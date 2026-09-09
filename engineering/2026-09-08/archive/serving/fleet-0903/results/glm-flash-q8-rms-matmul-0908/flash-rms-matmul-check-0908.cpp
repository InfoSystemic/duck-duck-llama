#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "repack.h"
#include "ggml-quants.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <sched.h>
#include <vector>

struct test_case {
    int k = 16384, rows = 24, tokens = 1, planes = 1, threads = 15;
    bool weighted = false, swapped = false, per_row_scale = false;
    bool packed = true, padded = false, reference = false, inplace = false;
    bool expose_norm = false, expose_activation = false, extra_use = false;
    int alias = 0;
    ggml_type type = GGML_TYPE_Q8_0;
};

using count_fn = uint64_t (*)(int);
static bool enabled(const char * name) { const char * v = std::getenv(name); return v && std::atoi(v) == 1; }

static bool run_case(const test_case & c, FILE * dump, bool timing, int id) {
    cpu_set_t saved, selected;
    if (sched_getaffinity(0, sizeof(saved), &saved)) std::abort();
    CPU_ZERO(&selected);
    int picked = 0;
    auto pp = ggml_threadpool_params_default(c.threads);
    for (int cpu = 0; cpu < CPU_SETSIZE && cpu < GGML_MAX_N_THREADS && picked < c.threads; ++cpu) {
        if (CPU_ISSET(cpu, &saved)) { CPU_SET(cpu, &selected); pp.cpumask[cpu] = true; ++picked; }
    }
    if (picked != c.threads) std::abort();
    pp.strict_cpu = true;
    auto pool = ggml_threadpool_new(&pp);
    auto backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, c.threads);
    ggml_backend_cpu_set_threadpool(backend, pool);
    ggml_backend_cpu_set_use_ref(backend, c.reference);
    auto wc = ggml_init({ggml_tensor_overhead()*4, nullptr, true});
    auto w = ggml_new_tensor_2d(wc, c.type, c.k, c.rows);
    ggml_set_name(w, c.rows == 24 ? "blk.0.hc_attn_fn.weight" : "blk.0.attn_output.weight");
    auto wb = ggml_backend_alloc_ctx_tensors_from_buft(wc,
        c.packed ? ggml_backend_cpu_repack_buffer_type() : ggml_backend_cpu_buffer_type());
    if (!wb || (c.packed && c.type == GGML_TYPE_Q8_0 && !w->extra)) std::abort();
    std::vector<float> weight_values(size_t(c.k)*c.rows), decoded(weight_values.size());
    for (size_t i = 0; i < weight_values.size(); ++i) {
        weight_values[i] = float(int((i*37+19)%257)-128)/8192.0f;
    }
    std::vector<unsigned char> quantized(ggml_nbytes(w));
    if (c.type == GGML_TYPE_Q8_0) {
        quantize_row_q8_0_ref(weight_values.data(), reinterpret_cast<block_q8_0 *>(quantized.data()), weight_values.size());
        dequantize_row_q8_0(reinterpret_cast<const block_q8_0 *>(quantized.data()), decoded.data(), decoded.size());
    } else {
        std::memcpy(quantized.data(), weight_values.data(), quantized.size());
        decoded = weight_values;
    }
    ggml_backend_tensor_set(w, quantized.data(), 0, quantized.size());
    auto ctx = ggml_init({128U*1024U*1024U, nullptr, false});
    auto graph = ggml_new_graph(ctx);
    struct group { ggml_tensor * x; ggml_tensor * scale; ggml_tensor * norm; ggml_tensor * act; ggml_tensor * out; ggml_tensor * extra; };
    std::vector<group> groups;
    const int group_count = timing ? 90 : 1;
    for (int i = 0; i < group_count; ++i) {
        auto storage = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.k+(c.padded ? 32 : 0), c.tokens, c.planes);
        auto x = c.padded ? ggml_view_3d(ctx, storage, c.k, c.tokens, c.planes, storage->nb[1], storage->nb[2], 0) : storage;
        ggml_set_input(x);
        auto scale = c.weighted ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.k, c.per_row_scale ? c.tokens : 1) : nullptr;
        auto norm = c.inplace ? ggml_rms_norm_inplace(ctx, x, 1e-6f) : ggml_rms_norm(ctx, x, 1e-6f);
        auto act = c.weighted ? (c.swapped ? ggml_mul(ctx, scale, norm) : ggml_mul(ctx, norm, scale)) : norm;
        auto out = ggml_mul_mat(ctx, w, act);
        if (c.alias == 1) out->data = act->data;
        if (c.alias == 2) out->data = x->data;
        if (c.expose_norm) ggml_set_output(norm);
        if (c.expose_activation) ggml_set_output(act);
        ggml_set_output(out);
        ggml_build_forward_expand(graph, out);
        auto extra = c.extra_use ? ggml_scale(ctx, norm, 0.5f) : nullptr;
        if (extra) { ggml_set_output(extra); ggml_build_forward_expand(graph, extra); }
        groups.push_back({x,scale,norm,act,out,extra});
    }
    const auto counter = reinterpret_cast<count_fn>(dlsym(RTLD_DEFAULT, "ggml_cpu_rms_matmul_fusion_count"));
    const bool audit = counter && enabled("GGML_CPU_RMS_MATMUL_AUDIT");
    const bool expected_fused = counter && enabled("GGML_CPU_RMS_MATMUL_FUSION") &&
        !enabled("GGML_CPU_DISABLE_FUSION") && !c.reference && c.threads > 1 &&
        c.k*c.tokens*c.planes > 4096 && c.tokens <= 3 && c.planes == 1 &&
        c.packed && c.type == GGML_TYPE_Q8_0 && !c.padded && !c.inplace &&
        !c.expose_norm && !c.expose_activation && !c.extra_use;
    bool good = true;
    double max_scaled_error = 0.0;
    const int samples = timing ? 1 : 3;
    uint64_t fused_count = 0;
    for (int sample = 0; sample < samples; ++sample) {
        std::vector<std::vector<float>> originals;
        for (auto & g : groups) {
            std::vector<float> values(size_t(c.k)*c.tokens*c.planes);
            for (int p = 0; p < c.planes; ++p) for (int t = 0; t < c.tokens; ++t) {
                auto row = reinterpret_cast<float *>(static_cast<char *>(g.x->data)+p*g.x->nb[2]+t*g.x->nb[1]);
                for (int k = 0; k < c.k; ++k) {
                    size_t ix = (size_t(p)*c.tokens+t)*c.k+k;
                    row[k] = values[ix] = sample == 2 ? 0.0f : float(int((ix*53+sample*71+17)%1021)-510)/128.0f;
                }
            }
            if (g.scale) for (int64_t j = 0; j < ggml_nelements(g.scale); ++j) {
                static_cast<float *>(g.scale->data)[j] = float(int((j*7+sample*13)%41)-20)/16.0f;
            }
            originals.push_back(std::move(values));
        }
        uint64_t before = audit ? counter(c.weighted) : 0;
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) good = false;
        uint64_t delta = audit ? counter(c.weighted)-before : 0;
        if (audit && delta != uint64_t(expected_fused ? group_count : 0)) good = false;
        fused_count += delta;
        for (size_t gi = 0; gi < groups.size(); ++gi) {
            auto & g = groups[gi];
            std::vector<float> normalized(c.k), activation(c.k);
            std::vector<block_q8_0> qinput(c.k/32);
            for (int p = 0; p < c.planes; ++p) for (int t = 0; t < c.tokens; ++t) {
                const auto x = originals[gi].data()+(size_t(p)*c.tokens+t)*c.k;
                double sum = 0.0;
                for (int k = 0; k < c.k; ++k) sum += double(x[k]*x[k]);
                const float mean = sum/c.k, scale = 1.0f/std::sqrt(mean+1e-6f);
                for (int k = 0; k < c.k; ++k) {
                    normalized[k] = x[k]*scale;
                    if (g.scale) normalized[k] *= static_cast<float *>(g.scale->data)[(c.per_row_scale ? t*c.k : 0)+k];
                }
                if (c.type == GGML_TYPE_Q8_0) {
                    ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float(normalized.data(),qinput.data(),c.k);
                    dequantize_row_q8_0(qinput.data(),activation.data(),c.k);
                } else activation = normalized;
                const auto actual = reinterpret_cast<float *>(static_cast<char *>(g.out->data)+p*g.out->nb[2]+t*g.out->nb[1]);
                for (int r = 0; r < c.rows; ++r) {
                    double expected = 0.0;
                    for (int k = 0; k < c.k; ++k) expected += double(decoded[size_t(r)*c.k+k])*activation[k];
                    const double error = std::abs(actual[r]-expected)/std::max(1.0,std::abs(expected));
                    max_scaled_error = std::max(max_scaled_error,error);
                    if (!std::isfinite(actual[r]) || error > 1e-4) good = false;
                }
            }
            if (dump) {
                std::fwrite(g.out->data,1,ggml_nbytes(g.out),dump);
                if (c.expose_norm) std::fwrite(g.norm->data,1,ggml_nbytes(g.norm),dump);
                if (c.expose_activation) std::fwrite(g.act->data,1,ggml_nbytes(g.act),dump);
                if (g.extra) std::fwrite(g.extra->data,1,ggml_nbytes(g.extra),dump);
            }
        }
    }
    double ms = 0.0;
    if (timing && good) {
        std::vector<double> times;
        for (int i = 0; i < 80; ++i) {
            auto start = std::chrono::steady_clock::now();
            if (ggml_backend_graph_compute(backend,graph) != GGML_STATUS_SUCCESS) good = false;
            times.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
        }
        std::sort(times.begin(),times.end()); ms = (times[39]+times[40])/2;
    }
    std::printf("%s id=%d k=%d rows=%d tokens=%d planes=%d threads=%d weighted=%d packed=%d fused=%llu groups=%d max_error=%.9g ms=%.6f\n",
        good ? "PASS" : "FAIL",id,c.k,c.rows,c.tokens,c.planes,c.threads,c.weighted,c.packed,
        (unsigned long long)fused_count,group_count,max_scaled_error,ms);
    std::fflush(stdout);
    ggml_backend_buffer_free(wb); ggml_free(wc); ggml_free(ctx); ggml_backend_free(backend); ggml_threadpool_free(pool);
    if (sched_setaffinity(0,sizeof(saved),&saved)) std::abort();
    return good;
}

int main(int argc, char ** argv) {
    setenv("GGML_CPU_Q8_0_REPACK","1",1);
    setenv("GGML_CPU_Q8_0_REPACK_FORCE","1",1);
    setenv("GGML_CPU_X16_Q8_0","1",1);
    const bool timing = argc == 2 && !std::strcmp(argv[1],"--timing");
    FILE * dump = timing ? nullptr : argc == 2 ? std::fopen(argv[1],"wb") : nullptr;
    if (!timing && !dump) return 2;
    int cases = 0, failures = 0;
    if (timing) {
        for (bool weighted : {false,true}) for (int tokens : {1,3}) {
            test_case c; c.weighted=weighted; c.tokens=tokens;
            failures += !run_case(c,nullptr,true,cases++);
        }
    } else {
        for (int rows : {24,64}) for (bool weighted : {false,true})
            for (int tokens : {1,2,3,4,64}) for (int threads : {1,4,15}) {
                test_case c; c.rows=rows; c.weighted=weighted; c.tokens=tokens; c.threads=threads;
                failures += !run_case(c,dump,false,cases++);
            }
        for (bool weighted : {false,true}) for (int variant=0; variant<12; ++variant) {
            test_case c; c.weighted=weighted; c.tokens=3;
            switch (variant) {
                case 0: c.packed=false; break;
                case 1: c.padded=true; break;
                case 2: c.reference=true; break;
                case 3: c.inplace=true; break;
                case 4: c.expose_norm=true; break;
                case 5: c.expose_activation=true; break;
                case 6: c.extra_use=true; break;
                case 7: c.planes=2; break;
                case 8: c.alias=1; break;
                case 9: c.alias=2; break;
                case 10: c.k=4096; c.tokens=1; break;
                case 11: c.type=GGML_TYPE_F32; c.packed=false; break;
            }
            failures += !run_case(c,dump,false,cases++);
        }
        for (bool swapped : {false,true}) {
            test_case c; c.weighted=true; c.tokens=3; c.per_row_scale=true; c.swapped=swapped;
            failures += !run_case(c,dump,false,cases++);
        }
        if (std::fclose(dump)) ++failures;
    }
    std::printf("SUMMARY cases=%d failures=%d\n",cases,failures);
    return failures ? 1 : 0;
}
