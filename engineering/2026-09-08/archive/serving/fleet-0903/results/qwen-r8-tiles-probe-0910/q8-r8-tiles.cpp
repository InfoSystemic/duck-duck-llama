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
#include <random>
#include <string>
#include <utility>
#include <vector>
#include <dlfcn.h>
#include <link.h>
#include <sched.h>

struct run_result { std::vector<float> values; double ms; uint64_t weight_digest = 0; };

static run_result run(bool packed, int k, int rows, int tokens, int threads, bool moe, bool fused, ggml_type type, bool iqk = false) {
    const int experts = moe ? 512 : 1;
    const int used = moe ? 10 : 1;
    auto backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, threads);
    ggml_threadpool_t pool = nullptr;
    cpu_set_t original_affinity;
    const bool restore_affinity = std::getenv("REPACK_TEST_PERSISTENT_POOL") && std::getenv("REPACK_TEST_PIN_POOL");
    if (std::getenv("REPACK_TEST_PERSISTENT_POOL")) {
        auto params = ggml_threadpool_params_default(threads);
        if (restore_affinity) {
            if (sched_getaffinity(0, sizeof(original_affinity), &original_affinity) != 0) std::abort();
            if (CPU_COUNT(&original_affinity) != threads) std::abort();
            for (int cpu = 0; cpu < GGML_MAX_N_THREADS; ++cpu) {
                params.cpumask[cpu] = cpu < CPU_SETSIZE && CPU_ISSET(cpu, &original_affinity);
            }
            params.strict_cpu = true;
        }
        pool = ggml_threadpool_new(&params);
        if (!pool) std::abort();
        ggml_backend_cpu_set_threadpool(backend, pool);
    }
    auto wc = ggml_init({ggml_tensor_overhead() * 4, nullptr, true});
    auto ic = ggml_init({ggml_tensor_overhead() * 4, nullptr, true});
    auto gc = ggml_init({ggml_tensor_overhead() * 32 + ggml_graph_overhead(), nullptr, true});
    auto w = ggml_new_tensor_3d(wc, type, k, rows, experts);
    auto u = fused ? ggml_new_tensor_3d(wc, type, k, rows, experts) : nullptr;
    ggml_set_name(w, moe ? "blk.0.ffn_down_exps.weight" : k == 1536 ? "blk.0.ssm_out.weight" : "blk.0.hc_attn_down.weight");
    if (std::getenv("REPACK_TEST_FULL_GLM_Q5")) {
        ggml_set_name(w, moe ? "blk.0.ffn_down_exps.weight" : "blk.0.attn_output.weight");
    }
    if (u) ggml_set_name(u, "blk.0.ffn_up_exps.weight");
    auto wb = ggml_backend_alloc_ctx_tensors_from_buft(wc,
        packed ? ggml_backend_cpu_repack_buffer_type() : ggml_backend_cpu_buffer_type());
    if (!wb || (packed && (!w->extra || (u && !u->extra)))) std::abort();
    std::mt19937 rng(72491);
    std::normal_distribution<float> q5_values(0.0f, 0.02f);
    std::vector<uint8_t> weights(ggml_nbytes(w));
    uint64_t weight_digest = 14695981039346656037ULL;
    auto fill_weights = [&](ggml_tensor * tensor) {
        const size_t block_size = ggml_type_size(type);
        for (size_t i = 0; i < weights.size() / block_size; ++i) {
            auto * block = weights.data() + i * block_size;
            const auto d = ggml_fp32_to_fp16(0.0003f * (1 + i % 5));
            std::memcpy(block, &d, sizeof(d));
            for (size_t j = sizeof(d); j < block_size; ++j) {
                block[j] = type == GGML_TYPE_Q8_0 ? uint8_t(int(rng() % 255) - 127) : uint8_t(rng());
            }
            if (type == GGML_TYPE_Q5_K) {
                float source[QK_K];
                for (float & value : source) value = q5_values(rng);
                quantize_row_q5_K_ref(source, reinterpret_cast<block_q5_K *>(block), QK_K);
            }
        }
        if (std::getenv("REPACK_TEST_FULL_GLM_Q5")) {
            for (uint8_t value : weights) weight_digest = (weight_digest ^ value) * 1099511628211ULL;
        }
        ggml_backend_tensor_set(tensor, weights.data(), 0, ggml_nbytes(tensor));
    };
    fill_weights(w);
    if (u) fill_weights(u);
    const bool padded = std::getenv("REPACK_TEST_PADDED") != nullptr;
    const int src_rows = moe && !fused && std::getenv("REPACK_TEST_DOWN") ? used : 1;
    ggml_tensor * x;
    if (padded) {
        auto storage = ggml_new_tensor_3d(ic, GGML_TYPE_F32, k + 32, moe ? src_rows : tokens, moe ? tokens : 1);
        x = ggml_view_3d(ic, storage, k, storage->ne[1], storage->ne[2], storage->nb[1], storage->nb[2], 0);
    } else {
        x = moe ? ggml_new_tensor_3d(ic, GGML_TYPE_F32, k, src_rows, tokens)
                : ggml_new_tensor_2d(ic, GGML_TYPE_F32, k, tokens);
    }
    auto ids = moe ? ggml_new_tensor_2d(ic, GGML_TYPE_I32, used, tokens) : nullptr;
    auto ib = ggml_backend_alloc_ctx_tensors_from_buft(ic, ggml_backend_cpu_buffer_type());
    if (!ib) std::abort();
    ggml_set_input(x);
    if (ids) ggml_set_input(ids);
    auto y = moe ? ggml_mul_mat_id(gc, w, x, ids) : ggml_mul_mat(gc, w, x);
    if (u) {
        auto z = moe ? ggml_mul_mat_id(gc, u, x, ids) : ggml_mul_mat(gc, u, x);
        auto gate_output = y;
        if (moe && std::getenv("REPACK_TEST_CLAMP")) {
            y = ggml_clamp(gc, y, -INFINITY, 0.125f);
            z = ggml_clamp(gc, z, -0.25f, 0.25f);
        }
        y = ggml_swiglu_split(gc, y, z);
        if (moe && std::getenv("REPACK_TEST_CLAMP_CONSUMER")) {
            y = ggml_add(gc, y, gate_output);
        }
    }
    ggml_set_output(y);
    auto graph = ggml_new_graph(gc);
    ggml_build_forward_expand(graph, y);
    auto alloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    if (!ggml_gallocr_alloc_graph(alloc, graph)) std::abort();
    std::vector<float> activations(ggml_nbytes(x) / sizeof(float));
    for (size_t i = 0; i < activations.size(); ++i) activations[i] = std::sin(float(i) * 0.0123f) + 0.3f * std::cos(float(i) * 0.079f);
    ggml_backend_tensor_set(x, activations.data(), 0, ggml_nbytes(x));
    if (ids) {
        std::vector<int32_t> routes(used * tokens);
        for (int t = 0; t < tokens; ++t)
            for (int j = 0; j < used; ++j) routes[t * used + j] = (j + 3 * t) % experts;
        ggml_backend_tensor_set(ids, routes.data(), 0, ggml_nbytes(ids));
    }
    using mat_fn = bool (*)(long,long,long,int,const void *,long,int,const void *,long,float *,long,int,int);
    using moe_fn = bool (*)(long,long,long,int,int,const void *,long,int,const void *,long,float *,long,long,const void *,int,int);
    void * library = nullptr;
    mat_fn iqk_mat = nullptr;
    moe_fn iqk_moe = nullptr;
    std::vector<block_q8_K> q8;
    struct iqk_q8_K { float d, sum; int8_t qs[QK_K]; int16_t bsums[QK_K/16]; };
    static_assert(sizeof(iqk_q8_K)==296);
    std::vector<iqk_q8_K> iqk_q8;
    struct mapping { int32_t i1, i2; };
    std::vector<std::vector<mapping>> mappings(experts);
    if (iqk) {
        if (packed || fused || threads != 1) std::abort();
        static void * handle = dlmopen(LM_ID_NEWLM, std::getenv("IQK_BRIDGE_LIBRARY"), RTLD_NOW | RTLD_LOCAL);
        library = handle;
        if (!library) { std::fprintf(stderr,"dlmopen: %s\n",dlerror()); std::abort(); }
        static const bool initialized = [&]() {
            auto init = reinterpret_cast<void * (*)(ggml_init_params)>(dlsym(library,"ggml_init"));
            auto free_ctx = reinterpret_cast<void (*)(void *)>(dlsym(library,"ggml_free"));
            if (!init || !free_ctx) std::abort();
            void * context=init({128*1024,nullptr,false});
            if (!context) std::abort();
            free_ctx(context);
            return true;
        }();
        (void) initialized;
        iqk_mat = reinterpret_cast<mat_fn>(dlsym(library,"iqk_mul_mat"));
        iqk_moe = reinterpret_cast<moe_fn>(dlsym(library,"iqk_mul_mat_moe"));
        auto type_name = reinterpret_cast<const char * (*)(int)>(dlsym(library,"ggml_type_name"));
        auto type_size = reinterpret_cast<size_t (*)(int)>(dlsym(library,"ggml_type_size"));
        if (!iqk_mat || !iqk_moe || !type_name || !type_size ||
                std::strcmp(type_name(type),ggml_type_name(type)) || type_size(type)!=ggml_type_size(type) ||
                std::strcmp(type_name(GGML_TYPE_Q8_K),"q8_K") || type_size(GGML_TYPE_Q8_K)!=sizeof(iqk_q8_K)) {
            std::fprintf(stderr,"IQK ABI check failed: type=%d name=%s bytes=%zu expected=%s/%zu Q8=%s/%zu expected=%zu functions=%d/%d\n",
                         int(type),type_name ? type_name(type) : "null",type_size ? type_size(type) : 0,ggml_type_name(type),ggml_type_size(type),
                         type_name ? type_name(GGML_TYPE_Q8_K) : "null",type_size ? type_size(GGML_TYPE_Q8_K) : 0,sizeof(iqk_q8_K),!!iqk_mat,!!iqk_moe);
            std::abort();
        }
        q8.resize((k/QK_K) * (moe ? src_rows : 1) * tokens);
        iqk_q8.resize(q8.size());
        for (int t=0;t<tokens;++t) for (int j=0;j<used;++j) mappings[(j+3*t)%experts].push_back({j,t});
    }
    uint64_t route_phase=0;
    auto compute = [&]() {
        if (ids) {
            if (iqk) std::abort();
            std::vector<int32_t> routes(used*tokens);
            for (int t=0;t<tokens;++t) for (int j=0;j<used;++j)
                routes[t*used+j]=(j+3*t+131*route_phase)%experts;
            ggml_backend_tensor_set(ids,routes.data(),0,ggml_nbytes(ids));
            ++route_phase;
        }
        if (!iqk) return ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS;
        const int nrows = moe ? src_rows : tokens;
        const int planes = moe ? tokens : 1;
        const size_t row_bytes = (k/QK_K)*sizeof(iqk_q8_K);
        for (int p=0;p<planes;++p) for (int r=0;r<nrows;++r) {
            const auto * input=(const float *)((const char *)x->data+p*x->nb[2]+r*x->nb[1]);
            ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float(input,q8.data()+(p*nrows+r)*(k/QK_K),k);
        }
        for (size_t b=0;b<q8.size();++b) {
            iqk_q8[b].d=q8[b].d;
            std::memcpy(iqk_q8[b].qs,q8[b].qs,sizeof(q8[b].qs));
            std::memcpy(iqk_q8[b].bsums,q8[b].bsums,sizeof(q8[b].bsums));
            int sum=0; for (int16_t v:q8[b].bsums) sum+=v;
            iqk_q8[b].sum=q8[b].d*sum;
        }
        if (!moe) return iqk_mat(rows,tokens,k,type,w->data,w->nb[1],GGML_TYPE_Q8_K,iqk_q8.data(),row_bytes,
                                (float *)y->data,y->nb[1]/sizeof(float),0,1);
        for (int e=0;e<experts;++e) if (!mappings[e].empty()) {
            if (!iqk_moe(rows,mappings[e].size(),k,src_rows,type,(const char *)w->data+e*w->nb[2],w->nb[1],
                         GGML_TYPE_Q8_K,iqk_q8.data(),row_bytes,(float *)y->data,y->nb[1],y->nb[2],mappings[e].data(),0,1)) return false;
        }
        return true;
    };
    run_result result;
    result.weight_digest = weight_digest;
    result.values.resize(ggml_nelements(y));
    if (!compute()) { std::fprintf(stderr,"compute rejected iqk=%d type=%s moe=%d rows=%d tokens=%d\n",iqk,ggml_type_name(type),moe,rows,tokens); std::abort(); }
    int repeats = std::getenv("REPACK_TEST_Q5_REAL_SHAPES") ? 50 : 5;
    if (const char * value = std::getenv("REPACK_TEST_REPEATS")) {
        char * end = nullptr;
        const long parsed = std::strtol(value, &end, 10);
        if (end == value || *end != '\0' || parsed < 1 || parsed > 1000) std::abort();
        repeats = int(parsed);
    }
    const bool median_timing = std::getenv("REPACK_TEST_TIMING_MEDIAN") != nullptr;
    std::vector<double> samples;
    if (median_timing) samples.reserve(repeats);
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeats; ++i) {
        const auto iteration_start = median_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        if (!compute()) std::abort();
        if (median_timing) samples.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - iteration_start).count());
    }
    result.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / repeats;
    if (median_timing) {
        std::sort(samples.begin(), samples.end());
        result.ms = 0.5 * (samples[(repeats - 1) / 2] + samples[repeats / 2]);
    }
    ggml_backend_tensor_get(y, result.values.data(), 0, ggml_nbytes(y));
    ggml_gallocr_free(alloc);
    ggml_backend_buffer_free(wb);
    ggml_backend_buffer_free(ib);
    ggml_free(ic);
    ggml_free(gc); ggml_free(wc); ggml_backend_free(backend);
    if (pool) ggml_threadpool_free(pool);
    if (restore_affinity && sched_setaffinity(0, sizeof(original_affinity), &original_affinity) != 0) std::abort();
    return result;
}


int main(int argc,char ** argv) {
    if (argc!=2) return 2;
    setenv("GGML_CPU_Q8_0_REPACK","1",1);
    setenv("GGML_CPU_Q8_0_REPACK_FORCE","1",1);
    setenv("REPACK_TEST_DOWN","1",1);
    ggml_backend_load_all();
    Dl_info runtime{};
    if (!dladdr(reinterpret_cast<void *>(ggml_backend_cpu_init),&runtime)) std::abort();
    std::printf("CPU_LIBRARY %s\n",runtime.dli_fname);
    FILE * dump=std::fopen(argv[1],"wb");
    if (!dump) return 2;
    const int threads=std::getenv("REPACK_TEST_THREADS") ? std::atoi(std::getenv("REPACK_TEST_THREADS")) : 15;
    int failures=0,cases=0;
    for (auto shape:std::vector<std::pair<int,int>>{{160,2560},{1536,2560},{10240,64},{10240,96}})
    for (int tokens:{1,3,5}) {
        const int k=shape.first,rows=shape.second;
        const bool moe=k==160;
        auto ref=run(false,k,rows,tokens,threads,moe,false,GGML_TYPE_Q8_0);
        auto got=run(true,k,rows,tokens,threads,moe,false,GGML_TYPE_Q8_0);
        bool okay=ref.values.size()==got.values.size() && ref.weight_digest==got.weight_digest;
        float max_abs=0,max_rel=0;
        for (size_t i=0;i<got.values.size();++i) {
            const float delta=std::abs(ref.values[i]-got.values[i]);
            max_abs=std::max(max_abs,delta);
            max_rel=std::max(max_rel,delta/(1+std::abs(ref.values[i])));
            okay &= std::isfinite(got.values[i]) && delta<=2e-4f*(1+std::abs(ref.values[i]));
        }
        if (std::fwrite(got.values.data(),sizeof(float),got.values.size(),dump)!=got.values.size()) return 2;
        uint64_t hash=14695981039346656037ULL;
        const auto * bytes=reinterpret_cast<const uint8_t *>(got.values.data());
        for (size_t i=0;i<got.values.size()*sizeof(float);++i) hash=(hash^bytes[i])*1099511628211ULL;
        std::printf("%s k=%d rows=%d threads=%d tokens=%d moe=%d native_ms=%.6f packed_ms=%.6f max_abs=%.8g max_scaled=%.8g weights=%016llx hash=%016llx\n",
            okay?"PASS":"FAIL",k,rows,threads,tokens,moe,ref.ms,got.ms,max_abs,max_rel,
            (unsigned long long)got.weight_digest,(unsigned long long)hash);
        std::fflush(stdout);++cases;failures+=!okay;
    }
    if (std::fclose(dump)) return 2;
    std::printf("Q8_R8_TILES cases=%d failures=%d\n",cases,failures);
    return failures?1:0;
}
