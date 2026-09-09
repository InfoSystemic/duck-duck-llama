#include <fstream>
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
    const int experts = moe ? 288 : 1;
    const int used = moe ? 8 : 1;
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
    ggml_set_name(w, "blk.0.ffn_gate_exps.weight");
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
            for (int j = 0; j < used; ++j) routes[t * used + j] = (j + 8 * t + 250) % experts;
        std::vector<bool> touched(experts, false);
        for (int32_t id : routes) touched[id] = true;
        const int active = std::count(touched.begin(), touched.end(), true);
        if (experts != 288 || active != 288) std::abort();
        std::printf("ACTIVE_EXPERTS %d\n", active);
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
    auto compute = [&]() {
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

int main(int argc, char ** argv) {
    setenv("GGML_CPU_IQ2_XXS_REPACK", "1", 1);
    setenv("GGML_CPU_IQ2_XS_REPACK", "1", 1);
    setenv("GGML_CPU_IQ3_XXS_REPACK", "1", 1);
    setenv("GGML_CPU_MOE_GATE_UP_FUSION", "1", 1);
    setenv("GGML_CPU_FFN_GATE_UP_FUSION", "1", 1);
    setenv("GGML_CPU_Q8_0_REPACK", "1", 1);
    setenv("GGML_CPU_Q8_0_REPACK_FORCE", "1", 1);
    setenv("GGML_CPU_Q5_K_REPACK", "1", 1);
    ggml_backend_load_all();
    Dl_info loaded{};
    if (!dladdr(reinterpret_cast<void *>(ggml_backend_cpu_init), &loaded)) std::abort();
    std::printf("CPU_LIBRARY %s\n", loaded.dli_fname);

    int failures = 0, cases = 0;
    const bool q8 = argc > 1 && std::string(argv[1]) == "q8";
    const bool q5_pair = argc > 1 && std::string(argv[1]) == "q5-pair";
    const bool iqk_compare = argc > 1 && std::string(argv[1]) == "iqk-compare";
    if (q5_pair) setenv("GGML_CPU_X16_Q5_K", "1", 1);
    const bool qwen_iq = argc > 1 && std::string(argv[1]) == "iq-r16-qwen";
    const bool iq_r16 = iqk_compare || qwen_iq || (argc > 1 && std::string(argv[1]) == "iq-r16");
    const bool dense_work_sharing = std::getenv("REPACK_TEST_DENSE_WORK_SHARING") != nullptr;
    const bool work_sharing = dense_work_sharing || std::getenv("REPACK_TEST_WORK_SHARING") != nullptr;
    const bool q5_real_shapes = q5_pair && std::getenv("REPACK_TEST_Q5_REAL_SHAPES") != nullptr;
    const bool full_glm_q5 = q5_pair && std::getenv("REPACK_TEST_FULL_GLM_Q5") != nullptr;
    const bool iq_real_shapes = iq_r16 && std::getenv("REPACK_TEST_IQ_REAL_SHAPES") != nullptr;
    const bool iq_pair_tails = iq_r16 && std::getenv("REPACK_TEST_IQ_PAIR_TAILS") != nullptr;
    const std::vector<ggml_type> types = q8 ? std::vector<ggml_type>{GGML_TYPE_Q8_0}
        : q5_pair ? std::vector<ggml_type>{GGML_TYPE_Q5_K}
        : qwen_iq ? std::vector<ggml_type>{GGML_TYPE_IQ2_XS, GGML_TYPE_IQ3_XXS}
        : iq_r16 ? std::vector<ggml_type>{GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ3_XXS}
        : argc > 1 ? std::vector<ggml_type>{GGML_TYPE_IQ2_XS}
        : std::vector<ggml_type>{GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS};
    const std::vector<std::pair<int, int>> shapes = full_glm_q5 ? std::vector<std::pair<int, int>>{{4096, 6144}, {2048, 4096}, {6144, 512}, {512, 6144}, {1536, 2048}, {2048, 1024}, {1536, 576}}
        : q5_real_shapes ? std::vector<std::pair<int, int>>{{4096, 4096}, {1536, 4096}, {3072, 4096}, {4096, 512}}
        : iq_real_shapes ? std::vector<std::pair<int, int>>{{4096, 512}, {512, 4096}}
        : iq_pair_tails ? std::vector<std::pair<int, int>>{{256, 48}, {768, 80}, {1536, 112}}
        : q8 ? std::vector<std::pair<int, int>>{{4096, 512}, {512, 4096}}
        : iqk_compare ? std::vector<std::pair<int,int>>{{4096,512}}
        : q5_pair && !work_sharing ? std::vector<std::pair<int, int>>{{4096, 128}, {768, 80}, {256, 24}}
        : work_sharing ? std::vector<std::pair<int, int>>{{4096, 512}, {768, 80}, {256, 24}}
        : iq_r16 ? std::vector<std::pair<int, int>>{{4096, 128}, {768, 80}, {256, 24}}
        : std::vector<std::pair<int, int>>{{4096, 128}};
    const std::vector<int> token_counts = full_glm_q5 ? std::vector<int>{1, 2, 3} : q5_real_shapes ? std::vector<int>{3} : iq_real_shapes || iq_pair_tails ? std::vector<int>{1, 2, 3} : std::getenv("REPACK_TEST_SMALL_BATCHES")
        ? std::vector<int>{1, 2, 3, 4, 9} : std::vector<int>{64};
    std::vector<int> thread_counts = work_sharing || q5_real_shapes ? std::vector<int>{1, 4, 15} : std::vector<int>{1, 4};
    if (const char * value = std::getenv("REPACK_TEST_THREADS")) {
        char * end = nullptr;
        const long parsed = std::strtol(value, &end, 10);
        if (end == value || *end != '\0' || parsed < 1 || parsed > GGML_MAX_N_THREADS) std::abort();
        thread_counts = {int(parsed)};
    }
    int selected_full_case = -1, full_case_ordinal = 0;
    if (const char * value = std::getenv("REPACK_TEST_FULL_CASE_INDEX")) {
        char * end = nullptr;
        const long parsed = std::strtol(value, &end, 10);
        if (!full_glm_q5 || end == value || *end != '\0' || parsed < 0 || parsed > 4096) std::abort();
        selected_full_case = int(parsed);
    }
    for (auto type : types) for (const auto & shape : shapes) for (int threads : thread_counts) for (int tokens : token_counts) for (bool moe : {false, true}) for (bool fused : {false, true}) {
        if (!moe || (shape.first == 512 && fused)) continue;
        if (q5_real_shapes && (moe || fused)) continue;
        if (full_glm_q5 && (fused || (moe && shape != std::make_pair(512, 6144)))) continue;
        if (iqk_compare && (threads!=1 || tokens==9 || fused)) continue;
        const int full_case_index = full_case_ordinal++;
        if (selected_full_case >= 0 && full_case_index != selected_full_case) continue;
        const int k = shape.first, rows = shape.second;
        auto ref = run(false, k, rows, tokens, threads, moe, fused, type);
        auto got = run(true, k, rows, tokens, threads, moe, fused, type);
        run_result ik;
        if (iqk_compare) ik = run(false,k,rows,tokens,threads,moe,fused,type,true);
        bool okay = ref.values.size() == got.values.size() && ref.weight_digest == got.weight_digest;
        float max_abs = 0, max_rel = 0;
        for (size_t i = 0; i < ref.values.size(); ++i) {
            const float delta = std::abs(ref.values[i] - got.values[i]);
            max_abs = std::max(max_abs, delta);
            max_rel = std::max(max_rel, delta / (1 + std::abs(ref.values[i])));
            okay &= std::isfinite(got.values[i]) && delta <= 2e-4f * (1 + std::abs(ref.values[i]));
            if (iqk_compare) okay &= std::isfinite(ik.values[i]) && std::abs(ik.values[i]-ref.values[i]) <= 2e-4f*(1+std::abs(ref.values[i]));
        }
        if (const char * dir = std::getenv("REPACK_TEST_OUTPUT_DIR")) {
            const std::string path = std::string(dir) + "/" + std::to_string(k) + "-" + std::to_string(rows) + "-" +
                std::to_string(tokens) + "-" + std::to_string(fused) + ".f32";
            std::ofstream output(path, std::ios::binary);
            output.write(reinterpret_cast<const char *>(got.values.data()), got.values.size() * sizeof(float));
            if (!output) std::abort();
        }
        ++cases; failures += !okay;
        uint64_t checksum = 14695981039346656037ULL;
        const auto * bytes = reinterpret_cast<const uint8_t *>(got.values.data());
        for (size_t i = 0; i < got.values.size() * sizeof(float); ++i) checksum = (checksum ^ bytes[i]) * 1099511628211ULL;
        std::printf("%s type=%s k=%d rows=%d threads=%d tokens=%d moe=%d fused=%d max_abs=%.8g max_scaled=%.8g native_ms=%.6f packed_ms=%.6f hash=%016llx\n",
                    okay ? "PASS" : "FAIL", ggml_type_name(type), k, rows, threads, tokens, moe, fused, max_abs, max_rel, ref.ms, got.ms, (unsigned long long) checksum);
        std::fflush(stdout);
        if (full_glm_q5) {
            std::printf("FULL_INPUT k=%d rows=%d threads=%d tokens=%d moe=%d digest=%016llx case_index=%d\n",
                        k, rows, threads, tokens, moe, (unsigned long long) got.weight_digest, full_case_index);
            std::fflush(stdout);
        }
        if (iqk_compare) {
            float ik_max=0;
            for (size_t i=0;i<ref.values.size();++i) ik_max=std::max(ik_max,std::abs(ik.values[i]-ref.values[i])/(1+std::abs(ref.values[i])));
            std::printf("IQK type=%s tokens=%d moe=%d ms=%.6f r16_ms=%.6f max_scaled=%.8g\n",ggml_type_name(type),tokens,moe,ik.ms,got.ms,ik_max);
            std::fflush(stdout);
        }
    }
    std::printf("Repacking: %d cases, %d failures\n", cases, failures);
    if (selected_full_case >= 0 && cases != 1) return 1;
    return failures ? 1 : 0;
}
