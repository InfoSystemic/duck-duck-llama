#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
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

static uint64_t digest(const void * data, size_t bytes, uint64_t hash = 14695981039346656037ULL) {
    const auto p = static_cast<const unsigned char *>(data);
    for (size_t i = 0; i < bytes; ++i) hash = (hash ^ p[i])*1099511628211ULL;
    return hash;
}

static bool run(ggml_backend_t backend, ggml_tensor * gate, ggml_tensor * up, int nr, bool padded,
                bool consumer, bool timing, bool rotating, bool disjoint, FILE * dump, int id, int used = 10) {
    const int k = 2560, nc = 160, experts = 512;
    auto ctx = ggml_init({16U*1024U*1024U, nullptr, false});
    auto storage = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k+(padded ? 32 : 0), 1, nr);
    auto x = padded ? ggml_view_3d(ctx, storage, k, 1, nr, storage->nb[1], storage->nb[2], 0) : storage;
    auto routes = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, used, nr);
    auto g = ggml_mul_mat_id(ctx, gate, x, routes), u = ggml_mul_mat_id(ctx, up, x, routes);
    auto y = ggml_swiglu_split(ctx, g, u);
    auto graph = ggml_new_graph(ctx);
    ggml_set_output(y);
    ggml_build_forward_expand(graph, y);
    auto extra = consumer ? ggml_scale(ctx,g,0.25f) : nullptr;
    if (extra) { ggml_set_output(extra); ggml_build_forward_expand(graph,extra); }
    std::vector<float> input(ggml_nelements(storage),12345.0f);
    std::vector<int32_t> ids(used*nr);
    std::vector<double> samples;
    uint64_t hash = 14695981039346656037ULL;
    bool seen[512] = {};
    bool good = true;
    using count_fn = uint64_t (*)();
    const auto counter = reinterpret_cast<count_fn>(dlsym(RTLD_DEFAULT,"ggml_cpu_qwen_q6_single_count"));
    const uint64_t initial_calls = counter ? counter() : 0;
    const int iterations = timing ? 128 : 3;
    for (int iteration = 0; iteration < iterations+5; ++iteration) {
        const int sample = timing ? 0 : std::max(0,iteration-5);
        for (int t = 0; t < nr; ++t) for (int j = 0; j < k; ++j) {
            input[size_t(t)*(k+(padded ? 32 : 0))+j] = sample == 2 ? 0.0f : float((j*53+t*17+sample*31)%1021-510)/2048.0f;
        }
        std::memcpy(storage->data,input.data(),input.size()*sizeof(float));
        const int shift = rotating ? used*std::max(0,iteration-5) : 0;
        for (int t = 0; t < nr; ++t) for (int j = 0; j < used; ++j) ids[t*used+j] = ((shift+j+t*(disjoint ? used : 3))*73+19)%experts;
        std::memcpy(routes->data,ids.data(),ids.size()*sizeof(int32_t));
        const auto start = std::chrono::steady_clock::now();
        if (ggml_backend_graph_compute(backend,graph) != GGML_STATUS_SUCCESS) std::abort();
        const double us = std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();
        if (iteration < 5) continue;
        for (int expert : ids) seen[expert] = true;
        samples.push_back(us);
        good &= !std::memcmp(storage->data,input.data(),input.size()*sizeof(float));
        good &= !std::memcmp(routes->data,ids.data(),ids.size()*sizeof(int32_t));
        for (int64_t i = 0; i < ggml_nelements(y); ++i) good &= std::isfinite(static_cast<float *>(y->data)[i]);
        hash = digest(y->data,ggml_nbytes(y),hash);
        if (extra) hash = digest(extra->data,ggml_nbytes(extra),hash);
        if (dump && std::fwrite(y->data,1,ggml_nbytes(y),dump) != ggml_nbytes(y)) std::abort();
        if (dump && extra && std::fwrite(extra->data,1,ggml_nbytes(extra),dump) != ggml_nbytes(extra)) std::abort();
    }
    std::sort(samples.begin(),samples.end());
    const uint64_t calls = counter ? counter()-initial_calls : 0;
    const int visited = std::count(seen,seen+512,true);
    std::printf("MOE_CASE {\"id\":%d,\"nr\":%d,\"used\":%d,\"padded\":%s,\"consumer\":%s,\"timing\":%s,\"rotating\":%s,\"disjoint\":%s,\"samples\":%zu,\"median_us\":%.6f,\"hash\":%llu,\"selected_calls\":%llu,\"active_experts\":%d,\"passed\":%s}\n",
                id,nr,used,padded?"true":"false",consumer?"true":"false",timing?"true":"false",rotating?"true":"false",
                disjoint?"true":"false",samples.size(),(samples[(samples.size()-1)/2]+samples[samples.size()/2])/2,
                (unsigned long long)hash,(unsigned long long)calls,visited,good?"true":"false");
    std::fflush(stdout);
    ggml_free(ctx);
    return good;
}

int main(int argc, char ** argv) {
    if (argc != 3) return 2;
    const bool timing = !std::strcmp(argv[1],"--timing");
    const int threads = std::atoi(argv[2]);
    if (threads != 1 && threads != 4 && threads != 15) return 2;
    FILE * dump = timing ? nullptr : std::fopen(argv[1],"wb");
    if (!timing && !dump) return 2;
    Dl_info runtime{};
    if (!dladdr(reinterpret_cast<void *>(ggml_backend_cpu_init),&runtime)) std::abort();
    std::printf("CPU_LIBRARY %s\n",runtime.dli_fname);
    std::printf("MOE_COUNTER_PRESENT %d\n",dlsym(RTLD_DEFAULT,"ggml_cpu_qwen_q6_single_count") != nullptr);
    cpu_set_t available;
    if (sched_getaffinity(0,sizeof(available),&available) || CPU_COUNT(&available) != 15) std::abort();
    auto pp = ggml_threadpool_params_default(threads);
    int assigned = 0;
    for (int i = 0; i < GGML_MAX_N_THREADS; ++i) {
        pp.cpumask[i] = CPU_ISSET(i,&available) && assigned < threads;
        if (pp.cpumask[i]) ++assigned;
    }
    if (assigned != threads) std::abort();
    pp.strict_cpu = true;
    auto pool = ggml_threadpool_new(&pp);
    auto backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_threadpool(backend,pool);
    ggml_backend_cpu_set_n_threads(backend,threads);
    auto wc = ggml_init({ggml_tensor_overhead()*4,nullptr,true});
    auto gate = ggml_new_tensor_3d(wc,GGML_TYPE_Q6_K,2560,160,512);
    auto up = ggml_new_tensor_3d(wc,GGML_TYPE_Q6_K,2560,160,512);
    ggml_set_name(gate,"blk.0.ffn_gate_exps.weight");
    ggml_set_name(up,"blk.0.ffn_up_exps.weight");
    auto wb = ggml_backend_alloc_ctx_tensors_from_buft(wc,ggml_backend_cpu_repack_buffer_type());
    if (!wb || !gate->extra || !up->extra) std::abort();
    std::vector<block_q6_K> values(ggml_nelements(gate)/QK_K);
    for (int arm = 0; arm < 2; ++arm) {
        for (size_t b = 0; b < values.size(); ++b) {
            auto & v = values[b];
            v.d = ggml_fp32_to_fp16(float(1+b%7)/65536.0f);
            for (size_t j = 0; j < sizeof(v.ql); ++j) v.ql[j] = (b*71+j*43+arm*23+13)%256;
            for (size_t j = 0; j < sizeof(v.qh); ++j) v.qh[j] = (b*31+j*53+arm*31+17)%256;
            for (size_t j = 0; j < sizeof(v.scales); ++j) v.scales[j] = int((b*13+j*7+arm*19)%63)-31;
        }
        auto w = arm ? up : gate;
        ggml_backend_tensor_set(w,values.data(),0,ggml_nbytes(w));
    }
    const uint64_t before = digest(up->data,ggml_nbytes(up),digest(gate->data,ggml_nbytes(gate)));
    int cases = 0, failures = 0;
    if (timing) {
        for (bool rotating : {false,true}) {
            failures += !run(backend,gate,up,1,false,false,true,rotating,false,nullptr,cases++);
            for (bool disjoint : {false,true}) failures += !run(backend,gate,up,5,false,false,true,rotating,disjoint,nullptr,cases++);
        }
    } else {
        for (int nr : {1,2,3,4,5,8,64}) for (bool padded : {false,true}) for (bool consumer : {false,true}) {
            failures += !run(backend,gate,up,nr,padded,consumer,false,false,false,dump,cases++);
        }
    }
    if (!timing) for (int nr : {1,5}) failures += !run(backend,gate,up,nr,false,false,false,false,false,dump,cases++,8);
    const bool weights_preserved = before == digest(up->data,ggml_nbytes(up),digest(gate->data,ggml_nbytes(gate)));
    failures += !weights_preserved;
    if (dump && std::fclose(dump)) ++failures;
    ggml_backend_buffer_free(wb);
    ggml_free(wc);
    ggml_backend_free(backend);
    ggml_threadpool_free(pool);
    std::printf("MOE_SUMMARY {\"cases\":%d,\"failures\":%d,\"weights_preserved\":%s}\n",cases,failures,weights_preserved?"true":"false");
    return failures ? 1 : 0;
}
