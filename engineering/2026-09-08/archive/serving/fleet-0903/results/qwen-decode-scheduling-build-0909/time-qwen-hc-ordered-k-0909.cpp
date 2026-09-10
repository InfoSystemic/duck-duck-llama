#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "repack.h"
#include "ggml-quants.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <sched.h>
#include <vector>

static void run(int nc, int nr, int count, ggml_backend_t backend) {
    const int k = 10240;
    auto wc = ggml_init({ggml_tensor_overhead()*size_t(count+4), nullptr, true});
    std::vector<ggml_tensor *> weights;
    for (int i = 0; i < count; ++i) {
        auto w = ggml_new_tensor_2d(wc, GGML_TYPE_Q8_0, k, nc);
        ggml_set_name(w, "blk.0.hc_attn_down.weight");
        weights.push_back(w);
    }
    auto wb = ggml_backend_alloc_ctx_tensors_from_buft(wc, ggml_backend_cpu_repack_buffer_type());
    if (!wb) std::abort();
    std::vector<block_q8_0> weight(size_t(k/32)*nc);
    for (int i = 0; i < count; ++i) {
        if (!weights[i]->extra) std::abort();
        for (size_t b = 0; b < weight.size(); ++b) {
            weight[b].d = ggml_fp32_to_fp16(float(1+(b+i)%11)/8192.0f);
            for (int j = 0; j < 32; ++j) weight[b].qs[j] = int((b*71+j*43+i*11+13)%255)-127;
        }
        ggml_backend_tensor_set(weights[i], weight.data(), 0, ggml_nbytes(weights[i]));
    }
    auto ctx = ggml_init({32U*1024U*1024U, nullptr, false});
    auto x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, nr);
    for (int i = 0; i < k*nr; ++i) static_cast<float *>(x->data)[i] = float((i*53+17)%1021-510)/128.0f;
    auto graph = ggml_new_graph(ctx);
    std::vector<ggml_tensor *> outputs;
    for (auto w : weights) {
        auto y = ggml_mul_mat(ctx, w, x);
        ggml_set_output(y);
        ggml_build_forward_expand(graph, y);
        outputs.push_back(y);
    }
    for (int i = 0; i < 5; ++i) if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) std::abort();
    uint64_t hash = 14695981039346656037ULL;
    for (auto y : outputs) for (size_t j = 0; j < ggml_nbytes(y); ++j) hash = (hash ^ static_cast<unsigned char *>(y->data)[j])*1099511628211ULL;
    std::vector<double> times;
    for (int i = 0; i < 40; ++i) {
        auto start = std::chrono::steady_clock::now();
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) std::abort();
        times.push_back(std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/count);
    }
    std::sort(times.begin(), times.end());
    std::printf("HC_TIME {\"nc\":%d,\"nr\":%d,\"matrices\":%d,\"weight_bytes\":%zu,\"samples\":40,\"median_us\":%.6f,\"p10_us\":%.6f,\"p90_us\":%.6f,\"hash\":%llu}\n",
                nc,nr,count,ggml_nbytes(weights[0])*count,(times[19]+times[20])/2,times[4],times[35],(unsigned long long)hash);
    std::fflush(stdout);
    ggml_backend_buffer_free(wb);
    ggml_free(wc);
    ggml_free(ctx);
}

int main() {
    Dl_info runtime{};
    if (!dladdr(reinterpret_cast<void *>(ggml_backend_cpu_init), &runtime)) std::abort();
    std::printf("CPU_LIBRARY %s\n", runtime.dli_fname);
    auto pp = ggml_threadpool_params_default(15);
    cpu_set_t available;
    if (sched_getaffinity(0, sizeof(available), &available) || CPU_COUNT(&available) != 15) std::abort();
    for (int i = 0; i < GGML_MAX_N_THREADS; ++i) pp.cpumask[i] = i < CPU_SETSIZE && CPU_ISSET(i, &available);
    pp.strict_cpu = true;
    auto pool = ggml_threadpool_new(&pp);
    auto backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, 15);
    ggml_backend_cpu_set_threadpool(backend, pool);
    for (int count : {1,96}) for (int nc : {64,96}) for (int nr : {1,4,5}) run(nc,nr,count,backend);
    ggml_backend_free(backend);
    ggml_threadpool_free(pool);
    return 0;
}
