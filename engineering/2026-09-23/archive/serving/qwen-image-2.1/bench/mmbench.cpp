// Throughput of ggml's CPU mul_mat at Qwen-Image-2.1 DiT shapes, linked against the same
// static ggml that sd-cli uses.
//   mmbench <q8_0|f16|f32> <M out> <K in> <N tokens> <threads> <iters>
#include "ggml-cpu.h"
#include "ggml.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 7) {
        fprintf(stderr, "usage: %s <q8_0|f16|f32> <M> <K> <N> <threads> <iters>\n", argv[0]);
        return 2;
    }
    const char* tname = argv[1];
    const int64_t M = atoll(argv[2]), K = atoll(argv[3]), N = atoll(argv[4]);
    const int nth = atoi(argv[5]), iters = atoi(argv[6]);
    const ggml_type wt = !strcmp(tname, "q8_0") ? GGML_TYPE_Q8_0
                       : !strcmp(tname, "f16")  ? GGML_TYPE_F16
                                                : GGML_TYPE_F32;

    const size_t mem = ggml_row_size(wt, K) * M + (size_t)K * N * 4 + (size_t)M * N * 4 + (512ull << 20);
    ggml_context* ctx = ggml_init({mem, nullptr, false});
    ggml_tensor* W = ggml_new_tensor_2d(ctx, wt, K, M);
    ggml_tensor* X = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> u(-0.05f, 0.05f);
    std::vector<float> w(K * M);
    for (auto& v : w) v = u(rng);
    if (wt == GGML_TYPE_F32) {
        memcpy(W->data, w.data(), w.size() * sizeof(float));
    } else {
        ggml_quantize_chunk(wt, w.data(), W->data, 0, M, K, nullptr);
    }
    for (int64_t i = 0; i < K * N; i++) ((float*)X->data)[i] = u(rng);

    ggml_tensor* Y = ggml_mul_mat(ctx, W, X);
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, Y);

    ggml_graph_compute_with_ctx(ctx, gf, nth);  // warm-up: page faults, work buffer
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; i++) ggml_graph_compute_with_ctx(ctx, gf, nth);
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / iters;

    printf("%-5s M=%-6lld K=%-6lld N=%-5lld t=%-3d %8.1f ms  %6.2f TFLOPS  (%.0f GFLOPS/thread)\n", tname,
           (long long)M, (long long)K, (long long)N, nth, s * 1e3, 2.0 * M * K * N / s / 1e12,
           2.0 * M * K * N / s / 1e9 / nth);
    ggml_free(ctx);
    return 0;
}
