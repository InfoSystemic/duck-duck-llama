#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <vector>

int main() {
    ggml_backend_load_all();
    auto backend = ggml_backend_cpu_init();
    int cases = 0;
    int failures = 0;
    for (int n : {32, 288, 512, 1024}) {
        for (int k : std::set<int>{1, 8, 32, n}) {
            for (int rows : {1, 3, 9}) for (int pattern = 0; pattern < 4; ++pattern)
            for (int threads : {1, 4}) for (bool padded : {false, true}) {
                ggml_backend_cpu_set_n_threads(backend, threads);
                auto ctx = ggml_init({4*1024*1024, nullptr, false});
                auto storage = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n + (padded ? 8 : 0), rows);
                auto input = ggml_view_2d(ctx, storage, n, rows, storage->nb[1], 0);
                std::mt19937 rng(42 + n + rows);
                std::uniform_real_distribution<float> distribution(-2.0f, 2.0f);
                std::vector<std::vector<int32_t>> reference(rows, std::vector<int32_t>(n));
                for (int row = 0; row < rows; ++row) {
                    auto data = (float *) ((char *) input->data + row*input->nb[1]);
                    for (int i = 0; i < n; ++i) {
                        data[i] = distribution(rng);
                        if (pattern == 1) { data[i] = std::round(data[i]*4); }
                        if (pattern == 2) { data[i] = i % 2 ? 0.0f : -0.0f; }
                    }
                    if (pattern == 3) {
                        data[0] = std::numeric_limits<float>::infinity();
                        data[n - 1] = -std::numeric_limits<float>::infinity();
                    }
                    std::iota(reference[row].begin(), reference[row].end(), 0);
                    std::sort(reference[row].begin(), reference[row].end(),
                        [&](int32_t a, int32_t b) { return data[a] > data[b]; });
                }
                auto output = ggml_argsort_top_k(ctx, input, k);
                auto graph = ggml_new_graph(ctx);
                ggml_build_forward_expand(graph, output);
                const auto begin = std::chrono::steady_clock::now();
                bool ok = true;
                for (int iteration = 0; iteration < 5; ++iteration) {
                    ok = ok && ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
                }
                const double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - begin).count() / 5;
                uint64_t hash = 1469598103934665603ull;
                for (int row = 0; row < rows; ++row) {
                    auto data = (const int32_t *) ((const char *) output->data + row*output->nb[1]);
                    for (int i = 0; i < k; ++i) {
                        ok = ok && data[i] == reference[row][i];
                        hash ^= (uint32_t) data[i];
                        hash *= 1099511628211ull;
                    }
                }
                std::printf("%s n=%d k=%d rows=%d pattern=%d threads=%d padded=%d ms=%.6f hash=%016llx\n",
                    ok ? "PASS" : "FAIL", n, k, rows, pattern, threads, padded, ms,
                    (unsigned long long) hash);
                ++cases;
                failures += !ok;
                ggml_free(ctx);
            }
        }
    }
    ggml_backend_free(backend);
    std::printf("cases=%d failures=%d\n", cases, failures);
    return failures ? 1 : 0;
}
