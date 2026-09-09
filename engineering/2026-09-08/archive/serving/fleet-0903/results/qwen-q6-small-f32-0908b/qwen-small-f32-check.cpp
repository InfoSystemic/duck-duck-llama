#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static void check(int k, int rows, int tokens, int threads, bool padded, bool timing) {
    const int layers = timing ? 96 : 4;
    auto backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, threads);
    auto ctx = ggml_init({4 * 1024 * 1024, nullptr, true});
    auto graph = ggml_new_graph_custom(ctx, 2048, false);
    std::mt19937 rng(941);
    std::normal_distribution<float> random(0.0f, 0.03f);
    std::vector<ggml_tensor *> weights, inputs, products;
    std::vector<std::vector<float>> weight_values, input_values;
    ggml_tensor * sum = nullptr;
    for (int layer = 0; layer < layers; ++layer) {
        auto w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, rows);
        auto storage = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k + (padded ? 16 : 0), tokens);
        auto x = padded ? ggml_view_2d(ctx, storage, k, tokens, storage->nb[1], 0) : storage;
        ggml_set_input(storage);
        auto y = ggml_mul_mat(ctx, w, x);
        ggml_set_name(y, "small-f32-product");
        ggml_set_output(y);
        auto activated = ggml_sigmoid(ctx, ggml_scale(ctx, y, 0.5f));
        sum = sum ? ggml_add(ctx, sum, activated) : activated;
        weights.push_back(w);
        inputs.push_back(storage);
        products.push_back(y);
        weight_values.emplace_back(k * rows);
        input_values.emplace_back((k + (padded ? 16 : 0)) * tokens);
        for (float & value : weight_values.back()) value = random(rng);
        for (float & value : input_values.back()) value = random(rng) * 7;
    }
    ggml_set_output(sum);
    ggml_build_forward_expand(graph, sum);
    auto storage = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!storage) std::abort();
    for (int i = 0; i < layers; ++i) {
        ggml_backend_tensor_set(weights[i], weight_values[i].data(), 0, ggml_nbytes(weights[i]));
        ggml_backend_tensor_set(inputs[i], input_values[i].data(), 0, ggml_nbytes(inputs[i]));
    }
    auto compute = [&]() {
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) std::abort();
        ggml_backend_synchronize(backend);
    };
    compute();
    uint64_t hash = 14695981039346656037ULL;
    double max_error = 0;
    bool okay = true;
    for (int i = 0; i < layers; ++i) {
        std::vector<float> values(rows * tokens);
        ggml_backend_tensor_get(products[i], values.data(), 0, ggml_nbytes(products[i]));
        for (int t = 0; t < tokens; ++t) for (int r = 0; r < rows; ++r) {
            double expected = 0;
            for (int j = 0; j < k; ++j) {
                expected += double(weight_values[i][r * k + j]) * input_values[i][t * (k + (padded ? 16 : 0)) + j];
            }
            const float actual = values[t * rows + r];
            const double error = std::abs(actual - expected);
            max_error = std::max(max_error, error);
            okay &= std::isfinite(actual) && error <= 2e-5 * (1 + std::abs(expected));
        }
        const auto * bytes = reinterpret_cast<const unsigned char *>(values.data());
        for (size_t j = 0; j < values.size() * sizeof(float); ++j) hash = (hash ^ bytes[j]) * 1099511628211ULL;
    }
    std::vector<float> output(rows * tokens);
    ggml_backend_tensor_get(sum, output.data(), 0, ggml_nbytes(sum));
    const auto * bytes = reinterpret_cast<const unsigned char *>(output.data());
    for (size_t j = 0; j < output.size() * sizeof(float); ++j) hash = (hash ^ bytes[j]) * 1099511628211ULL;
    std::vector<double> samples;
    for (int i = 0; i < (timing ? 70 : 2); ++i) {
        const auto start = std::chrono::steady_clock::now();
        compute();
        const double elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        if (i >= 10) samples.push_back(elapsed);
    }
    std::vector<float> repeated(output.size());
    ggml_backend_tensor_get(sum, repeated.data(), 0, ggml_nbytes(sum));
    okay &= std::memcmp(output.data(), repeated.data(), ggml_nbytes(sum)) == 0;
    std::sort(samples.begin(), samples.end());
    const double median = samples.empty() ? 0 : 0.5 * (samples[(samples.size() - 1) / 2] + samples[samples.size() / 2]);
    std::printf("%s k=%d rows=%d tokens=%d threads=%d padded=%d layers=%d error=%.9g hash=%016llx ms=%.6f\n",
                okay ? "PASS" : "FAIL", k, rows, tokens, threads, padded, layers, max_error, (unsigned long long) hash, median);
    std::fflush(stdout);
    if (!okay) std::abort();
    ggml_backend_buffer_free(storage);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

int main(int argc, char ** argv) {
    const bool timing = argc == 2 && std::string(argv[1]) == "--timing";
    if (argc != 1 && !timing) return 2;
    if (timing) {
        for (int tokens : {1, 5}) check(10240, 4, tokens, 15, false, true);
    } else {
        for (int threads : {1, 15}) for (bool padded : {false, true}) {
            for (int tokens : {1, 4, 5, 8, 9}) check(10240, 4, tokens, threads, padded, false);
            for (int k : {2048, 4096, 16384}) check(k, 4, 5, threads, padded, false);
            for (int rows : {1, 2, 8}) check(10240, rows, 5, threads, padded, false);
        }
    }
}
