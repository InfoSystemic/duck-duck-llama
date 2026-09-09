#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "ggml-cpu-impl.h"
#include "simd-mappings.h"
#include "repack.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include "q8-candidates.h"
#if defined(QWEN_Q8_BATCH)
#include "qwen-q8-batch.h"
#endif

using Fn = void (*)(int, float *, size_t, const void *, const void *, int, int);
static Fn functions[] = {ggml_gemv_q8_0_x16_q8_0, candidate1, candidate2, candidate4};

struct Fixture {
    int k, rows, tokens, nb;
    std::vector<block_q8_0_x16> weights;
    std::vector<block_q8_0> activations;
    std::vector<float> output;
    Fixture(int k_, int rows_, int tokens_, unsigned seed):
        k(k_), rows(rows_), tokens(tokens_), nb(k / 32),
        weights(size_t(rows / 16) * nb), activations(tokens * nb), output(tokens * rows + 2, -999.0f) {
        std::mt19937 rng(seed);
        for (auto & w : weights) {
            for (int r = 0; r < 16; ++r) w.d[r] = ggml_fp32_to_fp16(0.0004f * int(rng() % 31));
            for (auto & q : w.qs) q = uint8_t(rng());
        }
        for (auto & a : activations) {
            a.d = ggml_fp32_to_fp16(0.0007f * int(rng() % 29));
            for (auto & q : a.qs) q = int8_t(rng());
        }
    }
    void run(Fn fn, int batch = 1) {
        for (int tile = 0; tile < rows; tile += 64) {
            int width = std::min(64, rows - tile);
            for (int t = 0; t < tokens; t += batch)
                fn(k, output.data() + 1 + t * rows + tile, rows, weights.data() + size_t(tile / 16) * nb,
                   activations.data() + t * nb, std::min(batch, tokens - t), width);
        }
        if (output.front() != -999.0f || output.back() != -999.0f) std::abort();
    }
    float canonical(int t, int row) {
        float result = 0;
        for (int b = 0; b < nb; ++b) {
            const auto & w = weights[size_t(row / 16) * nb + b];
            const auto & a = activations[t * nb + b];
            int32_t dot = 0;
            for (int j = 0; j < 32; ++j)
                dot += (int(w.qs[(j / 4 * 16 + row % 16) * 4 + j % 4]) - 128) * int(a.qs[j]);
            result = std::fma(float(dot), ggml_fp16_to_fp32(w.d[row % 16]) * ggml_fp16_to_fp32(a.d), result);
        }
        return result;
    }
};

int main(int argc, char **) {
    ggml_cpu_init();
    auto ctx = ggml_init({1024 * 1024, nullptr, false});
    ggml_free(ctx);
    size_t compared = 0;
    for (int k : {32, 320, 640, 1536, 2560, 6144, 10240, 16384, 16416})
        for (int rows : {16, 64, 128})
            for (int tokens : {1, 2, 3, 4, 5, 6, 7, 8, 9}) {
                Fixture f(k, rows, tokens, 731 + k + rows + tokens);
                f.run(functions[0]);
                auto reference = f.output;
                for (int t = 0; t < tokens; ++t)
                    for (int r = 0; r < rows; ++r) {
                        float expected = f.canonical(t, r), actual = reference[1 + t * rows + r];
                        if (std::memcmp(&expected, &actual, 4)) {
                            std::fprintf(stderr, "canonical differs k=%d row=%d t=%d %.9g %.9g\n", k, r, t, expected, actual);
                            return 2;
                        }
                    }
                for (int mode = 1; mode < 4; ++mode) {
                    f.run(functions[mode]);
                    if (std::memcmp(reference.data(), f.output.data(), reference.size() * 4)) return 3;
                    compared += rows * tokens;
                }
#if defined(QWEN_Q8_BATCH)
                for (int batch : {2, 3, 4, 5, 6, 7, 8}) {
                    f.run(qwen_q8_batch_dispatch, batch);
                    if (std::memcmp(reference.data(), f.output.data(), reference.size() * 4)) return 4;
                    compared += rows * tokens;
                }
#endif
            }
    std::printf("{\"correctness_passed\":true,\"bit_exact_values\":%zu}\n", compared);
    std::fflush(stdout);
    if (argc > 1) return 0;
    volatile float sink = 0;
    for (int k : {640, 2560, 6144})
        for (int tokens : {4, 5, 6, 8, 9})
            for (bool cold : {false, true}) {
                const int rows = cold ? (int((64ULL << 20) / (544ULL * (k / 32) * 4)) * 64) : 64;
                Fixture f(k, rows, tokens, 911);
                for (int round = 0; round < 3; ++round)
                    for (int order = 0; order < 3; ++order) {
                        int mode = 1 + (order + round) % 3;
                        auto start = std::chrono::steady_clock::now();
                        double elapsed = 0;
                        int passes = 0;
                        do {
#if defined(QWEN_Q8_BATCH)
                            if (mode == 1) f.run(candidate1);
                            else f.run(qwen_q8_batch_dispatch, mode == 2 ? 4 : 8);
#else
                            f.run(functions[mode]);
#endif
                            sink = sink + f.output[1];
                            ++passes;
                            elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                        } while (elapsed < 0.12);
                        std::printf("{\"k\":%d,\"tokens\":%d,\"cold\":%s,\"round\":%d,\"chains\":%d,\"bytes\":%zu,\"passes\":%d,\"seconds\":%.9f,\"GB_s\":%.6f}\n",
                            k, tokens, cold ? "true" : "false", round, mode == 3 ? 8 : mode == 2 ? 4 : 1,
                            f.weights.size() * sizeof(block_q8_0_x16), passes, elapsed,
                            f.weights.size() * sizeof(block_q8_0_x16) * passes / elapsed / 1e9);
                        std::fflush(stdout);
                    }
            }
    return !std::isfinite(sink);
}
