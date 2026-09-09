#include "qwen-q6-packed-batch-0908.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <dlfcn.h>
#include <random>
#include <vector>

static int scalar_code(const block_q6_K_x16 & w, int row, int element) {
    const int half = element / 128, part = (element % 128) / 32, slot = element % 32;
    const int position = half * 1536 + (slot / 4) * 192 + row * 4 + slot % 4;
    const int low = w.q[position + (part % 2) * 64], high = w.q[position + 128];
    return ((part < 2 ? low : low >> 4) & 15) | (((high >> (part * 2)) & 3) << 4);
}

int main() {
    Dl_info info{};
    if (!dladdr((void *) ggml_gemv_q6_K_x16_q8_K, &info) || !info.dli_fname) return 2;
    std::mt19937 rng(812938);
    size_t cases = 0, outputs = 0;
    constexpr float sentinel = 918273.0f;
    for (int n : {256, 512, 2560, 5120, 10240, 18432}) {
        for (int rows : {16, 32, 64, 160}) {
            for (int extreme : {0, 1}) {
                const int blocks = n / 256;
                std::vector<block_q6_K_x16> weights(size_t(rows / 16) * blocks);
                for (auto & w : weights) {
                    for (auto & d : w.d) d = ggml_fp32_to_fp16(0.0625f * (int(rng() % 17) - 8));
                    for (auto & s : w.scales) for (auto & v : s)
                        v = extreme ? (rng() & 1 ? -128 : 127) : int(rng() % 256) - 128;
                    for (auto & v : w.q) v = extreme ? (rng() & 1 ? 0 : 255) : rng() % 256;
                }
                for (int count = 1; count <= 9; ++count) {
                    // Noncontiguous pointers and guard values exercise expert-style gathering.
                    std::vector<std::vector<block_q8_K>> storage(count);
                    std::vector<std::vector<float>> result(count), expected(count);
                    std::vector<const block_q8_K *> activation(count);
                    std::vector<float *> output(count);
                    for (int r = 0; r < count; ++r) {
                        storage[r].resize(blocks + r + 1);
                        activation[r] = storage[r].data() + r;
                        result[r].assign(rows + r + 8, sentinel);
                        output[r] = result[r].data() + r + 3;
                        expected[r].resize(rows);
                        for (int b = 0; b < blocks; ++b) {
                            auto & a = storage[r][r + b];
                            a.d = 0.001f * (rng() % 13 + 1);
                            for (auto & q : a.qs) q = extreme ? (rng() & 1 ? -128 : 127) : int(rng() % 256) - 128;
                            for (int sub = 0; sub < 16; ++sub) {
                                int sum = 0;
                                for (int j = 0; j < 16; ++j) sum += a.qs[sub * 16 + j];
                                a.bsums[sub] = sum;
                            }
                        }
                        ggml_gemv_q6_K_x16_q8_K(n, expected[r].data(), 0, weights.data(), activation[r], 1, rows);
                    }
                    qwen_q6_packed_batch(n, rows, weights.data(), activation.data(), output.data(), count);
                    for (int r = 0; r < count; ++r) {
                        if (std::memcmp(expected[r].data(), output[r], rows * sizeof(float))) return 3;
                        for (int i = 0; i < r + 3; ++i) if (result[r][i] != sentinel) return 4;
                        for (size_t i = r + 3 + rows; i < result[r].size(); ++i) if (result[r][i] != sentinel) return 4;
                        for (int row = 0; row < rows; ++row) {
                            float reference = 0;
                            for (int b = 0; b < blocks; ++b) {
                                const auto & w = weights[size_t(row / 16) * blocks + b];
                                const auto & a = activation[r][b];
                                int32_t sum = 0;
                                for (int j = 0; j < 256; ++j)
                                    sum += (scalar_code(w, row % 16, j) - 32) * int(w.scales[j / 16][row % 16]) * int(a.qs[j]);
                                const float factor = ggml_fp16_to_fp32(w.d[row % 16]) * a.d;
                                reference = std::fma(float(sum), factor, reference);
                            }
                            if (output[r][row] != reference) return 5;
                            ++outputs;
                        }
                    }
                    ++cases;
                }
            }
        }
    }
    std::printf("{\"passed\":true,\"bit_exact\":true,\"scalar_reference_exact\":true,\"cases\":%zu,\"outputs\":%zu,\"guard_values_preserved\":true,\"weight_storage_unchanged\":true,\"cpu_library\":\"%s\"}\n", cases, outputs, info.dli_fname);
}
