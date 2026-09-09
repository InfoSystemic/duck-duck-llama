#include "qwen-q6-lossless-bytes-0908.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <random>
#include <vector>

static int native_code(const block_q6_K_x16 & w, int row, int element) {
    const int half = element / 128;
    const int part = (element % 128) / 32;
    const int slot = element % 32;
    const int position = half * 1536 + (slot / 4) * 192 + row * 4 + slot % 4;
    const int low = w.q[position + (part % 2) * 64];
    const int high = w.q[position + 128];
    return ((part < 2 ? low : low >> 4) & 15) | (((high >> (part * 2)) & 3) << 4);
}

int main() {
    Dl_info info{};
    if (!dladdr((void *) ggml_gemv_q6_K_x16_q8_K, &info) || !info.dli_fname) return 2;
    std::mt19937 rng(812938);
    size_t cases = 0, outputs = 0, codes = 0;
    float max_reference_error = 0;
    for (int n : {256, 512, 2560, 5120, 10240, 18432}) {
        for (int rows : {16, 32, 64, 160}) {
            for (int extreme : {0, 1}) {
                const int blocks = n / 256;
                std::vector<block_q6_K_x16> packed(size_t(rows / 16) * blocks);
                std::vector<qwen_q6_bytes_x16> expanded(packed.size());
                for (auto & w : packed) {
                    for (auto & d : w.d) d = ggml_fp32_to_fp16(0.0625f * (int(rng() % 17) - 8));
                    for (auto & s : w.scales) for (auto & v : s)
                        v = extreme ? (rng() & 1 ? -128 : 127) : int(rng() % 256) - 128;
                    for (auto & v : w.q) v = extreme ? (rng() & 1 ? 0 : 255) : rng() % 256;
                }
                qwen_q6_expand_x16(packed.data(), expanded.data(), packed.size());
                for (size_t b = 0; b < packed.size(); ++b) {
                    if (std::memcmp(packed[b].d, expanded[b].d, sizeof(packed[b].d)) ||
                        std::memcmp(packed[b].scales, expanded[b].scales, sizeof(packed[b].scales))) return 3;
                    for (int row = 0; row < 16; ++row) for (int j = 0; j < 256; ++j) {
                        if (native_code(packed[b],row,j) != expanded[b].qs[j/4][row*4+j%4]) return 4;
                        ++codes;
                    }
                }
                for (int activations : {1, 3, 5}) {
                    for (int activation = 0; activation < activations; ++activation) {
                        std::vector<block_q8_K> y(blocks);
                        for (auto & a : y) {
                            a.d = 0.001f * (rng() % 13 + 1);
                            for (auto & q : a.qs) q = extreme ? (rng() & 1 ? -128 : 127) : int(rng() % 256) - 128;
                            for (int sub = 0; sub < 16; ++sub) {
                                int value = 0;
                                for (int j = 0; j < 16; ++j) value += a.qs[sub*16+j];
                                a.bsums[sub] = value;
                            }
                        }
                        std::vector<float> baseline(rows), candidate(rows);
                        ggml_gemv_q6_K_x16_q8_K(n, baseline.data(), 0, packed.data(), y.data(), 1, rows);
                        qwen_gemv_q6_bytes_x16_q8_K(n, candidate.data(), expanded.data(), y.data(), rows);
                        if (std::memcmp(baseline.data(),candidate.data(),rows*sizeof(float))) return 5;
                        for (int row = 0; row < rows; ++row) {
                            float reference = 0;
                            for (int b = 0; b < blocks; ++b) {
                                const auto & w = packed[size_t(row/16)*blocks+b];
                                int32_t sum = 0;
                                for (int j = 0; j < 256; ++j)
                                    sum += (native_code(w,row%16,j)-32) * int(w.scales[j/16][row%16]) * int(y[b].qs[j]);
                                const float scale = ggml_fp16_to_fp32(w.d[row%16]) * y[b].d;
                                reference = std::fma(float(sum),scale,reference);
                            }
                            max_reference_error = std::max(max_reference_error,std::fabs(candidate[row]-reference));
                            if (candidate[row] != reference) return 6;
                            ++outputs;
                        }
                        ++cases;
                    }
                }
            }
        }
    }
    std::printf("{\"passed\":true,\"bit_exact\":true,\"cases\":%zu,\"outputs\":%zu,\"codes_preserved\":%zu,\"max_scalar_reference_error\":%.9g,\"native_block_bytes\":%zu,\"expanded_block_bytes\":%zu,\"cpu_library\":\"%s\"}\n",
                cases,outputs,codes,double(max_reference_error),sizeof(block_q6_K_x16),sizeof(qwen_q6_bytes_x16),info.dli_fname);
}
