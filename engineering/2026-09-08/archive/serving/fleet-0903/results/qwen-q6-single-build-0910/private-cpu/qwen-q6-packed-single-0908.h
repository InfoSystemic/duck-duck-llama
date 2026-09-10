#pragma once

#include "repack.h"
#include <cassert>
#include <cstring>
#include <immintrin.h>

// Apply the same sub-block correction to one, two, or three activation rows.
// Six-bit codes, scales, weight storage, and FP32 accumulation order are unchanged.
template<int NR>
static inline void qwen_q6_packed_batch_impl(int n, int rows,
        const block_q6_K_x16 * weights, const block_q8_K * const * activation,
        float * const * output) {
    static_assert(NR >= 1 && NR <= 3);
    const int blocks = n / 256;
    const __m512i low_mask = _mm512_set1_epi8(15);
    const __m512i high_mask = _mm512_set1_epi8(48);
    for (int group = 0; group < rows / 16; ++group) {
        const block_q6_K_x16 * w = weights + size_t(group) * blocks;
        __m512 sum[NR];
        for (int r = 0; r < NR; ++r) sum[r] = _mm512_setzero_ps();
        for (int b = 0; b < blocks; ++b) {
            __m512i product[NR];
            for (int r = 0; r < NR; ++r) product[r] = _mm512_setzero_si512();
            for (int half = 0; half < 2; ++half) {
                for (int part = 0; part < 2; ++part) {
                    __m512i dot[4][NR];
                    for (int quarter = 0; quarter < 4; ++quarter)
                        for (int r = 0; r < NR; ++r) dot[quarter][r] = _mm512_setzero_si512();
                    for (int step = 0; step < 4; ++step) {
                        const uint8_t * q = w[b].q + half * 1536 + (part * 4 + step) * 192;
                        const __m512i a = _mm512_loadu_si512(q);
                        const __m512i z = _mm512_loadu_si512(q + 64);
                        const __m512i high = _mm512_loadu_si512(q + 128);
                        const __m512i codes[4] = {
                            _mm512_or_si512(_mm512_and_si512(a, low_mask), _mm512_and_si512(_mm512_slli_epi16(high, 4), high_mask)),
                            _mm512_or_si512(_mm512_and_si512(z, low_mask), _mm512_and_si512(_mm512_slli_epi16(high, 2), high_mask)),
                            _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(a, 4), low_mask), _mm512_and_si512(high, high_mask)),
                            _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(z, 4), low_mask), _mm512_and_si512(_mm512_srli_epi16(high, 2), high_mask))
                        };
                        for (int quarter = 0; quarter < 4; ++quarter) {
                            for (int r = 0; r < NR; ++r) {
                                int32_t packed;
                                std::memcpy(&packed, activation[r][b].qs + half * 128 + quarter * 32 + part * 16 + step * 4, sizeof(packed));
                                dot[quarter][r] = _mm512_dpbusd_epi32(dot[quarter][r], codes[quarter], _mm512_set1_epi32(packed));
                            }
                        }
                    }
                    for (int quarter = 0; quarter < 4; ++quarter) {
                        const int sub = half * 8 + quarter * 2 + part;
                        const __m512i scale = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *) w[b].scales[sub]));
                        for (int r = 0; r < NR; ++r) {
                            const __m512i corrected = _mm512_sub_epi32(dot[quarter][r], _mm512_set1_epi32(32 * int(activation[r][b].bsums[sub])));
                            product[r] = _mm512_add_epi32(product[r], _mm512_mullo_epi32(corrected, scale));
                        }
                    }
                }
            }
            const __m512 factors = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) w[b].d));
            for (int r = 0; r < NR; ++r) {
                const __m512 scale = _mm512_mul_ps(factors, _mm512_set1_ps(activation[r][b].d));
                sum[r] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(product[r]), scale, sum[r]);
            }
        }
        for (int r = 0; r < NR; ++r) _mm512_storeu_ps(output[r] + group * 16, sum[r]);
    }
}

static inline void qwen_q6_packed_batch(int n, int rows,
        const block_q6_K_x16 * weights, const block_q8_K * const * activation,
        float * const * output, int count) {
    assert(n > 0 && n % 256 == 0 && rows > 0 && rows % 16 == 0 && count > 0);
    int r = 0;
    for (; r + 3 <= count; r += 3) qwen_q6_packed_batch_impl<3>(n, rows, weights, activation + r, output + r);
    if (r + 2 <= count) {
        qwen_q6_packed_batch_impl<2>(n, rows, weights, activation + r, output + r);
    } else if (r < count) {
        qwen_q6_packed_batch_impl<1>(n, rows, weights, activation + r, output + r);
    }
}
