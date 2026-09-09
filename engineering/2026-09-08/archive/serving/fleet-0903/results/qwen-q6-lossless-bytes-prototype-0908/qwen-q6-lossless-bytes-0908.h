#pragma once

#include "repack.h"
#include <cassert>
#include <cstring>
#include <immintrin.h>

// Expanded storage of the SAME Q6 codes, scales, and fp16 block factors.
// There is no floating-point conversion or re-quantization when repacking.
struct qwen_q6_bytes_x16 {
    ggml_half d[16];
    int8_t scales[16][16];
    uint8_t qs[64][64];
};
static_assert(sizeof(qwen_q6_bytes_x16) == 4384);

static inline void qwen_q6_expand_x16(const block_q6_K_x16 * src, qwen_q6_bytes_x16 * dst, size_t blocks) {
    for (size_t b = 0; b < blocks; ++b) {
        std::memcpy(dst[b].d, src[b].d, sizeof(src[b].d));
        std::memcpy(dst[b].scales, src[b].scales, sizeof(src[b].scales));
        for (int h = 0; h < 2; ++h) {
            for (int i = 0; i < 8; ++i) {
                const uint8_t * q = src[b].q + h * 1536 + i * 192;
                for (int lane = 0; lane < 64; ++lane) {
                    const uint8_t a = q[lane], z = q[lane + 64], high = q[lane + 128];
                    dst[b].qs[h*32 + i][lane]      = (a & 15) | ((high << 4) & 48);
                    dst[b].qs[h*32 + 8 + i][lane]  = (z & 15) | ((high << 2) & 48);
                    dst[b].qs[h*32 + 16 + i][lane] = (a >> 4) | (high & 48);
                    dst[b].qs[h*32 + 24 + i][lane] = (z >> 4) | ((high >> 2) & 48);
                }
            }
        }
    }
}

// This prototype implements one activation row. Multi-row and graph integration
// are deliberately separate follow-up work after correctness and timing checks.
static inline void qwen_gemv_q6_bytes_x16_q8_K(int n, float * output,
        const qwen_q6_bytes_x16 * weights, const block_q8_K * activation, int rows) {
    assert(n > 0 && n % 256 == 0 && rows > 0 && rows % 16 == 0);
    const int blocks = n / 256;
    for (int group = 0; group < rows / 16; ++group) {
        const qwen_q6_bytes_x16 * w = weights + size_t(group) * blocks;
        __m512 sum = _mm512_setzero_ps();
        for (int b = 0; b < blocks; ++b) {
            __m512i product = _mm512_setzero_si512();
            __m512i correction = _mm512_setzero_si512();
            const block_q8_K & y = activation[b];
            for (int sub = 0; sub < 16; ++sub) {
                __m512i dot = _mm512_setzero_si512();
                for (int step = 0; step < 4; ++step) {
                    int32_t packed;
                    std::memcpy(&packed, y.qs + sub * 16 + step * 4, sizeof(packed));
                    dot = _mm512_dpbusd_epi32(dot,
                        _mm512_loadu_si512(w[b].qs[sub * 4 + step]), _mm512_set1_epi32(packed));
                }
                const __m512i scale = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *) w[b].scales[sub]));
                product = _mm512_add_epi32(product, _mm512_mullo_epi32(dot, scale));
                correction = _mm512_add_epi32(correction,
                    _mm512_mullo_epi32(scale, _mm512_set1_epi32(y.bsums[sub])));
            }
            const __m512i corrected = _mm512_sub_epi32(product, _mm512_slli_epi32(correction, 5));
            const __m512 scales = _mm512_mul_ps(
                _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) w[b].d)), _mm512_set1_ps(y.d));
            sum = _mm512_fmadd_ps(_mm512_cvtepi32_ps(corrected), scales, sum);
        }
        _mm512_storeu_ps(output + group * 16, sum);
    }
}
