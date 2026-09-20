#pragma once
#include <immintrin.h>
static inline int32_t glm_q8_activation_sum(const int8_t * values) {
    const __m256i q = _mm256_loadu_si256((const __m256i *) values);
    const __m256i biased = _mm256_xor_si256(q, _mm256_set1_epi8(char(0x80)));
    const __m256i sums = _mm256_sad_epu8(biased, _mm256_setzero_si256());
    const __m128i pairs = _mm_add_epi64(_mm256_castsi256_si128(sums), _mm256_extracti128_si256(sums, 1));
    return int32_t(_mm_cvtsi128_si64(pairs) + _mm_extract_epi64(pairs, 1)) - 4096;
}
// Private Q8 row batching. Each output retains the original integer and FP32 order.
template<int NR, int CHAINS, int SUM_MODE>
__attribute__((noinline)) static void glm_q8_batch_candidate(int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    GGML_ASSERT(nr == NR && n % QK8_0 == 0 && nc % 16 == 0);
    static_assert(CHAINS == 1 || CHAINS == 2 || CHAINS == 4);
    static_assert(SUM_MODE >= 0 && SUM_MODE <= 2);
    const bool inline_sum = SUM_MODE == 2 && nc == 16;
    const int nb = n / QK8_0;
    const auto * weights = (const block_q8_0_x16 *) vx;
    const auto * activations = (const block_q8_0 *) vy;
    int32_t sums_stack[NR * 512];
    float scales_stack[NR * 512];
    int32_t * sums = sums_stack;
    float * scales = scales_stack;
    std::vector<int32_t> sums_heap;
    std::vector<float> scales_heap;
    if (nb > 512 && !inline_sum) {
        sums_heap.resize(NR * nb); scales_heap.resize(NR * nb);
        sums = sums_heap.data(); scales = scales_heap.data();
    }
    if (!inline_sum) {
    for (int t = 0; t < NR; ++t) {
        for (int b = 0; b < nb; ++b) {
            const auto & a = activations[t * nb + b];
            if constexpr (SUM_MODE == 0) {
            const __m512i wide = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i *) a.qs));
            const __m256i s16 = _mm256_add_epi16(_mm512_castsi512_si256(wide), _mm512_extracti64x4_epi64(wide, 1));
            sums[t * nb + b] = _mm512_reduce_add_epi32(_mm512_cvtepi16_epi32(s16));
            } else {
                sums[t * nb + b] = glm_q8_activation_sum(a.qs);
            }
            scales[t * nb + b] = GGML_CPU_FP16_TO_FP32(a.d);
        }
    }
    }
    for (int g = 0; g < nc / 16; ++g) {
        const auto * wp = weights + size_t(g) * nb;
        __m512 result[NR];
        for (int t = 0; t < NR; ++t) result[t] = _mm512_setzero_ps();
        for (int b = 0; b < nb; ++b) {
            __m512i accum[NR][CHAINS];
            for (int t = 0; t < NR; ++t) for (int c = 0; c < CHAINS; ++c) accum[t][c] = _mm512_setzero_si512();
            for (int j = 0; j < 8; ++j) {
                const __m512i w = _mm512_loadu_si512((const void *) (wp[b].qs + j * 64));
                for (int t = 0; t < NR; ++t) {
                    const int8_t * q = activations[t * nb + b].qs + 4 * j;
                    accum[t][j % CHAINS] = _mm512_dpbusd_epi32(accum[t][j % CHAINS], w, _mm512_set1_epi32(*(const int32_t *) q));
                }
            }
            const __m512 wd = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) wp[b].d));
            for (int t = 0; t < NR; ++t) {
                __m512i total;
                if constexpr (CHAINS == 1) total = accum[t][0];
                else if constexpr (CHAINS == 2) total = _mm512_add_epi32(accum[t][0], accum[t][1]);
                else total = _mm512_add_epi32(_mm512_add_epi32(accum[t][0], accum[t][1]), _mm512_add_epi32(accum[t][2], accum[t][3]));
                const int32_t sum = inline_sum ? glm_q8_activation_sum(activations[t * nb + b].qs) : sums[t * nb + b];
                const float scale = inline_sum ? GGML_CPU_FP16_TO_FP32(activations[t * nb + b].d) : scales[t * nb + b];
                const __m512i corrected = _mm512_sub_epi32(total, _mm512_set1_epi32(128 * sum));
                result[t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(corrected),
                    _mm512_mul_ps(wd, _mm512_set1_ps(scale)), result[t]);
            }
        }
        for (int t = 0; t < NR; ++t) _mm512_storeu_ps(s + t * bs + g * 16, result[t]);
    }
}
