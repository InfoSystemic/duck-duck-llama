#pragma once
#include <immintrin.h>
#define GGML_X16_BC32(p) _mm512_set1_epi32(*(const int32_t *)(p))
static inline int32_t q8_activation_sum(const int8_t * values) {
    const __m256i q = _mm256_loadu_si256((const __m256i *) values);
    const __m256i biased = _mm256_xor_si256(q, _mm256_set1_epi8(char(0x80)));
    const __m256i sums = _mm256_sad_epu8(biased, _mm256_setzero_si256());
    const __m128i pairs = _mm_add_epi64(_mm256_castsi256_si128(sums), _mm256_extracti128_si256(sums, 1));
    return int32_t(_mm_cvtsi128_si64(pairs) + _mm_extract_epi64(pairs, 1)) - 4096;
}
template<int MODE> __attribute__((noinline)) static void candidate_q8(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
    static_assert(MODE >= 0 && MODE <= 2);
    GGML_ASSERT(nr == 1 && n % QK8_0 == 0 && nc % 16 == 0);
    GGML_UNUSED(bs);
    const int nb = n / QK8_0;
    const block_q8_0_x16 * vxb = (const block_q8_0_x16 *) vx; const block_q8_0 * vy8 = (const block_q8_0 *) vy;
    // per-block activation sums (for the +128 bias correction) and scales
    int32_t ysum_stack[512]; float yd_stack[512];
    int32_t * ysum = ysum_stack; float * yd = yd_stack;
    std::vector<int32_t> ysum_heap; std::vector<float> yd_heap;
    if (nb > 512 && !(MODE == 2 && nc == 16)) { ysum_heap.resize(nb); yd_heap.resize(nb); ysum = ysum_heap.data(); yd = yd_heap.data(); }
    if constexpr (MODE == 0) {
    for (int b = 0; b < nb; b++) {
        const __m256i q = _mm256_loadu_si256((const __m256i *) vy8[b].qs);
        const __m512i w = _mm512_cvtepi8_epi16(q);
        const __m256i lo = _mm512_castsi512_si256(w), hi = _mm512_extracti64x4_epi64(w, 1);
        const __m256i s16 = _mm256_add_epi16(lo, hi);
        ysum[b] = _mm512_reduce_add_epi32(_mm512_cvtepi16_epi32(s16));
        yd[b] = GGML_CPU_FP16_TO_FP32(vy8[b].d);
    }
    } else if (MODE != 2 || nc != 16) {
        for (int b = 0; b < nb; b++) {
            ysum[b] = q8_activation_sum(vy8[b].qs);
        yd[b] = GGML_CPU_FP16_TO_FP32(vy8[b].d);
    }
    }
    for (int g = 0; g < nc / 16; g++) { const block_q8_0_x16 * bp = vxb + (size_t) g * nb; __m512 accf = _mm512_setzero_ps();
        for (int b = 0; b < nb; b++) {
            __m512i acc = _mm512_setzero_si512();
            const uint8_t * q0 = bp[b].qs; const int8_t * y0 = vy8[b].qs;
            for (int i0 = 0; i0 < 8; i0++) acc = _mm512_dpbusd_epi32(acc, _mm512_loadu_si512((const void *)(q0 + i0 * 64)), GGML_X16_BC32(y0 + 4*i0));
            const int32_t sum = MODE == 2 && nc == 16 ? q8_activation_sum(vy8[b].qs) : ysum[b];
            const float scale = MODE == 2 && nc == 16 ? GGML_CPU_FP16_TO_FP32(vy8[b].d) : yd[b];
            acc = _mm512_sub_epi32(acc, _mm512_set1_epi32(128 * sum));
            accf = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), _mm512_mul_ps(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) bp[b].d)), _mm512_set1_ps(scale)), accf);
        }
        _mm512_storeu_ps(s + g * 16, accf); }
    return;
#endif
    ggml_gemv_q8_0_x16_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

#undef GGML_X16_BC32
