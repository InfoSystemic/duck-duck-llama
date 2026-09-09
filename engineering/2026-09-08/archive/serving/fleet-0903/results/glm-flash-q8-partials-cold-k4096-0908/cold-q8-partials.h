// NR=1 diagnostic, copied from source SHA256 133c8d2018a62cf063d76291160e1e0817d00b24c0d1eb879e88375af0fd746c
// Only integer additions regroup; FP32 block accumulation order is retained.
#pragma once
#include <immintrin.h>
#define GGML_X16_BC32(p) _mm512_set1_epi32(*(const int32_t *)(p))
template<int PARTS> __attribute__((noinline)) static void candidate_q8(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
    static_assert(PARTS == 1 || PARTS == 2 || PARTS == 4);
    GGML_ASSERT(nr == 1 && n % QK8_0 == 0 && nc % 16 == 0);
    GGML_UNUSED(bs);
    const int nb = n / QK8_0;
    const block_q8_0_x16 * vxb = (const block_q8_0_x16 *) vx; const block_q8_0 * vy8 = (const block_q8_0 *) vy;
    // per-block activation sums (for the +128 bias correction) and scales
    int32_t ysum_stack[512]; float yd_stack[512];
    int32_t * ysum = ysum_stack; float * yd = yd_stack;
    std::vector<int32_t> ysum_heap; std::vector<float> yd_heap;
    if (nb > 512) { ysum_heap.resize(nb); yd_heap.resize(nb); ysum = ysum_heap.data(); yd = yd_heap.data(); }
    for (int b = 0; b < nb; b++) {
        const __m256i q = _mm256_loadu_si256((const __m256i *) vy8[b].qs);
        const __m512i w = _mm512_cvtepi8_epi16(q);
        const __m256i lo = _mm512_castsi512_si256(w), hi = _mm512_extracti64x4_epi64(w, 1);
        const __m256i s16 = _mm256_add_epi16(lo, hi);
        ysum[b] = _mm512_reduce_add_epi32(_mm512_cvtepi16_epi32(s16));
        yd[b] = GGML_CPU_FP16_TO_FP32(vy8[b].d);
    }
    for (int g = 0; g < nc / 16; g++) { const block_q8_0_x16 * bp = vxb + (size_t) g * nb; __m512 accf = _mm512_setzero_ps();
        for (int b = 0; b < nb; b++) {
            __m512i acc = _mm512_setzero_si512();
            const uint8_t * q0 = bp[b].qs; const int8_t * y0 = vy8[b].qs;
            if constexpr (PARTS == 1) {
            for (int i0 = 0; i0 < 8; i0++) acc = _mm512_dpbusd_epi32(acc, _mm512_loadu_si512((const void *)(q0 + i0 * 64)), GGML_X16_BC32(y0 + 4*i0));
            } else {
                __m512i partial[PARTS];
                for (int p = 0; p < PARTS; ++p) partial[p] = _mm512_setzero_si512();
                for (int i0 = 0; i0 < 8; ++i0) {
                    partial[i0 % PARTS] = _mm512_dpbusd_epi32(partial[i0 % PARTS],
                        _mm512_loadu_si512((const void *)(q0 + i0 * 64)), GGML_X16_BC32(y0 + 4*i0));
                }
                acc = partial[0];
                for (int p = 1; p < PARTS; ++p) acc = _mm512_add_epi32(acc, partial[p]);
            }
            acc = _mm512_sub_epi32(acc, _mm512_set1_epi32(128 * ysum[b]));
            accf = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), _mm512_mul_ps(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) bp[b].d)), _mm512_set1_ps(yd[b])), accf);
        }
        _mm512_storeu_ps(s + g * 16, accf); }
    return;
#endif
    ggml_gemv_q8_0_x16_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

#undef GGML_X16_BC32
