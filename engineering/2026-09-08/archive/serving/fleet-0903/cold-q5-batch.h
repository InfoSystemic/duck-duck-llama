// Private multirow candidate for Full's existing Q5 x16 weight layout.
#pragma once
#include <cstring>
#include <immintrin.h>

namespace cold_q5_batch {
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
template<int SHIFT>
static inline __m512i high_bits(__m512i bits) {
    if constexpr (SHIFT >= 0) {
        return _mm512_and_si512(_mm512_slli_epi16(bits, SHIFT), _mm512_set1_epi8(0x10));
    } else {
        return _mm512_and_si512(_mm512_srli_epi16(bits, -SHIFT), _mm512_set1_epi8(0x10));
    }
}

template<int NR, int PAIR>
__attribute__((always_inline)) static inline void accumulate_pair(
        const block_q5_K_x16 & weight, const block_q8_K * input, int nb, int block,
        __m512i (&sums)[NR], __m512i (&minimums)[NR]) {
    __m512i low[NR], high[NR];
    for (int row = 0; row < NR; ++row) {
        low[row] = _mm512_setzero_si512();
        high[row] = _mm512_setzero_si512();
    }
    const __m512i mask = _mm512_set1_epi8(15);
#pragma GCC unroll 1
    for (int part = 0; part < 8; ++part) {
        const auto * packed = weight.qsh + part * 320;
        const __m512i nibble = _mm512_loadu_si512(packed + PAIR * 64);
        const __m512i fifth = _mm512_loadu_si512(packed + 256);
        const __m512i q_low = _mm512_or_si512(_mm512_and_si512(nibble, mask), high_bits<4 - 2 * PAIR>(fifth));
        const __m512i q_high = _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(nibble, 4), mask),
                                            high_bits<3 - 2 * PAIR>(fifth));
        for (int row = 0; row < NR; ++row) {
            const auto & activation = input[row * nb + block];
            int32_t a_low, a_high;
            std::memcpy(&a_low, activation.qs + PAIR * 64 + part * 4, sizeof(a_low));
            std::memcpy(&a_high, activation.qs + PAIR * 64 + 32 + part * 4, sizeof(a_high));
            low[row] = _mm512_dpbusd_epi32(low[row], q_low, _mm512_set1_epi32(a_low));
            high[row] = _mm512_dpbusd_epi32(high[row], q_high, _mm512_set1_epi32(a_high));
        }
    }
    const __m512i scale_low = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *) weight.scales[PAIR * 2]));
    const __m512i scale_high = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *) weight.scales[PAIR * 2 + 1]));
    const __m512i min_low = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *) weight.mins[PAIR * 2]));
    const __m512i min_high = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *) weight.mins[PAIR * 2 + 1]));
    for (int row = 0; row < NR; ++row) {
        const auto & activation = input[row * nb + block];
        sums[row] = _mm512_add_epi32(sums[row], _mm512_mullo_epi32(low[row], scale_low));
        sums[row] = _mm512_add_epi32(sums[row], _mm512_mullo_epi32(high[row], scale_high));
        const int32_t bs_low = int32_t(activation.bsums[PAIR * 4]) + activation.bsums[PAIR * 4 + 1];
        const int32_t bs_high = int32_t(activation.bsums[PAIR * 4 + 2]) + activation.bsums[PAIR * 4 + 3];
        minimums[row] = _mm512_add_epi32(minimums[row], _mm512_mullo_epi32(min_low, _mm512_set1_epi32(bs_low)));
        minimums[row] = _mm512_add_epi32(minimums[row], _mm512_mullo_epi32(min_high, _mm512_set1_epi32(bs_high)));
    }
}

template<int NR>
static void kernel(int n, float * output, size_t stride, const void * weights, const void * activations, int nc) {
    const int nb = n / QK_K;
    const auto * w = (const block_q5_K_x16 *) weights;
    const auto * a = (const block_q8_K *) activations;
    for (int group = 0; group < nc / 16; ++group) {
        const auto * blocks = w + size_t(group) * nb;
        __m512 accumulators[NR];
        for (int row = 0; row < NR; ++row) accumulators[row] = _mm512_setzero_ps();
        for (int block = 0; block < nb; ++block) {
            __m512i sums[NR], minimums[NR];
            for (int row = 0; row < NR; ++row) {
                sums[row] = _mm512_setzero_si512();
                minimums[row] = _mm512_setzero_si512();
            }
            accumulate_pair<NR, 0>(blocks[block], a, nb, block, sums, minimums);
            accumulate_pair<NR, 1>(blocks[block], a, nb, block, sums, minimums);
            accumulate_pair<NR, 2>(blocks[block], a, nb, block, sums, minimums);
            accumulate_pair<NR, 3>(blocks[block], a, nb, block, sums, minimums);
            const __m512 d = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) blocks[block].d));
            const __m512 dmin = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) blocks[block].dmin));
            for (int row = 0; row < NR; ++row) {
                const __m512 ad = _mm512_set1_ps(a[row * nb + block].d);
                accumulators[row] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(sums[row]), _mm512_mul_ps(d, ad), accumulators[row]);
                accumulators[row] = _mm512_fnmadd_ps(_mm512_cvtepi32_ps(minimums[row]), _mm512_mul_ps(dmin, ad), accumulators[row]);
            }
        }
        for (int row = 0; row < NR; ++row) _mm512_storeu_ps(output + row * stride + group * 16, accumulators[row]);
    }
}
#endif

static void gemv(int n, float * output, size_t stride, const void * weights, const void * activations, int nr, int nc) {
    GGML_ASSERT(nr >= 1 && nr <= 3 && n % QK_K == 0 && nc % 16 == 0 && (nr == 1 || stride >= size_t(nc)));
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
    if (nr == 3) return kernel<3>(n, output, stride, weights, activations, nc);
    if (nr == 2) return kernel<2>(n, output, stride, weights, activations, nc);
#endif
    for (int row = 0; row < nr; ++row) {
        ggml_gemv_q5_K_x16_q8_K(n, output + row * stride, 0, weights,
                               (const block_q8_K *) activations + row * (n / QK_K), 1, nc);
    }
}
} // namespace cold_q5_batch
