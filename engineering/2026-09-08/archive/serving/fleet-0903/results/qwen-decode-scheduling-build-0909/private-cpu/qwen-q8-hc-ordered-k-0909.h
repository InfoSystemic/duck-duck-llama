#pragma once
#include <immintrin.h>

struct alignas(64) qwen_q8_hc_partial {
    int32_t dot[8];
    float scale[8];
};

static inline __m128i qwen_q8_hc_dot4(__m512i weights, __m128i activation) {
    __m512i dots = _mm512_dpbusd_epi32(_mm512_setzero_si512(), weights,
                                      _mm512_broadcast_i32x4(activation));
    dots = _mm512_add_epi32(dots, _mm512_shuffle_epi32(dots, (_MM_PERM_ENUM) _MM_SHUFFLE(2,3,0,1)));
    dots = _mm512_add_epi32(dots, _mm512_shuffle_epi32(dots, (_MM_PERM_ENUM) _MM_SHUFFLE(1,0,3,2)));
    return _mm512_castsi512_si128(_mm512_maskz_compress_epi32((__mmask16) 0x1111, dots));
}

static inline int qwen_q8_hc_sum16(__m128i values) {
    const __m128i pairs = _mm_maddubs_epi16(_mm_set1_epi8(1), values);
    __m128i sums = _mm_madd_epi16(pairs, _mm_set1_epi16(1));
    sums = _mm_hadd_epi32(sums, sums);
    sums = _mm_hadd_epi32(sums, sums);
    return _mm_cvtsi128_si32(sums);
}

static void qwen_q8_hc_prepare(int k, int nc, int nr, const block_q8_0x8 * weights,
                               const block_q8_0 * activation, int first, int last,
                               qwen_q8_hc_partial * partial) {
    const int nb = k / QK8_0, groups = nc / 8;
    for (int b = first; b < last; ++b) {
        __m128i aq[8][2];
        int sums[8];
        float scales[8];
        for (int row = 0; row < nr; ++row) {
            const block_q8_0 & a = activation[row * nb + b];
            aq[row][0] = _mm_loadu_si128((const __m128i *) a.qs);
            aq[row][1] = _mm_loadu_si128((const __m128i *) (a.qs + 16));
            sums[row] = qwen_q8_hc_sum16(aq[row][0]) + qwen_q8_hc_sum16(aq[row][1]);
            scales[row] = GGML_CPU_FP16_TO_FP32(a.d);
        }
        for (int g = 0; g < groups; ++g) {
            const block_q8_0x8 & w = weights[g * nb + b];
            const __m512i w0 = _mm512_loadu_si512((const void *) (w.qs + 0));
            const __m512i w1 = _mm512_loadu_si512((const void *) (w.qs + 64));
            const __m512i w2 = _mm512_loadu_si512((const void *) (w.qs + 128));
            const __m512i w3 = _mm512_loadu_si512((const void *) (w.qs + 192));
            const __m256 wd = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) w.d));
            for (int row = 0; row < nr; ++row) {
                const __m128i low = _mm_add_epi32(qwen_q8_hc_dot4(w0, aq[row][0]), qwen_q8_hc_dot4(w2, aq[row][1]));
                const __m128i high = _mm_add_epi32(qwen_q8_hc_dot4(w1, aq[row][0]), qwen_q8_hc_dot4(w3, aq[row][1]));
                const __m256i dot = _mm256_sub_epi32(_mm256_set_m128i(high, low), _mm256_set1_epi32(128 * sums[row]));
                qwen_q8_hc_partial & out = partial[(row * groups + g) * nb + b];
                _mm256_storeu_si256((__m256i *) out.dot, dot);
                _mm256_storeu_ps(out.scale, _mm256_mul_ps(wd, _mm256_set1_ps(scales[row])));
            }
        }
    }
}

static void qwen_q8_hc_finish(int k, int nc, int nr, const qwen_q8_hc_partial * partial,
                              float * output, size_t stride, int ith, int nth) {
    const int nb = k / QK8_0, groups = nc / 8;
    for (int task = ith; task < nr * groups; task += nth) {
        __m256 sum = _mm256_setzero_ps();
        const qwen_q8_hc_partial * p = partial + task * nb;
        for (int b = 0; b < nb; ++b) {
            sum = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i *) p[b].dot)),
                                 _mm256_loadu_ps(p[b].scale), sum);
        }
        _mm256_storeu_ps(output + (task / groups) * stride + (task % groups) * 8, sum);
    }
}
