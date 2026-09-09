// Private Q8 row batching. Each output retains the original integer and FP32 order.
#pragma once
template<int NR>
static void qwen_q8_batch(int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    GGML_ASSERT(nr == NR && n % QK8_0 == 0 && nc % 16 == 0);
    const int nb = n / QK8_0;
    const auto * weights = (const block_q8_0_x16 *) vx;
    const auto * activations = (const block_q8_0 *) vy;
    int32_t sums_stack[NR * 512];
    float scales_stack[NR * 512];
    int32_t * sums = sums_stack;
    float * scales = scales_stack;
    std::vector<int32_t> sums_heap;
    std::vector<float> scales_heap;
    if (nb > 512) {
        sums_heap.resize(NR * nb); scales_heap.resize(NR * nb);
        sums = sums_heap.data(); scales = scales_heap.data();
    }
    for (int t = 0; t < NR; ++t) {
        for (int b = 0; b < nb; ++b) {
            const auto & a = activations[t * nb + b];
            const __m512i wide = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i *) a.qs));
            const __m256i s16 = _mm256_add_epi16(_mm512_castsi512_si256(wide), _mm512_extracti64x4_epi64(wide, 1));
            sums[t * nb + b] = _mm512_reduce_add_epi32(_mm512_cvtepi16_epi32(s16));
            scales[t * nb + b] = GGML_CPU_FP16_TO_FP32(a.d);
        }
    }
    for (int g = 0; g < nc / 16; ++g) {
        const auto * wp = weights + size_t(g) * nb;
        __m512 result[NR];
        for (int t = 0; t < NR; ++t) result[t] = _mm512_setzero_ps();
        for (int b = 0; b < nb; ++b) {
            __m512i accum[NR];
            for (int t = 0; t < NR; ++t) accum[t] = _mm512_setzero_si512();
            for (int j = 0; j < 8; ++j) {
                const __m512i w = _mm512_loadu_si512((const void *) (wp[b].qs + j * 64));
                for (int t = 0; t < NR; ++t) {
                    const int8_t * q = activations[t * nb + b].qs + 4 * j;
                    accum[t] = _mm512_dpbusd_epi32(accum[t], w, _mm512_set1_epi32(*(const int32_t *) q));
                }
            }
            const __m512 wd = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) wp[b].d));
            for (int t = 0; t < NR; ++t) {
                const __m512i corrected = _mm512_sub_epi32(accum[t], _mm512_set1_epi32(128 * sums[t * nb + b]));
                result[t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(corrected),
                    _mm512_mul_ps(wd, _mm512_set1_ps(scales[t * nb + b])), result[t]);
            }
        }
        for (int t = 0; t < NR; ++t) _mm512_storeu_ps(s + t * bs + g * 16, result[t]);
    }
}

static void qwen_q8_batch_dispatch(int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    switch (nr) {
        case 2: return qwen_q8_batch<2>(n, s, bs, vx, vy, nr, nc);
        case 3: return qwen_q8_batch<3>(n, s, bs, vx, vy, nr, nc);
        case 4: return qwen_q8_batch<4>(n, s, bs, vx, vy, nr, nc);
        case 5: return qwen_q8_batch<5>(n, s, bs, vx, vy, nr, nc);
        case 6: return qwen_q8_batch<6>(n, s, bs, vx, vy, nr, nc);
        case 7: return qwen_q8_batch<7>(n, s, bs, vx, vy, nr, nc);
        case 8: return qwen_q8_batch<8>(n, s, bs, vx, vy, nr, nc);
        default: GGML_ASSERT(nr == 1); return ggml_gemv_q8_0_x16_q8_0(n, s, bs, vx, vy, nr, nc);
    }
}
