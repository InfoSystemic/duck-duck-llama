// Generated from unchanged Full source SHA256 0d112976b6f81449df82915b358944a331fa692bb4fa3b399c463632c435b2bb
#pragma once
#include <immintrin.h>
#define GGML_X16_BC32(p) _mm512_set1_epi32(*(const int32_t *)(p))
template<int AHEAD> static void candidate_q4(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
    GGML_ASSERT(nr == 1 && n % QK_K == 0 && nc % 16 == 0);
    GGML_UNUSED(bs);
    const int nb = n / QK_K; const __m512i m4 = _mm512_set1_epi8(0x0F);
    const block_q4_K_x16 * vxb = (const block_q4_K_x16 *) vx; const block_q8_K * vy8 = (const block_q8_K *) vy;
    for (int g = 0; g < nc / 16; g++) { const block_q4_K_x16 * bp = vxb + (size_t) g * nb; __m512 accf = _mm512_setzero_ps();
        for (int b = 0; b < nb; b++) {
            if constexpr (AHEAD > 0) {
                if (g * nb + b + AHEAD < (nc / 16) * nb) {
                    const char * future = (const char *) (bp + b + AHEAD);
                    for (size_t offset = 0; offset < sizeof(*bp); offset += 64) {
                        _mm_prefetch(future + offset, _MM_HINT_T0);
                    }
                }
            }
 const block_q8_K * a = vy8 + b; const uint8_t * qs = bp[b].qs;
            __m512i iacc = _mm512_setzero_si512(), imin = _mm512_setzero_si512();
            for (int j = 0; j < 4; j++) {
                __m512i A0 = _mm512_setzero_si512(), A1 = A0, B0 = A0, B1 = A0; const uint8_t * qj = qs + j * 512; const int8_t * ya = a->qs + 64 * j; const int8_t * yb = ya + 32;
#define STEP(i0, AA, BB) { const __m512i v = _mm512_loadu_si512((const void *)(qj + (i0) * 64)); \
                AA = _mm512_dpbusd_epi32(AA, _mm512_and_si512(v, m4), GGML_X16_BC32(ya + 4*(i0))); \
                BB = _mm512_dpbusd_epi32(BB, _mm512_and_si512(_mm512_srli_epi16(v, 4), m4), GGML_X16_BC32(yb + 4*(i0))); }
                STEP(0,A0,B0) STEP(1,A1,B1) STEP(2,A0,B0) STEP(3,A1,B1) STEP(4,A0,B0) STEP(5,A1,B1) STEP(6,A0,B0) STEP(7,A1,B1)
#undef STEP
                const __m512i scA = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *) bp[b].scales[2*j]));
                const __m512i scB = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *) bp[b].scales[2*j+1]));
                iacc = _mm512_add_epi32(iacc, _mm512_mullo_epi32(_mm512_add_epi32(A0, A1), scA));
                iacc = _mm512_add_epi32(iacc, _mm512_mullo_epi32(_mm512_add_epi32(B0, B1), scB));
                const __m512i mA = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *) bp[b].mins[2*j]));
                const __m512i mB = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *) bp[b].mins[2*j+1]));
                imin = _mm512_add_epi32(imin, _mm512_mullo_epi32(mA, _mm512_set1_epi32((int32_t) a->bsums[4*j] + a->bsums[4*j+1])));
                imin = _mm512_add_epi32(imin, _mm512_mullo_epi32(mB, _mm512_set1_epi32((int32_t) a->bsums[4*j+2] + a->bsums[4*j+3])));
            }
            const __m512 ad = _mm512_set1_ps(a->d);
            accf = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc), _mm512_mul_ps(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) bp[b].d)), ad), accf);
            accf = _mm512_fnmadd_ps(_mm512_cvtepi32_ps(imin), _mm512_mul_ps(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) bp[b].dmin)), ad), accf);
        }
        _mm512_storeu_ps(s + g * 16, accf); }
    return;
#endif
    ggml_gemv_q4_K_x16_q8_K_generic(n, s, bs, vx, vy, nr, nc);
}

template<int AHEAD> static void candidate_q5(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
    GGML_ASSERT(nr == 1 && n % QK_K == 0 && nc % 16 == 0);
    GGML_UNUSED(bs);
    const int nb = n / QK_K; const __m512i m4 = _mm512_set1_epi8(0x0F); const __m512i m10 = _mm512_set1_epi8(0x10);
    const block_q5_K_x16 * vxb = (const block_q5_K_x16 *) vx; const block_q8_K * vy8 = (const block_q8_K *) vy;
    for (int g = 0; g < nc / 16; g++) { const block_q5_K_x16 * bp = vxb + (size_t) g * nb; __m512 accf = _mm512_setzero_ps();
        for (int b = 0; b < nb; b++) {
            if constexpr (AHEAD > 0) {
                if (g * nb + b + AHEAD < (nc / 16) * nb) {
                    const char * future = (const char *) (bp + b + AHEAD);
                    for (size_t offset = 0; offset < sizeof(*bp); offset += 64) {
                        _mm_prefetch(future + offset, _MM_HINT_T0);
                    }
                }
            }
 const block_q8_K * a = vy8 + b; const uint8_t * q = bp[b].qsh; const int8_t * y = a->qs;
            __m512i A0 = _mm512_setzero_si512(), B0 = A0, A1 = A0, B1 = A0, A2 = A0, B2 = A0, A3 = A0, B3 = A0;
#define Q5STEP(i0) { const uint8_t * qi = q + (i0) * 320; const __m512i h = _mm512_loadu_si512((const void *)(qi + 256)); \
            const __m512i v0 = _mm512_loadu_si512((const void *)(qi)), v1 = _mm512_loadu_si512((const void *)(qi + 64)); \
            const __m512i v2 = _mm512_loadu_si512((const void *)(qi + 128)), v3 = _mm512_loadu_si512((const void *)(qi + 192)); \
            A0 = _mm512_dpbusd_epi32(A0, _mm512_or_si512(_mm512_and_si512(v0, m4), _mm512_and_si512(_mm512_slli_epi16(h, 4), m10)), GGML_X16_BC32(y + 4*(i0))); \
            B0 = _mm512_dpbusd_epi32(B0, _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(v0, 4), m4), _mm512_and_si512(_mm512_slli_epi16(h, 3), m10)), GGML_X16_BC32(y + 32 + 4*(i0))); \
            A1 = _mm512_dpbusd_epi32(A1, _mm512_or_si512(_mm512_and_si512(v1, m4), _mm512_and_si512(_mm512_slli_epi16(h, 2), m10)), GGML_X16_BC32(y + 64 + 4*(i0))); \
            B1 = _mm512_dpbusd_epi32(B1, _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(v1, 4), m4), _mm512_and_si512(_mm512_slli_epi16(h, 1), m10)), GGML_X16_BC32(y + 96 + 4*(i0))); \
            A2 = _mm512_dpbusd_epi32(A2, _mm512_or_si512(_mm512_and_si512(v2, m4), _mm512_and_si512(h, m10)), GGML_X16_BC32(y + 128 + 4*(i0))); \
            B2 = _mm512_dpbusd_epi32(B2, _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(v2, 4), m4), _mm512_and_si512(_mm512_srli_epi16(h, 1), m10)), GGML_X16_BC32(y + 160 + 4*(i0))); \
            A3 = _mm512_dpbusd_epi32(A3, _mm512_or_si512(_mm512_and_si512(v3, m4), _mm512_and_si512(_mm512_srli_epi16(h, 2), m10)), GGML_X16_BC32(y + 192 + 4*(i0))); \
            B3 = _mm512_dpbusd_epi32(B3, _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(v3, 4), m4), _mm512_and_si512(_mm512_srli_epi16(h, 3), m10)), GGML_X16_BC32(y + 224 + 4*(i0))); }
            Q5STEP(0) Q5STEP(1) Q5STEP(2) Q5STEP(3) Q5STEP(4) Q5STEP(5) Q5STEP(6) Q5STEP(7)
#undef Q5STEP
            __m512i iacc = _mm512_setzero_si512(), imin = _mm512_setzero_si512();
            const __m512i accs[8] = {A0, B0, A1, B1, A2, B2, A3, B3};
            for (int sb = 0; sb < 8; sb++) {
                iacc = _mm512_add_epi32(iacc, _mm512_mullo_epi32(accs[sb], _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *) bp[b].scales[sb]))));
                imin = _mm512_add_epi32(imin, _mm512_mullo_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *) bp[b].mins[sb])), _mm512_set1_epi32((int32_t) a->bsums[2*sb] + a->bsums[2*sb+1])));
            }
            const __m512 ad = _mm512_set1_ps(a->d);
            accf = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc), _mm512_mul_ps(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) bp[b].d)), ad), accf);
            accf = _mm512_fnmadd_ps(_mm512_cvtepi32_ps(imin), _mm512_mul_ps(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) bp[b].dmin)), ad), accf);
        }
        _mm512_storeu_ps(s + g * 16, accf); }
    return;
#endif
    ggml_gemv_q5_K_x16_q8_K_generic(n, s, bs, vx, vy, nr, nc);
}

#undef GGML_X16_BC32
