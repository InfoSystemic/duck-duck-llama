#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml-backend-impl.h"

#include "ggml-impl.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include "simd-mappings.h"
#include "traits.h"

#include <cmath>
#include <vector>
#include <cstring>
#include <cassert>
#include <atomic>
#include <cstdlib> // for qsort
#include <cstdio>  // for GGML_ASSERT

#define GGML_CPU_CLANG_WORKAROUND
#include "/home/user/InfoSystemic/AI-Server/engines/llama.cpp-q4e-goal-0904/ggml/src/ggml-cpu/repack.h"

#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Woverlength-strings"
#endif

#define UNUSED GGML_UNUSED

#if defined(__AVX__)
#if defined(__F16C__)
#if defined(__AVX512F__)
#define GGML_F32Cx8x2_LOAD(x, y)     _mm512_cvtph_ps(_mm256_set_m128i(_mm_loadu_si128((const __m128i *)(y)), _mm_loadu_si128((const __m128i *)(x))))
#define GGML_F32Cx16_REPEAT_LOAD(x)  _mm512_cvtph_ps(_mm256_set_m128i(x, x))
#endif
// the  _mm256_cvt intrinsics require F16C
#define GGML_F32Cx8_LOAD(x)     _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(x)))
#define GGML_F32Cx8_REPEAT_LOAD(x, loadMask)     _mm256_cvtph_ps(_mm_shuffle_epi32(_mm_maskload_epi32((int const*)(x), loadMask), 68))
#define GGML_F32Cx8_REARRANGE_LOAD(x, arrangeMask)     _mm256_cvtph_ps(_mm_shuffle_epi8(_mm_loadu_si128((const __m128i *) x), arrangeMask))
#else
#if defined(__AVX512F__)
static inline __m512 __avx512_f32cx8x2_load(ggml_fp16_t *x, ggml_fp16_t *y) {
    float tmp[16];

    for (int i = 0; i < 8; i++) {
        tmp[i] = GGML_CPU_FP16_TO_FP32(x[i]);
    }

    for (int i = 0; i < 8; i++) {
        tmp[i + 8] = GGML_CPU_FP16_TO_FP32(y[i]);
    }

    return _mm512_loadu_ps(tmp);
}
static inline __m512 __avx512_repeat_f32cx16_load(__m128i x) {
    float tmp[16];
    uint16_t tmphalf[8];
    _mm_storeu_si128((__m128i*)tmphalf, x);

    for (int i = 0; i < 4; i++) {
        tmp[i] = GGML_CPU_FP16_TO_FP32(tmphalf[i]);
        tmp[i + 4] = GGML_CPU_FP16_TO_FP32(tmphalf[i]);
        tmp[i + 8] = GGML_CPU_FP16_TO_FP32(tmphalf[i]);
        tmp[i + 12] = GGML_CPU_FP16_TO_FP32(tmphalf[i]);
    }

    return _mm512_loadu_ps(tmp);
}
#endif
static inline __m256 __avx_f32cx8_load(ggml_fp16_t *x) {
    float tmp[8];

    for (int i = 0; i < 8; i++) {
        tmp[i] = GGML_CPU_FP16_TO_FP32(x[i]);
    }

    return _mm256_loadu_ps(tmp);
}
static inline __m256 __avx_repeat_f32cx8_load(ggml_fp16_t *x) {
    float tmp[8];

    for (int i = 0; i < 4; i++) {
        tmp[i] = GGML_CPU_FP16_TO_FP32(x[i]);
        tmp[i + 4] = GGML_CPU_FP16_TO_FP32(x[i]);
    }

    return _mm256_loadu_ps(tmp);
}
static inline __m256 __avx_rearranged_f32cx8_load(ggml_fp16_t *x, __m128i arrangeMask) {
    uint16_t tmphalf[8];
    float tmp[8];

    _mm_storeu_si128((__m128i*)tmphalf, _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *) x), arrangeMask));
    for (int i = 0; i < 8; i++) {
        tmp[i] = GGML_CPU_FP16_TO_FP32(tmphalf[i]);
    }

    return _mm256_loadu_ps(tmp);
}

#define GGML_F32Cx8_LOAD(x)     __avx_f32cx8_load(x)
#define GGML_F32Cx8_REPEAT_LOAD(x, loadMask)     __avx_repeat_f32cx8_load(x)
#define GGML_F32Cx8_REARRANGE_LOAD(x, arrangeMask)     __avx_rearranged_f32cx8_load(x, arrangeMask)
#if defined(__AVX512F__)
#define GGML_F32Cx8x2_LOAD(x, y)     __avx512_f32cx8x2_load(x, y)
#define GGML_F32Cx16_REPEAT_LOAD(x)  __avx512_repeat_f32cx16_load(x)
#endif
#endif
#endif

static inline int nearest_int(float fval) {
    assert(fabsf(fval) <= 4194303.f);
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

#if defined(__AVX2__) || defined(__AVX512F__)
#if defined(__AVX512F__)
// add int16_t pairwise and return as 512 bit int vector, then add the accumulator
static inline __m512i sum_i16_pairs_acc_int32x16(const __m512i acc, const __m512i x) {
    const __m512i ones = _mm512_set1_epi16(1);
    return _mm512_add_epi32(acc, _mm512_madd_epi16(ones, x));
}

static inline __m512i mul_sum_us8_pairs_acc_int32x16(const __m512i acc, const __m512i ax, const __m512i sy) {
#if defined(__AVX512VNNI__)
    return _mm512_dpbusd_epi32(acc, ax, sy);
#else
    // Perform multiplication and create 16-bit values
    const __m512i dot = _mm512_maddubs_epi16(ax, sy);
    return sum_i16_pairs_acc_int32x16(acc, dot);
#endif
}

// multiply int8_t, add results pairwise twice and return as 512 bit int vector，then add the accumulator
static inline __m512i mul_sum_i8_pairs_acc_int32x16(const __m512i acc, const __m512i x, const __m512i y) {
    const __m512i zero = _mm512_setzero_si512();
    // Get absolute values of x vectors
    const __m512i ax = _mm512_abs_epi8(x);
    // Sign the values of the y vectors
    __mmask64 blt0 = _mm512_movepi8_mask(x);
    const __m512i sy = _mm512_mask_sub_epi8(y, blt0, zero, y);
    return mul_sum_us8_pairs_acc_int32x16(acc, ax, sy);
}
#endif

// add int16_t pairwise and return as 256 bit int vector, then add the accumulator
static inline __m256i sum_i16_pairs_acc_int32x8(const __m256i acc, const __m256i x) {
    const __m256i ones = _mm256_set1_epi16(1);
    return _mm256_add_epi32(acc, _mm256_madd_epi16(ones, x));
}

static inline __m256i mul_sum_us8_pairs_acc_int32x8(const __m256i acc, const __m256i ax, const __m256i sy) {
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    return _mm256_dpbusd_epi32(acc, ax, sy);
#elif defined(__AVXVNNI__)
    return _mm256_dpbusd_avx_epi32(acc, ax, sy);
#else
    // Perform multiplication and create 16-bit values
    const __m256i dot = _mm256_maddubs_epi16(ax, sy);
    return sum_i16_pairs_acc_int32x8(acc, dot);
#endif
}

// Integer variant of the function defined in ggml-quants.c
// multiply int8_t, add results pairwise twice and return as 256 bit int vector, then add the accumulator
static inline __m256i mul_sum_i8_pairs_acc_int32x8(const __m256i acc, const __m256i x, const __m256i y) {
#if defined(__AVXVNNIINT8__)
    return _mm256_dpbssd_epi32(acc, x, y);
#else
    // Get absolute values of x vectors
    const __m256i ax = _mm256_sign_epi8(x, x);
    // Sign the values of the y vectors
    const __m256i sy = _mm256_sign_epi8(y, x);
    return mul_sum_us8_pairs_acc_int32x8(acc, ax, sy);
#endif
}
#endif

void ggml_quantize_mat_q8_0_4x8(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    block_q8_0x4 * GGML_RESTRICT y = (block_q8_0x4 *) vy;

#if defined(__AVX2__) || defined(__AVX__)
    float id[4];
    __m256 srcv[4][4];
    __m256 idvec[4];

    for (int i = 0; i < nb; i++) {
        for (int row_iter = 0; row_iter < 4; row_iter++) {
            // Load elements into 4 AVX vectors
            __m256 v0 = _mm256_loadu_ps( x + row_iter * k + i * 32 );
            __m256 v1 = _mm256_loadu_ps( x + row_iter * k + i * 32 + 8 );
            __m256 v2 = _mm256_loadu_ps( x + row_iter * k + i * 32 + 16 );
            __m256 v3 = _mm256_loadu_ps( x + row_iter * k + i * 32 + 24 );

            // Compute max(abs(e)) for the block
            const __m256 signBit = _mm256_set1_ps( -0.0f );
            __m256 maxAbs = _mm256_andnot_ps( signBit, v0 );
            maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v1 ) );
            maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v2 ) );
            maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v3 ) );

            __m128 max4 = _mm_max_ps( _mm256_extractf128_ps( maxAbs, 1 ), _mm256_castps256_ps128( maxAbs ) );
            max4 = _mm_max_ps( max4, _mm_movehl_ps( max4, max4 ) );
            max4 = _mm_max_ss( max4, _mm_movehdup_ps( max4 ) );
            const float maxScalar = _mm_cvtss_f32( max4 );

            // Divided by 127.f to mirror results in quantize_row_q8_0
            const float d = maxScalar  / 127.f;
            id[row_iter] = ( maxScalar != 0.0f ) ? 127.f / maxScalar : 0.0f; //d ? 1.0f / d : 0.0f;

            // Store the scale for the individual block
            y[i].d[row_iter] = GGML_CPU_FP32_TO_FP16(d);

            // Store the values in blocks of eight values - Aim is to use these later for block interleaving
            srcv[row_iter][0] = v0;
            srcv[row_iter][1] = v1;
            srcv[row_iter][2] = v2;
            srcv[row_iter][3] = v3;
            idvec[row_iter] = _mm256_set1_ps(id[row_iter]);
        }

        // The loop iterates four times - The aim is to get 4 corresponding chunks of eight bytes from the original weight blocks that are interleaved
        for (int j = 0; j < 4; j++) {
            // Apply the multiplier
            __m256 v0 = _mm256_mul_ps(srcv[0][j], idvec[0]);
            __m256 v1 = _mm256_mul_ps(srcv[1][j], idvec[1]);
            __m256 v2 = _mm256_mul_ps(srcv[2][j], idvec[2]);
            __m256 v3 = _mm256_mul_ps(srcv[3][j], idvec[3]);

            // Round to nearest integer
            v0 = _mm256_round_ps( v0, _MM_ROUND_NEAREST );
            v1 = _mm256_round_ps( v1, _MM_ROUND_NEAREST );
            v2 = _mm256_round_ps( v2, _MM_ROUND_NEAREST );
            v3 = _mm256_round_ps( v3, _MM_ROUND_NEAREST );

            // Convert floats to integers
            __m256i i0 = _mm256_cvtps_epi32( v0 );
            __m256i i1 = _mm256_cvtps_epi32( v1 );
            __m256i i2 = _mm256_cvtps_epi32( v2 );
            __m256i i3 = _mm256_cvtps_epi32( v3 );

#if defined(__AVX2__)
            // Convert int32 to int16
            i0 = _mm256_packs_epi32( i0, i1 );
            i2 = _mm256_packs_epi32( i2, i3 );
            // Convert int16 to int8
            i0 = _mm256_packs_epi16( i0, i2 );

            //  Permute and store the quantized weights in the required order after the pack instruction
            const __m256i perm = _mm256_setr_epi32( 0, 4, 1, 5, 2, 6, 3, 7 );
            i0 = _mm256_permutevar8x32_epi32( i0, perm );

            _mm256_storeu_si256((__m256i *)(y[i].qs + 32 * j), i0);
#else
            // Since we don't have in AVX some necessary functions,
            // we split the registers in half and call AVX2 analogs from SSE
            __m128i ni0 = _mm256_castsi256_si128( i0 );
            __m128i ni1 = _mm256_extractf128_si256( i0, 1);
            __m128i ni2 = _mm256_castsi256_si128( i1 );
            __m128i ni3 = _mm256_extractf128_si256( i1, 1);
            __m128i ni4 = _mm256_castsi256_si128( i2 );
            __m128i ni5 = _mm256_extractf128_si256( i2, 1);
            __m128i ni6 = _mm256_castsi256_si128( i3 );
            __m128i ni7 = _mm256_extractf128_si256( i3, 1);

            // Convert int32 to int16
            ni0 = _mm_packs_epi32( ni0, ni1 );
            ni2 = _mm_packs_epi32( ni2, ni3 );
            ni4 = _mm_packs_epi32( ni4, ni5 );
            ni6 = _mm_packs_epi32( ni6, ni7 );
            // Convert int16 to int8
            ni0 = _mm_packs_epi16( ni0, ni2 );
            ni4 = _mm_packs_epi16( ni4, ni6 );
            _mm_storeu_si128((__m128i *)(y[i].qs + 32 * j), ni0);
            _mm_storeu_si128((__m128i *)(y[i].qs + 32 * j + 16), ni4);
#endif
        }
    }

#else
    UNUSED(nb);
    UNUSED(y);
    ggml_quantize_mat_q8_0_4x8_generic(x, vy, k);
#endif
}

void ggml_quantize_mat_q8_K_4x8(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(QK_K == 256);
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    block_q8_Kx4 * GGML_RESTRICT y = (block_q8_Kx4 *) vy;

#if defined(__AVX2__)
    float iscale[4];
    __m256 srcv[4][32];
    __m256 iscale_vec[4];

    for (int i = 0; i < nb; i++) {
        for (int row_iter = 0; row_iter < 4; row_iter++) {
            // Load elements into 4 AVX vectors
            __m256 v0 = _mm256_loadu_ps( x + row_iter * k + i * 256 );
            __m256 v1 = _mm256_loadu_ps( x + row_iter * k + i * 256 + 8 );
            __m256 v2 = _mm256_loadu_ps( x + row_iter * k + i * 256 + 16 );
            __m256 v3 = _mm256_loadu_ps( x + row_iter * k + i * 256 + 24 );

            // Compute max(abs(e)) for the block
            const __m256 signBit = _mm256_set1_ps( -0.0f );
            __m256 abs0 = _mm256_andnot_ps( signBit, v0 );
            __m256 abs1 = _mm256_andnot_ps( signBit, v1 );
            __m256 abs2 = _mm256_andnot_ps( signBit, v2 );
            __m256 abs3 = _mm256_andnot_ps( signBit, v3 );

            __m256 maxAbs = _mm256_max_ps( abs0, abs1 );
            maxAbs = _mm256_max_ps( maxAbs, abs2 );
            maxAbs = _mm256_max_ps( maxAbs, abs3 );

            __m256 mask0 = _mm256_cmp_ps( maxAbs, v0, _CMP_EQ_OQ );
            __m256 mask1 = _mm256_cmp_ps( maxAbs, v1, _CMP_EQ_OQ );
            __m256 mask2 = _mm256_cmp_ps( maxAbs, v2, _CMP_EQ_OQ );
            __m256 mask3 = _mm256_cmp_ps( maxAbs, v3, _CMP_EQ_OQ );

            __m256 maskAbs = _mm256_or_ps(_mm256_or_ps(mask0, mask1),_mm256_or_ps(mask2, mask3));

            srcv[row_iter][0] = v0;
            srcv[row_iter][1] = v1;
            srcv[row_iter][2] = v2;
            srcv[row_iter][3] = v3;

            for (int sb = 1; sb < 8; sb++) {
                // Temporarily stores absolute quant values
                __m256 tempAbs = maxAbs;

                // Load elements into 4 AVX vectors
                __m256 v0 = _mm256_loadu_ps( x + row_iter * k + i * 256 + sb * 32);
                __m256 v1 = _mm256_loadu_ps( x + row_iter * k + i * 256 + sb * 32 + 8 );
                __m256 v2 = _mm256_loadu_ps( x + row_iter * k + i * 256 + sb * 32 + 16 );
                __m256 v3 = _mm256_loadu_ps( x + row_iter * k + i * 256 + sb * 32 + 24 );

                // Compute max(abs(e)) for the block
                __m256 abs0 = _mm256_andnot_ps( signBit, v0 );
                __m256 abs1 = _mm256_andnot_ps( signBit, v1 );
                __m256 abs2 = _mm256_andnot_ps( signBit, v2 );
                __m256 abs3 = _mm256_andnot_ps( signBit, v3 );

                maxAbs = _mm256_max_ps( maxAbs, abs0 );
                maxAbs = _mm256_max_ps( maxAbs, abs1 );
                maxAbs = _mm256_max_ps( maxAbs, abs2 );
                maxAbs = _mm256_max_ps( maxAbs, abs3 );

                __m256 mask_prev = _mm256_cmp_ps( tempAbs, maxAbs, _CMP_EQ_OQ );
                maskAbs = _mm256_and_ps( maskAbs, mask_prev );

                mask0 = _mm256_cmp_ps( maxAbs, v0, _CMP_EQ_OQ );
                mask1 = _mm256_cmp_ps( maxAbs, v1, _CMP_EQ_OQ );
                mask2 = _mm256_cmp_ps( maxAbs, v2, _CMP_EQ_OQ );
                mask3 = _mm256_cmp_ps( maxAbs, v3, _CMP_EQ_OQ );

                __m256 mask_curr = _mm256_or_ps(_mm256_or_ps(mask0, mask1),_mm256_or_ps(mask2, mask3));
                maskAbs =  _mm256_or_ps(maskAbs, mask_curr);

                srcv[row_iter][sb * 4] = v0;
                srcv[row_iter][sb * 4 + 1] = v1;
                srcv[row_iter][sb * 4 + 2] = v2;
                srcv[row_iter][sb * 4 + 3] = v3;
            }

            __m128 max4 = _mm_max_ps( _mm256_extractf128_ps( maxAbs, 1 ), _mm256_castps256_ps128( maxAbs ) );
            max4 = _mm_max_ps( max4, _mm_movehl_ps( max4, max4 ) );
            max4 = _mm_max_ss( max4, _mm_movehdup_ps( max4 ) );
            const float maxScalar = _mm_cvtss_f32( max4 );

            __m256 maxScalarVec = _mm256_set1_ps(maxScalar);

            __m256 mask_next = _mm256_cmp_ps( maxScalarVec, maxAbs, _CMP_EQ_OQ );
            __m256 finalMask = _mm256_and_ps(maskAbs, mask_next);

            const int mask = _mm256_movemask_ps(finalMask);
            iscale[row_iter] = ( maxScalar != 0.0f ) ? 127.f / maxScalar : 0.0f;

            if(mask) {
                iscale[row_iter] = ( maxScalar != 0.0f ) ? -127.f / maxScalar: 0.0f;
            }

            y[i].d[row_iter] = maxScalar ? 1/iscale[row_iter] : 0;
            iscale_vec[row_iter] = _mm256_set1_ps(iscale[row_iter]);
        }

        __m256i quants_interleaved[32];
        for (int j = 0; j < 32; j++) {
            // Apply the multiplier
            __m256 v0 = _mm256_mul_ps(srcv[0][j], iscale_vec[0]);
            __m256 v1 = _mm256_mul_ps(srcv[1][j], iscale_vec[1]);
            __m256 v2 = _mm256_mul_ps(srcv[2][j], iscale_vec[2]);
            __m256 v3 = _mm256_mul_ps(srcv[3][j], iscale_vec[3]);

            // Round to nearest integer
            v0 = _mm256_round_ps( v0, _MM_ROUND_NEAREST );
            v1 = _mm256_round_ps( v1, _MM_ROUND_NEAREST );
            v2 = _mm256_round_ps( v2, _MM_ROUND_NEAREST );
            v3 = _mm256_round_ps( v3, _MM_ROUND_NEAREST );

            // Convert floats to integers
            __m256i i0 = _mm256_cvtps_epi32( v0 );
            __m256i i1 = _mm256_cvtps_epi32( v1 );
            __m256i i2 = _mm256_cvtps_epi32( v2 );
            __m256i i3 = _mm256_cvtps_epi32( v3 );

            // Convert int32 to int16
            i0 = _mm256_packs_epi32( i0, i1 );
            i2 = _mm256_packs_epi32( i2, i3 );
            // Convert int16 to int8
            i0 = _mm256_packs_epi16( i0, i2 );

            //  Permute and store the quantized weights in the required order after the pack instruction
            const __m256i perm = _mm256_setr_epi32( 0, 4, 1, 5, 2, 6, 3, 7 );
            i0 = _mm256_permutevar8x32_epi32( i0, perm );

            _mm256_storeu_si256((__m256i *)(y[i].qs + 32 * j), i0);
            quants_interleaved[j] = i0;
        }

        // Masks to shuffle the quants of corresponding sub blocks for rearranging quants for vectorized bsums computation
        __m256i shuffle_mask_sb2 = _mm256_castsi128_si256(_mm_setr_epi8(0, 1, 0, 1, 4, 5, 6, 7, 8, 9, 8, 9, 12, 13, 14, 15));
        shuffle_mask_sb2 = _mm256_permute2f128_si256(shuffle_mask_sb2, shuffle_mask_sb2, 0);
        __m256i shuffle_mask_sb3 = _mm256_castsi128_si256(_mm_setr_epi8(0, 1, 2, 3, 0, 1, 6, 7, 8, 9, 10, 11, 8, 9, 14, 15));
        shuffle_mask_sb3 = _mm256_permute2f128_si256(shuffle_mask_sb3, shuffle_mask_sb3, 0);
        __m256i shuffle_mask_sb4 = _mm256_castsi128_si256(_mm_setr_epi8(0, 1, 2, 3, 4, 5, 0, 1, 8, 9, 10, 11, 12, 13, 8, 9));
        shuffle_mask_sb4 = _mm256_permute2f128_si256(shuffle_mask_sb4, shuffle_mask_sb4, 0);

        for (int k = 0; k < 4; k++) {
            // Quants from four different sub blocks are taken
            __m256i q0 = quants_interleaved[k * 8 + 0];
            __m256i q1 = quants_interleaved[k * 8 + 1];
            __m256i q2 = quants_interleaved[k * 8 + 2];
            __m256i q3 = quants_interleaved[k * 8 + 3];
            __m256i q4 = quants_interleaved[k * 8 + 4];
            __m256i q5 = quants_interleaved[k * 8 + 5];
            __m256i q6 = quants_interleaved[k * 8 + 6];
            __m256i q7 = quants_interleaved[k * 8 + 7];


            // The below code block has the first half of different sub blocks shuffled and blended so as to process 2 values from each sub block at a time
            __m256i sb2_h1_shuffled = _mm256_shuffle_epi8(q2, shuffle_mask_sb2);
            __m256i sb_h1_interleaved = _mm256_blend_epi16(q0, sb2_h1_shuffled, 34);
            __m256i sb3_h1_shuffled = _mm256_shuffle_epi8(q4, shuffle_mask_sb3);
            sb_h1_interleaved = _mm256_blend_epi16(sb_h1_interleaved, sb3_h1_shuffled, 68);
            __m256i sb4_h1_shuffled = _mm256_shuffle_epi8(q6, shuffle_mask_sb4);
            sb_h1_interleaved = _mm256_blend_epi16(sb_h1_interleaved, sb4_h1_shuffled, 136);

            __m256i one = _mm256_set1_epi8(1);
            __m256i bsums_r1 = _mm256_maddubs_epi16(one, sb_h1_interleaved);

            for (int l = 0; l < 3; l++) {
                // Quants value shifted to process next two values from each sub block
                q0 = _mm256_srli_epi64(q0, 16);
                q2 = _mm256_srli_epi64(q2, 16);
                q4 = _mm256_srli_epi64(q4, 16);
                q6 = _mm256_srli_epi64(q6, 16);

                sb2_h1_shuffled = _mm256_shuffle_epi8(q2, shuffle_mask_sb2);
                sb_h1_interleaved = _mm256_blend_epi16(q0, sb2_h1_shuffled, 34);
                sb3_h1_shuffled = _mm256_shuffle_epi8(q4, shuffle_mask_sb3);
                sb_h1_interleaved = _mm256_blend_epi16(sb_h1_interleaved, sb3_h1_shuffled, 68);
                sb4_h1_shuffled = _mm256_shuffle_epi8(q6, shuffle_mask_sb4);
                sb_h1_interleaved = _mm256_blend_epi16(sb_h1_interleaved, sb4_h1_shuffled, 136);

                bsums_r1 = _mm256_add_epi16(bsums_r1, _mm256_maddubs_epi16(one, sb_h1_interleaved));
            }

            // The below code block has the second half of different sub blocks shuffled and blended so as to process 2 values from each sub block at a time
            __m256i sb2_h2_shuffled = _mm256_shuffle_epi8(q3, shuffle_mask_sb2);
            __m256i sb_h2_interleaved = _mm256_blend_epi16(q1, sb2_h2_shuffled, 34);
            __m256i sb3_h2_shuffled = _mm256_shuffle_epi8(q5, shuffle_mask_sb3);
            sb_h2_interleaved = _mm256_blend_epi16(sb_h2_interleaved, sb3_h2_shuffled, 68);
            __m256i sb4_h2_shuffled = _mm256_shuffle_epi8(q7, shuffle_mask_sb4);
            sb_h2_interleaved = _mm256_blend_epi16(sb_h2_interleaved, sb4_h2_shuffled, 136);

            __m256i bsums_r2 = _mm256_maddubs_epi16(one, sb_h2_interleaved);

            for (int l = 0; l < 3; l++) {
                // Quants value shifted to process next two values from each sub block
                q1 = _mm256_srli_epi64(q1, 16);
                q3 = _mm256_srli_epi64(q3, 16);
                q5 = _mm256_srli_epi64(q5, 16);
                q7 = _mm256_srli_epi64(q7, 16);

                sb2_h2_shuffled = _mm256_shuffle_epi8(q3, shuffle_mask_sb2);
                sb_h2_interleaved = _mm256_blend_epi16(q1, sb2_h2_shuffled, 34);
                sb3_h2_shuffled = _mm256_shuffle_epi8(q5, shuffle_mask_sb3);
                sb_h2_interleaved = _mm256_blend_epi16(sb_h2_interleaved, sb3_h2_shuffled, 68);
                sb4_h2_shuffled = _mm256_shuffle_epi8(q7, shuffle_mask_sb4);
                sb_h2_interleaved = _mm256_blend_epi16(sb_h2_interleaved, sb4_h2_shuffled, 136);

                bsums_r2 = _mm256_add_epi16(bsums_r2, _mm256_maddubs_epi16(one, sb_h2_interleaved));
            }

            // Overall bsums in interleaved fashion computed by adding results of both halves
            __m256i bsums_r = _mm256_add_epi16(bsums_r1, bsums_r2);
            _mm256_storeu_si256((__m256i *)(y[i].bsums + 16 * k), bsums_r);
        }
    }

#else
    UNUSED(nb);
    UNUSED(y);
    ggml_quantize_mat_q8_K_4x8_generic(x, vy, k);
#endif
}

//
// GEMV/GEMM templates
//

#if defined(__AVX2__) || defined(__AVX512F__)

// GEMV for 8x blocks of 32 4-bit quants with a single scale factor per block
template<typename block_tx8>
static void gemv_q4_b32_8x8_q8_0_lut_avx(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc, __m256i signextendlut) {
    static_assert(
            std::is_same_v<block_tx8, block_q4_0x8> ||
            std::is_same_v<block_tx8, block_iq4_nlx8> ||
            std::is_same_v<block_tx8, block_mxfp4x8>,
            "Unsupported block type");

    const int qk = QK8_0;
    const int nb = n / qk;

    UNUSED(bs);

    __m256i finalpermutemask = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);

    // Permute mask used for easier vector processing at later stages
    const __m256i m4b = _mm256_set1_epi8(0x0F);

    int64_t b_nb = n / 32;

    const block_tx8  * b_ptr_start = (const block_tx8  *)vx;
    const block_q8_0 * a_ptr_start = (const block_q8_0 *)vy;

    // Process Q8_0 blocks one by one
    for (int64_t y = 0; y < nr; y++) {

        // Pointers to LHS blocks of block_q8_0 format
        const block_q8_0 * a_ptr = a_ptr_start + (y * nb);

        // Take group of eight blocks at each pass of the loop and perform dot product operation
        for (int64_t x = 0; x < nc / 8; x++) {

            // Pointers to RHS blocks
            const block_tx8 * b_ptr = b_ptr_start + (x * b_nb);

            // Master FP accumulator
            __m256 acc_row = _mm256_setzero_ps();

            for (int64_t b = 0; b < nb; b++) {
                // Load 8 blocks of 32 interleaved as 8 bytes (B0 - B7)
                const __m256i rhs_raw_vec_0123_0 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs));
                const __m256i rhs_raw_vec_4567_0 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs) + 1);
                const __m256i rhs_raw_vec_0123_1 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs) + 2);
                const __m256i rhs_raw_vec_4567_1 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs) + 3);

                // 4-bit -> 8-bit - Sign is maintained
                const __m256i rhs_vec_0123_0 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_vec_0123_0, m4b)); // B0(0-7) B1(0-7) B2(0-7) B3(0-7)
                const __m256i rhs_vec_4567_0 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_vec_4567_0, m4b)); // B4(0-7) B5(0-7) B6(0-7) B7(0-7)
                const __m256i rhs_vec_0123_1 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_vec_0123_1, m4b)); // B0(8-15) B1(8-15) B2(8-15) B3(8-15)
                const __m256i rhs_vec_4567_1 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_vec_4567_1, m4b)); // B0(8-15) B1(8-15) B2(8-15) B3(8-15)

                const __m256i rhs_vec_0123_2 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_0, 4), m4b)); // B0(16-23) B1(16-23) B2(16-23) B3(16-23)
                const __m256i rhs_vec_4567_2 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_0, 4), m4b)); // B4(16-23) B5(16-23) B6(16-23) B7(16-23)
                const __m256i rhs_vec_0123_3 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_1, 4), m4b)); // B0(24-31) B1(24-31) B2(24-31) B3(24-31)
                const __m256i rhs_vec_4567_3 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_1, 4), m4b)); // B4(24-31) B5(24-31) B6(24-31) B7(24-31)

                // Load the scale values for the 8 blocks interleaved in block_tx8
                __m256 col_scale_f32;
                if constexpr (
                        std::is_same_v<block_tx8, block_q4_0x8> ||
                        std::is_same_v<block_tx8, block_iq4_nlx8>) {
                    const __m128i changemask = _mm_set_epi8(15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0);
                    col_scale_f32 = GGML_F32Cx8_REARRANGE_LOAD(b_ptr[b].d, changemask);
                } else if constexpr (std::is_same_v<block_tx8, block_mxfp4x8>) {
                    // Load 8 E8M0 exponents and convert to float via LUT
                    // Rearranged to match changemask order: 0,4,1,5,2,6,3,7
                    col_scale_f32 = _mm256_set_ps(
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[b].e[7]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[b].e[3]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[b].e[6]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[b].e[2]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[b].e[5]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[b].e[1]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[b].e[4]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[b].e[0]));
                }

                // Load and convert to FP32 scale from block_q8_0
                const __m256 row_scale_f32 = _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(a_ptr[b].d));

                // Load the block values in block_q8_0 in batches of 16 bytes and replicate the same across 256 bit vector
                __m256i lhs_vec_0 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)a_ptr[b].qs));
                __m256i lhs_vec_1 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(a_ptr[b].qs + 16)));

                lhs_vec_0 = _mm256_permute2f128_si256(lhs_vec_0, lhs_vec_0, 0); // A0 (0-15) A0(0-15)
                lhs_vec_1 = _mm256_permute2f128_si256(lhs_vec_1, lhs_vec_1, 0); // A0 (16-31) A0(16-31))

                __m256i iacc = _mm256_setzero_si256();

                // Dot product done within 32 bit lanes and accumulated in the same vector
                // B0(0-3) B4(0-3) B1(0-3) B5(0-3) B2(0-3) B6(0-3) B3(0-3) B7(0-3) with A0(0-3)
                // B0(4-7) B4(4-7) B1(4-7) B5(4-7) B2(4-7) B6(4-7) B3(4-7) B7(4-7) with A0(4-7)
                // ...........................................................................
                // B0(28-31) B4(28-31) B1(28-31) B5(28-31) B2(28-31) B6(28-31) B3(28-31) B7(28-31) with A0(28-31)

                iacc = mul_sum_i8_pairs_acc_int32x8(iacc, _mm256_blend_epi32(rhs_vec_0123_0 ,_mm256_shuffle_epi32(rhs_vec_4567_0, 177), 170), _mm256_shuffle_epi32(lhs_vec_0, 0));
                iacc = mul_sum_i8_pairs_acc_int32x8(iacc, _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_0, 177) ,rhs_vec_4567_0, 170), _mm256_shuffle_epi32(lhs_vec_0, 85));

                iacc = mul_sum_i8_pairs_acc_int32x8(iacc, _mm256_blend_epi32(rhs_vec_0123_1 ,_mm256_shuffle_epi32(rhs_vec_4567_1, 177), 170), _mm256_shuffle_epi32(lhs_vec_0, 170));
                iacc = mul_sum_i8_pairs_acc_int32x8(iacc, _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_1, 177) ,rhs_vec_4567_1, 170), _mm256_shuffle_epi32(lhs_vec_0, 255));

                iacc = mul_sum_i8_pairs_acc_int32x8(iacc, _mm256_blend_epi32(rhs_vec_0123_2 ,_mm256_shuffle_epi32(rhs_vec_4567_2, 177), 170), _mm256_shuffle_epi32(lhs_vec_1, 0));
                iacc = mul_sum_i8_pairs_acc_int32x8(iacc, _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_2, 177) ,rhs_vec_4567_2, 170), _mm256_shuffle_epi32(lhs_vec_1, 85));

                iacc = mul_sum_i8_pairs_acc_int32x8(iacc, _mm256_blend_epi32(rhs_vec_0123_3 ,_mm256_shuffle_epi32(rhs_vec_4567_3, 177), 170), _mm256_shuffle_epi32(lhs_vec_1, 170));
                iacc = mul_sum_i8_pairs_acc_int32x8(iacc, _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_3, 177) ,rhs_vec_4567_3, 170), _mm256_shuffle_epi32(lhs_vec_1, 255));

                // Accumulated values multiplied with appropriate scales
                acc_row = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc), _mm256_mul_ps(col_scale_f32, row_scale_f32), acc_row);
            }

            // Accumulated output values permuted so as to be stored in appropriate order post accumulation
            acc_row = _mm256_permutevar8x32_ps(acc_row, finalpermutemask);
            _mm256_storeu_ps(s + (y * nr + x * 8), acc_row);
        }
    }
}

// GEMM for 8x blocks of 32 4-bit quants with a single scale factor per block
template<typename block_tx8>
static void gemm_q4_b32_8x8_q8_0_lut_avx(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc, __m256i signextendlut) {
    static_assert(
            std::is_same_v<block_tx8, block_q4_0x8> ||
            std::is_same_v<block_tx8, block_iq4_nlx8> ||
            std::is_same_v<block_tx8, block_mxfp4x8>,
            "Unsupported block type");

    const int qk = QK8_0;
    const int nb = n / qk;

    const block_tx8    * b_ptr_start = (const block_tx8    *)vx;
    const block_q8_0x4 * a_ptr_start = (const block_q8_0x4 *)vy;

    int64_t b_nb = n / 32;
    int64_t y = 0;
    // Mask to mask out nibbles from packed bytes
    const __m256i m4b = _mm256_set1_epi8(0x0F);
    const __m128i loadMask = _mm_blend_epi32(_mm_setzero_si128(), _mm_set1_epi32(0xFFFFFFFF), 3);
    // Permute mask used for easier vector processing at later stages
    __m256i requiredOrder = _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4);
    int64_t xstart = 0;
    int anr = nr - nr%16; // Used to align nr with boundary of 16
#if defined(__AVX512BW__) && defined(__AVX512DQ__)
    int anc = nc - nc%16; // Used to align nc with boundary of 16
                          // Mask to mask out nibbles from packed bytes expanded to 512 bit length
    const __m512i m4bexpanded = _mm512_set1_epi8(0x0F);
    // Lookup table to convert signed nibbles to signed bytes expanded to 512 bit length
    __m512i signextendlutexpanded = _mm512_inserti32x8(_mm512_castsi256_si512(signextendlut), signextendlut, 1);

    // Take group of four block_q8_0x4 structures at each pass of the loop and perform dot product operation
    for (; y < anr / 4; y += 4) {

        const block_q8_0x4 * a_ptrs[4];

        a_ptrs[0] = a_ptr_start + (y * nb);
        for (int i = 0; i < 3; ++i) {
            a_ptrs[i + 1] = a_ptrs[i] + nb;
        }

        // Take group of two block_tx8 structures at each pass of the loop and perform dot product operation
        for (int64_t x = 0; x < anc / 8; x += 2) {

            const block_tx8 * b_ptr_0 = b_ptr_start + ((x)     * b_nb);
            const block_tx8 * b_ptr_1 = b_ptr_start + ((x + 1) * b_nb);

            // Master FP accumulators
            __m512 acc_rows[16];
            for (int i = 0; i < 16; i++) {
                acc_rows[i] = _mm512_setzero_ps();
            }

            for (int64_t b = 0; b < nb; b++) {
                // Load the sixteen blocks of quantized values interleaved with each other in chunks of eight - B0,B1 ....BE,BF
                const __m256i rhs_raw_mat_0123_0 = _mm256_loadu_si256((const __m256i *)(b_ptr_0[b].qs));
                const __m256i rhs_raw_mat_4567_0 = _mm256_loadu_si256((const __m256i *)(b_ptr_0[b].qs + 32));
                const __m256i rhs_raw_mat_0123_1 = _mm256_loadu_si256((const __m256i *)(b_ptr_0[b].qs + 64));
                const __m256i rhs_raw_mat_4567_1 = _mm256_loadu_si256((const __m256i *)(b_ptr_0[b].qs + 96));

                const __m256i rhs_raw_mat_89AB_0 = _mm256_loadu_si256((const __m256i *)(b_ptr_1[b].qs));
                const __m256i rhs_raw_mat_CDEF_0 = _mm256_loadu_si256((const __m256i *)(b_ptr_1[b].qs + 32));
                const __m256i rhs_raw_mat_89AB_1 = _mm256_loadu_si256((const __m256i *)(b_ptr_1[b].qs + 64));
                const __m256i rhs_raw_mat_CDEF_1 = _mm256_loadu_si256((const __m256i *)(b_ptr_1[b].qs + 96));

                // Save the values in the following vectors in the formats B0B1B4B5B8B9BCBD, B2B3B6B7BABBBEBF for further processing and storing of values
                const __m256i rhs_raw_mat_0145_0 = _mm256_blend_epi32(rhs_raw_mat_0123_0, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_0, requiredOrder), 240);
                const __m256i rhs_raw_mat_2367_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_0, requiredOrder), rhs_raw_mat_4567_0, 240);
                const __m256i rhs_raw_mat_0145_1 = _mm256_blend_epi32(rhs_raw_mat_0123_1, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_1, requiredOrder), 240);
                const __m256i rhs_raw_mat_2367_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_1, requiredOrder), rhs_raw_mat_4567_1, 240);

                const __m256i rhs_raw_mat_89CD_0 = _mm256_blend_epi32(rhs_raw_mat_89AB_0, _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_0, requiredOrder), 240);
                const __m256i rhs_raw_mat_ABEF_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_0, requiredOrder), rhs_raw_mat_CDEF_0, 240);
                const __m256i rhs_raw_mat_89CD_1 = _mm256_blend_epi32(rhs_raw_mat_89AB_1, _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_1, requiredOrder), 240);
                const __m256i rhs_raw_mat_ABEF_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_1, requiredOrder), rhs_raw_mat_CDEF_1, 240);

                const __m512i rhs_raw_mat_014589CD_0 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_0145_0), rhs_raw_mat_89CD_0, 1);
                const __m512i rhs_raw_mat_2367ABEF_0 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_2367_0), rhs_raw_mat_ABEF_0, 1);
                const __m512i rhs_raw_mat_014589CD_1 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_0145_1), rhs_raw_mat_89CD_1, 1);
                const __m512i rhs_raw_mat_2367ABEF_1 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_2367_1), rhs_raw_mat_ABEF_1, 1);

                // 4-bit -> 8-bit - Sign is maintained
                const __m512i rhs_mat_014589CD_0 = _mm512_shuffle_epi8(signextendlutexpanded, _mm512_and_si512(rhs_raw_mat_014589CD_0, m4bexpanded)); //B0(0-7) B1(0-7) B4(0-7) B5(0-7) B8(0-7) B9(0-7) BC(0-7) BD(0-7)
                const __m512i rhs_mat_2367ABEF_0 = _mm512_shuffle_epi8(signextendlutexpanded, _mm512_and_si512(rhs_raw_mat_2367ABEF_0, m4bexpanded)); //B2(0-7) B3(0-7) B6(0-7) B7(0-7) BA(0-7) BB(0-7) BE(0-7) BF(0-7)

                const __m512i rhs_mat_014589CD_1 = _mm512_shuffle_epi8(signextendlutexpanded, _mm512_and_si512(rhs_raw_mat_014589CD_1, m4bexpanded)); //B0(8-15) B1(8-15) B4(8-15) B5(8-15) B8(8-15) B9(8-15) BC(8-15) BD(8-15)
                const __m512i rhs_mat_2367ABEF_1 = _mm512_shuffle_epi8(signextendlutexpanded, _mm512_and_si512(rhs_raw_mat_2367ABEF_1, m4bexpanded)); //B2(8-15) B3(8-15) B6(8-15) B7(8-15) BA(8-15) BB(8-15) BE(8-15) BF(8-15)

                const __m512i rhs_mat_014589CD_2 = _mm512_shuffle_epi8(signextendlutexpanded, _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_0, 4), m4bexpanded)); //B0(16-23) B1(16-23) B4(16-23) B5(16-23) B8(16-23) B9(16-23) BC(16-23) BD(16-23)
                const __m512i rhs_mat_2367ABEF_2 = _mm512_shuffle_epi8(signextendlutexpanded, _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_0, 4), m4bexpanded)); //B2(16-23) B3(16-23) B6(16-23) B7(16-23) BA(16-23) BB(16-23) BE(16-23) BF(16-23)

                const __m512i rhs_mat_014589CD_3 = _mm512_shuffle_epi8(signextendlutexpanded, _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_1, 4), m4bexpanded)); //B0(24-31) B1(24-31) B4(24-31) B5(24-31) B8(24-31) B9(24-31) BC(24-31) BD(24-31)
                const __m512i rhs_mat_2367ABEF_3 = _mm512_shuffle_epi8(signextendlutexpanded, _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_1, 4), m4bexpanded)); //B2(24-31) B3(24-31) B6(24-31) B7(24-31) BA(24-31) BB(24-31) BE(24-31) BF(24-31)

                // Shuffle pattern one - right side input
                const __m512i rhs_mat_014589CD_0_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_0, (_MM_PERM_ENUM)136); //B0(0-3) B1(0-3) B0(0-3) B1(0-3) B4(0-3) B5(0-3) B4(0-3) B5(0-3) B8(0-3) B9(0-3) B8(0-3) B9(0-3) BC(0-3) BD(0-3) BC(0-3) BD(0-3)
                const __m512i rhs_mat_2367ABEF_0_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_0, (_MM_PERM_ENUM)136); //B2(0-3) B3(0-3) B2(0-3) B3(0-3) B6(0-3) B7(0-3) B6(0-3) B7(0-3) BA(0-3) BB(0-3) BA(0-3) BB(0-3) BE(0-3) BF(0-3) BE(0-3) BF(0-3)

                const __m512i rhs_mat_014589CD_1_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_1, (_MM_PERM_ENUM)136); //B0(8-11) B1(8-11) B0(8-11) B1(8-11) B4(8-11) B5(8-11) B4(8-11) B5(8-11) B8(8-11) B9(8-11) B8(8-11) B9(8-11) BC(8-11) BD(8-11) BC(8-11) BD(8-11)
                const __m512i rhs_mat_2367ABEF_1_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_1, (_MM_PERM_ENUM)136); //B2(8-11) B3(8-11) B2(8-11) B3(8-11) B6(8-11) B7(8-11) B6(8-11) B7(8-11) BA(8-11) BB(8-11) BA(8-11) BB(8-11) BE(8-11) BF(8-11) BE(8-11) BF(8-11)

                const __m512i rhs_mat_014589CD_2_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_2, (_MM_PERM_ENUM)136); //B0(16-19) B1(16-19) B0(16-19) B1(16-19) B4(16-19) B5(16-19) B4(16-19) B5(16-19) B8(16-19) B9(16-19) B8(16-19) B9(16-19) BC(16-19) BD(16-19) BC(16-19) BD(16-19)
                const __m512i rhs_mat_2367ABEF_2_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_2, (_MM_PERM_ENUM)136); //B2(16-19) B3(16-19) B2(16-19) B3(16-19) B6(16-19) B7(16-19) B6(16-19) B7(16-19) BA(16-19) BB(16-19) BA(16-19) BB(16-19) BE(16-19) BF(16-19) BE(16-19) BF(16-19)

                const __m512i rhs_mat_014589CD_3_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_3, (_MM_PERM_ENUM)136); //B0(24-27) B1(24-27) B0(24-27) B1(24-27) B4(24-27) B5(24-27) B4(24-27) B5(24-27) B8(24-27) B9(24-27) B8(24-27) B9(24-27) BC(24-27) BD(24-27) BC(24-27) BD(24-27)
                const __m512i rhs_mat_2367ABEF_3_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_3, (_MM_PERM_ENUM)136); //B2(24-27) B3(24-27) B2(24-27) B3(24-27) B6(24-27) B7(24-27) B6(24-27) B7(24-27) BA(24-27) BB(24-27) BA(24-27) BB(24-27) BE(24-27) BF(24-27) BE(24-27) BF(24-27)

                // Shuffle pattern two - right side input

                const __m512i rhs_mat_014589CD_0_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_0, (_MM_PERM_ENUM)221); //B0(4-7) B1(4-7) B0(4-7) B1(4-7) B4(4-7) B5(4-7) B4(4-7) B5(4-7) B8(4-7) B9(4-7) B8(4-7) B9(4-7) BC(4-7) BD(4-7) BC(4-7) BD(4-7)
                const __m512i rhs_mat_2367ABEF_0_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_0, (_MM_PERM_ENUM)221); //B2(4-7) B3(4-7) B2(4-7) B3(4-7) B6(4-7) B7(4-7) B6(4-7) B7(4-7) BA(4-7) BB(4-7) BA(4-7) BB(4-7) BE(4-7) BF(4-7) BE(4-7) BF(4-7)

                const __m512i rhs_mat_014589CD_1_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_1, (_MM_PERM_ENUM)221); //B0(12-15) B1(12-15) B0(12-15) B1(12-15) B4(12-15) B5(12-15) B4(12-15) B5(12-15) B8(12-15) B9(12-15) B8(12-15) B9(12-15) BC(12-15) BD(12-15) BC(12-15) BD(12-15)
                const __m512i rhs_mat_2367ABEF_1_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_1, (_MM_PERM_ENUM)221); //B2(12-15) B3(12-15) B2(12-15) B3(12-15) B6(12-15) B7(12-15) B6(12-15) B7(12-15) BA(12-15) BB(12-15) BA(12-15) BB(12-15) BE(12-15) BF(12-15) BE(12-15) BF(12-15)

                const __m512i rhs_mat_014589CD_2_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_2, (_MM_PERM_ENUM)221); //B0(20-23) B1(20-23) B0(20-23) B1(20-23) B4(20-23) B5(20-23) B4(20-23) B5(20-23) B8(20-23) B9(20-23) B8(20-23) B9(20-23) BC(20-23) BD(20-23) BC(20-23) BD(20-23)
                const __m512i rhs_mat_2367ABEF_2_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_2, (_MM_PERM_ENUM)221); //B2(20-23) B3(20-23) B2(20-23) B3(20-23) B6(20-23) B7(20-23) B6(20-23) B7(20-23) BA(20-23) BB(20-23) BA(20-23) BB(20-23) BE(20-23) BF(20-23) BE(20-23) BF(20-23)

                const __m512i rhs_mat_014589CD_3_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_3, (_MM_PERM_ENUM)221); //B0(28-31) B1(28-31) B0(28-31) B1(28-31) B4(28-31) B5(28-31) B4(28-31) B5(28-31) B8(28-31) B9(28-31) B8(28-31) B9(28-31) BC(28-31) BD(28-31) BC(28-31) BD(28-31)
                const __m512i rhs_mat_2367ABEF_3_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_3, (_MM_PERM_ENUM)221); //B2(28-31) B3(28-31) B2(28-31) B3(28-31) B6(28-31) B7(28-31) B6(28-31) B7(28-31) BA(28-31) BB(28-31) BA(28-31) BB(28-31) BE(28-31) BF(28-31) BE(28-31) BF(28-31)

                // Scale values - Load the weight scale values of two block_tx8
                __m512 col_scale_f32;
                if constexpr (
                        std::is_same_v<block_tx8, block_q4_0x8> ||
                        std::is_same_v<block_tx8, block_iq4_nlx8>) {
                    col_scale_f32 = GGML_F32Cx8x2_LOAD(b_ptr_0[b].d, b_ptr_1[b].d);
                } else if constexpr (std::is_same_v<block_tx8, block_mxfp4x8>) {
                    //TODO: simd-ify
                    col_scale_f32 = _mm512_set_ps(
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_1[b].e[7]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_1[b].e[6]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_1[b].e[5]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_1[b].e[4]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_1[b].e[3]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_1[b].e[2]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_1[b].e[1]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_1[b].e[0]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_0[b].e[7]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_0[b].e[6]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_0[b].e[5]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_0[b].e[4]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_0[b].e[3]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_0[b].e[2]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_0[b].e[1]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_0[b].e[0]));
                }

                // Process LHS in pairs of rows
                for (int rp = 0; rp < 4; rp++) {

                    // Load the four blocks of quantized values interleaved with each other in chunks of eight - A0,A1,A2,A3
                    // Loaded as set of 128 bit vectors and repeated and stored into a 256 bit vector before again repeating into 512 bit vector
                    __m256i lhs_mat_ymm_0123_0 = _mm256_loadu_si256((const __m256i *)((a_ptrs[rp][b].qs)));
                    __m256i lhs_mat_ymm_01_0 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_0, lhs_mat_ymm_0123_0, 0);
                    __m256i lhs_mat_ymm_23_0 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_0, lhs_mat_ymm_0123_0, 17);
                    __m256i lhs_mat_ymm_0123_1 = _mm256_loadu_si256((const __m256i *)((a_ptrs[rp][b].qs + 32)));
                    __m256i lhs_mat_ymm_01_1 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_1, lhs_mat_ymm_0123_1, 0);
                    __m256i lhs_mat_ymm_23_1 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_1, lhs_mat_ymm_0123_1, 17);
                    __m256i lhs_mat_ymm_0123_2 = _mm256_loadu_si256((const __m256i *)((a_ptrs[rp][b].qs + 64)));
                    __m256i lhs_mat_ymm_01_2 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_2, lhs_mat_ymm_0123_2, 0);
                    __m256i lhs_mat_ymm_23_2 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_2, lhs_mat_ymm_0123_2, 17);
                    __m256i lhs_mat_ymm_0123_3 = _mm256_loadu_si256((const __m256i *)((a_ptrs[rp][b].qs + 96)));
                    __m256i lhs_mat_ymm_01_3 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_3, lhs_mat_ymm_0123_3, 0);
                    __m256i lhs_mat_ymm_23_3 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_3, lhs_mat_ymm_0123_3, 17);

                    __m512i lhs_mat_01_0 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_0), lhs_mat_ymm_01_0, 1);
                    __m512i lhs_mat_23_0 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_0), lhs_mat_ymm_23_0, 1);
                    __m512i lhs_mat_01_1 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_1), lhs_mat_ymm_01_1, 1);
                    __m512i lhs_mat_23_1 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_1), lhs_mat_ymm_23_1, 1);
                    __m512i lhs_mat_01_2 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_2), lhs_mat_ymm_01_2, 1);
                    __m512i lhs_mat_23_2 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_2), lhs_mat_ymm_23_2, 1);
                    __m512i lhs_mat_01_3 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_3), lhs_mat_ymm_01_3, 1);
                    __m512i lhs_mat_23_3 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_3), lhs_mat_ymm_23_3, 1);

                    // Shuffle pattern one - left side input

                    const __m512i lhs_mat_01_0_sp1 = _mm512_shuffle_epi32(lhs_mat_01_0, (_MM_PERM_ENUM)160);  //A0(0-3) A0(0-3) A1(0-3) A1(0-3) A0(0-3) A0(0-3) A1(0-3) A1(0-3) A0(0-3) A0(0-3) A1(0-3) A1(0-3) A0(0-3) A0(0-3) A1(0-3) A1(0-3)
                    const __m512i lhs_mat_23_0_sp1 = _mm512_shuffle_epi32(lhs_mat_23_0, (_MM_PERM_ENUM)160);  //A2(0-3) A2(0-3) A3(0-3) A3(0-3) A2(0-3) A2(0-3) A3(0-3) A3(0-3) A2(0-3) A2(0-3) A3(0-3) A3(0-3) A2(0-3) A2(0-3) A3(0-3) A3(0-3)

                    const __m512i lhs_mat_01_1_sp1 = _mm512_shuffle_epi32(lhs_mat_01_1, (_MM_PERM_ENUM)160);  //A0(8-11) A0(8-11) A1(8-11) A1(8-11) A0(8-11) A0(8-11) A1(8-11) A1(8-11) A0(8-11) A0(8-11) A1(8-11) A1(8-11) A0(8-11) A0(8-11) A1(8-11) A1(8-11)
                    const __m512i lhs_mat_23_1_sp1 = _mm512_shuffle_epi32(lhs_mat_23_1, (_MM_PERM_ENUM)160);  //A2(8-11) A2(8-11) A3(8-11) A3(8-11) A2(8-11) A2(8-11) A3(8-11) A3(8-11) A2(8-11) A2(8-11) A3(8-11) A3(8-11) A2(8-11) A2(8-11) A3(8-11) A3(8-11)

                    const __m512i lhs_mat_01_2_sp1 = _mm512_shuffle_epi32(lhs_mat_01_2, (_MM_PERM_ENUM)160);  //A0(16-19) A0(16-19) A1(16-19) A1(16-19) A0(16-19) A0(16-19) A1(16-19) A1(16-19) A0(16-19) A0(16-19) A1(16-19) A1(16-19) A0(16-19) A0(16-19) A1(16-19) A1(16-19)
                    const __m512i lhs_mat_23_2_sp1 = _mm512_shuffle_epi32(lhs_mat_23_2, (_MM_PERM_ENUM)160);  //A2(16-19) A2(16-19) A3(16-19) A3(16-19) A2(16-19) A2(16-19) A3(16-19) A3(16-19) A2(16-19) A2(16-19) A3(16-19) A3(16-19) A2(16-19) A2(16-19) A3(16-19) A3(16-19)

                    const __m512i lhs_mat_01_3_sp1 = _mm512_shuffle_epi32(lhs_mat_01_3, (_MM_PERM_ENUM)160);  //A0(24-27) A0(24-27) A1(24-27) A1(24-27) A0(24-27) A0(24-27) A1(24-27) A1(24-27) A0(24-27) A0(24-27) A1(24-27) A1(24-27) A0(24-27) A0(24-27) A1(24-27) A1(24-27)
                    const __m512i lhs_mat_23_3_sp1 = _mm512_shuffle_epi32(lhs_mat_23_3, (_MM_PERM_ENUM)160);  //A2(24-27) A2(24-27) A3(24-27) A3(24-27) A2(24-27) A2(24-27) A3(24-27) A3(24-27) A2(24-27) A2(24-27) A3(24-27) A3(24-27) A2(24-27) A2(24-27) A3(24-27) A3(24-27)

                    // Shuffle pattern two - left side input

                    const __m512i lhs_mat_01_0_sp2 = _mm512_shuffle_epi32(lhs_mat_01_0, (_MM_PERM_ENUM)245);  //A0(4-7) A0(4-7) A1(4-7) A1(4-7) A0(4-7) A0(4-7) A1(4-7) A1(4-7) A0(4-7) A0(4-7) A1(4-7) A1(4-7) A0(4-7) A0(4-7) A1(4-7) A1(4-7)
                    const __m512i lhs_mat_23_0_sp2 = _mm512_shuffle_epi32(lhs_mat_23_0, (_MM_PERM_ENUM)245);  //A2(4-7) A2(4-7) A3(4-7) A3(4-7) A2(4-7) A2(4-7) A3(4-7) A3(4-7) A2(4-7) A2(4-7) A3(4-7) A3(4-7) A2(4-7) A2(4-7) A3(4-7) A3(4-7)

                    const __m512i lhs_mat_01_1_sp2 = _mm512_shuffle_epi32(lhs_mat_01_1, (_MM_PERM_ENUM)245);  //A0(12-15) A0(12-15) A1(12-15) A1(12-15) A0(12-15) A0(12-15) A1(12-15) A1(12-15) A0(12-15) A0(12-15) A1(12-15) A1(12-15) A0(12-15) A0(12-15) A1(12-15) A1(12-15)
                    const __m512i lhs_mat_23_1_sp2 = _mm512_shuffle_epi32(lhs_mat_23_1, (_MM_PERM_ENUM)245);  //A2(12-15) A2(12-15) A3(12-15) A3(12-15) A2(12-15) A2(12-15) A3(12-15) A3(12-15) A2(12-15) A2(12-15) A3(12-15) A3(12-15) A2(12-15) A2(12-15) A3(12-15) A3(12-15)

                    const __m512i lhs_mat_01_2_sp2 = _mm512_shuffle_epi32(lhs_mat_01_2, (_MM_PERM_ENUM)245);  //A0(20-23) A0(20-23) A1(20-23) A1(20-23) A0(20-23) A0(20-23) A1(20-23) A1(20-23) A0(20-23) A0(20-23) A1(20-23) A1(20-23) A0(20-23) A0(20-23) A1(20-23) A1(20-23)
                    const __m512i lhs_mat_23_2_sp2 = _mm512_shuffle_epi32(lhs_mat_23_2, (_MM_PERM_ENUM)245);  //A2(20-23) A2(20-23) A3(20-23) A3(20-23) A2(20-23) A2(20-23) A3(20-23) A3(20-23) A2(20-23) A2(20-23) A3(20-23) A3(20-23) A2(20-23) A2(20-23) A3(20-23) A3(20-23)

                    const __m512i lhs_mat_01_3_sp2 = _mm512_shuffle_epi32(lhs_mat_01_3, (_MM_PERM_ENUM)245);  //A0(28-31) A0(28-31) A1(28-31) A1(28-31) A0(28-31) A0(28-31) A1(28-31) A1(28-31) A0(28-31) A0(28-31) A1(28-31) A1(28-31) A0(28-31) A0(28-31) A1(28-31) A1(28-31)
                    const __m512i lhs_mat_23_3_sp2 = _mm512_shuffle_epi32(lhs_mat_23_3, (_MM_PERM_ENUM)245);  //A2(28-31) A2(28-31) A3(28-31) A3(28-31) A2(28-31) A2(28-31) A3(28-31) A3(28-31) A2(28-31) A2(28-31) A3(28-31) A3(28-31) A2(28-31) A2(28-31) A3(28-31) A3(28-31)

                    // The values arranged in shuffle patterns are operated with dot product operation within 32 bit lane i.e corresponding bytes and multiplied and added into 32 bit integers within 32 bit lane
                    // Resembles MMLAs into 2x2 matrices in ARM Version
                    const __m512i zero = _mm512_setzero_epi32();
                    __m512i iacc_mat_00_sp1 = mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_01_3_sp1, rhs_mat_014589CD_3_sp1), lhs_mat_01_2_sp1, rhs_mat_014589CD_2_sp1), lhs_mat_01_1_sp1, rhs_mat_014589CD_1_sp1), lhs_mat_01_0_sp1, rhs_mat_014589CD_0_sp1);
                    __m512i iacc_mat_01_sp1 = mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_01_3_sp1, rhs_mat_2367ABEF_3_sp1), lhs_mat_01_2_sp1, rhs_mat_2367ABEF_2_sp1), lhs_mat_01_1_sp1, rhs_mat_2367ABEF_1_sp1), lhs_mat_01_0_sp1, rhs_mat_2367ABEF_0_sp1);
                    __m512i iacc_mat_10_sp1 = mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_23_3_sp1, rhs_mat_014589CD_3_sp1), lhs_mat_23_2_sp1, rhs_mat_014589CD_2_sp1), lhs_mat_23_1_sp1, rhs_mat_014589CD_1_sp1), lhs_mat_23_0_sp1, rhs_mat_014589CD_0_sp1);
                    __m512i iacc_mat_11_sp1 = mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_23_3_sp1, rhs_mat_2367ABEF_3_sp1), lhs_mat_23_2_sp1, rhs_mat_2367ABEF_2_sp1), lhs_mat_23_1_sp1, rhs_mat_2367ABEF_1_sp1), lhs_mat_23_0_sp1, rhs_mat_2367ABEF_0_sp1);
                    __m512i iacc_mat_00_sp2 = mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_01_3_sp2, rhs_mat_014589CD_3_sp2), lhs_mat_01_2_sp2, rhs_mat_014589CD_2_sp2), lhs_mat_01_1_sp2, rhs_mat_014589CD_1_sp2), lhs_mat_01_0_sp2, rhs_mat_014589CD_0_sp2);
                    __m512i iacc_mat_01_sp2 = mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_01_3_sp2, rhs_mat_2367ABEF_3_sp2), lhs_mat_01_2_sp2, rhs_mat_2367ABEF_2_sp2), lhs_mat_01_1_sp2, rhs_mat_2367ABEF_1_sp2), lhs_mat_01_0_sp2, rhs_mat_2367ABEF_0_sp2);
                    __m512i iacc_mat_10_sp2 = mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_23_3_sp2, rhs_mat_014589CD_3_sp2), lhs_mat_23_2_sp2, rhs_mat_014589CD_2_sp2), lhs_mat_23_1_sp2, rhs_mat_014589CD_1_sp2), lhs_mat_23_0_sp2, rhs_mat_014589CD_0_sp2);
                    __m512i iacc_mat_11_sp2 = mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_23_3_sp2, rhs_mat_2367ABEF_3_sp2), lhs_mat_23_2_sp2, rhs_mat_2367ABEF_2_sp2), lhs_mat_23_1_sp2, rhs_mat_2367ABEF_1_sp2), lhs_mat_23_0_sp2, rhs_mat_2367ABEF_0_sp2);

                    // Output of both shuffle patterns are added in order to sum dot product outputs of all 32 values in block
                    __m512i iacc_mat_00 = _mm512_add_epi32(iacc_mat_00_sp1, iacc_mat_00_sp2);
                    __m512i iacc_mat_01 = _mm512_add_epi32(iacc_mat_01_sp1, iacc_mat_01_sp2);
                    __m512i iacc_mat_10 = _mm512_add_epi32(iacc_mat_10_sp1, iacc_mat_10_sp2);
                    __m512i iacc_mat_11 = _mm512_add_epi32(iacc_mat_11_sp1, iacc_mat_11_sp2);


                    // Straighten out to make 4 row vectors
                    __m512i iacc_row_0 = _mm512_mask_blend_epi32(0xCCCC, iacc_mat_00, _mm512_shuffle_epi32(iacc_mat_01, (_MM_PERM_ENUM)78));
                    __m512i iacc_row_1 = _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32(iacc_mat_00, (_MM_PERM_ENUM)78), iacc_mat_01);
                    __m512i iacc_row_2 = _mm512_mask_blend_epi32(0xCCCC, iacc_mat_10, _mm512_shuffle_epi32(iacc_mat_11, (_MM_PERM_ENUM)78));
                    __m512i iacc_row_3 = _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32(iacc_mat_10, (_MM_PERM_ENUM)78), iacc_mat_11);

                    // Load the scale(d) values for all the 4 Q8_0 blocks and repeat it across lanes
                    const __m128i row_scale_f16 = _mm_shuffle_epi32(_mm_maskload_epi32((int const*)(a_ptrs[rp][b].d), loadMask), 68);
                    const __m512 row_scale_f32 = GGML_F32Cx16_REPEAT_LOAD(row_scale_f16);

                    // Multiply with appropriate scales and accumulate
                    acc_rows[rp * 4]     = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_0), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 0)),   acc_rows[rp * 4]);
                    acc_rows[rp * 4 + 1] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_1), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 85)),  acc_rows[rp * 4 + 1]);
                    acc_rows[rp * 4 + 2] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_2), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_rows[rp * 4 + 2]);
                    acc_rows[rp * 4 + 3] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_3), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_rows[rp * 4 + 3]);
                }
            }

            // Store the accumulated values
            for (int i = 0; i < 16; i++) {
                _mm512_storeu_ps((float *)(s + ((y * 4 + i) * bs + x * 8)), acc_rows[i]);
            }
        }
    }

    // Take a block_q8_0x4 structures at each pass of the loop and perform dot product operation
    for (; y < nr / 4; y ++) {
        const block_q8_0x4 * a_ptr = a_ptr_start + (y * nb);

        // Take group of two block_tx8 structures at each pass of the loop and perform dot product operation
        for (int64_t x = 0; x < anc / 8; x += 2) {

            const block_tx8 * b_ptr_0 = b_ptr_start + ((x)     * b_nb);
            const block_tx8 * b_ptr_1 = b_ptr_start + ((x + 1) * b_nb);

            // Master FP accumulators
            __m512 acc_rows[4];
            for (int i = 0; i < 4; i++) {
                acc_rows[i] = _mm512_setzero_ps();
            }

            for (int64_t b = 0; b < nb; b++) {
                // Load the sixteen blocks of quantized values interleaved with each other in chunks of eight - B0,B1 ....BE,BF
                const __m256i rhs_raw_mat_0123_0 = _mm256_loadu_si256((const __m256i *)(b_ptr_0[b].qs));
                const __m256i rhs_raw_mat_4567_0 = _mm256_loadu_si256((const __m256i *)(b_ptr_0[b].qs + 32));
                const __m256i rhs_raw_mat_0123_1 = _mm256_loadu_si256((const __m256i *)(b_ptr_0[b].qs + 64));
                const __m256i rhs_raw_mat_4567_1 = _mm256_loadu_si256((const __m256i *)(b_ptr_0[b].qs + 96));

                const __m256i rhs_raw_mat_89AB_0 = _mm256_loadu_si256((const __m256i *)(b_ptr_1[b].qs));
                const __m256i rhs_raw_mat_CDEF_0 = _mm256_loadu_si256((const __m256i *)(b_ptr_1[b].qs + 32));
                const __m256i rhs_raw_mat_89AB_1 = _mm256_loadu_si256((const __m256i *)(b_ptr_1[b].qs + 64));
                const __m256i rhs_raw_mat_CDEF_1 = _mm256_loadu_si256((const __m256i *)(b_ptr_1[b].qs + 96));

                // Save the values in the following vectors in the formats B0B1B4B5, B2B3B6B7 for further processing and storing of values
                const __m256i rhs_raw_mat_0145_0 = _mm256_blend_epi32(rhs_raw_mat_0123_0, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_0, requiredOrder), 240);
                const __m256i rhs_raw_mat_2367_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_0, requiredOrder), rhs_raw_mat_4567_0, 240);
                const __m256i rhs_raw_mat_0145_1 = _mm256_blend_epi32(rhs_raw_mat_0123_1, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_1, requiredOrder), 240);
                const __m256i rhs_raw_mat_2367_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_1, requiredOrder), rhs_raw_mat_4567_1, 240);

                const __m256i rhs_raw_mat_89CD_0 = _mm256_blend_epi32(rhs_raw_mat_89AB_0, _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_0, requiredOrder), 240);
                const __m256i rhs_raw_mat_ABEF_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_0, requiredOrder), rhs_raw_mat_CDEF_0, 240);
                const __m256i rhs_raw_mat_89CD_1 = _mm256_blend_epi32(rhs_raw_mat_89AB_1, _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_1, requiredOrder), 240);
                const __m256i rhs_raw_mat_ABEF_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_1, requiredOrder), rhs_raw_mat_CDEF_1, 240);

                const __m512i rhs_raw_mat_014589CD_0 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_0145_0), rhs_raw_mat_89CD_0, 1);
                const __m512i rhs_raw_mat_2367ABEF_0 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_2367_0), rhs_raw_mat_ABEF_0, 1);
                const __m512i rhs_raw_mat_014589CD_1 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_0145_1), rhs_raw_mat_89CD_1, 1);
                const __m512i rhs_raw_mat_2367ABEF_1 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_2367_1), rhs_raw_mat_ABEF_1, 1);

                // 4-bit -> 8-bit - Sign is maintained
                const __m512i rhs_mat_014589CD_0 = _mm512_shuffle_epi8(signextendlutexpanded, _mm512_and_si512(rhs_raw_mat_014589CD_0, m4bexpanded)); //B0(0-7) B1(0-7) B4(0-7) B5(0-7) B8(0-7) B9(0-7) BC(0-7) BD(0-7)
                const __m512i rhs_mat_2367ABEF_0 = _mm512_shuffle_epi8(signextendlutexpanded, _mm512_and_si512(rhs_raw_mat_2367ABEF_0, m4bexpanded)); //B2(0-7) B3(0-7) B6(0-7) B7(0-7) BA(0-7) BB(0-7) BE(0-7) BF(0-7)

                const __m512i rhs_mat_014589CD_1 = _mm512_shuffle_epi8(signextendlutexpanded, _mm512_and_si512(rhs_raw_mat_014589CD_1, m4bexpanded)); //B0(8-15) B1(8-15) B4(8-15) B5(8-15) B8(8-15) B9(8-15) BC(8-15) BD(8-15)
                const __m512i rhs_mat_2367ABEF_1 = _mm512_shuffle_epi8(signextendlutexpanded, _mm512_and_si512(rhs_raw_mat_2367ABEF_1, m4bexpanded)); //B2(8-15) B3(8-15) B6(8-15) B7(8-15) BA(8-15) BB(8-15) BE(8-15) BF(8-15)

                const __m512i rhs_mat_014589CD_2 = _mm512_shuffle_epi8(signextendlutexpanded, _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_0, 4), m4bexpanded)); //B0(16-23) B1(16-23) B4(16-23) B5(16-23) B8(16-23) B9(16-23) BC(16-23) BD(16-23)
                const __m512i rhs_mat_2367ABEF_2 = _mm512_shuffle_epi8(signextendlutexpanded, _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_0, 4), m4bexpanded)); //B2(16-23) B3(16-23) B6(16-23) B7(16-23) BA(16-23) BB(16-23) BE(16-23) BF(16-23)

                const __m512i rhs_mat_014589CD_3 = _mm512_shuffle_epi8(signextendlutexpanded, _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_1, 4), m4bexpanded)); //B0(24-31) B1(24-31) B4(24-31) B5(24-31) B8(24-31) B9(24-31) BC(24-31) BD(24-31)
                const __m512i rhs_mat_2367ABEF_3 = _mm512_shuffle_epi8(signextendlutexpanded, _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_1, 4), m4bexpanded)); //B2(24-31) B3(24-31) B6(24-31) B7(24-31) BA(24-31) BB(24-31) BE(24-31) BF(24-31)

                // Shuffle pattern one - right side input
                const __m512i rhs_mat_014589CD_0_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_0, (_MM_PERM_ENUM)136); //B0(0-3) B1(0-3) B0(0-3) B1(0-3) B4(0-3) B5(0-3) B4(0-3) B5(0-3) B8(0-3) B9(0-3) B8(0-3) B9(0-3) BC(0-3) BD(0-3) BC(0-3) BD(0-3)
                const __m512i rhs_mat_2367ABEF_0_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_0, (_MM_PERM_ENUM)136); //B2(0-3) B3(0-3) B2(0-3) B3(0-3) B6(0-3) B7(0-3) B6(0-3) B7(0-3) BA(0-3) BB(0-3) BA(0-3) BB(0-3) BE(0-3) BF(0-3) BE(0-3) BF(0-3)

                const __m512i rhs_mat_014589CD_1_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_1, (_MM_PERM_ENUM)136); //B0(8-11) B1(8-11) B0(8-11) B1(8-11) B4(8-11) B5(8-11) B4(8-11) B5(8-11) B8(8-11) B9(8-11) B8(8-11) B9(8-11) BC(8-11) BD(8-11) BC(8-11) BD(8-11)
                const __m512i rhs_mat_2367ABEF_1_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_1, (_MM_PERM_ENUM)136); //B2(8-11) B3(8-11) B2(8-11) B3(8-11) B6(8-11) B7(8-11) B6(8-11) B7(8-11) BA(8-11) BB(8-11) BA(8-11) BB(8-11) BE(8-11) BF(8-11) BE(8-11) BF(8-11)

                const __m512i rhs_mat_014589CD_2_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_2, (_MM_PERM_ENUM)136); //B0(16-19) B1(16-19) B0(16-19) B1(16-19) B4(16-19) B5(16-19) B4(16-19) B5(16-19) B8(16-19) B9(16-19) B8(16-19) B9(16-19) BC(16-19) BD(16-19) BC(16-19) BD(16-19)
                const __m512i rhs_mat_2367ABEF_2_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_2, (_MM_PERM_ENUM)136); //B2(16-19) B3(16-19) B2(16-19) B3(16-19) B6(16-19) B7(16-19) B6(16-19) B7(16-19) BA(16-19) BB(16-19) BA(16-19) BB(16-19) BE(16-19) BF(16-19) BE(16-19) BF(16-19)

                const __m512i rhs_mat_014589CD_3_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_3, (_MM_PERM_ENUM)136); //B0(24-27) B1(24-27) B0(24-27) B1(24-27) B4(24-27) B5(24-27) B4(24-27) B5(24-27) B8(24-27) B9(24-27) B8(24-27) B9(24-27) BC(24-27) BD(24-27) BC(24-27) BD(24-27)
                const __m512i rhs_mat_2367ABEF_3_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_3, (_MM_PERM_ENUM)136); //B2(24-27) B3(24-27) B2(24-27) B3(24-27) B6(24-27) B7(24-27) B6(24-27) B7(24-27) BA(24-27) BB(24-27) BA(24-27) BB(24-27) BE(24-27) BF(24-27) BE(24-27) BF(24-27)

                // Shuffle pattern two - right side input

                const __m512i rhs_mat_014589CD_0_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_0, (_MM_PERM_ENUM)221); //B0(4-7) B1(4-7) B0(4-7) B1(4-7) B4(4-7) B5(4-7) B4(4-7) B5(4-7) B8(4-7) B9(4-7) B8(4-7) B9(4-7) BC(4-7) BD(4-7) BC(4-7) BD(4-7)
                const __m512i rhs_mat_2367ABEF_0_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_0, (_MM_PERM_ENUM)221); //B2(4-7) B3(4-7) B2(4-7) B3(4-7) B6(4-7) B7(4-7) B6(4-7) B7(4-7) BA(4-7) BB(4-7) BA(4-7) BB(4-7) BE(4-7) BF(4-7) BE(4-7) BF(4-7)

                const __m512i rhs_mat_014589CD_1_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_1, (_MM_PERM_ENUM)221); //B0(12-15) B1(12-15) B0(12-15) B1(12-15) B4(12-15) B5(12-15) B4(12-15) B5(12-15) B8(12-15) B9(12-15) B8(12-15) B9(12-15) BC(12-15) BD(12-15) BC(12-15) BD(12-15)
                const __m512i rhs_mat_2367ABEF_1_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_1, (_MM_PERM_ENUM)221); //B2(12-15) B3(12-15) B2(12-15) B3(12-15) B6(12-15) B7(12-15) B6(12-15) B7(12-15) BA(12-15) BB(12-15) BA(12-15) BB(12-15) BE(12-15) BF(12-15) BE(12-15) BF(12-15)

                const __m512i rhs_mat_014589CD_2_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_2, (_MM_PERM_ENUM)221); //B0(20-23) B1(20-23) B0(20-23) B1(20-23) B4(20-23) B5(20-23) B4(20-23) B5(20-23) B8(20-23) B9(20-23) B8(20-23) B9(20-23) BC(20-23) BD(20-23) BC(20-23) BD(20-23)
                const __m512i rhs_mat_2367ABEF_2_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_2, (_MM_PERM_ENUM)221); //B2(20-23) B3(20-23) B2(20-23) B3(20-23) B6(20-23) B7(20-23) B6(20-23) B7(20-23) BA(20-23) BB(20-23) BA(20-23) BB(20-23) BE(20-23) BF(20-23) BE(20-23) BF(20-23)

                const __m512i rhs_mat_014589CD_3_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_3, (_MM_PERM_ENUM)221); //B0(28-31) B1(28-31) B0(28-31) B1(28-31) B4(28-31) B5(28-31) B4(28-31) B5(28-31) B8(28-31) B9(28-31) B8(28-31) B9(28-31) BC(28-31) BD(28-31) BC(28-31) BD(28-31)
                const __m512i rhs_mat_2367ABEF_3_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_3, (_MM_PERM_ENUM)221); //B2(28-31) B3(28-31) B2(28-31) B3(28-31) B6(28-31) B7(28-31) B6(28-31) B7(28-31) BA(28-31) BB(28-31) BA(28-31) BB(28-31) BE(28-31) BF(28-31) BE(28-31) BF(28-31)


                // Scale values - Load the weight scale values of two block_tx8
                __m512 col_scale_f32;
                if constexpr (
                        std::is_same_v<block_tx8, block_q4_0x8> ||
                        std::is_same_v<block_tx8, block_iq4_nlx8>) {
                    col_scale_f32 = GGML_F32Cx8x2_LOAD(b_ptr_0[b].d, b_ptr_1[b].d);
                } else if constexpr (std::is_same_v<block_tx8, block_mxfp4x8>) {
                    //TODO: simd-ify
                    col_scale_f32 = _mm512_set_ps(
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_1[b].e[7]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_1[b].e[6]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_1[b].e[5]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_1[b].e[4]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_1[b].e[3]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_1[b].e[2]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_1[b].e[1]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_1[b].e[0]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_0[b].e[7]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_0[b].e[6]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_0[b].e[5]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_0[b].e[4]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_0[b].e[3]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_0[b].e[2]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_0[b].e[1]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr_0[b].e[0]));
                }

                // Load the four blocks of quantized values interleaved with each other in chunks of eight - A0,A1,A2,A3
                // Loaded as set of 128 bit vectors and repeated and stored into a 256 bit vector before again repeating into 512 bit vector
                __m256i lhs_mat_ymm_0123_0 = _mm256_loadu_si256((const __m256i *)((a_ptr[b].qs)));
                __m256i lhs_mat_ymm_01_0 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_0, lhs_mat_ymm_0123_0, 0);
                __m256i lhs_mat_ymm_23_0 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_0, lhs_mat_ymm_0123_0, 17);
                __m256i lhs_mat_ymm_0123_1 = _mm256_loadu_si256((const __m256i *)((a_ptr[b].qs + 32)));
                __m256i lhs_mat_ymm_01_1 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_1, lhs_mat_ymm_0123_1, 0);
                __m256i lhs_mat_ymm_23_1 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_1, lhs_mat_ymm_0123_1, 17);
                __m256i lhs_mat_ymm_0123_2 = _mm256_loadu_si256((const __m256i *)((a_ptr[b].qs + 64)));
                __m256i lhs_mat_ymm_01_2 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_2, lhs_mat_ymm_0123_2, 0);
                __m256i lhs_mat_ymm_23_2 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_2, lhs_mat_ymm_0123_2, 17);
                __m256i lhs_mat_ymm_0123_3 = _mm256_loadu_si256((const __m256i *)((a_ptr[b].qs + 96)));
                __m256i lhs_mat_ymm_01_3 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_3, lhs_mat_ymm_0123_3, 0);
                __m256i lhs_mat_ymm_23_3 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_3, lhs_mat_ymm_0123_3, 17);

                __m512i lhs_mat_01_0 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_0), lhs_mat_ymm_01_0, 1);
                __m512i lhs_mat_23_0 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_0), lhs_mat_ymm_23_0, 1);
                __m512i lhs_mat_01_1 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_1), lhs_mat_ymm_01_1, 1);
                __m512i lhs_mat_23_1 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_1), lhs_mat_ymm_23_1, 1);
                __m512i lhs_mat_01_2 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_2), lhs_mat_ymm_01_2, 1);
                __m512i lhs_mat_23_2 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_2), lhs_mat_ymm_23_2, 1);
                __m512i lhs_mat_01_3 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_3), lhs_mat_ymm_01_3, 1);
                __m512i lhs_mat_23_3 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_3), lhs_mat_ymm_23_3, 1);

                // Shuffle pattern one - left side input

                const __m512i lhs_mat_01_0_sp1 = _mm512_shuffle_epi32(lhs_mat_01_0, (_MM_PERM_ENUM)160);  //A0(0-3) A0(0-3) A1(0-3) A1(0-3) A0(0-3) A0(0-3) A1(0-3) A1(0-3) A0(0-3) A0(0-3) A1(0-3) A1(0-3) A0(0-3) A0(0-3) A1(0-3) A1(0-3)
                const __m512i lhs_mat_23_0_sp1 = _mm512_shuffle_epi32(lhs_mat_23_0, (_MM_PERM_ENUM)160);  //A2(0-3) A2(0-3) A3(0-3) A3(0-3) A2(0-3) A2(0-3) A3(0-3) A3(0-3) A2(0-3) A2(0-3) A3(0-3) A3(0-3) A2(0-3) A2(0-3) A3(0-3) A3(0-3)

                const __m512i lhs_mat_01_1_sp1 = _mm512_shuffle_epi32(lhs_mat_01_1, (_MM_PERM_ENUM)160);  //A0(8-11) A0(8-11) A1(8-11) A1(8-11) A0(8-11) A0(8-11) A1(8-11) A1(8-11) A0(8-11) A0(8-11) A1(8-11) A1(8-11) A0(8-11) A0(8-11) A1(8-11) A1(8-11)
                const __m512i lhs_mat_23_1_sp1 = _mm512_shuffle_epi32(lhs_mat_23_1, (_MM_PERM_ENUM)160);  //A2(8-11) A2(8-11) A3(8-11) A3(8-11) A2(8-11) A2(8-11) A3(8-11) A3(8-11) A2(8-11) A2(8-11) A3(8-11) A3(8-11) A2(8-11) A2(8-11) A3(8-11) A3(8-11)

                const __m512i lhs_mat_01_2_sp1 = _mm512_shuffle_epi32(lhs_mat_01_2, (_MM_PERM_ENUM)160);  //A0(16-19) A0(16-19) A1(16-19) A1(16-19) A0(16-19) A0(16-19) A1(16-19) A1(16-19) A0(16-19) A0(16-19) A1(16-19) A1(16-19) A0(16-19) A0(16-19) A1(16-19) A1(16-19)
                const __m512i lhs_mat_23_2_sp1 = _mm512_shuffle_epi32(lhs_mat_23_2, (_MM_PERM_ENUM)160);  //A2(16-19) A2(16-19) A3(16-19) A3(16-19) A2(16-19) A2(16-19) A3(16-19) A3(16-19) A2(16-19) A2(16-19) A3(16-19) A3(16-19) A2(16-19) A2(16-19) A3(16-19) A3(16-19)

                const __m512i lhs_mat_01_3_sp1 = _mm512_shuffle_epi32(lhs_mat_01_3, (_MM_PERM_ENUM)160);  //A0(24-27) A0(24-27) A1(24-27) A1(24-27) A0(24-27) A0(24-27) A1(24-27) A1(24-27) A0(24-27) A0(24-27) A1(24-27) A1(24-27) A0(24-27) A0(24-27) A1(24-27) A1(24-27)
                const __m512i lhs_mat_23_3_sp1 = _mm512_shuffle_epi32(lhs_mat_23_3, (_MM_PERM_ENUM)160);  //A2(24-27) A2(24-27) A3(24-27) A3(24-27) A2(24-27) A2(24-27) A3(24-27) A3(24-27) A2(24-27) A2(24-27) A3(24-27) A3(24-27) A2(24-27) A2(24-27) A3(24-27) A3(24-27)

                // Shuffle pattern two - left side input

                const __m512i lhs_mat_01_0_sp2 = _mm512_shuffle_epi32(lhs_mat_01_0, (_MM_PERM_ENUM)245);  //A0(4-7) A0(4-7) A1(4-7) A1(4-7) A0(4-7) A0(4-7) A1(4-7) A1(4-7) A0(4-7) A0(4-7) A1(4-7) A1(4-7) A0(4-7) A0(4-7) A1(4-7) A1(4-7)
                const __m512i lhs_mat_23_0_sp2 = _mm512_shuffle_epi32(lhs_mat_23_0, (_MM_PERM_ENUM)245);  //A2(4-7) A2(4-7) A3(4-7) A3(4-7) A2(4-7) A2(4-7) A3(4-7) A3(4-7) A2(4-7) A2(4-7) A3(4-7) A3(4-7) A2(4-7) A2(4-7) A3(4-7) A3(4-7)

                const __m512i lhs_mat_01_1_sp2 = _mm512_shuffle_epi32(lhs_mat_01_1, (_MM_PERM_ENUM)245);  //A0(12-15) A0(12-15) A1(12-15) A1(12-15) A0(12-15) A0(12-15) A1(12-15) A1(12-15) A0(12-15) A0(12-15) A1(12-15) A1(12-15) A0(12-15) A0(12-15) A1(12-15) A1(12-15)
                const __m512i lhs_mat_23_1_sp2 = _mm512_shuffle_epi32(lhs_mat_23_1, (_MM_PERM_ENUM)245);  //A2(12-15) A2(12-15) A3(12-15) A3(12-15) A2(12-15) A2(12-15) A3(12-15) A3(12-15) A2(12-15) A2(12-15) A3(12-15) A3(12-15) A2(12-15) A2(12-15) A3(12-15) A3(12-15)

                const __m512i lhs_mat_01_2_sp2 = _mm512_shuffle_epi32(lhs_mat_01_2, (_MM_PERM_ENUM)245);  //A0(20-23) A0(20-23) A1(20-23) A1(20-23) A0(20-23) A0(20-23) A1(20-23) A1(20-23) A0(20-23) A0(20-23) A1(20-23) A1(20-23) A0(20-23) A0(20-23) A1(20-23) A1(20-23)
                const __m512i lhs_mat_23_2_sp2 = _mm512_shuffle_epi32(lhs_mat_23_2, (_MM_PERM_ENUM)245);  //A2(20-23) A2(20-23) A3(20-23) A3(20-23) A2(20-23) A2(20-23) A3(20-23) A3(20-23) A2(20-23) A2(20-23) A3(20-23) A3(20-23) A2(20-23) A2(20-23) A3(20-23) A3(20-23)

                const __m512i lhs_mat_01_3_sp2 = _mm512_shuffle_epi32(lhs_mat_01_3, (_MM_PERM_ENUM)245);  //A0(28-31) A0(28-31) A1(28-31) A1(28-31) A0(28-31) A0(28-31) A1(28-31) A1(28-31) A0(28-31) A0(28-31) A1(28-31) A1(28-31) A0(28-31) A0(28-31) A1(28-31) A1(28-31)
                const __m512i lhs_mat_23_3_sp2 = _mm512_shuffle_epi32(lhs_mat_23_3, (_MM_PERM_ENUM)245);  //A2(28-31) A2(28-31) A3(28-31) A3(28-31) A2(28-31) A2(28-31) A3(28-31) A3(28-31) A2(28-31) A2(28-31) A3(28-31) A3(28-31) A2(28-31) A2(28-31) A3(28-31) A3(28-31)

                // The values arranged in shuffle patterns are operated with dot product operation within 32 bit lane i.e corresponding bytes and multiplied and added into 32 bit integers within 32 bit lane
                // Resembles MMLAs into 2x2 matrices in ARM Version
                const __m512i zero = _mm512_setzero_epi32();
                __m512i iacc_mat_00_sp1 = mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_01_3_sp1, rhs_mat_014589CD_3_sp1), lhs_mat_01_2_sp1, rhs_mat_014589CD_2_sp1), lhs_mat_01_1_sp1, rhs_mat_014589CD_1_sp1), lhs_mat_01_0_sp1, rhs_mat_014589CD_0_sp1);
                __m512i iacc_mat_01_sp1 = mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_01_3_sp1, rhs_mat_2367ABEF_3_sp1), lhs_mat_01_2_sp1, rhs_mat_2367ABEF_2_sp1), lhs_mat_01_1_sp1, rhs_mat_2367ABEF_1_sp1), lhs_mat_01_0_sp1, rhs_mat_2367ABEF_0_sp1);
                __m512i iacc_mat_10_sp1 = mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_23_3_sp1, rhs_mat_014589CD_3_sp1), lhs_mat_23_2_sp1, rhs_mat_014589CD_2_sp1), lhs_mat_23_1_sp1, rhs_mat_014589CD_1_sp1), lhs_mat_23_0_sp1, rhs_mat_014589CD_0_sp1);
                __m512i iacc_mat_11_sp1 = mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_23_3_sp1, rhs_mat_2367ABEF_3_sp1), lhs_mat_23_2_sp1, rhs_mat_2367ABEF_2_sp1), lhs_mat_23_1_sp1, rhs_mat_2367ABEF_1_sp1), lhs_mat_23_0_sp1, rhs_mat_2367ABEF_0_sp1);
                __m512i iacc_mat_00_sp2 = mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_01_3_sp2, rhs_mat_014589CD_3_sp2), lhs_mat_01_2_sp2, rhs_mat_014589CD_2_sp2), lhs_mat_01_1_sp2, rhs_mat_014589CD_1_sp2), lhs_mat_01_0_sp2, rhs_mat_014589CD_0_sp2);
                __m512i iacc_mat_01_sp2 = mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_01_3_sp2, rhs_mat_2367ABEF_3_sp2), lhs_mat_01_2_sp2, rhs_mat_2367ABEF_2_sp2), lhs_mat_01_1_sp2, rhs_mat_2367ABEF_1_sp2), lhs_mat_01_0_sp2, rhs_mat_2367ABEF_0_sp2);
                __m512i iacc_mat_10_sp2 = mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_23_3_sp2, rhs_mat_014589CD_3_sp2), lhs_mat_23_2_sp2, rhs_mat_014589CD_2_sp2), lhs_mat_23_1_sp2, rhs_mat_014589CD_1_sp2), lhs_mat_23_0_sp2, rhs_mat_014589CD_0_sp2);
                __m512i iacc_mat_11_sp2 = mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_23_3_sp2, rhs_mat_2367ABEF_3_sp2), lhs_mat_23_2_sp2, rhs_mat_2367ABEF_2_sp2), lhs_mat_23_1_sp2, rhs_mat_2367ABEF_1_sp2), lhs_mat_23_0_sp2, rhs_mat_2367ABEF_0_sp2);

                // Output of both shuffle patterns are added in order to sum dot product outputs of all 32 values in block
                __m512i iacc_mat_00 = _mm512_add_epi32(iacc_mat_00_sp1, iacc_mat_00_sp2);
                __m512i iacc_mat_01 = _mm512_add_epi32(iacc_mat_01_sp1, iacc_mat_01_sp2);
                __m512i iacc_mat_10 = _mm512_add_epi32(iacc_mat_10_sp1, iacc_mat_10_sp2);
                __m512i iacc_mat_11 = _mm512_add_epi32(iacc_mat_11_sp1, iacc_mat_11_sp2);


                // Straighten out to make 4 row vectors
                __m512i iacc_row_0 = _mm512_mask_blend_epi32(0xCCCC, iacc_mat_00, _mm512_shuffle_epi32(iacc_mat_01, (_MM_PERM_ENUM)78));
                __m512i iacc_row_1 = _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32(iacc_mat_00, (_MM_PERM_ENUM)78), iacc_mat_01);
                __m512i iacc_row_2 = _mm512_mask_blend_epi32(0xCCCC, iacc_mat_10, _mm512_shuffle_epi32(iacc_mat_11, (_MM_PERM_ENUM)78));
                __m512i iacc_row_3 = _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32(iacc_mat_10, (_MM_PERM_ENUM)78), iacc_mat_11);

                // Load the scale(d) values for all the 4 Q8_0 blocks and repeat it across lanes
                const __m128i row_scale_f16 = _mm_shuffle_epi32(_mm_maskload_epi32((int const*)(a_ptr[b].d), loadMask), 68);
                const __m512 row_scale_f32 = GGML_F32Cx16_REPEAT_LOAD(row_scale_f16);

                // Multiply with appropriate scales and accumulate
                acc_rows[0] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_0), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 0)),   acc_rows[0]);
                acc_rows[1] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_1), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 85)),  acc_rows[1]);
                acc_rows[2] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_2), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_rows[2]);
                acc_rows[3] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_3), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_rows[3]);
            }

            // Store the accumulated values
            for (int i = 0; i < 4; i++) {
                _mm512_storeu_ps((float *)(s + ((y * 4 + i) * bs + x * 8)), acc_rows[i]);
            }
        }
    }
    if (anc != nc) {
        xstart = anc/8;
        y = 0;
    }
#endif // __AVX512BW__ && __AVX512DQ__

    // Take group of four block_q8_0x4 structures at each pass of the loop and perform dot product operation

    for (; y < anr / 4; y += 4) {
        const block_q8_0x4 * a_ptrs[4];

        a_ptrs[0] = a_ptr_start + (y * nb);
        for (int i = 0; i < 3; ++i) {
            a_ptrs[i + 1] = a_ptrs[i] + nb;
        }

        // Take group of eight block_tx8 structures at each pass of the loop and perform dot product operation
        for (int64_t x = xstart; x < nc / 8; x++) {

            const block_tx8 * b_ptr = b_ptr_start + (x * b_nb);

            // Master FP accumulators
            __m256 acc_rows[16];
            for (int i = 0; i < 16; i++) {
                acc_rows[i] = _mm256_setzero_ps();
            }

            for (int64_t b = 0; b < nb; b++) {
                // Load the eight blocks of quantized values interleaved with each other in chunks of eight - B0,B1 ....B6,B7
                const __m256i rhs_raw_mat_0123_0 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs));
                const __m256i rhs_raw_mat_4567_0 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 32));
                const __m256i rhs_raw_mat_0123_1 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 64));
                const __m256i rhs_raw_mat_4567_1 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 96));

                // Save the values in the following vectors in the formats B0B1B4B5, B2B3B6B7 for further processing and storing of values
                const __m256i rhs_raw_mat_0145_0 = _mm256_blend_epi32(rhs_raw_mat_0123_0, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_0, requiredOrder), 240);
                const __m256i rhs_raw_mat_2367_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_0, requiredOrder), rhs_raw_mat_4567_0, 240);
                const __m256i rhs_raw_mat_0145_1 = _mm256_blend_epi32(rhs_raw_mat_0123_1, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_1, requiredOrder), 240);
                const __m256i rhs_raw_mat_2367_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_1, requiredOrder), rhs_raw_mat_4567_1, 240);

                // 4-bit -> 8-bit - Sign is maintained
                const __m256i rhs_mat_0145_0 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_mat_0145_0, m4b)); //B0(0-7) B1(0-7) B4(0-7) B5(0-7)
                const __m256i rhs_mat_2367_0 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_mat_2367_0, m4b)); //B2(0-7) B3(0-7) B6(0-7) B7(0-7)

                const __m256i rhs_mat_0145_1 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_mat_0145_1, m4b)); //B0(8-15) B1(8-15) B4(8-15) B5(8-15)
                const __m256i rhs_mat_2367_1 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_mat_2367_1, m4b)); //B2(8-15) B3(8-15) B6(8-15) B7(8-15)

                const __m256i rhs_mat_0145_2 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_0, 4), m4b)); //B0(16-23) B1(16-23) B4(16-23) B5(16-23)
                const __m256i rhs_mat_2367_2 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_0, 4), m4b)); //B2(16-23) B3(16-23) B6(16-23) B7(16-23)

                const __m256i rhs_mat_0145_3 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_1, 4), m4b)); //B0(24-31) B1(24-31) B4(24-31) B5(24-31)
                const __m256i rhs_mat_2367_3 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_1, 4), m4b)); //B2(24-31) B3(24-31) B6(24-31) B7(24-31)

                // Shuffle pattern one - right side input
                const __m256i rhs_mat_0145_0_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_0, 136);  //B0(0-3) B1(0-3) B0(0-3) B1(0-3) B4(0-3) B5(0-3) B4(0-3) B5(0-3)
                const __m256i rhs_mat_2367_0_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_0, 136);  //B2(0-3) B3(0-3) B2(0-3) B3(0-3) B6(0-3) B7(0-3) B6(0-3) B7(0-3)

                const __m256i rhs_mat_0145_1_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_1, 136);  //B0(8-11) B1(8-11) B0(8-11) B1(8-11) B4(8-11) B5(8-11) B4(8-11) B5(8-11)
                const __m256i rhs_mat_2367_1_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_1, 136);  //B2(8-11) B3(8-11) B2(8-11) B3(8-11) B6(8-11) B7(8-11) B6(8-11) B7(8-11)

                const __m256i rhs_mat_0145_2_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_2, 136);  //B0(16-19) B1(16-19) B0(16-19) B1(16-19) B4(16-19) B5(16-19) B4(16-19) B5(16-19)
                const __m256i rhs_mat_2367_2_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_2, 136);  //B2(16-19) B3(16-19) B2(16-19) B3(16-19) B6(16-19) B7(16-19) B6(16-19) B7(16-19)

                const __m256i rhs_mat_0145_3_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_3, 136);  //B0(24-27) B1(24-27) B0(24-27) B1(24-27) B4(24-27) B5(24-27) B4(24-27) B5(24-27)
                const __m256i rhs_mat_2367_3_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_3, 136);  //B2(24-27) B3(24-27) B2(24-27) B3(24-27) B6(24-27) B7(24-27) B6(24-27) B7(24-27)

                // Shuffle pattern two - right side input

                const __m256i rhs_mat_0145_0_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_0, 221);  //B0(4-7) B1(4-7) B0(4-7) B1(4-7) B4(4-7) B5(4-7) B4(4-7) B5(4-7)
                const __m256i rhs_mat_2367_0_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_0, 221);  //B2(4-7) B3(4-7) B2(4-7) B3(4-7) B6(4-7) B7(4-7) B6(4-7) B7(4-7)

                const __m256i rhs_mat_0145_1_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_1, 221);  //B0(12-15) B1(12-15) B0(12-15) B1(12-15) B4(12-15) B5(12-15) B4(12-15) B5(12-15)
                const __m256i rhs_mat_2367_1_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_1, 221);  //B2(12-15) B3(12-15) B2(12-15) B3(12-15) B6(12-15) B7(12-15) B6(12-15) B7(12-15)

                const __m256i rhs_mat_0145_2_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_2, 221);  //B0(20-23) B1(20-23) B0(20-23) B1(20-23) B4(20-23) B5(20-23) B4(20-23) B5(20-23)
                const __m256i rhs_mat_2367_2_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_2, 221);  //B2(20-23) B3(20-23) B2(20-23) B3(20-23) B6(20-23) B7(20-23) B6(20-23) B7(20-23)

                const __m256i rhs_mat_0145_3_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_3, 221);  //B0(28-31) B1(28-31) B0(28-31) B1(28-31) B4(28-31) B5(28-31) B4(28-31) B5(28-31)
                const __m256i rhs_mat_2367_3_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_3, 221);  //B2(28-31) B3(28-31) B2(28-31) B3(28-31) B6(28-31) B7(28-31) B6(28-31) B7(28-31)

                // Scale values - Load the wight scale values of block_tx8
                __m256 col_scale_f32;
                if constexpr (
                        std::is_same_v<block_tx8, block_q4_0x8> ||
                        std::is_same_v<block_tx8, block_iq4_nlx8>) {
                    col_scale_f32 = GGML_F32Cx8_LOAD(b_ptr[b].d);
                } else if constexpr (std::is_same_v<block_tx8, block_mxfp4x8>) {
                    col_scale_f32 = _mm256_set_ps(
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[b].e[7]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[b].e[6]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[b].e[5]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[b].e[4]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[b].e[3]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[b].e[2]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[b].e[1]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[b].e[0]));
                }

                // Process LHS in groups of four
                for (int rp = 0; rp < 4; rp++) {
                    // Load the four blocks of quantized values interleaved with each other in chunks of eight - A0,A1,A2,A3
                    // Loaded as set of 128 bit vectors and repeated into a 256 bit vector
                    __m256i lhs_mat_0123_0 = _mm256_loadu_si256((const __m256i *)((a_ptrs[rp][b].qs)));
                    __m256i lhs_mat_01_0 = _mm256_permute2f128_si256(lhs_mat_0123_0, lhs_mat_0123_0, 0);
                    __m256i lhs_mat_23_0 = _mm256_permute2f128_si256(lhs_mat_0123_0, lhs_mat_0123_0, 17);
                    __m256i lhs_mat_0123_1 = _mm256_loadu_si256((const __m256i *)((a_ptrs[rp][b].qs + 32)));
                    __m256i lhs_mat_01_1 = _mm256_permute2f128_si256(lhs_mat_0123_1, lhs_mat_0123_1, 0);
                    __m256i lhs_mat_23_1 = _mm256_permute2f128_si256(lhs_mat_0123_1, lhs_mat_0123_1, 17);
                    __m256i lhs_mat_0123_2 = _mm256_loadu_si256((const __m256i *)((a_ptrs[rp][b].qs + 64)));
                    __m256i lhs_mat_01_2 = _mm256_permute2f128_si256(lhs_mat_0123_2, lhs_mat_0123_2, 0);
                    __m256i lhs_mat_23_2 = _mm256_permute2f128_si256(lhs_mat_0123_2, lhs_mat_0123_2, 17);
                    __m256i lhs_mat_0123_3 = _mm256_loadu_si256((const __m256i *)((a_ptrs[rp][b].qs + 96)));
                    __m256i lhs_mat_01_3 = _mm256_permute2f128_si256(lhs_mat_0123_3, lhs_mat_0123_3, 0);
                    __m256i lhs_mat_23_3 = _mm256_permute2f128_si256(lhs_mat_0123_3, lhs_mat_0123_3, 17);

                    // Shuffle pattern one - left side input
                    const __m256i lhs_mat_01_0_sp1 = _mm256_shuffle_epi32(lhs_mat_01_0, 160);  //A0(0-3) A0(0-3) A1(0-3) A1(0-3) A0(0-3) A0(0-3) A1(0-3) A1(0-3)
                    const __m256i lhs_mat_23_0_sp1 = _mm256_shuffle_epi32(lhs_mat_23_0, 160);  //A2(0-3) A2(0-3) A3(0-3) A3(0-3) A2(0-3) A2(0-3) A3(0-3) A3(0-3)

                    const __m256i lhs_mat_01_1_sp1 = _mm256_shuffle_epi32(lhs_mat_01_1, 160);  //A0(8-11) A0(8-11) A1(8-11) A1(8-11) A0(8-11) A0(8-11) A1(8-11) A1(8-11)
                    const __m256i lhs_mat_23_1_sp1 = _mm256_shuffle_epi32(lhs_mat_23_1, 160);  //A2(8-11) A2(8-11) A3(8-11) A3(8-11) A2(8-11) A2(8-11) A3(8-11) A3(8-11)

                    const __m256i lhs_mat_01_2_sp1 = _mm256_shuffle_epi32(lhs_mat_01_2, 160);  //A0(16-19) A0(16-19) A1(16-19) A1(16-19) A0(16-19) A0(16-19) A1(16-19) A1(16-19)
                    const __m256i lhs_mat_23_2_sp1 = _mm256_shuffle_epi32(lhs_mat_23_2, 160);  //A2(16-19) A2(16-19) A3(16-19) A3(16-19) A2(16-19) A2(16-19) A3(16-19) A3(16-19)

                    const __m256i lhs_mat_01_3_sp1 = _mm256_shuffle_epi32(lhs_mat_01_3, 160);  //A0(24-27) A0(24-27) A1(24-27) A1(24-27) A0(24-27) A0(24-27) A1(24-27) A1(24-27)
                    const __m256i lhs_mat_23_3_sp1 = _mm256_shuffle_epi32(lhs_mat_23_3, 160);  //A2(24-27) A2(24-27) A3(24-27) A3(24-27) A2(24-27) A2(24-27) A3(24-27) A3(24-27)

                    // Shuffle pattern two - left side input
                    const __m256i lhs_mat_01_0_sp2 = _mm256_shuffle_epi32(lhs_mat_01_0, 245);  //A0(4-7) A0(4-7) A1(4-7) A1(4-7) A0(4-7) A0(4-7) A1(4-7) A1(4-7)
                    const __m256i lhs_mat_23_0_sp2 = _mm256_shuffle_epi32(lhs_mat_23_0, 245);  //A2(4-7) A2(4-7) A3(4-7) A3(4-7) A2(4-7) A2(4-7) A3(4-7) A3(4-7)

                    const __m256i lhs_mat_01_1_sp2 = _mm256_shuffle_epi32(lhs_mat_01_1, 245);  //A0(12-15) A0(12-15) A1(12-15) A1(12-15) A0(12-15) A0(12-15) A1(12-15) A1(12-15)
                    const __m256i lhs_mat_23_1_sp2 = _mm256_shuffle_epi32(lhs_mat_23_1, 245);  //A2(12-15) A2(12-15) A3(12-15) A3(12-15) A2(12-15) A2(12-15) A3(12-15) A3(12-15)

                    const __m256i lhs_mat_01_2_sp2 = _mm256_shuffle_epi32(lhs_mat_01_2, 245);  //A0(20-23) A0(20-23) A1(20-23) A1(20-23) A0(20-23) A0(20-23) A1(20-23) A1(20-23)
                    const __m256i lhs_mat_23_2_sp2 = _mm256_shuffle_epi32(lhs_mat_23_2, 245);  //A2(20-23) A2(20-23) A3(20-23) A3(20-23) A2(20-23) A2(20-23) A3(20-23) A3(20-23)

                    const __m256i lhs_mat_01_3_sp2 = _mm256_shuffle_epi32(lhs_mat_01_3, 245);  //A0(28-31) A0(28-31) A1(28-31) A1(28-31) A0(28-31) A0(28-31) A1(28-31) A1(28-31)
                    const __m256i lhs_mat_23_3_sp2 = _mm256_shuffle_epi32(lhs_mat_23_3, 245);  //A2(28-31) A2(28-31) A3(28-31) A3(28-31) A2(28-31) A2(28-31) A3(28-31) A3(28-31)

                    // The values arranged in shuffle patterns are operated with dot product operation within 32 bit lane i.e corresponding bytes and multiplied and added into 32 bit integers within 32 bit lane
                    // Resembles MMLAs into 2x2 matrices in ARM Version
                    const __m256i zero = _mm256_setzero_si256();
                    __m256i iacc_mat_00_sp1 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_01_3_sp1, rhs_mat_0145_3_sp1), lhs_mat_01_2_sp1, rhs_mat_0145_2_sp1), lhs_mat_01_1_sp1, rhs_mat_0145_1_sp1), lhs_mat_01_0_sp1, rhs_mat_0145_0_sp1);
                    __m256i iacc_mat_01_sp1 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_01_3_sp1, rhs_mat_2367_3_sp1), lhs_mat_01_2_sp1, rhs_mat_2367_2_sp1), lhs_mat_01_1_sp1, rhs_mat_2367_1_sp1), lhs_mat_01_0_sp1, rhs_mat_2367_0_sp1);
                    __m256i iacc_mat_10_sp1 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_23_3_sp1, rhs_mat_0145_3_sp1), lhs_mat_23_2_sp1, rhs_mat_0145_2_sp1), lhs_mat_23_1_sp1, rhs_mat_0145_1_sp1), lhs_mat_23_0_sp1, rhs_mat_0145_0_sp1);
                    __m256i iacc_mat_11_sp1 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_23_3_sp1, rhs_mat_2367_3_sp1), lhs_mat_23_2_sp1, rhs_mat_2367_2_sp1), lhs_mat_23_1_sp1, rhs_mat_2367_1_sp1), lhs_mat_23_0_sp1, rhs_mat_2367_0_sp1);
                    __m256i iacc_mat_00_sp2 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_01_3_sp2, rhs_mat_0145_3_sp2), lhs_mat_01_2_sp2, rhs_mat_0145_2_sp2), lhs_mat_01_1_sp2, rhs_mat_0145_1_sp2), lhs_mat_01_0_sp2, rhs_mat_0145_0_sp2);
                    __m256i iacc_mat_01_sp2 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_01_3_sp2, rhs_mat_2367_3_sp2), lhs_mat_01_2_sp2, rhs_mat_2367_2_sp2), lhs_mat_01_1_sp2, rhs_mat_2367_1_sp2), lhs_mat_01_0_sp2, rhs_mat_2367_0_sp2);
                    __m256i iacc_mat_10_sp2 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_23_3_sp2, rhs_mat_0145_3_sp2), lhs_mat_23_2_sp2, rhs_mat_0145_2_sp2), lhs_mat_23_1_sp2, rhs_mat_0145_1_sp2), lhs_mat_23_0_sp2, rhs_mat_0145_0_sp2);
                    __m256i iacc_mat_11_sp2 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_23_3_sp2, rhs_mat_2367_3_sp2), lhs_mat_23_2_sp2, rhs_mat_2367_2_sp2), lhs_mat_23_1_sp2, rhs_mat_2367_1_sp2), lhs_mat_23_0_sp2, rhs_mat_2367_0_sp2);

                    // Output of both shuffle patterns are added in order to sum dot product outputs of all 32 values in block
                    __m256i iacc_mat_00 = _mm256_add_epi32(iacc_mat_00_sp1, iacc_mat_00_sp2);
                    __m256i iacc_mat_01 = _mm256_add_epi32(iacc_mat_01_sp1, iacc_mat_01_sp2);
                    __m256i iacc_mat_10 = _mm256_add_epi32(iacc_mat_10_sp1, iacc_mat_10_sp2);
                    __m256i iacc_mat_11 = _mm256_add_epi32(iacc_mat_11_sp1, iacc_mat_11_sp2);

                    // Straighten out to make 4 row vectors
                    __m256i iacc_row_0 = _mm256_blend_epi32(iacc_mat_00, _mm256_shuffle_epi32(iacc_mat_01, 78), 204);
                    __m256i iacc_row_1 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_00, 78), iacc_mat_01, 204);
                    __m256i iacc_row_2 = _mm256_blend_epi32(iacc_mat_10, _mm256_shuffle_epi32(iacc_mat_11, 78), 204);
                    __m256i iacc_row_3 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_10, 78), iacc_mat_11, 204);

                    // Load the scale(d) values for all the 4 Q8_0 blocks and repeat it across lanes
                    const __m256 row_scale_f32 = GGML_F32Cx8_REPEAT_LOAD(a_ptrs[rp][b].d, loadMask);

                    // Multiply with appropriate scales and accumulate
                    acc_rows[rp * 4] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_0), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 0)), acc_rows[rp * 4]);
                    acc_rows[rp * 4 + 1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_1), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 85)), acc_rows[rp * 4 + 1]);
                    acc_rows[rp * 4 + 2] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_2), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_rows[rp * 4 + 2]);
                    acc_rows[rp * 4 + 3] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_3), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32,  255)), acc_rows[rp * 4 + 3]);
                }
            }

            // Store the accumulated values
            for (int i = 0; i < 16; i++) {
                _mm256_storeu_ps((float *)(s + ((y * 4 + i) * bs + x * 8)), acc_rows[i]);
            }
        }
    }

    // Take a block_q8_0x4 structures at each pass of the loop and perform dot product operation
    for (; y < nr / 4; y ++) {
        const block_q8_0x4 * a_ptr = a_ptr_start + (y * nb);

        // Load the eight blocks of quantized values interleaved with each other in chunks of eight - B0,B1 ....B6,B7
        for (int64_t x = xstart; x < nc / 8; x++) {
            const block_tx8 * b_ptr = b_ptr_start + (x * b_nb);

            // Master FP accumulators
            __m256 acc_rows[4];
            for (int i = 0; i < 4; i++) {
                acc_rows[i] = _mm256_setzero_ps();
            }

            for (int64_t b = 0; b < nb; b++) {
                // Load the eight block_q8_0 quantized values interleaved with each other in chunks of eight - B0,B1 ....B6,B7
                const __m256i rhs_raw_mat_0123_0 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs));
                const __m256i rhs_raw_mat_4567_0 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 32));
                const __m256i rhs_raw_mat_0123_1 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 64));
                const __m256i rhs_raw_mat_4567_1 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 96));

                // Save the values in the following vectors in the formats B0B1B4B5, B2B3B6B7 for further processing and storing of values
                const __m256i rhs_raw_mat_0145_0 = _mm256_blend_epi32(rhs_raw_mat_0123_0, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_0, requiredOrder), 240);
                const __m256i rhs_raw_mat_2367_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_0, requiredOrder), rhs_raw_mat_4567_0, 240);
                const __m256i rhs_raw_mat_0145_1 = _mm256_blend_epi32(rhs_raw_mat_0123_1, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_1, requiredOrder), 240);
                const __m256i rhs_raw_mat_2367_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_1, requiredOrder), rhs_raw_mat_4567_1, 240);

                // 4-bit -> 8-bit - Sign is maintained
                const __m256i rhs_mat_0145_0 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_mat_0145_0, m4b));  //B0(0-7) B1(0-7) B4(0-7) B5(0-7)
                const __m256i rhs_mat_2367_0 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_mat_2367_0, m4b));  //B2(0-7) B3(0-7) B6(0-7) B7(0-7)

                const __m256i rhs_mat_0145_1 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_mat_0145_1, m4b));  //B0(8-15) B1(8-15) B4(8-15) B5(8-15)
                const __m256i rhs_mat_2367_1 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_mat_2367_1, m4b));  //B2(8-15) B3(8-15) B6(8-15) B7(8-15)

                const __m256i rhs_mat_0145_2 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_0, 4), m4b));  //B0(16-23) B1(16-23) B4(16-23) B5(16-23)
                const __m256i rhs_mat_2367_2 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_0, 4), m4b));  //B2(16-23) B3(16-23) B6(16-23) B7(16-23)

                const __m256i rhs_mat_0145_3 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_1, 4), m4b));  //B0(24-31) B1(24-31) B4(24-31) B5(24-31)
                const __m256i rhs_mat_2367_3 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_1, 4), m4b));  //B2(24-31) B3(24-31) B6(24-31) B7(24-31)

                // Shuffle pattern one - right side input
                const __m256i rhs_mat_0145_0_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_0, 136);  //B0(0-3) B1(0-3) B0(0-3) B1(0-3) B4(0-3) B5(0-3) B4(0-3) B5(0-3)
                const __m256i rhs_mat_2367_0_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_0, 136);  //B2(0-3) B3(0-3) B2(0-3) B3(0-3) B6(0-3) B7(0-3) B6(0-3) B7(0-3)

                const __m256i rhs_mat_0145_1_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_1, 136);  //B0(8-11) B1(8-11) B0(8-11) B1(8-11) B4(8-11) B5(8-11) B4(8-11) B5(8-11)
                const __m256i rhs_mat_2367_1_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_1, 136);  //B2(8-11) B3(8-11) B2(8-11) B3(8-11) B6(8-11) B7(8-11) B6(8-11) B7(8-11)

                const __m256i rhs_mat_0145_2_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_2, 136);  //B0(16-19) B1(16-19) B0(16-19) B1(16-19) B4(16-19) B5(16-19) B4(16-19) B5(16-19)
                const __m256i rhs_mat_2367_2_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_2, 136);  //B2(16-19) B3(16-19) B2(16-19) B3(16-19) B6(16-19) B7(16-19) B6(16-19) B7(16-19)

                const __m256i rhs_mat_0145_3_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_3, 136);  //B0(24-27) B1(24-27) B0(24-27) B1(24-27) B4(24-27) B5(24-27) B4(24-27) B5(24-27)
                const __m256i rhs_mat_2367_3_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_3, 136);  //B2(24-27) B3(24-27) B2(24-27) B3(24-27) B6(24-27) B7(24-27) B6(24-27) B7(24-27)

                // Shuffle pattern two - right side input

                const __m256i rhs_mat_0145_0_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_0, 221);  //B0(4-7) B1(4-7) B0(4-7) B1(4-7) B4(4-7) B5(4-7) B4(4-7) B5(4-7)
                const __m256i rhs_mat_2367_0_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_0, 221);  //B2(4-7) B3(4-7) B2(4-7) B3(4-7) B6(4-7) B7(4-7) B6(4-7) B7(4-7)

                const __m256i rhs_mat_0145_1_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_1, 221);  //B0(12-15) B1(12-15) B0(12-15) B1(12-15) B4(12-15) B5(12-15) B4(12-15) B5(12-15)
                const __m256i rhs_mat_2367_1_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_1, 221);  //B2(12-15) B3(12-15) B2(12-15) B3(12-15) B6(12-15) B7(12-15) B6(12-15) B7(12-15)

                const __m256i rhs_mat_0145_2_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_2, 221);  //B0(20-23) B1(20-23) B0(20-23) B1(20-23) B4(20-23) B5(20-23) B4(20-23) B5(20-23)
                const __m256i rhs_mat_2367_2_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_2, 221);  //B2(20-23) B3(20-23) B2(20-23) B3(20-23) B6(20-23) B7(20-23) B6(20-23) B7(20-23)

                const __m256i rhs_mat_0145_3_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_3, 221);  //B0(28-31) B1(28-31) B0(28-31) B1(28-31) B4(28-31) B5(28-31) B4(28-31) B5(28-31)
                const __m256i rhs_mat_2367_3_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_3, 221);  //B2(28-31) B3(28-31) B2(28-31) B3(28-31) B6(28-31) B7(28-31) B6(28-31) B7(28-31)

                // Scale values - Load the wight scale values of block_tx8
                __m256 col_scale_f32;
                if constexpr (
                        std::is_same_v<block_tx8, block_q4_0x8> ||
                        std::is_same_v<block_tx8, block_iq4_nlx8>) {
                    col_scale_f32 = GGML_F32Cx8_LOAD(b_ptr[b].d);
                } else if constexpr (std::is_same_v<block_tx8, block_mxfp4x8>) {
                    col_scale_f32 = _mm256_set_ps(
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[b].e[7]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[b].e[6]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[b].e[5]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[b].e[4]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[b].e[3]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[b].e[2]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[b].e[1]),
                        GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[b].e[0]));
                }

                // Load the four blocks of quantized values interleaved with each other in chunks of eight - A0,A1,A2,A3
                // Loaded as set of 128 bit vectors and repeated into a 256 bit vector
                __m256i lhs_mat_0123_0 = _mm256_loadu_si256((const __m256i *)((a_ptr[b].qs)));
                __m256i lhs_mat_01_0 = _mm256_permute2f128_si256(lhs_mat_0123_0, lhs_mat_0123_0, 0);
                __m256i lhs_mat_23_0 = _mm256_permute2f128_si256(lhs_mat_0123_0, lhs_mat_0123_0, 17);
                __m256i lhs_mat_0123_1 = _mm256_loadu_si256((const __m256i *)((a_ptr[b].qs + 32)));
                __m256i lhs_mat_01_1 = _mm256_permute2f128_si256(lhs_mat_0123_1, lhs_mat_0123_1, 0);
                __m256i lhs_mat_23_1 = _mm256_permute2f128_si256(lhs_mat_0123_1, lhs_mat_0123_1, 17);
                __m256i lhs_mat_0123_2 = _mm256_loadu_si256((const __m256i *)((a_ptr[b].qs + 64)));
                __m256i lhs_mat_01_2 = _mm256_permute2f128_si256(lhs_mat_0123_2, lhs_mat_0123_2, 0);
                __m256i lhs_mat_23_2 = _mm256_permute2f128_si256(lhs_mat_0123_2, lhs_mat_0123_2, 17);
                __m256i lhs_mat_0123_3 = _mm256_loadu_si256((const __m256i *)((a_ptr[b].qs + 96)));
                __m256i lhs_mat_01_3 = _mm256_permute2f128_si256(lhs_mat_0123_3, lhs_mat_0123_3, 0);
                __m256i lhs_mat_23_3 = _mm256_permute2f128_si256(lhs_mat_0123_3, lhs_mat_0123_3, 17);

                // Shuffle pattern one - left side input

                const __m256i lhs_mat_01_0_sp1 = _mm256_shuffle_epi32(lhs_mat_01_0, 160);  //A0(0-3) A0(0-3) A1(0-3) A1(0-3) A0(0-3) A0(0-3) A1(0-3) A1(0-3)
                const __m256i lhs_mat_23_0_sp1 = _mm256_shuffle_epi32(lhs_mat_23_0, 160);  //A2(0-3) A2(0-3) A3(0-3) A3(0-3) A2(0-3) A2(0-3) A3(0-3) A3(0-3)

                const __m256i lhs_mat_01_1_sp1 = _mm256_shuffle_epi32(lhs_mat_01_1, 160);  //A0(8-11) A0(8-11) A1(8-11) A1(8-11) A0(8-11) A0(8-11) A1(8-11) A1(8-11)
                const __m256i lhs_mat_23_1_sp1 = _mm256_shuffle_epi32(lhs_mat_23_1, 160);  //A2(8-11) A2(8-11) A3(8-11) A3(8-11) A2(8-11) A2(8-11) A3(8-11) A3(8-11)

                const __m256i lhs_mat_01_2_sp1 = _mm256_shuffle_epi32(lhs_mat_01_2, 160);  //A0(16-19) A0(16-19) A1(16-19) A1(16-19) A0(16-19) A0(16-19) A1(16-19) A1(16-19)
                const __m256i lhs_mat_23_2_sp1 = _mm256_shuffle_epi32(lhs_mat_23_2, 160);  //A2(16-19) A2(16-19) A3(16-19) A3(16-19) A2(16-19) A2(16-19) A3(16-19) A3(16-19)

                const __m256i lhs_mat_01_3_sp1 = _mm256_shuffle_epi32(lhs_mat_01_3, 160);  //A0(24-27) A0(24-27) A1(24-27) A1(24-27) A0(24-27) A0(24-27) A1(24-27) A1(24-27)
                const __m256i lhs_mat_23_3_sp1 = _mm256_shuffle_epi32(lhs_mat_23_3, 160);  //A2(24-27) A2(24-27) A3(24-27) A3(24-27) A2(24-27) A2(24-27) A3(24-27) A3(24-27)

                // Shuffle pattern two - left side input

                const __m256i lhs_mat_01_0_sp2 = _mm256_shuffle_epi32(lhs_mat_01_0, 245);  //A0(4-7) A0(4-7) A1(4-7) A1(4-7) A0(4-7) A0(4-7) A1(4-7) A1(4-7)
                const __m256i lhs_mat_23_0_sp2 = _mm256_shuffle_epi32(lhs_mat_23_0, 245);  //A2(4-7) A2(4-7) A3(4-7) A3(4-7) A2(4-7) A2(4-7) A3(4-7) A3(4-7)

                const __m256i lhs_mat_01_1_sp2 = _mm256_shuffle_epi32(lhs_mat_01_1, 245);  //A0(12-15) A0(12-15) A1(12-15) A1(12-15) A0(12-15) A0(12-15) A1(12-15) A1(12-15)
                const __m256i lhs_mat_23_1_sp2 = _mm256_shuffle_epi32(lhs_mat_23_1, 245);  //A2(12-15) A2(12-15) A3(12-15) A3(12-15) A2(12-15) A2(12-15) A3(12-15) A3(12-15)

                const __m256i lhs_mat_01_2_sp2 = _mm256_shuffle_epi32(lhs_mat_01_2, 245);  //A0(20-23) A0(20-23) A1(20-23) A1(20-23) A0(20-23) A0(20-23) A1(20-23) A1(20-23)
                const __m256i lhs_mat_23_2_sp2 = _mm256_shuffle_epi32(lhs_mat_23_2, 245);  //A2(20-23) A2(20-23) A3(20-23) A3(20-23) A2(20-23) A2(20-23) A3(20-23) A3(20-23)

                const __m256i lhs_mat_01_3_sp2 = _mm256_shuffle_epi32(lhs_mat_01_3, 245);  //A0(28-31) A0(28-31) A1(28-31) A1(28-31) A0(28-31) A0(28-31) A1(28-31) A1(28-31)
                const __m256i lhs_mat_23_3_sp2 = _mm256_shuffle_epi32(lhs_mat_23_3, 245);  //A2(28-31) A2(28-31) A3(28-31) A3(28-31) A2(28-31) A2(28-31) A3(28-31) A3(28-31)

                // The values arranged in shuffle patterns are operated with dot product operation within 32 bit lane i.e corresponding bytes and multiplied and added into 32 bit integers within 32 bit lane
                // Resembles MMLAs into 2x2 matrices in ARM Version
                const __m256i zero = _mm256_setzero_si256();
                __m256i iacc_mat_00_sp1 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_01_3_sp1, rhs_mat_0145_3_sp1), lhs_mat_01_2_sp1, rhs_mat_0145_2_sp1), lhs_mat_01_1_sp1, rhs_mat_0145_1_sp1), lhs_mat_01_0_sp1, rhs_mat_0145_0_sp1);
                __m256i iacc_mat_01_sp1 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_01_3_sp1, rhs_mat_2367_3_sp1), lhs_mat_01_2_sp1, rhs_mat_2367_2_sp1), lhs_mat_01_1_sp1, rhs_mat_2367_1_sp1), lhs_mat_01_0_sp1, rhs_mat_2367_0_sp1);
                __m256i iacc_mat_10_sp1 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_23_3_sp1, rhs_mat_0145_3_sp1), lhs_mat_23_2_sp1, rhs_mat_0145_2_sp1), lhs_mat_23_1_sp1, rhs_mat_0145_1_sp1), lhs_mat_23_0_sp1, rhs_mat_0145_0_sp1);
                __m256i iacc_mat_11_sp1 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_23_3_sp1, rhs_mat_2367_3_sp1), lhs_mat_23_2_sp1, rhs_mat_2367_2_sp1), lhs_mat_23_1_sp1, rhs_mat_2367_1_sp1), lhs_mat_23_0_sp1, rhs_mat_2367_0_sp1);
                __m256i iacc_mat_00_sp2 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_01_3_sp2, rhs_mat_0145_3_sp2), lhs_mat_01_2_sp2, rhs_mat_0145_2_sp2), lhs_mat_01_1_sp2, rhs_mat_0145_1_sp2), lhs_mat_01_0_sp2, rhs_mat_0145_0_sp2);
                __m256i iacc_mat_01_sp2 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_01_3_sp2, rhs_mat_2367_3_sp2), lhs_mat_01_2_sp2, rhs_mat_2367_2_sp2), lhs_mat_01_1_sp2, rhs_mat_2367_1_sp2), lhs_mat_01_0_sp2, rhs_mat_2367_0_sp2);
                __m256i iacc_mat_10_sp2 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_23_3_sp2, rhs_mat_0145_3_sp2), lhs_mat_23_2_sp2, rhs_mat_0145_2_sp2), lhs_mat_23_1_sp2, rhs_mat_0145_1_sp2), lhs_mat_23_0_sp2, rhs_mat_0145_0_sp2);
                __m256i iacc_mat_11_sp2 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_23_3_sp2, rhs_mat_2367_3_sp2), lhs_mat_23_2_sp2, rhs_mat_2367_2_sp2), lhs_mat_23_1_sp2, rhs_mat_2367_1_sp2), lhs_mat_23_0_sp2, rhs_mat_2367_0_sp2);

                // Output of both shuffle patterns are added in order to sum dot product outputs of all 32 values in block
                __m256i iacc_mat_00 = _mm256_add_epi32(iacc_mat_00_sp1, iacc_mat_00_sp2);
                __m256i iacc_mat_01 = _mm256_add_epi32(iacc_mat_01_sp1, iacc_mat_01_sp2);
                __m256i iacc_mat_10 = _mm256_add_epi32(iacc_mat_10_sp1, iacc_mat_10_sp2);
                __m256i iacc_mat_11 = _mm256_add_epi32(iacc_mat_11_sp1, iacc_mat_11_sp2);


                // Straighten out to make 4 row vectors
                __m256i iacc_row_0 = _mm256_blend_epi32(iacc_mat_00, _mm256_shuffle_epi32(iacc_mat_01, 78), 204);
                __m256i iacc_row_1 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_00, 78), iacc_mat_01, 204);
                __m256i iacc_row_2 = _mm256_blend_epi32(iacc_mat_10, _mm256_shuffle_epi32(iacc_mat_11, 78), 204);
                __m256i iacc_row_3 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_10, 78), iacc_mat_11, 204);

                // Load the scale(d) values for all the 4 Q8_0 blocks and repeat it across lanes
                const __m256 row_scale_f32 = GGML_F32Cx8_REPEAT_LOAD(a_ptr[b].d, loadMask);

                // Multiply with appropriate scales and accumulate
                acc_rows[0] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_0), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 0)), acc_rows[0]);
                acc_rows[1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_1), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 85)), acc_rows[1]);
                acc_rows[2] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_2), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_rows[2]);
                acc_rows[3] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_3), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_rows[3]);
            }

            // Store the accumulated values
            for (int i = 0; i < 4; i++) {
                _mm256_storeu_ps((float *)(s + ((y * 4 + i) * bs + x * 8)), acc_rows[i]);
            }
        }
    }
}

#endif // defined(__AVX2__) || defined(__AVX512F__)

void ggml_gemv_q4_0_8x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
#if defined(__AVX2__) || defined(__AVX512F__)
    {
        // Lookup table to convert signed nibbles to signed bytes
        __m256i signextendlut = _mm256_castsi128_si256(_mm_set_epi8(-1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0));
        signextendlut = _mm256_permute2f128_si256(signextendlut, signextendlut, 0);

        gemv_q4_b32_8x8_q8_0_lut_avx<block_q4_0x8>(n, s, bs, vx, vy, nr, nc, signextendlut);

        return;
    }
#endif

    ggml_gemv_q4_0_8x8_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemv_q4_K_8x8_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    assert (n % qk == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

#if defined(__AVX2__)
    // Lookup table to convert signed nibbles to signed bytes
    __m256i signextendlut = _mm256_castsi128_si256(_mm_set_epi8(-1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0));
    signextendlut = _mm256_permute2f128_si256(signextendlut, signextendlut, 0);
    // Shuffle masks to rearrange delta and scale values to multiply with appropriate scales
    __m128i deltamask = _mm_set_epi8(15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0);
    __m128i scalemask = _mm_set_epi8(7, 7, 3, 3, 6, 6, 2, 2, 5, 5, 1, 1, 4, 4, 0, 0);
    // Permute mask used for easier vector processing at later stages
    __m256i finalpermutemask = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);

    // Mask to extract nibbles from bytes
    const __m256i m4b = _mm256_set1_epi8(0x0F);

    int64_t b_nb = n / QK_K;

    const block_q4_Kx8 * b_ptr_start = (const block_q4_Kx8 *)vx;
    const block_q8_K * a_ptr_start = (const block_q8_K *)vy;

    // Process Q8_K blocks one by one
    for (int64_t y = 0; y < nr; y++) {

        // Pointers to LHS blocks of block_q8_K format
        const block_q8_K * a_ptr = a_ptr_start + (y * nb);

        // Take group of eight interleaved block_q4_K structures at each pass of the loop and perform dot product operation
        for (int64_t x = 0; x < nc / 8; x++) {

            // Pointers to RHS blocks
            const block_q4_Kx8 * b_ptr = b_ptr_start + (x * b_nb);

            // Master FP accumulators
            __m256 acc_row = _mm256_setzero_ps();
            __m256 acc_min_rows = _mm256_setzero_ps();

            for (int64_t b = 0; b < nb; b++) {

                // Load and convert to FP32 scale from block_q8_K
                const __m256 row_scale_f32 = _mm256_set1_ps((a_ptr[b].d));

                // Load the scale values for the 8 blocks interleaved in block_q4_Kx8
                // col_scale_f32 rearranged so as to multiply with appropriate quants
                const __m256 col_scale_f32 = GGML_F32Cx8_REARRANGE_LOAD(b_ptr[b].d, deltamask);
                const __m256 col_dmin_f32 = GGML_F32Cx8_LOAD(b_ptr[b].dmin);

                __m256i iacc_b = _mm256_setzero_si256();
                __m256i iacc_min_b = _mm256_setzero_si256();

                const __m256i q8sums = _mm256_loadu_si256((const __m256i * )(a_ptr[b].bsums));
                __m256i q8s = _mm256_castsi128_si256(_mm_hadd_epi16(_mm256_castsi256_si128(q8sums), _mm256_extracti128_si256(q8sums, 1)));
                q8s = _mm256_permute2f128_si256(q8s, q8s, 0);

                // Processes two sub blocks from each Q4_K in each iteration
                for (int sb = 0; sb < QK_K / 64; sb++) {

                    // Load the eight block_q4_K for two sub blocks quantized values interleaved with each other in chunks of eight - B0,B1 ....B6,B7
                    const __m256i rhs_raw_vec_0123_0 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + sb * 256));
                    const __m256i rhs_raw_vec_4567_0 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 32 + sb * 256));
                    const __m256i rhs_raw_vec_0123_1 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 64 + sb * 256));
                    const __m256i rhs_raw_vec_4567_1 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 96 + sb * 256));
                    const __m256i rhs_raw_vec_0123_2 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 128 + sb * 256));
                    const __m256i rhs_raw_vec_4567_2 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 160 + sb * 256));
                    const __m256i rhs_raw_vec_0123_3 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 192 + sb * 256));
                    const __m256i rhs_raw_vec_4567_3 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 224 + sb * 256));

                    // 4-bit -> 8-bit
                    // Values of the first sub block of eight block_q4_K structures for the sb loop
                    const __m256i rhs_vec_0123_00 = _mm256_and_si256(rhs_raw_vec_0123_0, m4b);
                    const __m256i rhs_vec_4567_00 = _mm256_and_si256(rhs_raw_vec_4567_0, m4b);
                    const __m256i rhs_vec_0123_01 = _mm256_and_si256(rhs_raw_vec_0123_1, m4b);
                    const __m256i rhs_vec_4567_01 = _mm256_and_si256(rhs_raw_vec_4567_1, m4b);
                    const __m256i rhs_vec_0123_02 = _mm256_and_si256(rhs_raw_vec_0123_2, m4b);
                    const __m256i rhs_vec_4567_02 = _mm256_and_si256(rhs_raw_vec_4567_2, m4b);
                    const __m256i rhs_vec_0123_03 = _mm256_and_si256(rhs_raw_vec_0123_3, m4b);
                    const __m256i rhs_vec_4567_03 = _mm256_and_si256(rhs_raw_vec_4567_3, m4b);

                    // Values of the second sub block of eight block_q4_K structures when sb = 1
                    const __m256i rhs_vec_0123_10 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_0, 4), m4b);
                    const __m256i rhs_vec_4567_10 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_0, 4), m4b);
                    const __m256i rhs_vec_0123_11 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_1, 4), m4b);
                    const __m256i rhs_vec_4567_11 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_1, 4), m4b);
                    const __m256i rhs_vec_0123_12 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_2, 4), m4b);
                    const __m256i rhs_vec_4567_12 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_2, 4), m4b);
                    const __m256i rhs_vec_0123_13 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_3, 4), m4b);
                    const __m256i rhs_vec_4567_13 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_3, 4), m4b);

                    uint32_t utmp_0[4], utmp_1[4];

                    // Scales and Mins of corresponding sub blocks from different Q8_K structures are stored together
                    // The below block is for eg to extract first sub block's scales and mins from different Q4_K structures for the sb loop
                    memcpy(utmp_0, b_ptr[b].scales + 24 * sb, 12);
                    utmp_0[3] = ((utmp_0[2] >> 4) & kmask2) | (((utmp_0[1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_0 = utmp_0[1] & kmask1;
                    utmp_0[1] = (utmp_0[2] & kmask2) | (((utmp_0[0] >> 6) & kmask3) << 4);
                    utmp_0[2] = uaux_0;
                    utmp_0[0] &= kmask1;

                    // The below block is for eg to extract second sub block's scales and mins from different Q4_K structures for the sb loop
                    memcpy(utmp_1, b_ptr[b].scales + 12 + sb * 24, 12);
                    utmp_1[3] = ((utmp_1[2] >> 4) & kmask2) | (((utmp_1[1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_1 = utmp_1[1] & kmask1;
                    utmp_1[1] = (utmp_1[2] & kmask2) | (((utmp_1[0] >> 6) & kmask3) << 4);
                    utmp_1[2] = uaux_1;
                    utmp_1[0] &= kmask1;

                    // Scales of first sub block in the sb loop
                    const __m128i mins_and_scales_0 = _mm_set_epi32(utmp_0[3], utmp_0[2], utmp_0[1], utmp_0[0]);
                    __m128i scales_rearrange_0 = _mm_shuffle_epi8(mins_and_scales_0, scalemask);
                    __m256i scales_0 = _mm256_cvtepu8_epi16(scales_rearrange_0);

                    // Scales of second sub block in the sb loop
                    __m128i mins_and_scales_1 = _mm_set_epi32(utmp_1[3], utmp_1[2], utmp_1[1], utmp_1[0]);
                    __m128i scales_rearrange_1 = _mm_shuffle_epi8(mins_and_scales_1, scalemask);
                    __m256i scales_1 = _mm256_cvtepu8_epi16(scales_rearrange_1);

                    // Mins of first and second sub block of Q4_K block are arranged side by side
                    __m256i mins_01 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(_mm_shuffle_epi32(mins_and_scales_0, 78), _mm_shuffle_epi32(mins_and_scales_1, 78)));

                    // Load the two sub block values corresponding to sb in block_q8_K in batches of 16 bytes and replicate the same across 256 bit vector
                    __m256i lhs_vec_00 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(a_ptr[b].qs + sb * 64)));
                    __m256i lhs_vec_01 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(a_ptr[b].qs + 16 + sb * 64)));
                    __m256i lhs_vec_10 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(a_ptr[b].qs + 32 + sb * 64)));
                    __m256i lhs_vec_11 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(a_ptr[b].qs + 48 + sb * 64)));

                    lhs_vec_00 = _mm256_permute2f128_si256(lhs_vec_00, lhs_vec_00, 0);
                    lhs_vec_01 = _mm256_permute2f128_si256(lhs_vec_01, lhs_vec_01, 0);
                    lhs_vec_10 = _mm256_permute2f128_si256(lhs_vec_10, lhs_vec_10, 0);
                    lhs_vec_11 = _mm256_permute2f128_si256(lhs_vec_11, lhs_vec_11, 0);

                    // Dot product done within 32 bit lanes and accumulated in the same vector
                    // First done for first sub block and then for second sub block in each sb
                    // B0(0-3) B4(0-3) B1(0-3) B5(0-3) B2(0-3) B6(0-3) B3(0-3) B7(0-3) with A0(0-3)
                    // B0(4-7) B4(4-7) B1(4-7) B5(4-7) B2(4-7) B6(4-7) B3(4-7) B7(4-7) with A0(4-7)
                    // ...........................................................................
                    // B0(28-31) B4(28-31) B1(28-31) B5(28-31) B2(28-31) B6(28-31) B3(28-31) B7(28-31) with A0(28-31)


                    __m256i iacc_0 = _mm256_setzero_si256();
                    __m256i iacc_1 = _mm256_setzero_si256();

                    iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_00 ,_mm256_shuffle_epi32(rhs_vec_4567_00, 177), 170), _mm256_shuffle_epi32(lhs_vec_00, 0)));
                    iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_00, 177) ,rhs_vec_4567_00, 170), _mm256_shuffle_epi32(lhs_vec_00, 85)));

                    iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_01 ,_mm256_shuffle_epi32(rhs_vec_4567_01, 177), 170), _mm256_shuffle_epi32(lhs_vec_00, 170)));
                    iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_01, 177) ,rhs_vec_4567_01, 170), _mm256_shuffle_epi32(lhs_vec_00, 255)));

                    iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_02 ,_mm256_shuffle_epi32(rhs_vec_4567_02, 177), 170), _mm256_shuffle_epi32(lhs_vec_01, 0)));
                    iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_02, 177) ,rhs_vec_4567_02, 170), _mm256_shuffle_epi32(lhs_vec_01, 85)));

                    iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_03 ,_mm256_shuffle_epi32(rhs_vec_4567_03, 177), 170), _mm256_shuffle_epi32(lhs_vec_01, 170)));
                    iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_03, 177) ,rhs_vec_4567_03, 170), _mm256_shuffle_epi32(lhs_vec_01, 255)));

                    iacc_0 = _mm256_madd_epi16(iacc_0, scales_0);

                    iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_10 ,_mm256_shuffle_epi32(rhs_vec_4567_10, 177), 170), _mm256_shuffle_epi32(lhs_vec_10, 0)));
                    iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_10, 177) ,rhs_vec_4567_10, 170), _mm256_shuffle_epi32(lhs_vec_10, 85)));

                    iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_11 ,_mm256_shuffle_epi32(rhs_vec_4567_11, 177), 170), _mm256_shuffle_epi32(lhs_vec_10, 170)));
                    iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_11, 177) ,rhs_vec_4567_11, 170), _mm256_shuffle_epi32(lhs_vec_10, 255)));

                    iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_12 ,_mm256_shuffle_epi32(rhs_vec_4567_12, 177), 170), _mm256_shuffle_epi32(lhs_vec_11, 0)));
                    iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_12, 177) ,rhs_vec_4567_12, 170), _mm256_shuffle_epi32(lhs_vec_11, 85)));

                    iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_13 ,_mm256_shuffle_epi32(rhs_vec_4567_13, 177), 170), _mm256_shuffle_epi32(lhs_vec_11, 170)));
                    iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_13, 177) ,rhs_vec_4567_13, 170), _mm256_shuffle_epi32(lhs_vec_11, 255)));

                    iacc_1 = _mm256_madd_epi16(iacc_1, scales_1);

                    // Accumulate the iacc value for one sb
                    __m256i iacc_sb = _mm256_add_epi32(iacc_0, iacc_1);

                    // Broadcast the bsums of the two sub blocks  of the iteration of Q8_K across the vector
                    // Multiply-Add with corresponding mins of Q4_Kx8 with bsums
                    __m256i q8s_sb = _mm256_shuffle_epi32(q8s, 0);
                    __m256i iacc_min_sb = _mm256_madd_epi16(q8s_sb, mins_01);
                    q8s = _mm256_bsrli_epi128(q8s, 4);

                    // Accumulate for the complete block
                    iacc_b = _mm256_add_epi32(iacc_b, iacc_sb);
                    iacc_min_b = _mm256_add_epi32(iacc_min_b, iacc_min_sb);
                }

                // Multiply-Add with scale values for the complete super block
                acc_row = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_b), _mm256_mul_ps(col_scale_f32, row_scale_f32), acc_row);
                acc_min_rows = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_min_b), _mm256_mul_ps(col_dmin_f32, row_scale_f32), acc_min_rows);

            }

            // Accumulated output values permuted so as to be stored in appropriate order post accumulation
            acc_row = _mm256_permutevar8x32_ps(acc_row, finalpermutemask);
            _mm256_storeu_ps(s + (y * nr + x * 8), _mm256_sub_ps(acc_row, acc_min_rows));
        }
    }

#else
    UNUSED(kmask1);
    UNUSED(kmask2);
    UNUSED(kmask3);
    ggml_gemv_q4_K_8x8_q8_K_generic(n, s, bs, vx, vy, nr, nc);
#endif
}

void ggml_gemv_iq4_nl_8x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
#if defined(__AVX2__)
    __m256i signextendlut = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*)kvalues_iq4nl));
    signextendlut = _mm256_permute2f128_si256(signextendlut, signextendlut, 0);

    gemv_q4_b32_8x8_q8_0_lut_avx<block_iq4_nlx8>(n, s, bs, vx, vy, nr, nc, signextendlut);

    return;
#endif

    ggml_gemv_iq4_nl_8x8_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemv_mxfp4_8x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
#if defined(__AVX2__)
    __m256i signextendlut = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*)kvalues_mxfp4));
    signextendlut = _mm256_permute2f128_si256(signextendlut, signextendlut, 0);

    gemv_q4_b32_8x8_q8_0_lut_avx<block_mxfp4x8>(n, s, bs, vx, vy, nr, nc, signextendlut);

    return;
#endif

    ggml_gemv_mxfp4_8x8_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemv_q2_K_8x8_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;

    assert (n % qk == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

#if defined(__AVX2__)
    // Lookup table to convert signed nibbles to signed bytes
    __m256i signextendlut = _mm256_castsi128_si256(_mm_set_epi8(-1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0));
    signextendlut = _mm256_permute2f128_si256(signextendlut, signextendlut, 0);
    // Shuffle masks to rearrange delta values to multiply with appropriate scales
    __m128i deltamask = _mm_set_epi8(15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0);
    // Permute mask used for easier vector processing at later stages
    __m256i finalpermutemask = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);

    const __m256i m3b = _mm256_set1_epi8(3);
    const __m128i m4b_sse = _mm_set1_epi8(0xF);

    //Mask to get appropriate scales
    __m128i scalemask1 = _mm_set_epi8(14,14,6,6,12,12,4,4,10,10,2,2,8,8,0,0);
    __m128i scalemask2 = _mm_set_epi8(15,15,7,7,13,13,5,5,11,11,3,3,9,9,1,1);

    int64_t b_nb = n / QK_K;

    const block_q2_Kx8 * b_ptr_start = (const block_q2_Kx8 *)vx;
    const block_q8_K * a_ptr_start = (const block_q8_K *)vy;

    // Process Q8_K blocks one by one
    for (int64_t y = 0; y < nr; y++) {

        // Pointers to LHS blocks of block_q8_K format
        const block_q8_K * a_ptr = a_ptr_start + (y * nb);

        // Take group of eight interleaved block_q2_K structures at each pass of the loop and perform dot product operation
        for(int64_t x = 0; x < nc / 8; x++) {

            // Pointers to RHS blocks
            const block_q2_Kx8 * b_ptr = b_ptr_start + (x * b_nb);

            // Master FP accumulators
            __m256 acc_row = _mm256_setzero_ps();
            __m256 acc_min_rows = _mm256_setzero_ps();

            for (int64_t b = 0; b < nb; b++) {

                // Load and convert to FP32 delta from block_q8_K
                const __m256 row_scale_f32 = _mm256_set1_ps((a_ptr[b].d));

                // Load the delta values for the 8 blocks interleaved in block_q2_Kx8
                // col_scale_f32 rearranged so as to multiply with appropriate quants
                const __m256 col_scale_f32 = GGML_F32Cx8_REARRANGE_LOAD(b_ptr[b].d, deltamask);
                const __m256 col_dmin_f32 = GGML_F32Cx8_LOAD(b_ptr[b].dmin);

                __m256i iacc_b = _mm256_setzero_si256();
                __m256i iacc_min_b = _mm256_setzero_si256();

                // Processes eight sub blocks from each Q2_K in each iteration
                for(int sb = 0; sb < QK_K / 128; sb++) {

                    // Load the eight block_q2_K for eight sub blocks quantized values interleaved with each other in chunks of eight - B0,B1 ....B6,B7
                    const __m256i rhs_raw_vec_0123_0 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + sb * 256));
                    const __m256i rhs_raw_vec_4567_0 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 32 + sb * 256));
                    const __m256i rhs_raw_vec_0123_1 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 64 + sb * 256));
                    const __m256i rhs_raw_vec_4567_1 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 96 + sb * 256));
                    const __m256i rhs_raw_vec_0123_2 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 128 + sb * 256));
                    const __m256i rhs_raw_vec_4567_2 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 160 + sb * 256));
                    const __m256i rhs_raw_vec_0123_3 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 192 + sb * 256));
                    const __m256i rhs_raw_vec_4567_3 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 224 + sb * 256));

                    // 2-bit -> 8-bit
                    // Values of the 0th,2nd,4th,6th sub blocks of eight block_q2_K structures for the sb loop
                    const __m256i rhs_vec_0123_00 = _mm256_and_si256(rhs_raw_vec_0123_0, m3b); //B00(0-7) B01(0-7) B02(0-7) B03(0-7)
                    const __m256i rhs_vec_0123_20 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_0, 2), m3b); //B20(0-7) B21(0-7) B22(0-7) B23(0-7)
                    const __m256i rhs_vec_0123_40 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_0, 4), m3b); //B40(0-7) B41(0-7) B42(0-7) B43(0-7)
                    const __m256i rhs_vec_0123_60 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_0, 6), m3b); //B60(0-7) B61(0-7) B62(0-7) B63(0-7)

                    const __m256i rhs_vec_4567_00 = _mm256_and_si256(rhs_raw_vec_4567_0, m3b); //B04(0-7) B05(0-7) B06(0-7) B07(0-7)
                    const __m256i rhs_vec_4567_20 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_0, 2), m3b); //B24(0-7) B25(0-7) B26(0-7) B27(0-7)
                    const __m256i rhs_vec_4567_40 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_0, 4), m3b); //B44(0-7) B45(0-7) B46(0-7) B47(0-7)
                    const __m256i rhs_vec_4567_60 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_0, 6), m3b); //B64(0-7) B65(0-7) B66(0-7) B67(0-7)

                    const __m256i rhs_vec_0123_01 = _mm256_and_si256(rhs_raw_vec_0123_1, m3b); //B00(8-15) B01(8-15) B02(8-15) B03(8-15)
                    const __m256i rhs_vec_0123_21 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_1, 2), m3b); //B20(8-15) B21(8-15) B22(8-15) B23(8-15)
                    const __m256i rhs_vec_0123_41 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_1, 4), m3b); //B40(8-15) B41(8-15) B42(8-15) B43(8-15)
                    const __m256i rhs_vec_0123_61 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_1, 6), m3b); //B60(8-15) B61(8-15) B62(8-15) B63(8-15)

                    const __m256i rhs_vec_4567_01 = _mm256_and_si256(rhs_raw_vec_4567_1, m3b); //B04(8-15) B05(8-15) B06(8-15) B07(8-15)
                    const __m256i rhs_vec_4567_21 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_1, 2), m3b); //B24(8-15) B25(8-15) B26(8-15) B27(8-15)
                    const __m256i rhs_vec_4567_41 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_1, 4), m3b); //B44(8-15) B45(8-15) B46(8-15) B47(8-15)
                    const __m256i rhs_vec_4567_61 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_1, 6), m3b); //B64(8-15) B65(8-15) B66(8-15) B67(8-15)

                    // Values of the 1st,3rd,5th,7th sub blocks of eight block_q2_K structures for the sb loop
                    const __m256i rhs_vec_0123_10 = _mm256_and_si256(rhs_raw_vec_0123_2, m3b); //B10(0-7) B11(0-7) B12(0-7) B13(0-7)
                    const __m256i rhs_vec_0123_30 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_2, 2), m3b); //B30(0-7) B31(0-7) B32(0-7) B33(0-7)
                    const __m256i rhs_vec_0123_50 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_2, 4), m3b); //B50(0-7) B51(0-7) B52(0-7) B53(0-7)
                    const __m256i rhs_vec_0123_70 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_2, 6), m3b); //B70(0-7) B71(0-7) B72(0-7) B73(0-7)

                    const __m256i rhs_vec_4567_10 = _mm256_and_si256(rhs_raw_vec_4567_2, m3b); //B14(0-7) B15(0-7) B16(0-7) B17(0-7)
                    const __m256i rhs_vec_4567_30 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_2, 2), m3b); //B34(0-7) B35(0-7) B36(0-7) B37(0-7)
                    const __m256i rhs_vec_4567_50 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_2, 4), m3b); //B54(0-7) B55(0-7) B56(0-7) B57(0-7)
                    const __m256i rhs_vec_4567_70 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_2, 6), m3b); //B74(0-7) B75(0-7) B76(0-7) B77(0-7)

                    const __m256i rhs_vec_0123_11 = _mm256_and_si256(rhs_raw_vec_0123_3, m3b); //B10(8-15) B11(8-15) B12(8-15) B13(8-15)
                    const __m256i rhs_vec_0123_31 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_3, 2), m3b); //B30(8-15) B31(8-15) B32(8-15) B33(8-15)
                    const __m256i rhs_vec_0123_51 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_3, 4), m3b); //B50(8-15) B51(8-15) B52(8-15) B53(8-15)
                    const __m256i rhs_vec_0123_71 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_3, 6), m3b); //B70(8-15) B71(8-15) B72(8-15) B73(8-15)

                    const __m256i rhs_vec_4567_11 = _mm256_and_si256(rhs_raw_vec_4567_3, m3b); //B14(8-15) B15(8-15) B16(8-15) B17(8-15)
                    const __m256i rhs_vec_4567_31 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_3, 2), m3b); //B34(8-15) B35(8-15) B36(8-15) B37(8-15)
                    const __m256i rhs_vec_4567_51 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_3, 4), m3b); //B54(8-15) B55(8-15) B56(8-15) B57(8-15)
                    const __m256i rhs_vec_4567_71 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_3, 6), m3b); //B74(8-15) B75(8-15) B76(8-15) B77(8-15)

                    //Scales and Mins of corresponding sub blocks from different Q2_K structures are stored together
                    //s00 m00  s01 m01   s10 m10  s11 m11  s20 m20  s21 m21   s30 m30  s31 m31  s40 m40  s41 m41   s50 m50  s51 m51  s60 m60  s61 m61   s70 m70  s71 m71

                    const __m128i mins_and_scales_01 = _mm_loadu_si128((const __m128i *)(b_ptr[b].scales + sb * 64));
                    const __m128i mins_and_scales_23 = _mm_loadu_si128((const __m128i *)(b_ptr[b].scales + 16 + sb * 64));
                    const __m128i mins_and_scales_45 = _mm_loadu_si128((const __m128i *)(b_ptr[b].scales + 32 + sb * 64));
                    const __m128i mins_and_scales_67 = _mm_loadu_si128((const __m128i *)(b_ptr[b].scales + 48 + sb * 64));

                    // Extract scales which is lower half from mins_and_scales
                    const __m128i scales_01 = _mm_and_si128(mins_and_scales_01, m4b_sse);
                    const __m128i scales_23 = _mm_and_si128(mins_and_scales_23, m4b_sse);
                    const __m128i scales_45 = _mm_and_si128(mins_and_scales_45, m4b_sse);
                    const __m128i scales_67 = _mm_and_si128(mins_and_scales_67, m4b_sse);

                    // Extract mins which is upper half from mins_and_scales
                    const __m256i mins_01 = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(mins_and_scales_01, 4), m4b_sse));
                    const __m256i mins_23 = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(mins_and_scales_23, 4), m4b_sse));
                    const __m256i mins_45 = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(mins_and_scales_45, 4), m4b_sse));
                    const __m256i mins_67 = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(mins_and_scales_67, 4), m4b_sse));

                    // Scales of sub blocks in the sb loop
                    // Scales of the 0th sub block from each super block
                    __m128i scales_rearrange_0 = _mm_shuffle_epi8(scales_01, scalemask1);
                    __m256i scales_0 = _mm256_cvtepu8_epi16(scales_rearrange_0);

                    // Scales of the 1st sub block from each super block
                    __m128i scales_rearrange_1 = _mm_shuffle_epi8(scales_01, scalemask2);
                    __m256i scales_1 = _mm256_cvtepu8_epi16(scales_rearrange_1);

                    // Scales of the 2nd sub block from each super block
                    __m128i scales_rearrange_2 = _mm_shuffle_epi8(scales_23, scalemask1);
                    __m256i scales_2 = _mm256_cvtepu8_epi16(scales_rearrange_2);

                    // Scales of the 3rd sub block from each super block
                    __m128i scales_rearrange_3 = _mm_shuffle_epi8(scales_23, scalemask2);
                    __m256i scales_3 = _mm256_cvtepu8_epi16(scales_rearrange_3);

                    // Scales of the 4th sub block from each super block
                    __m128i scales_rearrange_4 = _mm_shuffle_epi8(scales_45, scalemask1);
                    __m256i scales_4 = _mm256_cvtepu8_epi16(scales_rearrange_4);

                    // Scales of the 5th sub block from each super block
                    __m128i scales_rearrange_5 = _mm_shuffle_epi8(scales_45, scalemask2);
                    __m256i scales_5 = _mm256_cvtepu8_epi16(scales_rearrange_5);

                    // Scales of the 6th sub block from each super block
                    __m128i scales_rearrange_6 = _mm_shuffle_epi8(scales_67, scalemask1);
                    __m256i scales_6 = _mm256_cvtepu8_epi16(scales_rearrange_6);

                    // Scales of the 7th sub block from each super block
                    __m128i scales_rearrange_7 = _mm_shuffle_epi8(scales_67, scalemask2);
                    __m256i scales_7 = _mm256_cvtepu8_epi16(scales_rearrange_7);

                    // Load the sub block values corresponding to sb in block_q8_K in batches of 16 bytes and replicate the same across 256 bit vector
                    __m256i lhs_vec_0 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(a_ptr[b].qs + sb * 128)));
                    __m256i lhs_vec_1 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(a_ptr[b].qs + 16 + sb * 128)));
                    __m256i lhs_vec_2 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(a_ptr[b].qs + 32 + sb * 128)));
                    __m256i lhs_vec_3 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(a_ptr[b].qs + 48 + sb * 128)));
                    __m256i lhs_vec_4 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(a_ptr[b].qs + 64 + sb * 128)));
                    __m256i lhs_vec_5 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(a_ptr[b].qs + 80 + sb * 128)));
                    __m256i lhs_vec_6 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(a_ptr[b].qs + 96 + sb * 128)));
                    __m256i lhs_vec_7 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(a_ptr[b].qs + 112 + sb * 128)));

                    lhs_vec_0 = _mm256_permute2f128_si256(lhs_vec_0, lhs_vec_0, 0);
                    lhs_vec_1 = _mm256_permute2f128_si256(lhs_vec_1, lhs_vec_1, 0);
                    lhs_vec_2 = _mm256_permute2f128_si256(lhs_vec_2, lhs_vec_2, 0);
                    lhs_vec_3 = _mm256_permute2f128_si256(lhs_vec_3, lhs_vec_3, 0);
                    lhs_vec_4 = _mm256_permute2f128_si256(lhs_vec_4, lhs_vec_4, 0);
                    lhs_vec_5 = _mm256_permute2f128_si256(lhs_vec_5, lhs_vec_5, 0);
                    lhs_vec_6 = _mm256_permute2f128_si256(lhs_vec_6, lhs_vec_6, 0);
                    lhs_vec_7 = _mm256_permute2f128_si256(lhs_vec_7, lhs_vec_7, 0);

                    __m256i iacc_0 = _mm256_setzero_si256();
                    __m256i iacc_1 = _mm256_setzero_si256();
                    __m256i iacc_2 = _mm256_setzero_si256();
                    __m256i iacc_3 = _mm256_setzero_si256();
                    __m256i iacc_4 = _mm256_setzero_si256();
                    __m256i iacc_5 = _mm256_setzero_si256();
                    __m256i iacc_6 = _mm256_setzero_si256();
                    __m256i iacc_7 = _mm256_setzero_si256();

                    // Dot product done within 32 bit lanes and accumulated in the same vector
                    // First done for 0th sub block and then for seven (1st - 7th) other sub blocks processed for each sb (sb < QK_K/128 loop)                    // B0(0-3) B4(0-3) B1(0-3) B5(0-3) B2(0-3) B6(0-3) B3(0-3) B7(0-3) with A0(0-3)
                    // B0(4-7) B4(4-7) B1(4-7) B5(4-7) B2(4-7) B6(4-7) B3(4-7) B7(4-7) with A0(4-7)
                    // B0(8-11) B4(8-11) B1(8-11) B5(8-11) B2(8-11) B6(8-11) B3(8-11) B7(8-11) with A0(8-11)
                    // B0(12-15) B4(12-15) B1(12-15) B5(12-15) B2(12-15) B6(12-15) B3(12-15) B7(12-15) with A0(12-15)

                    iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_00 ,_mm256_shuffle_epi32(rhs_vec_4567_00, 177), 170), _mm256_shuffle_epi32(lhs_vec_0, 0)));
                    iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_00, 177) ,rhs_vec_4567_00, 170), _mm256_shuffle_epi32(lhs_vec_0, 85)));

                    iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_01 ,_mm256_shuffle_epi32(rhs_vec_4567_01, 177), 170), _mm256_shuffle_epi32(lhs_vec_0, 170)));
                    iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_01, 177) ,rhs_vec_4567_01, 170), _mm256_shuffle_epi32(lhs_vec_0, 255)));

                    iacc_0 = _mm256_madd_epi16(iacc_0, scales_0);

                    iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_10 ,_mm256_shuffle_epi32(rhs_vec_4567_10, 177), 170), _mm256_shuffle_epi32(lhs_vec_1, 0)));
                    iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_10, 177) ,rhs_vec_4567_10, 170), _mm256_shuffle_epi32(lhs_vec_1, 85)));

                    iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_11 ,_mm256_shuffle_epi32(rhs_vec_4567_11, 177), 170), _mm256_shuffle_epi32(lhs_vec_1, 170)));
                    iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_11, 177) ,rhs_vec_4567_11, 170), _mm256_shuffle_epi32(lhs_vec_1, 255)));

                    iacc_1 = _mm256_madd_epi16(iacc_1, scales_1);

                    iacc_2 = _mm256_add_epi16(iacc_2, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_20 ,_mm256_shuffle_epi32(rhs_vec_4567_20, 177), 170), _mm256_shuffle_epi32(lhs_vec_2, 0)));
                    iacc_2 = _mm256_add_epi16(iacc_2, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_20, 177) ,rhs_vec_4567_20, 170), _mm256_shuffle_epi32(lhs_vec_2, 85)));

                    iacc_2 = _mm256_add_epi16(iacc_2, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_21 ,_mm256_shuffle_epi32(rhs_vec_4567_21, 177), 170), _mm256_shuffle_epi32(lhs_vec_2, 170)));
                    iacc_2 = _mm256_add_epi16(iacc_2, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_21, 177) ,rhs_vec_4567_21, 170), _mm256_shuffle_epi32(lhs_vec_2, 255)));

                    iacc_2 = _mm256_madd_epi16(iacc_2, scales_2);

                    iacc_3 = _mm256_add_epi16(iacc_3, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_30 ,_mm256_shuffle_epi32(rhs_vec_4567_30, 177), 170), _mm256_shuffle_epi32(lhs_vec_3, 0)));
                    iacc_3 = _mm256_add_epi16(iacc_3, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_30, 177) ,rhs_vec_4567_30, 170), _mm256_shuffle_epi32(lhs_vec_3, 85)));

                    iacc_3 = _mm256_add_epi16(iacc_3, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_31 ,_mm256_shuffle_epi32(rhs_vec_4567_31, 177), 170), _mm256_shuffle_epi32(lhs_vec_3, 170)));
                    iacc_3 = _mm256_add_epi16(iacc_3, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_31, 177) ,rhs_vec_4567_31, 170), _mm256_shuffle_epi32(lhs_vec_3, 255)));

                    iacc_3 = _mm256_madd_epi16(iacc_3, scales_3);

                    iacc_4 = _mm256_add_epi16(iacc_4, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_40 ,_mm256_shuffle_epi32(rhs_vec_4567_40, 177), 170), _mm256_shuffle_epi32(lhs_vec_4, 0)));
                    iacc_4 = _mm256_add_epi16(iacc_4, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_40, 177) ,rhs_vec_4567_40, 170), _mm256_shuffle_epi32(lhs_vec_4, 85)));

                    iacc_4 = _mm256_add_epi16(iacc_4, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_41 ,_mm256_shuffle_epi32(rhs_vec_4567_41, 177), 170), _mm256_shuffle_epi32(lhs_vec_4, 170)));
                    iacc_4 = _mm256_add_epi16(iacc_4, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_41, 177) ,rhs_vec_4567_41, 170), _mm256_shuffle_epi32(lhs_vec_4, 255)));

                    iacc_4 = _mm256_madd_epi16(iacc_4, scales_4);

                    iacc_5 = _mm256_add_epi16(iacc_5, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_50 ,_mm256_shuffle_epi32(rhs_vec_4567_50, 177), 170), _mm256_shuffle_epi32(lhs_vec_5, 0)));
                    iacc_5 = _mm256_add_epi16(iacc_5, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_50, 177) ,rhs_vec_4567_50, 170), _mm256_shuffle_epi32(lhs_vec_5, 85)));

                    iacc_5 = _mm256_add_epi16(iacc_5, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_51 ,_mm256_shuffle_epi32(rhs_vec_4567_51, 177), 170), _mm256_shuffle_epi32(lhs_vec_5, 170)));
                    iacc_5 = _mm256_add_epi16(iacc_5, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_51, 177) ,rhs_vec_4567_51, 170), _mm256_shuffle_epi32(lhs_vec_5, 255)));

                    iacc_5 = _mm256_madd_epi16(iacc_5, scales_5);

                    iacc_6 = _mm256_add_epi16(iacc_6, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_60 ,_mm256_shuffle_epi32(rhs_vec_4567_60, 177), 170), _mm256_shuffle_epi32(lhs_vec_6, 0)));
                    iacc_6 = _mm256_add_epi16(iacc_6, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_60, 177) ,rhs_vec_4567_60, 170), _mm256_shuffle_epi32(lhs_vec_6, 85)));

                    iacc_6 = _mm256_add_epi16(iacc_6, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_61 ,_mm256_shuffle_epi32(rhs_vec_4567_61, 177), 170), _mm256_shuffle_epi32(lhs_vec_6, 170)));
                    iacc_6 = _mm256_add_epi16(iacc_6, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_61, 177) ,rhs_vec_4567_61, 170), _mm256_shuffle_epi32(lhs_vec_6, 255)));

                    iacc_6 = _mm256_madd_epi16(iacc_6, scales_6);

                    iacc_7 = _mm256_add_epi16(iacc_7, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_70 ,_mm256_shuffle_epi32(rhs_vec_4567_70, 177), 170), _mm256_shuffle_epi32(lhs_vec_7, 0)));
                    iacc_7 = _mm256_add_epi16(iacc_7, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_70, 177) ,rhs_vec_4567_70, 170), _mm256_shuffle_epi32(lhs_vec_7, 85)));

                    iacc_7 = _mm256_add_epi16(iacc_7, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_71 ,_mm256_shuffle_epi32(rhs_vec_4567_71, 177), 170), _mm256_shuffle_epi32(lhs_vec_7, 170)));
                    iacc_7 = _mm256_add_epi16(iacc_7, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_71, 177) ,rhs_vec_4567_71, 170), _mm256_shuffle_epi32(lhs_vec_7, 255)));

                    iacc_7 = _mm256_madd_epi16(iacc_7, scales_7);

                    // Accumulate the iacc value for one sb
                    __m256i iacc_sb = _mm256_add_epi32(_mm256_add_epi32(_mm256_add_epi32(iacc_0, iacc_1), _mm256_add_epi32(iacc_2, iacc_3)), _mm256_add_epi32(_mm256_add_epi32(iacc_4, iacc_5), _mm256_add_epi32(iacc_6, iacc_7)));

                    __m128i q8sums = _mm_loadu_si128((const __m128i *)(a_ptr[b].bsums + sb * 8));
                    __m256i q8s = _mm256_castsi128_si256(q8sums);
                    q8s= _mm256_permute2f128_si256(q8s, q8s, 0);

                    // Broadcast the bsums of the two corresponding subblocks of q8_k
                    // Multiply-Add with corresponding mins of Q2_Kx8 with bsums
                    __m256i iacc_min_sb_01 = _mm256_madd_epi16(_mm256_shuffle_epi32(q8s, 0), mins_01);
                    __m256i iacc_min_sb_23 = _mm256_madd_epi16(_mm256_shuffle_epi32(q8s, 85), mins_23);
                    __m256i iacc_min_sb_45 = _mm256_madd_epi16(_mm256_shuffle_epi32(q8s, 170), mins_45);
                    __m256i iacc_min_sb_67 = _mm256_madd_epi16(_mm256_shuffle_epi32(q8s, 255), mins_67);

                    __m256i iacc_min_sb = _mm256_add_epi32(_mm256_add_epi32(iacc_min_sb_01, iacc_min_sb_23), _mm256_add_epi32(iacc_min_sb_45,iacc_min_sb_67));

                    // Accumulate for the complete block
                    iacc_b = _mm256_add_epi32(iacc_b, iacc_sb);
                    iacc_min_b = _mm256_add_epi32(iacc_min_b, iacc_min_sb);
                }

                //Multiply-Add with scale values for complete super block
                acc_row = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_b), _mm256_mul_ps(col_scale_f32, row_scale_f32), acc_row);
                acc_min_rows = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_min_b), _mm256_mul_ps(col_dmin_f32, row_scale_f32), acc_min_rows);
            }
            // Accumulated output values permuted so as to be stored in appropriate order post accumulation
            acc_row = _mm256_permutevar8x32_ps(acc_row, finalpermutemask);
            _mm256_storeu_ps(s + (y * nr + x * 8), _mm256_sub_ps(acc_row, acc_min_rows));
        }
    }
#else

    ggml_gemv_q2_K_8x8_q8_K_generic(n, s, bs, vx, vy, nr, nc);

#endif
}

void ggml_gemv_iq2_xs_r8_q8_K(
        int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy,
        int nr, int nc) {
#if defined(__AVX512F__) && defined(__AVX512BW__)
    constexpr int ncols_interleaved = 8;
    constexpr int values_per_subblock = 16;
    constexpr int subblocks_per_block = QK_K / values_per_subblock;

    assert(n % QK_K == 0);
    assert(nc % ncols_interleaved == 0);

    const int nb = n / QK_K;
    const block_iq2_xs_r8 * b_ptr_start = (const block_iq2_xs_r8 *) vx;
    const block_q8_K * a_ptr_start = (const block_q8_K *) vy;

    // Bias the signed alphabet into the unsigned VNNI operand range.  Q8_K
    // already stores the sum of each 16-value activation subblock, so the
    // added 64*sum(q8) can be removed once per four output rows.
    const __m128i lut128 = _mm_setr_epi8(
        21, 39, 56, 72, 89, 107, 64, 64,
        64, 64, 64, 64, 64,  64, 64, 64);
    const __m512i lut = _mm512_broadcast_i32x4(lut128);
    const __m512i low_nibble_u16 = _mm512_set1_epi16(0x000f);
    const __m512i high_nibble_u16 = _mm512_set1_epi16(0x0f00);
    const __m128i nibble_mask = _mm_set1_epi32(0x0f);
    const __m128i one = _mm_set1_epi32(1);

    for (int y = 0; y < nr; ++y) {
        const block_q8_K * a_ptr = a_ptr_start + y * nb;
        float * out = s + y * bs;

        for (int x = 0; x < nc / ncols_interleaved; ++x) {
            const block_iq2_xs_r8 * b_ptr = b_ptr_start + x * nb;
            __m256 sumf = _mm256_setzero_ps();

            for (int b = 0; b < nb; ++b) {
                __m128i isum[2] = {
                    _mm_setzero_si128(),
                    _mm_setzero_si128(),
                };

                for (int sb = 0; sb < subblocks_per_block; ++sb) {
                    const __m128i aq128 = _mm_loadu_si128(
                        (const __m128i *) (a_ptr[b].qs + sb * values_per_subblock));
                    const __m512i aq = _mm512_broadcast_i32x4(aq128);
                    const int ib32 = sb / 2;
                    const bool high_scale = (sb & 1) != 0;

                    for (int row_group = 0; row_group < 2; ++row_group) {
                        const __m256i packed_codes = _mm256_loadu_si256(
                            (const __m256i *) (b_ptr[b].qs + (sb * 2 + row_group) * 32));
                        const __m512i packed_u16 = _mm512_cvtepu8_epi16(packed_codes);
                        const __m512i codes = _mm512_or_si512(
                            _mm512_and_si512(packed_u16, low_nibble_u16),
                            _mm512_and_si512(_mm512_slli_epi16(packed_u16, 4), high_nibble_u16));

                        const __m512i weights = _mm512_shuffle_epi8(lut, codes);
                        __m512i dots = mul_sum_us8_pairs_acc_int32x16(
                            _mm512_setzero_si512(), weights, aq);
                        dots = _mm512_add_epi32(
                            dots, _mm512_shuffle_epi32(dots, (_MM_PERM_ENUM) _MM_SHUFFLE(2, 3, 0, 1)));
                        dots = _mm512_add_epi32(
                            dots, _mm512_shuffle_epi32(dots, (_MM_PERM_ENUM) _MM_SHUFFLE(1, 0, 3, 2)));
                        __m128i row_dots = _mm512_castsi512_si128(
                            _mm512_maskz_compress_epi32((__mmask16) 0x1111, dots));
                        row_dots = _mm_sub_epi32(
                            row_dots, _mm_set1_epi32(64 * a_ptr[b].bsums[sb]));

                        uint32_t packed_scales;
                        memcpy(
                            &packed_scales,
                            b_ptr[b].scales + ib32 * ncols_interleaved + row_group * 4,
                            sizeof(packed_scales));
                        __m128i scales = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) packed_scales));
                        scales = high_scale ? _mm_srli_epi32(scales, 4) : _mm_and_si128(scales, nibble_mask);
                        scales = _mm_add_epi32(_mm_slli_epi32(scales, 1), one);
                        isum[row_group] = _mm_add_epi32(
                            isum[row_group], _mm_mullo_epi32(row_dots, scales));
                    }
                }

                const __m256i isum8 = _mm256_set_m128i(isum[1], isum[0]);
                const __m256 weight_scale = _mm256_cvtph_ps(
                    _mm_loadu_si128((const __m128i *) b_ptr[b].d));
                const __m256 block_scale = _mm256_mul_ps(
                    weight_scale, _mm256_set1_ps(a_ptr[b].d * 0.125f));
                sumf = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(isum8),
                    block_scale, sumf);
            }

            _mm256_storeu_ps(out + x * ncols_interleaved, sumf);
        }
    }
    return;
#endif
    ggml_gemv_iq2_xs_r8_q8_K_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemm_iq2_xs_r8_q8_K(
        int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy,
        int nr, int nc) {
    ggml_gemv_iq2_xs_r8_q8_K(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemv_iq3_xxs_r8_q8_K(
        int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy,
        int nr, int nc) {
#if defined(__AVX512F__) && defined(__AVX512BW__)
    constexpr int ncols_interleaved = 8;
    constexpr int values_per_subblock = 16;
    constexpr int subblocks_per_block = QK_K / values_per_subblock;

    assert(n % QK_K == 0);
    assert(nc % ncols_interleaved == 0);

    const int nb = n / QK_K;
    const block_iq3_xxs_r8 * b_ptr_start = (const block_iq3_xxs_r8 *) vx;
    const block_q8_K * a_ptr_start = (const block_q8_K *) vy;

    // Bias the signed IQ3 alphabet by 64 so VNNI can consume it directly as
    // its unsigned operand.  Q8_K records each 16-value activation sum, which
    // lets us subtract the exact 64*sum(q8) correction after the dot product.
    // This removes the abs/sign-mask sequence from every routed inner loop.
    const __m128i lut128 = _mm_setr_epi8(
          2,  12,  20,  28,  36,  44,  52, 60,
         68,  76,  84,  92, 100, 108, 116, 126);
    const __m512i lut = _mm512_broadcast_i32x4(lut128);
    const __m512i low_nibble_u16 = _mm512_set1_epi16(0x000f);
    const __m512i high_nibble_u16 = _mm512_set1_epi16(0x0f00);
    const __m128i one = _mm_set1_epi32(1);

    for (int y = 0; y < nr; ++y) {
        const block_q8_K * a_ptr = a_ptr_start + y * nb;
        float * out = s + y * bs;

        for (int x = 0; x < nc / ncols_interleaved; ++x) {
            const block_iq3_xxs_r8 * b_ptr = b_ptr_start + x * nb;
            __m256 sumf = _mm256_setzero_ps();

            for (int b = 0; b < nb; ++b) {
                __m128i isum[2] = {
                    _mm_setzero_si128(),
                    _mm_setzero_si128(),
                };

                for (int sb = 0; sb < subblocks_per_block; ++sb) {
                    const __m128i aq128 = _mm_loadu_si128(
                        (const __m128i *) (a_ptr[b].qs + sb * values_per_subblock));
                    const __m512i aq = _mm512_broadcast_i32x4(aq128);
                    const int ib32 = sb / 2;
                    const uint32_t packed_scales = b_ptr[b].scales[ib32];

                    for (int row_group = 0; row_group < 2; ++row_group) {
                        const __m256i packed_codes = _mm256_loadu_si256(
                            (const __m256i *) (b_ptr[b].qs + (sb * 2 + row_group) * 32));
                        const __m512i packed_u16 = _mm512_cvtepu8_epi16(packed_codes);
                        const __m512i codes = _mm512_or_si512(
                            _mm512_and_si512(packed_u16, low_nibble_u16),
                            _mm512_and_si512(_mm512_slli_epi16(packed_u16, 4), high_nibble_u16));

                        const __m512i weights = _mm512_shuffle_epi8(lut, codes);
                        __m512i dots = mul_sum_us8_pairs_acc_int32x16(
                            _mm512_setzero_si512(), weights, aq);
                        dots = _mm512_add_epi32(
                            dots, _mm512_shuffle_epi32(dots, (_MM_PERM_ENUM) _MM_SHUFFLE(2, 3, 0, 1)));
                        dots = _mm512_add_epi32(
                            dots, _mm512_shuffle_epi32(dots, (_MM_PERM_ENUM) _MM_SHUFFLE(1, 0, 3, 2)));
                        __m128i row_dots = _mm512_castsi512_si128(
                            _mm512_maskz_compress_epi32((__mmask16) 0x1111, dots));
                        row_dots = _mm_sub_epi32(
                            row_dots, _mm_set1_epi32(64 * a_ptr[b].bsums[sb]));

                        const uint32_t packed =
                            (packed_scales >> (16 * row_group)) & 0xffff;
                        const uint32_t expanded =
                            (packed & 0x000f) |
                            ((packed & 0x00f0) << 4) |
                            ((packed & 0x0f00) << 8) |
                            ((packed & 0xf000) << 12);
                        __m128i scales = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) expanded));
                        scales = _mm_add_epi32(_mm_slli_epi32(scales, 1), one);
                        isum[row_group] = _mm_add_epi32(
                            isum[row_group], _mm_mullo_epi32(row_dots, scales));
                    }
                }

                const __m256i isum8 = _mm256_set_m128i(isum[1], isum[0]);
                const __m256 weight_scale = _mm256_cvtph_ps(
                    _mm_loadu_si128((const __m128i *) b_ptr[b].d));
                const __m256 block_scale = _mm256_mul_ps(
                    weight_scale, _mm256_set1_ps(a_ptr[b].d * 0.25f));
                sumf = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(isum8),
                    block_scale, sumf);
            }

            _mm256_storeu_ps(out + x * ncols_interleaved, sumf);
        }
    }
    return;
#endif
    ggml_gemv_iq3_xxs_r8_q8_K_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemm_iq3_xxs_r8_q8_K(
        int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy,
        int nr, int nc) {
    ggml_gemv_iq3_xxs_r8_q8_K(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemv_q4_K_r8_q8_K(
        int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy,
        int nr, int nc) {
#if defined(__AVX512F__) && defined(__AVX512BW__)
    static const bool trace_enabled = []() {
        const char * value = std::getenv("GGML_CPU_Q4_K_REPACK_TRACE");
        return value != nullptr && strcmp(value, "1") == 0;
    }();
    static std::atomic_flag traced = ATOMIC_FLAG_INIT;
    if (trace_enabled && !traced.test_and_set(std::memory_order_relaxed)) {
        fprintf(stderr, "q4_K_r8: AVX-512/VNNI GEMV n=%d nr=%d nc=%d\n", n, nr, nc);
    }

    constexpr int ncols_interleaved = 8;
    constexpr int values_per_subblock = 16;
    constexpr int subblocks_per_block = QK_K / values_per_subblock;

    assert(n % QK_K == 0);
    assert(nc % ncols_interleaved == 0);

    const int nb = n / QK_K;
    const block_q4_K_r8 * b_ptr_start = (const block_q4_K_r8 *) vx;
    const block_q8_K * a_ptr_start = (const block_q8_K *) vy;
    const __m512i low_nibble_u16 = _mm512_set1_epi16(0x000f);
    const __m512i high_nibble_u16 = _mm512_set1_epi16(0x0f00);

    for (int y = 0; y < nr; ++y) {
        const block_q8_K * a_ptr = a_ptr_start + y * nb;
        float * out = s + y * bs;

        for (int x = 0; x < nc / ncols_interleaved; ++x) {
            const block_q4_K_r8 * b_ptr = b_ptr_start + x * nb;
            __m256 sumf = _mm256_setzero_ps();

            for (int b = 0; b < nb; ++b) {
                __m128i isum[2] = {
                    _mm_setzero_si128(),
                    _mm_setzero_si128(),
                };
                __m128i imin[2] = {
                    _mm_setzero_si128(),
                    _mm_setzero_si128(),
                };

                for (int ib32 = 0; ib32 < subblocks_per_block / 2; ++ib32) {
                    __m128i row_dots[2] = {
                        _mm_setzero_si128(),
                        _mm_setzero_si128(),
                    };
                    int bsum_total = 0;

                    for (int half = 0; half < 2; ++half) {
                        const int sb = ib32 * 2 + half;
                        const __m128i aq128 = _mm_loadu_si128(
                            (const __m128i *) (a_ptr[b].qs + sb * values_per_subblock));
                        const __m512i aq = _mm512_broadcast_i32x4(aq128);
                        bsum_total += a_ptr[b].bsums[sb];

                        for (int row_group = 0; row_group < 2; ++row_group) {
                            const int group = sb * 2 + row_group;
                            const __m256i packed_low = _mm256_loadu_si256(
                                (const __m256i *) (b_ptr[b].ql + group * 32));
                            const __m512i packed_u16 = _mm512_cvtepu8_epi16(packed_low);
                            const __m512i weights = _mm512_or_si512(
                                _mm512_and_si512(packed_u16, low_nibble_u16),
                                _mm512_and_si512(
                                    _mm512_slli_epi16(packed_u16, 4), high_nibble_u16));
                            __m512i dots = mul_sum_us8_pairs_acc_int32x16(
                                _mm512_setzero_si512(), weights, aq);
                            dots = _mm512_add_epi32(
                                dots, _mm512_shuffle_epi32(dots, (_MM_PERM_ENUM) _MM_SHUFFLE(2, 3, 0, 1)));
                            dots = _mm512_add_epi32(
                                dots, _mm512_shuffle_epi32(dots, (_MM_PERM_ENUM) _MM_SHUFFLE(1, 0, 3, 2)));
                            row_dots[row_group] = _mm_add_epi32(
                                row_dots[row_group], _mm512_castsi512_si128(
                                    _mm512_maskz_compress_epi32((__mmask16) 0x1111, dots)));
                        }
                    }

                    const __m128i bsum = _mm_set1_epi32(bsum_total);
                    for (int row_group = 0; row_group < 2; ++row_group) {
                        uint32_t packed_scales;
                        uint32_t packed_mins;
                        memcpy(
                            &packed_scales,
                            b_ptr[b].scales + ib32 * ncols_interleaved + row_group * 4,
                            sizeof(packed_scales));
                        memcpy(
                            &packed_mins,
                            b_ptr[b].mins + ib32 * ncols_interleaved + row_group * 4,
                            sizeof(packed_mins));
                        const __m128i scales = _mm_cvtepu8_epi32(
                            _mm_cvtsi32_si128((int) packed_scales));
                        const __m128i mins = _mm_cvtepu8_epi32(
                            _mm_cvtsi32_si128((int) packed_mins));
                        isum[row_group] = _mm_add_epi32(
                            isum[row_group], _mm_mullo_epi32(row_dots[row_group], scales));
                        imin[row_group] = _mm_add_epi32(
                            imin[row_group], _mm_mullo_epi32(mins, bsum));
                    }
                }

                const __m256i isum8 = _mm256_set_m128i(isum[1], isum[0]);
                const __m256i imin8 = _mm256_set_m128i(imin[1], imin[0]);
                const __m256 activation_scale = _mm256_set1_ps(a_ptr[b].d);
                const __m256 weight_scale = _mm256_mul_ps(
                    _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) b_ptr[b].d)),
                    activation_scale);
                const __m256 min_scale = _mm256_mul_ps(
                    _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) b_ptr[b].dmin)),
                    activation_scale);
                sumf = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(isum8), weight_scale, sumf);
                sumf = _mm256_sub_ps(
                    sumf, _mm256_mul_ps(_mm256_cvtepi32_ps(imin8), min_scale));
            }

            _mm256_storeu_ps(out + x * ncols_interleaved, sumf);
        }
    }
    return;
#endif
    ggml_gemv_q4_K_r8_q8_K_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemv_q5_K_r8_q8_K(
        int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy,
        int nr, int nc) {
#if defined(__AVX512F__) && defined(__AVX512BW__)
    static const bool trace_enabled = []() {
        const char * value = std::getenv("GGML_CPU_Q5_K_REPACK_TRACE");
        return value != nullptr && strcmp(value, "1") == 0;
    }();
    static std::atomic_flag traced = ATOMIC_FLAG_INIT;
    if (trace_enabled && !traced.test_and_set(std::memory_order_relaxed)) {
        fprintf(stderr, "q5_K_r8: AVX-512/VNNI GEMV n=%d nr=%d nc=%d\n", n, nr, nc);
    }

    constexpr int ncols_interleaved = 8;
    constexpr int values_per_subblock = 16;
    constexpr int subblocks_per_block = QK_K / values_per_subblock;

    assert(n % QK_K == 0);
    assert(nc % ncols_interleaved == 0);

    const int nb = n / QK_K;
    const block_q5_K_r8 * b_ptr_start = (const block_q5_K_r8 *) vx;
    const block_q8_K * a_ptr_start = (const block_q8_K *) vy;
    const __m512i low_nibble_u16 = _mm512_set1_epi16(0x000f);
    const __m512i high_nibble_u16 = _mm512_set1_epi16(0x0f00);
    const __m512i high_bit_value = _mm512_set1_epi8(16);

    for (int y = 0; y < nr; ++y) {
        const block_q8_K * a_ptr = a_ptr_start + y * nb;
        float * out = s + y * bs;

        for (int x = 0; x < nc / ncols_interleaved; ++x) {
            const block_q5_K_r8 * b_ptr = b_ptr_start + x * nb;
            __m256 sumf = _mm256_setzero_ps();

            for (int b = 0; b < nb; ++b) {
                __m128i isum[2] = {
                    _mm_setzero_si128(),
                    _mm_setzero_si128(),
                };
                __m128i imin[2] = {
                    _mm_setzero_si128(),
                    _mm_setzero_si128(),
                };

                for (int ib32 = 0; ib32 < subblocks_per_block / 2; ++ib32) {
                    __m128i row_dots[2] = {
                        _mm_setzero_si128(),
                        _mm_setzero_si128(),
                    };
                    int bsum_total = 0;

                    for (int half = 0; half < 2; ++half) {
                        const int sb = ib32 * 2 + half;
                        const __m128i aq128 = _mm_loadu_si128(
                            (const __m128i *) (a_ptr[b].qs + sb * values_per_subblock));
                        const __m512i aq = _mm512_broadcast_i32x4(aq128);
                        bsum_total += a_ptr[b].bsums[sb];

                        for (int row_group = 0; row_group < 2; ++row_group) {
                            const int group = sb * 2 + row_group;
                            const __m256i packed_low = _mm256_loadu_si256(
                                (const __m256i *) (b_ptr[b].ql + group * 32));
                            const __m512i packed_u16 = _mm512_cvtepu8_epi16(packed_low);
                            __m512i weights = _mm512_or_si512(
                                _mm512_and_si512(packed_u16, low_nibble_u16),
                                _mm512_and_si512(
                                    _mm512_slli_epi16(packed_u16, 4), high_nibble_u16));
                            weights = _mm512_mask_add_epi8(
                                weights, (__mmask64) b_ptr[b].qh[group], weights, high_bit_value);
                            __m512i dots = mul_sum_us8_pairs_acc_int32x16(
                                _mm512_setzero_si512(), weights, aq);
                            dots = _mm512_add_epi32(
                                dots, _mm512_shuffle_epi32(dots, (_MM_PERM_ENUM) _MM_SHUFFLE(2, 3, 0, 1)));
                            dots = _mm512_add_epi32(
                                dots, _mm512_shuffle_epi32(dots, (_MM_PERM_ENUM) _MM_SHUFFLE(1, 0, 3, 2)));
                            row_dots[row_group] = _mm_add_epi32(
                                row_dots[row_group], _mm512_castsi512_si128(
                                    _mm512_maskz_compress_epi32((__mmask16) 0x1111, dots)));
                        }
                    }

                    const __m128i bsum = _mm_set1_epi32(bsum_total);
                    for (int row_group = 0; row_group < 2; ++row_group) {
                        uint32_t packed_scales;
                        uint32_t packed_mins;
                        memcpy(
                            &packed_scales,
                            b_ptr[b].scales + ib32 * ncols_interleaved + row_group * 4,
                            sizeof(packed_scales));
                        memcpy(
                            &packed_mins,
                            b_ptr[b].mins + ib32 * ncols_interleaved + row_group * 4,
                            sizeof(packed_mins));
                        const __m128i scales = _mm_cvtepu8_epi32(
                            _mm_cvtsi32_si128((int) packed_scales));
                        const __m128i mins = _mm_cvtepu8_epi32(
                            _mm_cvtsi32_si128((int) packed_mins));
                        isum[row_group] = _mm_add_epi32(
                            isum[row_group], _mm_mullo_epi32(row_dots[row_group], scales));
                        imin[row_group] = _mm_add_epi32(
                            imin[row_group], _mm_mullo_epi32(mins, bsum));
                    }
                }

                const __m256i isum8 = _mm256_set_m128i(isum[1], isum[0]);
                const __m256i imin8 = _mm256_set_m128i(imin[1], imin[0]);
                const __m256 activation_scale = _mm256_set1_ps(a_ptr[b].d);
                const __m256 weight_scale = _mm256_mul_ps(
                    _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) b_ptr[b].d)),
                    activation_scale);
                const __m256 min_scale = _mm256_mul_ps(
                    _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) b_ptr[b].dmin)),
                    activation_scale);
                sumf = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(isum8), weight_scale, sumf);
                sumf = _mm256_sub_ps(
                    sumf, _mm256_mul_ps(_mm256_cvtepi32_ps(imin8), min_scale));
            }

            _mm256_storeu_ps(out + x * ncols_interleaved, sumf);
        }
    }
    return;
#endif
    ggml_gemv_q5_K_r8_q8_K_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemm_q4_K_r8_q8_K(
        int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy,
        int nr, int nc) {
#if defined(__AVX512F__) && defined(__AVX512BW__)
    constexpr int ncols_interleaved = 8;
    assert(n % QK_K == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    const int nb = n / QK_K;
    const block_q4_K_r8 * b_ptr_start = (const block_q4_K_r8 *) vx;
    const block_q8_Kx4 * a_ptr_start = (const block_q8_Kx4 *) vy;
    const __m512i low_nibble_u16 = _mm512_set1_epi16(0x000f);
    const __m512i high_nibble_u16 = _mm512_set1_epi16(0x0f00);

    for (int y = 0; y < nr / 4; ++y) {
        const block_q8_Kx4 * a_ptr = a_ptr_start + y * nb;

        for (int x = 0; x < nc / ncols_interleaved; ++x) {
            const block_q4_K_r8 * b_ptr = b_ptr_start + x * nb;
            __m256 sumf[4] = {
                _mm256_setzero_ps(),
                _mm256_setzero_ps(),
                _mm256_setzero_ps(),
                _mm256_setzero_ps(),
            };

            for (int b = 0; b < nb; ++b) {
                __m128i isum[4][2];
                __m128i imin[4][2];
                for (int m = 0; m < 4; ++m) {
                    isum[m][0] = _mm_setzero_si128();
                    isum[m][1] = _mm_setzero_si128();
                    imin[m][0] = _mm_setzero_si128();
                    imin[m][1] = _mm_setzero_si128();
                }

                for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
                    __m128i row_dots[4][2];
                    for (int m = 0; m < 4; ++m) {
                        row_dots[m][0] = _mm_setzero_si128();
                        row_dots[m][1] = _mm_setzero_si128();
                    }
                    int bsum_total[4] = {};

                    for (int half = 0; half < 2; ++half) {
                        const int sb = ib32 * 2 + half;
                        __m512i weights[2];
                        for (int row_group = 0; row_group < 2; ++row_group) {
                            const int group = sb * 2 + row_group;
                            const __m256i packed_low = _mm256_loadu_si256(
                                (const __m256i *) (b_ptr[b].ql + group * 32));
                            const __m512i packed_u16 = _mm512_cvtepu8_epi16(packed_low);
                            weights[row_group] = _mm512_or_si512(
                                _mm512_and_si512(packed_u16, low_nibble_u16),
                                _mm512_and_si512(
                                    _mm512_slli_epi16(packed_u16, 4), high_nibble_u16));
                        }

                        for (int m = 0; m < 4; ++m) {
                            uint64_t aq_lo;
                            uint64_t aq_hi;
                            memcpy(&aq_lo, a_ptr[b].qs + (sb * 2 + 0) * 32 + m * 8, sizeof(aq_lo));
                            memcpy(&aq_hi, a_ptr[b].qs + (sb * 2 + 1) * 32 + m * 8, sizeof(aq_hi));
                            const __m128i aq128 = _mm_set_epi64x((int64_t) aq_hi, (int64_t) aq_lo);
                            const __m512i aq = _mm512_broadcast_i32x4(aq128);
                            bsum_total[m] += a_ptr[b].bsums[(sb / 4) * 16 + m * 4 + sb % 4];

                            for (int row_group = 0; row_group < 2; ++row_group) {
                                __m512i dots = mul_sum_us8_pairs_acc_int32x16(
                                    _mm512_setzero_si512(), weights[row_group], aq);
                                dots = _mm512_add_epi32(
                                    dots, _mm512_shuffle_epi32(dots, (_MM_PERM_ENUM) _MM_SHUFFLE(2, 3, 0, 1)));
                                dots = _mm512_add_epi32(
                                    dots, _mm512_shuffle_epi32(dots, (_MM_PERM_ENUM) _MM_SHUFFLE(1, 0, 3, 2)));
                                const __m128i reduced = _mm512_castsi512_si128(
                                    _mm512_maskz_compress_epi32((__mmask16) 0x1111, dots));
                                row_dots[m][row_group] = _mm_add_epi32(
                                    row_dots[m][row_group], reduced);
                            }
                        }
                    }

                    for (int row_group = 0; row_group < 2; ++row_group) {
                        uint32_t packed_scales;
                        uint32_t packed_mins;
                        memcpy(&packed_scales,
                            b_ptr[b].scales + ib32 * ncols_interleaved + row_group * 4,
                            sizeof(packed_scales));
                        memcpy(&packed_mins,
                            b_ptr[b].mins + ib32 * ncols_interleaved + row_group * 4,
                            sizeof(packed_mins));
                        const __m128i scales = _mm_cvtepu8_epi32(
                            _mm_cvtsi32_si128((int) packed_scales));
                        const __m128i mins = _mm_cvtepu8_epi32(
                            _mm_cvtsi32_si128((int) packed_mins));

                        for (int m = 0; m < 4; ++m) {
                            isum[m][row_group] = _mm_add_epi32(
                                isum[m][row_group], _mm_mullo_epi32(row_dots[m][row_group], scales));
                            imin[m][row_group] = _mm_add_epi32(
                                imin[m][row_group], _mm_mullo_epi32(
                                    mins, _mm_set1_epi32(bsum_total[m])));
                        }
                    }
                }

                const __m256 weight_d = _mm256_cvtph_ps(
                    _mm_loadu_si128((const __m128i *) b_ptr[b].d));
                const __m256 weight_dmin = _mm256_cvtph_ps(
                    _mm_loadu_si128((const __m128i *) b_ptr[b].dmin));
                for (int m = 0; m < 4; ++m) {
                    const __m256i isum8 = _mm256_set_m128i(isum[m][1], isum[m][0]);
                    const __m256i imin8 = _mm256_set_m128i(imin[m][1], imin[m][0]);
                    const __m256 activation_scale = _mm256_set1_ps(a_ptr[b].d[m]);
                    sumf[m] = _mm256_fmadd_ps(
                        _mm256_cvtepi32_ps(isum8),
                        _mm256_mul_ps(weight_d, activation_scale), sumf[m]);
                    sumf[m] = _mm256_sub_ps(
                        sumf[m], _mm256_mul_ps(
                            _mm256_cvtepi32_ps(imin8),
                            _mm256_mul_ps(weight_dmin, activation_scale)));
                }
            }

            for (int m = 0; m < 4; ++m) {
                _mm256_storeu_ps(s + (y * 4 + m) * bs + x * ncols_interleaved, sumf[m]);
            }
        }
    }
    return;
#endif
    ggml_gemm_q4_K_r8_q8_K_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemm_q5_K_r8_q8_K(
        int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy,
        int nr, int nc) {
#if defined(__AVX512F__) && defined(__AVX512BW__)
    constexpr int ncols_interleaved = 8;
    assert(n % QK_K == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    const int nb = n / QK_K;
    const block_q5_K_r8 * b_ptr_start = (const block_q5_K_r8 *) vx;
    const block_q8_Kx4 * a_ptr_start = (const block_q8_Kx4 *) vy;
    const __m512i low_nibble_u16 = _mm512_set1_epi16(0x000f);
    const __m512i high_nibble_u16 = _mm512_set1_epi16(0x0f00);
    const __m512i high_bit_value = _mm512_set1_epi8(16);

    for (int y = 0; y < nr / 4; ++y) {
        const block_q8_Kx4 * a_ptr = a_ptr_start + y * nb;

        for (int x = 0; x < nc / ncols_interleaved; ++x) {
            const block_q5_K_r8 * b_ptr = b_ptr_start + x * nb;
            __m256 sumf[4] = {
                _mm256_setzero_ps(),
                _mm256_setzero_ps(),
                _mm256_setzero_ps(),
                _mm256_setzero_ps(),
            };

            for (int b = 0; b < nb; ++b) {
                __m128i isum[4][2];
                __m128i imin[4][2];
                for (int m = 0; m < 4; ++m) {
                    isum[m][0] = _mm_setzero_si128();
                    isum[m][1] = _mm_setzero_si128();
                    imin[m][0] = _mm_setzero_si128();
                    imin[m][1] = _mm_setzero_si128();
                }

                for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
                    __m128i row_dots[4][2];
                    for (int m = 0; m < 4; ++m) {
                        row_dots[m][0] = _mm_setzero_si128();
                        row_dots[m][1] = _mm_setzero_si128();
                    }
                    int bsum_total[4] = {};

                    for (int half = 0; half < 2; ++half) {
                        const int sb = ib32 * 2 + half;
                        __m512i weights[2];
                        for (int row_group = 0; row_group < 2; ++row_group) {
                            const int group = sb * 2 + row_group;
                            const __m256i packed_low = _mm256_loadu_si256(
                                (const __m256i *) (b_ptr[b].ql + group * 32));
                            const __m512i packed_u16 = _mm512_cvtepu8_epi16(packed_low);
                            weights[row_group] = _mm512_or_si512(
                                _mm512_and_si512(packed_u16, low_nibble_u16),
                                _mm512_and_si512(
                                    _mm512_slli_epi16(packed_u16, 4), high_nibble_u16));
                            weights[row_group] = _mm512_mask_add_epi8(
                                weights[row_group], (__mmask64) b_ptr[b].qh[group],
                                weights[row_group], high_bit_value);
                        }

                        for (int m = 0; m < 4; ++m) {
                            uint64_t aq_lo;
                            uint64_t aq_hi;
                            memcpy(&aq_lo, a_ptr[b].qs + (sb * 2 + 0) * 32 + m * 8, sizeof(aq_lo));
                            memcpy(&aq_hi, a_ptr[b].qs + (sb * 2 + 1) * 32 + m * 8, sizeof(aq_hi));
                            const __m128i aq128 = _mm_set_epi64x((int64_t) aq_hi, (int64_t) aq_lo);
                            const __m512i aq = _mm512_broadcast_i32x4(aq128);
                            bsum_total[m] += a_ptr[b].bsums[(sb / 4) * 16 + m * 4 + sb % 4];

                            for (int row_group = 0; row_group < 2; ++row_group) {
                                __m512i dots = mul_sum_us8_pairs_acc_int32x16(
                                    _mm512_setzero_si512(), weights[row_group], aq);
                                dots = _mm512_add_epi32(
                                    dots, _mm512_shuffle_epi32(dots, (_MM_PERM_ENUM) _MM_SHUFFLE(2, 3, 0, 1)));
                                dots = _mm512_add_epi32(
                                    dots, _mm512_shuffle_epi32(dots, (_MM_PERM_ENUM) _MM_SHUFFLE(1, 0, 3, 2)));
                                const __m128i reduced = _mm512_castsi512_si128(
                                    _mm512_maskz_compress_epi32((__mmask16) 0x1111, dots));
                                row_dots[m][row_group] = _mm_add_epi32(
                                    row_dots[m][row_group], reduced);
                            }
                        }
                    }

                    for (int row_group = 0; row_group < 2; ++row_group) {
                        uint32_t packed_scales;
                        uint32_t packed_mins;
                        memcpy(&packed_scales,
                            b_ptr[b].scales + ib32 * ncols_interleaved + row_group * 4,
                            sizeof(packed_scales));
                        memcpy(&packed_mins,
                            b_ptr[b].mins + ib32 * ncols_interleaved + row_group * 4,
                            sizeof(packed_mins));
                        const __m128i scales = _mm_cvtepu8_epi32(
                            _mm_cvtsi32_si128((int) packed_scales));
                        const __m128i mins = _mm_cvtepu8_epi32(
                            _mm_cvtsi32_si128((int) packed_mins));

                        for (int m = 0; m < 4; ++m) {
                            isum[m][row_group] = _mm_add_epi32(
                                isum[m][row_group], _mm_mullo_epi32(row_dots[m][row_group], scales));
                            imin[m][row_group] = _mm_add_epi32(
                                imin[m][row_group], _mm_mullo_epi32(
                                    mins, _mm_set1_epi32(bsum_total[m])));
                        }
                    }
                }

                const __m256 weight_d = _mm256_cvtph_ps(
                    _mm_loadu_si128((const __m128i *) b_ptr[b].d));
                const __m256 weight_dmin = _mm256_cvtph_ps(
                    _mm_loadu_si128((const __m128i *) b_ptr[b].dmin));
                for (int m = 0; m < 4; ++m) {
                    const __m256i isum8 = _mm256_set_m128i(isum[m][1], isum[m][0]);
                    const __m256i imin8 = _mm256_set_m128i(imin[m][1], imin[m][0]);
                    const __m256 activation_scale = _mm256_set1_ps(a_ptr[b].d[m]);
                    sumf[m] = _mm256_fmadd_ps(
                        _mm256_cvtepi32_ps(isum8),
                        _mm256_mul_ps(weight_d, activation_scale), sumf[m]);
                    sumf[m] = _mm256_sub_ps(
                        sumf[m], _mm256_mul_ps(
                            _mm256_cvtepi32_ps(imin8),
                            _mm256_mul_ps(weight_dmin, activation_scale)));
                }
            }

            for (int m = 0; m < 4; ++m) {
                _mm256_storeu_ps(s + (y * 4 + m) * bs + x * ncols_interleaved, sumf[m]);
            }
        }
    }
    return;
#endif
    ggml_gemm_q5_K_r8_q8_K_generic(n, s, bs, vx, vy, nr, nc);
}

static inline int q8_0_sum_i8x16(__m128i values) {
    const __m128i pair_sums = _mm_maddubs_epi16(_mm_set1_epi8(1), values);
    __m128i sums = _mm_madd_epi16(pair_sums, _mm_set1_epi16(1));
    sums = _mm_hadd_epi32(sums, sums);
    sums = _mm_hadd_epi32(sums, sums);
    return _mm_cvtsi128_si32(sums);
}

#if defined(__AVX512F__) && defined(__AVX512BW__)
static inline __m128i q8_0_vnni_dot_4rows(__m512i weights, __m128i activation) {
    const __m512i aq = _mm512_broadcast_i32x4(activation);
    __m512i dots = mul_sum_us8_pairs_acc_int32x16(
        _mm512_setzero_si512(), weights, aq);
    dots = _mm512_add_epi32(
        dots, _mm512_shuffle_epi32(
            dots, (_MM_PERM_ENUM) _MM_SHUFFLE(2, 3, 0, 1)));
    dots = _mm512_add_epi32(
        dots, _mm512_shuffle_epi32(
            dots, (_MM_PERM_ENUM) _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm512_castsi512_si128(
        _mm512_maskz_compress_epi32((__mmask16) 0x1111, dots));
}

template <int NR>
static void q8_0_vnni_gemv_rows(
        int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy,
        int nc) {
    static_assert(NR >= 1 && NR <= 3);
    constexpr int ncols_interleaved = 8;

    const int nb = n / QK8_0;
    const block_q8_0x8 * b_ptr_start = (const block_q8_0x8 *) vx;
    const block_q8_0 * a_ptr_start = (const block_q8_0 *) vy;

    for (int x = 0; x < nc / ncols_interleaved; ++x) {
        const block_q8_0x8 * b_ptr = b_ptr_start + x * nb;
        __m256 sumf[NR];
        for (int row = 0; row < NR; ++row) {
            sumf[row] = _mm256_setzero_ps();
        }

        for (int b = 0; b < nb; ++b) {
            __m128i row_dots[NR][2];
            int activation_sum[NR] = {};
            for (int row = 0; row < NR; ++row) {
                row_dots[row][0] = _mm_setzero_si128();
                row_dots[row][1] = _mm_setzero_si128();
            }

            for (int half = 0; half < 2; ++half) {
                const __m512i weights[2] = {
                    _mm512_loadu_si512((const void *) (b_ptr[b].qs + (half * 2 + 0) * 64)),
                    _mm512_loadu_si512((const void *) (b_ptr[b].qs + (half * 2 + 1) * 64)),
                };

                for (int row = 0; row < NR; ++row) {
                    const block_q8_0 * a_ptr = a_ptr_start + row * nb;
                    const __m128i aq = _mm_loadu_si128(
                        (const __m128i *) (a_ptr[b].qs + half * 16));
                    activation_sum[row] += q8_0_sum_i8x16(aq);
                    row_dots[row][0] = _mm_add_epi32(
                        row_dots[row][0], q8_0_vnni_dot_4rows(weights[0], aq));
                    row_dots[row][1] = _mm_add_epi32(
                        row_dots[row][1], q8_0_vnni_dot_4rows(weights[1], aq));
                }
            }

            const __m256 weight_scale = _mm256_cvtph_ps(
                _mm_loadu_si128((const __m128i *) b_ptr[b].d));
            for (int row = 0; row < NR; ++row) {
                const block_q8_0 * a_ptr = a_ptr_start + row * nb;
                __m256i isum = _mm256_set_m128i(row_dots[row][1], row_dots[row][0]);
                isum = _mm256_sub_epi32(
                    isum, _mm256_set1_epi32(activation_sum[row] * 128));
                const __m256 scale = _mm256_mul_ps(
                    weight_scale,
                    _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(a_ptr[b].d)));
                sumf[row] = _mm256_fmadd_ps(
                    _mm256_cvtepi32_ps(isum), scale, sumf[row]);
            }
        }

        for (int row = 0; row < NR; ++row) {
            _mm256_storeu_ps(s + row * bs + x * ncols_interleaved, sumf[row]);
        }
    }
}

// Reuse each quantized activation block (and, critically, its signed-byte
// sum) across several adjacent eight-column weight groups. The original
// x-major loop recomputes that sum for every output group even though it only
// depends on the activation row and K block. Keeping the K loop innermost for
// each output accumulator preserves its arithmetic order exactly.
template <int NR, int X_TILE>
static void q8_0_vnni_gemv_rows_x_tiled(
        int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy,
        int nc) {
    static_assert(NR >= 1 && NR <= 3);
    static_assert(X_TILE == 2 || X_TILE == 4 || X_TILE == 8);
    constexpr int ncols_interleaved = 8;

    const int nb = n / QK8_0;
    const int nx = nc / ncols_interleaved;
    const block_q8_0x8 * b_ptr_start = (const block_q8_0x8 *) vx;
    const block_q8_0 * a_ptr_start = (const block_q8_0 *) vy;

    for (int x = 0; x < nx; x += X_TILE) {
        const int nx_tile = nx - x < X_TILE ? nx - x : X_TILE;
        __m256 sumf[NR][X_TILE];
        for (int row = 0; row < NR; ++row) {
            for (int tx = 0; tx < nx_tile; ++tx) {
                sumf[row][tx] = _mm256_setzero_ps();
            }
        }

        for (int b = 0; b < nb; ++b) {
            __m128i activations[NR][2];
            int activation_sum[NR] = {};
            for (int row = 0; row < NR; ++row) {
                const block_q8_0 * a_ptr = a_ptr_start + row * nb;
                for (int half = 0; half < 2; ++half) {
                    activations[row][half] = _mm_loadu_si128(
                        (const __m128i *) (a_ptr[b].qs + half * 16));
                    activation_sum[row] += q8_0_sum_i8x16(activations[row][half]);
                }
            }

            for (int tx = 0; tx < nx_tile; ++tx) {
                const block_q8_0x8 * b_ptr = b_ptr_start + (x + tx) * nb;
                __m128i row_dots[NR][2];
                for (int row = 0; row < NR; ++row) {
                    row_dots[row][0] = _mm_setzero_si128();
                    row_dots[row][1] = _mm_setzero_si128();
                }

                for (int half = 0; half < 2; ++half) {
                    const __m512i weights[2] = {
                        _mm512_loadu_si512((const void *) (b_ptr[b].qs + (half * 2 + 0) * 64)),
                        _mm512_loadu_si512((const void *) (b_ptr[b].qs + (half * 2 + 1) * 64)),
                    };
                    for (int row = 0; row < NR; ++row) {
                        row_dots[row][0] = _mm_add_epi32(
                            row_dots[row][0], q8_0_vnni_dot_4rows(weights[0], activations[row][half]));
                        row_dots[row][1] = _mm_add_epi32(
                            row_dots[row][1], q8_0_vnni_dot_4rows(weights[1], activations[row][half]));
                    }
                }

                const __m256 weight_scale = _mm256_cvtph_ps(
                    _mm_loadu_si128((const __m128i *) b_ptr[b].d));
                for (int row = 0; row < NR; ++row) {
                    const block_q8_0 * a_ptr = a_ptr_start + row * nb;
                    __m256i isum = _mm256_set_m128i(row_dots[row][1], row_dots[row][0]);
                    isum = _mm256_sub_epi32(
                        isum, _mm256_set1_epi32(activation_sum[row] * 128));
                    const __m256 scale = _mm256_mul_ps(
                        weight_scale,
                        _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(a_ptr[b].d)));
                    sumf[row][tx] = _mm256_fmadd_ps(
                        _mm256_cvtepi32_ps(isum), scale, sumf[row][tx]);
                }
            }
        }

        for (int row = 0; row < NR; ++row) {
            for (int tx = 0; tx < nx_tile; ++tx) {
                _mm256_storeu_ps(
                    s + row * bs + (x + tx) * ncols_interleaved, sumf[row][tx]);
            }
        }
    }
}

template <int X_TILE>
static bool q8_0_vnni_gemv_rows_x_tiled_dispatch(
        int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy,
        int nr, int nc) {
    switch (nr) {
        case 1: q8_0_vnni_gemv_rows_x_tiled<1, X_TILE>(n, s, bs, vx, vy, nc); return true;
        case 2: q8_0_vnni_gemv_rows_x_tiled<2, X_TILE>(n, s, bs, vx, vy, nc); return true;
        case 3: q8_0_vnni_gemv_rows_x_tiled<3, X_TILE>(n, s, bs, vx, vy, nc); return true;
        default: return false;
    }
}

static int q8_0_vnni_x_tile() {
    static const int value = []() {
        const char * text = std::getenv("GGML_CPU_Q8_0_REPACK_X_TILE");
        if (text == nullptr) {
            return 1;
        }
        if (strcmp(text, "auto") == 0) {
            return 0;
        }
        const int parsed = atoi(text);
        return parsed == 2 || parsed == 4 || parsed == 8 ? parsed : 1;
    }();
    return value;
}

#endif

void ggml_gemv_q8_0_8x8_q8_0(
        int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy,
        int nr, int nc) {
#if defined(__AVX512F__) && defined(__AVX512BW__)
    static const bool trace_enabled = []() {
        const char * value = std::getenv("GGML_CPU_Q8_0_REPACK_TRACE");
        return value != nullptr && strcmp(value, "1") == 0;
    }();
    static std::atomic_flag traced = ATOMIC_FLAG_INIT;
    if (trace_enabled && !traced.test_and_set(std::memory_order_relaxed)) {
        const int configured_tile = q8_0_vnni_x_tile();
        const int active_tile = configured_tile == 0 ? (nr <= 2 ? 2 : 4) : configured_tile;
        fprintf(stderr, "q8_0_r8: AVX-512/VNNI GEMV n=%d nr=%d nc=%d x_tile=%d\n",
                n, nr, nc, active_tile);
    }

    assert(n % QK8_0 == 0);
    assert(nc % 8 == 0);
    const int configured_tile = q8_0_vnni_x_tile();
    const int active_tile = configured_tile == 0 ? (nr <= 2 ? 2 : 4) : configured_tile;
    switch (active_tile) {
        case 2: if (q8_0_vnni_gemv_rows_x_tiled_dispatch<2>(n, s, bs, vx, vy, nr, nc)) return; break;
        case 4: if (q8_0_vnni_gemv_rows_x_tiled_dispatch<4>(n, s, bs, vx, vy, nr, nc)) return; break;
        case 8: if (q8_0_vnni_gemv_rows_x_tiled_dispatch<8>(n, s, bs, vx, vy, nr, nc)) return; break;
        default: break;
    }
    switch (nr) {
        case 1: q8_0_vnni_gemv_rows<1>(n, s, bs, vx, vy, nc); return;
        case 2: q8_0_vnni_gemv_rows<2>(n, s, bs, vx, vy, nc); return;
        case 3: q8_0_vnni_gemv_rows<3>(n, s, bs, vx, vy, nc); return;
        default: break;
    }
#endif
    ggml_gemv_q8_0_8x8_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

template <int X_TILE>
static void q8_0_vnni_gemm_x_tiled(
        int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy,
        int nr, int nc) {
    static_assert(X_TILE == 2 || X_TILE == 4 || X_TILE == 8);
    constexpr int ncols_interleaved = 8;

    const int nb = n / QK8_0;
    const int nx = nc / ncols_interleaved;
    const block_q8_0x8 * b_ptr_start = (const block_q8_0x8 *) vx;
    const block_q8_0x4 * a_ptr_start = (const block_q8_0x4 *) vy;

    for (int y = 0; y < nr / 4; ++y) {
        const block_q8_0x4 * a_ptr = a_ptr_start + y * nb;
        for (int x = 0; x < nx; x += X_TILE) {
            const int nx_tile = nx - x < X_TILE ? nx - x : X_TILE;
            __m256 sumf[4][X_TILE];
            for (int row = 0; row < 4; ++row) {
                for (int tx = 0; tx < nx_tile; ++tx) {
                    sumf[row][tx] = _mm256_setzero_ps();
                }
            }

            for (int b = 0; b < nb; ++b) {
                __m128i activations[4][2];
                int activation_sum[4] = {};
                for (int half = 0; half < 2; ++half) {
                    for (int row = 0; row < 4; ++row) {
                        const int activation_offset = half * 64 + row * 8;
                        const __m128i aq_lo = _mm_loadl_epi64(
                            (const __m128i *) (a_ptr[b].qs + activation_offset));
                        const __m128i aq_hi = _mm_loadl_epi64(
                            (const __m128i *) (a_ptr[b].qs + activation_offset + 32));
                        activations[row][half] = _mm_unpacklo_epi64(aq_lo, aq_hi);
                        activation_sum[row] += q8_0_sum_i8x16(activations[row][half]);
                    }
                }

                for (int tx = 0; tx < nx_tile; ++tx) {
                    const block_q8_0x8 * b_ptr = b_ptr_start + (x + tx) * nb;
                    __m128i row_dots[4][2];
                    for (int row = 0; row < 4; ++row) {
                        row_dots[row][0] = _mm_setzero_si128();
                        row_dots[row][1] = _mm_setzero_si128();
                    }

                    for (int half = 0; half < 2; ++half) {
                        const __m512i weights[2] = {
                            _mm512_loadu_si512((const void *) (b_ptr[b].qs + (half * 2 + 0) * 64)),
                            _mm512_loadu_si512((const void *) (b_ptr[b].qs + (half * 2 + 1) * 64)),
                        };
                        for (int row = 0; row < 4; ++row) {
                            row_dots[row][0] = _mm_add_epi32(
                                row_dots[row][0], q8_0_vnni_dot_4rows(weights[0], activations[row][half]));
                            row_dots[row][1] = _mm_add_epi32(
                                row_dots[row][1], q8_0_vnni_dot_4rows(weights[1], activations[row][half]));
                        }
                    }

                    const __m256 weight_scale = _mm256_cvtph_ps(
                        _mm_loadu_si128((const __m128i *) b_ptr[b].d));
                    for (int row = 0; row < 4; ++row) {
                        __m256i isum = _mm256_set_m128i(row_dots[row][1], row_dots[row][0]);
                        isum = _mm256_sub_epi32(
                            isum, _mm256_set1_epi32(activation_sum[row] * 128));
                        const __m256 scale = _mm256_mul_ps(
                            weight_scale,
                            _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(a_ptr[b].d[row])));
                        sumf[row][tx] = _mm256_fmadd_ps(
                            _mm256_cvtepi32_ps(isum), scale, sumf[row][tx]);
                    }
                }
            }

            for (int row = 0; row < 4; ++row) {
                for (int tx = 0; tx < nx_tile; ++tx) {
                    _mm256_storeu_ps(
                        s + (y * 4 + row) * bs + (x + tx) * ncols_interleaved,
                        sumf[row][tx]);
                }
            }
        }
    }
}

void ggml_gemm_q8_0_8x8_q8_0(
        int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy,
        int nr, int nc) {
#if defined(__AVX512F__) && defined(__AVX512BW__)
    constexpr int ncols_interleaved = 8;
    assert(n % QK8_0 == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    const int configured_tile = q8_0_vnni_x_tile();
    switch (configured_tile == 0 ? 8 : configured_tile) {
        case 2: q8_0_vnni_gemm_x_tiled<2>(n, s, bs, vx, vy, nr, nc); return;
        case 4: q8_0_vnni_gemm_x_tiled<4>(n, s, bs, vx, vy, nr, nc); return;
        case 8: q8_0_vnni_gemm_x_tiled<8>(n, s, bs, vx, vy, nr, nc); return;
        default: break;
    }

    const int nb = n / QK8_0;
    const block_q8_0x8 * b_ptr_start = (const block_q8_0x8 *) vx;
    const block_q8_0x4 * a_ptr_start = (const block_q8_0x4 *) vy;

    for (int y = 0; y < nr / 4; ++y) {
        const block_q8_0x4 * a_ptr = a_ptr_start + y * nb;
        for (int x = 0; x < nc / ncols_interleaved; ++x) {
            const block_q8_0x8 * b_ptr = b_ptr_start + x * nb;
            __m256 sumf[4] = {
                _mm256_setzero_ps(), _mm256_setzero_ps(),
                _mm256_setzero_ps(), _mm256_setzero_ps(),
            };

            for (int b = 0; b < nb; ++b) {
                __m128i row_dots[4][2];
                int activation_sum[4] = {};
                for (int row = 0; row < 4; ++row) {
                    row_dots[row][0] = _mm_setzero_si128();
                    row_dots[row][1] = _mm_setzero_si128();
                }

                for (int half = 0; half < 2; ++half) {
                    const __m512i weights[2] = {
                        _mm512_loadu_si512((const void *) (b_ptr[b].qs + (half * 2 + 0) * 64)),
                        _mm512_loadu_si512((const void *) (b_ptr[b].qs + (half * 2 + 1) * 64)),
                    };

                    for (int row = 0; row < 4; ++row) {
                        const int activation_offset = half * 64 + row * 8;
                        const __m128i aq_lo = _mm_loadl_epi64(
                            (const __m128i *) (a_ptr[b].qs + activation_offset));
                        const __m128i aq_hi = _mm_loadl_epi64(
                            (const __m128i *) (a_ptr[b].qs + activation_offset + 32));
                        const __m128i aq = _mm_unpacklo_epi64(aq_lo, aq_hi);
                        activation_sum[row] += q8_0_sum_i8x16(aq);
                        row_dots[row][0] = _mm_add_epi32(
                            row_dots[row][0], q8_0_vnni_dot_4rows(weights[0], aq));
                        row_dots[row][1] = _mm_add_epi32(
                            row_dots[row][1], q8_0_vnni_dot_4rows(weights[1], aq));
                    }
                }

                const __m256 weight_scale = _mm256_cvtph_ps(
                    _mm_loadu_si128((const __m128i *) b_ptr[b].d));
                for (int row = 0; row < 4; ++row) {
                    __m256i isum = _mm256_set_m128i(row_dots[row][1], row_dots[row][0]);
                    isum = _mm256_sub_epi32(
                        isum, _mm256_set1_epi32(activation_sum[row] * 128));
                    const __m256 scale = _mm256_mul_ps(
                        weight_scale,
                        _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(a_ptr[b].d[row])));
                    sumf[row] = _mm256_fmadd_ps(
                        _mm256_cvtepi32_ps(isum), scale, sumf[row]);
                }
            }

            for (int row = 0; row < 4; ++row) {
                _mm256_storeu_ps(
                    s + (y * 4 + row) * bs + x * ncols_interleaved,
                    sumf[row]);
            }
        }
    }
    return;
#endif
    ggml_gemm_q8_0_8x8_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemm_q4_0_8x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
#if defined(__AVX2__) || defined(__AVX512F__)
    {
        // Lookup table to convert signed nibbles to signed bytes
        __m256i signextendlut = _mm256_castsi128_si256(_mm_set_epi8(-1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0));
        signextendlut = _mm256_permute2f128_si256(signextendlut, signextendlut, 0);

        gemm_q4_b32_8x8_q8_0_lut_avx<block_q4_0x8>(n, s, bs, vx, vy, nr, nc, signextendlut);

        return;
    }
#endif // defined(__AVX2__) || defined(__AVX512F__)

    ggml_gemm_q4_0_8x8_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemm_q4_K_8x8_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    assert (n % qk == 0);
    assert (nr % 4 == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

#if defined(__AVX2__) || defined(__AVX512F__)
    const block_q4_Kx8 * b_ptr_start = (const block_q4_Kx8 * ) vx;
    const block_q8_Kx4 * a_ptr_start = (const block_q8_Kx4 * ) vy;
    int64_t b_nb = n / QK_K;
    int64_t y = 0;

    // Mask to mask out nibbles from packed bytes
    const __m256i m4b = _mm256_set1_epi8(0x0F);
    // Permute mask used for easier vector processing at later stages
    __m256i requiredOrder = _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4);
    int64_t xstart = 0;
    int anr = nr - nr % 16;; // Used to align nr with boundary of 16
#if defined(__AVX512BW__) && defined(__AVX512DQ__)
    int anc = nc - nc % 16; // Used to align nc with boundary of 16
    // Mask to mask out nibbles from packed bytes expanded to 512 bit length
    const __m512i m4bexpanded = _mm512_set1_epi8(0x0F);
    //Take group of four block_q8_Kx4 structures at each pass of the loop and perform dot product operation
    for (; y < anr / 4; y += 4) {

        const block_q8_Kx4 * a_ptrs[4];

        a_ptrs[0] = a_ptr_start + (y * nb);
        for (int i = 0; i < 3; ++i) {
            a_ptrs[i + 1] = a_ptrs[i] + nb;
        }

        // Take group of eight block_q4_kx8 structures at each pass of the loop and perform dot product operation
        for (int64_t x = 0; x < anc / 8; x += 2) {

            const block_q4_Kx8 * b_ptr_0 = b_ptr_start + ((x) * b_nb);
            const block_q4_Kx8 * b_ptr_1 = b_ptr_start + ((x + 1) * b_nb);

            // Master FP accumulators
            __m512 acc_rows[16];
            for (int i = 0; i < 16; i++) {
                acc_rows[i] = _mm512_setzero_ps();
            }

            __m512 acc_min_rows[16];
            for (int i = 0; i < 16; i++) {
                acc_min_rows[i] = _mm512_setzero_ps();
            }

            // For super block
            for (int64_t b = 0; b < nb; b++) {
                // Scale values - Load the sixteen scale values from two block_q4_kx8 structures
                const __m512 col_scale_f32 = GGML_F32Cx8x2_LOAD(b_ptr_0[b].d, b_ptr_1[b].d);

                // dmin values - Load the sixteen dmin values from two block_q4_kx8 structures
                const __m512 col_dmin_f32 = GGML_F32Cx8x2_LOAD(b_ptr_0[b].dmin, b_ptr_1[b].dmin);

                // Loop to iterate over the eight sub blocks of a super block - two sub blocks are processed per iteration
                for (int sb = 0; sb < QK_K / 64; sb++) {

                    const __m256i rhs_raw_mat_0123_0 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + sb * 256));
                    const __m256i rhs_raw_mat_4567_0 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 32 + sb * 256));
                    const __m256i rhs_raw_mat_0123_1 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 64 + sb * 256));
                    const __m256i rhs_raw_mat_4567_1 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 96 + sb * 256));
                    const __m256i rhs_raw_mat_0123_2 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 128 + sb * 256));
                    const __m256i rhs_raw_mat_4567_2 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 160 + sb * 256));
                    const __m256i rhs_raw_mat_0123_3 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 192 + sb * 256));
                    const __m256i rhs_raw_mat_4567_3 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 224 + sb * 256));

                    const __m256i rhs_raw_mat_89AB_0 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + sb * 256));
                    const __m256i rhs_raw_mat_CDEF_0 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 32 + sb * 256));
                    const __m256i rhs_raw_mat_89AB_1 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 64 + sb * 256));
                    const __m256i rhs_raw_mat_CDEF_1 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 96 + sb * 256));
                    const __m256i rhs_raw_mat_89AB_2 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 128 + sb * 256));
                    const __m256i rhs_raw_mat_CDEF_2 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 160 + sb * 256));
                    const __m256i rhs_raw_mat_89AB_3 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 192 + sb * 256));
                    const __m256i rhs_raw_mat_CDEF_3 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 224 + sb * 256));

                    const __m256i rhs_raw_mat_0145_0 = _mm256_blend_epi32(rhs_raw_mat_0123_0, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_0, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_0, requiredOrder), rhs_raw_mat_4567_0, 240);
                    const __m256i rhs_raw_mat_0145_1 = _mm256_blend_epi32(rhs_raw_mat_0123_1, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_1, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_1, requiredOrder), rhs_raw_mat_4567_1, 240);
                    const __m256i rhs_raw_mat_0145_2 = _mm256_blend_epi32(rhs_raw_mat_0123_2, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_2, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_2 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_2, requiredOrder), rhs_raw_mat_4567_2, 240);
                    const __m256i rhs_raw_mat_0145_3 = _mm256_blend_epi32(rhs_raw_mat_0123_3, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_3, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_3 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_3, requiredOrder), rhs_raw_mat_4567_3, 240);

                    const __m256i rhs_raw_mat_89CD_0 = _mm256_blend_epi32(rhs_raw_mat_89AB_0, _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_0, requiredOrder), 240);
                    const __m256i rhs_raw_mat_ABEF_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_0, requiredOrder), rhs_raw_mat_CDEF_0, 240);
                    const __m256i rhs_raw_mat_89CD_1 = _mm256_blend_epi32(rhs_raw_mat_89AB_1, _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_1, requiredOrder), 240);
                    const __m256i rhs_raw_mat_ABEF_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_1, requiredOrder), rhs_raw_mat_CDEF_1, 240);
                    const __m256i rhs_raw_mat_89CD_2 = _mm256_blend_epi32(rhs_raw_mat_89AB_2, _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_2, requiredOrder), 240);
                    const __m256i rhs_raw_mat_ABEF_2 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_2, requiredOrder), rhs_raw_mat_CDEF_2, 240);
                    const __m256i rhs_raw_mat_89CD_3 = _mm256_blend_epi32(rhs_raw_mat_89AB_3, _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_3, requiredOrder), 240);
                    const __m256i rhs_raw_mat_ABEF_3 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_3, requiredOrder), rhs_raw_mat_CDEF_3, 240);

                    const __m512i rhs_raw_mat_014589CD_0 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_0145_0), rhs_raw_mat_89CD_0, 1);
                    const __m512i rhs_raw_mat_2367ABEF_0 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_2367_0), rhs_raw_mat_ABEF_0, 1);
                    const __m512i rhs_raw_mat_014589CD_1 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_0145_1), rhs_raw_mat_89CD_1, 1);
                    const __m512i rhs_raw_mat_2367ABEF_1 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_2367_1), rhs_raw_mat_ABEF_1, 1);

                    const __m512i rhs_raw_mat_014589CD_2 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_0145_2), rhs_raw_mat_89CD_2, 1);
                    const __m512i rhs_raw_mat_2367ABEF_2 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_2367_2), rhs_raw_mat_ABEF_2, 1);
                    const __m512i rhs_raw_mat_014589CD_3 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_0145_3), rhs_raw_mat_89CD_3, 1);
                    const __m512i rhs_raw_mat_2367ABEF_3 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_2367_3), rhs_raw_mat_ABEF_3, 1);

                    //4-bit -> 8-bit
                    const __m512i rhs_mat_014589CD_00 = _mm512_and_si512(rhs_raw_mat_014589CD_0, m4bexpanded); //B00(0-7) B01(0-7) B04(0-7) B05(0-7) B08(0-7) B09(0-7) B0C(0-7) B0D(0-7)
                    const __m512i rhs_mat_2367ABEF_00 = _mm512_and_si512(rhs_raw_mat_2367ABEF_0, m4bexpanded); //B02(0-7) B03(0-7) B06(0-7) B07(0-7) B0A(0-7) B0B(0-7) B0E(0-7) B0F(0-7)
                    const __m512i rhs_mat_014589CD_01 = _mm512_and_si512(rhs_raw_mat_014589CD_1, m4bexpanded); //B00(8-15) B01(8-15) B04(8-15) B05(8-15) B08(8-15) B09(8-15) B0C(8-15) B0D(8-15)
                    const __m512i rhs_mat_2367ABEF_01 = _mm512_and_si512(rhs_raw_mat_2367ABEF_1, m4bexpanded); //B02(8-15) B03(8-15) B06(8-15) B07(8-15) B0A(8-15) B0B(8-15) B0E(8-15) B0F(8-15)

                    const __m512i rhs_mat_014589CD_02 = _mm512_and_si512(rhs_raw_mat_014589CD_2, m4bexpanded); //B00(16-23) B01(16-23) B04(16-23) B05(16-23) B08(16-23) B09(16-23) B0C(16-23) B0D(16-23)
                    const __m512i rhs_mat_2367ABEF_02 = _mm512_and_si512(rhs_raw_mat_2367ABEF_2, m4bexpanded); //B02(16-23) B03(16-23) B06(16-23) B07(16-23) B0A(16-23) B0B(16-23) B0E(16-23) B0F(16-23)
                    const __m512i rhs_mat_014589CD_03 = _mm512_and_si512(rhs_raw_mat_014589CD_3, m4bexpanded); //B00(24-31) B01(24-31) B04(24-31) B05(24-31) B08(24-31) B09(24-31) B0C(24-31) B0D(24-31)
                    const __m512i rhs_mat_2367ABEF_03 = _mm512_and_si512(rhs_raw_mat_2367ABEF_3, m4bexpanded); //B02(24-31) B03(24-31) B06(24-31) B07(24-31) B0A(24-31) B0B(24-31) B0E(24-31) B0F(24-31)

                    const __m512i rhs_mat_014589CD_10 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_0, 4), m4bexpanded); //B10(0-7) B11(0-7) B14(0-7) B15(0-7) B18(0-7) B19(0-7) B1C(0-7) B1D(0-7)
                    const __m512i rhs_mat_2367ABEF_10 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_0, 4), m4bexpanded); //B12(0-7) B13(0-7) B16(0-7) B17(0-7) B1A(0-7) B1B(0-7) B1E(0-7) B1F(0-7)
                    const __m512i rhs_mat_014589CD_11 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_1, 4), m4bexpanded); //B10(8-15) B11(8-15) B14(8-15) B15(8-15) B18(8-15) B19(8-15) B1C(8-15) B1D(8-15)
                    const __m512i rhs_mat_2367ABEF_11 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_1, 4), m4bexpanded); //B12(8-15) B13(8-15) B16(8-15) B17(8-15) B1A(8-15) B1B(8-15) B1E(8-15) B1F(8-15)

                    const __m512i rhs_mat_014589CD_12 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_2, 4), m4bexpanded); //B10(16-23) B11(16-23) B14(16-23) B15(16-23) B18(16-23) B19(16-23) B1C(16-23) B1D(16-23)
                    const __m512i rhs_mat_2367ABEF_12 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_2, 4), m4bexpanded); //B12(16-23) B13(16-23) B16(16-23) B17(16-23) B1A(16-23) B1B(16-23) B1E(16-23) B1F(16-23)
                    const __m512i rhs_mat_014589CD_13 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_3, 4), m4bexpanded); //B10(24-31) B11(24-31) B14(24-31) B15(24-31) B18(24-31) B19(24-31) B1C(24-31) B1D(24-31)
                    const __m512i rhs_mat_2367ABEF_13 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_3, 4), m4bexpanded); //B12(24-31) B13(24-31) B16(24-31) B17(24-31) B1A(24-31) B1B(24-31) B1E(24-31) B1F(24-31)

                    // Shuffle pattern one - right side input
                    const __m512i rhs_mat_014589CD_00_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_00, (_MM_PERM_ENUM)136); //B00(0-3) B01(0-3) B00(0-3) B01(0-3) B04(0-3) B05(0-3) B04(0-3) B05(0-3) B08(0-3) B09(0-3) B08(0-3) B09(0-3) B0C(0-3) B0D(0-3) B0C(0-3) B0D(0-3)
                    const __m512i rhs_mat_2367ABEF_00_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_00, (_MM_PERM_ENUM)136); //B02(0-3) B03(0-3) B02(0-3) B03(0-3) B06(0-3) B07(0-3) B06(0-3) B07(0-3) B0A(0-3) B0B(0-3) B0A(0-3) B0B(0-3) B0E(0-3) B0F(0-3) B0E(0-3) B0F(0-3)
                    const __m512i rhs_mat_014589CD_01_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_01, (_MM_PERM_ENUM)136); //B00(8-11) B01(8-11) B00(8-11) B01(8-11) B04(8-11) B05(8-11) B04(8-11) B05(8-11) B08(8-11) B09(8-11) B08(8-11) B09(8-11) B0C(8-11) B0D(8-11) B0C(8-11) B0D(8-11)
                    const __m512i rhs_mat_2367ABEF_01_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_01, (_MM_PERM_ENUM)136); //B02(8-11) B03(8-11) B02(8-11) B03(8-11) B06(8-11) B07(8-11) B06(8-11) B07(8-11) B0A(8-11) B0B(8-11) B0A(8-11) B0B(8-11) B0E(8-11) B0F(8-11) B0E(8-11) B0F(8-11)
                    const __m512i rhs_mat_014589CD_02_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_02, (_MM_PERM_ENUM)136); //B00(16-19) B01(16-19) B00(16-19) B01(16-19) B04(16-19) B05(16-19) B04(16-19) B05(16-19) B08(16-19) B09(16-19) B08(16-19) B09(16-19) B0C(16-19) B0D(16-19) B0C(16-19) B0D(16-19)
                    const __m512i rhs_mat_2367ABEF_02_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_02, (_MM_PERM_ENUM)136); //B02(16-19) B03(16-19) B02(16-19) B03(16-19) B06(16-19) B07(16-19) B06(16-19) B07(16-19) B0A(16-19) B0B(16-19) B0A(16-19) B0B(16-19) B0E(16-19) B0F(16-19) B0E(16-19) B0F(16-19)
                    const __m512i rhs_mat_014589CD_03_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_03, (_MM_PERM_ENUM)136); //B00(24-27) B01(24-27) B00(24-27) B01(24-27) B04(24-27) B05(24-27) B04(24-27) B05(24-27) B08(24-27) B09(24-27) B08(24-27) B09(24-27) B0C(24-27) B0D(24-27) B0C(24-27) B0D(24-27)
                    const __m512i rhs_mat_2367ABEF_03_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_03, (_MM_PERM_ENUM)136); //B02(24-27) B03(24-27) B02(24-27) B03(24-27) B06(24-27) B07(24-27) B06(24-27) B07(24-27) B0A(24-27) B0B(24-27) B0A(24-27) B0B(24-27) B0E(24-27) B0F(24-27) B0E(24-27) B0F(24-27)

                    const __m512i rhs_mat_014589CD_10_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_10, (_MM_PERM_ENUM)136); //B10(0-3) B11(0-3) B10(0-3) B11(0-3) B14(0-3) B15(0-3) B14(0-3) B15(0-3) B18(0-3) B19(0-3) B18(0-3) B19(0-3) B1C(0-3) B1D(0-3) B1C(0-3) B1D(0-3)
                    const __m512i rhs_mat_2367ABEF_10_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_10, (_MM_PERM_ENUM)136); //B12(0-3) B13(0-3) B12(0-3) B13(0-3) B16(0-3) B17(0-3) B16(0-3) B17(0-3) B1A(0-3) B1B(0-3) B1A(0-3) B1B(0-3) B1E(0-3) B1F(0-3) B1E(0-3) B1F(0-3)
                    const __m512i rhs_mat_014589CD_11_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_11, (_MM_PERM_ENUM)136); //B10(8-11) B11(8-11) B10(8-11) B11(8-11) B14(8-11) B15(8-11) B14(8-11) B15(8-11) B18(8-11) B19(8-11) B18(8-11) B19(8-11) B1C(8-11) B1D(8-11) B1C(8-11) B1D(8-11)
                    const __m512i rhs_mat_2367ABEF_11_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_11, (_MM_PERM_ENUM)136); //B12(8-11) B13(8-11) B12(8-11) B13(8-11) B16(8-11) B17(8-11) B16(8-11) B17(8-11) B1A(8-11) B1B(8-11) B1A(8-11) B1B(8-11) B1E(8-11) B1F(8-11) B1E(8-11) B1F(8-11)
                    const __m512i rhs_mat_014589CD_12_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_12, (_MM_PERM_ENUM)136); //B10(16-19) B11(16-19) B10(16-19) B11(16-19) B14(16-19) B15(16-19) B14(16-19) B15(16-19) B18(16-19) B19(16-19) B18(16-19) B19(16-19) B1C(16-19) B1D(16-19) B1C(16-19) B1D(16-19)
                    const __m512i rhs_mat_2367ABEF_12_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_12, (_MM_PERM_ENUM)136); //B12(16-19) B13(16-19) B12(16-19) B13(16-19) B16(16-19) B17(16-19) B16(16-19) B17(16-19) B1A(16-19) B1B(16-19) B1A(16-19) B1B(16-19) B1E(16-19) B1F(16-19) B1E(16-19) B1F(16-19)
                    const __m512i rhs_mat_014589CD_13_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_13, (_MM_PERM_ENUM)136); //B10(24-27) B11(24-27) B10(24-27) B11(24-27) B14(24-27) B15(24-27) B14(24-27) B15(24-27) B18(24-27) B19(24-27) B18(24-27) B19(24-27) B1C(24-27) B1D(24-27) B1C(24-27) B1D(24-27)
                    const __m512i rhs_mat_2367ABEF_13_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_13, (_MM_PERM_ENUM)136); //B12(24-27) B13(24-27) B12(24-27) B13(24-27) B16(24-27) B17(24-27) B16(24-27) B17(24-27) B1A(24-27) B1B(24-27) B1A(24-27) B1B(24-27) B1E(24-27) B1F(24-27) B1E(24-27) B1F(24-27)

                    // Shuffle pattern two - right side input
                    const __m512i rhs_mat_014589CD_00_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_00, (_MM_PERM_ENUM)221); //B00(4-7) B01(4-7) B00(4-7) B01(4-7) B04(4-7) B05(4-7) B04(4-7) B05(4-7) B08(4-7) B09(4-7) B08(4-7) B09(4-7) B0C(4-7) B0D(4-7) B0C(4-7) B0D(4-7)
                    const __m512i rhs_mat_2367ABEF_00_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_00, (_MM_PERM_ENUM)221); //B02(4-7) B03(4-7) B02(4-7) B03(4-7) B06(4-7) B07(4-7) B06(4-7) B07(4-7) B0A(4-7) B0B(4-7) B0A(4-7) B0B(4-7) B0E(4-7) B0F(4-7) B0E(4-7) B0F(4-7)
                    const __m512i rhs_mat_014589CD_01_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_01, (_MM_PERM_ENUM)221); //B00(12-15) B01(12-15) B00(12-15) B01(12-15) B04(12-15) B05(12-15) B04(12-15) B05(12-15) B08(12-15) B09(12-15) B08(12-15) B09(12-15) B0C(12-15) B0D(12-15) B0C(12-15) B0D(12-15)
                    const __m512i rhs_mat_2367ABEF_01_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_01, (_MM_PERM_ENUM)221); //B02(12-15) B03(12-15) B02(12-15) B03(12-15) B06(12-15) B07(12-15) B06(12-15) B07(12-15) B0A(12-15) B0B(12-15) B0A(12-15) B0B(12-15) B0E(12-15) B0F(12-15) B0E(12-15) B0F(12-15)
                    const __m512i rhs_mat_014589CD_02_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_02, (_MM_PERM_ENUM)221); //B00(20-23) B01(20-23) B00(20-23) B01(20-23) B04(20-23) B05(20-23) B04(20-23) B05(20-23) B08(20-23) B09(20-23) B08(20-23) B09(20-23) B0C(20-23) B0D(20-23) B0C(20-23) B0D(20-23)
                    const __m512i rhs_mat_2367ABEF_02_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_02, (_MM_PERM_ENUM)221); //B02(20-23) B03(20-23) B02(20-23) B03(20-23) B06(20-23) B07(20-23) B06(20-23) B07(20-23) B0A(20-23) B0B(20-23) B0A(20-23) B0B(20-23) B0E(20-23) B0F(20-23) B0E(20-23) B0F(20-23)
                    const __m512i rhs_mat_014589CD_03_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_03, (_MM_PERM_ENUM)221); //B00(28-31) B01(28-31) B00(28-31) B01(28-31) B04(28-31) B05(28-31) B04(28-31) B05(28-31) B08(28-31) B09(28-31) B08(28-31) B09(28-31) B0C(28-31) B0D(28-31) B0C(28-31) 0BD(28-31)
                    const __m512i rhs_mat_2367ABEF_03_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_03, (_MM_PERM_ENUM)221); //B02(28-31) B03(28-31) B02(28-31) B03(28-31) B06(28-31) B07(28-31) B06(28-31) B07(28-31) B0A(28-31) B0B(28-31) B0A(28-31) B0B(28-31) B0E(28-31) B0F(28-31) B0E(28-31) B0F(28-31)

                    const __m512i rhs_mat_014589CD_10_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_10, (_MM_PERM_ENUM)221); //B10(4-7) B11(4-7) B10(4-7) B11(4-7) B14(4-7) B15(4-7) B14(4-7) B15(4-7) B18(4-7) B19(4-7) B18(4-7) B19(4-7) B1C(4-7) B1D(4-7) B1C(4-7) B1D(4-7)
                    const __m512i rhs_mat_2367ABEF_10_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_10, (_MM_PERM_ENUM)221); //B12(4-7) B13(4-7) B12(4-7) B13(4-7) B16(4-7) B17(4-7) B16(4-7) B17(4-7) B1A(4-7) B1B(4-7) B1A(4-7) B1B(4-7) B1E(4-7) B1F(4-7) B1E(4-7) B1F(4-7)
                    const __m512i rhs_mat_014589CD_11_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_11, (_MM_PERM_ENUM)221); //B10(12-15) B11(12-15) B10(12-15) B11(12-15) B14(12-15) B15(12-15) B14(12-15) B15(12-15) B18(12-15) B19(12-15) B18(12-15) B19(12-15) B1C(12-15) B1D(12-15) B1C(12-15) B1D(12-15)
                    const __m512i rhs_mat_2367ABEF_11_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_11, (_MM_PERM_ENUM)221); //B12(12-15) B13(12-15) B12(12-15) B13(12-15) B16(12-15) B17(12-15) B16(12-15) B17(12-15) B1A(12-15) B1B(12-15) B1A(12-15) B1B(12-15) B1E(12-15) B1F(12-15) B1E(12-15) B1F(12-15)
                    const __m512i rhs_mat_014589CD_12_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_12, (_MM_PERM_ENUM)221); //B10(20-23) B11(20-23) B10(20-23) B11(20-23) B14(20-23) B15(20-23) B14(20-23) B15(20-23) B18(20-23) B19(20-23) B18(20-23) B19(20-23) B1C(20-23) B1D(20-23) B1C(20-23) B1D(20-23)
                    const __m512i rhs_mat_2367ABEF_12_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_12, (_MM_PERM_ENUM)221); //B12(20-23) B13(20-23) B12(20-23) B13(20-23) B16(20-23) B17(20-23) B16(20-23) B17(20-23) B1A(20-23) B1B(20-23) B1A(20-23) B1B(20-23) B1E(20-23) B1F(20-23) B1E(20-23) B1F(20-23)
                    const __m512i rhs_mat_014589CD_13_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_13, (_MM_PERM_ENUM)221); //B10(28-31) B11(28-31) B10(28-31) B11(28-31) B14(28-31) B15(28-31) B14(28-31) B15(28-31) B18(28-31) B19(28-31) B18(28-31) B19(28-31) B1C(28-31) B1D(28-31) B1C(28-31) B1D(28-31)
                    const __m512i rhs_mat_2367ABEF_13_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_13, (_MM_PERM_ENUM)221); //B12(28-31) B13(28-31) B12(28-31) B13(28-31) B16(28-31) B17(28-31) B16(28-31) B17(28-31) B1A(28-31) B1B(28-31) B1A(28-31) B1B(28-31) B1E(28-31) B1F(28-31) B1E(28-31) B1F(28-31)

                    uint32_t utmp_00[4], utmp_01[4], utmp_10[4], utmp_11[4];

                    // Scales and Mins of corresponding sub blocks from different Q4_K structures are stored together
                    // The below block is for eg to extract first sub block's scales and mins from different Q4_K structures for the sb loop
                    memcpy(utmp_00, b_ptr_0[b].scales + 24 * sb, 12);
                    utmp_00[3] = ((utmp_00[2] >> 4) & kmask2) | (((utmp_00[1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_00 = utmp_00[1] & kmask1;
                    utmp_00[1] = (utmp_00[2] & kmask2) | (((utmp_00[0] >> 6) & kmask3) << 4);
                    utmp_00[2] = uaux_00;
                    utmp_00[0] &= kmask1;

                    // The below block is for eg to extract second sub block's scales and mins from different Q4_K structures for the sb loop
                    memcpy(utmp_01, b_ptr_0[b].scales + 12 + sb * 24, 12);
                    utmp_01[3] = ((utmp_01[2] >> 4) & kmask2) | (((utmp_01[1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_01 = utmp_01[1] & kmask1;
                    utmp_01[1] = (utmp_01[2] & kmask2) | (((utmp_01[0] >> 6) & kmask3) << 4);
                    utmp_01[2] = uaux_01;
                    utmp_01[0] &= kmask1;

                    memcpy(utmp_10, b_ptr_1[b].scales + sb * 24, 12);
                    utmp_10[3] = ((utmp_10[2] >> 4) & kmask2) | (((utmp_10[1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_10 = utmp_10[1] & kmask1;
                    utmp_10[1] = (utmp_10[2] & kmask2) | (((utmp_10[0] >> 6) & kmask3) << 4);
                    utmp_10[2] = uaux_10;
                    utmp_10[0] &= kmask1;

                    // The below block is for eg to extract second sub block's scales and mins from different Q4_K structures for the sb loop
                    memcpy(utmp_11, b_ptr_1[b].scales + 12 + sb * 24, 12);
                    utmp_11[3] = ((utmp_11[2] >> 4) & kmask2) | (((utmp_11[1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_11 = utmp_11[1] & kmask1;
                    utmp_11[1] = (utmp_11[2] & kmask2) | (((utmp_11[0] >> 6) & kmask3) << 4);
                    utmp_11[2] = uaux_11;
                    utmp_11[0] &= kmask1;

                    // Scales of first sub block in the sb loop
                    const __m256i mins_and_scales_0 = _mm256_set_epi32(utmp_10[3], utmp_10[2], utmp_10[1], utmp_10[0], utmp_00[3], utmp_00[2], utmp_00[1], utmp_00[0]);
                    const __m512i scales_0 = _mm512_cvtepu8_epi16(_mm256_unpacklo_epi8(mins_and_scales_0, mins_and_scales_0));

                    // Scales of second sub block in the sb loop
                    const __m256i mins_and_scales_1 = _mm256_set_epi32(utmp_11[3], utmp_11[2], utmp_11[1], utmp_11[0], utmp_01[3], utmp_01[2], utmp_01[1], utmp_01[0]);
                    const __m512i scales_1 = _mm512_cvtepu8_epi16(_mm256_unpacklo_epi8(mins_and_scales_1, mins_and_scales_1));

                    // Mins of first and second sub block of Q4_K block are arranged side by side
                    const __m512i mins_01 = _mm512_cvtepu8_epi16(_mm256_unpacklo_epi8(_mm256_shuffle_epi32(mins_and_scales_0, 78), _mm256_shuffle_epi32(mins_and_scales_1, 78)));

                    const __m512i scale_014589CD_0 = _mm512_shuffle_epi32(scales_0, (_MM_PERM_ENUM)68);
                    const __m512i scale_2367ABEF_0 = _mm512_shuffle_epi32(scales_0, (_MM_PERM_ENUM)238);

                    const __m512i scale_014589CD_1 = _mm512_shuffle_epi32(scales_1, (_MM_PERM_ENUM)68);
                    const __m512i scale_2367ABEF_1 = _mm512_shuffle_epi32(scales_1, (_MM_PERM_ENUM)238);

                    for (int rp = 0; rp < 4; rp++) {

                        // Load the four block_q8_k quantized values interleaved with each other in chunks of eight bytes - A0,A1,A2,A3
                        // Loaded as set of 128 bit vectors and repeated and stored into a 256 bit vector before again repeating into 512 bit vector
                        __m256i lhs_mat_ymm_0123_00 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 256 * sb)));
                        __m256i lhs_mat_ymm_01_00 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_00, lhs_mat_ymm_0123_00, 0);
                        __m256i lhs_mat_ymm_23_00 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_00, lhs_mat_ymm_0123_00, 17);
                        __m256i lhs_mat_ymm_0123_01 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 32 + 256 * sb)));
                        __m256i lhs_mat_ymm_01_01 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_01, lhs_mat_ymm_0123_01, 0);
                        __m256i lhs_mat_ymm_23_01 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_01, lhs_mat_ymm_0123_01, 17);
                        __m256i lhs_mat_ymm_0123_02 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 64 + 256 * sb)));
                        __m256i lhs_mat_ymm_01_02 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_02, lhs_mat_ymm_0123_02, 0);
                        __m256i lhs_mat_ymm_23_02 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_02, lhs_mat_ymm_0123_02, 17);
                        __m256i lhs_mat_ymm_0123_03 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 96 + 256 * sb)));
                        __m256i lhs_mat_ymm_01_03 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_03, lhs_mat_ymm_0123_03, 0);
                        __m256i lhs_mat_ymm_23_03 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_03, lhs_mat_ymm_0123_03, 17);
                        __m256i lhs_mat_ymm_0123_10 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 128 + 256 * sb)));
                        __m256i lhs_mat_ymm_01_10 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_10, lhs_mat_ymm_0123_10, 0);
                        __m256i lhs_mat_ymm_23_10 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_10, lhs_mat_ymm_0123_10, 17);
                        __m256i lhs_mat_ymm_0123_11 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 160 + 256 * sb)));
                        __m256i lhs_mat_ymm_01_11 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_11, lhs_mat_ymm_0123_11, 0);
                        __m256i lhs_mat_ymm_23_11 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_11, lhs_mat_ymm_0123_11, 17);
                        __m256i lhs_mat_ymm_0123_12 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 192 + 256 * sb)));
                        __m256i lhs_mat_ymm_01_12 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_12, lhs_mat_ymm_0123_12, 0);
                        __m256i lhs_mat_ymm_23_12 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_12, lhs_mat_ymm_0123_12, 17);
                        __m256i lhs_mat_ymm_0123_13 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 224 + 256 * sb)));
                        __m256i lhs_mat_ymm_01_13 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_13, lhs_mat_ymm_0123_13, 0);
                        __m256i lhs_mat_ymm_23_13 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_13, lhs_mat_ymm_0123_13, 17);

                        __m512i lhs_mat_01_00 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_00), lhs_mat_ymm_01_00, 1);
                        __m512i lhs_mat_23_00 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_00), lhs_mat_ymm_23_00, 1);
                        __m512i lhs_mat_01_01 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_01), lhs_mat_ymm_01_01, 1);
                        __m512i lhs_mat_23_01 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_01), lhs_mat_ymm_23_01, 1);
                        __m512i lhs_mat_01_02 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_02), lhs_mat_ymm_01_02, 1);
                        __m512i lhs_mat_23_02 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_02), lhs_mat_ymm_23_02, 1);
                        __m512i lhs_mat_01_03 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_03), lhs_mat_ymm_01_03, 1);
                        __m512i lhs_mat_23_03 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_03), lhs_mat_ymm_23_03, 1);

                        __m512i lhs_mat_01_10 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_10), lhs_mat_ymm_01_10, 1);
                        __m512i lhs_mat_23_10 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_10), lhs_mat_ymm_23_10, 1);
                        __m512i lhs_mat_01_11 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_11), lhs_mat_ymm_01_11, 1);
                        __m512i lhs_mat_23_11 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_11), lhs_mat_ymm_23_11, 1);
                        __m512i lhs_mat_01_12 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_12), lhs_mat_ymm_01_12, 1);
                        __m512i lhs_mat_23_12 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_12), lhs_mat_ymm_23_12, 1);
                        __m512i lhs_mat_01_13 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_13), lhs_mat_ymm_01_13, 1);
                        __m512i lhs_mat_23_13 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_13), lhs_mat_ymm_23_13, 1);

                        // Bsums are loaded - four bsums are loaded (for two sub blocks) for the different Q8_K blocks
                        __m256i lhs_bsums_0123_01 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].bsums + 16 * sb)));
                        __m256i lhs_bsums_hsum_ymm_0123_01 = _mm256_castsi128_si256(_mm_hadd_epi16(_mm256_castsi256_si128(lhs_bsums_0123_01), _mm256_extractf128_si256(lhs_bsums_0123_01, 1)));
                        lhs_bsums_hsum_ymm_0123_01 = _mm256_permute2x128_si256(lhs_bsums_hsum_ymm_0123_01, lhs_bsums_hsum_ymm_0123_01, 0);
                        __m512i lhs_bsums_hsum_0123_01 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_bsums_hsum_ymm_0123_01), lhs_bsums_hsum_ymm_0123_01, 1);

                        // Shuffle pattern one - left side input
                        const __m512i lhs_mat_01_00_sp1 = _mm512_shuffle_epi32(lhs_mat_01_00, (_MM_PERM_ENUM)160); //A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3) A00(0-3) A01(0-3) A01(0-3)
                        const __m512i lhs_mat_23_00_sp1 = _mm512_shuffle_epi32(lhs_mat_23_00, (_MM_PERM_ENUM)160); //A02(0-3) A02(0-3) A03(0-3) A03(0-3) A02(0-3) A02(0-3) A03(0-3) A03(0-3) A02(0-3) A02(0-3) A03(0-3) A03(0-3) A02(0-3) A02(0-3) A03(0-3) A03(0-3)
                        const __m512i lhs_mat_01_01_sp1 = _mm512_shuffle_epi32(lhs_mat_01_01, (_MM_PERM_ENUM)160); //A00(8-11) A00(8-11) A01(8-11) A01(8-11) A00(8-11) A00(8-11) A01(8-11) A01(8-11) A00(8-11) A00(8-11) A01(8-11) A01(8-11) A00(8-11) A00(8-11) A01(8-11) A01(8-11)
                        const __m512i lhs_mat_23_01_sp1 = _mm512_shuffle_epi32(lhs_mat_23_01, (_MM_PERM_ENUM)160); //A02(8-11) A02(8-11) A03(8-11) A03(8-11) A02(8-11) A02(8-11) A03(8-11) A03(8-11) A02(8-11) A02(8-11) A03(8-11) A03(8-11) A02(8-11) A02(8-11) A03(8-11) A03(8-11)
                        const __m512i lhs_mat_01_02_sp1 = _mm512_shuffle_epi32(lhs_mat_01_02, (_MM_PERM_ENUM)160); //A00(16-19) A00(16-19) A01(16-19) A01(16-19) A00(16-19) A00(16-19) A01(16-19) A01(16-19) A00(16-19) A00(16-19) A01(16-19) A01(16-19) A00(16-19) A00(16-19) A01(16-19) A01(16-19)
                        const __m512i lhs_mat_23_02_sp1 = _mm512_shuffle_epi32(lhs_mat_23_02, (_MM_PERM_ENUM)160); //A02(16-19) A02(16-19) A03(16-19) A03(16-19) A02(16-19) A02(16-19) A03(16-19) A03(16-19) A02(16-19) A02(16-19) A03(16-19) A03(16-19) A02(16-19) A02(16-19) A03(16-19) A03(16-19)
                        const __m512i lhs_mat_01_03_sp1 = _mm512_shuffle_epi32(lhs_mat_01_03, (_MM_PERM_ENUM)160); //A00(24-27) A00(24-27) A01(24-27) A01(24-27) A00(24-27) A00(24-27) A01(24-27) A01(24-27) A00(24-27) A00(24-27) A01(24-27) A01(24-27) A00(24-27) A00(24-27) A01(24-27) A01(24-27)
                        const __m512i lhs_mat_23_03_sp1 = _mm512_shuffle_epi32(lhs_mat_23_03, (_MM_PERM_ENUM)160); //A02(24-27) A02(24-27) A03(24-27) A03(24-27) A02(24-27) A02(24-27) A03(24-27) A03(24-27) A02(24-27) A02(24-27) A03(24-27) A03(24-27) A02(24-27) A02(24-27) A03(24-27) A03(24-27)

                        const __m512i lhs_mat_01_10_sp1 = _mm512_shuffle_epi32(lhs_mat_01_10, (_MM_PERM_ENUM)160); //A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3) A10(0-3) A11(0-3) A11(0-3)
                        const __m512i lhs_mat_23_10_sp1 = _mm512_shuffle_epi32(lhs_mat_23_10, (_MM_PERM_ENUM)160); //A12(0-3) A12(0-3) A13(0-3) A13(0-3) A12(0-3) A12(0-3) A13(0-3) A13(0-3) A12(0-3) A12(0-3) A13(0-3) A13(0-3) A12(0-3) A12(0-3) A13(0-3) A13(0-3)
                        const __m512i lhs_mat_01_11_sp1 = _mm512_shuffle_epi32(lhs_mat_01_11, (_MM_PERM_ENUM)160); //A10(8-11) A10(8-11) A11(8-11) A11(8-11) A10(8-11) A10(8-11) A11(8-11) A11(8-11) A10(8-11) A10(8-11) A11(8-11) A11(8-11) A10(8-11) A10(8-11) A11(8-11) A11(8-11)
                        const __m512i lhs_mat_23_11_sp1 = _mm512_shuffle_epi32(lhs_mat_23_11, (_MM_PERM_ENUM)160); //A12(8-11) A12(8-11) A13(8-11) A13(8-11) A12(8-11) A12(8-11) A13(8-11) A13(8-11) A12(8-11) A12(8-11) A13(8-11) A13(8-11) A12(8-11) A12(8-11) A13(8-11) A13(8-11)
                        const __m512i lhs_mat_01_12_sp1 = _mm512_shuffle_epi32(lhs_mat_01_12, (_MM_PERM_ENUM)160); //A10(16-19) A10(16-19) A11(16-19) A11(16-19) A10(16-19) A10(16-19) A11(16-19) A11(16-19) A10(16-19) A10(16-19) A11(16-19) A11(16-19) A10(16-19) A10(16-19) A11(16-19) A11(16-19)
                        const __m512i lhs_mat_23_12_sp1 = _mm512_shuffle_epi32(lhs_mat_23_12, (_MM_PERM_ENUM)160); //A12(16-19) A12(16-19) A13(16-19) A13(16-19) A12(16-19) A12(16-19) A13(16-19) A13(16-19) A12(16-19) A12(16-19) A13(16-19) A13(16-19) A12(16-19) A12(16-19) A13(16-19) A13(16-19)
                        const __m512i lhs_mat_01_13_sp1 = _mm512_shuffle_epi32(lhs_mat_01_13, (_MM_PERM_ENUM)160); //A10(24-27) A10(24-27) A11(24-27) A11(24-27) A10(24-27) A10(24-27) A11(24-27) A11(24-27) A10(24-27) A10(24-27) A11(24-27) A11(24-27) A10(24-27) A10(24-27) A11(24-27) A11(24-27)
                        const __m512i lhs_mat_23_13_sp1 = _mm512_shuffle_epi32(lhs_mat_23_13, (_MM_PERM_ENUM)160); //A12(24-27) A12(24-27) A13(24-27) A13(24-27) A12(24-27) A12(24-27) A13(24-27) A13(24-27) A12(24-27) A12(24-27) A13(24-27) A13(24-27) A12(24-27) A12(24-27) A13(24-27) A13(24-27)

                        const __m512i lhs_mat_01_00_sp2 = _mm512_shuffle_epi32(lhs_mat_01_00, (_MM_PERM_ENUM)245); //A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7) A00(4-7) A01(4-7) A01(4-7)
                        const __m512i lhs_mat_23_00_sp2 = _mm512_shuffle_epi32(lhs_mat_23_00, (_MM_PERM_ENUM)245); //A02(4-7) A02(4-7) A03(4-7) A03(4-7) A02(4-7) A02(4-7) A03(4-7) A03(4-7) A02(4-7) A02(4-7) A03(4-7) A03(4-7) A02(4-7) A02(4-7) A03(4-7) A03(4-7)
                        const __m512i lhs_mat_01_01_sp2 = _mm512_shuffle_epi32(lhs_mat_01_01, (_MM_PERM_ENUM)245); //A00(12-15) A00(12-15) A01(12-15) A01(12-15) A00(12-15) A00(12-15) A01(12-15) A01(12-15) A00(12-15) A00(12-15) A01(12-15) A01(12-15) A00(12-15) A00(12-15) A01(12-15) A01(12-15)
                        const __m512i lhs_mat_23_01_sp2 = _mm512_shuffle_epi32(lhs_mat_23_01, (_MM_PERM_ENUM)245); //A02(12-15) A02(12-15) A03(12-15) A03(12-15) A02(12-15) A02(12-15) A03(12-15) A03(12-15) A02(12-15) A02(12-15) A03(12-15) A03(12-15) A02(12-15) A02(12-15) A03(12-15) A03(12-15)
                        const __m512i lhs_mat_01_02_sp2 = _mm512_shuffle_epi32(lhs_mat_01_02, (_MM_PERM_ENUM)245); //A00(20-23) A00(20-23) A01(20-23) A01(20-23) A00(20-23) A00(20-23) A01(20-23) A01(20-23) A00(20-23) A00(20-23) A01(20-23) A01(20-23) A00(20-23) A00(20-23) A01(20-23) A01(20-23)
                        const __m512i lhs_mat_23_02_sp2 = _mm512_shuffle_epi32(lhs_mat_23_02, (_MM_PERM_ENUM)245); //A02(20-23) A02(20-23) A03(20-23) A03(20-23) A02(20-23) A02(20-23) A03(20-23) A03(20-23) A02(20-23) A02(20-23) A03(20-23) A03(20-23) A02(20-23) A02(20-23) A03(20-23) A03(20-23)
                        const __m512i lhs_mat_01_03_sp2 = _mm512_shuffle_epi32(lhs_mat_01_03, (_MM_PERM_ENUM)245); //A00(28-31) A00(28-31) A01(28-31) A01(28-31) A00(28-31) A00(28-31) A01(28-31) A01(28-31) A00(28-31) A00(28-31) A01(28-31) A01(28-31) A00(28-31) A00(28-31) A01(28-31) A01(28-31)
                        const __m512i lhs_mat_23_03_sp2 = _mm512_shuffle_epi32(lhs_mat_23_03, (_MM_PERM_ENUM)245); //A02(28-31) A02(28-31) A03(28-31) A03(28-31) A02(28-31) A02(28-31) A03(28-31) A03(28-31) A02(28-31) A02(28-31) A03(28-31) A03(28-31) A02(28-31) A02(28-31) A03(28-31) A03(28-31)

                        const __m512i lhs_mat_01_10_sp2 = _mm512_shuffle_epi32(lhs_mat_01_10, (_MM_PERM_ENUM)245); //A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7) A10(4-7) A11(4-7) A11(4-7)
                        const __m512i lhs_mat_23_10_sp2 = _mm512_shuffle_epi32(lhs_mat_23_10, (_MM_PERM_ENUM)245); //A12(4-7) A12(4-7) A13(4-7) A13(4-7) A12(4-7) A12(4-7) A13(4-7) A13(4-7) A12(4-7) A12(4-7) A13(4-7) A13(4-7) A12(4-7) A12(4-7) A13(4-7) A13(4-7)
                        const __m512i lhs_mat_01_11_sp2 = _mm512_shuffle_epi32(lhs_mat_01_11, (_MM_PERM_ENUM)245); //A10(12-15) A10(12-15) A11(12-15) A11(12-15) A10(12-15) A10(12-15) A11(12-15) A11(12-15) A10(12-15) A10(12-15) A11(12-15) A11(12-15) A10(12-15) A10(12-15) A11(12-15) A11(12-15)
                        const __m512i lhs_mat_23_11_sp2 = _mm512_shuffle_epi32(lhs_mat_23_11, (_MM_PERM_ENUM)245); //A12(12-15) A12(12-15) A13(12-15) A13(12-15) A12(12-15) A12(12-15) A13(12-15) A13(12-15) A12(12-15) A12(12-15) A13(12-15) A13(12-15) A12(12-15) A12(12-15) A13(12-15) A13(12-15)
                        const __m512i lhs_mat_01_12_sp2 = _mm512_shuffle_epi32(lhs_mat_01_12, (_MM_PERM_ENUM)245); //A10(20-23) A10(20-23) A11(20-23) A11(20-23) A10(20-23) A10(20-23) A11(20-23) A11(20-23) A10(20-23) A10(20-23) A11(20-23) A11(20-23) A10(20-23) A10(20-23) A11(20-23) A11(20-23)
                        const __m512i lhs_mat_23_12_sp2 = _mm512_shuffle_epi32(lhs_mat_23_12, (_MM_PERM_ENUM)245); //A12(20-23) A12(20-23) A13(20-23) A13(20-23) A12(20-23) A12(20-23) A13(20-23) A13(20-23) A12(20-23) A12(20-23) A13(20-23) A13(20-23) A12(20-23) A12(20-23) A13(20-23) A13(20-23)
                        const __m512i lhs_mat_01_13_sp2 = _mm512_shuffle_epi32(lhs_mat_01_13, (_MM_PERM_ENUM)245); //A10(28-31) A10(28-31) A11(28-31) A11(28-31) A10(28-31) A10(28-31) A11(28-31) A11(28-31) A10(28-31) A10(28-31) A11(28-31) A11(28-31) A10(28-31) A10(28-31) A11(28-31) A11(28-31)
                        const __m512i lhs_mat_23_13_sp2 = _mm512_shuffle_epi32(lhs_mat_23_13, (_MM_PERM_ENUM)245); //A12(28-31) A12(28-31) A13(28-31) A13(28-31) A12(28-31) A12(28-31) A13(28-31) A13(28-31) A12(28-31) A12(28-31) A13(28-31) A13(28-31) A12(28-31) A12(28-31) A13(28-31) A13(28-31)

                        // The values arranged in shuffle patterns are operated with dot product operation within 32 bit lane i.e corresponding bytes and multiplied and added into 32 bit integers within 32 bit lane
                        __m512i iacc_mat_00_0_sp1 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_03_sp1, lhs_mat_01_03_sp1), _mm512_maddubs_epi16(rhs_mat_014589CD_02_sp1, lhs_mat_01_02_sp1)), _mm512_maddubs_epi16(rhs_mat_014589CD_01_sp1, lhs_mat_01_01_sp1)), _mm512_maddubs_epi16(rhs_mat_014589CD_00_sp1, lhs_mat_01_00_sp1));
                        __m512i iacc_mat_01_0_sp1 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_03_sp1, lhs_mat_01_03_sp1), _mm512_maddubs_epi16(rhs_mat_2367ABEF_02_sp1, lhs_mat_01_02_sp1)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_01_sp1, lhs_mat_01_01_sp1)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_00_sp1, lhs_mat_01_00_sp1));
                        __m512i iacc_mat_10_0_sp1 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_03_sp1, lhs_mat_23_03_sp1), _mm512_maddubs_epi16(rhs_mat_014589CD_02_sp1, lhs_mat_23_02_sp1)), _mm512_maddubs_epi16(rhs_mat_014589CD_01_sp1, lhs_mat_23_01_sp1)), _mm512_maddubs_epi16(rhs_mat_014589CD_00_sp1, lhs_mat_23_00_sp1));
                        __m512i iacc_mat_11_0_sp1 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_03_sp1, lhs_mat_23_03_sp1), _mm512_maddubs_epi16(rhs_mat_2367ABEF_02_sp1, lhs_mat_23_02_sp1)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_01_sp1, lhs_mat_23_01_sp1)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_00_sp1, lhs_mat_23_00_sp1));
                        __m512i iacc_mat_00_1_sp1 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_13_sp1, lhs_mat_01_13_sp1), _mm512_maddubs_epi16(rhs_mat_014589CD_12_sp1, lhs_mat_01_12_sp1)), _mm512_maddubs_epi16(rhs_mat_014589CD_11_sp1, lhs_mat_01_11_sp1)), _mm512_maddubs_epi16(rhs_mat_014589CD_10_sp1, lhs_mat_01_10_sp1));
                        __m512i iacc_mat_01_1_sp1 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_13_sp1, lhs_mat_01_13_sp1), _mm512_maddubs_epi16(rhs_mat_2367ABEF_12_sp1, lhs_mat_01_12_sp1)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_11_sp1, lhs_mat_01_11_sp1)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_10_sp1, lhs_mat_01_10_sp1));
                        __m512i iacc_mat_10_1_sp1 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_13_sp1, lhs_mat_23_13_sp1), _mm512_maddubs_epi16(rhs_mat_014589CD_12_sp1, lhs_mat_23_12_sp1)), _mm512_maddubs_epi16(rhs_mat_014589CD_11_sp1, lhs_mat_23_11_sp1)), _mm512_maddubs_epi16(rhs_mat_014589CD_10_sp1, lhs_mat_23_10_sp1));
                        __m512i iacc_mat_11_1_sp1 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_13_sp1, lhs_mat_23_13_sp1), _mm512_maddubs_epi16(rhs_mat_2367ABEF_12_sp1, lhs_mat_23_12_sp1)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_11_sp1, lhs_mat_23_11_sp1)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_10_sp1, lhs_mat_23_10_sp1));

                        __m512i iacc_mat_00_0_sp2 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_03_sp2, lhs_mat_01_03_sp2), _mm512_maddubs_epi16(rhs_mat_014589CD_02_sp2, lhs_mat_01_02_sp2)), _mm512_maddubs_epi16(rhs_mat_014589CD_01_sp2, lhs_mat_01_01_sp2)), _mm512_maddubs_epi16(rhs_mat_014589CD_00_sp2, lhs_mat_01_00_sp2));
                        __m512i iacc_mat_01_0_sp2 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_03_sp2, lhs_mat_01_03_sp2), _mm512_maddubs_epi16(rhs_mat_2367ABEF_02_sp2, lhs_mat_01_02_sp2)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_01_sp2, lhs_mat_01_01_sp2)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_00_sp2, lhs_mat_01_00_sp2));
                        __m512i iacc_mat_10_0_sp2 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_03_sp2, lhs_mat_23_03_sp2), _mm512_maddubs_epi16(rhs_mat_014589CD_02_sp2, lhs_mat_23_02_sp2)), _mm512_maddubs_epi16(rhs_mat_014589CD_01_sp2, lhs_mat_23_01_sp2)), _mm512_maddubs_epi16(rhs_mat_014589CD_00_sp2, lhs_mat_23_00_sp2));
                        __m512i iacc_mat_11_0_sp2 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_03_sp2, lhs_mat_23_03_sp2), _mm512_maddubs_epi16(rhs_mat_2367ABEF_02_sp2, lhs_mat_23_02_sp2)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_01_sp2, lhs_mat_23_01_sp2)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_00_sp2, lhs_mat_23_00_sp2));
                        __m512i iacc_mat_00_1_sp2 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_13_sp2, lhs_mat_01_13_sp2), _mm512_maddubs_epi16(rhs_mat_014589CD_12_sp2, lhs_mat_01_12_sp2)), _mm512_maddubs_epi16(rhs_mat_014589CD_11_sp2, lhs_mat_01_11_sp2)), _mm512_maddubs_epi16(rhs_mat_014589CD_10_sp2, lhs_mat_01_10_sp2));
                        __m512i iacc_mat_01_1_sp2 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_13_sp2, lhs_mat_01_13_sp2), _mm512_maddubs_epi16(rhs_mat_2367ABEF_12_sp2, lhs_mat_01_12_sp2)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_11_sp2, lhs_mat_01_11_sp2)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_10_sp2, lhs_mat_01_10_sp2));
                        __m512i iacc_mat_10_1_sp2 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_13_sp2, lhs_mat_23_13_sp2), _mm512_maddubs_epi16(rhs_mat_014589CD_12_sp2, lhs_mat_23_12_sp2)), _mm512_maddubs_epi16(rhs_mat_014589CD_11_sp2, lhs_mat_23_11_sp2)), _mm512_maddubs_epi16(rhs_mat_014589CD_10_sp2, lhs_mat_23_10_sp2));
                        __m512i iacc_mat_11_1_sp2 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_13_sp2, lhs_mat_23_13_sp2), _mm512_maddubs_epi16(rhs_mat_2367ABEF_12_sp2, lhs_mat_23_12_sp2)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_11_sp2, lhs_mat_23_11_sp2)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_10_sp2, lhs_mat_23_10_sp2));

                        // Output of both shuffle patterns are added in order to sum dot product outputs of all 32 values in block
                        __m512i iacc_mat_00_0 = _mm512_add_epi16(iacc_mat_00_0_sp1, iacc_mat_00_0_sp2);
                        __m512i iacc_mat_01_0 = _mm512_add_epi16(iacc_mat_01_0_sp1, iacc_mat_01_0_sp2);
                        __m512i iacc_mat_10_0 = _mm512_add_epi16(iacc_mat_10_0_sp1, iacc_mat_10_0_sp2);
                        __m512i iacc_mat_11_0 = _mm512_add_epi16(iacc_mat_11_0_sp1, iacc_mat_11_0_sp2);

                        __m512i iacc_mat_00_1 = _mm512_add_epi16(iacc_mat_00_1_sp1, iacc_mat_00_1_sp2);
                        __m512i iacc_mat_01_1 = _mm512_add_epi16(iacc_mat_01_1_sp1, iacc_mat_01_1_sp2);
                        __m512i iacc_mat_10_1 = _mm512_add_epi16(iacc_mat_10_1_sp1, iacc_mat_10_1_sp2);
                        __m512i iacc_mat_11_1 = _mm512_add_epi16(iacc_mat_11_1_sp1, iacc_mat_11_1_sp2);

                        iacc_mat_00_0 = _mm512_madd_epi16(iacc_mat_00_0, scale_014589CD_0);
                        iacc_mat_01_0 = _mm512_madd_epi16(iacc_mat_01_0, scale_2367ABEF_0);
                        iacc_mat_10_0 = _mm512_madd_epi16(iacc_mat_10_0, scale_014589CD_0);
                        iacc_mat_11_0 = _mm512_madd_epi16(iacc_mat_11_0, scale_2367ABEF_0);

                        iacc_mat_00_1 = _mm512_madd_epi16(iacc_mat_00_1, scale_014589CD_1);
                        iacc_mat_01_1 = _mm512_madd_epi16(iacc_mat_01_1, scale_2367ABEF_1);
                        iacc_mat_10_1 = _mm512_madd_epi16(iacc_mat_10_1, scale_014589CD_1);
                        iacc_mat_11_1 = _mm512_madd_epi16(iacc_mat_11_1, scale_2367ABEF_1);

                        // Straighten out to make 4 row vectors (4 for each sub block which are accumulated together in the next step)
                        __m512i iacc_row_0_0 = _mm512_mask_blend_epi32(0xCCCC, iacc_mat_00_0, _mm512_shuffle_epi32(iacc_mat_01_0, (_MM_PERM_ENUM)78));
                        __m512i iacc_row_1_0 = _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32(iacc_mat_00_0, (_MM_PERM_ENUM)78), iacc_mat_01_0);
                        __m512i iacc_row_2_0 = _mm512_mask_blend_epi32(0xCCCC, iacc_mat_10_0, _mm512_shuffle_epi32(iacc_mat_11_0, (_MM_PERM_ENUM)78));
                        __m512i iacc_row_3_0 = _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32(iacc_mat_10_0, (_MM_PERM_ENUM)78), iacc_mat_11_0);
                        __m512i iacc_row_0_1 = _mm512_mask_blend_epi32(0xCCCC, iacc_mat_00_1, _mm512_shuffle_epi32(iacc_mat_01_1, (_MM_PERM_ENUM)78));
                        __m512i iacc_row_1_1 = _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32(iacc_mat_00_1, (_MM_PERM_ENUM)78), iacc_mat_01_1);
                        __m512i iacc_row_2_1 = _mm512_mask_blend_epi32(0xCCCC, iacc_mat_10_1, _mm512_shuffle_epi32(iacc_mat_11_1, (_MM_PERM_ENUM)78));
                        __m512i iacc_row_3_1 = _mm512_mask_blend_epi32(0xCCCC,_mm512_shuffle_epi32(iacc_mat_10_1, (_MM_PERM_ENUM)78), iacc_mat_11_1);

                        __m512i iacc_row_0 = _mm512_add_epi32(iacc_row_0_0, iacc_row_0_1);
                        __m512i iacc_row_1 = _mm512_add_epi32(iacc_row_1_0, iacc_row_1_1);
                        __m512i iacc_row_2 = _mm512_add_epi32(iacc_row_2_0, iacc_row_2_1);
                        __m512i iacc_row_3 = _mm512_add_epi32(iacc_row_3_0, iacc_row_3_1);

                        // Load the scale(d) values for all the 4 Q8_k blocks and repeat it across lanes
                        const __m128 row_scale_f32_sse = _mm_load_ps(a_ptrs[rp][b].d);
                        const __m256 row_scale_f32_ymm = _mm256_set_m128(row_scale_f32_sse, row_scale_f32_sse);
                        const __m512 row_scale_f32 = _mm512_insertf32x8(_mm512_castps256_ps512(row_scale_f32_ymm), row_scale_f32_ymm, 1);

                        // Multiply with appropriate scales and accumulate (for both d and dmin) below
                        acc_rows[rp * 4] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_0), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 0)), acc_rows[rp * 4]);
                        acc_rows[rp * 4  + 1] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_1), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 85)), acc_rows[rp * 4 + 1]);
                        acc_rows[rp * 4 + 2] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_2), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_rows[rp * 4 + 2]);
                        acc_rows[rp * 4 + 3] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_3), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_rows[rp * 4 + 3]);

                        __m512i iacc_row_min_0 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_hsum_0123_01, (_MM_PERM_ENUM)0), mins_01);
                        __m512i iacc_row_min_1 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_hsum_0123_01, (_MM_PERM_ENUM)85), mins_01);
                        __m512i iacc_row_min_2 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_hsum_0123_01, (_MM_PERM_ENUM)170), mins_01);
                        __m512i iacc_row_min_3 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_hsum_0123_01, (_MM_PERM_ENUM)255), mins_01);

                        acc_min_rows[rp * 4] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_min_0), _mm512_mul_ps(col_dmin_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 0)), acc_min_rows[rp * 4]);
                        acc_min_rows[rp * 4 + 1] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_min_1), _mm512_mul_ps(col_dmin_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 85)), acc_min_rows[rp * 4 + 1]);
                        acc_min_rows[rp * 4 + 2] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_min_2), _mm512_mul_ps(col_dmin_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_min_rows[rp * 4 + 2]);
                        acc_min_rows[rp * 4 + 3] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_min_3), _mm512_mul_ps(col_dmin_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_min_rows[rp * 4 + 3]);
                    }
                }
            }
            // Store the accumulated values
            for (int i = 0; i < 16; i++) {
                _mm512_storeu_ps((float * )(s + ((y * 4 + i) * bs + x * 8)), _mm512_sub_ps(acc_rows[i], acc_min_rows[i]));
            }
        }
    }

    for (; y < nr / 4; y++) {

        const block_q8_Kx4 * a_ptr = a_ptr_start + (y * nb);

        // Take group of eight block_q4_kx8 structures at each pass of the loop and perform dot product operation
        for (int64_t x = 0; x < anc / 8; x += 2) {

            const block_q4_Kx8 * b_ptr_0 = b_ptr_start + ((x) * b_nb);
            const block_q4_Kx8 * b_ptr_1 = b_ptr_start + ((x + 1) * b_nb);

            // Master FP accumulators
            __m512 acc_rows[4];
            for (int i = 0; i < 4; i++) {
                acc_rows[i] = _mm512_setzero_ps();
            }

            __m512 acc_min_rows[4];
            for (int i = 0; i < 4; i++) {
                acc_min_rows[i] = _mm512_setzero_ps();
            }

            // For super block
            for (int64_t b = 0; b < nb; b++) {
                // Scale values - Load the sixteen scale values from two block_q4_kx8 structures
                const __m512 col_scale_f32 = GGML_F32Cx8x2_LOAD(b_ptr_0[b].d, b_ptr_1[b].d);

                // dmin values - Load the sixteen dmin values from two block_q4_kx8 structures
                const __m512 col_dmin_f32 = GGML_F32Cx8x2_LOAD(b_ptr_0[b].dmin, b_ptr_1[b].dmin);

                // Loop to iterate over the eight sub blocks of a super block - two sub blocks are processed per iteration
                for (int sb = 0; sb < QK_K / 64; sb++) {

                    const __m256i rhs_raw_mat_0123_0 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + sb * 256));
                    const __m256i rhs_raw_mat_4567_0 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 32 + sb * 256));
                    const __m256i rhs_raw_mat_0123_1 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 64 + sb * 256));
                    const __m256i rhs_raw_mat_4567_1 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 96 + sb * 256));
                    const __m256i rhs_raw_mat_0123_2 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 128 + sb * 256));
                    const __m256i rhs_raw_mat_4567_2 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 160 + sb * 256));
                    const __m256i rhs_raw_mat_0123_3 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 192 + sb * 256));
                    const __m256i rhs_raw_mat_4567_3 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 224 + sb * 256));

                    const __m256i rhs_raw_mat_89AB_0 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + sb * 256));
                    const __m256i rhs_raw_mat_CDEF_0 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 32 + sb * 256));
                    const __m256i rhs_raw_mat_89AB_1 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 64 + sb * 256));
                    const __m256i rhs_raw_mat_CDEF_1 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 96 + sb * 256));
                    const __m256i rhs_raw_mat_89AB_2 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 128 + sb * 256));
                    const __m256i rhs_raw_mat_CDEF_2 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 160 + sb * 256));
                    const __m256i rhs_raw_mat_89AB_3 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 192 + sb * 256));
                    const __m256i rhs_raw_mat_CDEF_3 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 224 + sb * 256));

                    const __m256i rhs_raw_mat_0145_0 = _mm256_blend_epi32(rhs_raw_mat_0123_0, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_0, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_0, requiredOrder), rhs_raw_mat_4567_0, 240);
                    const __m256i rhs_raw_mat_0145_1 = _mm256_blend_epi32(rhs_raw_mat_0123_1, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_1, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_1, requiredOrder), rhs_raw_mat_4567_1, 240);
                    const __m256i rhs_raw_mat_0145_2 = _mm256_blend_epi32(rhs_raw_mat_0123_2, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_2, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_2 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_2, requiredOrder), rhs_raw_mat_4567_2, 240);
                    const __m256i rhs_raw_mat_0145_3 = _mm256_blend_epi32(rhs_raw_mat_0123_3, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_3, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_3 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_3, requiredOrder), rhs_raw_mat_4567_3, 240);

                    const __m256i rhs_raw_mat_89CD_0 = _mm256_blend_epi32(rhs_raw_mat_89AB_0, _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_0, requiredOrder), 240);
                    const __m256i rhs_raw_mat_ABEF_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_0, requiredOrder), rhs_raw_mat_CDEF_0, 240);
                    const __m256i rhs_raw_mat_89CD_1 = _mm256_blend_epi32(rhs_raw_mat_89AB_1, _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_1, requiredOrder), 240);
                    const __m256i rhs_raw_mat_ABEF_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_1, requiredOrder), rhs_raw_mat_CDEF_1, 240);
                    const __m256i rhs_raw_mat_89CD_2 = _mm256_blend_epi32(rhs_raw_mat_89AB_2, _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_2, requiredOrder), 240);
                    const __m256i rhs_raw_mat_ABEF_2 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_2, requiredOrder), rhs_raw_mat_CDEF_2, 240);
                    const __m256i rhs_raw_mat_89CD_3 = _mm256_blend_epi32(rhs_raw_mat_89AB_3, _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_3, requiredOrder), 240);
                    const __m256i rhs_raw_mat_ABEF_3 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_3, requiredOrder), rhs_raw_mat_CDEF_3, 240);

                    const __m512i rhs_raw_mat_014589CD_0 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_0145_0), rhs_raw_mat_89CD_0, 1);
                    const __m512i rhs_raw_mat_2367ABEF_0 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_2367_0), rhs_raw_mat_ABEF_0, 1);
                    const __m512i rhs_raw_mat_014589CD_1 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_0145_1), rhs_raw_mat_89CD_1, 1);
                    const __m512i rhs_raw_mat_2367ABEF_1 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_2367_1), rhs_raw_mat_ABEF_1, 1);

                    const __m512i rhs_raw_mat_014589CD_2 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_0145_2), rhs_raw_mat_89CD_2, 1);
                    const __m512i rhs_raw_mat_2367ABEF_2 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_2367_2), rhs_raw_mat_ABEF_2, 1);
                    const __m512i rhs_raw_mat_014589CD_3 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_0145_3), rhs_raw_mat_89CD_3, 1);
                    const __m512i rhs_raw_mat_2367ABEF_3 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_2367_3), rhs_raw_mat_ABEF_3, 1);

                    //4-bit -> 8-bit
                    const __m512i rhs_mat_014589CD_00 = _mm512_and_si512(rhs_raw_mat_014589CD_0, m4bexpanded); //B00(0-7) B01(0-7) B04(0-7) B05(0-7) B08(0-7) B09(0-7) B0C(0-7) B0D(0-7)
                    const __m512i rhs_mat_2367ABEF_00 = _mm512_and_si512(rhs_raw_mat_2367ABEF_0, m4bexpanded); //B02(0-7) B03(0-7) B06(0-7) B07(0-7) B0A(0-7) B0B(0-7) B0E(0-7) B0F(0-7)
                    const __m512i rhs_mat_014589CD_01 = _mm512_and_si512(rhs_raw_mat_014589CD_1, m4bexpanded); //B00(8-15) B01(8-15) B04(8-15) B05(8-15) B08(8-15) B09(8-15) B0C(8-15) B0D(8-15)
                    const __m512i rhs_mat_2367ABEF_01 = _mm512_and_si512(rhs_raw_mat_2367ABEF_1, m4bexpanded); //B02(8-15) B03(8-15) B06(8-15) B07(8-15) B0A(8-15) B0B(8-15) B0E(8-15) B0F(8-15)

                    const __m512i rhs_mat_014589CD_02 = _mm512_and_si512(rhs_raw_mat_014589CD_2, m4bexpanded); //B00(16-23) B01(16-23) B04(16-23) B05(16-23) B08(16-23) B09(16-23) B0C(16-23) B0D(16-23)
                    const __m512i rhs_mat_2367ABEF_02 = _mm512_and_si512(rhs_raw_mat_2367ABEF_2, m4bexpanded); //B02(16-23) B03(16-23) B06(16-23) B07(16-23) B0A(16-23) B0B(16-23) B0E(16-23) B0F(16-23)
                    const __m512i rhs_mat_014589CD_03 = _mm512_and_si512(rhs_raw_mat_014589CD_3, m4bexpanded); //B00(24-31) B01(24-31) B04(24-31) B05(24-31) B08(24-31) B09(24-31) B0C(24-31) B0D(24-31)
                    const __m512i rhs_mat_2367ABEF_03 = _mm512_and_si512(rhs_raw_mat_2367ABEF_3, m4bexpanded); //B02(24-31) B03(24-31) B06(24-31) B07(24-31) B0A(24-31) B0B(24-31) B0E(24-31) B0F(24-31)

                    const __m512i rhs_mat_014589CD_10 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_0, 4), m4bexpanded); //B10(0-7) B11(0-7) B14(0-7) B15(0-7) B18(0-7) B19(0-7) B1C(0-7) B1D(0-7)
                    const __m512i rhs_mat_2367ABEF_10 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_0, 4), m4bexpanded); //B12(0-7) B13(0-7) B16(0-7) B17(0-7) B1A(0-7) B1B(0-7) B1E(0-7) B1F(0-7)
                    const __m512i rhs_mat_014589CD_11 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_1, 4), m4bexpanded); //B10(8-15) B11(8-15) B14(8-15) B15(8-15) B18(8-15) B19(8-15) B1C(8-15) B1D(8-15)
                    const __m512i rhs_mat_2367ABEF_11 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_1, 4), m4bexpanded); //B12(8-15) B13(8-15) B16(8-15) B17(8-15) B1A(8-15) B1B(8-15) B1E(8-15) B1F(8-15)

                    const __m512i rhs_mat_014589CD_12 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_2, 4), m4bexpanded); //B10(16-23) B11(16-23) B14(16-23) B15(16-23) B18(16-23) B19(16-23) B1C(16-23) B1D(16-23)
                    const __m512i rhs_mat_2367ABEF_12 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_2, 4), m4bexpanded); //B12(16-23) B13(16-23) B16(16-23) B17(16-23) B1A(16-23) B1B(16-23) B1E(16-23) B1F(16-23)
                    const __m512i rhs_mat_014589CD_13 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_3, 4), m4bexpanded); //B10(24-31) B11(24-31) B14(24-31) B15(24-31) B18(24-31) B19(24-31) B1C(24-31) B1D(24-31)
                    const __m512i rhs_mat_2367ABEF_13 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_3, 4), m4bexpanded); //B12(24-31) B13(24-31) B16(24-31) B17(24-31) B1A(24-31) B1B(24-31) B1E(24-31) B1F(24-31)

                    // Shuffle pattern one - right side input
                    const __m512i rhs_mat_014589CD_00_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_00, (_MM_PERM_ENUM)136); //B00(0-3) B01(0-3) B00(0-3) B01(0-3) B04(0-3) B05(0-3) B04(0-3) B05(0-3) B08(0-3) B09(0-3) B08(0-3) B09(0-3) B0C(0-3) B0D(0-3) B0C(0-3) B0D(0-3)
                    const __m512i rhs_mat_2367ABEF_00_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_00, (_MM_PERM_ENUM)136); //B02(0-3) B03(0-3) B02(0-3) B03(0-3) B06(0-3) B07(0-3) B06(0-3) B07(0-3) B0A(0-3) B0B(0-3) B0A(0-3) B0B(0-3) B0E(0-3) B0F(0-3) B0E(0-3) B0F(0-3)
                    const __m512i rhs_mat_014589CD_01_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_01, (_MM_PERM_ENUM)136); //B00(8-11) B01(8-11) B00(8-11) B01(8-11) B04(8-11) B05(8-11) B04(8-11) B05(8-11) B08(8-11) B09(8-11) B08(8-11) B09(8-11) B0C(8-11) B0D(8-11) B0C(8-11) B0D(8-11)
                    const __m512i rhs_mat_2367ABEF_01_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_01, (_MM_PERM_ENUM)136); //B02(8-11) B03(8-11) B02(8-11) B03(8-11) B06(8-11) B07(8-11) B06(8-11) B07(8-11) B0A(8-11) B0B(8-11) B0A(8-11) B0B(8-11) B0E(8-11) B0F(8-11) B0E(8-11) B0F(8-11)
                    const __m512i rhs_mat_014589CD_02_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_02, (_MM_PERM_ENUM)136); //B00(16-19) B01(16-19) B00(16-19) B01(16-19) B04(16-19) B05(16-19) B04(16-19) B05(16-19) B08(16-19) B09(16-19) B08(16-19) B09(16-19) B0C(16-19) B0D(16-19) B0C(16-19) B0D(16-19)
                    const __m512i rhs_mat_2367ABEF_02_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_02, (_MM_PERM_ENUM)136); //B02(16-19) B03(16-19) B02(16-19) B03(16-19) B06(16-19) B07(16-19) B06(16-19) B07(16-19) B0A(16-19) B0B(16-19) B0A(16-19) B0B(16-19) B0E(16-19) B0F(16-19) B0E(16-19) B0F(16-19)
                    const __m512i rhs_mat_014589CD_03_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_03, (_MM_PERM_ENUM)136); //B00(24-27) B01(24-27) B00(24-27) B01(24-27) B04(24-27) B05(24-27) B04(24-27) B05(24-27) B08(24-27) B09(24-27) B08(24-27) B09(24-27) B0C(24-27) B0D(24-27) B0C(24-27) B0D(24-27)
                    const __m512i rhs_mat_2367ABEF_03_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_03, (_MM_PERM_ENUM)136); //B02(24-27) B03(24-27) B02(24-27) B03(24-27) B06(24-27) B07(24-27) B06(24-27) B07(24-27) B0A(24-27) B0B(24-27) B0A(24-27) B0B(24-27) B0E(24-27) B0F(24-27) B0E(24-27) B0F(24-27)

                    const __m512i rhs_mat_014589CD_10_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_10, (_MM_PERM_ENUM)136); //B10(0-3) B11(0-3) B10(0-3) B11(0-3) B14(0-3) B15(0-3) B14(0-3) B15(0-3) B18(0-3) B19(0-3) B18(0-3) B19(0-3) B1C(0-3) B1D(0-3) B1C(0-3) B1D(0-3)
                    const __m512i rhs_mat_2367ABEF_10_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_10, (_MM_PERM_ENUM)136); //B12(0-3) B13(0-3) B12(0-3) B13(0-3) B16(0-3) B17(0-3) B16(0-3) B17(0-3) B1A(0-3) B1B(0-3) B1A(0-3) B1B(0-3) B1E(0-3) B1F(0-3) B1E(0-3) B1F(0-3)
                    const __m512i rhs_mat_014589CD_11_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_11, (_MM_PERM_ENUM)136); //B10(8-11) B11(8-11) B10(8-11) B11(8-11) B14(8-11) B15(8-11) B14(8-11) B15(8-11) B18(8-11) B19(8-11) B18(8-11) B19(8-11) B1C(8-11) B1D(8-11) B1C(8-11) B1D(8-11)
                    const __m512i rhs_mat_2367ABEF_11_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_11, (_MM_PERM_ENUM)136); //B12(8-11) B13(8-11) B12(8-11) B13(8-11) B16(8-11) B17(8-11) B16(8-11) B17(8-11) B1A(8-11) B1B(8-11) B1A(8-11) B1B(8-11) B1E(8-11) B1F(8-11) B1E(8-11) B1F(8-11)
                    const __m512i rhs_mat_014589CD_12_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_12, (_MM_PERM_ENUM)136); //B10(16-19) B11(16-19) B10(16-19) B11(16-19) B14(16-19) B15(16-19) B14(16-19) B15(16-19) B18(16-19) B19(16-19) B18(16-19) B19(16-19) B1C(16-19) B1D(16-19) B1C(16-19) B1D(16-19)
                    const __m512i rhs_mat_2367ABEF_12_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_12, (_MM_PERM_ENUM)136); //B12(16-19) B13(16-19) B12(16-19) B13(16-19) B16(16-19) B17(16-19) B16(16-19) B17(16-19) B1A(16-19) B1B(16-19) B1A(16-19) B1B(16-19) B1E(16-19) B1F(16-19) B1E(16-19) B1F(16-19)
                    const __m512i rhs_mat_014589CD_13_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_13, (_MM_PERM_ENUM)136); //B10(24-27) B11(24-27) B10(24-27) B11(24-27) B14(24-27) B15(24-27) B14(24-27) B15(24-27) B18(24-27) B19(24-27) B18(24-27) B19(24-27) B1C(24-27) B1D(24-27) B1C(24-27) B1D(24-27)
                    const __m512i rhs_mat_2367ABEF_13_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_13, (_MM_PERM_ENUM)136); //B12(24-27) B13(24-27) B12(24-27) B13(24-27) B16(24-27) B17(24-27) B16(24-27) B17(24-27) B1A(24-27) B1B(24-27) B1A(24-27) B1B(24-27) B1E(24-27) B1F(24-27) B1E(24-27) B1F(24-27)

                    // Shuffle pattern two - right side input
                    const __m512i rhs_mat_014589CD_00_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_00, (_MM_PERM_ENUM)221); //B00(4-7) B01(4-7) B00(4-7) B01(4-7) B04(4-7) B05(4-7) B04(4-7) B05(4-7) B08(4-7) B09(4-7) B08(4-7) B09(4-7) B0C(4-7) B0D(4-7) B0C(4-7) B0D(4-7)
                    const __m512i rhs_mat_2367ABEF_00_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_00, (_MM_PERM_ENUM)221); //B02(4-7) B03(4-7) B02(4-7) B03(4-7) B06(4-7) B07(4-7) B06(4-7) B07(4-7) B0A(4-7) B0B(4-7) B0A(4-7) B0B(4-7) B0E(4-7) B0F(4-7) B0E(4-7) B0F(4-7)
                    const __m512i rhs_mat_014589CD_01_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_01, (_MM_PERM_ENUM)221); //B00(12-15) B01(12-15) B00(12-15) B01(12-15) B04(12-15) B05(12-15) B04(12-15) B05(12-15) B08(12-15) B09(12-15) B08(12-15) B09(12-15) B0C(12-15) B0D(12-15) B0C(12-15) B0D(12-15)
                    const __m512i rhs_mat_2367ABEF_01_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_01, (_MM_PERM_ENUM)221); //B02(12-15) B03(12-15) B02(12-15) B03(12-15) B06(12-15) B07(12-15) B06(12-15) B07(12-15) B0A(12-15) B0B(12-15) B0A(12-15) B0B(12-15) B0E(12-15) B0F(12-15) B0E(12-15) B0F(12-15)
                    const __m512i rhs_mat_014589CD_02_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_02, (_MM_PERM_ENUM)221); //B00(20-23) B01(20-23) B00(20-23) B01(20-23) B04(20-23) B05(20-23) B04(20-23) B05(20-23) B08(20-23) B09(20-23) B08(20-23) B09(20-23) B0C(20-23) B0D(20-23) B0C(20-23) B0D(20-23)
                    const __m512i rhs_mat_2367ABEF_02_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_02, (_MM_PERM_ENUM)221); //B02(20-23) B03(20-23) B02(20-23) B03(20-23) B06(20-23) B07(20-23) B06(20-23) B07(20-23) B0A(20-23) B0B(20-23) B0A(20-23) B0B(20-23) B0E(20-23) B0F(20-23) B0E(20-23) B0F(20-23)
                    const __m512i rhs_mat_014589CD_03_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_03, (_MM_PERM_ENUM)221); //B00(28-31) B01(28-31) B00(28-31) B01(28-31) B04(28-31) B05(28-31) B04(28-31) B05(28-31) B08(28-31) B09(28-31) B08(28-31) B09(28-31) B0C(28-31) B0D(28-31) B0C(28-31) 0BD(28-31)
                    const __m512i rhs_mat_2367ABEF_03_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_03, (_MM_PERM_ENUM)221); //B02(28-31) B03(28-31) B02(28-31) B03(28-31) B06(28-31) B07(28-31) B06(28-31) B07(28-31) B0A(28-31) B0B(28-31) B0A(28-31) B0B(28-31) B0E(28-31) B0F(28-31) B0E(28-31) B0F(28-31)

                    const __m512i rhs_mat_014589CD_10_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_10, (_MM_PERM_ENUM)221); //B10(4-7) B11(4-7) B10(4-7) B11(4-7) B14(4-7) B15(4-7) B14(4-7) B15(4-7) B18(4-7) B19(4-7) B18(4-7) B19(4-7) B1C(4-7) B1D(4-7) B1C(4-7) B1D(4-7)
                    const __m512i rhs_mat_2367ABEF_10_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_10, (_MM_PERM_ENUM)221); //B12(4-7) B13(4-7) B12(4-7) B13(4-7) B16(4-7) B17(4-7) B16(4-7) B17(4-7) B1A(4-7) B1B(4-7) B1A(4-7) B1B(4-7) B1E(4-7) B1F(4-7) B1E(4-7) B1F(4-7)
                    const __m512i rhs_mat_014589CD_11_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_11, (_MM_PERM_ENUM)221); //B10(12-15) B11(12-15) B10(12-15) B11(12-15) B14(12-15) B15(12-15) B14(12-15) B15(12-15) B18(12-15) B19(12-15) B18(12-15) B19(12-15) B1C(12-15) B1D(12-15) B1C(12-15) B1D(12-15)
                    const __m512i rhs_mat_2367ABEF_11_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_11, (_MM_PERM_ENUM)221); //B12(12-15) B13(12-15) B12(12-15) B13(12-15) B16(12-15) B17(12-15) B16(12-15) B17(12-15) B1A(12-15) B1B(12-15) B1A(12-15) B1B(12-15) B1E(12-15) B1F(12-15) B1E(12-15) B1F(12-15)
                    const __m512i rhs_mat_014589CD_12_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_12, (_MM_PERM_ENUM)221); //B10(20-23) B11(20-23) B10(20-23) B11(20-23) B14(20-23) B15(20-23) B14(20-23) B15(20-23) B18(20-23) B19(20-23) B18(20-23) B19(20-23) B1C(20-23) B1D(20-23) B1C(20-23) B1D(20-23)
                    const __m512i rhs_mat_2367ABEF_12_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_12, (_MM_PERM_ENUM)221); //B12(20-23) B13(20-23) B12(20-23) B13(20-23) B16(20-23) B17(20-23) B16(20-23) B17(20-23) B1A(20-23) B1B(20-23) B1A(20-23) B1B(20-23) B1E(20-23) B1F(20-23) B1E(20-23) B1F(20-23)
                    const __m512i rhs_mat_014589CD_13_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_13, (_MM_PERM_ENUM)221); //B10(28-31) B11(28-31) B10(28-31) B11(28-31) B14(28-31) B15(28-31) B14(28-31) B15(28-31) B18(28-31) B19(28-31) B18(28-31) B19(28-31) B1C(28-31) B1D(28-31) B1C(28-31) B1D(28-31)
                    const __m512i rhs_mat_2367ABEF_13_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_13, (_MM_PERM_ENUM)221); //B12(28-31) B13(28-31) B12(28-31) B13(28-31) B16(28-31) B17(28-31) B16(28-31) B17(28-31) B1A(28-31) B1B(28-31) B1A(28-31) B1B(28-31) B1E(28-31) B1F(28-31) B1E(28-31) B1F(28-31)

                    uint32_t utmp_00[4], utmp_01[4], utmp_10[4], utmp_11[4];

                    // Scales and Mins of corresponding sub blocks from different Q4_K structures are stored together
                    // The below block is for eg to extract first sub block's scales and mins from different Q4_K structures for the sb loop
                    memcpy(utmp_00, b_ptr_0[b].scales + 24 * sb, 12);
                    utmp_00[3] = ((utmp_00[2] >> 4) & kmask2) | (((utmp_00[1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_00 = utmp_00[1] & kmask1;
                    utmp_00[1] = (utmp_00[2] & kmask2) | (((utmp_00[0] >> 6) & kmask3) << 4);
                    utmp_00[2] = uaux_00;
                    utmp_00[0] &= kmask1;

                    // The below block is for eg to extract second sub block's scales and mins from different Q4_K structures for the sb loop
                    memcpy(utmp_01, b_ptr_0[b].scales + 12 + sb * 24, 12);
                    utmp_01[3] = ((utmp_01[2] >> 4) & kmask2) | (((utmp_01[1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_01 = utmp_01[1] & kmask1;
                    utmp_01[1] = (utmp_01[2] & kmask2) | (((utmp_01[0] >> 6) & kmask3) << 4);
                    utmp_01[2] = uaux_01;
                    utmp_01[0] &= kmask1;

                    // The below block is for eg to extract first sub block's scales and mins from different Q4_K structures for the sb loop
                    memcpy(utmp_10, b_ptr_1[b].scales + sb * 24, 12);
                    utmp_10[3] = ((utmp_10[2] >> 4) & kmask2) | (((utmp_10[1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_10 = utmp_10[1] & kmask1;
                    utmp_10[1] = (utmp_10[2] & kmask2) | (((utmp_10[0] >> 6) & kmask3) << 4);
                    utmp_10[2] = uaux_10;
                    utmp_10[0] &= kmask1;

                    // The below block is for eg to extract second sub block's scales and mins from different Q4_K structures for the sb loop
                    memcpy(utmp_11, b_ptr_1[b].scales + 12 + sb * 24, 12);
                    utmp_11[3] = ((utmp_11[2] >> 4) & kmask2) | (((utmp_11[1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_11 = utmp_11[1] & kmask1;
                    utmp_11[1] = (utmp_11[2] & kmask2) | (((utmp_11[0] >> 6) & kmask3) << 4);
                    utmp_11[2] = uaux_11;
                    utmp_11[0] &= kmask1;

                    // Scales of first sub block in the sb loop
                    const __m256i mins_and_scales_0 = _mm256_set_epi32(utmp_10[3], utmp_10[2], utmp_10[1], utmp_10[0], utmp_00[3], utmp_00[2], utmp_00[1], utmp_00[0]);
                    const __m512i scales_0 = _mm512_cvtepu8_epi16(_mm256_unpacklo_epi8(mins_and_scales_0, mins_and_scales_0));

                    // Scales of second sub block in the sb loop
                    const __m256i mins_and_scales_1 = _mm256_set_epi32(utmp_11[3], utmp_11[2], utmp_11[1], utmp_11[0], utmp_01[3], utmp_01[2], utmp_01[1], utmp_01[0]);
                    const __m512i scales_1 = _mm512_cvtepu8_epi16(_mm256_unpacklo_epi8(mins_and_scales_1, mins_and_scales_1));

                    // Mins of first and second sub block of Q4_K block are arranged side by side
                    const __m512i mins_01 = _mm512_cvtepu8_epi16(_mm256_unpacklo_epi8(_mm256_shuffle_epi32(mins_and_scales_0, 78), _mm256_shuffle_epi32(mins_and_scales_1, 78)));

                    const __m512i scale_014589CD_0 = _mm512_shuffle_epi32(scales_0, (_MM_PERM_ENUM)68);
                    const __m512i scale_2367ABEF_0 = _mm512_shuffle_epi32(scales_0, (_MM_PERM_ENUM)238);

                    const __m512i scale_014589CD_1 = _mm512_shuffle_epi32(scales_1, (_MM_PERM_ENUM)68);
                    const __m512i scale_2367ABEF_1 = _mm512_shuffle_epi32(scales_1, (_MM_PERM_ENUM)238);

                    // Load the four block_q8_k quantized values interleaved with each other in chunks of eight bytes - A0,A1,A2,A3
                    // Loaded as set of 128 bit vectors and repeated into a 256 bit vector
                    __m256i lhs_mat_ymm_0123_00 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 256 * sb)));
                    __m256i lhs_mat_ymm_01_00 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_00, lhs_mat_ymm_0123_00, 0);
                    __m256i lhs_mat_ymm_23_00 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_00, lhs_mat_ymm_0123_00, 17);
                    __m256i lhs_mat_ymm_0123_01 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 32 + 256 * sb)));
                    __m256i lhs_mat_ymm_01_01 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_01, lhs_mat_ymm_0123_01, 0);
                    __m256i lhs_mat_ymm_23_01 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_01, lhs_mat_ymm_0123_01, 17);
                    __m256i lhs_mat_ymm_0123_02 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 64 + 256 * sb)));
                    __m256i lhs_mat_ymm_01_02 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_02, lhs_mat_ymm_0123_02, 0);
                    __m256i lhs_mat_ymm_23_02 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_02, lhs_mat_ymm_0123_02, 17);
                    __m256i lhs_mat_ymm_0123_03 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 96 + 256 * sb)));
                    __m256i lhs_mat_ymm_01_03 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_03, lhs_mat_ymm_0123_03, 0);
                    __m256i lhs_mat_ymm_23_03 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_03, lhs_mat_ymm_0123_03, 17);
                    __m256i lhs_mat_ymm_0123_10 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 128 + 256 * sb)));
                    __m256i lhs_mat_ymm_01_10 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_10, lhs_mat_ymm_0123_10, 0);
                    __m256i lhs_mat_ymm_23_10 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_10, lhs_mat_ymm_0123_10, 17);
                    __m256i lhs_mat_ymm_0123_11 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 160 + 256 * sb)));
                    __m256i lhs_mat_ymm_01_11 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_11, lhs_mat_ymm_0123_11, 0);
                    __m256i lhs_mat_ymm_23_11 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_11, lhs_mat_ymm_0123_11, 17);
                    __m256i lhs_mat_ymm_0123_12 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 192 + 256 * sb)));
                    __m256i lhs_mat_ymm_01_12 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_12, lhs_mat_ymm_0123_12, 0);
                    __m256i lhs_mat_ymm_23_12 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_12, lhs_mat_ymm_0123_12, 17);
                    __m256i lhs_mat_ymm_0123_13 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 224 + 256 * sb)));
                    __m256i lhs_mat_ymm_01_13 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_13, lhs_mat_ymm_0123_13, 0);
                    __m256i lhs_mat_ymm_23_13 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_13, lhs_mat_ymm_0123_13, 17);

                    //Loaded as set of 128 bit vectors and repeated and stored into a 256 bit vector before again repeating into a 512 bit vector
                    __m512i lhs_mat_01_00 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_00), lhs_mat_ymm_01_00, 1);
                    __m512i lhs_mat_23_00 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_00), lhs_mat_ymm_23_00, 1);
                    __m512i lhs_mat_01_01 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_01), lhs_mat_ymm_01_01, 1);
                    __m512i lhs_mat_23_01 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_01), lhs_mat_ymm_23_01, 1);
                    __m512i lhs_mat_01_02 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_02), lhs_mat_ymm_01_02, 1);
                    __m512i lhs_mat_23_02 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_02), lhs_mat_ymm_23_02, 1);
                    __m512i lhs_mat_01_03 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_03), lhs_mat_ymm_01_03, 1);
                    __m512i lhs_mat_23_03 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_03), lhs_mat_ymm_23_03, 1);

                    __m512i lhs_mat_01_10 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_10), lhs_mat_ymm_01_10, 1);
                    __m512i lhs_mat_23_10 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_10), lhs_mat_ymm_23_10, 1);
                    __m512i lhs_mat_01_11 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_11), lhs_mat_ymm_01_11, 1);
                    __m512i lhs_mat_23_11 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_11), lhs_mat_ymm_23_11, 1);
                    __m512i lhs_mat_01_12 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_12), lhs_mat_ymm_01_12, 1);
                    __m512i lhs_mat_23_12 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_12), lhs_mat_ymm_23_12, 1);
                    __m512i lhs_mat_01_13 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_13), lhs_mat_ymm_01_13, 1);
                    __m512i lhs_mat_23_13 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_13), lhs_mat_ymm_23_13, 1);

                    // Bsums are loaded - four bsums are loaded (for two sub blocks) for the different Q8_K blocks
                    __m256i lhs_bsums_0123_01 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].bsums + 16 * sb)));
                    __m256i lhs_bsums_hsum_ymm_0123_01 = _mm256_castsi128_si256(_mm_hadd_epi16(_mm256_castsi256_si128(lhs_bsums_0123_01), _mm256_extractf128_si256(lhs_bsums_0123_01, 1)));
                    lhs_bsums_hsum_ymm_0123_01 = _mm256_permute2x128_si256(lhs_bsums_hsum_ymm_0123_01, lhs_bsums_hsum_ymm_0123_01, 0);
                    __m512i lhs_bsums_hsum_0123_01 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_bsums_hsum_ymm_0123_01), lhs_bsums_hsum_ymm_0123_01, 1);

                    // Shuffle pattern one - left side input
                    const __m512i lhs_mat_01_00_sp1 = _mm512_shuffle_epi32(lhs_mat_01_00, (_MM_PERM_ENUM)160); //A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3) A00(0-3) A01(0-3) A01(0-3)
                    const __m512i lhs_mat_23_00_sp1 = _mm512_shuffle_epi32(lhs_mat_23_00, (_MM_PERM_ENUM)160); //A02(0-3) A02(0-3) A03(0-3) A03(0-3) A02(0-3) A02(0-3) A03(0-3) A03(0-3) A02(0-3) A02(0-3) A03(0-3) A03(0-3) A02(0-3) A02(0-3) A03(0-3) A03(0-3)
                    const __m512i lhs_mat_01_01_sp1 = _mm512_shuffle_epi32(lhs_mat_01_01, (_MM_PERM_ENUM)160); //A00(8-11) A00(8-11) A01(8-11) A01(8-11) A00(8-11) A00(8-11) A01(8-11) A01(8-11) A00(8-11) A00(8-11) A01(8-11) A01(8-11) A00(8-11) A00(8-11) A01(8-11) A01(8-11)
                    const __m512i lhs_mat_23_01_sp1 = _mm512_shuffle_epi32(lhs_mat_23_01, (_MM_PERM_ENUM)160); //A02(8-11) A02(8-11) A03(8-11) A03(8-11) A02(8-11) A02(8-11) A03(8-11) A03(8-11) A02(8-11) A02(8-11) A03(8-11) A03(8-11) A02(8-11) A02(8-11) A03(8-11) A03(8-11)
                    const __m512i lhs_mat_01_02_sp1 = _mm512_shuffle_epi32(lhs_mat_01_02, (_MM_PERM_ENUM)160); //A00(16-19) A00(16-19) A01(16-19) A01(16-19) A00(16-19) A00(16-19) A01(16-19) A01(16-19) A00(16-19) A00(16-19) A01(16-19) A01(16-19) A00(16-19) A00(16-19) A01(16-19) A01(16-19)
                    const __m512i lhs_mat_23_02_sp1 = _mm512_shuffle_epi32(lhs_mat_23_02, (_MM_PERM_ENUM)160); //A02(16-19) A02(16-19) A03(16-19) A03(16-19) A02(16-19) A02(16-19) A03(16-19) A03(16-19) A02(16-19) A02(16-19) A03(16-19) A03(16-19) A02(16-19) A02(16-19) A03(16-19) A03(16-19)
                    const __m512i lhs_mat_01_03_sp1 = _mm512_shuffle_epi32(lhs_mat_01_03, (_MM_PERM_ENUM)160); //A00(24-27) A00(24-27) A01(24-27) A01(24-27) A00(24-27) A00(24-27) A01(24-27) A01(24-27) A00(24-27) A00(24-27) A01(24-27) A01(24-27) A00(24-27) A00(24-27) A01(24-27) A01(24-27)
                    const __m512i lhs_mat_23_03_sp1 = _mm512_shuffle_epi32(lhs_mat_23_03, (_MM_PERM_ENUM)160); //A02(24-27) A02(24-27) A03(24-27) A03(24-27) A02(24-27) A02(24-27) A03(24-27) A03(24-27) A02(24-27) A02(24-27) A03(24-27) A03(24-27) A02(24-27) A02(24-27) A03(24-27) A03(24-27)

                    const __m512i lhs_mat_01_10_sp1 = _mm512_shuffle_epi32(lhs_mat_01_10, (_MM_PERM_ENUM)160); //A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3) A10(0-3) A11(0-3) A11(0-3)
                    const __m512i lhs_mat_23_10_sp1 = _mm512_shuffle_epi32(lhs_mat_23_10, (_MM_PERM_ENUM)160); //A12(0-3) A12(0-3) A13(0-3) A13(0-3) A12(0-3) A12(0-3) A13(0-3) A13(0-3) A12(0-3) A12(0-3) A13(0-3) A13(0-3) A12(0-3) A12(0-3) A13(0-3) A13(0-3)
                    const __m512i lhs_mat_01_11_sp1 = _mm512_shuffle_epi32(lhs_mat_01_11, (_MM_PERM_ENUM)160); //A10(8-11) A10(8-11) A11(8-11) A11(8-11) A10(8-11) A10(8-11) A11(8-11) A11(8-11) A10(8-11) A10(8-11) A11(8-11) A11(8-11) A10(8-11) A10(8-11) A11(8-11) A11(8-11)
                    const __m512i lhs_mat_23_11_sp1 = _mm512_shuffle_epi32(lhs_mat_23_11, (_MM_PERM_ENUM)160); //A12(8-11) A12(8-11) A13(8-11) A13(8-11) A12(8-11) A12(8-11) A13(8-11) A13(8-11) A12(8-11) A12(8-11) A13(8-11) A13(8-11) A12(8-11) A12(8-11) A13(8-11) A13(8-11)
                    const __m512i lhs_mat_01_12_sp1 = _mm512_shuffle_epi32(lhs_mat_01_12, (_MM_PERM_ENUM)160); //A10(16-19) A10(16-19) A11(16-19) A11(16-19) A10(16-19) A10(16-19) A11(16-19) A11(16-19) A10(16-19) A10(16-19) A11(16-19) A11(16-19) A10(16-19) A10(16-19) A11(16-19) A11(16-19)
                    const __m512i lhs_mat_23_12_sp1 = _mm512_shuffle_epi32(lhs_mat_23_12, (_MM_PERM_ENUM)160); //A12(16-19) A12(16-19) A13(16-19) A13(16-19) A12(16-19) A12(16-19) A13(16-19) A13(16-19) A12(16-19) A12(16-19) A13(16-19) A13(16-19) A12(16-19) A12(16-19) A13(16-19) A13(16-19)
                    const __m512i lhs_mat_01_13_sp1 = _mm512_shuffle_epi32(lhs_mat_01_13, (_MM_PERM_ENUM)160); //A10(24-27) A10(24-27) A11(24-27) A11(24-27) A10(24-27) A10(24-27) A11(24-27) A11(24-27) A10(24-27) A10(24-27) A11(24-27) A11(24-27) A10(24-27) A10(24-27) A11(24-27) A11(24-27)
                    const __m512i lhs_mat_23_13_sp1 = _mm512_shuffle_epi32(lhs_mat_23_13, (_MM_PERM_ENUM)160); //A12(24-27) A12(24-27) A13(24-27) A13(24-27) A12(24-27) A12(24-27) A13(24-27) A13(24-27) A12(24-27) A12(24-27) A13(24-27) A13(24-27) A12(24-27) A12(24-27) A13(24-27) A13(24-27)

                    const __m512i lhs_mat_01_00_sp2 = _mm512_shuffle_epi32(lhs_mat_01_00, (_MM_PERM_ENUM)245); //A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7) A00(4-7) A01(4-7) A01(4-7)
                    const __m512i lhs_mat_23_00_sp2 = _mm512_shuffle_epi32(lhs_mat_23_00, (_MM_PERM_ENUM)245); //A02(4-7) A02(4-7) A03(4-7) A03(4-7) A02(4-7) A02(4-7) A03(4-7) A03(4-7) A02(4-7) A02(4-7) A03(4-7) A03(4-7) A02(4-7) A02(4-7) A03(4-7) A03(4-7)
                    const __m512i lhs_mat_01_01_sp2 = _mm512_shuffle_epi32(lhs_mat_01_01, (_MM_PERM_ENUM)245); //A00(12-15) A00(12-15) A01(12-15) A01(12-15) A00(12-15) A00(12-15) A01(12-15) A01(12-15) A00(12-15) A00(12-15) A01(12-15) A01(12-15) A00(12-15) A00(12-15) A01(12-15) A01(12-15)
                    const __m512i lhs_mat_23_01_sp2 = _mm512_shuffle_epi32(lhs_mat_23_01, (_MM_PERM_ENUM)245); //A02(12-15) A02(12-15) A03(12-15) A03(12-15) A02(12-15) A02(12-15) A03(12-15) A03(12-15) A02(12-15) A02(12-15) A03(12-15) A03(12-15) A02(12-15) A02(12-15) A03(12-15) A03(12-15)
                    const __m512i lhs_mat_01_02_sp2 = _mm512_shuffle_epi32(lhs_mat_01_02, (_MM_PERM_ENUM)245); //A00(20-23) A00(20-23) A01(20-23) A01(20-23) A00(20-23) A00(20-23) A01(20-23) A01(20-23) A00(20-23) A00(20-23) A01(20-23) A01(20-23) A00(20-23) A00(20-23) A01(20-23) A01(20-23)
                    const __m512i lhs_mat_23_02_sp2 = _mm512_shuffle_epi32(lhs_mat_23_02, (_MM_PERM_ENUM)245); //A02(20-23) A02(20-23) A03(20-23) A03(20-23) A02(20-23) A02(20-23) A03(20-23) A03(20-23) A02(20-23) A02(20-23) A03(20-23) A03(20-23) A02(20-23) A02(20-23) A03(20-23) A03(20-23)
                    const __m512i lhs_mat_01_03_sp2 = _mm512_shuffle_epi32(lhs_mat_01_03, (_MM_PERM_ENUM)245); //A00(28-31) A00(28-31) A01(28-31) A01(28-31) A00(28-31) A00(28-31) A01(28-31) A01(28-31) A00(28-31) A00(28-31) A01(28-31) A01(28-31) A00(28-31) A00(28-31) A01(28-31) A01(28-31)
                    const __m512i lhs_mat_23_03_sp2 = _mm512_shuffle_epi32(lhs_mat_23_03, (_MM_PERM_ENUM)245); //A02(28-31) A02(28-31) A03(28-31) A03(28-31) A02(28-31) A02(28-31) A03(28-31) A03(28-31) A02(28-31) A02(28-31) A03(28-31) A03(28-31) A02(28-31) A02(28-31) A03(28-31) A03(28-31)

                    const __m512i lhs_mat_01_10_sp2 = _mm512_shuffle_epi32(lhs_mat_01_10, (_MM_PERM_ENUM)245); //A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7) A10(4-7) A11(4-7) A11(4-7)
                    const __m512i lhs_mat_23_10_sp2 = _mm512_shuffle_epi32(lhs_mat_23_10, (_MM_PERM_ENUM)245); //A12(4-7) A12(4-7) A13(4-7) A13(4-7) A12(4-7) A12(4-7) A13(4-7) A13(4-7) A12(4-7) A12(4-7) A13(4-7) A13(4-7) A12(4-7) A12(4-7) A13(4-7) A13(4-7)
                    const __m512i lhs_mat_01_11_sp2 = _mm512_shuffle_epi32(lhs_mat_01_11, (_MM_PERM_ENUM)245); //A10(12-15) A10(12-15) A11(12-15) A11(12-15) A10(12-15) A10(12-15) A11(12-15) A11(12-15) A10(12-15) A10(12-15) A11(12-15) A11(12-15) A10(12-15) A10(12-15) A11(12-15) A11(12-15)
                    const __m512i lhs_mat_23_11_sp2 = _mm512_shuffle_epi32(lhs_mat_23_11, (_MM_PERM_ENUM)245); //A12(12-15) A12(12-15) A13(12-15) A13(12-15) A12(12-15) A12(12-15) A13(12-15) A13(12-15) A12(12-15) A12(12-15) A13(12-15) A13(12-15) A12(12-15) A12(12-15) A13(12-15) A13(12-15)
                    const __m512i lhs_mat_01_12_sp2 = _mm512_shuffle_epi32(lhs_mat_01_12, (_MM_PERM_ENUM)245); //A10(20-23) A10(20-23) A11(20-23) A11(20-23) A10(20-23) A10(20-23) A11(20-23) A11(20-23) A10(20-23) A10(20-23) A11(20-23) A11(20-23) A10(20-23) A10(20-23) A11(20-23) A11(20-23)
                    const __m512i lhs_mat_23_12_sp2 = _mm512_shuffle_epi32(lhs_mat_23_12, (_MM_PERM_ENUM)245); //A12(20-23) A12(20-23) A13(20-23) A13(20-23) A12(20-23) A12(20-23) A13(20-23) A13(20-23) A12(20-23) A12(20-23) A13(20-23) A13(20-23) A12(20-23) A12(20-23) A13(20-23) A13(20-23)
                    const __m512i lhs_mat_01_13_sp2 = _mm512_shuffle_epi32(lhs_mat_01_13, (_MM_PERM_ENUM)245); //A10(28-31) A10(28-31) A11(28-31) A11(28-31) A10(28-31) A10(28-31) A11(28-31) A11(28-31) A10(28-31) A10(28-31) A11(28-31) A11(28-31) A10(28-31) A10(28-31) A11(28-31) A11(28-31)
                    const __m512i lhs_mat_23_13_sp2 = _mm512_shuffle_epi32(lhs_mat_23_13, (_MM_PERM_ENUM)245); //A12(28-31) A12(28-31) A13(28-31) A13(28-31) A12(28-31) A12(28-31) A13(28-31) A13(28-31) A12(28-31) A12(28-31) A13(28-31) A13(28-31) A12(28-31) A12(28-31) A13(28-31) A13(28-31)

                    // The values arranged in shuffle patterns are operated with dot product operation within 32 bit lane i.e corresponding bytes and multiplied and added into 32 bit integers within 32 bit lane
                    __m512i iacc_mat_00_0_sp1 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_03_sp1, lhs_mat_01_03_sp1), _mm512_maddubs_epi16(rhs_mat_014589CD_02_sp1, lhs_mat_01_02_sp1)), _mm512_maddubs_epi16(rhs_mat_014589CD_01_sp1, lhs_mat_01_01_sp1)), _mm512_maddubs_epi16(rhs_mat_014589CD_00_sp1, lhs_mat_01_00_sp1));
                    __m512i iacc_mat_01_0_sp1 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_03_sp1, lhs_mat_01_03_sp1), _mm512_maddubs_epi16(rhs_mat_2367ABEF_02_sp1, lhs_mat_01_02_sp1)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_01_sp1, lhs_mat_01_01_sp1)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_00_sp1, lhs_mat_01_00_sp1));
                    __m512i iacc_mat_10_0_sp1 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_03_sp1, lhs_mat_23_03_sp1), _mm512_maddubs_epi16(rhs_mat_014589CD_02_sp1, lhs_mat_23_02_sp1)), _mm512_maddubs_epi16(rhs_mat_014589CD_01_sp1, lhs_mat_23_01_sp1)), _mm512_maddubs_epi16(rhs_mat_014589CD_00_sp1, lhs_mat_23_00_sp1));
                    __m512i iacc_mat_11_0_sp1 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_03_sp1, lhs_mat_23_03_sp1), _mm512_maddubs_epi16(rhs_mat_2367ABEF_02_sp1, lhs_mat_23_02_sp1)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_01_sp1, lhs_mat_23_01_sp1)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_00_sp1, lhs_mat_23_00_sp1));
                    __m512i iacc_mat_00_1_sp1 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_13_sp1, lhs_mat_01_13_sp1), _mm512_maddubs_epi16(rhs_mat_014589CD_12_sp1, lhs_mat_01_12_sp1)), _mm512_maddubs_epi16(rhs_mat_014589CD_11_sp1, lhs_mat_01_11_sp1)), _mm512_maddubs_epi16(rhs_mat_014589CD_10_sp1, lhs_mat_01_10_sp1));
                    __m512i iacc_mat_01_1_sp1 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_13_sp1, lhs_mat_01_13_sp1), _mm512_maddubs_epi16(rhs_mat_2367ABEF_12_sp1, lhs_mat_01_12_sp1)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_11_sp1, lhs_mat_01_11_sp1)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_10_sp1, lhs_mat_01_10_sp1));
                    __m512i iacc_mat_10_1_sp1 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_13_sp1, lhs_mat_23_13_sp1), _mm512_maddubs_epi16(rhs_mat_014589CD_12_sp1, lhs_mat_23_12_sp1)), _mm512_maddubs_epi16(rhs_mat_014589CD_11_sp1, lhs_mat_23_11_sp1)), _mm512_maddubs_epi16(rhs_mat_014589CD_10_sp1, lhs_mat_23_10_sp1));
                    __m512i iacc_mat_11_1_sp1 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_13_sp1, lhs_mat_23_13_sp1), _mm512_maddubs_epi16(rhs_mat_2367ABEF_12_sp1, lhs_mat_23_12_sp1)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_11_sp1, lhs_mat_23_11_sp1)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_10_sp1, lhs_mat_23_10_sp1));

                    __m512i iacc_mat_00_0_sp2 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_03_sp2, lhs_mat_01_03_sp2), _mm512_maddubs_epi16(rhs_mat_014589CD_02_sp2, lhs_mat_01_02_sp2)), _mm512_maddubs_epi16(rhs_mat_014589CD_01_sp2, lhs_mat_01_01_sp2)), _mm512_maddubs_epi16(rhs_mat_014589CD_00_sp2, lhs_mat_01_00_sp2));
                    __m512i iacc_mat_01_0_sp2 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_03_sp2, lhs_mat_01_03_sp2), _mm512_maddubs_epi16(rhs_mat_2367ABEF_02_sp2, lhs_mat_01_02_sp2)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_01_sp2, lhs_mat_01_01_sp2)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_00_sp2, lhs_mat_01_00_sp2));
                    __m512i iacc_mat_10_0_sp2 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_03_sp2, lhs_mat_23_03_sp2), _mm512_maddubs_epi16(rhs_mat_014589CD_02_sp2, lhs_mat_23_02_sp2)), _mm512_maddubs_epi16(rhs_mat_014589CD_01_sp2, lhs_mat_23_01_sp2)), _mm512_maddubs_epi16(rhs_mat_014589CD_00_sp2, lhs_mat_23_00_sp2));
                    __m512i iacc_mat_11_0_sp2 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_03_sp2, lhs_mat_23_03_sp2), _mm512_maddubs_epi16(rhs_mat_2367ABEF_02_sp2, lhs_mat_23_02_sp2)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_01_sp2, lhs_mat_23_01_sp2)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_00_sp2, lhs_mat_23_00_sp2));
                    __m512i iacc_mat_00_1_sp2 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_13_sp2, lhs_mat_01_13_sp2), _mm512_maddubs_epi16(rhs_mat_014589CD_12_sp2, lhs_mat_01_12_sp2)), _mm512_maddubs_epi16(rhs_mat_014589CD_11_sp2, lhs_mat_01_11_sp2)), _mm512_maddubs_epi16(rhs_mat_014589CD_10_sp2, lhs_mat_01_10_sp2));
                    __m512i iacc_mat_01_1_sp2 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_13_sp2, lhs_mat_01_13_sp2), _mm512_maddubs_epi16(rhs_mat_2367ABEF_12_sp2, lhs_mat_01_12_sp2)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_11_sp2, lhs_mat_01_11_sp2)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_10_sp2, lhs_mat_01_10_sp2));
                    __m512i iacc_mat_10_1_sp2 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_13_sp2, lhs_mat_23_13_sp2), _mm512_maddubs_epi16(rhs_mat_014589CD_12_sp2, lhs_mat_23_12_sp2)), _mm512_maddubs_epi16(rhs_mat_014589CD_11_sp2, lhs_mat_23_11_sp2)), _mm512_maddubs_epi16(rhs_mat_014589CD_10_sp2, lhs_mat_23_10_sp2));
                    __m512i iacc_mat_11_1_sp2 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_13_sp2, lhs_mat_23_13_sp2), _mm512_maddubs_epi16(rhs_mat_2367ABEF_12_sp2, lhs_mat_23_12_sp2)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_11_sp2, lhs_mat_23_11_sp2)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_10_sp2, lhs_mat_23_10_sp2));

                    // Output of both shuffle patterns are added in order to sum dot product outputs of all 32 values in block
                    __m512i iacc_mat_00_0 = _mm512_add_epi16(iacc_mat_00_0_sp1, iacc_mat_00_0_sp2);
                    __m512i iacc_mat_01_0 = _mm512_add_epi16(iacc_mat_01_0_sp1, iacc_mat_01_0_sp2);
                    __m512i iacc_mat_10_0 = _mm512_add_epi16(iacc_mat_10_0_sp1, iacc_mat_10_0_sp2);
                    __m512i iacc_mat_11_0 = _mm512_add_epi16(iacc_mat_11_0_sp1, iacc_mat_11_0_sp2);

                    __m512i iacc_mat_00_1 = _mm512_add_epi16(iacc_mat_00_1_sp1, iacc_mat_00_1_sp2);
                    __m512i iacc_mat_01_1 = _mm512_add_epi16(iacc_mat_01_1_sp1, iacc_mat_01_1_sp2);
                    __m512i iacc_mat_10_1 = _mm512_add_epi16(iacc_mat_10_1_sp1, iacc_mat_10_1_sp2);
                    __m512i iacc_mat_11_1 = _mm512_add_epi16(iacc_mat_11_1_sp1, iacc_mat_11_1_sp2);

                    iacc_mat_00_0 = _mm512_madd_epi16(iacc_mat_00_0, scale_014589CD_0);
                    iacc_mat_01_0 = _mm512_madd_epi16(iacc_mat_01_0, scale_2367ABEF_0);
                    iacc_mat_10_0 = _mm512_madd_epi16(iacc_mat_10_0, scale_014589CD_0);
                    iacc_mat_11_0 = _mm512_madd_epi16(iacc_mat_11_0, scale_2367ABEF_0);

                    iacc_mat_00_1 = _mm512_madd_epi16(iacc_mat_00_1, scale_014589CD_1);
                    iacc_mat_01_1 = _mm512_madd_epi16(iacc_mat_01_1, scale_2367ABEF_1);
                    iacc_mat_10_1 = _mm512_madd_epi16(iacc_mat_10_1, scale_014589CD_1);
                    iacc_mat_11_1 = _mm512_madd_epi16(iacc_mat_11_1, scale_2367ABEF_1);

                    // Straighten out to make 4 row vectors (4 for each sub block which are accumulated together in the next step)
                    __m512i iacc_row_0_0 = _mm512_mask_blend_epi32(0xCCCC, iacc_mat_00_0, _mm512_shuffle_epi32(iacc_mat_01_0, (_MM_PERM_ENUM)78));
                    __m512i iacc_row_1_0 = _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32(iacc_mat_00_0, (_MM_PERM_ENUM)78), iacc_mat_01_0);
                    __m512i iacc_row_2_0 = _mm512_mask_blend_epi32(0xCCCC, iacc_mat_10_0, _mm512_shuffle_epi32(iacc_mat_11_0, (_MM_PERM_ENUM)78));
                    __m512i iacc_row_3_0 = _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32(iacc_mat_10_0, (_MM_PERM_ENUM)78), iacc_mat_11_0);
                    __m512i iacc_row_0_1 = _mm512_mask_blend_epi32(0xCCCC, iacc_mat_00_1, _mm512_shuffle_epi32(iacc_mat_01_1, (_MM_PERM_ENUM)78));
                    __m512i iacc_row_1_1 = _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32(iacc_mat_00_1, (_MM_PERM_ENUM)78), iacc_mat_01_1);
                    __m512i iacc_row_2_1 = _mm512_mask_blend_epi32(0xCCCC, iacc_mat_10_1, _mm512_shuffle_epi32(iacc_mat_11_1, (_MM_PERM_ENUM)78));
                    __m512i iacc_row_3_1 = _mm512_mask_blend_epi32(0xCCCC,_mm512_shuffle_epi32(iacc_mat_10_1, (_MM_PERM_ENUM)78), iacc_mat_11_1);

                    __m512i iacc_row_0 = _mm512_add_epi32(iacc_row_0_0, iacc_row_0_1);
                    __m512i iacc_row_1 = _mm512_add_epi32(iacc_row_1_0, iacc_row_1_1);
                    __m512i iacc_row_2 = _mm512_add_epi32(iacc_row_2_0, iacc_row_2_1);
                    __m512i iacc_row_3 = _mm512_add_epi32(iacc_row_3_0, iacc_row_3_1);

                    // Load the scale(d) values for all the 4 Q8_k blocks and repeat it across lanes
                    const __m128 row_scale_f32_sse = _mm_load_ps(a_ptr[b].d);
                    const __m256 row_scale_f32_ymm = _mm256_set_m128(row_scale_f32_sse, row_scale_f32_sse);
                    const __m512 row_scale_f32 = _mm512_insertf32x8(_mm512_castps256_ps512(row_scale_f32_ymm), row_scale_f32_ymm, 1);

                    // Multiply with appropriate scales and accumulate (for both d and dmin) below
                    acc_rows[0] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_0), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 0)), acc_rows[0]);
                    acc_rows[1] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_1), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 85)), acc_rows[1]);
                    acc_rows[2] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_2), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_rows[2]);
                    acc_rows[3] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_3), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_rows[3]);

                    __m512i iacc_row_min_0 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_hsum_0123_01, (_MM_PERM_ENUM)0), mins_01);
                    __m512i iacc_row_min_1 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_hsum_0123_01, (_MM_PERM_ENUM)85), mins_01);
                    __m512i iacc_row_min_2 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_hsum_0123_01, (_MM_PERM_ENUM)170), mins_01);
                    __m512i iacc_row_min_3 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_hsum_0123_01, (_MM_PERM_ENUM)255), mins_01);

                    acc_min_rows[0] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_min_0), _mm512_mul_ps(col_dmin_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 0)), acc_min_rows[0]);
                    acc_min_rows[1] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_min_1), _mm512_mul_ps(col_dmin_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 85)), acc_min_rows[1]);
                    acc_min_rows[2] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_min_2), _mm512_mul_ps(col_dmin_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_min_rows[2]);
                    acc_min_rows[3] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_min_3), _mm512_mul_ps(col_dmin_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_min_rows[3]);
                }
            }
            // Store accumulated values
            for (int i = 0; i < 4; i++) {
                _mm512_storeu_ps((float * )(s + ((y * 4 + i) * bs + x * 8)), _mm512_sub_ps(acc_rows[i], acc_min_rows[i]));
            }
        }
    }
    if (anc != nc) {
        xstart = anc/8;
        y = 0;
    }
#endif // __AVX512BW__ && __AVX512DQ__

    // Take group of four block_q8_Kx4 structures at each pass of the loop and perform dot product operation
    for (; y < anr / 4; y += 4) {

        const block_q8_Kx4 * a_ptrs[4];

        a_ptrs[0] = a_ptr_start + (y * nb);
        for (int i = 0; i < 3; ++i) {
            a_ptrs[i + 1] = a_ptrs[i] + nb;
        }

        // Take group of eight block_q4_kx8 structures at each pass of the loop and perform dot product operation
        for (int64_t x = xstart; x < nc / 8; x++) {

            const block_q4_Kx8 * b_ptr = b_ptr_start + (x * b_nb);

            // Master FP accumulators
            __m256 acc_rows[16];
            for (int i = 0; i < 16; i++) {
                acc_rows[i] = _mm256_setzero_ps();
            }

            __m256 acc_min_rows[16];
            for (int i = 0; i < 16; i++) {
                acc_min_rows[i] = _mm256_setzero_ps();
            }

            // For super block
            for (int64_t b = 0; b < nb; b++) {

                // Scale values - Load the eight scale values of block_q4_kx8
                const __m256 col_scale_f32 = GGML_F32Cx8_LOAD(b_ptr[b].d);

                // dmin values - Load the eight dmin values of block_q4_kx8
                const __m256 col_dmin_f32 = GGML_F32Cx8_LOAD(b_ptr[b].dmin);

                // Loop to iterate over the eight sub blocks of a super block - two sub blocks are processed per iteration
                for (int sb = 0; sb < QK_K / 64; sb++) {

                    // Load the eight block_q4_K for two sub blocks quantized values interleaved with each other in chunks of eight bytes - B0,B1 ....B6,B7
                    const __m256i rhs_raw_mat_0123_0 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + sb * 256));
                    const __m256i rhs_raw_mat_4567_0 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 32 + sb * 256));
                    const __m256i rhs_raw_mat_0123_1 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 64 + sb * 256));
                    const __m256i rhs_raw_mat_4567_1 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 96 + sb * 256));
                    const __m256i rhs_raw_mat_0123_2 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 128 + sb * 256));
                    const __m256i rhs_raw_mat_4567_2 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 160 + sb * 256));
                    const __m256i rhs_raw_mat_0123_3 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 192 + sb * 256));
                    const __m256i rhs_raw_mat_4567_3 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 224 + sb * 256));

                    // Save the values in the following vectors in the formats B0B1B4B5, B2B3B6B7 for further processing and storing of values
                    const __m256i rhs_raw_mat_0145_0 = _mm256_blend_epi32(rhs_raw_mat_0123_0, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_0, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_0, requiredOrder), rhs_raw_mat_4567_0, 240);
                    const __m256i rhs_raw_mat_0145_1 = _mm256_blend_epi32(rhs_raw_mat_0123_1, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_1, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_1, requiredOrder), rhs_raw_mat_4567_1, 240);
                    const __m256i rhs_raw_mat_0145_2 = _mm256_blend_epi32(rhs_raw_mat_0123_2, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_2, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_2 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_2, requiredOrder), rhs_raw_mat_4567_2, 240);
                    const __m256i rhs_raw_mat_0145_3 = _mm256_blend_epi32(rhs_raw_mat_0123_3, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_3, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_3 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_3, requiredOrder), rhs_raw_mat_4567_3, 240);

                    // 4-bit -> 8-bit
                    // First sub block of the two sub blocks processed in the iteration
                    const __m256i rhs_mat_0145_00 = _mm256_and_si256(rhs_raw_mat_0145_0, m4b); //B00(0-7) B01(0-7) B04(0-7) B05(0-7)
                    const __m256i rhs_mat_2367_00 = _mm256_and_si256(rhs_raw_mat_2367_0, m4b); //B02(0-7) B03(0-7) B06(0-7) B07(0-7)

                    const __m256i rhs_mat_0145_01 = _mm256_and_si256(rhs_raw_mat_0145_1, m4b); //B00(8-15) B01(8-15) B04(8-15) B05(8-15)
                    const __m256i rhs_mat_2367_01 = _mm256_and_si256(rhs_raw_mat_2367_1, m4b); //B02(8-15) B03(8-15) B06(8-15) B07(8-15)

                    const __m256i rhs_mat_0145_02 = _mm256_and_si256(rhs_raw_mat_0145_2, m4b); //B00(16-23) B01(16-23) B04(16-23) B05(16-23)
                    const __m256i rhs_mat_2367_02 = _mm256_and_si256(rhs_raw_mat_2367_2, m4b); //B02(16-23) B03(16-23) B06(16-23) B07(16-23)

                    const __m256i rhs_mat_0145_03 = _mm256_and_si256(rhs_raw_mat_0145_3, m4b); //B00(24-31) B01(24-31) B04(24-31) B05(24-31)
                    const __m256i rhs_mat_2367_03 = _mm256_and_si256(rhs_raw_mat_2367_3, m4b); //B02(24-31) B03(24-31) B06(24-31) B07(24-31)

                    // Second sub block of the two sub blocks processed in the iteration
                    const __m256i rhs_mat_0145_10 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_0, 4), m4b); //B10(0-7) B11(0-7) B14(0-7) B15(0-7)
                    const __m256i rhs_mat_2367_10 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_0, 4), m4b); //B12(0-7) B13(0-7) B16(0-7) B17(0-7)

                    const __m256i rhs_mat_0145_11 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_1, 4), m4b); //B10(8-15) B11(8-15) B14(8-15) B15(8-15)
                    const __m256i rhs_mat_2367_11 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_1, 4), m4b); //B12(8-15) B13(8-15) B16(8-15) B17(8-15)

                    const __m256i rhs_mat_0145_12 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_2, 4), m4b); //B10(16-23) B11(16-23) B14(16-23) B15(16-23)
                    const __m256i rhs_mat_2367_12 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_2, 4), m4b); //B12(16-23) B13(16-23) B16(16-23) B17(16-23)

                    const __m256i rhs_mat_0145_13 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_3, 4), m4b); //B10(24-31) B11(24-31) B14(24-31) B15(24-31)
                    const __m256i rhs_mat_2367_13 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_3, 4), m4b); //B12(24-31) B13(24-31) B16(24-31) B17(24-31)

                    // Shuffle pattern one - right side input
                    const __m256i rhs_mat_0145_00_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_00, 136); //B00(0-3) B01(0-3) B00(0-3) B01(0-3) B04(0-3) B05(0-3) B04(0-3) B05(0-3)
                    const __m256i rhs_mat_2367_00_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_00, 136); //B02(0-3) B03(0-3) B02(0-3) B03(0-3) B06(0-3) B07(0-3) B06(0-3) B07(0-3)

                    const __m256i rhs_mat_0145_01_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_01, 136); //B00(8-11) B01(8-11) B00(8-11) B01(8-11) B04(8-11) B05(8-11) B04(8-11) B05(8-11)
                    const __m256i rhs_mat_2367_01_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_01, 136); //B02(8-11) B03(8-11) B02(8-11) B03(8-11) B06(8-11) B07(8-11) B06(8-11) B07(8-11)

                    const __m256i rhs_mat_0145_02_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_02, 136); //B00(16-19) B01(16-19) B00(16-19) B01(16-19) B04(16-19) B05(16-19) B04(16-19) B05(16-19)
                    const __m256i rhs_mat_2367_02_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_02, 136); //B02(16-19) B03(16-19) B02(16-19) B03(16-19) B06(16-19) B07(16-19) B06(16-19) B07(16-19)

                    const __m256i rhs_mat_0145_03_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_03, 136); //B00(24-27) B01(24-27) B00(24-27) B01(24-27) B04(24-27) B05(24-27) B04(24-27) B05(24-27)
                    const __m256i rhs_mat_2367_03_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_03, 136); //B02(24-27) B03(24-27) B02(24-27) B03(24-27) B06(24-27) B07(24-27) B06(24-27) B07(24-27)

                    const __m256i rhs_mat_0145_10_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_10, 136); //B10(0-3) B11(0-3) B10(0-3) B11(0-3) B14(0-3) B15(0-3) B14(0-3) B15(0-3)
                    const __m256i rhs_mat_2367_10_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_10, 136); //B12(0-3) B13(0-3) B12(0-3) B13(0-3) B16(0-3) B17(0-3) B16(0-3) B17(0-3)

                    const __m256i rhs_mat_0145_11_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_11, 136); //B10(8-11) B11(8-11) B10(8-11) B11(8-11) B14(8-11) B15(8-11) B14(8-11) B15(8-11)
                    const __m256i rhs_mat_2367_11_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_11, 136); //B12(8-11) B13(8-11) B12(8-11) B13(8-11) B16(8-11) B17(8-11) B16(8-11) B17(8-11)

                    const __m256i rhs_mat_0145_12_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_12, 136); //B10(16-19) B11(16-19) B10(16-19) B11(16-19) B14(16-19) B15(16-19) B14(16-19) B15(16-19)
                    const __m256i rhs_mat_2367_12_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_12, 136); //B12(16-19) B13(16-19) B12(16-19) B13(16-19) B16(16-19) B17(16-19) B16(16-19) B17(16-19)

                    const __m256i rhs_mat_0145_13_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_13, 136); //B10(24-27) B11(24-27) B10(24-27) B11(24-27) B14(24-27) B15(24-27) B14(24-27) B15(24-27)
                    const __m256i rhs_mat_2367_13_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_13, 136); //B12(24-27) B13(24-27) B12(24-27) B13(24-27) B16(24-27) B17(24-27) B16(24-27) B17(24-27)


                    // Shuffle pattern two - right side input
                    const __m256i rhs_mat_0145_00_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_00, 221); //B00(4-7) B01(4-7) B00(4-7) B01(4-7) B04(4-7) B05(4-7) B04(4-7) B05(4-7)
                    const __m256i rhs_mat_2367_00_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_00, 221); //B02(4-7) B03(4-7) B02(4-7) B03(4-7) B06(4-7) B07(4-7) B06(4-7) B07(4-7)

                    const __m256i rhs_mat_0145_01_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_01, 221); //B00(12-15) B01(12-15) B00(12-15) B01(12-15) B04(12-15) B05(12-15) B04(12-15) B05(12-15)
                    const __m256i rhs_mat_2367_01_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_01, 221); //B02(12-15) B03(12-15) B02(12-15) B03(12-15) B06(12-15) B07(12-15) B06(12-15) B07(12-15)

                    const __m256i rhs_mat_0145_02_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_02, 221); //B00(20-23) B01(20-23) B00(20-23) B01(20-23) B04(20-23) B05(20-23) B04(20-23) B05(20-23)
                    const __m256i rhs_mat_2367_02_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_02, 221); //B02(20-23) B03(20-23) B02(20-23) B03(20-23) B06(20-23) B07(20-23) B06(20-23) B07(20-23)

                    const __m256i rhs_mat_0145_03_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_03, 221); //B00(28-31) B01(28-31) B00(28-31) B01(28-31) B04(28-31) B05(28-31) B04(28-31) B05(28-31)
                    const __m256i rhs_mat_2367_03_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_03, 221); //B02(28-31) B03(28-31) B02(28-31) B03(28-31) B06(28-31) B07(28-31) B06(28-31) B07(28-31)

                    const __m256i rhs_mat_0145_10_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_10, 221); //B10(4-7) B11(4-7) B10(4-7) B11(4-7) B14(4-7) B15(4-7) B14(4-7) B15(4-7)
                    const __m256i rhs_mat_2367_10_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_10, 221); //B12(4-7) B13(4-7) B12(4-7) B13(4-7) B16(4-7) B17(4-7) B16(4-7) B17(4-7)

                    const __m256i rhs_mat_0145_11_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_11, 221); //B10(12-15) B11(12-15) B10(12-15) B11(12-15) B14(12-15) B15(12-15) B14(12-15) B15(12-15)
                    const __m256i rhs_mat_2367_11_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_11, 221); //B12(12-15) B13(12-15) B12(12-15) B13(12-15) B16(12-15) B17(12-15) B16(12-15) B17(12-15)

                    const __m256i rhs_mat_0145_12_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_12, 221); //B10(20-23) B11(20-23) B10(20-23) B11(20-23) B14(20-23) B15(20-23) B14(20-23) B15(20-23)
                    const __m256i rhs_mat_2367_12_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_12, 221); //B12(20-23) B13(20-23) B12(20-23) B13(20-23) B16(20-23) B17(20-23) B16(20-23) B17(20-23)

                    const __m256i rhs_mat_0145_13_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_13, 221); //B10(28-31) B11(28-31) B10(28-31) B11(28-31) B14(28-31) B15(28-31) B14(28-31) B15(28-31)
                    const __m256i rhs_mat_2367_13_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_13, 221); //B12(28-31) B13(28-31) B12(28-31) B13(28-31) B16(28-31) B17(28-31) B16(28-31) B17(28-31)

                    uint32_t utmp_0[4], utmp_1[4];

                    // Scales and Mins of corresponding sub blocks from different Q4_K structures are stored together
                    // The below block is for eg to extract first sub block's scales and mins from different Q4_K structures for the sb loop
                    memcpy(utmp_0, b_ptr[b].scales + 24 * sb, 12);
                    utmp_0[3] = ((utmp_0[2] >> 4) & kmask2) | (((utmp_0[1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_0 = utmp_0[1] & kmask1;
                    utmp_0[1] = (utmp_0[2] & kmask2) | (((utmp_0[0] >> 6) & kmask3) << 4);
                    utmp_0[2] = uaux_0;
                    utmp_0[0] &= kmask1;

                    // The below block is for eg to extract second sub block's scales and mins from different Q4_K structures for the sb loop
                    memcpy(utmp_1, b_ptr[b].scales + 12 + sb * 24, 12);
                    utmp_1[3] = ((utmp_1[2] >> 4) & kmask2) | (((utmp_1[1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_1 = utmp_1[1] & kmask1;
                    utmp_1[1] = (utmp_1[2] & kmask2) | (((utmp_1[0] >> 6) & kmask3) << 4);
                    utmp_1[2] = uaux_1;
                    utmp_1[0] &= kmask1;

                    // Scales of first sub block in the sb loop
                    const __m128i mins_and_scales_0 = _mm_set_epi32(utmp_0[3], utmp_0[2], utmp_0[1], utmp_0[0]);
                    const __m256i scales_0 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(mins_and_scales_0, mins_and_scales_0));

                    // Scales of second sub block in the sb loop
                    const __m128i mins_and_scales_1 = _mm_set_epi32(utmp_1[3], utmp_1[2], utmp_1[1], utmp_1[0]);
                    const __m256i scales_1 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(mins_and_scales_1, mins_and_scales_1));

                    // Mins of first and second sub block of Q4_K block are arranged side by side
                    const __m256i mins_01 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(_mm_shuffle_epi32(mins_and_scales_0, 78), _mm_shuffle_epi32(mins_and_scales_1, 78)));

                    const __m256i scale_0145_0 = _mm256_shuffle_epi32(scales_0, 68);
                    const __m256i scale_2367_0 = _mm256_shuffle_epi32(scales_0, 238);

                    const __m256i scale_0145_1 = _mm256_shuffle_epi32(scales_1, 68);
                    const __m256i scale_2367_1 = _mm256_shuffle_epi32(scales_1, 238);

                    for (int rp = 0; rp < 4; rp++) {

                        // Load the four block_q8_k quantized values interleaved with each other in chunks of eight bytes - A0,A1,A2,A3
                        // Loaded as set of 128 bit vectors and repeated into a 256 bit vector
                        __m256i lhs_mat_0123_00 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 256 * sb)));
                        __m256i lhs_mat_01_00 = _mm256_permute2f128_si256(lhs_mat_0123_00, lhs_mat_0123_00, 0);
                        __m256i lhs_mat_23_00 = _mm256_permute2f128_si256(lhs_mat_0123_00, lhs_mat_0123_00, 17);
                        __m256i lhs_mat_0123_01 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 32 + 256 * sb)));
                        __m256i lhs_mat_01_01 = _mm256_permute2f128_si256(lhs_mat_0123_01, lhs_mat_0123_01, 0);
                        __m256i lhs_mat_23_01 = _mm256_permute2f128_si256(lhs_mat_0123_01, lhs_mat_0123_01, 17);
                        __m256i lhs_mat_0123_02 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 64 + 256 * sb)));
                        __m256i lhs_mat_01_02 = _mm256_permute2f128_si256(lhs_mat_0123_02, lhs_mat_0123_02, 0);
                        __m256i lhs_mat_23_02 = _mm256_permute2f128_si256(lhs_mat_0123_02, lhs_mat_0123_02, 17);
                        __m256i lhs_mat_0123_03 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 96 + 256 * sb)));
                        __m256i lhs_mat_01_03 = _mm256_permute2f128_si256(lhs_mat_0123_03, lhs_mat_0123_03, 0);
                        __m256i lhs_mat_23_03 = _mm256_permute2f128_si256(lhs_mat_0123_03, lhs_mat_0123_03, 17);
                        __m256i lhs_mat_0123_10 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 128 + 256 * sb)));
                        __m256i lhs_mat_01_10 = _mm256_permute2f128_si256(lhs_mat_0123_10, lhs_mat_0123_10, 0);
                        __m256i lhs_mat_23_10 = _mm256_permute2f128_si256(lhs_mat_0123_10, lhs_mat_0123_10, 17);
                        __m256i lhs_mat_0123_11 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 160 + 256 * sb)));
                        __m256i lhs_mat_01_11 = _mm256_permute2f128_si256(lhs_mat_0123_11, lhs_mat_0123_11, 0);
                        __m256i lhs_mat_23_11 = _mm256_permute2f128_si256(lhs_mat_0123_11, lhs_mat_0123_11, 17);
                        __m256i lhs_mat_0123_12 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 192 + 256 * sb)));
                        __m256i lhs_mat_01_12 = _mm256_permute2f128_si256(lhs_mat_0123_12, lhs_mat_0123_12, 0);
                        __m256i lhs_mat_23_12 = _mm256_permute2f128_si256(lhs_mat_0123_12, lhs_mat_0123_12, 17);
                        __m256i lhs_mat_0123_13 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 224 + 256 * sb)));
                        __m256i lhs_mat_01_13 = _mm256_permute2f128_si256(lhs_mat_0123_13, lhs_mat_0123_13, 0);
                        __m256i lhs_mat_23_13 = _mm256_permute2f128_si256(lhs_mat_0123_13, lhs_mat_0123_13, 17);

                        // Bsums are loaded - four bsums are loaded (for two sub blocks) for the different Q8_K blocks
                        __m256i lhs_bsums_0123_01 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].bsums + 16 * sb)));
                        __m256i lhs_bsums_hsum_0123_01 = _mm256_castsi128_si256(_mm_hadd_epi16(_mm256_castsi256_si128(lhs_bsums_0123_01), _mm256_extractf128_si256(lhs_bsums_0123_01, 1)));
                        lhs_bsums_hsum_0123_01 = _mm256_permute2x128_si256(lhs_bsums_hsum_0123_01, lhs_bsums_hsum_0123_01, 0);

                        // Shuffle pattern one - left side input
                        const __m256i lhs_mat_01_00_sp1 = _mm256_shuffle_epi32(lhs_mat_01_00, 160); //A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3) A00(0-3) A01(0-3) A01(0-3)
                        const __m256i lhs_mat_23_00_sp1 = _mm256_shuffle_epi32(lhs_mat_23_00, 160); //A02(0-3) A03(0-3) A02(0-3) A03(0-3) A02(0-3) A03(0-3) A02(0-3) A03(0-3)

                        const __m256i lhs_mat_01_01_sp1 = _mm256_shuffle_epi32(lhs_mat_01_01, 160);  //A00(8-11) A00(8-11) A01(8-11) A01(8-11) A00(8-11) A00(8-11) A01(8-11) A01(8-11)
                        const __m256i lhs_mat_23_01_sp1 = _mm256_shuffle_epi32(lhs_mat_23_01, 160);  //A02(8-11) A03(8-11) A02(8-11) A03(8-11) A02(8-11) A03(8-11) A02(8-11) A03(8-11)

                        const __m256i lhs_mat_01_02_sp1 = _mm256_shuffle_epi32(lhs_mat_01_02, 160);  //A00(16-19) A00(16-19) A01(16-19) A01(16-19) A00(16-19) A00(16-19) A01(16-19) A01(16-19)
                        const __m256i lhs_mat_23_02_sp1 = _mm256_shuffle_epi32(lhs_mat_23_02, 160);  //A02(16-19) A03(16-19) A02(16-19) A03(16-19) A02(16-19) A03(16-19) A02(16-19) A03(16-19)

                        const __m256i lhs_mat_01_03_sp1 = _mm256_shuffle_epi32(lhs_mat_01_03, 160);  //A00(24-27) A00(24-27) A01(24-27) A01(24-27) A00(24-27) A00(24-27) A01(24-27) A01(24-27)
                        const __m256i lhs_mat_23_03_sp1 = _mm256_shuffle_epi32(lhs_mat_23_03, 160); //A02(24-27) A03(24-27) A02(24-27) A03(24-27) A02(24-27) A03(24-27) A02(24-27) A03(24-27)

                        const __m256i lhs_mat_01_10_sp1 = _mm256_shuffle_epi32(lhs_mat_01_10, 160); //A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3) A10(0-3) A11(0-3) A11(0-3)
                        const __m256i lhs_mat_23_10_sp1 = _mm256_shuffle_epi32(lhs_mat_23_10, 160); //A12(0-3) A13(0-3) A12(0-3) A13(0-3) A12(0-3) A13(0-3) A12(0-3) A13(0-3)

                        const __m256i lhs_mat_01_11_sp1 = _mm256_shuffle_epi32(lhs_mat_01_11, 160);  //A10(8-11) A10(8-11) A11(8-11) A11(8-11) A10(8-11) A10(8-11) A11(8-11) A11(8-11)
                        const __m256i lhs_mat_23_11_sp1 = _mm256_shuffle_epi32(lhs_mat_23_11, 160);  //A12(8-11) A13(8-11) A12(8-11) A13(8-11) A12(8-11) A13(8-11) A12(8-11) A13(8-11)

                        const __m256i lhs_mat_01_12_sp1 = _mm256_shuffle_epi32(lhs_mat_01_12, 160);  //A10(16-19) A10(16-19) A11(16-19) A11(16-19) A10(16-19) A10(16-19) A11(16-19) A11(16-19)
                        const __m256i lhs_mat_23_12_sp1 = _mm256_shuffle_epi32(lhs_mat_23_12, 160);  //A12(16-19) A13(16-19) A12(16-19) A13(16-19) A12(16-19) A13(16-19) A12(16-19) A13(16-19)

                        const __m256i lhs_mat_01_13_sp1 = _mm256_shuffle_epi32(lhs_mat_01_13, 160); //A10(24-27) A10(24-27) A11(24-27) A11(24-27) A10(24-27) A10(24-27) A11(24-27) A11(24-27)
                        const __m256i lhs_mat_23_13_sp1 = _mm256_shuffle_epi32(lhs_mat_23_13, 160); //A12(24-27) A13(24-27) A12(24-27) A13(24-27) A12(24-27) A13(24-27) A12(24-27) A13(24-27)

                        // Shuffle pattern two- left side input
                        const __m256i lhs_mat_01_00_sp2 = _mm256_shuffle_epi32(lhs_mat_01_00, 245); //A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7) A00(4-7) A01(4-7) A01(4-7)
                        const __m256i lhs_mat_23_00_sp2 = _mm256_shuffle_epi32(lhs_mat_23_00, 245); //A02(4-7) A03(4-7) A02(4-7) A03(4-7) A02(4-7) A03(4-7) A02(4-7) A03(4-7)

                        const __m256i lhs_mat_01_01_sp2 = _mm256_shuffle_epi32(lhs_mat_01_01, 245); //A00(12-15) A00(12-15) A01(12-15) A01(12-15) A00(12-15) A00(12-15) A01(12-15) A01(12-15)
                        const __m256i lhs_mat_23_01_sp2 = _mm256_shuffle_epi32(lhs_mat_23_01, 245); //A02(12-15) A03(12-15) A02(12-15) A03(12-15) A02(12-15) A03(12-15) A02(12-15) A03(12-15)

                        const __m256i lhs_mat_01_02_sp2 = _mm256_shuffle_epi32(lhs_mat_01_02, 245); //A00(20-23) A00(20-23) A01(20-23) A01(20-23) A00(20-23) A00(20-23) A01(20-23) A01(20-23)
                        const __m256i lhs_mat_23_02_sp2 = _mm256_shuffle_epi32(lhs_mat_23_02, 245); //A02(20-23) A03(20-23) A02(20-23) A03(20-23) A02(20-23) A03(20-23) A02(20-23) A03(20-23)

                        const __m256i lhs_mat_01_03_sp2 = _mm256_shuffle_epi32(lhs_mat_01_03, 245); //A00(28-31) A00(28-31) A01(28-31) A01(28-31) A00(28-31) A00(28-31) A01(28-31) A01(28-31)
                        const __m256i lhs_mat_23_03_sp2 = _mm256_shuffle_epi32(lhs_mat_23_03, 245); //A02(28-31) A03(28-31) A02(28-31) A03(28-31) A02(28-31) A03(28-31) A02(28-31) A03(28-31)

                        const __m256i lhs_mat_01_10_sp2 = _mm256_shuffle_epi32(lhs_mat_01_10, 245); //A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7) A10(4-7) A11(4-7) A11(4-7)
                        const __m256i lhs_mat_23_10_sp2 = _mm256_shuffle_epi32(lhs_mat_23_10, 245); //A12(4-7) A13(4-7) A12(4-7) A13(4-7) A12(4-7) A13(4-7) A12(4-7) A13(4-7)

                        const __m256i lhs_mat_01_11_sp2 = _mm256_shuffle_epi32(lhs_mat_01_11, 245); //A10(12-15) A10(12-15) A11(12-15) A11(12-15) A10(12-15) A10(12-15) A11(12-15) A11(12-15)
                        const __m256i lhs_mat_23_11_sp2 = _mm256_shuffle_epi32(lhs_mat_23_11, 245); //A12(12-15) A13(12-15) A12(12-15) A13(12-15) A12(12-15) A13(12-15) A12(12-15) A13(12-15)

                        const __m256i lhs_mat_01_12_sp2 = _mm256_shuffle_epi32(lhs_mat_01_12, 245); //A10(20-23) A10(20-23) A11(20-23) A11(20-23) A10(20-23) A10(20-23) A11(20-23) A11(20-23)
                        const __m256i lhs_mat_23_12_sp2 = _mm256_shuffle_epi32(lhs_mat_23_12, 245); //A12(20-23) A13(20-23) A12(20-23) A13(20-23) A12(20-23) A13(20-23) A12(20-23) A13(20-23)

                        const __m256i lhs_mat_01_13_sp2 = _mm256_shuffle_epi32(lhs_mat_01_13, 245); //A10(28-31) A10(28-31) A11(28-31) A11(28-31) A10(28-31) A10(28-31) A11(28-31) A11(28-31)
                        const __m256i lhs_mat_23_13_sp2 = _mm256_shuffle_epi32(lhs_mat_23_13, 245); //A12(28-31) A13(28-31) A12(28-31) A13(28-31) A12(28-31) A13(28-31) A12(28-31) A13(28-31)

                        // The values arranged in shuffle patterns are operated with dot product operation within 32 bit lane i.e corresponding bytes and multiplied and added into 32 bit integers within 32 bit lane
                        __m256i iacc_mat_00_0_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_03_sp1, lhs_mat_01_03_sp1), _mm256_maddubs_epi16(rhs_mat_0145_02_sp1, lhs_mat_01_02_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_01_sp1, lhs_mat_01_01_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_00_sp1, lhs_mat_01_00_sp1));
                        __m256i iacc_mat_01_0_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_03_sp1, lhs_mat_01_03_sp1), _mm256_maddubs_epi16(rhs_mat_2367_02_sp1, lhs_mat_01_02_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_01_sp1, lhs_mat_01_01_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_00_sp1, lhs_mat_01_00_sp1));
                        __m256i iacc_mat_10_0_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_03_sp1, lhs_mat_23_03_sp1), _mm256_maddubs_epi16(rhs_mat_0145_02_sp1, lhs_mat_23_02_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_01_sp1, lhs_mat_23_01_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_00_sp1, lhs_mat_23_00_sp1));
                        __m256i iacc_mat_11_0_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_03_sp1, lhs_mat_23_03_sp1), _mm256_maddubs_epi16(rhs_mat_2367_02_sp1, lhs_mat_23_02_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_01_sp1, lhs_mat_23_01_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_00_sp1, lhs_mat_23_00_sp1));
                        __m256i iacc_mat_00_1_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_13_sp1, lhs_mat_01_13_sp1), _mm256_maddubs_epi16(rhs_mat_0145_12_sp1, lhs_mat_01_12_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_11_sp1, lhs_mat_01_11_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_10_sp1, lhs_mat_01_10_sp1));
                        __m256i iacc_mat_01_1_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_13_sp1, lhs_mat_01_13_sp1), _mm256_maddubs_epi16(rhs_mat_2367_12_sp1, lhs_mat_01_12_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_11_sp1, lhs_mat_01_11_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_10_sp1, lhs_mat_01_10_sp1));
                        __m256i iacc_mat_10_1_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_13_sp1, lhs_mat_23_13_sp1), _mm256_maddubs_epi16(rhs_mat_0145_12_sp1, lhs_mat_23_12_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_11_sp1, lhs_mat_23_11_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_10_sp1, lhs_mat_23_10_sp1));
                        __m256i iacc_mat_11_1_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_13_sp1, lhs_mat_23_13_sp1), _mm256_maddubs_epi16(rhs_mat_2367_12_sp1, lhs_mat_23_12_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_11_sp1, lhs_mat_23_11_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_10_sp1, lhs_mat_23_10_sp1));

                        __m256i iacc_mat_00_0_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_03_sp2, lhs_mat_01_03_sp2), _mm256_maddubs_epi16(rhs_mat_0145_02_sp2, lhs_mat_01_02_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_01_sp2, lhs_mat_01_01_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_00_sp2, lhs_mat_01_00_sp2));
                        __m256i iacc_mat_01_0_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_03_sp2, lhs_mat_01_03_sp2), _mm256_maddubs_epi16(rhs_mat_2367_02_sp2, lhs_mat_01_02_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_01_sp2, lhs_mat_01_01_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_00_sp2, lhs_mat_01_00_sp2));
                        __m256i iacc_mat_10_0_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_03_sp2, lhs_mat_23_03_sp2), _mm256_maddubs_epi16(rhs_mat_0145_02_sp2, lhs_mat_23_02_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_01_sp2, lhs_mat_23_01_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_00_sp2, lhs_mat_23_00_sp2));
                        __m256i iacc_mat_11_0_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_03_sp2, lhs_mat_23_03_sp2), _mm256_maddubs_epi16(rhs_mat_2367_02_sp2, lhs_mat_23_02_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_01_sp2, lhs_mat_23_01_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_00_sp2, lhs_mat_23_00_sp2));
                        __m256i iacc_mat_00_1_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_13_sp2, lhs_mat_01_13_sp2), _mm256_maddubs_epi16(rhs_mat_0145_12_sp2, lhs_mat_01_12_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_11_sp2, lhs_mat_01_11_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_10_sp2, lhs_mat_01_10_sp2));
                        __m256i iacc_mat_01_1_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_13_sp2, lhs_mat_01_13_sp2), _mm256_maddubs_epi16(rhs_mat_2367_12_sp2, lhs_mat_01_12_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_11_sp2, lhs_mat_01_11_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_10_sp2, lhs_mat_01_10_sp2));
                        __m256i iacc_mat_10_1_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_13_sp2, lhs_mat_23_13_sp2), _mm256_maddubs_epi16(rhs_mat_0145_12_sp2, lhs_mat_23_12_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_11_sp2, lhs_mat_23_11_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_10_sp2, lhs_mat_23_10_sp2));
                        __m256i iacc_mat_11_1_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_13_sp2, lhs_mat_23_13_sp2), _mm256_maddubs_epi16(rhs_mat_2367_12_sp2, lhs_mat_23_12_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_11_sp2, lhs_mat_23_11_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_10_sp2, lhs_mat_23_10_sp2));

                        // Output of both shuffle patterns are added in order to sum dot product outputs of all 32 values in block
                        __m256i iacc_mat_00_0 = _mm256_add_epi16(iacc_mat_00_0_sp1, iacc_mat_00_0_sp2);
                        __m256i iacc_mat_01_0 = _mm256_add_epi16(iacc_mat_01_0_sp1, iacc_mat_01_0_sp2);
                        __m256i iacc_mat_10_0 = _mm256_add_epi16(iacc_mat_10_0_sp1, iacc_mat_10_0_sp2);
                        __m256i iacc_mat_11_0 = _mm256_add_epi16(iacc_mat_11_0_sp1, iacc_mat_11_0_sp2);

                        __m256i iacc_mat_00_1 = _mm256_add_epi16(iacc_mat_00_1_sp1, iacc_mat_00_1_sp2);
                        __m256i iacc_mat_01_1 = _mm256_add_epi16(iacc_mat_01_1_sp1, iacc_mat_01_1_sp2);
                        __m256i iacc_mat_10_1 = _mm256_add_epi16(iacc_mat_10_1_sp1, iacc_mat_10_1_sp2);
                        __m256i iacc_mat_11_1 = _mm256_add_epi16(iacc_mat_11_1_sp1, iacc_mat_11_1_sp2);

                        // Output of both shuffle patterns are added in order to sum dot product outputs of all 32 values in block
                        iacc_mat_00_0 = _mm256_madd_epi16(iacc_mat_00_0, scale_0145_0);
                        iacc_mat_01_0 = _mm256_madd_epi16(iacc_mat_01_0, scale_2367_0);
                        iacc_mat_10_0 = _mm256_madd_epi16(iacc_mat_10_0, scale_0145_0);
                        iacc_mat_11_0 = _mm256_madd_epi16(iacc_mat_11_0, scale_2367_0);

                        iacc_mat_00_1 = _mm256_madd_epi16(iacc_mat_00_1, scale_0145_1);
                        iacc_mat_01_1 = _mm256_madd_epi16(iacc_mat_01_1, scale_2367_1);
                        iacc_mat_10_1 = _mm256_madd_epi16(iacc_mat_10_1, scale_0145_1);
                        iacc_mat_11_1 = _mm256_madd_epi16(iacc_mat_11_1, scale_2367_1);

                        // Straighten out to make 4 row vectors (4 for each sub block which are accumulated together in the next step)
                        __m256i iacc_row_0_0 = _mm256_blend_epi32(iacc_mat_00_0, _mm256_shuffle_epi32(iacc_mat_01_0, 78), 204);
                        __m256i iacc_row_1_0 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_00_0, 78), iacc_mat_01_0, 204);
                        __m256i iacc_row_2_0 = _mm256_blend_epi32(iacc_mat_10_0, _mm256_shuffle_epi32(iacc_mat_11_0, 78), 204);
                        __m256i iacc_row_3_0 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_10_0, 78), iacc_mat_11_0, 204);
                        __m256i iacc_row_0_1 = _mm256_blend_epi32(iacc_mat_00_1, _mm256_shuffle_epi32(iacc_mat_01_1, 78), 204);
                        __m256i iacc_row_1_1 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_00_1, 78), iacc_mat_01_1, 204);
                        __m256i iacc_row_2_1 = _mm256_blend_epi32(iacc_mat_10_1, _mm256_shuffle_epi32(iacc_mat_11_1, 78), 204);
                        __m256i iacc_row_3_1 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_10_1, 78), iacc_mat_11_1, 204);

                        __m256i iacc_row_0 = _mm256_add_epi32(iacc_row_0_0, iacc_row_0_1);
                        __m256i iacc_row_1 = _mm256_add_epi32(iacc_row_1_0, iacc_row_1_1);
                        __m256i iacc_row_2 = _mm256_add_epi32(iacc_row_2_0, iacc_row_2_1);
                        __m256i iacc_row_3 = _mm256_add_epi32(iacc_row_3_0, iacc_row_3_1);

                        // Load the scale(d) values for all the 4 Q8_k blocks and repeat it across lanes
                        const __m128 row_scale_f32_sse = _mm_load_ps(a_ptrs[rp][b].d);
                        const __m256 row_scale_f32 = _mm256_set_m128(row_scale_f32_sse, row_scale_f32_sse);//GGML_F32Cx8_REPEAT_LOAD(a_ptrs[rp][b].d, loadMask);

                        // Multiply with appropriate scales and accumulate (for both d and dmin) below
                        acc_rows[rp * 4] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_0), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 0)), acc_rows[rp * 4]);
                        acc_rows[rp * 4 + 1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_1), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 85)), acc_rows[rp * 4 + 1]);
                        acc_rows[rp * 4 + 2] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_2), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_rows[rp * 4 + 2]);
                        acc_rows[rp * 4 + 3] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_3), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_rows[rp * 4 + 3]);

                        __m256i iacc_row_min_0 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 0), mins_01);
                        __m256i iacc_row_min_1 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 85), mins_01);
                        __m256i iacc_row_min_2 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 170), mins_01);
                        __m256i iacc_row_min_3 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 255), mins_01);

                        acc_min_rows[rp * 4] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_0), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 0)), acc_min_rows[rp * 4]);
                        acc_min_rows[rp * 4 + 1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_1), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 85)), acc_min_rows[rp * 4 + 1]);
                        acc_min_rows[rp * 4 + 2] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_2), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_min_rows[rp * 4 + 2]);
                        acc_min_rows[rp * 4 + 3] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_3), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_min_rows[rp * 4 + 3]);

                    }
                }
            }
            // Store the accumulated values
            for (int i = 0; i < 16; i++) {
                _mm256_storeu_ps((float * )(s + ((y * 4 + i) * bs + x * 8)), _mm256_sub_ps(acc_rows[i], acc_min_rows[i]));
            }
        }
    }
    for (; y < nr / 4; y++) {

        const block_q8_Kx4 * a_ptr = a_ptr_start + (y * nb);

        for (int64_t x = xstart; x < nc / 8; x++) {

            const block_q4_Kx8 * b_ptr = b_ptr_start + (x * b_nb);

            // Master FP accumulators
            __m256 acc_rows[4];
            for (int i = 0; i < 4; i++) {
                acc_rows[i] = _mm256_setzero_ps();
            }

            __m256 acc_min_rows[4];
            for (int i = 0; i < 4; i++) {
                acc_min_rows[i] = _mm256_setzero_ps();
            }

            for (int64_t b = 0; b < nb; b++) {

                // Scale values - Load the eight scale values of block_q4_Kx8
                const __m256 col_scale_f32 = GGML_F32Cx8_LOAD(b_ptr[b].d);

                // dmin values - Load the eight dmin values of block_q4_Kx8
                const __m256 col_dmin_f32 = GGML_F32Cx8_LOAD(b_ptr[b].dmin);

                // Loop to iterate over the eight sub blocks of a super block - two sub blocks are processed per iteration
                for (int sb = 0; sb < QK_K / 64; sb++) {

                    // Load the eight block_q4_k for two sub blocks quantized values interleaved with each other in chunks of eight bytes - B0,B1 ....B6,B7
                    const __m256i rhs_raw_mat_0123_0 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + sb * 256));
                    const __m256i rhs_raw_mat_4567_0 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 32 + sb * 256));
                    const __m256i rhs_raw_mat_0123_1 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 64 + sb * 256));
                    const __m256i rhs_raw_mat_4567_1 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 96 + sb * 256));
                    const __m256i rhs_raw_mat_0123_2 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 128 + sb * 256));
                    const __m256i rhs_raw_mat_4567_2 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 160 + sb * 256));
                    const __m256i rhs_raw_mat_0123_3 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 192 + sb * 256));
                    const __m256i rhs_raw_mat_4567_3 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 224 + sb * 256));

                    // Save the values in the following vectors in the formats B0B1B4B5, B2B3B6B7 for further processing and storing of values
                    const __m256i rhs_raw_mat_0145_0 = _mm256_blend_epi32(rhs_raw_mat_0123_0, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_0, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_0, requiredOrder), rhs_raw_mat_4567_0, 240);
                    const __m256i rhs_raw_mat_0145_1 = _mm256_blend_epi32(rhs_raw_mat_0123_1, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_1, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_1, requiredOrder), rhs_raw_mat_4567_1, 240);
                    const __m256i rhs_raw_mat_0145_2 = _mm256_blend_epi32(rhs_raw_mat_0123_2, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_2, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_2 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_2, requiredOrder), rhs_raw_mat_4567_2, 240);
                    const __m256i rhs_raw_mat_0145_3 = _mm256_blend_epi32(rhs_raw_mat_0123_3, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_3, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_3 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_3, requiredOrder), rhs_raw_mat_4567_3, 240);

                    // 4-bit -> 8-bit
                    // First sub block of the two sub blocks processed in the iteration
                    const __m256i rhs_mat_0145_00 = _mm256_and_si256(rhs_raw_mat_0145_0, m4b); //B00(0-7) B01(0-7) B04(0-7) B05(0-7)
                    const __m256i rhs_mat_2367_00 = _mm256_and_si256(rhs_raw_mat_2367_0, m4b); //B02(0-7) B03(0-7) B06(0-7) B07(0-7)

                    const __m256i rhs_mat_0145_01 = _mm256_and_si256(rhs_raw_mat_0145_1, m4b); //B00(8-15) B01(8-15) B04(8-15) B05(8-15)
                    const __m256i rhs_mat_2367_01 = _mm256_and_si256(rhs_raw_mat_2367_1, m4b); //B02(8-15) B03(8-15) B06(8-15) B07(8-15)

                    const __m256i rhs_mat_0145_02 = _mm256_and_si256(rhs_raw_mat_0145_2, m4b); //B00(16-23) B01(16-23) B04(16-23) B05(16-23)
                    const __m256i rhs_mat_2367_02 = _mm256_and_si256(rhs_raw_mat_2367_2, m4b); //B02(16-23) B03(16-23) B06(16-23) B07(16-23)

                    const __m256i rhs_mat_0145_03 = _mm256_and_si256(rhs_raw_mat_0145_3, m4b); //B00(24-31) B01(24-31) B04(24-31) B05(24-31)
                    const __m256i rhs_mat_2367_03 = _mm256_and_si256(rhs_raw_mat_2367_3, m4b); //B02(24-31) B03(24-31) B06(24-31) B07(24-31)

                    // Second sub block of the two sub blocks processed in the iteration
                    const __m256i rhs_mat_0145_10 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_0, 4), m4b); //B10(0-7) B11(0-7) B14(0-7) B15(0-7)
                    const __m256i rhs_mat_2367_10 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_0, 4), m4b); //B12(0-7) B13(0-7) B16(0-7) B17(0-7)

                    const __m256i rhs_mat_0145_11 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_1, 4), m4b); //B10(8-15) B11(8-15) B14(8-15) B15(8-15)
                    const __m256i rhs_mat_2367_11 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_1, 4), m4b); //B12(8-15) B13(8-15) B16(8-15) B17(8-15)

                    const __m256i rhs_mat_0145_12 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_2, 4), m4b); //B10(16-23) B11(16-23) B14(16-23) B15(16-23)
                    const __m256i rhs_mat_2367_12 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_2, 4), m4b); //B12(16-23) B13(16-23) B16(16-23) B17(16-23)

                    const __m256i rhs_mat_0145_13 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_3, 4), m4b); //B10(24-31) B11(24-31) B14(24-31) B15(24-31)
                    const __m256i rhs_mat_2367_13 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_3, 4), m4b); //B12(24-31) B13(24-31) B16(24-31) B17(24-31)

                    // Shuffle pattern one - right side input
                    const __m256i rhs_mat_0145_00_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_00, 136); //B00(0-3) B01(0-3) B00(0-3) B01(0-3) B04(0-3) B05(0-3) B04(0-3) B05(0-3)
                    const __m256i rhs_mat_2367_00_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_00, 136); //B02(0-3) B03(0-3) B02(0-3) B03(0-3) B06(0-3) B07(0-3) B06(0-3) B07(0-3)

                    const __m256i rhs_mat_0145_01_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_01, 136); //B00(8-11) B01(8-11) B00(8-11) B01(8-11) B04(8-11) B05(8-11) B04(8-11) B05(8-11)
                    const __m256i rhs_mat_2367_01_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_01, 136); //B02(8-11) B03(8-11) B02(8-11) B03(8-11) B06(8-11) B07(8-11) B06(8-11) B07(8-11)

                    const __m256i rhs_mat_0145_02_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_02, 136); //B00(16-19) B01(16-19) B00(16-19) B01(16-19) B04(16-19) B05(16-19) B04(16-19) B05(16-19)
                    const __m256i rhs_mat_2367_02_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_02, 136); //B02(16-19) B03(16-19) B02(16-19) B03(16-19) B06(16-19) B07(16-19) B06(16-19) B07(16-19)

                    const __m256i rhs_mat_0145_03_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_03, 136); //B00(24-27) B01(24-27) B00(24-27) B01(24-27) B04(24-27) B05(24-27) B04(24-27) B05(24-27)
                    const __m256i rhs_mat_2367_03_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_03, 136); //B02(24-27) B03(24-27) B02(24-27) B03(24-27) B06(24-27) B07(24-27) B06(24-27) B07(24-27)

                    const __m256i rhs_mat_0145_10_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_10, 136); //B10(0-3) B11(0-3) B10(0-3) B11(0-3) B14(0-3) B15(0-3) B14(0-3) B15(0-3)
                    const __m256i rhs_mat_2367_10_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_10, 136); //B12(0-3) B13(0-3) B12(0-3) B13(0-3) B16(0-3) B17(0-3) B16(0-3) B17(0-3)

                    const __m256i rhs_mat_0145_11_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_11, 136); //B10(8-11) B11(8-11) B10(8-11) B11(8-11) B14(8-11) B15(8-11) B14(8-11) B15(8-11)
                    const __m256i rhs_mat_2367_11_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_11, 136); //B12(8-11) B13(8-11) B12(8-11) B13(8-11) B16(8-11) B17(8-11) B16(8-11) B17(8-11)

                    const __m256i rhs_mat_0145_12_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_12, 136); //B10(16-19) B11(16-19) B10(16-19) B11(16-19) B14(16-19) B15(16-19) B14(16-19) B15(16-19)
                    const __m256i rhs_mat_2367_12_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_12, 136); //B12(16-19) B13(16-19) B12(16-19) B13(16-19) B16(16-19) B17(16-19) B16(16-19) B17(16-19)

                    const __m256i rhs_mat_0145_13_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_13, 136); //B10(24-27) B11(24-27) B10(24-27) B11(24-27) B14(24-27) B15(24-27) B14(24-27) B15(24-27)
                    const __m256i rhs_mat_2367_13_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_13, 136); //B12(24-27) B13(24-27) B12(24-27) B13(24-27) B16(24-27) B17(24-27) B16(24-27) B17(24-27)

                    // Shuffle pattern two - right side input
                    const __m256i rhs_mat_0145_00_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_00, 221); //B00(4-7) B01(4-7) B00(4-7) B01(4-7) B04(4-7) B05(4-7) B04(4-7) B05(4-7)
                    const __m256i rhs_mat_2367_00_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_00, 221); //B02(4-7) B03(4-7) B02(4-7) B03(4-7) B06(4-7) B07(4-7) B06(4-7) B07(4-7)

                    const __m256i rhs_mat_0145_01_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_01, 221); //B00(12-15) B01(12-15) B00(12-15) B01(12-15) B04(12-15) B05(12-15) B04(12-15) B05(12-15)
                    const __m256i rhs_mat_2367_01_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_01, 221); //B02(12-15) B03(12-15) B02(12-15) B03(12-15) B06(12-15) B07(12-15) B06(12-15) B07(12-15)

                    const __m256i rhs_mat_0145_02_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_02, 221); //B00(20-23) B01(20-23) B00(20-23) B01(20-23) B04(20-23) B05(20-23) B04(20-23) B05(20-23)
                    const __m256i rhs_mat_2367_02_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_02, 221); //B02(20-23) B03(20-23) B02(20-23) B03(20-23) B06(20-23) B07(20-23) B06(20-23) B07(20-23)

                    const __m256i rhs_mat_0145_03_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_03, 221); //B00(28-31) B01(28-31) B00(28-31) B01(28-31) B04(28-31) B05(28-31) B04(28-31) B05(28-31)
                    const __m256i rhs_mat_2367_03_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_03, 221); //B02(28-31) B03(28-31) B02(28-31) B03(28-31) B06(28-31) B07(28-31) B06(28-31) B07(28-31)

                    const __m256i rhs_mat_0145_10_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_10, 221); //B10(4-7) B11(4-7) B10(4-7) B11(4-7) B14(4-7) B15(4-7) B14(4-7) B15(4-7)
                    const __m256i rhs_mat_2367_10_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_10, 221); //B12(4-7) B13(4-7) B12(4-7) B13(4-7) B16(4-7) B17(4-7) B16(4-7) B17(4-7)

                    const __m256i rhs_mat_0145_11_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_11, 221); //B10(12-15) B11(12-15) B10(12-15) B11(12-15) B14(12-15) B15(12-15) B14(12-15) B15(12-15)
                    const __m256i rhs_mat_2367_11_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_11, 221); //B12(12-15) B13(12-15) B12(12-15) B13(12-15) B16(12-15) B17(12-15) B16(12-15) B17(12-15)

                    const __m256i rhs_mat_0145_12_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_12, 221); //B10(20-23) B11(20-23) B10(20-23) B11(20-23) B14(20-23) B15(20-23) B14(20-23) B15(20-23)
                    const __m256i rhs_mat_2367_12_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_12, 221); //B12(20-23) B13(20-23) B12(20-23) B13(20-23) B16(20-23) B17(20-23) B16(20-23) B17(20-23)

                    const __m256i rhs_mat_0145_13_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_13, 221); //B10(28-31) B11(28-31) B10(28-31) B11(28-31) B14(28-31) B15(28-31) B14(28-31) B15(28-31)
                    const __m256i rhs_mat_2367_13_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_13, 221); //B12(28-31) B13(28-31) B12(28-31) B13(28-31) B16(28-31) B17(28-31) B16(28-31) B17(28-31)

                    uint32_t utmp_0[4], utmp_1[4];

                    // Scales and Mins of corresponding sub blocks from different Q4_K structures are stored together
                    // The below block is for eg to extract first sub block's scales and mins from different Q4_K structures for the sb loop
                    memcpy(utmp_0, b_ptr[b].scales + 24 * sb, 12);
                    utmp_0[3] = ((utmp_0[2] >> 4) & kmask2) | (((utmp_0[1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_0 = utmp_0[1] & kmask1;
                    utmp_0[1] = (utmp_0[2] & kmask2) | (((utmp_0[0] >> 6) & kmask3) << 4);
                    utmp_0[2] = uaux_0;
                    utmp_0[0] &= kmask1;

                    // The below block is for eg to extract second sub block's scales and mins from different Q4_K structures when sb = 1
                    memcpy(utmp_1, b_ptr[b].scales + 12 + sb * 24, 12);
                    utmp_1[3] = ((utmp_1[2] >> 4) & kmask2) | (((utmp_1[1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_1 = utmp_1[1] & kmask1;
                    utmp_1[1] = (utmp_1[2] & kmask2) | (((utmp_1[0] >> 6) & kmask3) << 4);
                    utmp_1[2] = uaux_1;
                    utmp_1[0] &= kmask1;

                    // Scales of first sub block in the sb loop
                    const __m128i mins_and_scales_0 = _mm_set_epi32(utmp_0[3], utmp_0[2], utmp_0[1], utmp_0[0]);
                    const __m256i scales_0 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(mins_and_scales_0, mins_and_scales_0));

                    // Scales of second sub block in the sb loop
                    const __m128i mins_and_scales_1 = _mm_set_epi32(utmp_1[3], utmp_1[2], utmp_1[1], utmp_1[0]);
                    const __m256i scales_1 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(mins_and_scales_1, mins_and_scales_1));

                    // Mins of first and second sub block of Q4_K block are arranged side by side
                    const __m256i mins_01 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(_mm_shuffle_epi32(mins_and_scales_0, 78), _mm_shuffle_epi32(mins_and_scales_1, 78)));

                    const __m256i scale_0145_0 = _mm256_shuffle_epi32(scales_0, 68);
                    const __m256i scale_2367_0 = _mm256_shuffle_epi32(scales_0, 238);

                    const __m256i scale_0145_1 = _mm256_shuffle_epi32(scales_1, 68);
                    const __m256i scale_2367_1 = _mm256_shuffle_epi32(scales_1, 238);

                    // Load the four block_q8_k quantized values interleaved with each other in chunks of eight bytes - A0,A1,A2,A3
                    // Loaded as set of 128 bit vectors and repeated into a 256 bit vector
                    __m256i lhs_mat_0123_00 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 256 * sb)));
                    __m256i lhs_mat_01_00 = _mm256_permute2f128_si256(lhs_mat_0123_00, lhs_mat_0123_00, 0);
                    __m256i lhs_mat_23_00 = _mm256_permute2f128_si256(lhs_mat_0123_00, lhs_mat_0123_00, 17);
                    __m256i lhs_mat_0123_01 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 32 + 256 * sb)));
                    __m256i lhs_mat_01_01 = _mm256_permute2f128_si256(lhs_mat_0123_01, lhs_mat_0123_01, 0);
                    __m256i lhs_mat_23_01 = _mm256_permute2f128_si256(lhs_mat_0123_01, lhs_mat_0123_01, 17);
                    __m256i lhs_mat_0123_02 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 64 + 256 * sb)));
                    __m256i lhs_mat_01_02 = _mm256_permute2f128_si256(lhs_mat_0123_02, lhs_mat_0123_02, 0);
                    __m256i lhs_mat_23_02 = _mm256_permute2f128_si256(lhs_mat_0123_02, lhs_mat_0123_02, 17);
                    __m256i lhs_mat_0123_03 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 96 + 256 * sb)));
                    __m256i lhs_mat_01_03 = _mm256_permute2f128_si256(lhs_mat_0123_03, lhs_mat_0123_03, 0);
                    __m256i lhs_mat_23_03 = _mm256_permute2f128_si256(lhs_mat_0123_03, lhs_mat_0123_03, 17);
                    __m256i lhs_mat_0123_10 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 128 + 256 * sb)));
                    __m256i lhs_mat_01_10 = _mm256_permute2f128_si256(lhs_mat_0123_10, lhs_mat_0123_10, 0);
                    __m256i lhs_mat_23_10 = _mm256_permute2f128_si256(lhs_mat_0123_10, lhs_mat_0123_10, 17);
                    __m256i lhs_mat_0123_11 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 160 + 256 * sb)));
                    __m256i lhs_mat_01_11 = _mm256_permute2f128_si256(lhs_mat_0123_11, lhs_mat_0123_11, 0);
                    __m256i lhs_mat_23_11 = _mm256_permute2f128_si256(lhs_mat_0123_11, lhs_mat_0123_11, 17);
                    __m256i lhs_mat_0123_12 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 192 + 256 * sb)));
                    __m256i lhs_mat_01_12 = _mm256_permute2f128_si256(lhs_mat_0123_12, lhs_mat_0123_12, 0);
                    __m256i lhs_mat_23_12 = _mm256_permute2f128_si256(lhs_mat_0123_12, lhs_mat_0123_12, 17);
                    __m256i lhs_mat_0123_13 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 224 + 256 * sb)));
                    __m256i lhs_mat_01_13 = _mm256_permute2f128_si256(lhs_mat_0123_13, lhs_mat_0123_13, 0);
                    __m256i lhs_mat_23_13 = _mm256_permute2f128_si256(lhs_mat_0123_13, lhs_mat_0123_13, 17);

                    // Bsums are loaded - four bsums are loaded (for two sub blocks) for the different Q8_K blocks
                    __m256i lhs_bsums_0123_01 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].bsums + 16 * sb)));
                    __m256i lhs_bsums_hsum_0123_01 = _mm256_castsi128_si256(_mm_hadd_epi16(_mm256_castsi256_si128(lhs_bsums_0123_01), _mm256_extractf128_si256(lhs_bsums_0123_01, 1)));
                    lhs_bsums_hsum_0123_01 = _mm256_permute2x128_si256(lhs_bsums_hsum_0123_01, lhs_bsums_hsum_0123_01, 0);

                    // Shuffle pattern one - left side input
                    const __m256i lhs_mat_01_00_sp1 = _mm256_shuffle_epi32(lhs_mat_01_00, 160); //A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3) A00(0-3) A01(0-3) A01(0-3)
                    const __m256i lhs_mat_23_00_sp1 = _mm256_shuffle_epi32(lhs_mat_23_00, 160); //A02(0-3) A03(0-3) A02(0-3) A03(0-3) A02(0-3) A03(0-3) A02(0-3) A03(0-3)

                    const __m256i lhs_mat_01_01_sp1 = _mm256_shuffle_epi32(lhs_mat_01_01, 160);  //A00(8-11) A00(8-11) A01(8-11) A01(8-11) A00(8-11) A00(8-11) A01(8-11) A01(8-11)
                    const __m256i lhs_mat_23_01_sp1 = _mm256_shuffle_epi32(lhs_mat_23_01, 160);  //A02(8-11) A03(8-11) A02(8-11) A03(8-11) A02(8-11) A03(8-11) A02(8-11) A03(8-11)

                    const __m256i lhs_mat_01_02_sp1 = _mm256_shuffle_epi32(lhs_mat_01_02, 160);  //A00(16-19) A00(16-19) A01(16-19) A01(16-19) A00(16-19) A00(16-19) A01(16-19) A01(16-19)
                    const __m256i lhs_mat_23_02_sp1 = _mm256_shuffle_epi32(lhs_mat_23_02, 160);  //A02(16-19) A03(16-19) A02(16-19) A03(16-19) A02(16-19) A03(16-19) A02(16-19) A03(16-19)

                    const __m256i lhs_mat_01_03_sp1 = _mm256_shuffle_epi32(lhs_mat_01_03, 160);  //A00(24-27) A00(24-27) A01(24-27) A01(24-27) A00(24-27) A00(24-27) A01(24-27) A01(24-27)
                    const __m256i lhs_mat_23_03_sp1 = _mm256_shuffle_epi32(lhs_mat_23_03, 160); //A02(24-27) A03(24-27) A02(24-27) A03(24-27) A02(24-27) A03(24-27) A02(24-27) A03(24-27)

                    const __m256i lhs_mat_01_10_sp1 = _mm256_shuffle_epi32(lhs_mat_01_10, 160); //A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3) A10(0-3) A11(0-3) A11(0-3)
                    const __m256i lhs_mat_23_10_sp1 = _mm256_shuffle_epi32(lhs_mat_23_10, 160); //A12(0-3) A13(0-3) A12(0-3) A13(0-3) A12(0-3) A13(0-3) A12(0-3) A13(0-3)

                    const __m256i lhs_mat_01_11_sp1 = _mm256_shuffle_epi32(lhs_mat_01_11, 160);  //A10(8-11) A10(8-11) A11(8-11) A11(8-11) A10(8-11) A10(8-11) A11(8-11) A11(8-11)
                    const __m256i lhs_mat_23_11_sp1 = _mm256_shuffle_epi32(lhs_mat_23_11, 160);  //A12(8-11) A13(8-11) A12(8-11) A13(8-11) A12(8-11) A13(8-11) A12(8-11) A13(8-11)

                    const __m256i lhs_mat_01_12_sp1 = _mm256_shuffle_epi32(lhs_mat_01_12, 160);  //A10(16-19) A10(16-19) A11(16-19) A11(16-19) A10(16-19) A10(16-19) A11(16-19) A11(16-19)
                    const __m256i lhs_mat_23_12_sp1 = _mm256_shuffle_epi32(lhs_mat_23_12, 160);  //A12(16-19) A13(16-19) A12(16-19) A13(16-19) A12(16-19) A13(16-19) A12(16-19) A13(16-19)

                    const __m256i lhs_mat_01_13_sp1 = _mm256_shuffle_epi32(lhs_mat_01_13, 160); //A10(24-27) A10(24-27) A11(24-27) A11(24-27) A10(24-27) A10(24-27) A11(24-27) A11(24-27)
                    const __m256i lhs_mat_23_13_sp1 = _mm256_shuffle_epi32(lhs_mat_23_13, 160); //A12(24-27) A13(24-27) A12(24-27) A13(24-27) A12(24-27) A13(24-27) A12(24-27) A13(24-27)

                    // Shuffle pattern two- left side input
                    const __m256i lhs_mat_01_00_sp2 = _mm256_shuffle_epi32(lhs_mat_01_00, 245); //A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7) A00(4-7) A01(4-7) A01(4-7)
                    const __m256i lhs_mat_23_00_sp2 = _mm256_shuffle_epi32(lhs_mat_23_00, 245); //A02(4-7) A03(4-7) A02(4-7) A03(4-7) A02(4-7) A03(4-7) A02(4-7) A03(4-7)

                    const __m256i lhs_mat_01_01_sp2 = _mm256_shuffle_epi32(lhs_mat_01_01, 245); //A00(12-15) A00(12-15) A01(12-15) A01(12-15) A00(12-15) A00(12-15) A01(12-15) A01(12-15)
                    const __m256i lhs_mat_23_01_sp2 = _mm256_shuffle_epi32(lhs_mat_23_01, 245); //A02(12-15) A03(12-15) A02(12-15) A03(12-15) A02(12-15) A03(12-15) A02(12-15) A03(12-15)

                    const __m256i lhs_mat_01_02_sp2 = _mm256_shuffle_epi32(lhs_mat_01_02, 245); //A00(20-23) A00(20-23) A01(20-23) A01(20-23) A00(20-23) A00(20-23) A01(20-23) A01(20-23)
                    const __m256i lhs_mat_23_02_sp2 = _mm256_shuffle_epi32(lhs_mat_23_02, 245); //A02(20-23) A03(20-23) A02(20-23) A03(20-23) A02(20-23) A03(20-23) A02(20-23) A03(20-23)

                    const __m256i lhs_mat_01_03_sp2 = _mm256_shuffle_epi32(lhs_mat_01_03, 245); //A00(28-31) A00(28-31) A01(28-31) A01(28-31) A00(28-31) A00(28-31) A01(28-31) A01(28-31)
                    const __m256i lhs_mat_23_03_sp2 = _mm256_shuffle_epi32(lhs_mat_23_03, 245); //A02(28-31) A03(28-31) A02(28-31) A03(28-31) A02(28-31) A03(28-31) A02(28-31) A03(28-31)

                    const __m256i lhs_mat_01_10_sp2 = _mm256_shuffle_epi32(lhs_mat_01_10, 245); //A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7) A10(4-7) A11(4-7) A11(4-7)
                    const __m256i lhs_mat_23_10_sp2 = _mm256_shuffle_epi32(lhs_mat_23_10, 245); //A12(4-7) A13(4-7) A12(4-7) A13(4-7) A12(4-7) A13(4-7) A12(4-7) A13(4-7)

                    const __m256i lhs_mat_01_11_sp2 = _mm256_shuffle_epi32(lhs_mat_01_11, 245); //A10(12-15) A10(12-15) A11(12-15) A11(12-15) A10(12-15) A10(12-15) A11(12-15) A11(12-15)
                    const __m256i lhs_mat_23_11_sp2 = _mm256_shuffle_epi32(lhs_mat_23_11, 245); //A12(12-15) A13(12-15) A12(12-15) A13(12-15) A12(12-15) A13(12-15) A12(12-15) A13(12-15)

                    const __m256i lhs_mat_01_12_sp2 = _mm256_shuffle_epi32(lhs_mat_01_12, 245); //A10(20-23) A10(20-23) A11(20-23) A11(20-23) A10(20-23) A10(20-23) A11(20-23) A11(20-23)
                    const __m256i lhs_mat_23_12_sp2 = _mm256_shuffle_epi32(lhs_mat_23_12, 245); //A12(20-23) A13(20-23) A12(20-23) A13(20-23) A12(20-23) A13(20-23) A12(20-23) A13(20-23)

                    const __m256i lhs_mat_01_13_sp2 = _mm256_shuffle_epi32(lhs_mat_01_13, 245); //A10(28-31) A10(28-31) A11(28-31) A11(28-31) A10(28-31) A10(28-31) A11(28-31) A11(28-31)
                    const __m256i lhs_mat_23_13_sp2 = _mm256_shuffle_epi32(lhs_mat_23_13, 245); //A12(28-31) A13(28-31) A12(28-31) A13(28-31) A12(28-31) A13(28-31) A12(28-31) A13(28-31)

                    // The values arranged in shuffle patterns are operated with dot product operation within 32 bit lane i.e corresponding bytes and multiplied and added into 32 bit integers within 32 bit lane
                    __m256i iacc_mat_00_0_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_03_sp1, lhs_mat_01_03_sp1), _mm256_maddubs_epi16(rhs_mat_0145_02_sp1, lhs_mat_01_02_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_01_sp1, lhs_mat_01_01_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_00_sp1, lhs_mat_01_00_sp1));
                    __m256i iacc_mat_01_0_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_03_sp1, lhs_mat_01_03_sp1), _mm256_maddubs_epi16(rhs_mat_2367_02_sp1, lhs_mat_01_02_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_01_sp1, lhs_mat_01_01_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_00_sp1, lhs_mat_01_00_sp1));
                    __m256i iacc_mat_10_0_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_03_sp1, lhs_mat_23_03_sp1), _mm256_maddubs_epi16(rhs_mat_0145_02_sp1, lhs_mat_23_02_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_01_sp1, lhs_mat_23_01_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_00_sp1, lhs_mat_23_00_sp1));
                    __m256i iacc_mat_11_0_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_03_sp1, lhs_mat_23_03_sp1), _mm256_maddubs_epi16(rhs_mat_2367_02_sp1, lhs_mat_23_02_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_01_sp1, lhs_mat_23_01_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_00_sp1, lhs_mat_23_00_sp1));
                    __m256i iacc_mat_00_1_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_13_sp1, lhs_mat_01_13_sp1), _mm256_maddubs_epi16(rhs_mat_0145_12_sp1, lhs_mat_01_12_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_11_sp1, lhs_mat_01_11_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_10_sp1, lhs_mat_01_10_sp1));
                    __m256i iacc_mat_01_1_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_13_sp1, lhs_mat_01_13_sp1), _mm256_maddubs_epi16(rhs_mat_2367_12_sp1, lhs_mat_01_12_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_11_sp1, lhs_mat_01_11_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_10_sp1, lhs_mat_01_10_sp1));
                    __m256i iacc_mat_10_1_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_13_sp1, lhs_mat_23_13_sp1), _mm256_maddubs_epi16(rhs_mat_0145_12_sp1, lhs_mat_23_12_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_11_sp1, lhs_mat_23_11_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_10_sp1, lhs_mat_23_10_sp1));
                    __m256i iacc_mat_11_1_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_13_sp1, lhs_mat_23_13_sp1), _mm256_maddubs_epi16(rhs_mat_2367_12_sp1, lhs_mat_23_12_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_11_sp1, lhs_mat_23_11_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_10_sp1, lhs_mat_23_10_sp1));

                    __m256i iacc_mat_00_0_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_03_sp2, lhs_mat_01_03_sp2), _mm256_maddubs_epi16(rhs_mat_0145_02_sp2, lhs_mat_01_02_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_01_sp2, lhs_mat_01_01_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_00_sp2, lhs_mat_01_00_sp2));
                    __m256i iacc_mat_01_0_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_03_sp2, lhs_mat_01_03_sp2), _mm256_maddubs_epi16(rhs_mat_2367_02_sp2, lhs_mat_01_02_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_01_sp2, lhs_mat_01_01_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_00_sp2, lhs_mat_01_00_sp2));
                    __m256i iacc_mat_10_0_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_03_sp2, lhs_mat_23_03_sp2), _mm256_maddubs_epi16(rhs_mat_0145_02_sp2, lhs_mat_23_02_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_01_sp2, lhs_mat_23_01_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_00_sp2, lhs_mat_23_00_sp2));
                    __m256i iacc_mat_11_0_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_03_sp2, lhs_mat_23_03_sp2), _mm256_maddubs_epi16(rhs_mat_2367_02_sp2, lhs_mat_23_02_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_01_sp2, lhs_mat_23_01_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_00_sp2, lhs_mat_23_00_sp2));
                    __m256i iacc_mat_00_1_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_13_sp2, lhs_mat_01_13_sp2), _mm256_maddubs_epi16(rhs_mat_0145_12_sp2, lhs_mat_01_12_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_11_sp2, lhs_mat_01_11_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_10_sp2, lhs_mat_01_10_sp2));
                    __m256i iacc_mat_01_1_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_13_sp2, lhs_mat_01_13_sp2), _mm256_maddubs_epi16(rhs_mat_2367_12_sp2, lhs_mat_01_12_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_11_sp2, lhs_mat_01_11_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_10_sp2, lhs_mat_01_10_sp2));
                    __m256i iacc_mat_10_1_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_13_sp2, lhs_mat_23_13_sp2), _mm256_maddubs_epi16(rhs_mat_0145_12_sp2, lhs_mat_23_12_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_11_sp2, lhs_mat_23_11_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_10_sp2, lhs_mat_23_10_sp2));
                    __m256i iacc_mat_11_1_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_13_sp2, lhs_mat_23_13_sp2), _mm256_maddubs_epi16(rhs_mat_2367_12_sp2, lhs_mat_23_12_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_11_sp2, lhs_mat_23_11_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_10_sp2, lhs_mat_23_10_sp2));

                    // Output of both shuffle patterns are added in order to sum dot product outputs of all 32 values in block
                    __m256i iacc_mat_00_0 = _mm256_add_epi16(iacc_mat_00_0_sp1, iacc_mat_00_0_sp2);
                    __m256i iacc_mat_01_0 = _mm256_add_epi16(iacc_mat_01_0_sp1, iacc_mat_01_0_sp2);
                    __m256i iacc_mat_10_0 = _mm256_add_epi16(iacc_mat_10_0_sp1, iacc_mat_10_0_sp2);
                    __m256i iacc_mat_11_0 = _mm256_add_epi16(iacc_mat_11_0_sp1, iacc_mat_11_0_sp2);

                    __m256i iacc_mat_00_1 = _mm256_add_epi16(iacc_mat_00_1_sp1, iacc_mat_00_1_sp2);
                    __m256i iacc_mat_01_1 = _mm256_add_epi16(iacc_mat_01_1_sp1, iacc_mat_01_1_sp2);
                    __m256i iacc_mat_10_1 = _mm256_add_epi16(iacc_mat_10_1_sp1, iacc_mat_10_1_sp2);
                    __m256i iacc_mat_11_1 = _mm256_add_epi16(iacc_mat_11_1_sp1, iacc_mat_11_1_sp2);

                    // Output of both shuffle patterns are added in order to sum dot product outputs of all 32 values in block
                    iacc_mat_00_0 = _mm256_madd_epi16(iacc_mat_00_0, scale_0145_0);
                    iacc_mat_01_0 = _mm256_madd_epi16(iacc_mat_01_0, scale_2367_0);
                    iacc_mat_10_0 = _mm256_madd_epi16(iacc_mat_10_0, scale_0145_0);
                    iacc_mat_11_0 = _mm256_madd_epi16(iacc_mat_11_0, scale_2367_0);

                    iacc_mat_00_1 = _mm256_madd_epi16(iacc_mat_00_1, scale_0145_1);
                    iacc_mat_01_1 = _mm256_madd_epi16(iacc_mat_01_1, scale_2367_1);
                    iacc_mat_10_1 = _mm256_madd_epi16(iacc_mat_10_1, scale_0145_1);
                    iacc_mat_11_1 = _mm256_madd_epi16(iacc_mat_11_1, scale_2367_1);

                    // Straighten out to make 4 row vectors (4 for each sub block which are accumulated together in the next step)
                    __m256i iacc_row_0_0 = _mm256_blend_epi32(iacc_mat_00_0, _mm256_shuffle_epi32(iacc_mat_01_0, 78), 204);
                    __m256i iacc_row_1_0 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_00_0, 78), iacc_mat_01_0, 204);
                    __m256i iacc_row_2_0 = _mm256_blend_epi32(iacc_mat_10_0, _mm256_shuffle_epi32(iacc_mat_11_0, 78), 204);
                    __m256i iacc_row_3_0 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_10_0, 78), iacc_mat_11_0, 204);
                    __m256i iacc_row_0_1 = _mm256_blend_epi32(iacc_mat_00_1, _mm256_shuffle_epi32(iacc_mat_01_1, 78), 204);
                    __m256i iacc_row_1_1 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_00_1, 78), iacc_mat_01_1, 204);
                    __m256i iacc_row_2_1 = _mm256_blend_epi32(iacc_mat_10_1, _mm256_shuffle_epi32(iacc_mat_11_1, 78), 204);
                    __m256i iacc_row_3_1 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_10_1, 78), iacc_mat_11_1, 204);

                    __m256i iacc_row_0 = _mm256_add_epi32(iacc_row_0_0, iacc_row_0_1);
                    __m256i iacc_row_1 = _mm256_add_epi32(iacc_row_1_0, iacc_row_1_1);
                    __m256i iacc_row_2 = _mm256_add_epi32(iacc_row_2_0, iacc_row_2_1);
                    __m256i iacc_row_3 = _mm256_add_epi32(iacc_row_3_0, iacc_row_3_1);

                    // Load the scale(d) values for all the 4 Q8_k blocks and repeat it across lanes
                    const __m128 row_scale_f32_sse = _mm_load_ps(a_ptr[b].d);
                    const __m256 row_scale_f32 = _mm256_set_m128(row_scale_f32_sse, row_scale_f32_sse); //GGML_F32Cx8_REPEAT_LOAD(a_ptrs[rp][b].d, loadMask);

                    // Multiply with appropriate scales and accumulate (for both d and dmin) below
                    acc_rows[0] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_0), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 0)), acc_rows[0]);
                    acc_rows[1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_1), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 85)), acc_rows[1]);
                    acc_rows[2] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_2), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_rows[2]);
                    acc_rows[3] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_3), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_rows[3]);

                    __m256i iacc_row_min_0 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 0), mins_01);
                    __m256i iacc_row_min_1 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 85), mins_01);
                    __m256i iacc_row_min_2 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 170), mins_01);
                    __m256i iacc_row_min_3 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 255), mins_01);

                    acc_min_rows[0] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_0), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 0)), acc_min_rows[0]);
                    acc_min_rows[1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_1), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 85)), acc_min_rows[1]);
                    acc_min_rows[2] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_2), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_min_rows[2]);
                    acc_min_rows[3] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_3), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_min_rows[3]);
                }
            }

            // Store the accumulated values
            for (int i = 0; i < 4; i++) {
                _mm256_storeu_ps((float * )(s + ((y * 4 + i) * bs + x * 8)), _mm256_sub_ps(acc_rows[i], acc_min_rows[i]));
            }
        }
    }

#else
    UNUSED(kmask1);
    UNUSED(kmask2);
    UNUSED(kmask3);
    ggml_gemm_q4_K_8x8_q8_K_generic(n, s, bs, vx, vy, nr, nc);
#endif
}

void ggml_gemm_iq4_nl_8x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
#if defined(__AVX2__) || defined(__AVX512F__)
    {
        __m256i signextendlut = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*)kvalues_iq4nl));
        signextendlut = _mm256_permute2f128_si256(signextendlut, signextendlut, 0);

        gemm_q4_b32_8x8_q8_0_lut_avx<block_iq4_nlx8>(n, s, bs, vx, vy, nr, nc, signextendlut);

        return;
    }
#endif // defined(__AVX2__) || defined(__AVX512F__)

    ggml_gemm_iq4_nl_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemm_mxfp4_8x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
#if defined(__AVX2__) || defined(__AVX512F__)
    {
        __m256i signextendlut = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*)kvalues_mxfp4));
        signextendlut = _mm256_permute2f128_si256(signextendlut, signextendlut, 0);

        gemm_q4_b32_8x8_q8_0_lut_avx<block_mxfp4x8>(n, s, bs, vx, vy, nr, nc, signextendlut);

        return;
    }
#endif // defined(__AVX2__) || defined(__AVX512F__)

    ggml_gemm_mxfp4_8x8_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemm_q2_K_8x8_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;

    assert (n % qk == 0);
    assert (nr % 4 == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

#if defined(__AVX2__) || defined(__AVX512F__)
    const block_q2_Kx8 * b_ptr_start = (const block_q2_Kx8 * ) vx;
    const block_q8_Kx4 * a_ptr_start = (const block_q8_Kx4 * ) vy;
    int64_t b_nb = n / QK_K;
    int64_t y = 0;

    // Permute mask used for easier vector processing at later stages
    __m256i requiredOrder = _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4);
    int64_t xstart = 0;
    int anr = nr - nr % 16; // Used to align nr with boundary of 16

    // Mask to convert 2 bit and 4 bit values into a bytes
    const __m256i m3b = _mm256_set1_epi8(3);
    const __m128i m4b_sse = _mm_set1_epi8(0xF);

    //Mask to get appropriate scales
    __m128i scalesmask1_sse = _mm_set_epi8(14,14,12,12,10,10,8,8,6,6,4,4,2,2,0,0);
    __m128i scalesmask2_sse = _mm_set_epi8(15,15,13,13,11,11,9,9,7,7,5,5,3,3,1,1);

    __m256i scalesmask1 = _mm256_castsi128_si256(scalesmask1_sse);
    scalesmask1 = _mm256_permute2f128_si256(scalesmask1, scalesmask1, 0);
    __m256i scalesmask2 = _mm256_castsi128_si256(scalesmask2_sse);
    scalesmask2 = _mm256_permute2f128_si256(scalesmask2, scalesmask2, 0);

#if defined(__AVX512BW__) && defined(__AVX512DQ__)

    int anc = nc - nc % 16; // Used to align nc with boundary of 16

    // Mask to mask out nibbles from packed bytes
    const __m256i m4b = _mm256_set1_epi8(0x0F);
    // Mask to mask out nibbles from packed bytes expanded to 512 bit length
    const __m512i m3bexpanded = _mm512_set1_epi8(3);
    //Take group of four block_q8_Kx4 structures at each pass of the loop and perform dot product operation
    for (; y < anr / 4; y += 4) {

        const block_q8_Kx4 * a_ptrs[4];

        a_ptrs[0] = a_ptr_start + (y * nb);
        for (int i = 0; i < 3; ++i) {
            a_ptrs[i + 1] = a_ptrs[i] + nb;
        }

        // Take group of eight block_q2_kx8 structures at each pass of the loop and perform dot product operation
        for (int64_t x = 0; x < anc / 8; x += 2) {

            const block_q2_Kx8 * b_ptr_0 = b_ptr_start + ((x) * b_nb);
            const block_q2_Kx8 * b_ptr_1 = b_ptr_start + ((x + 1) * b_nb);

            // Master FP accumulators
            __m512 acc_rows[16];
            for (int i = 0; i < 16; i++) {
                acc_rows[i] = _mm512_setzero_ps();
            }

            __m512 acc_min_rows[16];
            for (int i = 0; i < 16; i++) {
                acc_min_rows[i] = _mm512_setzero_ps();
            }
            // For super block
            for (int64_t b = 0; b < nb; b++) {
                // Delta values - Load the sixteen scale values from two block_q2_kx8 structures
                const __m512 col_scale_f32 = GGML_F32Cx8x2_LOAD(b_ptr_0[b].d, b_ptr_1[b].d);

                // dmin values - Load the sixteen dmin values from two block_q2_kx8 structures
                const __m512 col_dmin_f32 = GGML_F32Cx8x2_LOAD(b_ptr_0[b].dmin, b_ptr_1[b].dmin);

                // Loop to iterate over the sixteen sub blocks of a super block - eight sub blocks are processed per iteration
                for (int sb = 0; sb < QK_K / 128; sb++) {

                    // Load the eight block_q2_k for eight sub blocks quantized values interleaved with each other in chunks of eight bytes - B0,B1 ....B6,B7
                    const __m256i rhs_raw_mat_0123_0 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + sb * 256));
                    const __m256i rhs_raw_mat_4567_0 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 32 + sb * 256));
                    const __m256i rhs_raw_mat_0123_1 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 64 + sb * 256));
                    const __m256i rhs_raw_mat_4567_1 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 96 + sb * 256));
                    const __m256i rhs_raw_mat_0123_2 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 128 + sb * 256));
                    const __m256i rhs_raw_mat_4567_2 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 160 + sb * 256));
                    const __m256i rhs_raw_mat_0123_3 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 192 + sb * 256));
                    const __m256i rhs_raw_mat_4567_3 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 224 + sb * 256));

                    const __m256i rhs_raw_mat_89AB_0 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + sb * 256));
                    const __m256i rhs_raw_mat_CDEF_0 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 32 + sb * 256));
                    const __m256i rhs_raw_mat_89AB_1 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 64 + sb * 256));
                    const __m256i rhs_raw_mat_CDEF_1 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 96 + sb * 256));
                    const __m256i rhs_raw_mat_89AB_2 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 128 + sb * 256));
                    const __m256i rhs_raw_mat_CDEF_2 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 160 + sb * 256));
                    const __m256i rhs_raw_mat_89AB_3 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 192 + sb * 256));
                    const __m256i rhs_raw_mat_CDEF_3 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 224 + sb * 256));

                    const __m256i rhs_raw_mat_0145_0 = _mm256_blend_epi32(rhs_raw_mat_0123_0, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_0, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_0, requiredOrder), rhs_raw_mat_4567_0, 240);
                    const __m256i rhs_raw_mat_0145_1 = _mm256_blend_epi32(rhs_raw_mat_0123_1, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_1, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_1, requiredOrder), rhs_raw_mat_4567_1, 240);
                    const __m256i rhs_raw_mat_0145_2 = _mm256_blend_epi32(rhs_raw_mat_0123_2, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_2, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_2 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_2, requiredOrder), rhs_raw_mat_4567_2, 240);
                    const __m256i rhs_raw_mat_0145_3 = _mm256_blend_epi32(rhs_raw_mat_0123_3, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_3, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_3 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_3, requiredOrder), rhs_raw_mat_4567_3, 240);

                    const __m256i rhs_raw_mat_89CD_0 = _mm256_blend_epi32(rhs_raw_mat_89AB_0, _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_0, requiredOrder), 240);
                    const __m256i rhs_raw_mat_ABEF_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_0, requiredOrder), rhs_raw_mat_CDEF_0, 240);
                    const __m256i rhs_raw_mat_89CD_1 = _mm256_blend_epi32(rhs_raw_mat_89AB_1, _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_1, requiredOrder), 240);
                    const __m256i rhs_raw_mat_ABEF_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_1, requiredOrder), rhs_raw_mat_CDEF_1, 240);
                    const __m256i rhs_raw_mat_89CD_2 = _mm256_blend_epi32(rhs_raw_mat_89AB_2, _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_2, requiredOrder), 240);
                    const __m256i rhs_raw_mat_ABEF_2 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_2, requiredOrder), rhs_raw_mat_CDEF_2, 240);
                    const __m256i rhs_raw_mat_89CD_3 = _mm256_blend_epi32(rhs_raw_mat_89AB_3, _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_3, requiredOrder), 240);
                    const __m256i rhs_raw_mat_ABEF_3 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_3, requiredOrder), rhs_raw_mat_CDEF_3, 240);

                    const __m512i rhs_raw_mat_014589CD_0 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_0145_0), rhs_raw_mat_89CD_0, 1);
                    const __m512i rhs_raw_mat_2367ABEF_0 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_2367_0), rhs_raw_mat_ABEF_0, 1);
                    const __m512i rhs_raw_mat_014589CD_1 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_0145_1), rhs_raw_mat_89CD_1, 1);
                    const __m512i rhs_raw_mat_2367ABEF_1 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_2367_1), rhs_raw_mat_ABEF_1, 1);

                    const __m512i rhs_raw_mat_014589CD_2 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_0145_2), rhs_raw_mat_89CD_2, 1);
                    const __m512i rhs_raw_mat_2367ABEF_2 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_2367_2), rhs_raw_mat_ABEF_2, 1);
                    const __m512i rhs_raw_mat_014589CD_3 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_0145_3), rhs_raw_mat_89CD_3, 1);
                    const __m512i rhs_raw_mat_2367ABEF_3 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_2367_3), rhs_raw_mat_ABEF_3, 1);

                    //2-bit -> 8-bit
                    const __m512i rhs_mat_014589CD_00 = _mm512_and_si512(rhs_raw_mat_014589CD_0,m3bexpanded); //B00(0-7) B01(0-7) B04(0-7) B05(0-7) B08(0-7) B09(0-7) B0C(0-7) B0D(0-7)
                    const __m512i rhs_mat_2367ABEF_00 = _mm512_and_si512(rhs_raw_mat_2367ABEF_0,m3bexpanded); //B02(0-7) B03(0-7) B06(0-7) B07(0-7) B0A(0-7) B0B(0-7) B0E(0-7) B0F(0-7)
                    const __m512i rhs_mat_014589CD_01 = _mm512_and_si512(rhs_raw_mat_014589CD_1,m3bexpanded); //B00(8-15) B01(8-15) B04(8-15) B05(8-15) B08(8-15) B09(8-15) B0C(8-15) B0D(8-15)
                    const __m512i rhs_mat_2367ABEF_01 = _mm512_and_si512(rhs_raw_mat_2367ABEF_1,m3bexpanded); //B02(8-15) B03(8-15) B06(8-15) B07(8-15) B0A(8-15) B0B(8-15) B0E(8-15) B0F(8-15)
                    const __m512i rhs_mat_014589CD_10 = _mm512_and_si512(rhs_raw_mat_014589CD_2,m3bexpanded); //B10(0-7) B11(0-7) B14(0-7) B15(0-7) B18(0-7) B19(0-7) B1C(0-7) B1D(0-7)
                    const __m512i rhs_mat_2367ABEF_10 = _mm512_and_si512(rhs_raw_mat_2367ABEF_2,m3bexpanded); //B12(0-7) B13(0-7) B16(0-7) B17(0-7) B1A(0-7) B1B(0-7) B1E(0-7) B1F(0-7)
                    const __m512i rhs_mat_014589CD_11 = _mm512_and_si512(rhs_raw_mat_014589CD_3,m3bexpanded); //B10(8-15) B11(8-15) B14(8-15) B15(8-15) B18(8-15) B19(8-15) B1C(8-15) B1D(8-15)
                    const __m512i rhs_mat_2367ABEF_11 = _mm512_and_si512(rhs_raw_mat_2367ABEF_3,m3bexpanded); //B12(8-15) B13(8-15) B16(8-15) B17(8-15) B1A(8-15) B1B(8-15) B1E(8-15) B1F(8-15)

                    const __m512i rhs_mat_014589CD_20 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_0, 2), m3bexpanded); //B20(0-7) B21(0-7) B24(0-7) B25(0-7) B28(0-7) B29(0-7) B2C(0-7) B2D(0-7)
                    const __m512i rhs_mat_2367ABEF_20 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_0, 2), m3bexpanded); //B22(0-7) B23(0-7) B26(0-7) B27(0-7) B2A(0-7) B2B(0-7) B2E(0-7) B2F(0-7)

                    const __m512i rhs_mat_014589CD_21 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_1, 2), m3bexpanded); //B20(8-15) B21(8-15) B24(8-15) B25(8-15) B28(8-15) B29(8-15) B2C(8-15) B2D(8-15)
                    const __m512i rhs_mat_2367ABEF_21 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_1, 2), m3bexpanded); //B22(8-15) B23(8-15) B26(8-15) B27(8-15) B2A(8-15) B2B(8-15) B2E(8-15) B2F(8-15)

                    const __m512i rhs_mat_014589CD_30 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_2, 2), m3bexpanded); //B30(0-7) B31(0-7) B34(0-7) B35(0-7) B38(0-7) B39(0-7) B3C(0-7) B3D(0-7)
                    const __m512i rhs_mat_2367ABEF_30 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_2, 2), m3bexpanded); //B32(0-7) B33(0-7) B36(0-7) B37(0-7) B3A(0-7) B3B(0-7) B3E(0-7) B3F(0-7)

                    const __m512i rhs_mat_014589CD_31 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_3, 2), m3bexpanded); //B30(8-15) B31(8-15) B34(8-15) B35(8-15) B38(8-15) B39(8-15) B3C(8-15) B3D(8-15)
                    const __m512i rhs_mat_2367ABEF_31 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_3, 2), m3bexpanded); //B32(8-15) B33(8-15) B36(8-15) B37(8-15) B3A(8-15) B3B(8-15) B3E(8-15) B3F(8-15)

                    const __m512i rhs_mat_014589CD_40 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_0, 4), m3bexpanded); //B40(0-7) B41(0-7) B44(0-7) B45(0-7) B48(0-7) B49(0-7) B4C(0-7) B4D(0-7)
                    const __m512i rhs_mat_2367ABEF_40 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_0, 4), m3bexpanded); //B42(0-7) B43(0-7) B46(0-7) B47(0-7) B4A(0-7) B4B(0-7) B4E(0-7) B4F(0-7)

                    const __m512i rhs_mat_014589CD_41 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_1, 4), m3bexpanded); //B40(8-15) B41(8-15) B44(8-15) B45(8-15) B48(8-15) B49(8-15) B4C(8-15) B4D(8-15)
                    const __m512i rhs_mat_2367ABEF_41 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_1, 4), m3bexpanded); //B42(8-15) B43(8-15) B46(8-15) B47(8-15) B4A(8-15) B4B(8-15) B4E(8-15) B4F(8-15)

                    const __m512i rhs_mat_014589CD_50 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_2, 4), m3bexpanded); //B50(0-7) B51(0-7) B54(0-7) B55(0-7) B58(0-7) B59(0-7) B5C(0-7) B5D(0-7)
                    const __m512i rhs_mat_2367ABEF_50 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_2, 4), m3bexpanded); //B52(0-7) B53(0-7) B56(0-7) B57(0-7) B5A(0-7) B5B(0-7) B5E(0-7) B5F(0-7)

                    const __m512i rhs_mat_014589CD_51 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_3, 4), m3bexpanded); //B50(8-15) B51(8-15) B54(8-15) B55(8-15) B58(8-15) B59(8-15) B5C(8-15) B5D(8-15)
                    const __m512i rhs_mat_2367ABEF_51 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_3, 4), m3bexpanded); //B52(8-15) B53(8-15) B56(8-15) B57(8-15) B5A(8-15) B5B(8-15) B5E(8-15) B5F(8-15)

                    const __m512i rhs_mat_014589CD_60 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_0, 6), m3bexpanded); //B60(0-7) B61(0-7) B64(0-7) B65(0-7) B68(0-7) B69(0-7) B6C(0-7) B6D(0-7)
                    const __m512i rhs_mat_2367ABEF_60 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_0, 6), m3bexpanded); //B62(0-7) B63(0-7) B66(0-7) B67(0-7) B6A(0-7) B6B(0-7) B6E(0-7) B6F(0-7)

                    const __m512i rhs_mat_014589CD_61 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_1, 6), m3bexpanded); //B60(8-15) B61(8-15) B64(8-15) B65(8-15) B68(8-15) B69(8-15) B6C(8-15) B6D(8-15)
                    const __m512i rhs_mat_2367ABEF_61 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_1, 6), m3bexpanded); //B62(8-15) B63(8-15) B66(8-15) B67(8-15) B6A(8-15) B6B(8-15) B6E(8-15) B6F(8-15)

                    const __m512i rhs_mat_014589CD_70 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_2, 6), m3bexpanded); //B70(0-7) B71(0-7) B74(0-7) B75(0-7) B78(0-7) B79(0-7) B7C(0-7) B7D(0-7)
                    const __m512i rhs_mat_2367ABEF_70 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_2, 6), m3bexpanded); //B72(0-7) B73(0-7) B76(0-7) B77(0-7) B7A(0-7) B7B(0-7) B7E(0-7) B7F(0-7)

                    const __m512i rhs_mat_014589CD_71 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_3, 6), m3bexpanded); //B70(8-15) B71(8-15) B74(8-15) B75(8-15) B78(8-15) B79(8-15) B7C(8-15) B7D(8-15)
                    const __m512i rhs_mat_2367ABEF_71 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_3, 6), m3bexpanded); //B72(8-15) B73(8-15) B76(8-15) B77(8-15) B7A(8-15) B7B(8-15) B7E(8-15) B7F(8-15)

                    const __m512i rhs_mat_014589CD_00_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_00, (_MM_PERM_ENUM)136); //B00(0-3) B01(0-3) B00(0-3) B01(0-3) B04(0-3) B05(0-3) B04(0-3) B05(0-3) B08(0-3) B09(0-3) B08(0-3) B09(0-3) B0C(0-3) B0D(0-3) B0C(0-3) B0D(0-3)
                    const __m512i rhs_mat_2367ABEF_00_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_00, (_MM_PERM_ENUM)136); //B02(0-3) B03(0-3) B02(0-3) B03(0-3) B06(0-3) B07(0-3) B06(0-3) B07(0-3) B0A(0-3) B0B(0-3) B0A(0-3) B0B(0-3) B0E(0-3) B0F(0-3) B0E(0-3) B0F(0-3)

                    const __m512i rhs_mat_014589CD_01_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_01, (_MM_PERM_ENUM)136); //B00(8-11) B01(8-11) B00(8-11) B01(8-11) B04(8-11) B05(8-11) B04(8-11) B05(8-11) B08(8-11) B09(8-11) B08(8-11) B09(8-11) B0C(8-11) B0D(8-11) B0C(8-11) B0D(8-11)
                    const __m512i rhs_mat_2367ABEF_01_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_01, (_MM_PERM_ENUM)136); //B02(8-11) B03(8-11) B02(8-11) B03(8-11) B06(8-11) B07(8-11) B06(8-11) B07(8-11) B0A(8-11) B0B(8-11) B0A(8-11) B0B(8-11) B0E(8-11) B0F(8-11) B0E(8-11) B0F(8-11)

                    const __m512i rhs_mat_014589CD_10_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_10, (_MM_PERM_ENUM)136); //B10(0-3) B11(0-3) B10(0-3) B11(0-3) B14(0-3) B15(0-3) B14(0-3) B15(0-3) B18(0-3) B19(0-3) B18(0-3) B19(0-3) B1C(0-3) B1D(0-3) B1C(0-3) B1D(0-3)
                    const __m512i rhs_mat_2367ABEF_10_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_10, (_MM_PERM_ENUM)136); //B12(0-3) B13(0-3) B12(0-3) B13(0-3) B16(0-3) B17(0-3) B16(0-3) B17(0-3) B1A(0-3) B1B(0-3) B1A(0-3) B1B(0-3) B1E(0-3) B1F(0-3) B1E(0-3) B1F(0-3)

                    const __m512i rhs_mat_014589CD_11_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_11, (_MM_PERM_ENUM)136); //B10(8-11) B11(8-11) B10(8-11) B11(8-11) B14(8-11) B15(8-11) B14(8-11) B15(8-11) B18(8-11) B19(8-11) B18(8-11) B19(8-11) B1C(8-11) B1D(8-11) B1C(8-11) B1D(8-11)
                    const __m512i rhs_mat_2367ABEF_11_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_11, (_MM_PERM_ENUM)136); //B12(8-11) B13(8-11) B12(8-11) B13(8-11) B16(8-11) B17(8-11) B16(8-11) B17(8-11) B1A(8-11) B1B(8-11) B1A(8-11) B1B(8-11) B1E(8-11) B1F(8-11) B1E(8-11) B1F(8-11)

                    const __m512i rhs_mat_014589CD_20_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_20, (_MM_PERM_ENUM)136); //B20(0-3) B21(0-3) B20(0-3) B21(0-3) B24(0-3) B25(0-3) B24(0-3) B25(0-3) B28(0-3) B29(0-3) B28(0-3) B29(0-3) B2C(0-3) B2D(0-3) B2C(0-3) B2D(0-3)
                    const __m512i rhs_mat_2367ABEF_20_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_20, (_MM_PERM_ENUM)136); //B22(0-3) B23(0-3) B22(0-3) B23(0-3) B26(0-3) B27(0-3) B26(0-3) B27(0-3) B2A(0-3) B2B(0-3) B2A(0-3) B2B(0-3) B2E(0-3) B2F(0-3) B2E(0-3) B2F(0-3)

                    const __m512i rhs_mat_014589CD_21_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_21, (_MM_PERM_ENUM)136); //B20(8-11) B21(8-11) B20(8-11) B21(8-11) B24(8-11) B25(8-11) B24(8-11) B25(8-11) B28(8-11) B29(8-11) B28(8-11) B29(8-11) B2C(8-11) B2D(8-11) B2C(8-11) B2D(8-11)
                    const __m512i rhs_mat_2367ABEF_21_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_21, (_MM_PERM_ENUM)136); //B22(8-11) B23(8-11) B22(8-11) B23(8-11) B26(8-11) B27(8-11) B26(8-11) B27(8-11) B2A(8-11) B2B(8-11) B2A(8-11) B2B(8-11) B2E(8-11) B2F(8-11) B2E(8-11) B2F(8-11)

                    const __m512i rhs_mat_014589CD_30_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_30, (_MM_PERM_ENUM)136); ///B30(0-3) B31(0-3) B30(0-3) B31(0-3) B34(0-3) B35(0-3) B34(0-3) B35(0-3) B38(0-3) B39(0-3) B38(0-3) B39(0-3) B3C(0-3) B3D(0-3) B3C(0-3) B3D(0-3)
                    const __m512i rhs_mat_2367ABEF_30_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_30, (_MM_PERM_ENUM)136); //B32(0-3) B33(0-3) B32(0-3) B33(0-3) B36(0-3) B37(0-3) B36(0-3) B37(0-3) B3A(0-3) B3B(0-3) B3A(0-3) B3B(0-3) B3E(0-3) B3F(0-3) B3E(0-3) B3F(0-3)

                    const __m512i rhs_mat_014589CD_31_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_31, (_MM_PERM_ENUM)136); //B30(8-11) B31(8-11) B30(8-11) B31(8-11) B34(8-11) B35(8-11) B34(8-11) B35(8-11) B38(8-11) B39(8-11) B38(8-11) B39(8-11) B3C(8-11) B3D(8-11) B3C(8-11) B3D(8-11)
                    const __m512i rhs_mat_2367ABEF_31_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_31, (_MM_PERM_ENUM)136); //B32(8-11) B33(8-11) B32(8-11) B33(8-11) B36(8-11) B37(8-11) B36(8-11) B37(8-11) B3A(8-11) B3B(8-11) B3A(8-11) B3B(8-11) B3E(8-11) B3F(8-11) B3E(8-11) B3F(8-11)

                    const __m512i rhs_mat_014589CD_40_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_40, (_MM_PERM_ENUM)136); //B40(0-3) B41(0-3) B40(0-3) B41(0-3) B44(0-3) B45(0-3) B44(0-3) B45(0-3) B48(0-3) B49(0-3) B48(0-3) B49(0-3) B4C(0-3) B4D(0-3) B4C(0-3) B4D(0-3)
                    const __m512i rhs_mat_2367ABEF_40_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_40, (_MM_PERM_ENUM)136); //B42(0-3) B43(0-3) B42(0-3) B43(0-3) B46(0-3) B47(0-3) B46(0-3) B47(0-3) B4A(0-3) B4B(0-3) B4A(0-3) B4B(0-3) B4E(0-3) B4F(0-3) B4E(0-3) B4F(0-3)

                    const __m512i rhs_mat_014589CD_41_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_41, (_MM_PERM_ENUM)136); //B40(8-11) B41(8-11) B40(8-11) B41(8-11) B44(8-11) B45(8-11) B44(8-11) B45(8-11) B48(8-11) B49(8-11) B48(8-11) B49(8-11) B4C(8-11) B4D(8-11) B4C(8-11) B4D(8-11)
                    const __m512i rhs_mat_2367ABEF_41_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_41, (_MM_PERM_ENUM)136); //B42(8-11) B43(8-11) B42(8-11) B43(8-11) B46(8-11) B47(8-11) B46(8-11) B47(8-11) B4A(8-11) B4B(8-11) B4A(8-11) B4B(8-11) B4E(8-11) B4F(8-11) B4E(8-11) B4F(8-11)

                    const __m512i rhs_mat_014589CD_50_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_50, (_MM_PERM_ENUM)136); //B50(0-3) B51(0-3) B50(0-3) B51(0-3) B54(0-3) B55(0-3) B54(0-3) B55(0-3) B58(0-3) B59(0-3) B58(0-3) B59(0-3) B5C(0-3) B5D(0-3) B5C(0-3) B5D(0-3)
                    const __m512i rhs_mat_2367ABEF_50_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_50, (_MM_PERM_ENUM)136); //B52(0-3) B53(0-3) B52(0-3) B53(0-3) B56(0-3) B57(0-3) B56(0-3) B57(0-3) B5A(0-3) B5B(0-3) B5A(0-3) B5B(0-3) B5E(0-3) B5F(0-3) B5E(0-3) B5F(0-3)

                    const __m512i rhs_mat_014589CD_51_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_51, (_MM_PERM_ENUM)136); //B50(8-11) B51(8-11) B50(8-11) B51(8-11) B54(8-11) B55(8-11) B54(8-11) B55(8-11) B58(8-11) B59(8-11) B58(8-11) B59(8-11) B5C(8-11) B5D(8-11) B5C(8-11) B5D(8-11)
                    const __m512i rhs_mat_2367ABEF_51_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_51, (_MM_PERM_ENUM)136); //B52(8-11) B53(8-11) B52(8-11) B53(8-11) B56(8-11) B57(8-11) B56(8-11) B57(8-11) B5A(8-11) B5B(8-11) B5A(8-11) B5B(8-11) B5E(8-11) B5F(8-11) B5E(8-11) B5F(8-11)

                    const __m512i rhs_mat_014589CD_60_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_60, (_MM_PERM_ENUM)136); //B60(0-3) B61(0-3) B60(0-3) B61(0-3) B64(0-3) B65(0-3) B64(0-3) B65(0-3) B68(0-3) B69(0-3) B68(0-3) B69(0-3) B6C(0-3) B6D(0-3) B6C(0-3) B6D(0-3)
                    const __m512i rhs_mat_2367ABEF_60_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_60, (_MM_PERM_ENUM)136); //B62(0-3) B63(0-3) B62(0-3) B63(0-3) B66(0-3) B67(0-3) B66(0-3) B67(0-3) B6A(0-3) B6B(0-3) B6A(0-3) B6B(0-3) B6E(0-3) B6F(0-3) B6E(0-3) B6F(0-3)

                    const __m512i rhs_mat_014589CD_61_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_61, (_MM_PERM_ENUM)136); //B60(8-11) B61(8-11) B60(8-11) B61(8-11) B64(8-11) B65(8-11) B64(8-11) B65(8-11) B68(8-11) B69(8-11) B68(8-11) B69(8-11) B6C(8-11) B6D(8-11) B6C(8-11) B6D(8-11)
                    const __m512i rhs_mat_2367ABEF_61_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_61, (_MM_PERM_ENUM)136); //B62(8-11) B63(8-11) B62(8-11) B63(8-11) B66(8-11) B67(8-11) B66(8-11) B67(8-11) B6A(8-11) B6B(8-11) B6A(8-11) B6B(8-11) B6E(8-11) B6F(8-11) B6E(8-11) B6F(8-11)

                    const __m512i rhs_mat_014589CD_70_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_70, (_MM_PERM_ENUM)136); //B70(0-3) B71(0-3) B70(0-3) B71(0-3) B74(0-3) B75(0-3) B74(0-3) B75(0-3) B78(0-3) B79(0-3) B78(0-3) B79(0-3) B7C(0-3) B7D(0-3) B7C(0-3) B7D(0-3)
                    const __m512i rhs_mat_2367ABEF_70_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_70, (_MM_PERM_ENUM)136); //B72(0-3) B73(0-3) B72(0-3) B73(0-3) B76(0-3) B77(0-3) B76(0-3) B77(0-3) B7A(0-3) B7B(0-3) B7A(0-3) B7B(0-3) B7E(0-3) B7F(0-3) B7E(0-3) B7F(0-3)

                    const __m512i rhs_mat_014589CD_71_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_71, (_MM_PERM_ENUM)136); //B00(8-11) B01(8-11) B00(8-11) B01(8-11) B04(8-11) B05(8-11) B04(8-11) B05(8-11) B08(8-11) B09(8-11) B08(8-11) B09(8-11) B0C(8-11) B0D(8-11) B0C(8-11) B0D(8-11)
                    const __m512i rhs_mat_2367ABEF_71_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_71, (_MM_PERM_ENUM)136); //B72(8-11) B73(8-11) B72(8-11) B73(8-11) B76(8-11) B77(8-11) B76(8-11) B77(8-11) B7A(8-11) B7B(8-11) B7A(8-11) B7B(8-11) B7E(8-11) B7F(8-11) B7E(8-11) B7F(8-11)

                    const __m512i rhs_mat_014589CD_00_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_00, (_MM_PERM_ENUM)221); //B00(4-7) B01(4-7) B00(4-7) B01(4-7) B04(4-7) B05(4-7) B04(4-7) B05(4-7) B08(4-7) B09(4-7) B08(4-7) B09(4-7) B0C(4-7) B0D(4-7) B0C(4-7) B0D(4-7)
                    const __m512i rhs_mat_2367ABEF_00_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_00, (_MM_PERM_ENUM)221); //B02(4-7) B03(4-7) B02(4-7) B03(4-7) B06(4-7) B07(4-7) B06(4-7) B07(4-7) B0A(4-7) B0B(4-7) B0A(4-7) B0B(4-7) B0E(4-7) B0F(4-7) B0E(4-7) B0F(4-7)

                    const __m512i rhs_mat_014589CD_01_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_01, (_MM_PERM_ENUM)221); //B00(12-15) B01(12-15) B00(12-15) B01(12-15) B04(12-15) B05(12-15) B04(12-15) B05(12-15) B08(12-15) B09(12-15) B08(12-15) B09(12-15) B0C(12-15) B0D(12-15) B0C(12-15) B0D(12-15)
                    const __m512i rhs_mat_2367ABEF_01_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_01, (_MM_PERM_ENUM)221); //B02(12-15) B03(12-15) B02(12-15) B03(12-15) B06(12-15) B07(12-15) B06(12-15) B07(12-15) B0A(12-15) B0B(12-15) B0A(12-15) B0B(12-15) B0E(12-15) B0F(12-15) B0E(12-15) B0F(12-15)

                    const __m512i rhs_mat_014589CD_10_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_10, (_MM_PERM_ENUM)221); //B10(4-7) B11(4-7) B10(4-7) B11(4-7) B14(4-7) B15(4-7) B14(4-7) B15(4-7) B18(4-7) B19(4-7) B18(4-7) B19(4-7) B1C(4-7) B1D(4-7) B1C(4-7) B1D(4-7)
                    const __m512i rhs_mat_2367ABEF_10_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_10, (_MM_PERM_ENUM)221); //B12(4-7) B13(4-7) B12(4-7) B13(4-7) B16(4-7) B17(4-7) B16(4-7) B17(4-7) B1A(4-7) B1B(4-7) B1A(4-7) B1B(4-7) B1E(4-7) B1F(4-7) B1E(4-7) B1F(4-7)

                    const __m512i rhs_mat_014589CD_11_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_11, (_MM_PERM_ENUM)221); //B10(12-15) B11(12-15) B10(12-15) B11(12-15) B14(12-15) B15(12-15) B14(12-15) B15(12-15) B18(12-15) B19(12-15) B18(12-15) B19(12-15) B1C(12-15) B1D(12-15) B1C(12-15) B1D(12-15)
                    const __m512i rhs_mat_2367ABEF_11_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_11, (_MM_PERM_ENUM)221); //B12(12-15) B13(12-15) B12(12-15) B13(12-15) B16(12-15) B17(12-15) B16(12-15) B17(12-15) B1A(12-15) B1B(12-15) B1A(12-15) B1B(12-15) B1E(12-15) B1F(12-15) B1E(12-15) B1F(12-15)

                    const __m512i rhs_mat_014589CD_20_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_20, (_MM_PERM_ENUM)221); //B20(4-7) B21(4-7) B20(4-7) B21(4-7) B24(4-7) B25(4-7) B24(4-7) B25(4-7) B28(4-7) B29(4-7) B28(4-7) B29(4-7) B2C(4-7) B2D(4-7) B2C(4-7) B2D(4-7)
                    const __m512i rhs_mat_2367ABEF_20_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_20, (_MM_PERM_ENUM)221); //B22(4-7) B23(4-7) B22(4-7) B23(4-7) B26(4-7) B27(4-7) B26(4-7) B27(4-7) B2A(4-7) B2B(4-7) B2A(4-7) B2B(4-7) B2E(4-7) B2F(4-7) B2E(4-7) B2F(4-7)

                    const __m512i rhs_mat_014589CD_21_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_21, (_MM_PERM_ENUM)221); //B20(12-15) B21(12-15) B20(12-15) B21(12-15) B24(12-15) B25(12-15) B24(12-15) B25(12-15) B28(12-15) B29(12-15) B28(12-15) B29(12-15) B2C(12-15) B2D(12-15) B2C(12-15) B2D(12-15)
                    const __m512i rhs_mat_2367ABEF_21_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_21, (_MM_PERM_ENUM)221); //B22(12-15) B23(12-15) B22(12-15) B23(12-15) B26(12-15) B27(12-15) B26(12-15) B27(12-15) B2A(12-15) B2B(12-15) B2A(12-15) B2B(12-15) B2E(12-15) B2F(12-15) B2E(12-15) B2F(12-15)

                    const __m512i rhs_mat_014589CD_30_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_30, (_MM_PERM_ENUM)221); //B30(4-7) B31(4-7) B30(4-7) B31(4-7) B34(4-7) B35(4-7) B34(4-7) B35(4-7) B38(4-7) B39(4-7) B38(4-7) B39(4-7) B3C(4-7) B3D(4-7) B3C(4-7) B3D(4-7)
                    const __m512i rhs_mat_2367ABEF_30_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_30, (_MM_PERM_ENUM)221); //B32(4-7) B33(4-7) B32(4-7) B33(4-7) B36(4-7) B37(4-7) B36(4-7) B37(4-7) B3A(4-7) B3B(4-7) B3A(4-7) B3B(4-7) B3E(4-7) B3F(4-7) B3E(4-7) B3F(4-7)

                    const __m512i rhs_mat_014589CD_31_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_31, (_MM_PERM_ENUM)221); //B30(12-15) B31(12-15) B30(12-15) B31(12-15) B34(12-15) B35(12-15) B34(12-15) B35(12-15) B38(12-15) B39(12-15) B38(12-15) B39(12-15) B3C(12-15) B3D(12-15) B3C(12-15) B3D(12-15)
                    const __m512i rhs_mat_2367ABEF_31_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_31, (_MM_PERM_ENUM)221); //B32(12-15) B33(12-15) B32(12-15) B33(12-15) B36(12-15) B37(12-15) B36(12-15) B37(12-15) B3A(12-15) B3B(12-15) B3A(12-15) B3B(12-15) B3E(12-15) B3F(12-15) B3E(12-15) B3F(12-15)

                    const __m512i rhs_mat_014589CD_40_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_40, (_MM_PERM_ENUM)221); //B40(4-7) B41(4-7) B40(4-7) B41(4-7) B44(4-7) B45(4-7) B44(4-7) B45(4-7) B48(4-7) B49(4-7) B48(4-7) B49(4-7) B4C(4-7) B4D(4-7) B4C(4-7) B4D(4-7)
                    const __m512i rhs_mat_2367ABEF_40_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_40, (_MM_PERM_ENUM)221); //B42(4-7) B43(4-7) B42(4-7) B43(4-7) B46(4-7) B47(4-7) B46(4-7) B47(4-7) B4A(4-7) B4B(4-7) B4A(4-7) B4B(4-7) B4E(4-7) B4F(4-7) B4E(4-7) B4F(4-7)

                    const __m512i rhs_mat_014589CD_41_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_41, (_MM_PERM_ENUM)221); //B40(12-15) B41(12-15) B40(12-15) B41(12-15) B44(12-15) B45(12-15) B44(12-15) B45(12-15) B48(12-15) B49(12-15) B48(12-15) B49(12-15) B4C(12-15) B4D(12-15) B4C(12-15) B4D(12-15)
                    const __m512i rhs_mat_2367ABEF_41_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_41, (_MM_PERM_ENUM)221); //B42(12-15) B43(12-15) B42(12-15) B43(12-15) B46(12-15) B47(12-15) B46(12-15) B47(12-15) B4A(12-15) B4B(12-15) B4A(12-15) B4B(12-15) B4E(12-15) B4F(12-15) B4E(12-15) B4F(12-15)

                    const __m512i rhs_mat_014589CD_50_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_50, (_MM_PERM_ENUM)221); //B50(4-7) B51(4-7) B50(4-7) B51(4-7) B54(4-7) B55(4-7) B54(4-7) B55(4-7) B58(4-7) B59(4-7) B58(4-7) B59(4-7) B5C(4-7) B5D(4-7) B5C(4-7) B5D(4-7)
                    const __m512i rhs_mat_2367ABEF_50_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_50, (_MM_PERM_ENUM)221); //B52(4-7) B53(4-7) B52(4-7) B53(4-7) B56(4-7) B57(4-7) B56(4-7) B57(4-7) B5A(4-7) B5B(4-7) B5A(4-7) B5B(4-7) B5E(4-7) B5F(4-7) B5E(4-7) B5F(4-7)

                    const __m512i rhs_mat_014589CD_51_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_51, (_MM_PERM_ENUM)221); //B50(12-15) B51(12-15) B50(12-15) B51(12-15) B54(12-15) B55(12-15) B54(12-15) B55(12-15) B58(12-15) B59(12-15) B58(12-15) B59(12-15) B5C(12-15) B5D(12-15) B5C(12-15) B5D(12-15)
                    const __m512i rhs_mat_2367ABEF_51_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_51, (_MM_PERM_ENUM)221); //B52(12-15) B53(12-15) B52(12-15) B53(12-15) B56(12-15) B57(12-15) B56(12-15) B57(12-15) B5A(12-15) B5B(12-15) B5A(12-15) B5B(12-15) B5E(12-15) B5F(12-15) B5E(12-15) B5F(12-15)

                    const __m512i rhs_mat_014589CD_60_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_60, (_MM_PERM_ENUM)221); //B60(4-7) B61(4-7) B60(4-7) B61(4-7) B64(4-7) B65(4-7) B64(4-7) B65(4-7) B68(4-7) B69(4-7) B68(4-7) B69(4-7) B6C(4-7) B6D(4-7) B6C(4-7) B6D(4-7)
                    const __m512i rhs_mat_2367ABEF_60_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_60, (_MM_PERM_ENUM)221); //B62(4-7) B63(4-7) B62(4-7) B63(4-7) B66(4-7) B67(4-7) B66(4-7) B67(4-7) B6A(4-7) B6B(4-7) B6A(4-7) B6B(4-7) B6E(4-7) B6F(4-7) B6E(4-7) B6F(4-7)

                    const __m512i rhs_mat_014589CD_61_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_61, (_MM_PERM_ENUM)221); //B60(12-15) B61(12-15) B60(12-15) B61(12-15) B64(12-15) B65(12-15) B64(12-15) B65(12-15) B68(12-15) B69(12-15) B68(12-15) B69(12-15) B6C(12-15) B6D(12-15) B6C(12-15) B6D(12-15)
                    const __m512i rhs_mat_2367ABEF_61_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_61, (_MM_PERM_ENUM)221); //B62(12-15) B63(12-15) B62(12-15) B63(12-15) B66(12-15) B67(12-15) B66(12-15) B67(12-15) B6A(12-15) B6B(12-15) B6A(12-15) B6B(12-15) B6E(12-15) B6F(12-15) B6E(12-15) B6F(12-15)

                    const __m512i rhs_mat_014589CD_70_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_70, (_MM_PERM_ENUM)221); //B70(4-7) B71(4-7) B70(4-7) B71(4-7) B74(4-7) B75(4-7) B74(4-7) B75(4-7) B78(4-7) B79(4-7) B78(4-7) B79(4-7) B7C(4-7) B7D(4-7) B7C(4-7) B7D(4-7)
                    const __m512i rhs_mat_2367ABEF_70_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_70, (_MM_PERM_ENUM)221); //B72(4-7) B73(4-7) B72(4-7) B73(4-7) B76(4-7) B77(4-7) B76(4-7) B77(4-7) B7A(4-7) B7B(4-7) B7A(4-7) B7B(4-7) B7E(4-7) B7F(4-7) B7E(4-7) B7F(4-7)

                    const __m512i rhs_mat_014589CD_71_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_71, (_MM_PERM_ENUM)221); //B70(12-15) B71(12-15) B70(12-15) B71(12-15) B74(12-15) B75(12-15) B74(12-15) B75(12-15) B78(12-15) B79(12-15) B78(12-15) B79(12-15) B7C(12-15) B7D(12-15) B7C(12-15) B7D(12-15)
                    const __m512i rhs_mat_2367ABEF_71_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_71, (_MM_PERM_ENUM)221); //B72(12-15) B73(12-15) B72(12-15) B73(12-15) B76(12-15) B77(12-15) B76(12-15) B77(12-15) B7A(12-15) B7B(12-15) B7A(12-15) B7B(12-15) B7E(12-15) B7F(12-15) B7E(12-15) B7F(12-15)

                    //notation:superblock subblock
                    //s00 m00  s01 m01   s10 m10  s11 m11  s20 m20  s21 m21   s30 m30  s31 m31  s40 m40  s41 m41   s50 m50  s51 m51  s60 m60  s61 m61   s70 m70  s71 m71

                    const __m128i mins_and_scales_01_0 = _mm_loadu_si128((const __m128i *)(b_ptr_0[b].scales + sb * 64));
                    const __m128i mins_and_scales_23_0 = _mm_loadu_si128((const __m128i *)(b_ptr_0[b].scales + 16 + sb * 64));
                    const __m128i mins_and_scales_45_0 = _mm_loadu_si128((const __m128i *)(b_ptr_0[b].scales + 32 + sb * 64));
                    const __m128i mins_and_scales_67_0 = _mm_loadu_si128((const __m128i *)(b_ptr_0[b].scales + 48 + sb * 64));

                    const __m128i mins_and_scales_01_1 = _mm_loadu_si128((const __m128i *)(b_ptr_1[b].scales + sb * 64));
                    const __m128i mins_and_scales_23_1 = _mm_loadu_si128((const __m128i *)(b_ptr_1[b].scales + 16 + sb * 64));
                    const __m128i mins_and_scales_45_1 = _mm_loadu_si128((const __m128i *)(b_ptr_1[b].scales + 32 + sb * 64));
                    const __m128i mins_and_scales_67_1 = _mm_loadu_si128((const __m128i *)(b_ptr_1[b].scales + 48 + sb * 64));

                    // Combine mins and scales for sub-blocks: 0-1, 2-3, 4-5, 6-7 in the sb loop
                    const __m256i mins_and_scales_01 = _mm256_insertf128_si256(_mm256_castsi128_si256(mins_and_scales_01_0), mins_and_scales_01_1, 1);
                    const __m256i mins_and_scales_23 = _mm256_insertf128_si256(_mm256_castsi128_si256(mins_and_scales_23_0), mins_and_scales_23_1, 1);
                    const __m256i mins_and_scales_45 = _mm256_insertf128_si256(_mm256_castsi128_si256(mins_and_scales_45_0), mins_and_scales_45_1, 1);
                    const __m256i mins_and_scales_67 = _mm256_insertf128_si256(_mm256_castsi128_si256(mins_and_scales_67_0), mins_and_scales_67_1, 1);

                    // Extract scales which is lower half from mins_and_scales
                    const __m256i scales_01 = _mm256_and_si256(mins_and_scales_01, m4b);
                    const __m256i scales_23 = _mm256_and_si256(mins_and_scales_23, m4b);
                    const __m256i scales_45 = _mm256_and_si256(mins_and_scales_45, m4b);
                    const __m256i scales_67 = _mm256_and_si256(mins_and_scales_67, m4b);

                    // Extract mins which is upper half from mins_and_scales
                    const __m512i mins_01 = _mm512_cvtepu8_epi16(_mm256_and_si256(_mm256_srli_epi16(mins_and_scales_01, 4), m4b));
                    const __m512i mins_23 = _mm512_cvtepu8_epi16(_mm256_and_si256(_mm256_srli_epi16(mins_and_scales_23, 4), m4b));
                    const __m512i mins_45 = _mm512_cvtepu8_epi16(_mm256_and_si256(_mm256_srli_epi16(mins_and_scales_45, 4), m4b));
                    const __m512i mins_67 = _mm512_cvtepu8_epi16(_mm256_and_si256(_mm256_srli_epi16(mins_and_scales_67, 4), m4b));

                    const __m512i scales_0 = _mm512_cvtepu8_epi16(_mm256_shuffle_epi8(scales_01,scalesmask1));
                    const __m512i scales_1 = _mm512_cvtepu8_epi16(_mm256_shuffle_epi8(scales_01,scalesmask2));
                    const __m512i scales_2 = _mm512_cvtepu8_epi16(_mm256_shuffle_epi8(scales_23,scalesmask1));
                    const __m512i scales_3 = _mm512_cvtepu8_epi16(_mm256_shuffle_epi8(scales_23,scalesmask2));
                    const __m512i scales_4 = _mm512_cvtepu8_epi16(_mm256_shuffle_epi8(scales_45,scalesmask1));
                    const __m512i scales_5 = _mm512_cvtepu8_epi16(_mm256_shuffle_epi8(scales_45,scalesmask2));
                    const __m512i scales_6 = _mm512_cvtepu8_epi16(_mm256_shuffle_epi8(scales_67,scalesmask1));
                    const __m512i scales_7 = _mm512_cvtepu8_epi16(_mm256_shuffle_epi8(scales_67,scalesmask2));

                    const __m512i scale_014589CD_0 = _mm512_shuffle_epi32(scales_0, (_MM_PERM_ENUM)68);
                    const __m512i scale_2367ABEF_0 = _mm512_shuffle_epi32(scales_0, (_MM_PERM_ENUM)238);

                    const __m512i scale_014589CD_1 = _mm512_shuffle_epi32(scales_1, (_MM_PERM_ENUM)68);
                    const __m512i scale_2367ABEF_1 = _mm512_shuffle_epi32(scales_1, (_MM_PERM_ENUM)238);

                    const __m512i scale_014589CD_2 = _mm512_shuffle_epi32(scales_2, (_MM_PERM_ENUM)68);
                    const __m512i scale_2367ABEF_2 = _mm512_shuffle_epi32(scales_2, (_MM_PERM_ENUM)238);

                    const __m512i scale_014589CD_3 = _mm512_shuffle_epi32(scales_3, (_MM_PERM_ENUM)68);
                    const __m512i scale_2367ABEF_3 = _mm512_shuffle_epi32(scales_3, (_MM_PERM_ENUM)238);

                    const __m512i scale_014589CD_4 = _mm512_shuffle_epi32(scales_4, (_MM_PERM_ENUM)68);
                    const __m512i scale_2367ABEF_4 = _mm512_shuffle_epi32(scales_4, (_MM_PERM_ENUM)238);

                    const __m512i scale_014589CD_5 = _mm512_shuffle_epi32(scales_5, (_MM_PERM_ENUM)68);
                    const __m512i scale_2367ABEF_5 = _mm512_shuffle_epi32(scales_5, (_MM_PERM_ENUM)238);

                    const __m512i scale_014589CD_6 = _mm512_shuffle_epi32(scales_6, (_MM_PERM_ENUM)68);
                    const __m512i scale_2367ABEF_6 = _mm512_shuffle_epi32(scales_6, (_MM_PERM_ENUM)238);

                    const __m512i scale_014589CD_7 = _mm512_shuffle_epi32(scales_7, (_MM_PERM_ENUM)68);
                    const __m512i scale_2367ABEF_7 = _mm512_shuffle_epi32(scales_7, (_MM_PERM_ENUM)238);


                    for (int rp = 0; rp < 4; rp++) {

                        // Load the four block_q8_k quantized values interleaved with each other in chunks of eight bytes - A0,A1,A2,A3
                        // Loaded as set of 128 bit vectors and repeated and stored into a 256 bit vector before again repeating into 512 bit vector
                        __m256i lhs_mat_ymm_0123_00 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 512 * sb)));
                        __m256i lhs_mat_ymm_01_00 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_00, lhs_mat_ymm_0123_00, 0);
                        __m256i lhs_mat_ymm_23_00 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_00, lhs_mat_ymm_0123_00, 17);
                        __m256i lhs_mat_ymm_0123_01 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 32 + 512 * sb)));
                        __m256i lhs_mat_ymm_01_01 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_01, lhs_mat_ymm_0123_01, 0);
                        __m256i lhs_mat_ymm_23_01 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_01, lhs_mat_ymm_0123_01, 17);
                        __m256i lhs_mat_ymm_0123_10 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 64 + 512 * sb)));
                        __m256i lhs_mat_ymm_01_10 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_10, lhs_mat_ymm_0123_10, 0);
                        __m256i lhs_mat_ymm_23_10 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_10, lhs_mat_ymm_0123_10, 17);
                        __m256i lhs_mat_ymm_0123_11 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 96 + 512 * sb)));
                        __m256i lhs_mat_ymm_01_11 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_11, lhs_mat_ymm_0123_11, 0);
                        __m256i lhs_mat_ymm_23_11 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_11, lhs_mat_ymm_0123_11, 17);
                        __m256i lhs_mat_ymm_0123_20 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 128 + 512 * sb)));
                        __m256i lhs_mat_ymm_01_20 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_20, lhs_mat_ymm_0123_20, 0);
                        __m256i lhs_mat_ymm_23_20 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_20, lhs_mat_ymm_0123_20, 17);
                        __m256i lhs_mat_ymm_0123_21 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 160 + 512 * sb)));
                        __m256i lhs_mat_ymm_01_21 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_21, lhs_mat_ymm_0123_21, 0);
                        __m256i lhs_mat_ymm_23_21 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_21, lhs_mat_ymm_0123_21, 17);
                        __m256i lhs_mat_ymm_0123_30 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 192 + 512 * sb)));
                        __m256i lhs_mat_ymm_01_30 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_30, lhs_mat_ymm_0123_30, 0);
                        __m256i lhs_mat_ymm_23_30 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_30, lhs_mat_ymm_0123_30, 17);
                        __m256i lhs_mat_ymm_0123_31 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 224 + 512 * sb)));
                        __m256i lhs_mat_ymm_01_31 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_31, lhs_mat_ymm_0123_31, 0);
                        __m256i lhs_mat_ymm_23_31 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_31, lhs_mat_ymm_0123_31, 17);

                        __m256i lhs_mat_ymm_0123_40 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 256 + 512 * sb)));
                        __m256i lhs_mat_ymm_01_40 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_40, lhs_mat_ymm_0123_40, 0);
                        __m256i lhs_mat_ymm_23_40 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_40, lhs_mat_ymm_0123_40, 17);
                        __m256i lhs_mat_ymm_0123_41 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 288 + 512 * sb)));
                        __m256i lhs_mat_ymm_01_41 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_41, lhs_mat_ymm_0123_41, 0);
                        __m256i lhs_mat_ymm_23_41 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_41, lhs_mat_ymm_0123_41, 17);
                        __m256i lhs_mat_ymm_0123_50 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 320 + 512 * sb)));
                        __m256i lhs_mat_ymm_01_50 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_50, lhs_mat_ymm_0123_50, 0);
                        __m256i lhs_mat_ymm_23_50 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_50, lhs_mat_ymm_0123_50, 17);
                        __m256i lhs_mat_ymm_0123_51 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 352 + 512 * sb)));
                        __m256i lhs_mat_ymm_01_51 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_51, lhs_mat_ymm_0123_51, 0);
                        __m256i lhs_mat_ymm_23_51 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_51, lhs_mat_ymm_0123_51, 17);
                        __m256i lhs_mat_ymm_0123_60 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 384 + 512 * sb)));
                        __m256i lhs_mat_ymm_01_60 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_60, lhs_mat_ymm_0123_60, 0);
                        __m256i lhs_mat_ymm_23_60 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_60, lhs_mat_ymm_0123_60, 17);
                        __m256i lhs_mat_ymm_0123_61 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 416 + 512 * sb)));
                        __m256i lhs_mat_ymm_01_61 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_61, lhs_mat_ymm_0123_61, 0);
                        __m256i lhs_mat_ymm_23_61 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_61, lhs_mat_ymm_0123_61, 17);
                        __m256i lhs_mat_ymm_0123_70 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 448 + 512 * sb)));
                        __m256i lhs_mat_ymm_01_70 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_70, lhs_mat_ymm_0123_70, 0);
                        __m256i lhs_mat_ymm_23_70 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_70, lhs_mat_ymm_0123_70, 17);
                        __m256i lhs_mat_ymm_0123_71 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 480 + 512 * sb)));
                        __m256i lhs_mat_ymm_01_71 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_71, lhs_mat_ymm_0123_71, 0);
                        __m256i lhs_mat_ymm_23_71 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_71, lhs_mat_ymm_0123_71, 17);


                        __m512i lhs_mat_01_00 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_00), lhs_mat_ymm_01_00, 1);
                        __m512i lhs_mat_23_00 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_00), lhs_mat_ymm_23_00, 1);
                        __m512i lhs_mat_01_01 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_01), lhs_mat_ymm_01_01, 1);
                        __m512i lhs_mat_23_01 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_01), lhs_mat_ymm_23_01, 1);

                        __m512i lhs_mat_01_10 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_10), lhs_mat_ymm_01_10, 1);
                        __m512i lhs_mat_23_10 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_10), lhs_mat_ymm_23_10, 1);
                        __m512i lhs_mat_01_11 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_11), lhs_mat_ymm_01_11, 1);
                        __m512i lhs_mat_23_11 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_11), lhs_mat_ymm_23_11, 1);

                        __m512i lhs_mat_01_20 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_20), lhs_mat_ymm_01_20, 1);
                        __m512i lhs_mat_23_20 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_20), lhs_mat_ymm_23_20, 1);
                        __m512i lhs_mat_01_21 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_21), lhs_mat_ymm_01_21, 1);
                        __m512i lhs_mat_23_21 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_21), lhs_mat_ymm_23_21, 1);

                        __m512i lhs_mat_01_30 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_30), lhs_mat_ymm_01_30, 1);
                        __m512i lhs_mat_23_30 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_30), lhs_mat_ymm_23_30, 1);
                        __m512i lhs_mat_01_31 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_31), lhs_mat_ymm_01_31, 1);
                        __m512i lhs_mat_23_31 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_31), lhs_mat_ymm_23_31, 1);

                        __m512i lhs_mat_01_40 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_40), lhs_mat_ymm_01_40, 1);
                        __m512i lhs_mat_23_40 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_40), lhs_mat_ymm_23_40, 1);
                        __m512i lhs_mat_01_41 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_41), lhs_mat_ymm_01_41, 1);
                        __m512i lhs_mat_23_41 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_41), lhs_mat_ymm_23_41, 1);

                        __m512i lhs_mat_01_50 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_50), lhs_mat_ymm_01_50, 1);
                        __m512i lhs_mat_23_50 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_50), lhs_mat_ymm_23_50, 1);
                        __m512i lhs_mat_01_51 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_51), lhs_mat_ymm_01_51, 1);
                        __m512i lhs_mat_23_51 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_51), lhs_mat_ymm_23_51, 1);

                        __m512i lhs_mat_01_60 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_60), lhs_mat_ymm_01_60, 1);
                        __m512i lhs_mat_23_60 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_60), lhs_mat_ymm_23_60, 1);
                        __m512i lhs_mat_01_61 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_61), lhs_mat_ymm_01_61, 1);
                        __m512i lhs_mat_23_61 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_61), lhs_mat_ymm_23_61, 1);

                        __m512i lhs_mat_01_70 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_70), lhs_mat_ymm_01_70, 1);
                        __m512i lhs_mat_23_70 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_70), lhs_mat_ymm_23_70, 1);
                        __m512i lhs_mat_01_71 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_71), lhs_mat_ymm_01_71, 1);
                        __m512i lhs_mat_23_71 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_71), lhs_mat_ymm_23_71, 1);

                        // Bsums are loaded for the different Q8_K blocks
                        __m128i lhs_raw_bsums_01_0123 = _mm_loadu_si128((const __m128i *)((a_ptrs[rp][b].bsums + 32 * sb)));
                        __m128i lhs_raw_bsums_23_0123 = _mm_loadu_si128((const __m128i *)(a_ptrs[rp][b].bsums + 8 + 32 * sb));
                        __m128i lhs_raw_bsums_01_4567 = _mm_loadu_si128((const __m128i *)((a_ptrs[rp][b].bsums + 16 + 32 * sb)));
                        __m128i lhs_raw_bsums_23_4567 = _mm_loadu_si128((const __m128i *)(a_ptrs[rp][b].bsums + 24 + 32 * sb));

                        __m256i lhs_bsums_ymm_01_0123 = _mm256_inserti128_si256(_mm256_castsi128_si256(lhs_raw_bsums_01_0123), lhs_raw_bsums_01_0123, 1);
                        __m512i lhs_bsums_01_0123 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_bsums_ymm_01_0123), lhs_bsums_ymm_01_0123, 1);
                        __m256i lhs_bsums_ymm_23_0123 = _mm256_inserti128_si256(_mm256_castsi128_si256(lhs_raw_bsums_23_0123), lhs_raw_bsums_23_0123, 1);
                        __m512i lhs_bsums_23_0123 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_bsums_ymm_23_0123), lhs_bsums_ymm_23_0123, 1);                        __m256i lhs_bsums_ymm_01_4567 = _mm256_inserti128_si256(_mm256_castsi128_si256(lhs_raw_bsums_01_4567), lhs_raw_bsums_01_4567, 1);
                        __m512i lhs_bsums_01_4567 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_bsums_ymm_01_4567), lhs_bsums_ymm_01_4567, 1);
                        __m256i lhs_bsums_ymm_23_4567 = _mm256_inserti128_si256(_mm256_castsi128_si256(lhs_raw_bsums_23_4567), lhs_raw_bsums_23_4567, 1);
                        __m512i lhs_bsums_23_4567 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_bsums_ymm_23_4567), lhs_bsums_ymm_23_4567, 1);

                        // Shuffle pattern one - left side input
                        const __m512i lhs_mat_01_00_sp1 = _mm512_shuffle_epi32(lhs_mat_01_00, (_MM_PERM_ENUM)160); //A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3) A00(0-3) A01(0-3) A01(0-3)
                        const __m512i lhs_mat_23_00_sp1 = _mm512_shuffle_epi32(lhs_mat_23_00, (_MM_PERM_ENUM)160); //A02(0-3) A02(0-3) A03(0-3) A03(0-3) A02(0-3) A02(0-3) A03(0-3) A03(0-3) A02(0-3) A02(0-3) A03(0-3) A03(0-3) A02(0-3) A02(0-3) A03(0-3) A03(0-3)

                        const __m512i lhs_mat_01_01_sp1 = _mm512_shuffle_epi32(lhs_mat_01_01, (_MM_PERM_ENUM)160); //A00(8-11) A00(8-11) A01(8-11) A01(8-11) A00(8-11) A00(8-11) A01(8-11) A01(8-11) A00(8-11) A00(8-11) A01(8-11) A01(8-11) A00(8-11) A00(8-11) A01(8-11) A01(8-11)
                        const __m512i lhs_mat_23_01_sp1 = _mm512_shuffle_epi32(lhs_mat_23_01, (_MM_PERM_ENUM)160); //A02(8-11) A02(8-11) A03(8-11) A03(8-11) A02(8-11) A02(8-11) A03(8-11) A03(8-11) A02(8-11) A02(8-11) A03(8-11) A03(8-11) A02(8-11) A02(8-11) A03(8-11) A03(8-11)

                        const __m512i lhs_mat_01_10_sp1 = _mm512_shuffle_epi32(lhs_mat_01_10, (_MM_PERM_ENUM)160); //A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3) A10(0-3) A11(0-3) A11(0-3)
                        const __m512i lhs_mat_23_10_sp1 = _mm512_shuffle_epi32(lhs_mat_23_10, (_MM_PERM_ENUM)160); //A12(0-3) A12(0-3) A13(0-3) A13(0-3) A12(0-3) A12(0-3) A13(0-3) A13(0-3) A12(0-3) A12(0-3) A13(0-3) A13(0-3) A12(0-3) A12(0-3) A13(0-3) A13(0-3)

                        const __m512i lhs_mat_01_11_sp1 = _mm512_shuffle_epi32(lhs_mat_01_11, (_MM_PERM_ENUM)160); //A10(8-11) A10(8-11) A11(8-11) A11(8-11) A10(8-11) A10(8-11) A11(8-11) A11(8-11) A10(8-11) A10(8-11) A11(8-11) A11(8-11) A10(8-11) A10(8-11) A11(8-11) A11(8-11)
                        const __m512i lhs_mat_23_11_sp1 = _mm512_shuffle_epi32(lhs_mat_23_11, (_MM_PERM_ENUM)160); //A12(8-11) A12(8-11) A13(8-11) A13(8-11) A12(8-11) A12(8-11) A13(8-11) A13(8-11) A12(8-11) A12(8-11) A13(8-11) A13(8-11) A12(8-11) A12(8-11) A13(8-11) A13(8-11)

                        const __m512i lhs_mat_01_20_sp1 = _mm512_shuffle_epi32(lhs_mat_01_20, (_MM_PERM_ENUM)160); //A20(0-3) A20(0-3) A21(0-3) A21(0-3) A20(0-3) A20(0-3) A21(0-3) A21(0-3) A20(0-3) A20(0-3) A21(0-3) A21(0-3) A20(0-3) A20(0-3) A21(0-3) A21(0-3)
                        const __m512i lhs_mat_23_20_sp1 = _mm512_shuffle_epi32(lhs_mat_23_20, (_MM_PERM_ENUM)160); //A22(0-3) A22(0-3) A23(0-3) A23(0-3) A22(0-3) A22(0-3) A23(0-3) A23(0-3) A22(0-3) A22(0-3) A23(0-3) A23(0-3) A22(0-3) A22(0-3) A23(0-3) A23(0-3)

                        const __m512i lhs_mat_01_21_sp1 = _mm512_shuffle_epi32(lhs_mat_01_21, (_MM_PERM_ENUM)160); //A20(8-11) A20(8-11) A21(8-11) A21(8-11) A20(8-11) A20(8-11) A21(8-11) A21(8-11) A20(8-11) A20(8-11) A21(8-11) A21(8-11) A20(8-11) A20(8-11) A21(8-11) A21(8-11)
                        const __m512i lhs_mat_23_21_sp1 = _mm512_shuffle_epi32(lhs_mat_23_21, (_MM_PERM_ENUM)160); //A22(8-11) A22(8-11) A23(8-11) A23(8-11) A22(8-11) A22(8-11) A23(8-11) A23(8-11) A22(8-11) A22(8-11) A23(8-11) A23(8-11) A22(8-11) A22(8-11) A23(8-11) A23(8-11)

                        const __m512i lhs_mat_01_30_sp1 = _mm512_shuffle_epi32(lhs_mat_01_30, (_MM_PERM_ENUM)160); //A30(0-3) A30(0-3) A31(0-3) A31(0-3) A30(0-3) A30(0-3) A31(0-3) A31(0-3) A30(0-3) A30(0-3) A31(0-3) A31(0-3) A30(0-3) A30(0-3) A31(0-3) A31(0-3)
                        const __m512i lhs_mat_23_30_sp1 = _mm512_shuffle_epi32(lhs_mat_23_30, (_MM_PERM_ENUM)160); //A32(0-3) A32(0-3) A33(0-3) A33(0-3) A32(0-3) A32(0-3) A33(0-3) A33(0-3) A32(0-3) A32(0-3) A33(0-3) A33(0-3) A32(0-3) A32(0-3) A33(0-3) A33(0-3)

                        const __m512i lhs_mat_01_31_sp1 = _mm512_shuffle_epi32(lhs_mat_01_31, (_MM_PERM_ENUM)160); //A30(8-11) A30(8-11) A31(8-11) A31(8-11) A30(8-11) A30(8-11) A31(8-11) A31(8-11) A30(8-11) A30(8-11) A31(8-11) A31(8-11) A30(8-11) A30(8-11) A31(8-11) A31(8-11)
                        const __m512i lhs_mat_23_31_sp1 = _mm512_shuffle_epi32(lhs_mat_23_31, (_MM_PERM_ENUM)160); //A32(8-11) A32(8-11) A33(8-11) A33(8-11) A32(8-11) A32(8-11) A33(8-11) A33(8-11) A32(8-11) A32(8-11) A33(8-11) A33(8-11) A32(8-11) A32(8-11) A33(8-11) A33(8-11)

                        const __m512i lhs_mat_01_40_sp1 = _mm512_shuffle_epi32(lhs_mat_01_40, (_MM_PERM_ENUM)160); //A40(0-3) A40(0-3) A41(0-3) A41(0-3) A40(0-3) A40(0-3) A41(0-3) A41(0-3) A40(0-3) A40(0-3) A41(0-3) A41(0-3) A40(0-3) A40(0-3) A41(0-3) A41(0-3)
                        const __m512i lhs_mat_23_40_sp1 = _mm512_shuffle_epi32(lhs_mat_23_40, (_MM_PERM_ENUM)160); //A42(0-3) A42(0-3) A43(0-3) A43(0-3) A42(0-3) A42(0-3) A43(0-3) A43(0-3) A42(0-3) A42(0-3) A43(0-3) A43(0-3) A42(0-3) A42(0-3) A43(0-3) A43(0-3)

                        const __m512i lhs_mat_01_41_sp1 = _mm512_shuffle_epi32(lhs_mat_01_41, (_MM_PERM_ENUM)160); //A40(8-11) A40(8-11) A41(8-11) A41(8-11) A40(8-11) A40(8-11) A41(8-11) A41(8-11) A40(8-11) A40(8-11) A41(8-11) A41(8-11) A40(8-11) A40(8-11) A41(8-11) A41(8-11)
                        const __m512i lhs_mat_23_41_sp1 = _mm512_shuffle_epi32(lhs_mat_23_41, (_MM_PERM_ENUM)160); //A42(8-11) A42(8-11) A43(8-11) A43(8-11) A42(8-11) A42(8-11) A43(8-11) A43(8-11) A42(8-11) A42(8-11) A43(8-11) A43(8-11) A42(8-11) A42(8-11) A43(8-11) A43(8-11)

                        const __m512i lhs_mat_01_50_sp1 = _mm512_shuffle_epi32(lhs_mat_01_50, (_MM_PERM_ENUM)160); //A50(0-3) A50(0-3) A51(0-3) A51(0-3) A50(0-3) A50(0-3) A51(0-3) A51(0-3) A50(0-3) A50(0-3) A51(0-3) A51(0-3) A50(0-3) A50(0-3) A51(0-3) A51(0-3)
                        const __m512i lhs_mat_23_50_sp1 = _mm512_shuffle_epi32(lhs_mat_23_50, (_MM_PERM_ENUM)160); //A52(0-3) A52(0-3) A53(0-3) A53(0-3) A52(0-3) A52(0-3) A53(0-3) A53(0-3) A52(0-3) A52(0-3) A53(0-3) A53(0-3) A52(0-3) A52(0-3) A53(0-3) A53(0-3)

                        const __m512i lhs_mat_01_51_sp1 = _mm512_shuffle_epi32(lhs_mat_01_51, (_MM_PERM_ENUM)160); //A50(8-11) A50(8-11) A51(8-11) A51(8-11) A50(8-11) A50(8-11) A51(8-11) A51(8-11) A50(8-11) A50(8-11) A51(8-11) A51(8-11) A50(8-11) A50(8-11) A51(8-11) A51(8-11)
                        const __m512i lhs_mat_23_51_sp1 = _mm512_shuffle_epi32(lhs_mat_23_51, (_MM_PERM_ENUM)160); //A52(8-11) A52(8-11) A53(8-11) A53(8-11) A52(8-11) A52(8-11) A53(8-11) A53(8-11) A52(8-11) A52(8-11) A53(8-11) A53(8-11) A52(8-11) A52(8-11) A53(8-11) A53(8-11)

                        const __m512i lhs_mat_01_60_sp1 = _mm512_shuffle_epi32(lhs_mat_01_60, (_MM_PERM_ENUM)160); //A60(0-3) A60(0-3) A61(0-3) A61(0-3) A60(0-3) A60(0-3) A61(0-3) A61(0-3) A60(0-3) A60(0-3) A61(0-3) A61(0-3) A60(0-3) A60(0-3) A61(0-3) A61(0-3)
                        const __m512i lhs_mat_23_60_sp1 = _mm512_shuffle_epi32(lhs_mat_23_60, (_MM_PERM_ENUM)160); //A62(0-3) A62(0-3) A63(0-3) A63(0-3) A62(0-3) A62(0-3) A63(0-3) A63(0-3) A62(0-3) A62(0-3) A63(0-3) A63(0-3) A62(0-3) A62(0-3) A63(0-3) A63(0-3)

                        const __m512i lhs_mat_01_61_sp1 = _mm512_shuffle_epi32(lhs_mat_01_61, (_MM_PERM_ENUM)160); //A60(8-11) A60(8-11) A61(8-11) A61(8-11) A60(8-11) A60(8-11) A61(8-11) A61(8-11) A60(8-11) A60(8-11) A61(8-11) A61(8-11) A60(8-11) A60(8-11) A61(8-11) A61(8-11)
                        const __m512i lhs_mat_23_61_sp1 = _mm512_shuffle_epi32(lhs_mat_23_61, (_MM_PERM_ENUM)160); //A62(8-11) A62(8-11) A63(8-11) A63(8-11) A62(8-11) A62(8-11) A63(8-11) A63(8-11) A62(8-11) A62(8-11) A63(8-11) A63(8-11) A62(8-11) A62(8-11) A63(8-11) A63(8-11)

                        const __m512i lhs_mat_01_70_sp1 = _mm512_shuffle_epi32(lhs_mat_01_70, (_MM_PERM_ENUM)160); //A70(0-3) A70(0-3) A71(0-3) A71(0-3) A70(0-3) A70(0-3) A71(0-3) A71(0-3) A70(0-3) A70(0-3) A71(0-3) A71(0-3) A70(0-3) A70(0-3) A71(0-3) A71(0-3)
                        const __m512i lhs_mat_23_70_sp1 = _mm512_shuffle_epi32(lhs_mat_23_70, (_MM_PERM_ENUM)160); //A72(0-3) A72(0-3) A73(0-3) A73(0-3) A72(0-3) A72(0-3) A73(0-3) A73(0-3) A72(0-3) A72(0-3) A73(0-3) A73(0-3) A72(0-3) A72(0-3) A73(0-3) A73(0-3)

                        const __m512i lhs_mat_01_71_sp1 = _mm512_shuffle_epi32(lhs_mat_01_71, (_MM_PERM_ENUM)160); //A70(8-11) A70(8-11) A71(8-11) A71(8-11) A70(8-11) A70(8-11) A71(8-11) A71(8-11) A70(8-11) A70(8-11) A71(8-11) A71(8-11) A70(8-11) A70(8-11) A71(8-11) A71(8-11)
                        const __m512i lhs_mat_23_71_sp1 = _mm512_shuffle_epi32(lhs_mat_23_71, (_MM_PERM_ENUM)160); //A72(8-11) A72(8-11) A73(8-11) A73(8-11) A72(8-11) A72(8-11) A73(8-11) A73(8-11) A72(8-11) A72(8-11) A73(8-11) A73(8-11) A72(8-11) A72(8-11) A73(8-11) A73(8-11)

                        const __m512i lhs_mat_01_00_sp2 = _mm512_shuffle_epi32(lhs_mat_01_00, (_MM_PERM_ENUM)245); //A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7) A00(4-7) A01(4-7) A01(4-7)
                        const __m512i lhs_mat_23_00_sp2 = _mm512_shuffle_epi32(lhs_mat_23_00, (_MM_PERM_ENUM)245); //A02(4-7) A02(4-7) A03(4-7) A03(4-7) A02(4-7) A02(4-7) A03(4-7) A03(4-7) A02(4-7) A02(4-7) A03(4-7) A03(4-7) A02(4-7) A02(4-7) A03(4-7) A03(4-7)

                        const __m512i lhs_mat_01_01_sp2 = _mm512_shuffle_epi32(lhs_mat_01_01, (_MM_PERM_ENUM)245); //A00(12-15) A00(12-15) A01(12-15) A01(12-15) A00(12-15) A00(12-15) A01(12-15) A01(12-15) A00(12-15) A00(12-15) A01(12-15) A01(12-15) A00(12-15) A00(12-15) A01(12-15) A01(12-15)
                        const __m512i lhs_mat_23_01_sp2 = _mm512_shuffle_epi32(lhs_mat_23_01, (_MM_PERM_ENUM)245); //A02(12-15) A02(12-15) A03(12-15) A03(12-15) A02(12-15) A02(12-15) A03(12-15) A03(12-15) A02(12-15) A02(12-15) A03(12-15) A03(12-15) A02(12-15) A02(12-15) A03(12-15) A03(12-15)

                        const __m512i lhs_mat_01_10_sp2 = _mm512_shuffle_epi32(lhs_mat_01_10, (_MM_PERM_ENUM)245); //A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7) A10(4-7) A11(4-7) A11(4-7)
                        const __m512i lhs_mat_23_10_sp2 = _mm512_shuffle_epi32(lhs_mat_23_10, (_MM_PERM_ENUM)245); //A12(4-7) A12(4-7) A13(4-7) A13(4-7) A12(4-7) A12(4-7) A13(4-7) A13(4-7) A12(4-7) A12(4-7) A13(4-7) A13(4-7) A12(4-7) A12(4-7) A13(4-7) A13(4-7)

                        const __m512i lhs_mat_01_11_sp2 = _mm512_shuffle_epi32(lhs_mat_01_11, (_MM_PERM_ENUM)245); //A10(12-15) A10(12-15) A11(12-15) A11(12-15) A10(12-15) A10(12-15) A11(12-15) A11(12-15) A10(12-15) A10(12-15) A11(12-15) A11(12-15) A10(12-15) A10(12-15) A11(12-15) A11(12-15)
                        const __m512i lhs_mat_23_11_sp2 = _mm512_shuffle_epi32(lhs_mat_23_11, (_MM_PERM_ENUM)245); //A12(12-15) A12(12-15) A13(12-15) A13(12-15) A12(12-15) A12(12-15) A13(12-15) A13(12-15) A12(12-15) A12(12-15) A13(12-15) A13(12-15) A12(12-15) A12(12-15) A13(12-15) A13(12-15)

                        const __m512i lhs_mat_01_20_sp2 = _mm512_shuffle_epi32(lhs_mat_01_20, (_MM_PERM_ENUM)245); //A20(4-7) A20(4-7) A21(4-7) A21(4-7) A20(4-7) A20(4-7) A21(4-7) A21(4-7) A20(4-7) A20(4-7) A21(4-7) A21(4-7) A20(4-7) A20(4-7) A21(4-7) A21(4-7)
                        const __m512i lhs_mat_23_20_sp2 = _mm512_shuffle_epi32(lhs_mat_23_20, (_MM_PERM_ENUM)245); //A22(4-7) A22(4-7) A23(4-7) A23(4-7) A22(4-7) A22(4-7) A23(4-7) A23(4-7) A22(4-7) A22(4-7) A23(4-7) A23(4-7) A22(4-7) A22(4-7) A23(4-7) A23(4-7)

                        const __m512i lhs_mat_01_21_sp2 = _mm512_shuffle_epi32(lhs_mat_01_21, (_MM_PERM_ENUM)245); //A20(12-15) A20(12-15) A21(12-15) A21(12-15) A20(12-15) A20(12-15) A21(12-15) A21(12-15) A20(12-15) A20(12-15) A21(12-15) A21(12-15) A20(12-15) A20(12-15) A21(12-15) A21(12-15)
                        const __m512i lhs_mat_23_21_sp2 = _mm512_shuffle_epi32(lhs_mat_23_21, (_MM_PERM_ENUM)245); //A22(12-15) A22(12-15) A23(12-15) A23(12-15) A22(12-15) A22(12-15) A23(12-15) A23(12-15) A22(12-15) A22(12-15) A23(12-15) A23(12-15) A22(12-15) A22(12-15) A23(12-15) A23(12-15)

                        const __m512i lhs_mat_01_30_sp2 = _mm512_shuffle_epi32(lhs_mat_01_30, (_MM_PERM_ENUM)245); //A30(4-7) A30(4-7) A31(4-7) A31(4-7) A30(4-7) A30(4-7) A31(4-7) A31(4-7) A30(4-7) A30(4-7) A31(4-7) A31(4-7) A30(4-7) A30(4-7) A31(4-7) A31(4-7)
                        const __m512i lhs_mat_23_30_sp2 = _mm512_shuffle_epi32(lhs_mat_23_30, (_MM_PERM_ENUM)245); //A32(4-7) A32(4-7) A33(4-7) A33(4-7) A32(4-7) A32(4-7) A33(4-7) A33(4-7) A32(4-7) A32(4-7) A33(4-7) A33(4-7) A32(4-7) A32(4-7) A33(4-7) A33(4-7)

                        const __m512i lhs_mat_01_31_sp2 = _mm512_shuffle_epi32(lhs_mat_01_31, (_MM_PERM_ENUM)245); //A30(12-15) A30(12-15) A31(12-15) A31(12-15) A30(12-15) A30(12-15) A31(12-15) A31(12-15) A30(12-15) A30(12-15) A31(12-15) A31(12-15) A30(12-15) A30(12-15) A31(12-15) A31(12-15)
                        const __m512i lhs_mat_23_31_sp2 = _mm512_shuffle_epi32(lhs_mat_23_31, (_MM_PERM_ENUM)245); //A32(12-15) A32(12-15) A33(12-15) A33(12-15) A32(12-15) A32(12-15) A33(12-15) A33(12-15) A32(12-15) A32(12-15) A33(12-15) A33(12-15) A32(12-15) A32(12-15) A33(12-15) A33(12-15)

                        const __m512i lhs_mat_01_40_sp2 = _mm512_shuffle_epi32(lhs_mat_01_40, (_MM_PERM_ENUM)245); //A40(4-7) A40(4-7) A41(4-7) A41(4-7) A40(4-7) A40(4-7) A41(4-7) A41(4-7) A40(4-7) A40(4-7) A41(4-7) A41(4-7) A40(4-7) A40(4-7) A41(4-7) A41(4-7)
                        const __m512i lhs_mat_23_40_sp2 = _mm512_shuffle_epi32(lhs_mat_23_40, (_MM_PERM_ENUM)245); //A42(4-7) A42(4-7) A43(4-7) A43(4-7) A42(4-7) A42(4-7) A43(4-7) A43(4-7) A42(4-7) A42(4-7) A43(4-7) A43(4-7) A42(4-7) A42(4-7) A43(4-7) A43(4-7)

                        const __m512i lhs_mat_01_41_sp2 = _mm512_shuffle_epi32(lhs_mat_01_41, (_MM_PERM_ENUM)245); //A40(12-15) A40(12-15) A41(12-15) A41(12-15) A40(12-15) A40(12-15) A41(12-15) A41(12-15) A40(12-15) A40(12-15) A41(12-15) A41(12-15) A40(12-15) A40(12-15) A41(12-15) A41(12-15)
                        const __m512i lhs_mat_23_41_sp2 = _mm512_shuffle_epi32(lhs_mat_23_41, (_MM_PERM_ENUM)245); //A42(12-15) A42(12-15) A43(12-15) A43(12-15) A42(12-15) A42(12-15) A43(12-15) A43(12-15) A42(12-15) A42(12-15) A43(12-15) A43(12-15) A42(12-15) A42(12-15) A43(12-15) A43(12-15)

                        const __m512i lhs_mat_01_50_sp2 = _mm512_shuffle_epi32(lhs_mat_01_50, (_MM_PERM_ENUM)245); //A50(4-7) A50(4-7) A51(4-7) A51(4-7) A50(4-7) A50(4-7) A51(4-7) A51(4-7) A50(4-7) A50(4-7) A51(4-7) A51(4-7) A50(4-7) A50(4-7) A51(4-7) A51(4-7)
                        const __m512i lhs_mat_23_50_sp2 = _mm512_shuffle_epi32(lhs_mat_23_50, (_MM_PERM_ENUM)245); //A52(4-7) A52(4-7) A53(4-7) A53(4-7) A52(4-7) A52(4-7) A53(4-7) A53(4-7) A52(4-7) A52(4-7) A53(4-7) A53(4-7) A52(4-7) A52(4-7) A53(4-7) A53(4-7)

                        const __m512i lhs_mat_01_51_sp2 = _mm512_shuffle_epi32(lhs_mat_01_51, (_MM_PERM_ENUM)245); //A50(12-15) A50(12-15) A51(12-15) A51(12-15) A50(12-15) A50(12-15) A51(12-15) A51(12-15) A50(12-15) A50(12-15) A51(12-15) A51(12-15) A50(12-15) A50(12-15) A51(12-15) A51(12-15)
                        const __m512i lhs_mat_23_51_sp2 = _mm512_shuffle_epi32(lhs_mat_23_51, (_MM_PERM_ENUM)245); //A52(12-15) A52(12-15) A53(12-15) A53(12-15) A52(12-15) A52(12-15) A53(12-15) A53(12-15) A52(12-15) A52(12-15) A53(12-15) A53(12-15) A52(12-15) A52(12-15) A53(12-15) A53(12-15)

                        const __m512i lhs_mat_01_60_sp2 = _mm512_shuffle_epi32(lhs_mat_01_60, (_MM_PERM_ENUM)245); //A60(4-7) A60(4-7) A61(4-7) A61(4-7) A60(4-7) A60(4-7) A61(4-7) A61(4-7) A60(4-7) A60(4-7) A61(4-7) A61(4-7) A60(4-7) A60(4-7) A61(4-7) A61(4-7)
                        const __m512i lhs_mat_23_60_sp2 = _mm512_shuffle_epi32(lhs_mat_23_60, (_MM_PERM_ENUM)245); //A62(4-7) A62(4-7) A63(4-7) A63(4-7) A62(4-7) A62(4-7) A63(4-7) A63(4-7) A62(4-7) A62(4-7) A63(4-7) A63(4-7) A62(4-7) A62(4-7) A63(4-7) A63(4-7)

                        const __m512i lhs_mat_01_61_sp2 = _mm512_shuffle_epi32(lhs_mat_01_61, (_MM_PERM_ENUM)245); //A60(12-15) A60(12-15) A61(12-15) A61(12-15) A60(12-15) A60(12-15) A61(12-15) A61(12-15) A60(12-15) A60(12-15) A61(12-15) A61(12-15) A60(12-15) A60(12-15) A61(12-15) A61(12-15)
                        const __m512i lhs_mat_23_61_sp2 = _mm512_shuffle_epi32(lhs_mat_23_61, (_MM_PERM_ENUM)245); //A62(12-15) A62(12-15) A63(12-15) A63(12-15) A62(12-15) A62(12-15) A63(12-15) A63(12-15) A62(12-15) A62(12-15) A63(12-15) A63(12-15) A62(12-15) A62(12-15) A63(12-15) A63(12-15)

                        const __m512i lhs_mat_01_70_sp2 = _mm512_shuffle_epi32(lhs_mat_01_70, (_MM_PERM_ENUM)245); //A70(4-7) A70(4-7) A71(4-7) A71(4-7) A70(4-7) A70(4-7) A71(4-7) A71(4-7) A70(4-7) A70(4-7) A71(4-7) A71(4-7) A70(4-7) A70(4-7) A71(4-7) A71(4-7)
                        const __m512i lhs_mat_23_70_sp2 = _mm512_shuffle_epi32(lhs_mat_23_70, (_MM_PERM_ENUM)245); //A72(4-7) A72(4-7) A73(4-7) A73(4-7) A72(4-7) A72(4-7) A73(4-7) A73(4-7) A72(4-7) A72(4-7) A73(4-7) A73(4-7) A72(4-7) A72(4-7) A73(4-7) A73(4-7)

                        const __m512i lhs_mat_01_71_sp2 = _mm512_shuffle_epi32(lhs_mat_01_71, (_MM_PERM_ENUM)245); //A70(12-15) A70(12-15) A71(12-15) A71(12-15) A70(12-15) A70(12-15) A71(12-15) A71(12-15) A70(12-15) A70(12-15) A71(12-15) A71(12-15) A70(12-15) A70(12-15) A71(12-15) A71(12-15)
                        const __m512i lhs_mat_23_71_sp2 = _mm512_shuffle_epi32(lhs_mat_23_71, (_MM_PERM_ENUM)245); //A72(12-15) A72(12-15) A73(12-15) A73(12-15) A72(12-15) A72(12-15) A73(12-15) A73(12-15) A72(12-15) A72(12-15) A73(12-15) A73(12-15) A72(12-15) A72(12-15) A73(12-15) A73(12-15)

                        // The values arranged in shuffle patterns are operated with dot product operation within 32 bit lane i.e corresponding bytes and multiplied and added into 32 bit integers within 32 bit lane
                        __m512i iacc_mat_00_0_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_00_sp1, lhs_mat_01_00_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_01_sp1, lhs_mat_01_01_sp1));
                        __m512i iacc_mat_01_0_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_00_sp1, lhs_mat_01_00_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_01_sp1, lhs_mat_01_01_sp1));

                        __m512i iacc_mat_10_0_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_00_sp1, lhs_mat_23_00_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_01_sp1, lhs_mat_23_01_sp1));
                        __m512i iacc_mat_11_0_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_00_sp1, lhs_mat_23_00_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_01_sp1, lhs_mat_23_01_sp1));

                        __m512i iacc_mat_00_1_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_10_sp1, lhs_mat_01_10_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_11_sp1, lhs_mat_01_11_sp1));
                        __m512i iacc_mat_01_1_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_10_sp1, lhs_mat_01_10_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_11_sp1, lhs_mat_01_11_sp1));

                        __m512i iacc_mat_10_1_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_10_sp1, lhs_mat_23_10_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_11_sp1, lhs_mat_23_11_sp1));
                        __m512i iacc_mat_11_1_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_10_sp1, lhs_mat_23_10_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_11_sp1, lhs_mat_23_11_sp1));

                        __m512i iacc_mat_00_2_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_20_sp1, lhs_mat_01_20_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_21_sp1, lhs_mat_01_21_sp1));
                        __m512i iacc_mat_01_2_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_20_sp1, lhs_mat_01_20_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_21_sp1, lhs_mat_01_21_sp1));

                        __m512i iacc_mat_10_2_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_20_sp1, lhs_mat_23_20_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_21_sp1, lhs_mat_23_21_sp1));
                        __m512i iacc_mat_11_2_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_20_sp1, lhs_mat_23_20_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_21_sp1, lhs_mat_23_21_sp1));

                        __m512i iacc_mat_00_3_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_30_sp1, lhs_mat_01_30_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_31_sp1, lhs_mat_01_31_sp1));
                        __m512i iacc_mat_01_3_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_30_sp1, lhs_mat_01_30_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_31_sp1, lhs_mat_01_31_sp1));

                        __m512i iacc_mat_10_3_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_30_sp1, lhs_mat_23_30_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_31_sp1, lhs_mat_23_31_sp1));
                        __m512i iacc_mat_11_3_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_30_sp1, lhs_mat_23_30_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_31_sp1, lhs_mat_23_31_sp1));

                        __m512i iacc_mat_00_4_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_40_sp1, lhs_mat_01_40_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_41_sp1, lhs_mat_01_41_sp1));
                        __m512i iacc_mat_01_4_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_40_sp1, lhs_mat_01_40_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_41_sp1, lhs_mat_01_41_sp1));

                        __m512i iacc_mat_10_4_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_40_sp1, lhs_mat_23_40_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_41_sp1, lhs_mat_23_41_sp1));
                        __m512i iacc_mat_11_4_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_40_sp1, lhs_mat_23_40_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_41_sp1, lhs_mat_23_41_sp1));

                        __m512i iacc_mat_00_5_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_50_sp1, lhs_mat_01_50_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_51_sp1, lhs_mat_01_51_sp1));
                        __m512i iacc_mat_01_5_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_50_sp1, lhs_mat_01_50_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_51_sp1, lhs_mat_01_51_sp1));

                        __m512i iacc_mat_10_5_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_50_sp1, lhs_mat_23_50_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_51_sp1, lhs_mat_23_51_sp1));
                        __m512i iacc_mat_11_5_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_50_sp1, lhs_mat_23_50_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_51_sp1, lhs_mat_23_51_sp1));

                        __m512i iacc_mat_00_6_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_60_sp1, lhs_mat_01_60_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_61_sp1, lhs_mat_01_61_sp1));
                        __m512i iacc_mat_01_6_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_60_sp1, lhs_mat_01_60_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_61_sp1, lhs_mat_01_61_sp1));

                        __m512i iacc_mat_10_6_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_60_sp1, lhs_mat_23_60_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_61_sp1, lhs_mat_23_61_sp1));
                        __m512i iacc_mat_11_6_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_60_sp1, lhs_mat_23_60_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_61_sp1, lhs_mat_23_61_sp1));

                        __m512i iacc_mat_00_7_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_70_sp1, lhs_mat_01_70_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_71_sp1, lhs_mat_01_71_sp1));
                        __m512i iacc_mat_01_7_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_70_sp1, lhs_mat_01_70_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_71_sp1, lhs_mat_01_71_sp1));

                        __m512i iacc_mat_10_7_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_70_sp1, lhs_mat_23_70_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_71_sp1, lhs_mat_23_71_sp1));
                        __m512i iacc_mat_11_7_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_70_sp1, lhs_mat_23_70_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_71_sp1, lhs_mat_23_71_sp1));


                        __m512i iacc_mat_00_0_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_00_sp2, lhs_mat_01_00_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_01_sp2, lhs_mat_01_01_sp2));
                        __m512i iacc_mat_01_0_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_00_sp2, lhs_mat_01_00_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_01_sp2, lhs_mat_01_01_sp2));

                        __m512i iacc_mat_10_0_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_00_sp2, lhs_mat_23_00_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_01_sp2, lhs_mat_23_01_sp2));
                        __m512i iacc_mat_11_0_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_00_sp2, lhs_mat_23_00_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_01_sp2, lhs_mat_23_01_sp2));

                        __m512i iacc_mat_00_1_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_10_sp2, lhs_mat_01_10_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_11_sp2, lhs_mat_01_11_sp2));
                        __m512i iacc_mat_01_1_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_10_sp2, lhs_mat_01_10_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_11_sp2, lhs_mat_01_11_sp2));

                        __m512i iacc_mat_10_1_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_10_sp2, lhs_mat_23_10_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_11_sp2, lhs_mat_23_11_sp2));
                        __m512i iacc_mat_11_1_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_10_sp2, lhs_mat_23_10_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_11_sp2, lhs_mat_23_11_sp2));

                        __m512i iacc_mat_00_2_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_20_sp2, lhs_mat_01_20_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_21_sp2, lhs_mat_01_21_sp2));
                        __m512i iacc_mat_01_2_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_20_sp2, lhs_mat_01_20_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_21_sp2, lhs_mat_01_21_sp2));

                        __m512i iacc_mat_10_2_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_20_sp2, lhs_mat_23_20_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_21_sp2, lhs_mat_23_21_sp2));
                        __m512i iacc_mat_11_2_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_20_sp2, lhs_mat_23_20_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_21_sp2, lhs_mat_23_21_sp2));

                        __m512i iacc_mat_00_3_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_30_sp2, lhs_mat_01_30_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_31_sp2, lhs_mat_01_31_sp2));
                        __m512i iacc_mat_01_3_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_30_sp2, lhs_mat_01_30_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_31_sp2, lhs_mat_01_31_sp2));

                        __m512i iacc_mat_10_3_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_30_sp2, lhs_mat_23_30_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_31_sp2, lhs_mat_23_31_sp2));
                        __m512i iacc_mat_11_3_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_30_sp2, lhs_mat_23_30_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_31_sp2, lhs_mat_23_31_sp2));

                        __m512i iacc_mat_00_4_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_40_sp2, lhs_mat_01_40_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_41_sp2, lhs_mat_01_41_sp2));
                        __m512i iacc_mat_01_4_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_40_sp2, lhs_mat_01_40_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_41_sp2, lhs_mat_01_41_sp2));

                        __m512i iacc_mat_10_4_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_40_sp2, lhs_mat_23_40_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_41_sp2, lhs_mat_23_41_sp2));
                        __m512i iacc_mat_11_4_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_40_sp2, lhs_mat_23_40_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_41_sp2, lhs_mat_23_41_sp2));

                        __m512i iacc_mat_00_5_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_50_sp2, lhs_mat_01_50_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_51_sp2, lhs_mat_01_51_sp2));
                        __m512i iacc_mat_01_5_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_50_sp2, lhs_mat_01_50_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_51_sp2, lhs_mat_01_51_sp2));

                        __m512i iacc_mat_10_5_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_50_sp2, lhs_mat_23_50_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_51_sp2, lhs_mat_23_51_sp2));
                        __m512i iacc_mat_11_5_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_50_sp2, lhs_mat_23_50_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_51_sp2, lhs_mat_23_51_sp2));

                        __m512i iacc_mat_00_6_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_60_sp2, lhs_mat_01_60_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_61_sp2, lhs_mat_01_61_sp2));
                        __m512i iacc_mat_01_6_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_60_sp2, lhs_mat_01_60_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_61_sp2, lhs_mat_01_61_sp2));

                        __m512i iacc_mat_10_6_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_60_sp2, lhs_mat_23_60_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_61_sp2, lhs_mat_23_61_sp2));
                        __m512i iacc_mat_11_6_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_60_sp2, lhs_mat_23_60_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_61_sp2, lhs_mat_23_61_sp2));

                        __m512i iacc_mat_00_7_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_70_sp2, lhs_mat_01_70_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_71_sp2, lhs_mat_01_71_sp2));
                        __m512i iacc_mat_01_7_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_70_sp2, lhs_mat_01_70_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_71_sp2, lhs_mat_01_71_sp2));

                        __m512i iacc_mat_10_7_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_70_sp2, lhs_mat_23_70_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_71_sp2, lhs_mat_23_71_sp2));
                        __m512i iacc_mat_11_7_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_70_sp2, lhs_mat_23_70_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_71_sp2, lhs_mat_23_71_sp2));

                        // Combine results from both shuffle patterns for each output block
                        __m512i iacc_mat_00_0 = _mm512_add_epi16(iacc_mat_00_0_sp1, iacc_mat_00_0_sp2);
                        __m512i iacc_mat_01_0 = _mm512_add_epi16(iacc_mat_01_0_sp1, iacc_mat_01_0_sp2);
                        __m512i iacc_mat_10_0 = _mm512_add_epi16(iacc_mat_10_0_sp1, iacc_mat_10_0_sp2);
                        __m512i iacc_mat_11_0 = _mm512_add_epi16(iacc_mat_11_0_sp1, iacc_mat_11_0_sp2);

                        __m512i iacc_mat_00_1 = _mm512_add_epi16(iacc_mat_00_1_sp1, iacc_mat_00_1_sp2);
                        __m512i iacc_mat_01_1 = _mm512_add_epi16(iacc_mat_01_1_sp1, iacc_mat_01_1_sp2);
                        __m512i iacc_mat_10_1 = _mm512_add_epi16(iacc_mat_10_1_sp1, iacc_mat_10_1_sp2);
                        __m512i iacc_mat_11_1 = _mm512_add_epi16(iacc_mat_11_1_sp1, iacc_mat_11_1_sp2);

                        __m512i iacc_mat_00_2 = _mm512_add_epi16(iacc_mat_00_2_sp1, iacc_mat_00_2_sp2);
                        __m512i iacc_mat_01_2 = _mm512_add_epi16(iacc_mat_01_2_sp1, iacc_mat_01_2_sp2);
                        __m512i iacc_mat_10_2 = _mm512_add_epi16(iacc_mat_10_2_sp1, iacc_mat_10_2_sp2);
                        __m512i iacc_mat_11_2 = _mm512_add_epi16(iacc_mat_11_2_sp1, iacc_mat_11_2_sp2);

                        __m512i iacc_mat_00_3 = _mm512_add_epi16(iacc_mat_00_3_sp1, iacc_mat_00_3_sp2);
                        __m512i iacc_mat_01_3 = _mm512_add_epi16(iacc_mat_01_3_sp1, iacc_mat_01_3_sp2);
                        __m512i iacc_mat_10_3 = _mm512_add_epi16(iacc_mat_10_3_sp1, iacc_mat_10_3_sp2);
                        __m512i iacc_mat_11_3 = _mm512_add_epi16(iacc_mat_11_3_sp1, iacc_mat_11_3_sp2);

                        __m512i iacc_mat_00_4 = _mm512_add_epi16(iacc_mat_00_4_sp1, iacc_mat_00_4_sp2);
                        __m512i iacc_mat_01_4 = _mm512_add_epi16(iacc_mat_01_4_sp1, iacc_mat_01_4_sp2);
                        __m512i iacc_mat_10_4 = _mm512_add_epi16(iacc_mat_10_4_sp1, iacc_mat_10_4_sp2);
                        __m512i iacc_mat_11_4 = _mm512_add_epi16(iacc_mat_11_4_sp1, iacc_mat_11_4_sp2);

                        __m512i iacc_mat_00_5 = _mm512_add_epi16(iacc_mat_00_5_sp1, iacc_mat_00_5_sp2);
                        __m512i iacc_mat_01_5 = _mm512_add_epi16(iacc_mat_01_5_sp1, iacc_mat_01_5_sp2);
                        __m512i iacc_mat_10_5 = _mm512_add_epi16(iacc_mat_10_5_sp1, iacc_mat_10_5_sp2);
                        __m512i iacc_mat_11_5 = _mm512_add_epi16(iacc_mat_11_5_sp1, iacc_mat_11_5_sp2);

                        __m512i iacc_mat_00_6 = _mm512_add_epi16(iacc_mat_00_6_sp1, iacc_mat_00_6_sp2);
                        __m512i iacc_mat_01_6 = _mm512_add_epi16(iacc_mat_01_6_sp1, iacc_mat_01_6_sp2);
                        __m512i iacc_mat_10_6 = _mm512_add_epi16(iacc_mat_10_6_sp1, iacc_mat_10_6_sp2);
                        __m512i iacc_mat_11_6 = _mm512_add_epi16(iacc_mat_11_6_sp1, iacc_mat_11_6_sp2);

                        __m512i iacc_mat_00_7 = _mm512_add_epi16(iacc_mat_00_7_sp1, iacc_mat_00_7_sp2);
                        __m512i iacc_mat_01_7 = _mm512_add_epi16(iacc_mat_01_7_sp1, iacc_mat_01_7_sp2);
                        __m512i iacc_mat_10_7 = _mm512_add_epi16(iacc_mat_10_7_sp1, iacc_mat_10_7_sp2);
                        __m512i iacc_mat_11_7 = _mm512_add_epi16(iacc_mat_11_7_sp1, iacc_mat_11_7_sp2);

                        // Output of both shuffle patterns are added in order to sum dot product outputs of all 32 values in block
                        iacc_mat_00_0 = _mm512_madd_epi16(iacc_mat_00_0, scale_014589CD_0);
                        iacc_mat_01_0 = _mm512_madd_epi16(iacc_mat_01_0, scale_2367ABEF_0);
                        iacc_mat_10_0 = _mm512_madd_epi16(iacc_mat_10_0, scale_014589CD_0);
                        iacc_mat_11_0 = _mm512_madd_epi16(iacc_mat_11_0, scale_2367ABEF_0);

                        iacc_mat_00_1 = _mm512_madd_epi16(iacc_mat_00_1, scale_014589CD_1);
                        iacc_mat_01_1 = _mm512_madd_epi16(iacc_mat_01_1, scale_2367ABEF_1);
                        iacc_mat_10_1 = _mm512_madd_epi16(iacc_mat_10_1, scale_014589CD_1);
                        iacc_mat_11_1 = _mm512_madd_epi16(iacc_mat_11_1, scale_2367ABEF_1);

                        iacc_mat_00_2 = _mm512_madd_epi16(iacc_mat_00_2, scale_014589CD_2);
                        iacc_mat_01_2 = _mm512_madd_epi16(iacc_mat_01_2, scale_2367ABEF_2);
                        iacc_mat_10_2 = _mm512_madd_epi16(iacc_mat_10_2, scale_014589CD_2);
                        iacc_mat_11_2 = _mm512_madd_epi16(iacc_mat_11_2, scale_2367ABEF_2);

                        iacc_mat_00_3 = _mm512_madd_epi16(iacc_mat_00_3, scale_014589CD_3);
                        iacc_mat_01_3 = _mm512_madd_epi16(iacc_mat_01_3, scale_2367ABEF_3);
                        iacc_mat_10_3 = _mm512_madd_epi16(iacc_mat_10_3, scale_014589CD_3);
                        iacc_mat_11_3 = _mm512_madd_epi16(iacc_mat_11_3, scale_2367ABEF_3);

                        iacc_mat_00_4 = _mm512_madd_epi16(iacc_mat_00_4, scale_014589CD_4);
                        iacc_mat_01_4 = _mm512_madd_epi16(iacc_mat_01_4, scale_2367ABEF_4);
                        iacc_mat_10_4 = _mm512_madd_epi16(iacc_mat_10_4, scale_014589CD_4);
                        iacc_mat_11_4 = _mm512_madd_epi16(iacc_mat_11_4, scale_2367ABEF_4);

                        iacc_mat_00_5 = _mm512_madd_epi16(iacc_mat_00_5, scale_014589CD_5);
                        iacc_mat_01_5 = _mm512_madd_epi16(iacc_mat_01_5, scale_2367ABEF_5);
                        iacc_mat_10_5 = _mm512_madd_epi16(iacc_mat_10_5, scale_014589CD_5);
                        iacc_mat_11_5 = _mm512_madd_epi16(iacc_mat_11_5, scale_2367ABEF_5);

                        iacc_mat_00_6 = _mm512_madd_epi16(iacc_mat_00_6, scale_014589CD_6);
                        iacc_mat_01_6 = _mm512_madd_epi16(iacc_mat_01_6, scale_2367ABEF_6);
                        iacc_mat_10_6 = _mm512_madd_epi16(iacc_mat_10_6, scale_014589CD_6);
                        iacc_mat_11_6 = _mm512_madd_epi16(iacc_mat_11_6, scale_2367ABEF_6);

                        iacc_mat_00_7 = _mm512_madd_epi16(iacc_mat_00_7, scale_014589CD_7);
                        iacc_mat_01_7 = _mm512_madd_epi16(iacc_mat_01_7, scale_2367ABEF_7);
                        iacc_mat_10_7 = _mm512_madd_epi16(iacc_mat_10_7, scale_014589CD_7);
                        iacc_mat_11_7 = _mm512_madd_epi16(iacc_mat_11_7, scale_2367ABEF_7);

                        __m512i iacc_mat_00 = _mm512_add_epi32(_mm512_add_epi32(_mm512_add_epi32(iacc_mat_00_0, iacc_mat_00_1), _mm512_add_epi32(iacc_mat_00_2, iacc_mat_00_3)), _mm512_add_epi32(_mm512_add_epi32(iacc_mat_00_4, iacc_mat_00_5), _mm512_add_epi32(iacc_mat_00_6, iacc_mat_00_7)));
                        __m512i iacc_mat_01 = _mm512_add_epi32(_mm512_add_epi32(_mm512_add_epi32(iacc_mat_01_0, iacc_mat_01_1), _mm512_add_epi32(iacc_mat_01_2, iacc_mat_01_3)), _mm512_add_epi32(_mm512_add_epi32(iacc_mat_01_4, iacc_mat_01_5), _mm512_add_epi32(iacc_mat_01_6, iacc_mat_01_7)));
                        __m512i iacc_mat_10 = _mm512_add_epi32(_mm512_add_epi32(_mm512_add_epi32(iacc_mat_10_0, iacc_mat_10_1), _mm512_add_epi32(iacc_mat_10_2, iacc_mat_10_3)), _mm512_add_epi32(_mm512_add_epi32(iacc_mat_10_4, iacc_mat_10_5), _mm512_add_epi32(iacc_mat_10_6, iacc_mat_10_7)));
                        __m512i iacc_mat_11 = _mm512_add_epi32(_mm512_add_epi32(_mm512_add_epi32(iacc_mat_11_0, iacc_mat_11_1), _mm512_add_epi32(iacc_mat_11_2, iacc_mat_11_3)), _mm512_add_epi32(_mm512_add_epi32(iacc_mat_11_4, iacc_mat_11_5), _mm512_add_epi32(iacc_mat_11_6, iacc_mat_11_7)));

                        // Straighten out to make 4 row vectors
                        __m512i iacc_row_0 = _mm512_mask_blend_epi32(0xCCCC, iacc_mat_00, _mm512_shuffle_epi32(iacc_mat_01, (_MM_PERM_ENUM)78));
                        __m512i iacc_row_1 = _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32(iacc_mat_00, (_MM_PERM_ENUM)78), iacc_mat_01);
                        __m512i iacc_row_2 = _mm512_mask_blend_epi32(0xCCCC, iacc_mat_10, _mm512_shuffle_epi32(iacc_mat_11, (_MM_PERM_ENUM)78));
                        __m512i iacc_row_3 = _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32(iacc_mat_10, (_MM_PERM_ENUM)78), iacc_mat_11);

                        // Load the scale(d) values for all the 4 Q8_k blocks and repeat it across lanes
                        const __m128 row_scale_f32_sse = _mm_load_ps(a_ptrs[rp][b].d);
                        const __m256 row_scale_f32_ymm = _mm256_set_m128(row_scale_f32_sse, row_scale_f32_sse);
                        const __m512 row_scale_f32 = _mm512_insertf32x8(_mm512_castps256_ps512(row_scale_f32_ymm), row_scale_f32_ymm, 1);

                        // Multiply with appropriate scales and accumulate (for both d and dmin) below
                        acc_rows[rp * 4] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_0), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 0)), acc_rows[rp * 4]);
                        acc_rows[rp * 4  + 1] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_1), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 85)), acc_rows[rp * 4 + 1]);
                        acc_rows[rp * 4 + 2] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_2), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_rows[rp * 4 + 2]);
                        acc_rows[rp * 4 + 3] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_3), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_rows[rp * 4 + 3]);

                        // Take two bsums from two Q8_Ks at a time and multiply with corresponding mins values from each Q2_K
                        __m512i iacc_row_min_0_01 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_01_0123, (_MM_PERM_ENUM)0), mins_01);
                        __m512i iacc_row_min_1_01 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_01_0123, (_MM_PERM_ENUM)170), mins_01);
                        __m512i iacc_row_min_2_01 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_23_0123, (_MM_PERM_ENUM)0), mins_01);
                        __m512i iacc_row_min_3_01 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_23_0123, (_MM_PERM_ENUM)170), mins_01);

                        __m512i iacc_row_min_0_23 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_01_0123, (_MM_PERM_ENUM)85), mins_23);
                        __m512i iacc_row_min_1_23 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_01_0123, (_MM_PERM_ENUM)255), mins_23);
                        __m512i iacc_row_min_2_23 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_23_0123, (_MM_PERM_ENUM)85), mins_23);
                        __m512i iacc_row_min_3_23 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_23_0123, (_MM_PERM_ENUM)255), mins_23);

                        __m512i iacc_row_min_0_45 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_01_4567, (_MM_PERM_ENUM)0), mins_45);
                        __m512i iacc_row_min_1_45 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_01_4567, (_MM_PERM_ENUM)170), mins_45);
                        __m512i iacc_row_min_2_45 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_23_4567, (_MM_PERM_ENUM)0), mins_45);
                        __m512i iacc_row_min_3_45 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_23_4567, (_MM_PERM_ENUM)170), mins_45);

                        __m512i iacc_row_min_0_67 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_01_4567, (_MM_PERM_ENUM)85), mins_67);
                        __m512i iacc_row_min_1_67 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_01_4567, (_MM_PERM_ENUM)255), mins_67);
                        __m512i iacc_row_min_2_67 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_23_4567, (_MM_PERM_ENUM)85), mins_67);
                        __m512i iacc_row_min_3_67 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_23_4567, (_MM_PERM_ENUM)255), mins_67);

                        __m512i iacc_row_min_0 = _mm512_add_epi32(_mm512_add_epi32(iacc_row_min_0_01, iacc_row_min_0_23), _mm512_add_epi32(iacc_row_min_0_45,iacc_row_min_0_67));
                        __m512i iacc_row_min_1 = _mm512_add_epi32(_mm512_add_epi32(iacc_row_min_1_01, iacc_row_min_1_23), _mm512_add_epi32(iacc_row_min_1_45,iacc_row_min_1_67));
                        __m512i iacc_row_min_2 = _mm512_add_epi32(_mm512_add_epi32(iacc_row_min_2_01, iacc_row_min_2_23), _mm512_add_epi32(iacc_row_min_2_45,iacc_row_min_2_67));
                        __m512i iacc_row_min_3 = _mm512_add_epi32(_mm512_add_epi32(iacc_row_min_3_01, iacc_row_min_3_23), _mm512_add_epi32(iacc_row_min_3_45,iacc_row_min_3_67));

                        acc_min_rows[rp * 4] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_min_0), _mm512_mul_ps(col_dmin_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 0)), acc_min_rows[rp * 4]);
                        acc_min_rows[rp * 4 + 1] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_min_1), _mm512_mul_ps(col_dmin_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 85)), acc_min_rows[rp * 4 + 1]);
                        acc_min_rows[rp * 4 + 2] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_min_2), _mm512_mul_ps(col_dmin_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_min_rows[rp * 4 + 2]);
                        acc_min_rows[rp * 4 + 3] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_min_3), _mm512_mul_ps(col_dmin_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_min_rows[rp * 4 + 3]);
                    }
                }
            }
            // Store the accumulated values
            for (int i = 0; i < 16; i++) {
                _mm512_storeu_ps((float * )(s + ((y * 4 + i) * bs + x * 8)), _mm512_sub_ps(acc_rows[i], acc_min_rows[i]));
            }
        }
    }

    for (; y < nr / 4; y ++) {

        const block_q8_Kx4 * a_ptr = a_ptr_start + (y * nb);

        // Take group of eight block_q2_kx8 structures at each pass of the loop and perform dot product operation
        for (int64_t x = 0; x < anc / 8; x += 2) {

            const block_q2_Kx8 * b_ptr_0 = b_ptr_start + ((x) * b_nb);
            const block_q2_Kx8 * b_ptr_1 = b_ptr_start + ((x + 1) * b_nb);

            // Master FP accumulators
            __m512 acc_rows[4];
            for (int i = 0; i < 4; i++) {
                acc_rows[i] = _mm512_setzero_ps();
            }

            __m512 acc_min_rows[4];
            for (int i = 0; i < 4; i++) {
                acc_min_rows[i] = _mm512_setzero_ps();
            }
            // For super block
            for (int64_t b = 0; b < nb; b++) {
                // Delta values - Load the sixteen scale values from two block_q2_kx8 structures
                const __m512 col_scale_f32 = GGML_F32Cx8x2_LOAD(b_ptr_0[b].d, b_ptr_1[b].d);

                // dmin values - Load the sixteen dmin values from two block_q2_kx8 structures
                const __m512 col_dmin_f32 = GGML_F32Cx8x2_LOAD(b_ptr_0[b].dmin, b_ptr_1[b].dmin);

                // Loop to iterate over the sixteen sub blocks of a super block - eight sub blocks are processed per iteration
                for (int sb = 0; sb < QK_K / 128; sb++) {

                    // Load the eight block_q2_k for eight sub blocks quantized values interleaved with each other in chunks of eight bytes - B0,B1 ....B6,B7
                    const __m256i rhs_raw_mat_0123_0 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + sb * 256));
                    const __m256i rhs_raw_mat_4567_0 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 32 + sb * 256));
                    const __m256i rhs_raw_mat_0123_1 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 64 + sb * 256));
                    const __m256i rhs_raw_mat_4567_1 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 96 + sb * 256));
                    const __m256i rhs_raw_mat_0123_2 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 128 + sb * 256));
                    const __m256i rhs_raw_mat_4567_2 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 160 + sb * 256));
                    const __m256i rhs_raw_mat_0123_3 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 192 + sb * 256));
                    const __m256i rhs_raw_mat_4567_3 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 224 + sb * 256));

                    const __m256i rhs_raw_mat_89AB_0 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + sb * 256));
                    const __m256i rhs_raw_mat_CDEF_0 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 32 + sb * 256));
                    const __m256i rhs_raw_mat_89AB_1 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 64 + sb * 256));
                    const __m256i rhs_raw_mat_CDEF_1 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 96 + sb * 256));
                    const __m256i rhs_raw_mat_89AB_2 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 128 + sb * 256));
                    const __m256i rhs_raw_mat_CDEF_2 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 160 + sb * 256));
                    const __m256i rhs_raw_mat_89AB_3 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 192 + sb * 256));
                    const __m256i rhs_raw_mat_CDEF_3 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 224 + sb * 256));

                    const __m256i rhs_raw_mat_0145_0 = _mm256_blend_epi32(rhs_raw_mat_0123_0, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_0, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_0, requiredOrder), rhs_raw_mat_4567_0, 240);
                    const __m256i rhs_raw_mat_0145_1 = _mm256_blend_epi32(rhs_raw_mat_0123_1, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_1, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_1, requiredOrder), rhs_raw_mat_4567_1, 240);
                    const __m256i rhs_raw_mat_0145_2 = _mm256_blend_epi32(rhs_raw_mat_0123_2, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_2, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_2 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_2, requiredOrder), rhs_raw_mat_4567_2, 240);
                    const __m256i rhs_raw_mat_0145_3 = _mm256_blend_epi32(rhs_raw_mat_0123_3, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_3, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_3 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_3, requiredOrder), rhs_raw_mat_4567_3, 240);

                    const __m256i rhs_raw_mat_89CD_0 = _mm256_blend_epi32(rhs_raw_mat_89AB_0, _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_0, requiredOrder), 240);
                    const __m256i rhs_raw_mat_ABEF_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_0, requiredOrder), rhs_raw_mat_CDEF_0, 240);
                    const __m256i rhs_raw_mat_89CD_1 = _mm256_blend_epi32(rhs_raw_mat_89AB_1, _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_1, requiredOrder), 240);
                    const __m256i rhs_raw_mat_ABEF_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_1, requiredOrder), rhs_raw_mat_CDEF_1, 240);
                    const __m256i rhs_raw_mat_89CD_2 = _mm256_blend_epi32(rhs_raw_mat_89AB_2, _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_2, requiredOrder), 240);
                    const __m256i rhs_raw_mat_ABEF_2 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_2, requiredOrder), rhs_raw_mat_CDEF_2, 240);
                    const __m256i rhs_raw_mat_89CD_3 = _mm256_blend_epi32(rhs_raw_mat_89AB_3, _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_3, requiredOrder), 240);
                    const __m256i rhs_raw_mat_ABEF_3 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_3, requiredOrder), rhs_raw_mat_CDEF_3, 240);

                    const __m512i rhs_raw_mat_014589CD_0 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_0145_0), rhs_raw_mat_89CD_0, 1);
                    const __m512i rhs_raw_mat_2367ABEF_0 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_2367_0), rhs_raw_mat_ABEF_0, 1);
                    const __m512i rhs_raw_mat_014589CD_1 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_0145_1), rhs_raw_mat_89CD_1, 1);
                    const __m512i rhs_raw_mat_2367ABEF_1 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_2367_1), rhs_raw_mat_ABEF_1, 1);

                    const __m512i rhs_raw_mat_014589CD_2 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_0145_2), rhs_raw_mat_89CD_2, 1);
                    const __m512i rhs_raw_mat_2367ABEF_2 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_2367_2), rhs_raw_mat_ABEF_2, 1);
                    const __m512i rhs_raw_mat_014589CD_3 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_0145_3), rhs_raw_mat_89CD_3, 1);
                    const __m512i rhs_raw_mat_2367ABEF_3 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_2367_3), rhs_raw_mat_ABEF_3, 1);

                    //2-bit -> 8-bit
                    const __m512i rhs_mat_014589CD_00 = _mm512_and_si512(rhs_raw_mat_014589CD_0,m3bexpanded); //B00(0-7) B01(0-7) B04(0-7) B05(0-7) B08(0-7) B09(0-7) B0C(0-7) B0D(0-7)
                    const __m512i rhs_mat_2367ABEF_00 = _mm512_and_si512(rhs_raw_mat_2367ABEF_0,m3bexpanded); //B02(0-7) B03(0-7) B06(0-7) B07(0-7) B0A(0-7) B0B(0-7) B0E(0-7) B0F(0-7)
                    const __m512i rhs_mat_014589CD_01 = _mm512_and_si512(rhs_raw_mat_014589CD_1,m3bexpanded); //B00(8-15) B01(8-15) B04(8-15) B05(8-15) B08(8-15) B09(8-15) B0C(8-15) B0D(8-15)
                    const __m512i rhs_mat_2367ABEF_01 = _mm512_and_si512(rhs_raw_mat_2367ABEF_1,m3bexpanded); //B02(8-15) B03(8-15) B06(8-15) B07(8-15) B0A(8-15) B0B(8-15) B0E(8-15) B0F(8-15)
                    const __m512i rhs_mat_014589CD_10 = _mm512_and_si512(rhs_raw_mat_014589CD_2,m3bexpanded); //B10(0-7) B11(0-7) B14(0-7) B15(0-7) B18(0-7) B19(0-7) B1C(0-7) B1D(0-7)
                    const __m512i rhs_mat_2367ABEF_10 = _mm512_and_si512(rhs_raw_mat_2367ABEF_2,m3bexpanded); //B12(0-7) B13(0-7) B16(0-7) B17(0-7) B1A(0-7) B1B(0-7) B1E(0-7) B1F(0-7)
                    const __m512i rhs_mat_014589CD_11 = _mm512_and_si512(rhs_raw_mat_014589CD_3,m3bexpanded); //B10(8-15) B11(8-15) B14(8-15) B15(8-15) B18(8-15) B19(8-15) B1C(8-15) B1D(8-15)
                    const __m512i rhs_mat_2367ABEF_11 = _mm512_and_si512(rhs_raw_mat_2367ABEF_3,m3bexpanded); //B12(8-15) B13(8-15) B16(8-15) B17(8-15) B1A(8-15) B1B(8-15) B1E(8-15) B1F(8-15)

                    const __m512i rhs_mat_014589CD_20 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_0, 2), m3bexpanded); //B20(0-7) B21(0-7) B24(0-7) B25(0-7) B28(0-7) B29(0-7) B2C(0-7) B2D(0-7)
                    const __m512i rhs_mat_2367ABEF_20 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_0, 2), m3bexpanded); //B22(0-7) B23(0-7) B26(0-7) B27(0-7) B2A(0-7) B2B(0-7) B2E(0-7) B2F(0-7)

                    const __m512i rhs_mat_014589CD_21 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_1, 2), m3bexpanded); //B20(8-15) B21(8-15) B24(8-15) B25(8-15) B28(8-15) B29(8-15) B2C(8-15) B2D(8-15)
                    const __m512i rhs_mat_2367ABEF_21 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_1, 2), m3bexpanded); //B22(8-15) B23(8-15) B26(8-15) B27(8-15) B2A(8-15) B2B(8-15) B2E(8-15) B2F(8-15)

                    const __m512i rhs_mat_014589CD_30 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_2, 2), m3bexpanded); //B30(0-7) B31(0-7) B34(0-7) B35(0-7) B38(0-7) B39(0-7) B3C(0-7) B3D(0-7)
                    const __m512i rhs_mat_2367ABEF_30 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_2, 2), m3bexpanded); //B32(0-7) B33(0-7) B36(0-7) B37(0-7) B3A(0-7) B3B(0-7) B3E(0-7) B3F(0-7)

                    const __m512i rhs_mat_014589CD_31 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_3, 2), m3bexpanded); //B30(8-15) B31(8-15) B34(8-15) B35(8-15) B38(8-15) B39(8-15) B3C(8-15) B3D(8-15)
                    const __m512i rhs_mat_2367ABEF_31 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_3, 2), m3bexpanded); //B32(8-15) B33(8-15) B36(8-15) B37(8-15) B3A(8-15) B3B(8-15) B3E(8-15) B3F(8-15)

                    const __m512i rhs_mat_014589CD_40 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_0, 4), m3bexpanded); //B40(0-7) B41(0-7) B44(0-7) B45(0-7) B48(0-7) B49(0-7) B4C(0-7) B4D(0-7)
                    const __m512i rhs_mat_2367ABEF_40 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_0, 4), m3bexpanded); //B42(0-7) B43(0-7) B46(0-7) B47(0-7) B4A(0-7) B4B(0-7) B4E(0-7) B4F(0-7)

                    const __m512i rhs_mat_014589CD_41 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_1, 4), m3bexpanded); //B40(8-15) B41(8-15) B44(8-15) B45(8-15) B48(8-15) B49(8-15) B4C(8-15) B4D(8-15)
                    const __m512i rhs_mat_2367ABEF_41 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_1, 4), m3bexpanded); //B42(8-15) B43(8-15) B46(8-15) B47(8-15) B4A(8-15) B4B(8-15) B4E(8-15) B4F(8-15)

                    const __m512i rhs_mat_014589CD_50 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_2, 4), m3bexpanded); //B50(0-7) B51(0-7) B54(0-7) B55(0-7) B58(0-7) B59(0-7) B5C(0-7) B5D(0-7)
                    const __m512i rhs_mat_2367ABEF_50 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_2, 4), m3bexpanded); //B52(0-7) B53(0-7) B56(0-7) B57(0-7) B5A(0-7) B5B(0-7) B5E(0-7) B5F(0-7)

                    const __m512i rhs_mat_014589CD_51 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_3, 4), m3bexpanded); //B50(8-15) B51(8-15) B54(8-15) B55(8-15) B58(8-15) B59(8-15) B5C(8-15) B5D(8-15)
                    const __m512i rhs_mat_2367ABEF_51 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_3, 4), m3bexpanded); //B52(8-15) B53(8-15) B56(8-15) B57(8-15) B5A(8-15) B5B(8-15) B5E(8-15) B5F(8-15)

                    const __m512i rhs_mat_014589CD_60 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_0, 6), m3bexpanded); //B60(0-7) B61(0-7) B64(0-7) B65(0-7) B68(0-7) B69(0-7) B6C(0-7) B6D(0-7)
                    const __m512i rhs_mat_2367ABEF_60 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_0, 6), m3bexpanded); //B62(0-7) B63(0-7) B66(0-7) B67(0-7) B6A(0-7) B6B(0-7) B6E(0-7) B6F(0-7)

                    const __m512i rhs_mat_014589CD_61 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_1, 6), m3bexpanded); //B60(8-15) B61(8-15) B64(8-15) B65(8-15) B68(8-15) B69(8-15) B6C(8-15) B6D(8-15)
                    const __m512i rhs_mat_2367ABEF_61 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_1, 6), m3bexpanded); //B62(8-15) B63(8-15) B66(8-15) B67(8-15) B6A(8-15) B6B(8-15) B6E(8-15) B6F(8-15)

                    const __m512i rhs_mat_014589CD_70 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_2, 6), m3bexpanded); //B70(0-7) B71(0-7) B74(0-7) B75(0-7) B78(0-7) B79(0-7) B7C(0-7) B7D(0-7)
                    const __m512i rhs_mat_2367ABEF_70 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_2, 6), m3bexpanded); //B72(0-7) B73(0-7) B76(0-7) B77(0-7) B7A(0-7) B7B(0-7) B7E(0-7) B7F(0-7)

                    const __m512i rhs_mat_014589CD_71 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_3, 6), m3bexpanded); //B70(8-15) B71(8-15) B74(8-15) B75(8-15) B78(8-15) B79(8-15) B7C(8-15) B7D(8-15)
                    const __m512i rhs_mat_2367ABEF_71 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_3, 6), m3bexpanded); //B72(8-15) B73(8-15) B76(8-15) B77(8-15) B7A(8-15) B7B(8-15) B7E(8-15) B7F(8-15)

                    const __m512i rhs_mat_014589CD_00_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_00, (_MM_PERM_ENUM)136); //B00(0-3) B01(0-3) B00(0-3) B01(0-3) B04(0-3) B05(0-3) B04(0-3) B05(0-3) B08(0-3) B09(0-3) B08(0-3) B09(0-3) B0C(0-3) B0D(0-3) B0C(0-3) B0D(0-3)
                    const __m512i rhs_mat_2367ABEF_00_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_00, (_MM_PERM_ENUM)136); //B02(0-3) B03(0-3) B02(0-3) B03(0-3) B06(0-3) B07(0-3) B06(0-3) B07(0-3) B0A(0-3) B0B(0-3) B0A(0-3) B0B(0-3) B0E(0-3) B0F(0-3) B0E(0-3) B0F(0-3)

                    const __m512i rhs_mat_014589CD_01_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_01, (_MM_PERM_ENUM)136); //B00(8-11) B01(8-11) B00(8-11) B01(8-11) B04(8-11) B05(8-11) B04(8-11) B05(8-11) B08(8-11) B09(8-11) B08(8-11) B09(8-11) B0C(8-11) B0D(8-11) B0C(8-11) B0D(8-11)
                    const __m512i rhs_mat_2367ABEF_01_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_01, (_MM_PERM_ENUM)136); //B02(8-11) B03(8-11) B02(8-11) B03(8-11) B06(8-11) B07(8-11) B06(8-11) B07(8-11) B0A(8-11) B0B(8-11) B0A(8-11) B0B(8-11) B0E(8-11) B0F(8-11) B0E(8-11) B0F(8-11)

                    const __m512i rhs_mat_014589CD_10_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_10, (_MM_PERM_ENUM)136); //B10(0-3) B11(0-3) B10(0-3) B11(0-3) B14(0-3) B15(0-3) B14(0-3) B15(0-3) B18(0-3) B19(0-3) B18(0-3) B19(0-3) B1C(0-3) B1D(0-3) B1C(0-3) B1D(0-3)
                    const __m512i rhs_mat_2367ABEF_10_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_10, (_MM_PERM_ENUM)136); //B12(0-3) B13(0-3) B12(0-3) B13(0-3) B16(0-3) B17(0-3) B16(0-3) B17(0-3) B1A(0-3) B1B(0-3) B1A(0-3) B1B(0-3) B1E(0-3) B1F(0-3) B1E(0-3) B1F(0-3)

                    const __m512i rhs_mat_014589CD_11_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_11, (_MM_PERM_ENUM)136); //B10(8-11) B11(8-11) B10(8-11) B11(8-11) B14(8-11) B15(8-11) B14(8-11) B15(8-11) B18(8-11) B19(8-11) B18(8-11) B19(8-11) B1C(8-11) B1D(8-11) B1C(8-11) B1D(8-11)
                    const __m512i rhs_mat_2367ABEF_11_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_11, (_MM_PERM_ENUM)136); //B12(8-11) B13(8-11) B12(8-11) B13(8-11) B16(8-11) B17(8-11) B16(8-11) B17(8-11) B1A(8-11) B1B(8-11) B1A(8-11) B1B(8-11) B1E(8-11) B1F(8-11) B1E(8-11) B1F(8-11)

                    const __m512i rhs_mat_014589CD_20_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_20, (_MM_PERM_ENUM)136); //B20(0-3) B21(0-3) B20(0-3) B21(0-3) B24(0-3) B25(0-3) B24(0-3) B25(0-3) B28(0-3) B29(0-3) B28(0-3) B29(0-3) B2C(0-3) B2D(0-3) B2C(0-3) B2D(0-3)
                    const __m512i rhs_mat_2367ABEF_20_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_20, (_MM_PERM_ENUM)136); //B22(0-3) B23(0-3) B22(0-3) B23(0-3) B26(0-3) B27(0-3) B26(0-3) B27(0-3) B2A(0-3) B2B(0-3) B2A(0-3) B2B(0-3) B2E(0-3) B2F(0-3) B2E(0-3) B2F(0-3)

                    const __m512i rhs_mat_014589CD_21_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_21, (_MM_PERM_ENUM)136); //B20(8-11) B21(8-11) B20(8-11) B21(8-11) B24(8-11) B25(8-11) B24(8-11) B25(8-11) B28(8-11) B29(8-11) B28(8-11) B29(8-11) B2C(8-11) B2D(8-11) B2C(8-11) B2D(8-11)
                    const __m512i rhs_mat_2367ABEF_21_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_21, (_MM_PERM_ENUM)136); //B22(8-11) B23(8-11) B22(8-11) B23(8-11) B26(8-11) B27(8-11) B26(8-11) B27(8-11) B2A(8-11) B2B(8-11) B2A(8-11) B2B(8-11) B2E(8-11) B2F(8-11) B2E(8-11) B2F(8-11)
                    const __m512i rhs_mat_014589CD_30_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_30, (_MM_PERM_ENUM)136); ///B30(0-3) B31(0-3) B30(0-3) B31(0-3) B34(0-3) B35(0-3) B34(0-3) B35(0-3) B38(0-3) B39(0-3) B38(0-3) B39(0-3) B3C(0-3) B3D(0-3) B3C(0-3) B3D(0-3)
                    const __m512i rhs_mat_2367ABEF_30_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_30, (_MM_PERM_ENUM)136); //B32(0-3) B33(0-3) B32(0-3) B33(0-3) B36(0-3) B37(0-3) B36(0-3) B37(0-3) B3A(0-3) B3B(0-3) B3A(0-3) B3B(0-3) B3E(0-3) B3F(0-3) B3E(0-3) B3F(0-3)

                    const __m512i rhs_mat_014589CD_31_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_31, (_MM_PERM_ENUM)136); //B30(8-11) B31(8-11) B30(8-11) B31(8-11) B34(8-11) B35(8-11) B34(8-11) B35(8-11) B38(8-11) B39(8-11) B38(8-11) B39(8-11) B3C(8-11) B3D(8-11) B3C(8-11) B3D(8-11)
                    const __m512i rhs_mat_2367ABEF_31_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_31, (_MM_PERM_ENUM)136); //B32(8-11) B33(8-11) B32(8-11) B33(8-11) B36(8-11) B37(8-11) B36(8-11) B37(8-11) B3A(8-11) B3B(8-11) B3A(8-11) B3B(8-11) B3E(8-11) B3F(8-11) B3E(8-11) B3F(8-11)

                    const __m512i rhs_mat_014589CD_40_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_40, (_MM_PERM_ENUM)136); //B40(0-3) B41(0-3) B40(0-3) B41(0-3) B44(0-3) B45(0-3) B44(0-3) B45(0-3) B48(0-3) B49(0-3) B48(0-3) B49(0-3) B4C(0-3) B4D(0-3) B4C(0-3) B4D(0-3)
                    const __m512i rhs_mat_2367ABEF_40_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_40, (_MM_PERM_ENUM)136); //B42(0-3) B43(0-3) B42(0-3) B43(0-3) B46(0-3) B47(0-3) B46(0-3) B47(0-3) B4A(0-3) B4B(0-3) B4A(0-3) B4B(0-3) B4E(0-3) B4F(0-3) B4E(0-3) B4F(0-3)

                    const __m512i rhs_mat_014589CD_41_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_41, (_MM_PERM_ENUM)136); //B40(8-11) B41(8-11) B40(8-11) B41(8-11) B44(8-11) B45(8-11) B44(8-11) B45(8-11) B48(8-11) B49(8-11) B48(8-11) B49(8-11) B4C(8-11) B4D(8-11) B4C(8-11) B4D(8-11)
                    const __m512i rhs_mat_2367ABEF_41_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_41, (_MM_PERM_ENUM)136); //B42(8-11) B43(8-11) B42(8-11) B43(8-11) B46(8-11) B47(8-11) B46(8-11) B47(8-11) B4A(8-11) B4B(8-11) B4A(8-11) B4B(8-11) B4E(8-11) B4F(8-11) B4E(8-11) B4F(8-11)

                    const __m512i rhs_mat_014589CD_50_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_50, (_MM_PERM_ENUM)136); //B50(0-3) B51(0-3) B50(0-3) B51(0-3) B54(0-3) B55(0-3) B54(0-3) B55(0-3) B58(0-3) B59(0-3) B58(0-3) B59(0-3) B5C(0-3) B5D(0-3) B5C(0-3) B5D(0-3)
                    const __m512i rhs_mat_2367ABEF_50_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_50, (_MM_PERM_ENUM)136); //B52(0-3) B53(0-3) B52(0-3) B53(0-3) B56(0-3) B57(0-3) B56(0-3) B57(0-3) B5A(0-3) B5B(0-3) B5A(0-3) B5B(0-3) B5E(0-3) B5F(0-3) B5E(0-3) B5F(0-3)

                    const __m512i rhs_mat_014589CD_51_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_51, (_MM_PERM_ENUM)136); //B50(8-11) B51(8-11) B50(8-11) B51(8-11) B54(8-11) B55(8-11) B54(8-11) B55(8-11) B58(8-11) B59(8-11) B58(8-11) B59(8-11) B5C(8-11) B5D(8-11) B5C(8-11) B5D(8-11)
                    const __m512i rhs_mat_2367ABEF_51_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_51, (_MM_PERM_ENUM)136); //B52(8-11) B53(8-11) B52(8-11) B53(8-11) B56(8-11) B57(8-11) B56(8-11) B57(8-11) B5A(8-11) B5B(8-11) B5A(8-11) B5B(8-11) B5E(8-11) B5F(8-11) B5E(8-11) B5F(8-11)

                    const __m512i rhs_mat_014589CD_60_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_60, (_MM_PERM_ENUM)136); //B60(0-3) B61(0-3) B60(0-3) B61(0-3) B64(0-3) B65(0-3) B64(0-3) B65(0-3) B68(0-3) B69(0-3) B68(0-3) B69(0-3) B6C(0-3) B6D(0-3) B6C(0-3) B6D(0-3)
                    const __m512i rhs_mat_2367ABEF_60_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_60, (_MM_PERM_ENUM)136); //B62(0-3) B63(0-3) B62(0-3) B63(0-3) B66(0-3) B67(0-3) B66(0-3) B67(0-3) B6A(0-3) B6B(0-3) B6A(0-3) B6B(0-3) B6E(0-3) B6F(0-3) B6E(0-3) B6F(0-3)

                    const __m512i rhs_mat_014589CD_61_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_61, (_MM_PERM_ENUM)136); //B60(8-11) B61(8-11) B60(8-11) B61(8-11) B64(8-11) B65(8-11) B64(8-11) B65(8-11) B68(8-11) B69(8-11) B68(8-11) B69(8-11) B6C(8-11) B6D(8-11) B6C(8-11) B6D(8-11)
                    const __m512i rhs_mat_2367ABEF_61_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_61, (_MM_PERM_ENUM)136); //B62(8-11) B63(8-11) B62(8-11) B63(8-11) B66(8-11) B67(8-11) B66(8-11) B67(8-11) B6A(8-11) B6B(8-11) B6A(8-11) B6B(8-11) B6E(8-11) B6F(8-11) B6E(8-11) B6F(8-11)

                    const __m512i rhs_mat_014589CD_70_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_70, (_MM_PERM_ENUM)136); //B70(0-3) B71(0-3) B70(0-3) B71(0-3) B74(0-3) B75(0-3) B74(0-3) B75(0-3) B78(0-3) B79(0-3) B78(0-3) B79(0-3) B7C(0-3) B7D(0-3) B7C(0-3) B7D(0-3)
                    const __m512i rhs_mat_2367ABEF_70_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_70, (_MM_PERM_ENUM)136); //B72(0-3) B73(0-3) B72(0-3) B73(0-3) B76(0-3) B77(0-3) B76(0-3) B77(0-3) B7A(0-3) B7B(0-3) B7A(0-3) B7B(0-3) B7E(0-3) B7F(0-3) B7E(0-3) B7F(0-3)

                    const __m512i rhs_mat_014589CD_71_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_71, (_MM_PERM_ENUM)136); //B00(8-11) B01(8-11) B00(8-11) B01(8-11) B04(8-11) B05(8-11) B04(8-11) B05(8-11) B08(8-11) B09(8-11) B08(8-11) B09(8-11) B0C(8-11) B0D(8-11) B0C(8-11) B0D(8-11)
                    const __m512i rhs_mat_2367ABEF_71_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_71, (_MM_PERM_ENUM)136); //B72(8-11) B73(8-11) B72(8-11) B73(8-11) B76(8-11) B77(8-11) B76(8-11) B77(8-11) B7A(8-11) B7B(8-11) B7A(8-11) B7B(8-11) B7E(8-11) B7F(8-11) B7E(8-11) B7F(8-11)

                    const __m512i rhs_mat_014589CD_00_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_00, (_MM_PERM_ENUM)221); //B00(4-7) B01(4-7) B00(4-7) B01(4-7) B04(4-7) B05(4-7) B04(4-7) B05(4-7) B08(4-7) B09(4-7) B08(4-7) B09(4-7) B0C(4-7) B0D(4-7) B0C(4-7) B0D(4-7)
                    const __m512i rhs_mat_2367ABEF_00_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_00, (_MM_PERM_ENUM)221); //B02(4-7) B03(4-7) B02(4-7) B03(4-7) B06(4-7) B07(4-7) B06(4-7) B07(4-7) B0A(4-7) B0B(4-7) B0A(4-7) B0B(4-7) B0E(4-7) B0F(4-7) B0E(4-7) B0F(4-7)

                    const __m512i rhs_mat_014589CD_01_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_01, (_MM_PERM_ENUM)221); //B00(12-15) B01(12-15) B00(12-15) B01(12-15) B04(12-15) B05(12-15) B04(12-15) B05(12-15) B08(12-15) B09(12-15) B08(12-15) B09(12-15) B0C(12-15) B0D(12-15) B0C(12-15) B0D(12-15)
                    const __m512i rhs_mat_2367ABEF_01_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_01, (_MM_PERM_ENUM)221); //B02(12-15) B03(12-15) B02(12-15) B03(12-15) B06(12-15) B07(12-15) B06(12-15) B07(12-15) B0A(12-15) B0B(12-15) B0A(12-15) B0B(12-15) B0E(12-15) B0F(12-15) B0E(12-15) B0F(12-15)

                    const __m512i rhs_mat_014589CD_10_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_10, (_MM_PERM_ENUM)221); //B10(4-7) B11(4-7) B10(4-7) B11(4-7) B14(4-7) B15(4-7) B14(4-7) B15(4-7) B18(4-7) B19(4-7) B18(4-7) B19(4-7) B1C(4-7) B1D(4-7) B1C(4-7) B1D(4-7)
                    const __m512i rhs_mat_2367ABEF_10_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_10, (_MM_PERM_ENUM)221); //B12(4-7) B13(4-7) B12(4-7) B13(4-7) B16(4-7) B17(4-7) B16(4-7) B17(4-7) B1A(4-7) B1B(4-7) B1A(4-7) B1B(4-7) B1E(4-7) B1F(4-7) B1E(4-7) B1F(4-7)

                    const __m512i rhs_mat_014589CD_11_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_11, (_MM_PERM_ENUM)221); //B10(12-15) B11(12-15) B10(12-15) B11(12-15) B14(12-15) B15(12-15) B14(12-15) B15(12-15) B18(12-15) B19(12-15) B18(12-15) B19(12-15) B1C(12-15) B1D(12-15) B1C(12-15) B1D(12-15)
                    const __m512i rhs_mat_2367ABEF_11_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_11, (_MM_PERM_ENUM)221); //B12(12-15) B13(12-15) B12(12-15) B13(12-15) B16(12-15) B17(12-15) B16(12-15) B17(12-15) B1A(12-15) B1B(12-15) B1A(12-15) B1B(12-15) B1E(12-15) B1F(12-15) B1E(12-15) B1F(12-15)

                    const __m512i rhs_mat_014589CD_20_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_20, (_MM_PERM_ENUM)221); //B20(4-7) B21(4-7) B20(4-7) B21(4-7) B24(4-7) B25(4-7) B24(4-7) B25(4-7) B28(4-7) B29(4-7) B28(4-7) B29(4-7) B2C(4-7) B2D(4-7) B2C(4-7) B2D(4-7)
                    const __m512i rhs_mat_2367ABEF_20_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_20, (_MM_PERM_ENUM)221); //B22(4-7) B23(4-7) B22(4-7) B23(4-7) B26(4-7) B27(4-7) B26(4-7) B27(4-7) B2A(4-7) B2B(4-7) B2A(4-7) B2B(4-7) B2E(4-7) B2F(4-7) B2E(4-7) B2F(4-7)

                    const __m512i rhs_mat_014589CD_21_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_21, (_MM_PERM_ENUM)221); //B20(12-15) B21(12-15) B20(12-15) B21(12-15) B24(12-15) B25(12-15) B24(12-15) B25(12-15) B28(12-15) B29(12-15) B28(12-15) B29(12-15) B2C(12-15) B2D(12-15) B2C(12-15) B2D(12-15)
                    const __m512i rhs_mat_2367ABEF_21_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_21, (_MM_PERM_ENUM)221); //B22(12-15) B23(12-15) B22(12-15) B23(12-15) B26(12-15) B27(12-15) B26(12-15) B27(12-15) B2A(12-15) B2B(12-15) B2A(12-15) B2B(12-15) B2E(12-15) B2F(12-15) B2E(12-15) B2F(12-15)

                    const __m512i rhs_mat_014589CD_30_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_30, (_MM_PERM_ENUM)221); //B30(4-7) B31(4-7) B30(4-7) B31(4-7) B34(4-7) B35(4-7) B34(4-7) B35(4-7) B38(4-7) B39(4-7) B38(4-7) B39(4-7) B3C(4-7) B3D(4-7) B3C(4-7) B3D(4-7)
                    const __m512i rhs_mat_2367ABEF_30_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_30, (_MM_PERM_ENUM)221); //B32(4-7) B33(4-7) B32(4-7) B33(4-7) B36(4-7) B37(4-7) B36(4-7) B37(4-7) B3A(4-7) B3B(4-7) B3A(4-7) B3B(4-7) B3E(4-7) B3F(4-7) B3E(4-7) B3F(4-7)

                    const __m512i rhs_mat_014589CD_31_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_31, (_MM_PERM_ENUM)221); //B30(12-15) B31(12-15) B30(12-15) B31(12-15) B34(12-15) B35(12-15) B34(12-15) B35(12-15) B38(12-15) B39(12-15) B38(12-15) B39(12-15) B3C(12-15) B3D(12-15) B3C(12-15) B3D(12-15)
                    const __m512i rhs_mat_2367ABEF_31_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_31, (_MM_PERM_ENUM)221); //B32(12-15) B33(12-15) B32(12-15) B33(12-15) B36(12-15) B37(12-15) B36(12-15) B37(12-15) B3A(12-15) B3B(12-15) B3A(12-15) B3B(12-15) B3E(12-15) B3F(12-15) B3E(12-15) B3F(12-15)

                    const __m512i rhs_mat_014589CD_40_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_40, (_MM_PERM_ENUM)221); //B40(4-7) B41(4-7) B40(4-7) B41(4-7) B44(4-7) B45(4-7) B44(4-7) B45(4-7) B48(4-7) B49(4-7) B48(4-7) B49(4-7) B4C(4-7) B4D(4-7) B4C(4-7) B4D(4-7)
                    const __m512i rhs_mat_2367ABEF_40_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_40, (_MM_PERM_ENUM)221); //B42(4-7) B43(4-7) B42(4-7) B43(4-7) B46(4-7) B47(4-7) B46(4-7) B47(4-7) B4A(4-7) B4B(4-7) B4A(4-7) B4B(4-7) B4E(4-7) B4F(4-7) B4E(4-7) B4F(4-7)

                    const __m512i rhs_mat_014589CD_41_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_41, (_MM_PERM_ENUM)221); //B40(12-15) B41(12-15) B40(12-15) B41(12-15) B44(12-15) B45(12-15) B44(12-15) B45(12-15) B48(12-15) B49(12-15) B48(12-15) B49(12-15) B4C(12-15) B4D(12-15) B4C(12-15) B4D(12-15)
                    const __m512i rhs_mat_2367ABEF_41_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_41, (_MM_PERM_ENUM)221); //B42(12-15) B43(12-15) B42(12-15) B43(12-15) B46(12-15) B47(12-15) B46(12-15) B47(12-15) B4A(12-15) B4B(12-15) B4A(12-15) B4B(12-15) B4E(12-15) B4F(12-15) B4E(12-15) B4F(12-15)

                    const __m512i rhs_mat_014589CD_50_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_50, (_MM_PERM_ENUM)221); //B50(4-7) B51(4-7) B50(4-7) B51(4-7) B54(4-7) B55(4-7) B54(4-7) B55(4-7) B58(4-7) B59(4-7) B58(4-7) B59(4-7) B5C(4-7) B5D(4-7) B5C(4-7) B5D(4-7)
                    const __m512i rhs_mat_2367ABEF_50_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_50, (_MM_PERM_ENUM)221); //B52(4-7) B53(4-7) B52(4-7) B53(4-7) B56(4-7) B57(4-7) B56(4-7) B57(4-7) B5A(4-7) B5B(4-7) B5A(4-7) B5B(4-7) B5E(4-7) B5F(4-7) B5E(4-7) B5F(4-7)

                    const __m512i rhs_mat_014589CD_51_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_51, (_MM_PERM_ENUM)221); //B50(12-15) B51(12-15) B50(12-15) B51(12-15) B54(12-15) B55(12-15) B54(12-15) B55(12-15) B58(12-15) B59(12-15) B58(12-15) B59(12-15) B5C(12-15) B5D(12-15) B5C(12-15) B5D(12-15)
                    const __m512i rhs_mat_2367ABEF_51_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_51, (_MM_PERM_ENUM)221); //B52(12-15) B53(12-15) B52(12-15) B53(12-15) B56(12-15) B57(12-15) B56(12-15) B57(12-15) B5A(12-15) B5B(12-15) B5A(12-15) B5B(12-15) B5E(12-15) B5F(12-15) B5E(12-15) B5F(12-15)

                    const __m512i rhs_mat_014589CD_60_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_60, (_MM_PERM_ENUM)221); //B60(4-7) B61(4-7) B60(4-7) B61(4-7) B64(4-7) B65(4-7) B64(4-7) B65(4-7) B68(4-7) B69(4-7) B68(4-7) B69(4-7) B6C(4-7) B6D(4-7) B6C(4-7) B6D(4-7)
                    const __m512i rhs_mat_2367ABEF_60_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_60, (_MM_PERM_ENUM)221); //B62(4-7) B63(4-7) B62(4-7) B63(4-7) B66(4-7) B67(4-7) B66(4-7) B67(4-7) B6A(4-7) B6B(4-7) B6A(4-7) B6B(4-7) B6E(4-7) B6F(4-7) B6E(4-7) B6F(4-7)

                    const __m512i rhs_mat_014589CD_61_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_61, (_MM_PERM_ENUM)221); //B60(12-15) B61(12-15) B60(12-15) B61(12-15) B64(12-15) B65(12-15) B64(12-15) B65(12-15) B68(12-15) B69(12-15) B68(12-15) B69(12-15) B6C(12-15) B6D(12-15) B6C(12-15) B6D(12-15)
                    const __m512i rhs_mat_2367ABEF_61_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_61, (_MM_PERM_ENUM)221); //B62(12-15) B63(12-15) B62(12-15) B63(12-15) B66(12-15) B67(12-15) B66(12-15) B67(12-15) B6A(12-15) B6B(12-15) B6A(12-15) B6B(12-15) B6E(12-15) B6F(12-15) B6E(12-15) B6F(12-15)

                    const __m512i rhs_mat_014589CD_70_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_70, (_MM_PERM_ENUM)221); //B70(4-7) B71(4-7) B70(4-7) B71(4-7) B74(4-7) B75(4-7) B74(4-7) B75(4-7) B78(4-7) B79(4-7) B78(4-7) B79(4-7) B7C(4-7) B7D(4-7) B7C(4-7) B7D(4-7)
                    const __m512i rhs_mat_2367ABEF_70_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_70, (_MM_PERM_ENUM)221); //B72(4-7) B73(4-7) B72(4-7) B73(4-7) B76(4-7) B77(4-7) B76(4-7) B77(4-7) B7A(4-7) B7B(4-7) B7A(4-7) B7B(4-7) B7E(4-7) B7F(4-7) B7E(4-7) B7F(4-7)

                    const __m512i rhs_mat_014589CD_71_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_71, (_MM_PERM_ENUM)221); //B70(12-15) B71(12-15) B70(12-15) B71(12-15) B74(12-15) B75(12-15) B74(12-15) B75(12-15) B78(12-15) B79(12-15) B78(12-15) B79(12-15) B7C(12-15) B7D(12-15) B7C(12-15) B7D(12-15)
                    const __m512i rhs_mat_2367ABEF_71_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_71, (_MM_PERM_ENUM)221); //B72(12-15) B73(12-15) B72(12-15) B73(12-15) B76(12-15) B77(12-15) B76(12-15) B77(12-15) B7A(12-15) B7B(12-15) B7A(12-15) B7B(12-15) B7E(12-15) B7F(12-15) B7E(12-15) B7F(12-15)

                    //notation:superblock subblock
                    //s00 m00  s01 m01   s10 m10  s11 m11  s20 m20  s21 m21   s30 m30  s31 m31  s40 m40  s41 m41   s50 m50  s51 m51  s60 m60  s61 m61   s70 m70  s71 m71

                    const __m128i mins_and_scales_01_0 = _mm_loadu_si128((const __m128i *)(b_ptr_0[b].scales + sb * 64));
                    const __m128i mins_and_scales_23_0 = _mm_loadu_si128((const __m128i *)(b_ptr_0[b].scales + 16 + sb * 64));
                    const __m128i mins_and_scales_45_0 = _mm_loadu_si128((const __m128i *)(b_ptr_0[b].scales + 32 + sb * 64));
                    const __m128i mins_and_scales_67_0 = _mm_loadu_si128((const __m128i *)(b_ptr_0[b].scales + 48 + sb * 64));

                    const __m128i mins_and_scales_01_1 = _mm_loadu_si128((const __m128i *)(b_ptr_1[b].scales + sb * 64));
                    const __m128i mins_and_scales_23_1 = _mm_loadu_si128((const __m128i *)(b_ptr_1[b].scales + 16 + sb * 64));
                    const __m128i mins_and_scales_45_1 = _mm_loadu_si128((const __m128i *)(b_ptr_1[b].scales + 32 + sb * 64));
                    const __m128i mins_and_scales_67_1 = _mm_loadu_si128((const __m128i *)(b_ptr_1[b].scales + 48 + sb * 64));

                    // Combine mins and scales for sub-blocks: 0-1, 2-3, 4-5, 6-7 in the sb loop
                    const __m256i mins_and_scales_01 = _mm256_insertf128_si256(_mm256_castsi128_si256(mins_and_scales_01_0), mins_and_scales_01_1, 1);
                    const __m256i mins_and_scales_23 = _mm256_insertf128_si256(_mm256_castsi128_si256(mins_and_scales_23_0), mins_and_scales_23_1, 1);
                    const __m256i mins_and_scales_45 = _mm256_insertf128_si256(_mm256_castsi128_si256(mins_and_scales_45_0), mins_and_scales_45_1, 1);
                    const __m256i mins_and_scales_67 = _mm256_insertf128_si256(_mm256_castsi128_si256(mins_and_scales_67_0), mins_and_scales_67_1, 1);

                    // Extract scales which is lower half from mins_and_scales
                    const __m256i scales_01 = _mm256_and_si256(mins_and_scales_01, m4b);
                    const __m256i scales_23 = _mm256_and_si256(mins_and_scales_23, m4b);
                    const __m256i scales_45 = _mm256_and_si256(mins_and_scales_45, m4b);
                    const __m256i scales_67 = _mm256_and_si256(mins_and_scales_67, m4b);

                    // Extract mins which is upper half from mins_and_scales
                    const __m512i mins_01 = _mm512_cvtepu8_epi16(_mm256_and_si256(_mm256_srli_epi16(mins_and_scales_01, 4), m4b));
                    const __m512i mins_23 = _mm512_cvtepu8_epi16(_mm256_and_si256(_mm256_srli_epi16(mins_and_scales_23, 4), m4b));
                    const __m512i mins_45 = _mm512_cvtepu8_epi16(_mm256_and_si256(_mm256_srli_epi16(mins_and_scales_45, 4), m4b));
                    const __m512i mins_67 = _mm512_cvtepu8_epi16(_mm256_and_si256(_mm256_srli_epi16(mins_and_scales_67, 4), m4b));

                    const __m512i scales_0 = _mm512_cvtepu8_epi16(_mm256_shuffle_epi8(scales_01, scalesmask1));
                    const __m512i scales_1 = _mm512_cvtepu8_epi16(_mm256_shuffle_epi8(scales_01, scalesmask2));
                    const __m512i scales_2 = _mm512_cvtepu8_epi16(_mm256_shuffle_epi8(scales_23, scalesmask1));
                    const __m512i scales_3 = _mm512_cvtepu8_epi16(_mm256_shuffle_epi8(scales_23, scalesmask2));
                    const __m512i scales_4 = _mm512_cvtepu8_epi16(_mm256_shuffle_epi8(scales_45, scalesmask1));
                    const __m512i scales_5 = _mm512_cvtepu8_epi16(_mm256_shuffle_epi8(scales_45, scalesmask2));
                    const __m512i scales_6 = _mm512_cvtepu8_epi16(_mm256_shuffle_epi8(scales_67, scalesmask1));
                    const __m512i scales_7 = _mm512_cvtepu8_epi16(_mm256_shuffle_epi8(scales_67, scalesmask2));

                    const __m512i scale_014589CD_0 = _mm512_shuffle_epi32(scales_0, (_MM_PERM_ENUM)68);
                    const __m512i scale_2367ABEF_0 = _mm512_shuffle_epi32(scales_0, (_MM_PERM_ENUM)238);

                    const __m512i scale_014589CD_1 = _mm512_shuffle_epi32(scales_1, (_MM_PERM_ENUM)68);
                    const __m512i scale_2367ABEF_1 = _mm512_shuffle_epi32(scales_1, (_MM_PERM_ENUM)238);

                    const __m512i scale_014589CD_2 = _mm512_shuffle_epi32(scales_2, (_MM_PERM_ENUM)68);
                    const __m512i scale_2367ABEF_2 = _mm512_shuffle_epi32(scales_2, (_MM_PERM_ENUM)238);

                    const __m512i scale_014589CD_3 = _mm512_shuffle_epi32(scales_3, (_MM_PERM_ENUM)68);
                    const __m512i scale_2367ABEF_3 = _mm512_shuffle_epi32(scales_3, (_MM_PERM_ENUM)238);

                    const __m512i scale_014589CD_4 = _mm512_shuffle_epi32(scales_4, (_MM_PERM_ENUM)68);
                    const __m512i scale_2367ABEF_4 = _mm512_shuffle_epi32(scales_4, (_MM_PERM_ENUM)238);

                    const __m512i scale_014589CD_5 = _mm512_shuffle_epi32(scales_5, (_MM_PERM_ENUM)68);
                    const __m512i scale_2367ABEF_5 = _mm512_shuffle_epi32(scales_5, (_MM_PERM_ENUM)238);

                    const __m512i scale_014589CD_6 = _mm512_shuffle_epi32(scales_6, (_MM_PERM_ENUM)68);
                    const __m512i scale_2367ABEF_6 = _mm512_shuffle_epi32(scales_6, (_MM_PERM_ENUM)238);

                    const __m512i scale_014589CD_7 = _mm512_shuffle_epi32(scales_7, (_MM_PERM_ENUM)68);
                    const __m512i scale_2367ABEF_7 = _mm512_shuffle_epi32(scales_7, (_MM_PERM_ENUM)238);

                    // Load the four block_q8_k quantized values interleaved with each other in chunks of eight bytes - A0,A1,A2,A3
                    // Loaded as set of 128 bit vectors and repeated into a 256 bit vector
                    __m256i lhs_mat_ymm_0123_00 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 512 * sb)));
                    __m256i lhs_mat_ymm_01_00 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_00, lhs_mat_ymm_0123_00, 0);
                    __m256i lhs_mat_ymm_23_00 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_00, lhs_mat_ymm_0123_00, 17);
                    __m256i lhs_mat_ymm_0123_01 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 32 + 512 * sb)));
                    __m256i lhs_mat_ymm_01_01 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_01, lhs_mat_ymm_0123_01, 0);
                    __m256i lhs_mat_ymm_23_01 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_01, lhs_mat_ymm_0123_01, 17);
                    __m256i lhs_mat_ymm_0123_10 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 64 + 512 * sb)));
                    __m256i lhs_mat_ymm_01_10 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_10, lhs_mat_ymm_0123_10, 0);
                    __m256i lhs_mat_ymm_23_10 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_10, lhs_mat_ymm_0123_10, 17);
                    __m256i lhs_mat_ymm_0123_11 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 96 + 512 * sb)));
                    __m256i lhs_mat_ymm_01_11 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_11, lhs_mat_ymm_0123_11, 0);
                    __m256i lhs_mat_ymm_23_11 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_11, lhs_mat_ymm_0123_11, 17);
                    __m256i lhs_mat_ymm_0123_20 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 128 + 512 * sb)));
                    __m256i lhs_mat_ymm_01_20 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_20, lhs_mat_ymm_0123_20, 0);
                    __m256i lhs_mat_ymm_23_20 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_20, lhs_mat_ymm_0123_20, 17);
                    __m256i lhs_mat_ymm_0123_21 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 160 + 512 * sb)));
                    __m256i lhs_mat_ymm_01_21 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_21, lhs_mat_ymm_0123_21, 0);
                    __m256i lhs_mat_ymm_23_21 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_21, lhs_mat_ymm_0123_21, 17);
                    __m256i lhs_mat_ymm_0123_30 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 192 + 512 * sb)));
                    __m256i lhs_mat_ymm_01_30 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_30, lhs_mat_ymm_0123_30, 0);
                    __m256i lhs_mat_ymm_23_30 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_30, lhs_mat_ymm_0123_30, 17);
                    __m256i lhs_mat_ymm_0123_31 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 224 + 512 * sb)));
                    __m256i lhs_mat_ymm_01_31 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_31, lhs_mat_ymm_0123_31, 0);
                    __m256i lhs_mat_ymm_23_31 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_31, lhs_mat_ymm_0123_31, 17);

                    __m256i lhs_mat_ymm_0123_40 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 256 + 512 * sb)));
                    __m256i lhs_mat_ymm_01_40 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_40, lhs_mat_ymm_0123_40, 0);
                    __m256i lhs_mat_ymm_23_40 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_40, lhs_mat_ymm_0123_40, 17);
                    __m256i lhs_mat_ymm_0123_41 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 288 + 512 * sb)));
                    __m256i lhs_mat_ymm_01_41 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_41, lhs_mat_ymm_0123_41, 0);
                    __m256i lhs_mat_ymm_23_41 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_41, lhs_mat_ymm_0123_41, 17);
                    __m256i lhs_mat_ymm_0123_50 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 320 + 512 * sb)));
                    __m256i lhs_mat_ymm_01_50 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_50, lhs_mat_ymm_0123_50, 0);
                    __m256i lhs_mat_ymm_23_50 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_50, lhs_mat_ymm_0123_50, 17);
                    __m256i lhs_mat_ymm_0123_51 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 352 + 512 * sb)));
                    __m256i lhs_mat_ymm_01_51 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_51, lhs_mat_ymm_0123_51, 0);
                    __m256i lhs_mat_ymm_23_51 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_51, lhs_mat_ymm_0123_51, 17);
                    __m256i lhs_mat_ymm_0123_60 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 384 + 512 * sb)));
                    __m256i lhs_mat_ymm_01_60 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_60, lhs_mat_ymm_0123_60, 0);
                    __m256i lhs_mat_ymm_23_60 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_60, lhs_mat_ymm_0123_60, 17);
                    __m256i lhs_mat_ymm_0123_61 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 416 + 512 * sb)));
                    __m256i lhs_mat_ymm_01_61 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_61, lhs_mat_ymm_0123_61, 0);
                    __m256i lhs_mat_ymm_23_61 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_61, lhs_mat_ymm_0123_61, 17);
                    __m256i lhs_mat_ymm_0123_70 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 448 + 512 * sb)));
                    __m256i lhs_mat_ymm_01_70 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_70, lhs_mat_ymm_0123_70, 0);
                    __m256i lhs_mat_ymm_23_70 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_70, lhs_mat_ymm_0123_70, 17);
                    __m256i lhs_mat_ymm_0123_71 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 480 + 512 * sb)));
                    __m256i lhs_mat_ymm_01_71 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_71, lhs_mat_ymm_0123_71, 0);
                    __m256i lhs_mat_ymm_23_71 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_71, lhs_mat_ymm_0123_71, 17);

                    __m512i lhs_mat_01_00 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_00), lhs_mat_ymm_01_00, 1);
                    __m512i lhs_mat_23_00 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_00), lhs_mat_ymm_23_00, 1);
                    __m512i lhs_mat_01_01 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_01), lhs_mat_ymm_01_01, 1);
                    __m512i lhs_mat_23_01 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_01), lhs_mat_ymm_23_01, 1);

                    __m512i lhs_mat_01_10 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_10), lhs_mat_ymm_01_10, 1);
                    __m512i lhs_mat_23_10 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_10), lhs_mat_ymm_23_10, 1);
                    __m512i lhs_mat_01_11 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_11), lhs_mat_ymm_01_11, 1);
                    __m512i lhs_mat_23_11 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_11), lhs_mat_ymm_23_11, 1);

                    __m512i lhs_mat_01_20 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_20), lhs_mat_ymm_01_20, 1);
                    __m512i lhs_mat_23_20 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_20), lhs_mat_ymm_23_20, 1);
                    __m512i lhs_mat_01_21 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_21), lhs_mat_ymm_01_21, 1);
                    __m512i lhs_mat_23_21 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_21), lhs_mat_ymm_23_21, 1);

                    __m512i lhs_mat_01_30 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_30), lhs_mat_ymm_01_30, 1);
                    __m512i lhs_mat_23_30 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_30), lhs_mat_ymm_23_30, 1);
                    __m512i lhs_mat_01_31 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_31), lhs_mat_ymm_01_31, 1);
                    __m512i lhs_mat_23_31 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_31), lhs_mat_ymm_23_31, 1);

                    __m512i lhs_mat_01_40 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_40), lhs_mat_ymm_01_40, 1);
                    __m512i lhs_mat_23_40 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_40), lhs_mat_ymm_23_40, 1);
                    __m512i lhs_mat_01_41 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_41), lhs_mat_ymm_01_41, 1);
                    __m512i lhs_mat_23_41 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_41), lhs_mat_ymm_23_41, 1);

                    __m512i lhs_mat_01_50 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_50), lhs_mat_ymm_01_50, 1);
                    __m512i lhs_mat_23_50 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_50), lhs_mat_ymm_23_50, 1);
                    __m512i lhs_mat_01_51 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_51), lhs_mat_ymm_01_51, 1);
                    __m512i lhs_mat_23_51 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_51), lhs_mat_ymm_23_51, 1);

                    __m512i lhs_mat_01_60 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_60), lhs_mat_ymm_01_60, 1);
                    __m512i lhs_mat_23_60 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_60), lhs_mat_ymm_23_60, 1);
                    __m512i lhs_mat_01_61 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_61), lhs_mat_ymm_01_61, 1);
                    __m512i lhs_mat_23_61 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_61), lhs_mat_ymm_23_61, 1);

                    __m512i lhs_mat_01_70 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_70), lhs_mat_ymm_01_70, 1);
                    __m512i lhs_mat_23_70 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_70), lhs_mat_ymm_23_70, 1);
                    __m512i lhs_mat_01_71 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_71), lhs_mat_ymm_01_71, 1);
                    __m512i lhs_mat_23_71 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_71), lhs_mat_ymm_23_71, 1);

                    // Bsums are loaded for the different Q8_K blocks
                    __m128i lhs_raw_bsums_01_0123 = _mm_loadu_si128((const __m128i *)((a_ptr[b].bsums + 32 * sb)));
                    __m128i lhs_raw_bsums_23_0123 = _mm_loadu_si128((const __m128i *)(a_ptr[b].bsums + 8 + 32 * sb));
                    __m128i lhs_raw_bsums_01_4567 = _mm_loadu_si128((const __m128i *)((a_ptr[b].bsums + 16 + 32 * sb)));
                    __m128i lhs_raw_bsums_23_4567 = _mm_loadu_si128((const __m128i *)(a_ptr[b].bsums + 24 + 32 * sb));

                    __m256i lhs_bsums_ymm_01_0123 = _mm256_inserti128_si256(_mm256_castsi128_si256(lhs_raw_bsums_01_0123), lhs_raw_bsums_01_0123, 1);
                    __m512i lhs_bsums_01_0123 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_bsums_ymm_01_0123), lhs_bsums_ymm_01_0123, 1);
                    __m256i lhs_bsums_ymm_23_0123 = _mm256_inserti128_si256(_mm256_castsi128_si256(lhs_raw_bsums_23_0123), lhs_raw_bsums_23_0123, 1);
                    __m512i lhs_bsums_23_0123 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_bsums_ymm_23_0123), lhs_bsums_ymm_23_0123, 1);
                    __m256i lhs_bsums_ymm_01_4567 = _mm256_inserti128_si256(_mm256_castsi128_si256(lhs_raw_bsums_01_4567), lhs_raw_bsums_01_4567, 1);
                    __m512i lhs_bsums_01_4567 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_bsums_ymm_01_4567), lhs_bsums_ymm_01_4567, 1);
                    __m256i lhs_bsums_ymm_23_4567 = _mm256_inserti128_si256(_mm256_castsi128_si256(lhs_raw_bsums_23_4567), lhs_raw_bsums_23_4567, 1);
                    __m512i lhs_bsums_23_4567 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_bsums_ymm_23_4567), lhs_bsums_ymm_23_4567, 1);

                    // Shuffle pattern one - left side input
                    const __m512i lhs_mat_01_00_sp1 = _mm512_shuffle_epi32(lhs_mat_01_00, (_MM_PERM_ENUM)160); //A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3) A00(0-3) A01(0-3) A01(0-3)
                    const __m512i lhs_mat_23_00_sp1 = _mm512_shuffle_epi32(lhs_mat_23_00, (_MM_PERM_ENUM)160); //A02(0-3) A02(0-3) A03(0-3) A03(0-3) A02(0-3) A02(0-3) A03(0-3) A03(0-3) A02(0-3) A02(0-3) A03(0-3) A03(0-3) A02(0-3) A02(0-3) A03(0-3) A03(0-3)

                    const __m512i lhs_mat_01_01_sp1 = _mm512_shuffle_epi32(lhs_mat_01_01, (_MM_PERM_ENUM)160); //A00(8-11) A00(8-11) A01(8-11) A01(8-11) A00(8-11) A00(8-11) A01(8-11) A01(8-11) A00(8-11) A00(8-11) A01(8-11) A01(8-11) A00(8-11) A00(8-11) A01(8-11) A01(8-11)
                    const __m512i lhs_mat_23_01_sp1 = _mm512_shuffle_epi32(lhs_mat_23_01, (_MM_PERM_ENUM)160); //A02(8-11) A02(8-11) A03(8-11) A03(8-11) A02(8-11) A02(8-11) A03(8-11) A03(8-11) A02(8-11) A02(8-11) A03(8-11) A03(8-11) A02(8-11) A02(8-11) A03(8-11) A03(8-11)

                    const __m512i lhs_mat_01_10_sp1 = _mm512_shuffle_epi32(lhs_mat_01_10, (_MM_PERM_ENUM)160); //A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3) A10(0-3) A11(0-3) A11(0-3)
                    const __m512i lhs_mat_23_10_sp1 = _mm512_shuffle_epi32(lhs_mat_23_10, (_MM_PERM_ENUM)160); //A12(0-3) A12(0-3) A13(0-3) A13(0-3) A12(0-3) A12(0-3) A13(0-3) A13(0-3) A12(0-3) A12(0-3) A13(0-3) A13(0-3) A12(0-3) A12(0-3) A13(0-3) A13(0-3)

                    const __m512i lhs_mat_01_11_sp1 = _mm512_shuffle_epi32(lhs_mat_01_11, (_MM_PERM_ENUM)160); //A10(8-11) A10(8-11) A11(8-11) A11(8-11) A10(8-11) A10(8-11) A11(8-11) A11(8-11) A10(8-11) A10(8-11) A11(8-11) A11(8-11) A10(8-11) A10(8-11) A11(8-11) A11(8-11)
                    const __m512i lhs_mat_23_11_sp1 = _mm512_shuffle_epi32(lhs_mat_23_11, (_MM_PERM_ENUM)160); //A12(8-11) A12(8-11) A13(8-11) A13(8-11) A12(8-11) A12(8-11) A13(8-11) A13(8-11) A12(8-11) A12(8-11) A13(8-11) A13(8-11) A12(8-11) A12(8-11) A13(8-11) A13(8-11)

                    const __m512i lhs_mat_01_20_sp1 = _mm512_shuffle_epi32(lhs_mat_01_20, (_MM_PERM_ENUM)160); //A20(0-3) A20(0-3) A21(0-3) A21(0-3) A20(0-3) A20(0-3) A21(0-3) A21(0-3) A20(0-3) A20(0-3) A21(0-3) A21(0-3) A20(0-3) A20(0-3) A21(0-3) A21(0-3)
                    const __m512i lhs_mat_23_20_sp1 = _mm512_shuffle_epi32(lhs_mat_23_20, (_MM_PERM_ENUM)160); //A22(0-3) A22(0-3) A23(0-3) A23(0-3) A22(0-3) A22(0-3) A23(0-3) A23(0-3) A22(0-3) A22(0-3) A23(0-3) A23(0-3) A22(0-3) A22(0-3) A23(0-3) A23(0-3)

                    const __m512i lhs_mat_01_21_sp1 = _mm512_shuffle_epi32(lhs_mat_01_21, (_MM_PERM_ENUM)160); //A20(8-11) A20(8-11) A21(8-11) A21(8-11) A20(8-11) A20(8-11) A21(8-11) A21(8-11) A20(8-11) A20(8-11) A21(8-11) A21(8-11) A20(8-11) A20(8-11) A21(8-11) A21(8-11)
                    const __m512i lhs_mat_23_21_sp1 = _mm512_shuffle_epi32(lhs_mat_23_21, (_MM_PERM_ENUM)160); //A22(8-11) A22(8-11) A23(8-11) A23(8-11) A22(8-11) A22(8-11) A23(8-11) A23(8-11) A22(8-11) A22(8-11) A23(8-11) A23(8-11) A22(8-11) A22(8-11) A23(8-11) A23(8-11)

                    const __m512i lhs_mat_01_30_sp1 = _mm512_shuffle_epi32(lhs_mat_01_30, (_MM_PERM_ENUM)160); //A30(0-3) A30(0-3) A31(0-3) A31(0-3) A30(0-3) A30(0-3) A31(0-3) A31(0-3) A30(0-3) A30(0-3) A31(0-3) A31(0-3) A30(0-3) A30(0-3) A31(0-3) A31(0-3)
                    const __m512i lhs_mat_23_30_sp1 = _mm512_shuffle_epi32(lhs_mat_23_30, (_MM_PERM_ENUM)160); //A32(0-3) A32(0-3) A33(0-3) A33(0-3) A32(0-3) A32(0-3) A33(0-3) A33(0-3) A32(0-3) A32(0-3) A33(0-3) A33(0-3) A32(0-3) A32(0-3) A33(0-3) A33(0-3)

                    const __m512i lhs_mat_01_31_sp1 = _mm512_shuffle_epi32(lhs_mat_01_31, (_MM_PERM_ENUM)160); //A30(8-11) A30(8-11) A31(8-11) A31(8-11) A30(8-11) A30(8-11) A31(8-11) A31(8-11) A30(8-11) A30(8-11) A31(8-11) A31(8-11) A30(8-11) A30(8-11) A31(8-11) A31(8-11)
                    const __m512i lhs_mat_23_31_sp1 = _mm512_shuffle_epi32(lhs_mat_23_31, (_MM_PERM_ENUM)160); //A32(8-11) A32(8-11) A33(8-11) A33(8-11) A32(8-11) A32(8-11) A33(8-11) A33(8-11) A32(8-11) A32(8-11) A33(8-11) A33(8-11) A32(8-11) A32(8-11) A33(8-11) A33(8-11)

                    const __m512i lhs_mat_01_40_sp1 = _mm512_shuffle_epi32(lhs_mat_01_40, (_MM_PERM_ENUM)160); //A40(0-3) A40(0-3) A41(0-3) A41(0-3) A40(0-3) A40(0-3) A41(0-3) A41(0-3) A40(0-3) A40(0-3) A41(0-3) A41(0-3) A40(0-3) A40(0-3) A41(0-3) A41(0-3)
                    const __m512i lhs_mat_23_40_sp1 = _mm512_shuffle_epi32(lhs_mat_23_40, (_MM_PERM_ENUM)160); //A42(0-3) A42(0-3) A43(0-3) A43(0-3) A42(0-3) A42(0-3) A43(0-3) A43(0-3) A42(0-3) A42(0-3) A43(0-3) A43(0-3) A42(0-3) A42(0-3) A43(0-3) A43(0-3)

                    const __m512i lhs_mat_01_41_sp1 = _mm512_shuffle_epi32(lhs_mat_01_41, (_MM_PERM_ENUM)160); //A40(8-11) A40(8-11) A41(8-11) A41(8-11) A40(8-11) A40(8-11) A41(8-11) A41(8-11) A40(8-11) A40(8-11) A41(8-11) A41(8-11) A40(8-11) A40(8-11) A41(8-11) A41(8-11)
                    const __m512i lhs_mat_23_41_sp1 = _mm512_shuffle_epi32(lhs_mat_23_41, (_MM_PERM_ENUM)160); //A42(8-11) A42(8-11) A43(8-11) A43(8-11) A42(8-11) A42(8-11) A43(8-11) A43(8-11) A42(8-11) A42(8-11) A43(8-11) A43(8-11) A42(8-11) A42(8-11) A43(8-11) A43(8-11)

                    const __m512i lhs_mat_01_50_sp1 = _mm512_shuffle_epi32(lhs_mat_01_50, (_MM_PERM_ENUM)160); //A50(0-3) A50(0-3) A51(0-3) A51(0-3) A50(0-3) A50(0-3) A51(0-3) A51(0-3) A50(0-3) A50(0-3) A51(0-3) A51(0-3) A50(0-3) A50(0-3) A51(0-3) A51(0-3)
                    const __m512i lhs_mat_23_50_sp1 = _mm512_shuffle_epi32(lhs_mat_23_50, (_MM_PERM_ENUM)160); //A52(0-3) A52(0-3) A53(0-3) A53(0-3) A52(0-3) A52(0-3) A53(0-3) A53(0-3) A52(0-3) A52(0-3) A53(0-3) A53(0-3) A52(0-3) A52(0-3) A53(0-3) A53(0-3)

                    const __m512i lhs_mat_01_51_sp1 = _mm512_shuffle_epi32(lhs_mat_01_51, (_MM_PERM_ENUM)160); //A50(8-11) A50(8-11) A51(8-11) A51(8-11) A50(8-11) A50(8-11) A51(8-11) A51(8-11) A50(8-11) A50(8-11) A51(8-11) A51(8-11) A50(8-11) A50(8-11) A51(8-11) A51(8-11)
                    const __m512i lhs_mat_23_51_sp1 = _mm512_shuffle_epi32(lhs_mat_23_51, (_MM_PERM_ENUM)160); //A52(8-11) A52(8-11) A53(8-11) A53(8-11) A52(8-11) A52(8-11) A53(8-11) A53(8-11) A52(8-11) A52(8-11) A53(8-11) A53(8-11) A52(8-11) A52(8-11) A53(8-11) A53(8-11)

                    const __m512i lhs_mat_01_60_sp1 = _mm512_shuffle_epi32(lhs_mat_01_60, (_MM_PERM_ENUM)160); //A60(0-3) A60(0-3) A61(0-3) A61(0-3) A60(0-3) A60(0-3) A61(0-3) A61(0-3) A60(0-3) A60(0-3) A61(0-3) A61(0-3) A60(0-3) A60(0-3) A61(0-3) A61(0-3)
                    const __m512i lhs_mat_23_60_sp1 = _mm512_shuffle_epi32(lhs_mat_23_60, (_MM_PERM_ENUM)160); //A62(0-3) A62(0-3) A63(0-3) A63(0-3) A62(0-3) A62(0-3) A63(0-3) A63(0-3) A62(0-3) A62(0-3) A63(0-3) A63(0-3) A62(0-3) A62(0-3) A63(0-3) A63(0-3)

                    const __m512i lhs_mat_01_61_sp1 = _mm512_shuffle_epi32(lhs_mat_01_61, (_MM_PERM_ENUM)160); //A60(8-11) A60(8-11) A61(8-11) A61(8-11) A60(8-11) A60(8-11) A61(8-11) A61(8-11) A60(8-11) A60(8-11) A61(8-11) A61(8-11) A60(8-11) A60(8-11) A61(8-11) A61(8-11)
                    const __m512i lhs_mat_23_61_sp1 = _mm512_shuffle_epi32(lhs_mat_23_61, (_MM_PERM_ENUM)160); //A62(8-11) A62(8-11) A63(8-11) A63(8-11) A62(8-11) A62(8-11) A63(8-11) A63(8-11) A62(8-11) A62(8-11) A63(8-11) A63(8-11) A62(8-11) A62(8-11) A63(8-11) A63(8-11)

                    const __m512i lhs_mat_01_70_sp1 = _mm512_shuffle_epi32(lhs_mat_01_70, (_MM_PERM_ENUM)160); //A70(0-3) A70(0-3) A71(0-3) A71(0-3) A70(0-3) A70(0-3) A71(0-3) A71(0-3) A70(0-3) A70(0-3) A71(0-3) A71(0-3) A70(0-3) A70(0-3) A71(0-3) A71(0-3)
                    const __m512i lhs_mat_23_70_sp1 = _mm512_shuffle_epi32(lhs_mat_23_70, (_MM_PERM_ENUM)160); //A72(0-3) A72(0-3) A73(0-3) A73(0-3) A72(0-3) A72(0-3) A73(0-3) A73(0-3) A72(0-3) A72(0-3) A73(0-3) A73(0-3) A72(0-3) A72(0-3) A73(0-3) A73(0-3)

                    const __m512i lhs_mat_01_71_sp1 = _mm512_shuffle_epi32(lhs_mat_01_71, (_MM_PERM_ENUM)160); //A70(8-11) A70(8-11) A71(8-11) A71(8-11) A70(8-11) A70(8-11) A71(8-11) A71(8-11) A70(8-11) A70(8-11) A71(8-11) A71(8-11) A70(8-11) A70(8-11) A71(8-11) A71(8-11)
                    const __m512i lhs_mat_23_71_sp1 = _mm512_shuffle_epi32(lhs_mat_23_71, (_MM_PERM_ENUM)160); //A72(8-11) A72(8-11) A73(8-11) A73(8-11) A72(8-11) A72(8-11) A73(8-11) A73(8-11) A72(8-11) A72(8-11) A73(8-11) A73(8-11) A72(8-11) A72(8-11) A73(8-11) A73(8-11)

                    const __m512i lhs_mat_01_00_sp2 = _mm512_shuffle_epi32(lhs_mat_01_00, (_MM_PERM_ENUM)245); //A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7) A00(4-7) A01(4-7) A01(4-7)
                    const __m512i lhs_mat_23_00_sp2 = _mm512_shuffle_epi32(lhs_mat_23_00, (_MM_PERM_ENUM)245); //A02(4-7) A02(4-7) A03(4-7) A03(4-7) A02(4-7) A02(4-7) A03(4-7) A03(4-7) A02(4-7) A02(4-7) A03(4-7) A03(4-7) A02(4-7) A02(4-7) A03(4-7) A03(4-7)

                    const __m512i lhs_mat_01_01_sp2 = _mm512_shuffle_epi32(lhs_mat_01_01, (_MM_PERM_ENUM)245); //A00(12-15) A00(12-15) A01(12-15) A01(12-15) A00(12-15) A00(12-15) A01(12-15) A01(12-15) A00(12-15) A00(12-15) A01(12-15) A01(12-15) A00(12-15) A00(12-15) A01(12-15) A01(12-15)
                    const __m512i lhs_mat_23_01_sp2 = _mm512_shuffle_epi32(lhs_mat_23_01, (_MM_PERM_ENUM)245); //A02(12-15) A02(12-15) A03(12-15) A03(12-15) A02(12-15) A02(12-15) A03(12-15) A03(12-15) A02(12-15) A02(12-15) A03(12-15) A03(12-15) A02(12-15) A02(12-15) A03(12-15) A03(12-15)

                    const __m512i lhs_mat_01_10_sp2 = _mm512_shuffle_epi32(lhs_mat_01_10, (_MM_PERM_ENUM)245); //A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7) A10(4-7) A11(4-7) A11(4-7)
                    const __m512i lhs_mat_23_10_sp2 = _mm512_shuffle_epi32(lhs_mat_23_10, (_MM_PERM_ENUM)245); //A12(4-7) A12(4-7) A13(4-7) A13(4-7) A12(4-7) A12(4-7) A13(4-7) A13(4-7) A12(4-7) A12(4-7) A13(4-7) A13(4-7) A12(4-7) A12(4-7) A13(4-7) A13(4-7)

                    const __m512i lhs_mat_01_11_sp2 = _mm512_shuffle_epi32(lhs_mat_01_11, (_MM_PERM_ENUM)245); //A10(12-15) A10(12-15) A11(12-15) A11(12-15) A10(12-15) A10(12-15) A11(12-15) A11(12-15) A10(12-15) A10(12-15) A11(12-15) A11(12-15) A10(12-15) A10(12-15) A11(12-15) A11(12-15)
                    const __m512i lhs_mat_23_11_sp2 = _mm512_shuffle_epi32(lhs_mat_23_11, (_MM_PERM_ENUM)245); //A12(12-15) A12(12-15) A13(12-15) A13(12-15) A12(12-15) A12(12-15) A13(12-15) A13(12-15) A12(12-15) A12(12-15) A13(12-15) A13(12-15) A12(12-15) A12(12-15) A13(12-15) A13(12-15)

                    const __m512i lhs_mat_01_20_sp2 = _mm512_shuffle_epi32(lhs_mat_01_20, (_MM_PERM_ENUM)245); //A20(4-7) A20(4-7) A21(4-7) A21(4-7) A20(4-7) A20(4-7) A21(4-7) A21(4-7) A20(4-7) A20(4-7) A21(4-7) A21(4-7) A20(4-7) A20(4-7) A21(4-7) A21(4-7)
                    const __m512i lhs_mat_23_20_sp2 = _mm512_shuffle_epi32(lhs_mat_23_20, (_MM_PERM_ENUM)245); //A22(4-7) A22(4-7) A23(4-7) A23(4-7) A22(4-7) A22(4-7) A23(4-7) A23(4-7) A22(4-7) A22(4-7) A23(4-7) A23(4-7) A22(4-7) A22(4-7) A23(4-7) A23(4-7)

                    const __m512i lhs_mat_01_21_sp2 = _mm512_shuffle_epi32(lhs_mat_01_21, (_MM_PERM_ENUM)245); //A20(12-15) A20(12-15) A21(12-15) A21(12-15) A20(12-15) A20(12-15) A21(12-15) A21(12-15) A20(12-15) A20(12-15) A21(12-15) A21(12-15) A20(12-15) A20(12-15) A21(12-15) A21(12-15)
                    const __m512i lhs_mat_23_21_sp2 = _mm512_shuffle_epi32(lhs_mat_23_21, (_MM_PERM_ENUM)245); //A22(12-15) A22(12-15) A23(12-15) A23(12-15) A22(12-15) A22(12-15) A23(12-15) A23(12-15) A22(12-15) A22(12-15) A23(12-15) A23(12-15) A22(12-15) A22(12-15) A23(12-15) A23(12-15)

                    const __m512i lhs_mat_01_30_sp2 = _mm512_shuffle_epi32(lhs_mat_01_30, (_MM_PERM_ENUM)245); //A30(4-7) A30(4-7) A31(4-7) A31(4-7) A30(4-7) A30(4-7) A31(4-7) A31(4-7) A30(4-7) A30(4-7) A31(4-7) A31(4-7) A30(4-7) A30(4-7) A31(4-7) A31(4-7)
                    const __m512i lhs_mat_23_30_sp2 = _mm512_shuffle_epi32(lhs_mat_23_30, (_MM_PERM_ENUM)245); //A32(4-7) A32(4-7) A33(4-7) A33(4-7) A32(4-7) A32(4-7) A33(4-7) A33(4-7) A32(4-7) A32(4-7) A33(4-7) A33(4-7) A32(4-7) A32(4-7) A33(4-7) A33(4-7)

                    const __m512i lhs_mat_01_31_sp2 = _mm512_shuffle_epi32(lhs_mat_01_31, (_MM_PERM_ENUM)245); //A30(12-15) A30(12-15) A31(12-15) A31(12-15) A30(12-15) A30(12-15) A31(12-15) A31(12-15) A30(12-15) A30(12-15) A31(12-15) A31(12-15) A30(12-15) A30(12-15) A31(12-15) A31(12-15)
                    const __m512i lhs_mat_23_31_sp2 = _mm512_shuffle_epi32(lhs_mat_23_31, (_MM_PERM_ENUM)245); //A32(12-15) A32(12-15) A33(12-15) A33(12-15) A32(12-15) A32(12-15) A33(12-15) A33(12-15) A32(12-15) A32(12-15) A33(12-15) A33(12-15) A32(12-15) A32(12-15) A33(12-15) A33(12-15)

                    const __m512i lhs_mat_01_40_sp2 = _mm512_shuffle_epi32(lhs_mat_01_40, (_MM_PERM_ENUM)245); //A40(4-7) A40(4-7) A41(4-7) A41(4-7) A40(4-7) A40(4-7) A41(4-7) A41(4-7) A40(4-7) A40(4-7) A41(4-7) A41(4-7) A40(4-7) A40(4-7) A41(4-7) A41(4-7)
                    const __m512i lhs_mat_23_40_sp2 = _mm512_shuffle_epi32(lhs_mat_23_40, (_MM_PERM_ENUM)245); //A42(4-7) A42(4-7) A43(4-7) A43(4-7) A42(4-7) A42(4-7) A43(4-7) A43(4-7) A42(4-7) A42(4-7) A43(4-7) A43(4-7) A42(4-7) A42(4-7) A43(4-7) A43(4-7)

                    const __m512i lhs_mat_01_41_sp2 = _mm512_shuffle_epi32(lhs_mat_01_41, (_MM_PERM_ENUM)245); //A40(12-15) A40(12-15) A41(12-15) A41(12-15) A40(12-15) A40(12-15) A41(12-15) A41(12-15) A40(12-15) A40(12-15) A41(12-15) A41(12-15) A40(12-15) A40(12-15) A41(12-15) A41(12-15)
                    const __m512i lhs_mat_23_41_sp2 = _mm512_shuffle_epi32(lhs_mat_23_41, (_MM_PERM_ENUM)245); //A42(12-15) A42(12-15) A43(12-15) A43(12-15) A42(12-15) A42(12-15) A43(12-15) A43(12-15) A42(12-15) A42(12-15) A43(12-15) A43(12-15) A42(12-15) A42(12-15) A43(12-15) A43(12-15)

                    const __m512i lhs_mat_01_50_sp2 = _mm512_shuffle_epi32(lhs_mat_01_50, (_MM_PERM_ENUM)245); //A50(4-7) A50(4-7) A51(4-7) A51(4-7) A50(4-7) A50(4-7) A51(4-7) A51(4-7) A50(4-7) A50(4-7) A51(4-7) A51(4-7) A50(4-7) A50(4-7) A51(4-7) A51(4-7)
                    const __m512i lhs_mat_23_50_sp2 = _mm512_shuffle_epi32(lhs_mat_23_50, (_MM_PERM_ENUM)245); //A52(4-7) A52(4-7) A53(4-7) A53(4-7) A52(4-7) A52(4-7) A53(4-7) A53(4-7) A52(4-7) A52(4-7) A53(4-7) A53(4-7) A52(4-7) A52(4-7) A53(4-7) A53(4-7)

                    const __m512i lhs_mat_01_51_sp2 = _mm512_shuffle_epi32(lhs_mat_01_51, (_MM_PERM_ENUM)245); //A50(12-15) A50(12-15) A51(12-15) A51(12-15) A50(12-15) A50(12-15) A51(12-15) A51(12-15) A50(12-15) A50(12-15) A51(12-15) A51(12-15) A50(12-15) A50(12-15) A51(12-15) A51(12-15)
                    const __m512i lhs_mat_23_51_sp2 = _mm512_shuffle_epi32(lhs_mat_23_51, (_MM_PERM_ENUM)245); //A52(12-15) A52(12-15) A53(12-15) A53(12-15) A52(12-15) A52(12-15) A53(12-15) A53(12-15) A52(12-15) A52(12-15) A53(12-15) A53(12-15) A52(12-15) A52(12-15) A53(12-15) A53(12-15)

                    const __m512i lhs_mat_01_60_sp2 = _mm512_shuffle_epi32(lhs_mat_01_60, (_MM_PERM_ENUM)245); //A60(4-7) A60(4-7) A61(4-7) A61(4-7) A60(4-7) A60(4-7) A61(4-7) A61(4-7) A60(4-7) A60(4-7) A61(4-7) A61(4-7) A60(4-7) A60(4-7) A61(4-7) A61(4-7)
                    const __m512i lhs_mat_23_60_sp2 = _mm512_shuffle_epi32(lhs_mat_23_60, (_MM_PERM_ENUM)245); //A62(4-7) A62(4-7) A63(4-7) A63(4-7) A62(4-7) A62(4-7) A63(4-7) A63(4-7) A62(4-7) A62(4-7) A63(4-7) A63(4-7) A62(4-7) A62(4-7) A63(4-7) A63(4-7)

                    const __m512i lhs_mat_01_61_sp2 = _mm512_shuffle_epi32(lhs_mat_01_61, (_MM_PERM_ENUM)245); //A60(12-15) A60(12-15) A61(12-15) A61(12-15) A60(12-15) A60(12-15) A61(12-15) A61(12-15) A60(12-15) A60(12-15) A61(12-15) A61(12-15) A60(12-15) A60(12-15) A61(12-15) A61(12-15)
                    const __m512i lhs_mat_23_61_sp2 = _mm512_shuffle_epi32(lhs_mat_23_61, (_MM_PERM_ENUM)245); //A62(12-15) A62(12-15) A63(12-15) A63(12-15) A62(12-15) A62(12-15) A63(12-15) A63(12-15) A62(12-15) A62(12-15) A63(12-15) A63(12-15) A62(12-15) A62(12-15) A63(12-15) A63(12-15)

                    const __m512i lhs_mat_01_70_sp2 = _mm512_shuffle_epi32(lhs_mat_01_70, (_MM_PERM_ENUM)245); //A70(4-7) A70(4-7) A71(4-7) A71(4-7) A70(4-7) A70(4-7) A71(4-7) A71(4-7) A70(4-7) A70(4-7) A71(4-7) A71(4-7) A70(4-7) A70(4-7) A71(4-7) A71(4-7)
                    const __m512i lhs_mat_23_70_sp2 = _mm512_shuffle_epi32(lhs_mat_23_70, (_MM_PERM_ENUM)245); //A72(4-7) A72(4-7) A73(4-7) A73(4-7) A72(4-7) A72(4-7) A73(4-7) A73(4-7) A72(4-7) A72(4-7) A73(4-7) A73(4-7) A72(4-7) A72(4-7) A73(4-7) A73(4-7)

                    const __m512i lhs_mat_01_71_sp2 = _mm512_shuffle_epi32(lhs_mat_01_71, (_MM_PERM_ENUM)245); //A70(12-15) A70(12-15) A71(12-15) A71(12-15) A70(12-15) A70(12-15) A71(12-15) A71(12-15) A70(12-15) A70(12-15) A71(12-15) A71(12-15) A70(12-15) A70(12-15) A71(12-15) A71(12-15)
                    const __m512i lhs_mat_23_71_sp2 = _mm512_shuffle_epi32(lhs_mat_23_71, (_MM_PERM_ENUM)245); //A72(12-15) A72(12-15) A73(12-15) A73(12-15) A72(12-15) A72(12-15) A73(12-15) A73(12-15) A72(12-15) A72(12-15) A73(12-15) A73(12-15) A72(12-15) A72(12-15) A73(12-15) A73(12-15)

                    // The values arranged in shuffle patterns are operated with dot product operation within 32 bit lane i.e corresponding bytes and multiplied and added into 32 bit integers within 32 bit lane
                    __m512i iacc_mat_00_0_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_00_sp1, lhs_mat_01_00_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_01_sp1, lhs_mat_01_01_sp1));
                    __m512i iacc_mat_01_0_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_00_sp1, lhs_mat_01_00_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_01_sp1, lhs_mat_01_01_sp1));

                    __m512i iacc_mat_10_0_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_00_sp1, lhs_mat_23_00_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_01_sp1, lhs_mat_23_01_sp1));
                    __m512i iacc_mat_11_0_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_00_sp1, lhs_mat_23_00_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_01_sp1, lhs_mat_23_01_sp1));

                    __m512i iacc_mat_00_1_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_10_sp1, lhs_mat_01_10_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_11_sp1, lhs_mat_01_11_sp1));
                    __m512i iacc_mat_01_1_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_10_sp1, lhs_mat_01_10_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_11_sp1, lhs_mat_01_11_sp1));

                    __m512i iacc_mat_10_1_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_10_sp1, lhs_mat_23_10_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_11_sp1, lhs_mat_23_11_sp1));
                    __m512i iacc_mat_11_1_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_10_sp1, lhs_mat_23_10_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_11_sp1, lhs_mat_23_11_sp1));

                    __m512i iacc_mat_00_2_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_20_sp1, lhs_mat_01_20_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_21_sp1, lhs_mat_01_21_sp1));
                    __m512i iacc_mat_01_2_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_20_sp1, lhs_mat_01_20_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_21_sp1, lhs_mat_01_21_sp1));

                    __m512i iacc_mat_10_2_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_20_sp1, lhs_mat_23_20_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_21_sp1, lhs_mat_23_21_sp1));
                    __m512i iacc_mat_11_2_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_20_sp1, lhs_mat_23_20_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_21_sp1, lhs_mat_23_21_sp1));

                    __m512i iacc_mat_00_3_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_30_sp1, lhs_mat_01_30_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_31_sp1, lhs_mat_01_31_sp1));
                    __m512i iacc_mat_01_3_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_30_sp1, lhs_mat_01_30_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_31_sp1, lhs_mat_01_31_sp1));

                    __m512i iacc_mat_10_3_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_30_sp1, lhs_mat_23_30_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_31_sp1, lhs_mat_23_31_sp1));
                    __m512i iacc_mat_11_3_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_30_sp1, lhs_mat_23_30_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_31_sp1, lhs_mat_23_31_sp1));

                    __m512i iacc_mat_00_4_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_40_sp1, lhs_mat_01_40_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_41_sp1, lhs_mat_01_41_sp1));
                    __m512i iacc_mat_01_4_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_40_sp1, lhs_mat_01_40_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_41_sp1, lhs_mat_01_41_sp1));

                    __m512i iacc_mat_10_4_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_40_sp1, lhs_mat_23_40_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_41_sp1, lhs_mat_23_41_sp1));
                    __m512i iacc_mat_11_4_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_40_sp1, lhs_mat_23_40_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_41_sp1, lhs_mat_23_41_sp1));

                    __m512i iacc_mat_00_5_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_50_sp1, lhs_mat_01_50_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_51_sp1, lhs_mat_01_51_sp1));
                    __m512i iacc_mat_01_5_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_50_sp1, lhs_mat_01_50_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_51_sp1, lhs_mat_01_51_sp1));

                    __m512i iacc_mat_10_5_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_50_sp1, lhs_mat_23_50_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_51_sp1, lhs_mat_23_51_sp1));
                    __m512i iacc_mat_11_5_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_50_sp1, lhs_mat_23_50_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_51_sp1, lhs_mat_23_51_sp1));

                    __m512i iacc_mat_00_6_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_60_sp1, lhs_mat_01_60_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_61_sp1, lhs_mat_01_61_sp1));
                    __m512i iacc_mat_01_6_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_60_sp1, lhs_mat_01_60_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_61_sp1, lhs_mat_01_61_sp1));

                    __m512i iacc_mat_10_6_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_60_sp1, lhs_mat_23_60_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_61_sp1, lhs_mat_23_61_sp1));
                    __m512i iacc_mat_11_6_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_60_sp1, lhs_mat_23_60_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_61_sp1, lhs_mat_23_61_sp1));

                    __m512i iacc_mat_00_7_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_70_sp1, lhs_mat_01_70_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_71_sp1, lhs_mat_01_71_sp1));
                    __m512i iacc_mat_01_7_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_70_sp1, lhs_mat_01_70_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_71_sp1, lhs_mat_01_71_sp1));

                    __m512i iacc_mat_10_7_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_70_sp1, lhs_mat_23_70_sp1),_mm512_maddubs_epi16(rhs_mat_014589CD_71_sp1, lhs_mat_23_71_sp1));
                    __m512i iacc_mat_11_7_sp1 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_70_sp1, lhs_mat_23_70_sp1),_mm512_maddubs_epi16(rhs_mat_2367ABEF_71_sp1, lhs_mat_23_71_sp1));


                    __m512i iacc_mat_00_0_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_00_sp2, lhs_mat_01_00_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_01_sp2, lhs_mat_01_01_sp2));
                    __m512i iacc_mat_01_0_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_00_sp2, lhs_mat_01_00_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_01_sp2, lhs_mat_01_01_sp2));

                    __m512i iacc_mat_10_0_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_00_sp2, lhs_mat_23_00_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_01_sp2, lhs_mat_23_01_sp2));
                    __m512i iacc_mat_11_0_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_00_sp2, lhs_mat_23_00_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_01_sp2, lhs_mat_23_01_sp2));

                    __m512i iacc_mat_00_1_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_10_sp2, lhs_mat_01_10_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_11_sp2, lhs_mat_01_11_sp2));
                    __m512i iacc_mat_01_1_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_10_sp2, lhs_mat_01_10_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_11_sp2, lhs_mat_01_11_sp2));

                    __m512i iacc_mat_10_1_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_10_sp2, lhs_mat_23_10_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_11_sp2, lhs_mat_23_11_sp2));
                    __m512i iacc_mat_11_1_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_10_sp2, lhs_mat_23_10_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_11_sp2, lhs_mat_23_11_sp2));

                    __m512i iacc_mat_00_2_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_20_sp2, lhs_mat_01_20_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_21_sp2, lhs_mat_01_21_sp2));
                    __m512i iacc_mat_01_2_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_20_sp2, lhs_mat_01_20_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_21_sp2, lhs_mat_01_21_sp2));

                    __m512i iacc_mat_10_2_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_20_sp2, lhs_mat_23_20_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_21_sp2, lhs_mat_23_21_sp2));
                    __m512i iacc_mat_11_2_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_20_sp2, lhs_mat_23_20_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_21_sp2, lhs_mat_23_21_sp2));

                    __m512i iacc_mat_00_3_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_30_sp2, lhs_mat_01_30_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_31_sp2, lhs_mat_01_31_sp2));
                    __m512i iacc_mat_01_3_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_30_sp2, lhs_mat_01_30_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_31_sp2, lhs_mat_01_31_sp2));

                    __m512i iacc_mat_10_3_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_30_sp2, lhs_mat_23_30_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_31_sp2, lhs_mat_23_31_sp2));
                    __m512i iacc_mat_11_3_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_30_sp2, lhs_mat_23_30_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_31_sp2, lhs_mat_23_31_sp2));

                    __m512i iacc_mat_00_4_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_40_sp2, lhs_mat_01_40_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_41_sp2, lhs_mat_01_41_sp2));
                    __m512i iacc_mat_01_4_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_40_sp2, lhs_mat_01_40_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_41_sp2, lhs_mat_01_41_sp2));

                    __m512i iacc_mat_10_4_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_40_sp2, lhs_mat_23_40_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_41_sp2, lhs_mat_23_41_sp2));
                    __m512i iacc_mat_11_4_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_40_sp2, lhs_mat_23_40_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_41_sp2, lhs_mat_23_41_sp2));

                    __m512i iacc_mat_00_5_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_50_sp2, lhs_mat_01_50_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_51_sp2, lhs_mat_01_51_sp2));
                    __m512i iacc_mat_01_5_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_50_sp2, lhs_mat_01_50_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_51_sp2, lhs_mat_01_51_sp2));

                    __m512i iacc_mat_10_5_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_50_sp2, lhs_mat_23_50_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_51_sp2, lhs_mat_23_51_sp2));
                    __m512i iacc_mat_11_5_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_50_sp2, lhs_mat_23_50_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_51_sp2, lhs_mat_23_51_sp2));

                    __m512i iacc_mat_00_6_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_60_sp2, lhs_mat_01_60_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_61_sp2, lhs_mat_01_61_sp2));
                    __m512i iacc_mat_01_6_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_60_sp2, lhs_mat_01_60_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_61_sp2, lhs_mat_01_61_sp2));

                    __m512i iacc_mat_10_6_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_60_sp2, lhs_mat_23_60_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_61_sp2, lhs_mat_23_61_sp2));
                    __m512i iacc_mat_11_6_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_60_sp2, lhs_mat_23_60_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_61_sp2, lhs_mat_23_61_sp2));

                    __m512i iacc_mat_00_7_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_70_sp2, lhs_mat_01_70_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_71_sp2, lhs_mat_01_71_sp2));
                    __m512i iacc_mat_01_7_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_70_sp2, lhs_mat_01_70_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_71_sp2, lhs_mat_01_71_sp2));

                    __m512i iacc_mat_10_7_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_70_sp2, lhs_mat_23_70_sp2),_mm512_maddubs_epi16(rhs_mat_014589CD_71_sp2, lhs_mat_23_71_sp2));
                    __m512i iacc_mat_11_7_sp2 = _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_70_sp2, lhs_mat_23_70_sp2),_mm512_maddubs_epi16(rhs_mat_2367ABEF_71_sp2, lhs_mat_23_71_sp2));

                    // Combine results from both shuffle patterns for each output block
                    __m512i iacc_mat_00_0 = _mm512_add_epi16(iacc_mat_00_0_sp1, iacc_mat_00_0_sp2);
                    __m512i iacc_mat_01_0 = _mm512_add_epi16(iacc_mat_01_0_sp1, iacc_mat_01_0_sp2);
                    __m512i iacc_mat_10_0 = _mm512_add_epi16(iacc_mat_10_0_sp1, iacc_mat_10_0_sp2);
                    __m512i iacc_mat_11_0 = _mm512_add_epi16(iacc_mat_11_0_sp1, iacc_mat_11_0_sp2);

                    __m512i iacc_mat_00_1 = _mm512_add_epi16(iacc_mat_00_1_sp1, iacc_mat_00_1_sp2);
                    __m512i iacc_mat_01_1 = _mm512_add_epi16(iacc_mat_01_1_sp1, iacc_mat_01_1_sp2);
                    __m512i iacc_mat_10_1 = _mm512_add_epi16(iacc_mat_10_1_sp1, iacc_mat_10_1_sp2);
                    __m512i iacc_mat_11_1 = _mm512_add_epi16(iacc_mat_11_1_sp1, iacc_mat_11_1_sp2);

                    __m512i iacc_mat_00_2 = _mm512_add_epi16(iacc_mat_00_2_sp1, iacc_mat_00_2_sp2);
                    __m512i iacc_mat_01_2 = _mm512_add_epi16(iacc_mat_01_2_sp1, iacc_mat_01_2_sp2);
                    __m512i iacc_mat_10_2 = _mm512_add_epi16(iacc_mat_10_2_sp1, iacc_mat_10_2_sp2);
                    __m512i iacc_mat_11_2 = _mm512_add_epi16(iacc_mat_11_2_sp1, iacc_mat_11_2_sp2);

                    __m512i iacc_mat_00_3 = _mm512_add_epi16(iacc_mat_00_3_sp1, iacc_mat_00_3_sp2);
                    __m512i iacc_mat_01_3 = _mm512_add_epi16(iacc_mat_01_3_sp1, iacc_mat_01_3_sp2);
                    __m512i iacc_mat_10_3 = _mm512_add_epi16(iacc_mat_10_3_sp1, iacc_mat_10_3_sp2);
                    __m512i iacc_mat_11_3 = _mm512_add_epi16(iacc_mat_11_3_sp1, iacc_mat_11_3_sp2);

                    __m512i iacc_mat_00_4 = _mm512_add_epi16(iacc_mat_00_4_sp1, iacc_mat_00_4_sp2);
                    __m512i iacc_mat_01_4 = _mm512_add_epi16(iacc_mat_01_4_sp1, iacc_mat_01_4_sp2);
                    __m512i iacc_mat_10_4 = _mm512_add_epi16(iacc_mat_10_4_sp1, iacc_mat_10_4_sp2);
                    __m512i iacc_mat_11_4 = _mm512_add_epi16(iacc_mat_11_4_sp1, iacc_mat_11_4_sp2);

                    __m512i iacc_mat_00_5 = _mm512_add_epi16(iacc_mat_00_5_sp1, iacc_mat_00_5_sp2);
                    __m512i iacc_mat_01_5 = _mm512_add_epi16(iacc_mat_01_5_sp1, iacc_mat_01_5_sp2);
                    __m512i iacc_mat_10_5 = _mm512_add_epi16(iacc_mat_10_5_sp1, iacc_mat_10_5_sp2);
                    __m512i iacc_mat_11_5 = _mm512_add_epi16(iacc_mat_11_5_sp1, iacc_mat_11_5_sp2);

                    __m512i iacc_mat_00_6 = _mm512_add_epi16(iacc_mat_00_6_sp1, iacc_mat_00_6_sp2);
                    __m512i iacc_mat_01_6 = _mm512_add_epi16(iacc_mat_01_6_sp1, iacc_mat_01_6_sp2);
                    __m512i iacc_mat_10_6 = _mm512_add_epi16(iacc_mat_10_6_sp1, iacc_mat_10_6_sp2);
                    __m512i iacc_mat_11_6 = _mm512_add_epi16(iacc_mat_11_6_sp1, iacc_mat_11_6_sp2);

                    __m512i iacc_mat_00_7 = _mm512_add_epi16(iacc_mat_00_7_sp1, iacc_mat_00_7_sp2);
                    __m512i iacc_mat_01_7 = _mm512_add_epi16(iacc_mat_01_7_sp1, iacc_mat_01_7_sp2);
                    __m512i iacc_mat_10_7 = _mm512_add_epi16(iacc_mat_10_7_sp1, iacc_mat_10_7_sp2);
                    __m512i iacc_mat_11_7 = _mm512_add_epi16(iacc_mat_11_7_sp1, iacc_mat_11_7_sp2);

                    // Output of both shuffle patterns are added in order to sum dot product outputs of all 32 values in block
                    iacc_mat_00_0 = _mm512_madd_epi16(iacc_mat_00_0, scale_014589CD_0);
                    iacc_mat_01_0 = _mm512_madd_epi16(iacc_mat_01_0, scale_2367ABEF_0);
                    iacc_mat_10_0 = _mm512_madd_epi16(iacc_mat_10_0, scale_014589CD_0);
                    iacc_mat_11_0 = _mm512_madd_epi16(iacc_mat_11_0, scale_2367ABEF_0);

                    iacc_mat_00_1 = _mm512_madd_epi16(iacc_mat_00_1, scale_014589CD_1);
                    iacc_mat_01_1 = _mm512_madd_epi16(iacc_mat_01_1, scale_2367ABEF_1);
                    iacc_mat_10_1 = _mm512_madd_epi16(iacc_mat_10_1, scale_014589CD_1);
                    iacc_mat_11_1 = _mm512_madd_epi16(iacc_mat_11_1, scale_2367ABEF_1);

                    iacc_mat_00_2 = _mm512_madd_epi16(iacc_mat_00_2, scale_014589CD_2);
                    iacc_mat_01_2 = _mm512_madd_epi16(iacc_mat_01_2, scale_2367ABEF_2);
                    iacc_mat_10_2 = _mm512_madd_epi16(iacc_mat_10_2, scale_014589CD_2);
                    iacc_mat_11_2 = _mm512_madd_epi16(iacc_mat_11_2, scale_2367ABEF_2);

                    iacc_mat_00_3 = _mm512_madd_epi16(iacc_mat_00_3, scale_014589CD_3);
                    iacc_mat_01_3 = _mm512_madd_epi16(iacc_mat_01_3, scale_2367ABEF_3);
                    iacc_mat_10_3 = _mm512_madd_epi16(iacc_mat_10_3, scale_014589CD_3);
                    iacc_mat_11_3 = _mm512_madd_epi16(iacc_mat_11_3, scale_2367ABEF_3);

                    iacc_mat_00_4 = _mm512_madd_epi16(iacc_mat_00_4, scale_014589CD_4);
                    iacc_mat_01_4 = _mm512_madd_epi16(iacc_mat_01_4, scale_2367ABEF_4);
                    iacc_mat_10_4 = _mm512_madd_epi16(iacc_mat_10_4, scale_014589CD_4);
                    iacc_mat_11_4 = _mm512_madd_epi16(iacc_mat_11_4, scale_2367ABEF_4);

                    iacc_mat_00_5 = _mm512_madd_epi16(iacc_mat_00_5, scale_014589CD_5);
                    iacc_mat_01_5 = _mm512_madd_epi16(iacc_mat_01_5, scale_2367ABEF_5);
                    iacc_mat_10_5 = _mm512_madd_epi16(iacc_mat_10_5, scale_014589CD_5);
                    iacc_mat_11_5 = _mm512_madd_epi16(iacc_mat_11_5, scale_2367ABEF_5);

                    iacc_mat_00_6 = _mm512_madd_epi16(iacc_mat_00_6, scale_014589CD_6);
                    iacc_mat_01_6 = _mm512_madd_epi16(iacc_mat_01_6, scale_2367ABEF_6);
                    iacc_mat_10_6 = _mm512_madd_epi16(iacc_mat_10_6, scale_014589CD_6);
                    iacc_mat_11_6 = _mm512_madd_epi16(iacc_mat_11_6, scale_2367ABEF_6);

                    iacc_mat_00_7 = _mm512_madd_epi16(iacc_mat_00_7, scale_014589CD_7);
                    iacc_mat_01_7 = _mm512_madd_epi16(iacc_mat_01_7, scale_2367ABEF_7);
                    iacc_mat_10_7 = _mm512_madd_epi16(iacc_mat_10_7, scale_014589CD_7);
                    iacc_mat_11_7 = _mm512_madd_epi16(iacc_mat_11_7, scale_2367ABEF_7);

                    __m512i iacc_mat_00 = _mm512_add_epi32(_mm512_add_epi32(_mm512_add_epi32(iacc_mat_00_0, iacc_mat_00_1), _mm512_add_epi32(iacc_mat_00_2, iacc_mat_00_3)), _mm512_add_epi32(_mm512_add_epi32(iacc_mat_00_4, iacc_mat_00_5), _mm512_add_epi32(iacc_mat_00_6, iacc_mat_00_7)));
                    __m512i iacc_mat_01 = _mm512_add_epi32(_mm512_add_epi32(_mm512_add_epi32(iacc_mat_01_0, iacc_mat_01_1), _mm512_add_epi32(iacc_mat_01_2, iacc_mat_01_3)), _mm512_add_epi32(_mm512_add_epi32(iacc_mat_01_4, iacc_mat_01_5), _mm512_add_epi32(iacc_mat_01_6, iacc_mat_01_7)));
                    __m512i iacc_mat_10 = _mm512_add_epi32(_mm512_add_epi32(_mm512_add_epi32(iacc_mat_10_0, iacc_mat_10_1), _mm512_add_epi32(iacc_mat_10_2, iacc_mat_10_3)), _mm512_add_epi32(_mm512_add_epi32(iacc_mat_10_4, iacc_mat_10_5), _mm512_add_epi32(iacc_mat_10_6, iacc_mat_10_7)));
                    __m512i iacc_mat_11 = _mm512_add_epi32(_mm512_add_epi32(_mm512_add_epi32(iacc_mat_11_0, iacc_mat_11_1), _mm512_add_epi32(iacc_mat_11_2, iacc_mat_11_3)), _mm512_add_epi32(_mm512_add_epi32(iacc_mat_11_4, iacc_mat_11_5), _mm512_add_epi32(iacc_mat_11_6, iacc_mat_11_7)));

                    // Straighten out to make 4 row vectors
                    __m512i iacc_row_0 = _mm512_mask_blend_epi32(0xCCCC, iacc_mat_00, _mm512_shuffle_epi32(iacc_mat_01, (_MM_PERM_ENUM)78));
                    __m512i iacc_row_1 = _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32(iacc_mat_00, (_MM_PERM_ENUM)78), iacc_mat_01);
                    __m512i iacc_row_2 = _mm512_mask_blend_epi32(0xCCCC, iacc_mat_10, _mm512_shuffle_epi32(iacc_mat_11, (_MM_PERM_ENUM)78));
                    __m512i iacc_row_3 = _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32(iacc_mat_10, (_MM_PERM_ENUM)78), iacc_mat_11);

                    // Load the scale(d) values for all the 4 Q8_k blocks and repeat it across lanes
                    const __m128 row_scale_f32_sse = _mm_load_ps(a_ptr[b].d);
                    const __m256 row_scale_f32_ymm = _mm256_set_m128(row_scale_f32_sse, row_scale_f32_sse);
                    const __m512 row_scale_f32 = _mm512_insertf32x8(_mm512_castps256_ps512(row_scale_f32_ymm), row_scale_f32_ymm, 1);

                    // Multiply with appropiate scales and accumulate (for both d and dmin) below
                    acc_rows[0] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_0), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 0)), acc_rows[0]);
                    acc_rows[1] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_1), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 85)), acc_rows[1]);
                    acc_rows[2] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_2), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_rows[2]);
                    acc_rows[3] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_3), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_rows[3]);

                    // Take two bsums from two Q8_Ks at a time and multiply with corresponding mins values from each Q2_K
                    __m512i iacc_row_min_0_01 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_01_0123, (_MM_PERM_ENUM)0), mins_01);
                    __m512i iacc_row_min_1_01 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_01_0123, (_MM_PERM_ENUM)170), mins_01);
                    __m512i iacc_row_min_2_01 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_23_0123, (_MM_PERM_ENUM)0), mins_01);
                    __m512i iacc_row_min_3_01 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_23_0123, (_MM_PERM_ENUM)170), mins_01);

                    __m512i iacc_row_min_0_23 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_01_0123, (_MM_PERM_ENUM)85), mins_23);
                    __m512i iacc_row_min_1_23 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_01_0123, (_MM_PERM_ENUM)255), mins_23);
                    __m512i iacc_row_min_2_23 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_23_0123, (_MM_PERM_ENUM)85), mins_23);
                    __m512i iacc_row_min_3_23 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_23_0123, (_MM_PERM_ENUM)255), mins_23);

                    __m512i iacc_row_min_0_45 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_01_4567, (_MM_PERM_ENUM)0), mins_45);
                    __m512i iacc_row_min_1_45 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_01_4567, (_MM_PERM_ENUM)170), mins_45);
                    __m512i iacc_row_min_2_45 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_23_4567, (_MM_PERM_ENUM)0), mins_45);
                    __m512i iacc_row_min_3_45 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_23_4567, (_MM_PERM_ENUM)170), mins_45);

                    __m512i iacc_row_min_0_67 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_01_4567, (_MM_PERM_ENUM)85), mins_67);
                    __m512i iacc_row_min_1_67 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_01_4567, (_MM_PERM_ENUM)255), mins_67);
                    __m512i iacc_row_min_2_67 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_23_4567, (_MM_PERM_ENUM)85), mins_67);
                    __m512i iacc_row_min_3_67 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_23_4567, (_MM_PERM_ENUM)255), mins_67);

                    __m512i iacc_row_min_0 = _mm512_add_epi32(_mm512_add_epi32(iacc_row_min_0_01, iacc_row_min_0_23), _mm512_add_epi32(iacc_row_min_0_45,iacc_row_min_0_67));
                    __m512i iacc_row_min_1 = _mm512_add_epi32(_mm512_add_epi32(iacc_row_min_1_01, iacc_row_min_1_23), _mm512_add_epi32(iacc_row_min_1_45,iacc_row_min_1_67));
                    __m512i iacc_row_min_2 = _mm512_add_epi32(_mm512_add_epi32(iacc_row_min_2_01, iacc_row_min_2_23), _mm512_add_epi32(iacc_row_min_2_45,iacc_row_min_2_67));
                    __m512i iacc_row_min_3 = _mm512_add_epi32(_mm512_add_epi32(iacc_row_min_3_01, iacc_row_min_3_23), _mm512_add_epi32(iacc_row_min_3_45,iacc_row_min_3_67));

                    acc_min_rows[0] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_min_0), _mm512_mul_ps(col_dmin_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 0)), acc_min_rows[0]);
                    acc_min_rows[1] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_min_1), _mm512_mul_ps(col_dmin_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 85)), acc_min_rows[1]);
                    acc_min_rows[2] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_min_2), _mm512_mul_ps(col_dmin_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_min_rows[2]);
                    acc_min_rows[3] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_min_3), _mm512_mul_ps(col_dmin_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_min_rows[3]);
                }
            }
            // Store accumulated values
            for (int i = 0; i < 4; i++) {
                _mm512_storeu_ps((float * )(s + ((y * 4 + i) * bs + x * 8)), _mm512_sub_ps(acc_rows[i], acc_min_rows[i]));
            }
        }
    }

    if (anc != nc) {
        xstart = anc/8;
        y = 0;
    }

#endif // __AVX512BW__ && __AVX512DQ__

    // Take group of four block_q8_Kx4 structures at each pass of the loop and perform dot product operation
    for (; y < anr / 4; y += 4) {

        const block_q8_Kx4 * a_ptrs[4];

        a_ptrs[0] = a_ptr_start + (y * nb);
        for (int i = 0; i < 3; ++i) {
            a_ptrs[i + 1] = a_ptrs[i] + nb;
        }

        // Take group of eight block_q2_kx8 structures at each pass of the loop and perform dot product operation
        for (int64_t x = xstart; x < nc / 8; x++) {

            const block_q2_Kx8 * b_ptr = b_ptr_start + (x * b_nb);

            // Master FP accumulators
            __m256 acc_rows[16];
            for (int i = 0; i < 16; i++) {
                acc_rows[i] = _mm256_setzero_ps();
            }

            __m256 acc_min_rows[16];
            for (int i = 0; i < 16; i++) {
                acc_min_rows[i] = _mm256_setzero_ps();
            }

            // For super block
            for (int64_t b = 0; b < nb; b++) {
                // Delta values - Load the eight scale values of block_q2_kx8
                const __m256 col_scale_f32 = GGML_F32Cx8_LOAD(b_ptr[b].d);

                // dmin values - Load the eight dmin values of block_q2_kx8
                const __m256 col_dmin_f32 = GGML_F32Cx8_LOAD(b_ptr[b].dmin);

                // Loop to iterate over the sixteen sub blocks of a super block - eight sub blocks are processed per iteration
                for (int sb = 0; sb < QK_K / 128; sb++) {

                    // Load the eight block_q2_K for eight sub blocks quantized values interleaved with each other in chunks of eight bytes - B0,B1 ....B6,B7
                    const __m256i rhs_raw_mat_0123_0 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + sb * 256));
                    const __m256i rhs_raw_mat_4567_0 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 32 + sb * 256));
                    const __m256i rhs_raw_mat_0123_1 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 64 + sb * 256));
                    const __m256i rhs_raw_mat_4567_1 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 96 + sb * 256));
                    const __m256i rhs_raw_mat_0123_2 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 128 + sb * 256));
                    const __m256i rhs_raw_mat_4567_2 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 160 + sb * 256));
                    const __m256i rhs_raw_mat_0123_3 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 192 + sb * 256));
                    const __m256i rhs_raw_mat_4567_3 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 224 + sb * 256));

                    // Save the values in the following vectors in the formats B0B1B4B5, B2B3B6B7 for further processing and storing of values
                    //superblock    sub block   which part of sub block
                    const __m256i rhs_raw_mat_0145_0 = _mm256_blend_epi32(rhs_raw_mat_0123_0, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_0, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_0, requiredOrder), rhs_raw_mat_4567_0, 240);

                    const __m256i rhs_raw_mat_0145_1 = _mm256_blend_epi32(rhs_raw_mat_0123_1, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_1, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_1, requiredOrder), rhs_raw_mat_4567_1, 240);

                    const __m256i rhs_raw_mat_0145_2 = _mm256_blend_epi32(rhs_raw_mat_0123_2, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_2, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_2 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_2, requiredOrder), rhs_raw_mat_4567_2, 240);

                    const __m256i rhs_raw_mat_0145_3 = _mm256_blend_epi32(rhs_raw_mat_0123_3, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_3, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_3 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_3, requiredOrder), rhs_raw_mat_4567_3, 240);

                    // 2-bit -> 8-bit
                    // First sub block of the eight sub blocks processed in the iteration
                    const __m256i rhs_mat_0145_00 = _mm256_and_si256(rhs_raw_mat_0145_0, m3b); //B00(0-7) B01(0-7) B04(0-7) B05(0-7)
                    const __m256i rhs_mat_2367_00 = _mm256_and_si256(rhs_raw_mat_2367_0, m3b); //B02(0-7) B03(0-7) B06(0-7) B07(0-7)

                    const __m256i rhs_mat_0145_01 = _mm256_and_si256(rhs_raw_mat_0145_1, m3b); //B00(8-15) B01(8-15) B04(8-15) B05(8-15)
                    const __m256i rhs_mat_2367_01 = _mm256_and_si256(rhs_raw_mat_2367_1, m3b); //B02(8-15) B03(8-15) B06(8-15) B07(8-15)

                    // Second sub block of the eight sub blocks processed in the iteration
                    const __m256i rhs_mat_0145_10 = _mm256_and_si256(rhs_raw_mat_0145_2, m3b); //B10(0-7) B11(0-7) B14(0-7) B15(0-7)
                    const __m256i rhs_mat_2367_10 = _mm256_and_si256(rhs_raw_mat_2367_2, m3b); //B12(0-7) B13(0-7) B16(0-7) B17(0-7)

                    const __m256i rhs_mat_0145_11 = _mm256_and_si256(rhs_raw_mat_0145_3, m3b); //B10(8-15) B11(8-15) B14(8-15) B15(8-15)
                    const __m256i rhs_mat_2367_11 = _mm256_and_si256(rhs_raw_mat_2367_3, m3b); //B12(8-15) B13(8-15) B16(8-15) B17(8-15)

                    // Third sub block of the eight sub blocks processed in the iteration
                    const __m256i rhs_mat_0145_20 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_0, 2), m3b); //B20(0-7) B21(0-7) B24(0-7) B25(0-7)
                    const __m256i rhs_mat_2367_20 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_0, 2), m3b); //B22(0-7) B23(0-7) B26(0-7) B27(0-7)

                    const __m256i rhs_mat_0145_21 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_1, 2), m3b); //B20(8-15) B21(8-15) B24(8-15) B25(8-15)
                    const __m256i rhs_mat_2367_21 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_1, 2), m3b); //B22(8-15) B23(8-15) B26(8-15) B27(8-15)

                    // Fourth sub block of the eight sub blocks processed in the iteration
                    const __m256i rhs_mat_0145_30 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_2, 2), m3b); //B30(0-7) B31(0-7) B34(0-7) B35(0-7)
                    const __m256i rhs_mat_2367_30 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_2, 2), m3b); //B32(0-7) B33(0-7) B36(0-7) B37(0-7)

                    const __m256i rhs_mat_0145_31 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_3, 2), m3b); //B30(8-15) B31(8-15) B34(8-15) B35(8-15)
                    const __m256i rhs_mat_2367_31 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_3, 2), m3b); //B32(8-15) B33(8-15) B36(8-15) B37(8-15)

                    // Fifth sub block of the eight sub blocks processed in the iteration
                    const __m256i rhs_mat_0145_40 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_0, 4), m3b); //B40(0-7) B41(0-7) B44(0-7) B45(0-7)
                    const __m256i rhs_mat_2367_40 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_0, 4), m3b); //B42(0-7) B43(0-7) B46(0-7) B47(0-7)

                    const __m256i rhs_mat_0145_41 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_1, 4), m3b); //B40(8-15) B41(8-15) B44(8-15) B45(8-15)
                    const __m256i rhs_mat_2367_41 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_1, 4), m3b); //B42(8-15) B43(8-15) B46(8-15) B47(8-15)

                    // Sixth sub block of the eight sub blocks processed in the iteration
                    const __m256i rhs_mat_0145_50 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_2, 4), m3b); //B50(0-7) B51(0-7) B54(0-7) B55(0-7)
                    const __m256i rhs_mat_2367_50 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_2, 4), m3b); //B52(0-7) B53(0-7) B56(0-7) B57(0-7)

                    const __m256i rhs_mat_0145_51 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_3, 4), m3b); //B50(8-15) B51(8-15) B54(8-15) B55(8-15)
                    const __m256i rhs_mat_2367_51 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_3, 4), m3b); //B52(8-15) B53(8-15) B56(8-15) B57(8-15)

                    // Seventh sub block of the eight sub blocks processed in the iteration
                    const __m256i rhs_mat_0145_60 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_0, 6), m3b); //B60(0-7) B61(0-7) B64(0-7) B65(0-7)
                    const __m256i rhs_mat_2367_60 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_0, 6), m3b); //B62(0-7) B63(0-7) B66(0-7) B67(0-7)

                    const __m256i rhs_mat_0145_61 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_1, 6), m3b); //B60(8-15) B61(8-15) B64(8-15) B65(8-15)
                    const __m256i rhs_mat_2367_61 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_1, 6), m3b); //B62(8-15) B63(8-15) B66(8-15) B67(8-15)

                    // Eighth sub block of the eight sub blocks processed in the iteration
                    const __m256i rhs_mat_0145_70 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_2, 6), m3b); //B70(0-7) B71(0-7) B74(0-7) B75(0-7)
                    const __m256i rhs_mat_2367_70 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_2, 6), m3b); //B72(0-7) B73(0-7) B76(0-7) B77(0-7)

                    const __m256i rhs_mat_0145_71 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_3, 6), m3b); //B70(8-15) B71(8-15) B74(8-15) B75(8-15)
                    const __m256i rhs_mat_2367_71 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_3, 6), m3b); //B72(8-15) B73(8-15) B76(8-15) B77(8-15)

                    // Shuffle pattern one - right side input
                    const __m256i rhs_mat_0145_00_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_00, 136); //B00(0-3) B01(0-3) B00(0-3) B01(0-3) B04(0-3) B05(0-3) B04(0-3) B05(0-3)
                    const __m256i rhs_mat_2367_00_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_00, 136); //B02(0-3) B03(0-3) B02(0-3) B03(0-3) B06(0-3) B07(0-3) B06(0-3) B07(0-3)

                    const __m256i rhs_mat_0145_01_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_01, 136); //B00(8-11) B01(8-11) B00(8-11) B01(8-11) B04(8-11) B05(8-11) B04(8-11) B05(8-11)
                    const __m256i rhs_mat_2367_01_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_01, 136); //B02(8-11) B03(8-11) B02(8-11) B03(8-11) B06(8-11) B07(8-11) B06(8-11) B07(8-11)

                    const __m256i rhs_mat_0145_10_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_10, 136); //B10(0-3) B11(0-3) B10(0-3) B11(0-3) B14(0-3) B15(0-3) B14(0-3) B15(0-3)
                    const __m256i rhs_mat_2367_10_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_10, 136); //B12(0-3) B13(0-3) B12(0-3) B13(0-3) B16(0-3) B17(0-3) B16(0-3) B17(0-3)

                    const __m256i rhs_mat_0145_11_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_11, 136); //B10(8-11) B11(8-11) B10(8-11) B11(8-11) B14(8-11) B15(8-11) B14(8-11) B15(8-11)
                    const __m256i rhs_mat_2367_11_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_11, 136); //B12(8-11) B13(8-11) B12(8-11) B13(8-11) B16(8-11) B17(8-11) B16(8-11) B17(8-11)

                    const __m256i rhs_mat_0145_20_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_20, 136); //B20(0-3) B21(0-3) B20(0-3) B21(0-3) B24(0-3) B25(0-3) B24(0-3) B25(0-3)
                    const __m256i rhs_mat_2367_20_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_20, 136); //B22(0-3) B23(0-3) B22(0-3) B23(0-3) B26(0-3) B27(0-3) B26(0-3) B27(0-3)

                    const __m256i rhs_mat_0145_21_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_21, 136); //B20(8-11) B21(8-11) B20(8-11) B21(8-11) B24(8-11) B25(8-11) B24(8-11) B25(8-11)
                    const __m256i rhs_mat_2367_21_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_21, 136); //B22(8-11) B23(8-11) B22(8-11) B23(8-11) B26(8-11) B27(8-11) B26(8-11) B27(8-11)

                    const __m256i rhs_mat_0145_30_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_30, 136); //B30(0-3) B31(0-3) B30(0-3) B31(0-3) B34(0-3) B35(0-3) B34(0-3) B35(0-3)
                    const __m256i rhs_mat_2367_30_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_30, 136); //B32(0-3) B33(0-3) B32(0-3) B33(0-3) B36(0-3) B37(0-3) B36(0-3) B37(0-3)

                    const __m256i rhs_mat_0145_31_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_31, 136); //B30(8-11) B31(8-11) B30(8-11) B31(8-11) B34(8-11) B35(8-11) B34(8-11) B35(8-11
                    const __m256i rhs_mat_2367_31_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_31, 136); //B32(8-11) B33(8-11) B32(8-11) B33(8-11) B36(8-11) B37(8-11) B36(8-11) B37(8-11)

                    const __m256i rhs_mat_0145_40_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_40, 136); //B40(0-3) B41(0-3) B40(0-3) B41(0-3) B44(0-3) B45(0-3) B44(0-3) B45(0-3)
                    const __m256i rhs_mat_2367_40_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_40, 136); //B42(0-3) B43(0-3) B42(0-3) B43(0-3) B46(0-3) B47(0-3) B46(0-3) B47(0-3)

                    const __m256i rhs_mat_0145_41_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_41, 136); //B40(8-11) B41(8-11) B40(8-11) B41(8-11) B44(8-11) B45(8-11) B44(8-11) B45(8-11)
                    const __m256i rhs_mat_2367_41_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_41, 136); //B42(8-11) B43(8-11) B42(8-11) B43(8-11) B46(8-11) B47(8-11) B46(8-11) B47(8-11)

                    const __m256i rhs_mat_0145_50_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_50, 136); //B50(0-3) B51(0-3) B50(0-3) B51(0-3) B54(0-3) B55(0-3) B54(0-3) B55(0-3)
                    const __m256i rhs_mat_2367_50_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_50, 136); //B52(0-3) B53(0-3) B52(0-3) B53(0-3) B56(0-3) B57(0-3) B56(0-3) B57(0-3)

                    const __m256i rhs_mat_0145_51_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_51, 136); //B50(8-11) B51(8-11) B50(8-11) B51(8-11) B54(8-11) B55(8-11) B54(8-11) B55(8-11)
                    const __m256i rhs_mat_2367_51_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_51, 136); //B52(8-11) B53(8-11) B52(8-11) B53(8-11) B56(8-11) B57(8-11) B56(8-11) B57(8-11)

                    const __m256i rhs_mat_0145_60_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_60, 136); //B60(0-3) B61(0-3) B60(0-3) B61(0-3) B64(0-3) B65(0-3) B64(0-3) B65(0-3)
                    const __m256i rhs_mat_2367_60_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_60, 136); //B62(0-3) B63(0-3) B62(0-3) B63(0-3) B66(0-3) B67(0-3) B66(0-3) B67(0-3)

                    const __m256i rhs_mat_0145_61_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_61, 136); //B60(8-11) B61(8-11) B60(8-11) B61(8-11) B64(8-11) B65(8-11) B64(8-11) B65(8-11)
                    const __m256i rhs_mat_2367_61_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_61, 136); //B62(8-11) B63(8-11) B62(8-11) B63(8-11) B66(8-11) B67(8-11) B66(8-11) B67(8-11)

                    const __m256i rhs_mat_0145_70_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_70, 136); //B70(0-3) B71(0-3) B70(0-3) B71(0-3) B74(0-3) B75(0-3) B74(0-3) B75(0-3)
                    const __m256i rhs_mat_2367_70_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_70, 136); //B72(0-3) B73(0-3) B72(0-3) B73(0-3) B76(0-3) B77(0-3) B76(0-3) B77(0-3)

                    const __m256i rhs_mat_0145_71_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_71, 136); //B70(8-11) B71(8-11) B70(8-11) B71(8-11) B74(8-11) B75(8-11) B74(8-11) B75(8-11)
                    const __m256i rhs_mat_2367_71_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_71, 136); //B72(8-11) B73(8-11) B72(8-11) B73(8-11) B76(8-11) B77(8-11) B76(8-11) B77(8-11)


                    // Shuffle pattern two - right side input
                    const __m256i rhs_mat_0145_00_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_00, 221); //B00(4-7) B01(4-7) B00(4-7) B01(4-7) B04(4-7) B05(4-7) B04(4-7) B05(4-7)
                    const __m256i rhs_mat_2367_00_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_00, 221); //B02(4-7) B03(4-7) B02(4-7) B03(4-7) B06(4-7) B07(4-7) B06(4-7) B07(4-7)

                    const __m256i rhs_mat_0145_01_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_01, 221); //B00(12-15) B01(12-15) B00(12-15) B01(12-15) B04(12-15) B05(12-15) B04(12-15) B05(12-15)
                    const __m256i rhs_mat_2367_01_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_01, 221); //B02(12-15) B03(12-15) B02(12-15) B03(12-15) B06(12-15) B07(12-15) B06(12-15) B07(12-15)

                    const __m256i rhs_mat_0145_10_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_10, 221); //B10(4-7) B11(4-7) B10(4-7) B11(4-7) B14(4-7) B15(4-7) B14(4-7) B15(4-7)
                    const __m256i rhs_mat_2367_10_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_10, 221); //B12(4-7) B13(4-7) B12(4-7) B13(4-7) B16(4-7) B17(4-7) B16(4-7) B17(4-7)

                    const __m256i rhs_mat_0145_11_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_11, 221); //B10(12-15) B11(12-15) B10(12-15) B11(12-15) B14(12-15) B15(12-15) B14(12-15) B15(12-15)
                    const __m256i rhs_mat_2367_11_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_11, 221); //B12(12-15) B13(12-15) B12(12-15) B13(12-15) B16(12-15) B17(12-15) B16(12-15) B17(12-15)

                    const __m256i rhs_mat_0145_20_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_20, 221); //B20(4-7) B21(4-7) B20(4-7) B21(4-7) B24(4-7) B25(4-7) B24(4-7) B25(4-7)
                    const __m256i rhs_mat_2367_20_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_20, 221); //B22(4-7) B23(4-7) B22(4-7) B23(4-7) B26(4-7) B27(4-7) B26(4-7) B27(4-7)

                    const __m256i rhs_mat_0145_21_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_21, 221); //B20(12-15) B21(12-15) B20(12-15) B21(12-15) B24(12-15) B25(12-15) B24(12-15) B25(12-15)
                    const __m256i rhs_mat_2367_21_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_21, 221); //B22(12-15) B23(12-15) B22(12-15) B23(12-15) B26(12-15) B27(12-15) B26(12-15) B27(12-15)

                    const __m256i rhs_mat_0145_30_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_30, 221); //B30(4-7) B31(4-7) B30(4-7) B31(4-7) B34(4-7) B35(4-7) B34(4-7) B35(4-7)
                    const __m256i rhs_mat_2367_30_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_30, 221); //B32(4-7) B33(4-7) B32(4-7) B33(4-7) B36(4-7) B37(4-7) B36(4-7) B37(4-7)

                    const __m256i rhs_mat_0145_31_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_31, 221); //B30(12-15) B31(12-15) B30(12-15) B31(12-15) B34(12-15) B35(12-15) B34(12-15) B35(12-15)
                    const __m256i rhs_mat_2367_31_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_31, 221); //B32(12-15) B33(12-15) B32(12-15) B33(12-15) B36(12-15) B37(12-15) B36(12-15) B37(12-15)

                    const __m256i rhs_mat_0145_40_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_40, 221); //B40(4-7) B41(4-7) B40(4-7) B41(4-7) B44(4-7) B45(4-7) B44(4-7) B45(4-7)
                    const __m256i rhs_mat_2367_40_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_40, 221); //B42(4-7) B43(4-7) B42(4-7) B43(4-7) B46(4-7) B47(4-7) B46(4-7) B47(4-7)

                    const __m256i rhs_mat_0145_41_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_41, 221); //B40(12-15) B41(12-15) B40(12-15) B41(12-15) B44(12-15) B45(12-15) B44(12-15) B45(12-15)
                    const __m256i rhs_mat_2367_41_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_41, 221); //B42(12-15) B43(12-15) B42(12-15) B43(12-15) B46(12-15) B47(12-15) B46(12-15) B47(12-15)

                    const __m256i rhs_mat_0145_50_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_50, 221); //B50(4-7) B51(4-7) B50(4-7) B51(4-7) B54(4-7) B55(4-7) B54(4-7) B55(4-7)
                    const __m256i rhs_mat_2367_50_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_50, 221); //B52(4-7) B53(4-7) B52(4-7) B53(4-7) B56(4-7) B57(4-7) B56(4-7) B57(4-7)

                    const __m256i rhs_mat_0145_51_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_51, 221); //B50(12-15) B51(12-15) B50(12-15) B51(12-15) B54(12-15) B55(12-15) B54(12-15) B55(12-15)
                    const __m256i rhs_mat_2367_51_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_51, 221); //B52(12-15) B53(12-15) B52(12-15) B53(12-15) B56(12-15) B57(12-15) B56(12-15) B57(12-15)

                    const __m256i rhs_mat_0145_60_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_60, 221); //B60(4-7) B61(4-7) B60(4-7) B61(4-7) B64(4-7) B65(4-7) B64(4-7) B65(4-7)
                    const __m256i rhs_mat_2367_60_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_60, 221); //B62(4-7) B63(4-7) B62(4-7) B63(4-7) B66(4-7) B67(4-7) B66(4-7) B67(4-7)

                    const __m256i rhs_mat_0145_61_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_61, 221); //B60(12-15) B61(12-15) B60(12-15) B61(12-15) B64(12-15) B65(12-15) B64(12-15) B65(12-15)
                    const __m256i rhs_mat_2367_61_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_61, 221); //B62(12-15) B63(12-15) B62(12-15) B63(12-15) B66(12-15) B67(12-15) B66(12-15) B67(12-15)

                    const __m256i rhs_mat_0145_70_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_70, 221); //B70(4-7) B71(4-7) B70(4-7) B71(4-7) B74(4-7) B75(4-7) B74(4-7) B75(4-7)
                    const __m256i rhs_mat_2367_70_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_70, 221); //B72(4-7) B73(4-7) B72(4-7) B73(4-7) B76(4-7) B77(4-7) B76(4-7) B77(4-7)

                    const __m256i rhs_mat_0145_71_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_71, 221); //B70(12-15) B71(12-15) B70(12-15) B71(12-15) B74(12-15) B75(12-15) B74(12-15) B75(12-15)
                    const __m256i rhs_mat_2367_71_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_71, 221); //B72(12-15) B73(12-15) B72(12-15) B73(12-15) B76(12-15) B77(12-15) B76(12-15) B77(12-15)

                    //Scales and Mins of corresponding sub blocks from different Q2_K structures are stored together
                    //s00 m00  s01 m01   s10 m10  s11 m11  s20 m20  s21 m21   s30 m30  s31 m31  s40 m40  s41 m41   s50 m50  s51 m51  s60 m60  s61 m61   s70 m70  s71 m71

                    // Combine mins and scales for sub-blocks: 0-1, 2-3, 4-5, 6-7 in the sb loop
                    const __m128i mins_and_scales_01 = _mm_loadu_si128((const __m128i *)(b_ptr[b].scales + sb * 64));
                    const __m128i mins_and_scales_23 = _mm_loadu_si128((const __m128i *)(b_ptr[b].scales + 16 + sb * 64));
                    const __m128i mins_and_scales_45 = _mm_loadu_si128((const __m128i *)(b_ptr[b].scales + 32 + sb * 64));
                    const __m128i mins_and_scales_67 = _mm_loadu_si128((const __m128i *)(b_ptr[b].scales + 48 + sb * 64));

                    // Extract scales which is lower half from mins_and_scales
                    const __m128i scales_01 = _mm_and_si128(mins_and_scales_01, m4b_sse);
                    const __m128i scales_23 = _mm_and_si128(mins_and_scales_23, m4b_sse);
                    const __m128i scales_45 = _mm_and_si128(mins_and_scales_45, m4b_sse);
                    const __m128i scales_67 = _mm_and_si128(mins_and_scales_67, m4b_sse);

                    // Extract mins which is upper half from mins_and_scales
                    const __m256i mins_01 = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(mins_and_scales_01, 4), m4b_sse));
                    const __m256i mins_23 = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(mins_and_scales_23, 4), m4b_sse));
                    const __m256i mins_45 = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(mins_and_scales_45, 4), m4b_sse));
                    const __m256i mins_67 = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(mins_and_scales_67, 4), m4b_sse));

                    const __m256i scales_0 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(scales_01, scalesmask1_sse));
                    const __m256i scales_1 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(scales_01, scalesmask2_sse));

                    const __m256i scales_2 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(scales_23, scalesmask1_sse));
                    const __m256i scales_3 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(scales_23, scalesmask2_sse));

                    const __m256i scales_4 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(scales_45, scalesmask1_sse));
                    const __m256i scales_5 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(scales_45, scalesmask2_sse));

                    const __m256i scales_6 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(scales_67, scalesmask1_sse));
                    const __m256i scales_7 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(scales_67, scalesmask2_sse));

                    const __m256i scale_0145_0 = _mm256_shuffle_epi32(scales_0, 68);
                    const __m256i scale_2367_0 = _mm256_shuffle_epi32(scales_0, 238);

                    const __m256i scale_0145_1 = _mm256_shuffle_epi32(scales_1, 68);
                    const __m256i scale_2367_1 = _mm256_shuffle_epi32(scales_1, 238);

                    const __m256i scale_0145_2 = _mm256_shuffle_epi32(scales_2, 68);
                    const __m256i scale_2367_2 = _mm256_shuffle_epi32(scales_2, 238);

                    const __m256i scale_0145_3 = _mm256_shuffle_epi32(scales_3, 68);
                    const __m256i scale_2367_3 = _mm256_shuffle_epi32(scales_3, 238);

                    const __m256i scale_0145_4 = _mm256_shuffle_epi32(scales_4, 68);
                    const __m256i scale_2367_4 = _mm256_shuffle_epi32(scales_4, 238);

                    const __m256i scale_0145_5 = _mm256_shuffle_epi32(scales_5, 68);
                    const __m256i scale_2367_5 = _mm256_shuffle_epi32(scales_5, 238);

                    const __m256i scale_0145_6 = _mm256_shuffle_epi32(scales_6, 68);
                    const __m256i scale_2367_6 = _mm256_shuffle_epi32(scales_6, 238);

                    const __m256i scale_0145_7 = _mm256_shuffle_epi32(scales_7, 68);
                    const __m256i scale_2367_7 = _mm256_shuffle_epi32(scales_7, 238);


                    for (int rp = 0; rp < 4; rp++) {

                        // Load the four block_q8_k quantized values interleaved with each other in chunks of eight bytes - A0,A1,A2,A3
                        // Loaded as set of 128 bit vectors and repeated into a 256 bit vector
                        __m256i lhs_mat_0123_00 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 512 * sb)));
                        __m256i lhs_mat_01_00 = _mm256_permute2f128_si256(lhs_mat_0123_00, lhs_mat_0123_00, 0);
                        __m256i lhs_mat_23_00 = _mm256_permute2f128_si256(lhs_mat_0123_00, lhs_mat_0123_00, 17);
                        __m256i lhs_mat_0123_01 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 32 + 512 * sb)));
                        __m256i lhs_mat_01_01 = _mm256_permute2f128_si256(lhs_mat_0123_01, lhs_mat_0123_01, 0);
                        __m256i lhs_mat_23_01 = _mm256_permute2f128_si256(lhs_mat_0123_01, lhs_mat_0123_01, 17);
                        __m256i lhs_mat_0123_10 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 64 + 512 * sb)));
                        __m256i lhs_mat_01_10 = _mm256_permute2f128_si256(lhs_mat_0123_10, lhs_mat_0123_10, 0);
                        __m256i lhs_mat_23_10 = _mm256_permute2f128_si256(lhs_mat_0123_10, lhs_mat_0123_10, 17);
                        __m256i lhs_mat_0123_11 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 96 + 512 * sb)));
                        __m256i lhs_mat_01_11 = _mm256_permute2f128_si256(lhs_mat_0123_11, lhs_mat_0123_11, 0);
                        __m256i lhs_mat_23_11 = _mm256_permute2f128_si256(lhs_mat_0123_11, lhs_mat_0123_11, 17);
                        __m256i lhs_mat_0123_20 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 128 + 512 * sb)));
                        __m256i lhs_mat_01_20 = _mm256_permute2f128_si256(lhs_mat_0123_20, lhs_mat_0123_20, 0);
                        __m256i lhs_mat_23_20 = _mm256_permute2f128_si256(lhs_mat_0123_20, lhs_mat_0123_20, 17);
                        __m256i lhs_mat_0123_21 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 160 + 512 * sb)));
                        __m256i lhs_mat_01_21 = _mm256_permute2f128_si256(lhs_mat_0123_21, lhs_mat_0123_21, 0);
                        __m256i lhs_mat_23_21 = _mm256_permute2f128_si256(lhs_mat_0123_21, lhs_mat_0123_21, 17);
                        __m256i lhs_mat_0123_30 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 192 + 512 * sb)));
                        __m256i lhs_mat_01_30 = _mm256_permute2f128_si256(lhs_mat_0123_30, lhs_mat_0123_30, 0);
                        __m256i lhs_mat_23_30 = _mm256_permute2f128_si256(lhs_mat_0123_30, lhs_mat_0123_30, 17);
                        __m256i lhs_mat_0123_31 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 224 + 512 * sb)));
                        __m256i lhs_mat_01_31 = _mm256_permute2f128_si256(lhs_mat_0123_31, lhs_mat_0123_31, 0);
                        __m256i lhs_mat_23_31 = _mm256_permute2f128_si256(lhs_mat_0123_31, lhs_mat_0123_31, 17);

                        __m256i lhs_mat_0123_40 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 256 + 512 * sb)));
                        __m256i lhs_mat_01_40 = _mm256_permute2f128_si256(lhs_mat_0123_40, lhs_mat_0123_40, 0);
                        __m256i lhs_mat_23_40 = _mm256_permute2f128_si256(lhs_mat_0123_40, lhs_mat_0123_40, 17);
                        __m256i lhs_mat_0123_41 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 288 + 512 * sb)));
                        __m256i lhs_mat_01_41 = _mm256_permute2f128_si256(lhs_mat_0123_41, lhs_mat_0123_41, 0);
                        __m256i lhs_mat_23_41 = _mm256_permute2f128_si256(lhs_mat_0123_41, lhs_mat_0123_41, 17);
                        __m256i lhs_mat_0123_50 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 320 + 512 * sb)));
                        __m256i lhs_mat_01_50 = _mm256_permute2f128_si256(lhs_mat_0123_50, lhs_mat_0123_50, 0);
                        __m256i lhs_mat_23_50 = _mm256_permute2f128_si256(lhs_mat_0123_50, lhs_mat_0123_50, 17);
                        __m256i lhs_mat_0123_51 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 352 + 512 * sb)));
                        __m256i lhs_mat_01_51 = _mm256_permute2f128_si256(lhs_mat_0123_51, lhs_mat_0123_51, 0);
                        __m256i lhs_mat_23_51 = _mm256_permute2f128_si256(lhs_mat_0123_51, lhs_mat_0123_51, 17);
                        __m256i lhs_mat_0123_60 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 384 + 512 * sb)));
                        __m256i lhs_mat_01_60 = _mm256_permute2f128_si256(lhs_mat_0123_60, lhs_mat_0123_60, 0);
                        __m256i lhs_mat_23_60 = _mm256_permute2f128_si256(lhs_mat_0123_60, lhs_mat_0123_60, 17);
                        __m256i lhs_mat_0123_61 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 416 + 512 * sb)));
                        __m256i lhs_mat_01_61 = _mm256_permute2f128_si256(lhs_mat_0123_61, lhs_mat_0123_61, 0);
                        __m256i lhs_mat_23_61 = _mm256_permute2f128_si256(lhs_mat_0123_61, lhs_mat_0123_61, 17);
                        __m256i lhs_mat_0123_70 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 448 + 512 * sb)));
                        __m256i lhs_mat_01_70 = _mm256_permute2f128_si256(lhs_mat_0123_70, lhs_mat_0123_70, 0);
                        __m256i lhs_mat_23_70 = _mm256_permute2f128_si256(lhs_mat_0123_70, lhs_mat_0123_70, 17);
                        __m256i lhs_mat_0123_71 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 480 + 512 * sb)));
                        __m256i lhs_mat_01_71 = _mm256_permute2f128_si256(lhs_mat_0123_71, lhs_mat_0123_71, 0);
                        __m256i lhs_mat_23_71 = _mm256_permute2f128_si256(lhs_mat_0123_71, lhs_mat_0123_71, 17);

                        // Bsums are loaded for the different Q8_K blocks
                        __m128i lhs_raw_bsums_01_0123 = _mm_loadu_si128((const __m128i *)((a_ptrs[rp][b].bsums + 32 * sb)));
                        __m128i lhs_raw_bsums_23_0123 = _mm_loadu_si128((const __m128i *)(a_ptrs[rp][b].bsums + 8 + 32 * sb));
                        __m128i lhs_raw_bsums_01_4567 = _mm_loadu_si128((const __m128i *)((a_ptrs[rp][b].bsums + 16 + 32 * sb)));
                        __m128i lhs_raw_bsums_23_4567 = _mm_loadu_si128((const __m128i *)(a_ptrs[rp][b].bsums + 24 + 32 * sb));

                        // Shuffle pattern one - left side input
                        const __m256i lhs_mat_01_00_sp1 = _mm256_shuffle_epi32(lhs_mat_01_00, 160); //A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3) A00(0-3) A01(0-3) A01(0-3)
                        const __m256i lhs_mat_23_00_sp1 = _mm256_shuffle_epi32(lhs_mat_23_00, 160); //A02(0-3) A03(0-3) A02(0-3) A03(0-3) A02(0-3) A03(0-3) A02(0-3) A03(0-3)

                        const __m256i lhs_mat_01_01_sp1 = _mm256_shuffle_epi32(lhs_mat_01_01, 160); //A00(8-11) A00(8-11) A01(8-11) A01(8-11) A00(8-11) A00(8-11) A01(8-11) A01(8-11)
                        const __m256i lhs_mat_23_01_sp1 = _mm256_shuffle_epi32(lhs_mat_23_01, 160); //A02(8-11) A03(8-11) A02(8-11) A03(8-11) A02(8-11) A03(8-11) A02(8-11) A03(8-11)

                        const __m256i lhs_mat_01_10_sp1 = _mm256_shuffle_epi32(lhs_mat_01_10, 160); //A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3) A10(0-3) A11(0-3) A11(0-3)
                        const __m256i lhs_mat_23_10_sp1 = _mm256_shuffle_epi32(lhs_mat_23_10, 160); //A12(0-3) A13(0-3) A12(0-3) A13(0-3) A12(0-3) A13(0-3) A12(0-3) A13(0-3)

                        const __m256i lhs_mat_01_11_sp1 = _mm256_shuffle_epi32(lhs_mat_01_11, 160); //A10(8-11) A10(8-11) A11(8-11) A11(8-11) A10(8-11) A10(8-11) A11(8-11) A11(8-11)
                        const __m256i lhs_mat_23_11_sp1 = _mm256_shuffle_epi32(lhs_mat_23_11, 160); //A12(8-11) A13(8-11) A12(8-11) A13(8-11) A12(8-11) A13(8-11) A12(8-11) A13(8-11)

                        const __m256i lhs_mat_01_20_sp1 = _mm256_shuffle_epi32(lhs_mat_01_20, 160); //A20(0-3) A20(0-3) A21(0-3) A21(0-3) A20(0-3) A20(0-3) A21(0-3) A21(0-3)
                        const __m256i lhs_mat_23_20_sp1 = _mm256_shuffle_epi32(lhs_mat_23_20, 160); //A22(0-3) A23(0-3) A22(0-3) A23(0-3) A22(0-3) A23(0-3) A22(0-3) A23(0-3)

                        const __m256i lhs_mat_01_21_sp1 = _mm256_shuffle_epi32(lhs_mat_01_21, 160); //A20(8-11) A20(8-11) A21(8-11) A21(8-11) A20(8-11) A20(8-11) A21(8-11) A21(8-11)
                        const __m256i lhs_mat_23_21_sp1 = _mm256_shuffle_epi32(lhs_mat_23_21, 160); //A22(8-11) A23(8-11) A22(8-11) A23(8-11) A22(8-11) A23(8-11) A22(8-11) A23(8-11)

                        const __m256i lhs_mat_01_30_sp1 = _mm256_shuffle_epi32(lhs_mat_01_30, 160); //A30(0-3) A30(0-3) A31(0-3) A31(0-3) A30(0-3) A30(0-3) A31(0-3) A31(0-3)
                        const __m256i lhs_mat_23_30_sp1 = _mm256_shuffle_epi32(lhs_mat_23_30, 160); //A32(0-3) A33(0-3) A32(0-3) A33(0-3) A32(0-3) A33(0-3) A32(0-3) A33(0-3)

                        const __m256i lhs_mat_01_31_sp1 = _mm256_shuffle_epi32(lhs_mat_01_31, 160); //A30(8-11) A30(8-11) A31(8-11) A31(8-11) A30(8-11) A30(8-11) A31(8-11) A31(8-11)
                        const __m256i lhs_mat_23_31_sp1 = _mm256_shuffle_epi32(lhs_mat_23_31, 160); //A32(8-11) A33(8-11) A32(8-11) A33(8-11) A32(8-11) A33(8-11) A32(8-11) A33(8-11)

                        const __m256i lhs_mat_01_40_sp1 = _mm256_shuffle_epi32(lhs_mat_01_40, 160); //A40(0-3) A40(0-3) A41(0-3) A41(0-3) A40(0-3) A40(0-3) A41(0-3) A41(0-3)
                        const __m256i lhs_mat_23_40_sp1 = _mm256_shuffle_epi32(lhs_mat_23_40, 160); //A42(0-3) A43(0-3) A42(0-3) A43(0-3) A42(0-3) A43(0-3) A42(0-3) A43(0-3)

                        const __m256i lhs_mat_01_41_sp1 = _mm256_shuffle_epi32(lhs_mat_01_41, 160); //A40(8-11) A40(8-11) A41(8-11) A41(8-11) A40(8-11) A40(8-11) A41(8-11) A41(8-11)
                        const __m256i lhs_mat_23_41_sp1 = _mm256_shuffle_epi32(lhs_mat_23_41, 160); //A42(8-11) A43(8-11) A42(8-11) A43(8-11) A42(8-11) A43(8-11) A42(8-11) A43(8-11)

                        const __m256i lhs_mat_01_50_sp1 = _mm256_shuffle_epi32(lhs_mat_01_50, 160); //A50(0-3) A50(0-3) A51(0-3) A51(0-3) A50(0-3) A50(0-3) A51(0-3) A51(0-3)
                        const __m256i lhs_mat_23_50_sp1 = _mm256_shuffle_epi32(lhs_mat_23_50, 160); //A52(0-3) A53(0-3) A52(0-3) A53(0-3) A52(0-3) A53(0-3) A52(0-3) A53(0-3)

                        const __m256i lhs_mat_01_51_sp1 = _mm256_shuffle_epi32(lhs_mat_01_51, 160); //A50(8-11) A50(8-11) A51(8-11) A51(8-11) A50(8-11) A50(8-11) A51(8-11) A51(8-11)
                        const __m256i lhs_mat_23_51_sp1 = _mm256_shuffle_epi32(lhs_mat_23_51, 160); //A52(8-11) A53(8-11) A52(8-11) A53(8-11) A52(8-11) A53(8-11) A52(8-11) A53(8-11)

                        const __m256i lhs_mat_01_60_sp1 = _mm256_shuffle_epi32(lhs_mat_01_60, 160); //A60(0-3) A60(0-3) A61(0-3) A61(0-3) A60(0-3) A60(0-3) A61(0-3) A61(0-3)
                        const __m256i lhs_mat_23_60_sp1 = _mm256_shuffle_epi32(lhs_mat_23_60, 160); //A62(0-3) A63(0-3) A62(0-3) A63(0-3) A62(0-3) A63(0-3) A62(0-3) A63(0-3)

                        const __m256i lhs_mat_01_61_sp1 = _mm256_shuffle_epi32(lhs_mat_01_61, 160); //A60(8-11) A60(8-11) A61(8-11) A61(8-11) A60(8-11) A60(8-11) A61(8-11) A61(8-11)
                        const __m256i lhs_mat_23_61_sp1 = _mm256_shuffle_epi32(lhs_mat_23_61, 160); //A62(8-11) A63(8-11) A62(8-11) A63(8-11) A62(8-11) A63(8-11) A62(8-11) A63(8-11)

                        const __m256i lhs_mat_01_70_sp1 = _mm256_shuffle_epi32(lhs_mat_01_70, 160); //A70(0-3) A70(0-3) A71(0-3) A71(0-3) A70(0-3) A70(0-3) A71(0-3) A71(0-3)
                        const __m256i lhs_mat_23_70_sp1 = _mm256_shuffle_epi32(lhs_mat_23_70, 160); //A72(0-3) A73(0-3) A72(0-3) A73(0-3) A72(0-3) A73(0-3) A72(0-3) A73(0-3)

                        const __m256i lhs_mat_01_71_sp1 = _mm256_shuffle_epi32(lhs_mat_01_71, 160); //A70(8-11) A70(8-11) A71(8-11) A71(8-11) A70(8-11) A70(8-11) A71(8-11) A71(8-11)
                        const __m256i lhs_mat_23_71_sp1 = _mm256_shuffle_epi32(lhs_mat_23_71, 160); //A72(8-11) A73(8-11) A72(8-11) A73(8-11) A72(8-11) A73(8-11) A72(8-11) A73(8-11)

                        // Shuffle pattern two- left side input
                        const __m256i lhs_mat_01_00_sp2 = _mm256_shuffle_epi32(lhs_mat_01_00, 245); //A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7) A00(4-7) A01(4-7) A01(4-7)
                        const __m256i lhs_mat_23_00_sp2 = _mm256_shuffle_epi32(lhs_mat_23_00, 245); //A02(4-7) A03(4-7) A02(4-7) A03(4-7) A02(4-7) A03(4-7) A02(4-7) A03(4-7)

                        const __m256i lhs_mat_01_01_sp2 = _mm256_shuffle_epi32(lhs_mat_01_01, 245); //A00(12-15) A00(12-15) A01(12-15) A01(12-15) A00(12-15) A00(12-15) A01(12-15) A01(12-15)
                        const __m256i lhs_mat_23_01_sp2 = _mm256_shuffle_epi32(lhs_mat_23_01, 245); //A02(12-15) A03(12-15) A02(12-15) A03(12-15) A02(12-15) A03(12-15) A02(12-15) A03(12-15)

                        const __m256i lhs_mat_01_10_sp2 = _mm256_shuffle_epi32(lhs_mat_01_10, 245); //A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7) A10(4-7) A11(4-7) A11(4-7)
                        const __m256i lhs_mat_23_10_sp2 = _mm256_shuffle_epi32(lhs_mat_23_10, 245); //A12(4-7) A13(4-7) A12(4-7) A13(4-7) A12(4-7) A13(4-7) A12(4-7) A13(4-7)

                        const __m256i lhs_mat_01_11_sp2 = _mm256_shuffle_epi32(lhs_mat_01_11, 245); //A10(12-15) A10(12-15) A11(12-15) A11(12-15) A10(12-15) A10(12-15) A11(12-15) A11(12-15)
                        const __m256i lhs_mat_23_11_sp2 = _mm256_shuffle_epi32(lhs_mat_23_11, 245); //A12(12-15) A13(12-15) A12(12-15) A13(12-15) A12(12-15) A13(12-15) A12(12-15) A13(12-15)

                        const __m256i lhs_mat_01_20_sp2 = _mm256_shuffle_epi32(lhs_mat_01_20, 245); //A20(4-7) A20(4-7) A21(4-7) A21(4-7) A20(4-7) A20(4-7) A21(4-7) A21(4-7)
                        const __m256i lhs_mat_23_20_sp2 = _mm256_shuffle_epi32(lhs_mat_23_20, 245); //A22(4-7) A23(4-7) A22(4-7) A23(4-7) A22(4-7) A23(4-7) A22(4-7) A23(4-7)

                        const __m256i lhs_mat_01_21_sp2 = _mm256_shuffle_epi32(lhs_mat_01_21, 245); //A20(12-15) A20(12-15) A21(12-15) A21(12-15) A20(12-15) A20(12-15) A21(12-15) A21(12-15)
                        const __m256i lhs_mat_23_21_sp2 = _mm256_shuffle_epi32(lhs_mat_23_21, 245); //A22(12-15) A23(12-15) A22(12-15) A23(12-15) A22(12-15) A23(12-15) A22(12-15) A23(12-15)

                        const __m256i lhs_mat_01_30_sp2 = _mm256_shuffle_epi32(lhs_mat_01_30, 245); //A30(4-7) A30(4-7) A31(4-7) A31(4-7) A30(4-7) A30(4-7) A31(4-7) A31(4-7)
                        const __m256i lhs_mat_23_30_sp2 = _mm256_shuffle_epi32(lhs_mat_23_30, 245); //A32(4-7) A33(4-7) A32(4-7) A33(4-7) A32(4-7) A33(4-7) A32(4-7) A33(4-7)

                        const __m256i lhs_mat_01_31_sp2 = _mm256_shuffle_epi32(lhs_mat_01_31, 245); //A30(12-15) A30(12-15) A31(12-15) A31(12-15) A30(12-15) A30(12-15) A31(12-15) A31(12-15)
                        const __m256i lhs_mat_23_31_sp2 = _mm256_shuffle_epi32(lhs_mat_23_31, 245); //A32(12-15) A33(12-15) A32(12-15) A33(12-15) A32(12-15) A33(12-15) A32(12-15) A33(12-15)

                        const __m256i lhs_mat_01_40_sp2 = _mm256_shuffle_epi32(lhs_mat_01_40, 245); //A40(4-7) A40(4-7) A41(4-7) A41(4-7) A40(4-7) A40(4-7) A41(4-7) A41(4-7)
                        const __m256i lhs_mat_23_40_sp2 = _mm256_shuffle_epi32(lhs_mat_23_40, 245); //A42(4-7) A43(4-7) A42(4-7) A43(4-7) A42(4-7) A43(4-7) A42(4-7) A43(4-7)

                        const __m256i lhs_mat_01_41_sp2 = _mm256_shuffle_epi32(lhs_mat_01_41, 245); //A40(12-15) A40(12-15) A41(12-15) A41(12-15) A40(12-15) A40(12-15) A41(12-15) A41(12-15)
                        const __m256i lhs_mat_23_41_sp2 = _mm256_shuffle_epi32(lhs_mat_23_41, 245); //A42(12-15) A43(12-15) A42(12-15) A43(12-15) A42(12-15) A43(12-15) A42(12-15) A43(12-15)

                        const __m256i lhs_mat_01_50_sp2 = _mm256_shuffle_epi32(lhs_mat_01_50, 245); //A50(4-7) A50(4-7) A51(4-7) A51(4-7) A50(4-7) A50(4-7) A51(4-7) A51(4-7)
                        const __m256i lhs_mat_23_50_sp2 = _mm256_shuffle_epi32(lhs_mat_23_50, 245); //A52(4-7) A53(4-7) A52(4-7) A53(4-7) A52(4-7) A53(4-7) A52(4-7) A53(4-7)

                        const __m256i lhs_mat_01_51_sp2 = _mm256_shuffle_epi32(lhs_mat_01_51, 245); //A50(12-15) A50(12-15) A51(12-15) A51(12-15) A50(12-15) A50(12-15) A51(12-15) A51(12-15)
                        const __m256i lhs_mat_23_51_sp2 = _mm256_shuffle_epi32(lhs_mat_23_51, 245); //A52(12-15) A53(12-15) A52(12-15) A53(12-15) A52(12-15) A53(12-15) A52(12-15) A53(12-15)

                        const __m256i lhs_mat_01_60_sp2 = _mm256_shuffle_epi32(lhs_mat_01_60, 245); //A60(4-7) A60(4-7) A61(4-7) A61(4-7) A60(4-7) A60(4-7) A61(4-7) A61(4-7)
                        const __m256i lhs_mat_23_60_sp2 = _mm256_shuffle_epi32(lhs_mat_23_60, 245); //A62(4-7) A63(4-7) A62(4-7) A63(4-7) A62(4-7) A63(4-7) A62(4-7) A63(4-7)

                        const __m256i lhs_mat_01_61_sp2 = _mm256_shuffle_epi32(lhs_mat_01_61, 245); //A60(12-15) A60(12-15) A61(12-15) A61(12-15) A60(12-15) A60(12-15) A61(12-15) A61(12-15)
                        const __m256i lhs_mat_23_61_sp2 = _mm256_shuffle_epi32(lhs_mat_23_61, 245); //A62(12-15) A63(12-15) A62(12-15) A63(12-15) A62(12-15) A63(12-15) A62(12-15) A63(12-15)

                        const __m256i lhs_mat_01_70_sp2 = _mm256_shuffle_epi32(lhs_mat_01_70, 245); //A70(4-7) A70(4-7) A71(4-7) A71(4-7) A70(4-7) A70(4-7) A71(4-7) A71(4-7)
                        const __m256i lhs_mat_23_70_sp2 = _mm256_shuffle_epi32(lhs_mat_23_70, 245); //A72(4-7) A73(4-7) A72(4-7) A73(4-7) A72(4-7) A73(4-7) A72(4-7) A73(4-7)

                        const __m256i lhs_mat_01_71_sp2 = _mm256_shuffle_epi32(lhs_mat_01_71, 245); //A70(12-15) A70(12-15) A71(12-15) A71(12-15) A70(12-15) A70(12-15) A71(12-15) A71(12-15)
                        const __m256i lhs_mat_23_71_sp2 = _mm256_shuffle_epi32(lhs_mat_23_71, 245); //A72(12-15) A73(12-15) A72(12-15) A73(12-15) A72(12-15) A73(12-15) A72(12-15) A73(12-15)

                        // The values arranged in shuffle patterns are operated with dot product operation within 32 bit lane i.e corresponding bytes and multiplied and added into 32 bit integers within 32 bit lane
                        __m256i iacc_mat_00_0_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_00_sp1, lhs_mat_01_00_sp1),_mm256_maddubs_epi16(rhs_mat_0145_01_sp1, lhs_mat_01_01_sp1));
                        __m256i iacc_mat_01_0_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_00_sp1, lhs_mat_01_00_sp1),_mm256_maddubs_epi16(rhs_mat_2367_01_sp1, lhs_mat_01_01_sp1));

                        __m256i iacc_mat_10_0_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_00_sp1, lhs_mat_23_00_sp1),_mm256_maddubs_epi16(rhs_mat_0145_01_sp1, lhs_mat_23_01_sp1));
                        __m256i iacc_mat_11_0_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_00_sp1, lhs_mat_23_00_sp1),_mm256_maddubs_epi16(rhs_mat_2367_01_sp1, lhs_mat_23_01_sp1));

                        __m256i iacc_mat_00_1_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_10_sp1, lhs_mat_01_10_sp1),_mm256_maddubs_epi16(rhs_mat_0145_11_sp1, lhs_mat_01_11_sp1));
                        __m256i iacc_mat_01_1_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_10_sp1, lhs_mat_01_10_sp1),_mm256_maddubs_epi16(rhs_mat_2367_11_sp1, lhs_mat_01_11_sp1));

                        __m256i iacc_mat_10_1_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_10_sp1, lhs_mat_23_10_sp1),_mm256_maddubs_epi16(rhs_mat_0145_11_sp1, lhs_mat_23_11_sp1));
                        __m256i iacc_mat_11_1_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_10_sp1, lhs_mat_23_10_sp1),_mm256_maddubs_epi16(rhs_mat_2367_11_sp1, lhs_mat_23_11_sp1));

                        __m256i iacc_mat_00_2_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_20_sp1, lhs_mat_01_20_sp1),_mm256_maddubs_epi16(rhs_mat_0145_21_sp1, lhs_mat_01_21_sp1));
                        __m256i iacc_mat_01_2_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_20_sp1, lhs_mat_01_20_sp1),_mm256_maddubs_epi16(rhs_mat_2367_21_sp1, lhs_mat_01_21_sp1));

                        __m256i iacc_mat_10_2_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_20_sp1, lhs_mat_23_20_sp1),_mm256_maddubs_epi16(rhs_mat_0145_21_sp1, lhs_mat_23_21_sp1));
                        __m256i iacc_mat_11_2_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_20_sp1, lhs_mat_23_20_sp1),_mm256_maddubs_epi16(rhs_mat_2367_21_sp1, lhs_mat_23_21_sp1));

                        __m256i iacc_mat_00_3_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_30_sp1, lhs_mat_01_30_sp1),_mm256_maddubs_epi16(rhs_mat_0145_31_sp1, lhs_mat_01_31_sp1));
                        __m256i iacc_mat_01_3_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_30_sp1, lhs_mat_01_30_sp1),_mm256_maddubs_epi16(rhs_mat_2367_31_sp1, lhs_mat_01_31_sp1));

                        __m256i iacc_mat_10_3_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_30_sp1, lhs_mat_23_30_sp1),_mm256_maddubs_epi16(rhs_mat_0145_31_sp1, lhs_mat_23_31_sp1));
                        __m256i iacc_mat_11_3_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_30_sp1, lhs_mat_23_30_sp1),_mm256_maddubs_epi16(rhs_mat_2367_31_sp1, lhs_mat_23_31_sp1));

                        __m256i iacc_mat_00_4_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_40_sp1, lhs_mat_01_40_sp1),_mm256_maddubs_epi16(rhs_mat_0145_41_sp1, lhs_mat_01_41_sp1));
                        __m256i iacc_mat_01_4_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_40_sp1, lhs_mat_01_40_sp1),_mm256_maddubs_epi16(rhs_mat_2367_41_sp1, lhs_mat_01_41_sp1));

                        __m256i iacc_mat_10_4_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_40_sp1, lhs_mat_23_40_sp1),_mm256_maddubs_epi16(rhs_mat_0145_41_sp1, lhs_mat_23_41_sp1));
                        __m256i iacc_mat_11_4_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_40_sp1, lhs_mat_23_40_sp1),_mm256_maddubs_epi16(rhs_mat_2367_41_sp1, lhs_mat_23_41_sp1));

                        __m256i iacc_mat_00_5_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_50_sp1, lhs_mat_01_50_sp1),_mm256_maddubs_epi16(rhs_mat_0145_51_sp1, lhs_mat_01_51_sp1));
                        __m256i iacc_mat_01_5_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_50_sp1, lhs_mat_01_50_sp1),_mm256_maddubs_epi16(rhs_mat_2367_51_sp1, lhs_mat_01_51_sp1));

                        __m256i iacc_mat_10_5_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_50_sp1, lhs_mat_23_50_sp1),_mm256_maddubs_epi16(rhs_mat_0145_51_sp1, lhs_mat_23_51_sp1));
                        __m256i iacc_mat_11_5_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_50_sp1, lhs_mat_23_50_sp1),_mm256_maddubs_epi16(rhs_mat_2367_51_sp1, lhs_mat_23_51_sp1));

                        __m256i iacc_mat_00_6_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_60_sp1, lhs_mat_01_60_sp1),_mm256_maddubs_epi16(rhs_mat_0145_61_sp1, lhs_mat_01_61_sp1));
                        __m256i iacc_mat_01_6_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_60_sp1, lhs_mat_01_60_sp1),_mm256_maddubs_epi16(rhs_mat_2367_61_sp1, lhs_mat_01_61_sp1));

                        __m256i iacc_mat_10_6_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_60_sp1, lhs_mat_23_60_sp1),_mm256_maddubs_epi16(rhs_mat_0145_61_sp1, lhs_mat_23_61_sp1));
                        __m256i iacc_mat_11_6_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_60_sp1, lhs_mat_23_60_sp1),_mm256_maddubs_epi16(rhs_mat_2367_61_sp1, lhs_mat_23_61_sp1));

                        __m256i iacc_mat_00_7_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_70_sp1, lhs_mat_01_70_sp1),_mm256_maddubs_epi16(rhs_mat_0145_71_sp1, lhs_mat_01_71_sp1));
                        __m256i iacc_mat_01_7_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_70_sp1, lhs_mat_01_70_sp1),_mm256_maddubs_epi16(rhs_mat_2367_71_sp1, lhs_mat_01_71_sp1));

                        __m256i iacc_mat_10_7_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_70_sp1, lhs_mat_23_70_sp1),_mm256_maddubs_epi16(rhs_mat_0145_71_sp1, lhs_mat_23_71_sp1));
                        __m256i iacc_mat_11_7_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_70_sp1, lhs_mat_23_70_sp1),_mm256_maddubs_epi16(rhs_mat_2367_71_sp1, lhs_mat_23_71_sp1));


                        __m256i iacc_mat_00_0_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_00_sp2, lhs_mat_01_00_sp2),_mm256_maddubs_epi16(rhs_mat_0145_01_sp2, lhs_mat_01_01_sp2));
                        __m256i iacc_mat_01_0_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_00_sp2, lhs_mat_01_00_sp2),_mm256_maddubs_epi16(rhs_mat_2367_01_sp2, lhs_mat_01_01_sp2));

                        __m256i iacc_mat_10_0_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_00_sp2, lhs_mat_23_00_sp2),_mm256_maddubs_epi16(rhs_mat_0145_01_sp2, lhs_mat_23_01_sp2));
                        __m256i iacc_mat_11_0_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_00_sp2, lhs_mat_23_00_sp2),_mm256_maddubs_epi16(rhs_mat_2367_01_sp2, lhs_mat_23_01_sp2));

                        __m256i iacc_mat_00_1_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_10_sp2, lhs_mat_01_10_sp2),_mm256_maddubs_epi16(rhs_mat_0145_11_sp2, lhs_mat_01_11_sp2));
                        __m256i iacc_mat_01_1_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_10_sp2, lhs_mat_01_10_sp2),_mm256_maddubs_epi16(rhs_mat_2367_11_sp2, lhs_mat_01_11_sp2));

                        __m256i iacc_mat_10_1_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_10_sp2, lhs_mat_23_10_sp2),_mm256_maddubs_epi16(rhs_mat_0145_11_sp2, lhs_mat_23_11_sp2));
                        __m256i iacc_mat_11_1_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_10_sp2, lhs_mat_23_10_sp2),_mm256_maddubs_epi16(rhs_mat_2367_11_sp2, lhs_mat_23_11_sp2));

                        __m256i iacc_mat_00_2_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_20_sp2, lhs_mat_01_20_sp2),_mm256_maddubs_epi16(rhs_mat_0145_21_sp2, lhs_mat_01_21_sp2));
                        __m256i iacc_mat_01_2_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_20_sp2, lhs_mat_01_20_sp2),_mm256_maddubs_epi16(rhs_mat_2367_21_sp2, lhs_mat_01_21_sp2));

                        __m256i iacc_mat_10_2_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_20_sp2, lhs_mat_23_20_sp2),_mm256_maddubs_epi16(rhs_mat_0145_21_sp2, lhs_mat_23_21_sp2));
                        __m256i iacc_mat_11_2_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_20_sp2, lhs_mat_23_20_sp2),_mm256_maddubs_epi16(rhs_mat_2367_21_sp2, lhs_mat_23_21_sp2));

                        __m256i iacc_mat_00_3_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_30_sp2, lhs_mat_01_30_sp2),_mm256_maddubs_epi16(rhs_mat_0145_31_sp2, lhs_mat_01_31_sp2));
                        __m256i iacc_mat_01_3_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_30_sp2, lhs_mat_01_30_sp2),_mm256_maddubs_epi16(rhs_mat_2367_31_sp2, lhs_mat_01_31_sp2));

                        __m256i iacc_mat_10_3_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_30_sp2, lhs_mat_23_30_sp2),_mm256_maddubs_epi16(rhs_mat_0145_31_sp2, lhs_mat_23_31_sp2));
                        __m256i iacc_mat_11_3_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_30_sp2, lhs_mat_23_30_sp2),_mm256_maddubs_epi16(rhs_mat_2367_31_sp2, lhs_mat_23_31_sp2));

                        __m256i iacc_mat_00_4_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_40_sp2, lhs_mat_01_40_sp2),_mm256_maddubs_epi16(rhs_mat_0145_41_sp2, lhs_mat_01_41_sp2));
                        __m256i iacc_mat_01_4_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_40_sp2, lhs_mat_01_40_sp2),_mm256_maddubs_epi16(rhs_mat_2367_41_sp2, lhs_mat_01_41_sp2));

                        __m256i iacc_mat_10_4_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_40_sp2, lhs_mat_23_40_sp2),_mm256_maddubs_epi16(rhs_mat_0145_41_sp2, lhs_mat_23_41_sp2));
                        __m256i iacc_mat_11_4_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_40_sp2, lhs_mat_23_40_sp2),_mm256_maddubs_epi16(rhs_mat_2367_41_sp2, lhs_mat_23_41_sp2));

                        __m256i iacc_mat_00_5_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_50_sp2, lhs_mat_01_50_sp2),_mm256_maddubs_epi16(rhs_mat_0145_51_sp2, lhs_mat_01_51_sp2));
                        __m256i iacc_mat_01_5_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_50_sp2, lhs_mat_01_50_sp2),_mm256_maddubs_epi16(rhs_mat_2367_51_sp2, lhs_mat_01_51_sp2));

                        __m256i iacc_mat_10_5_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_50_sp2, lhs_mat_23_50_sp2),_mm256_maddubs_epi16(rhs_mat_0145_51_sp2, lhs_mat_23_51_sp2));
                        __m256i iacc_mat_11_5_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_50_sp2, lhs_mat_23_50_sp2),_mm256_maddubs_epi16(rhs_mat_2367_51_sp2, lhs_mat_23_51_sp2));

                        __m256i iacc_mat_00_6_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_60_sp2, lhs_mat_01_60_sp2),_mm256_maddubs_epi16(rhs_mat_0145_61_sp2, lhs_mat_01_61_sp2));
                        __m256i iacc_mat_01_6_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_60_sp2, lhs_mat_01_60_sp2),_mm256_maddubs_epi16(rhs_mat_2367_61_sp2, lhs_mat_01_61_sp2));

                        __m256i iacc_mat_10_6_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_60_sp2, lhs_mat_23_60_sp2),_mm256_maddubs_epi16(rhs_mat_0145_61_sp2, lhs_mat_23_61_sp2));
                        __m256i iacc_mat_11_6_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_60_sp2, lhs_mat_23_60_sp2),_mm256_maddubs_epi16(rhs_mat_2367_61_sp2, lhs_mat_23_61_sp2));

                        __m256i iacc_mat_00_7_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_70_sp2, lhs_mat_01_70_sp2),_mm256_maddubs_epi16(rhs_mat_0145_71_sp2, lhs_mat_01_71_sp2));
                        __m256i iacc_mat_01_7_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_70_sp2, lhs_mat_01_70_sp2),_mm256_maddubs_epi16(rhs_mat_2367_71_sp2, lhs_mat_01_71_sp2));

                        __m256i iacc_mat_10_7_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_70_sp2, lhs_mat_23_70_sp2),_mm256_maddubs_epi16(rhs_mat_0145_71_sp2, lhs_mat_23_71_sp2));
                        __m256i iacc_mat_11_7_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_70_sp2, lhs_mat_23_70_sp2),_mm256_maddubs_epi16(rhs_mat_2367_71_sp2, lhs_mat_23_71_sp2));

                        // Combine results from both shuffle patterns for each output block
                        __m256i iacc_mat_00_0 = _mm256_add_epi16(iacc_mat_00_0_sp1, iacc_mat_00_0_sp2);
                        __m256i iacc_mat_01_0 = _mm256_add_epi16(iacc_mat_01_0_sp1, iacc_mat_01_0_sp2);
                        __m256i iacc_mat_10_0 = _mm256_add_epi16(iacc_mat_10_0_sp1, iacc_mat_10_0_sp2);
                        __m256i iacc_mat_11_0 = _mm256_add_epi16(iacc_mat_11_0_sp1, iacc_mat_11_0_sp2);

                        __m256i iacc_mat_00_1 = _mm256_add_epi16(iacc_mat_00_1_sp1, iacc_mat_00_1_sp2);
                        __m256i iacc_mat_01_1 = _mm256_add_epi16(iacc_mat_01_1_sp1, iacc_mat_01_1_sp2);
                        __m256i iacc_mat_10_1 = _mm256_add_epi16(iacc_mat_10_1_sp1, iacc_mat_10_1_sp2);
                        __m256i iacc_mat_11_1 = _mm256_add_epi16(iacc_mat_11_1_sp1, iacc_mat_11_1_sp2);

                        __m256i iacc_mat_00_2 = _mm256_add_epi16(iacc_mat_00_2_sp1, iacc_mat_00_2_sp2);
                        __m256i iacc_mat_01_2 = _mm256_add_epi16(iacc_mat_01_2_sp1, iacc_mat_01_2_sp2);
                        __m256i iacc_mat_10_2 = _mm256_add_epi16(iacc_mat_10_2_sp1, iacc_mat_10_2_sp2);
                        __m256i iacc_mat_11_2 = _mm256_add_epi16(iacc_mat_11_2_sp1, iacc_mat_11_2_sp2);

                        __m256i iacc_mat_00_3 = _mm256_add_epi16(iacc_mat_00_3_sp1, iacc_mat_00_3_sp2);
                        __m256i iacc_mat_01_3 = _mm256_add_epi16(iacc_mat_01_3_sp1, iacc_mat_01_3_sp2);
                        __m256i iacc_mat_10_3 = _mm256_add_epi16(iacc_mat_10_3_sp1, iacc_mat_10_3_sp2);
                        __m256i iacc_mat_11_3 = _mm256_add_epi16(iacc_mat_11_3_sp1, iacc_mat_11_3_sp2);

                        __m256i iacc_mat_00_4 = _mm256_add_epi16(iacc_mat_00_4_sp1, iacc_mat_00_4_sp2);
                        __m256i iacc_mat_01_4 = _mm256_add_epi16(iacc_mat_01_4_sp1, iacc_mat_01_4_sp2);
                        __m256i iacc_mat_10_4 = _mm256_add_epi16(iacc_mat_10_4_sp1, iacc_mat_10_4_sp2);
                        __m256i iacc_mat_11_4 = _mm256_add_epi16(iacc_mat_11_4_sp1, iacc_mat_11_4_sp2);

                        __m256i iacc_mat_00_5 = _mm256_add_epi16(iacc_mat_00_5_sp1, iacc_mat_00_5_sp2);
                        __m256i iacc_mat_01_5 = _mm256_add_epi16(iacc_mat_01_5_sp1, iacc_mat_01_5_sp2);
                        __m256i iacc_mat_10_5 = _mm256_add_epi16(iacc_mat_10_5_sp1, iacc_mat_10_5_sp2);
                        __m256i iacc_mat_11_5 = _mm256_add_epi16(iacc_mat_11_5_sp1, iacc_mat_11_5_sp2);

                        __m256i iacc_mat_00_6 = _mm256_add_epi16(iacc_mat_00_6_sp1, iacc_mat_00_6_sp2);
                        __m256i iacc_mat_01_6 = _mm256_add_epi16(iacc_mat_01_6_sp1, iacc_mat_01_6_sp2);
                        __m256i iacc_mat_10_6 = _mm256_add_epi16(iacc_mat_10_6_sp1, iacc_mat_10_6_sp2);
                        __m256i iacc_mat_11_6 = _mm256_add_epi16(iacc_mat_11_6_sp1, iacc_mat_11_6_sp2);

                        __m256i iacc_mat_00_7 = _mm256_add_epi16(iacc_mat_00_7_sp1, iacc_mat_00_7_sp2);
                        __m256i iacc_mat_01_7 = _mm256_add_epi16(iacc_mat_01_7_sp1, iacc_mat_01_7_sp2);
                        __m256i iacc_mat_10_7 = _mm256_add_epi16(iacc_mat_10_7_sp1, iacc_mat_10_7_sp2);
                        __m256i iacc_mat_11_7 = _mm256_add_epi16(iacc_mat_11_7_sp1, iacc_mat_11_7_sp2);

                        // Output of both shuffle patterns are added in order to sum dot product outputs of all 32 values in block
                        iacc_mat_00_0 = _mm256_madd_epi16(iacc_mat_00_0, scale_0145_0);
                        iacc_mat_01_0 = _mm256_madd_epi16(iacc_mat_01_0, scale_2367_0);
                        iacc_mat_10_0 = _mm256_madd_epi16(iacc_mat_10_0, scale_0145_0);
                        iacc_mat_11_0 = _mm256_madd_epi16(iacc_mat_11_0, scale_2367_0);

                        iacc_mat_00_1 = _mm256_madd_epi16(iacc_mat_00_1, scale_0145_1);
                        iacc_mat_01_1 = _mm256_madd_epi16(iacc_mat_01_1, scale_2367_1);
                        iacc_mat_10_1 = _mm256_madd_epi16(iacc_mat_10_1, scale_0145_1);
                        iacc_mat_11_1 = _mm256_madd_epi16(iacc_mat_11_1, scale_2367_1);

                        iacc_mat_00_2 = _mm256_madd_epi16(iacc_mat_00_2, scale_0145_2);
                        iacc_mat_01_2 = _mm256_madd_epi16(iacc_mat_01_2, scale_2367_2);
                        iacc_mat_10_2 = _mm256_madd_epi16(iacc_mat_10_2, scale_0145_2);
                        iacc_mat_11_2 = _mm256_madd_epi16(iacc_mat_11_2, scale_2367_2);

                        iacc_mat_00_3 = _mm256_madd_epi16(iacc_mat_00_3, scale_0145_3);
                        iacc_mat_01_3 = _mm256_madd_epi16(iacc_mat_01_3, scale_2367_3);
                        iacc_mat_10_3 = _mm256_madd_epi16(iacc_mat_10_3, scale_0145_3);
                        iacc_mat_11_3 = _mm256_madd_epi16(iacc_mat_11_3, scale_2367_3);

                        iacc_mat_00_4 = _mm256_madd_epi16(iacc_mat_00_4, scale_0145_4);
                        iacc_mat_01_4 = _mm256_madd_epi16(iacc_mat_01_4, scale_2367_4);
                        iacc_mat_10_4 = _mm256_madd_epi16(iacc_mat_10_4, scale_0145_4);
                        iacc_mat_11_4 = _mm256_madd_epi16(iacc_mat_11_4, scale_2367_4);

                        iacc_mat_00_5 = _mm256_madd_epi16(iacc_mat_00_5, scale_0145_5);
                        iacc_mat_01_5 = _mm256_madd_epi16(iacc_mat_01_5, scale_2367_5);
                        iacc_mat_10_5 = _mm256_madd_epi16(iacc_mat_10_5, scale_0145_5);
                        iacc_mat_11_5 = _mm256_madd_epi16(iacc_mat_11_5, scale_2367_5);

                        iacc_mat_00_6 = _mm256_madd_epi16(iacc_mat_00_6, scale_0145_6);
                        iacc_mat_01_6 = _mm256_madd_epi16(iacc_mat_01_6, scale_2367_6);
                        iacc_mat_10_6 = _mm256_madd_epi16(iacc_mat_10_6, scale_0145_6);
                        iacc_mat_11_6 = _mm256_madd_epi16(iacc_mat_11_6, scale_2367_6);

                        iacc_mat_00_7 = _mm256_madd_epi16(iacc_mat_00_7, scale_0145_7);
                        iacc_mat_01_7 = _mm256_madd_epi16(iacc_mat_01_7, scale_2367_7);
                        iacc_mat_10_7 = _mm256_madd_epi16(iacc_mat_10_7, scale_0145_7);
                        iacc_mat_11_7 = _mm256_madd_epi16(iacc_mat_11_7, scale_2367_7);

                        __m256i iacc_mat_00 = _mm256_add_epi32(_mm256_add_epi32(_mm256_add_epi32(iacc_mat_00_0, iacc_mat_00_1), _mm256_add_epi32(iacc_mat_00_2, iacc_mat_00_3)), _mm256_add_epi32(_mm256_add_epi32(iacc_mat_00_4, iacc_mat_00_5), _mm256_add_epi32(iacc_mat_00_6, iacc_mat_00_7)));
                        __m256i iacc_mat_01 = _mm256_add_epi32(_mm256_add_epi32(_mm256_add_epi32(iacc_mat_01_0, iacc_mat_01_1), _mm256_add_epi32(iacc_mat_01_2, iacc_mat_01_3)), _mm256_add_epi32(_mm256_add_epi32(iacc_mat_01_4, iacc_mat_01_5), _mm256_add_epi32(iacc_mat_01_6, iacc_mat_01_7)));
                        __m256i iacc_mat_10 = _mm256_add_epi32(_mm256_add_epi32(_mm256_add_epi32(iacc_mat_10_0, iacc_mat_10_1), _mm256_add_epi32(iacc_mat_10_2, iacc_mat_10_3)), _mm256_add_epi32(_mm256_add_epi32(iacc_mat_10_4, iacc_mat_10_5), _mm256_add_epi32(iacc_mat_10_6, iacc_mat_10_7)));
                        __m256i iacc_mat_11 = _mm256_add_epi32(_mm256_add_epi32(_mm256_add_epi32(iacc_mat_11_0, iacc_mat_11_1), _mm256_add_epi32(iacc_mat_11_2, iacc_mat_11_3)), _mm256_add_epi32(_mm256_add_epi32(iacc_mat_11_4, iacc_mat_11_5), _mm256_add_epi32(iacc_mat_11_6, iacc_mat_11_7)));

                        // Straighten out to make 4 row vectors
                        __m256i iacc_row_0 = _mm256_blend_epi32(iacc_mat_00, _mm256_shuffle_epi32(iacc_mat_01, 78), 204);
                        __m256i iacc_row_1 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_00, 78), iacc_mat_01, 204);
                        __m256i iacc_row_2 = _mm256_blend_epi32(iacc_mat_10, _mm256_shuffle_epi32(iacc_mat_11, 78), 204);
                        __m256i iacc_row_3 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_10, 78), iacc_mat_11, 204);

                        // Load the scale(d) values for all the 4 Q8_k blocks and repeat it across lanes
                        const __m128 row_scale_f32_sse = _mm_load_ps(a_ptrs[rp][b].d);
                        const __m256 row_scale_f32 = _mm256_set_m128(row_scale_f32_sse, row_scale_f32_sse);

                        // Multiply with appropriate scales and accumulate (for both d and dmin) below
                        acc_rows[rp * 4] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_0), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 0)), acc_rows[rp * 4]);
                        acc_rows[rp * 4 + 1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_1), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 85)), acc_rows[rp * 4 + 1]);
                        acc_rows[rp * 4 + 2] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_2), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_rows[rp * 4 + 2]);
                        acc_rows[rp * 4 + 3] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_3), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_rows[rp * 4 + 3]);

                        __m256i lhs_bsums_01_0123 = _mm256_inserti128_si256(_mm256_castsi128_si256(lhs_raw_bsums_01_0123), lhs_raw_bsums_01_0123, 1);
                        __m256i lhs_bsums_23_0123 = _mm256_inserti128_si256(_mm256_castsi128_si256(lhs_raw_bsums_23_0123), lhs_raw_bsums_23_0123, 1);
                        __m256i lhs_bsums_01_4567 = _mm256_inserti128_si256(_mm256_castsi128_si256(lhs_raw_bsums_01_4567), lhs_raw_bsums_01_4567, 1);
                        __m256i lhs_bsums_23_4567 = _mm256_inserti128_si256(_mm256_castsi128_si256(lhs_raw_bsums_23_4567), lhs_raw_bsums_23_4567, 1);

                       // Take two bsums from two Q8_Ks at a time and multiply with corresponding mins values from each Q2_K
                        __m256i iacc_row_min_0_01 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_01_0123, 0), mins_01);
                        __m256i iacc_row_min_1_01 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_01_0123, 170), mins_01);
                        __m256i iacc_row_min_2_01 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_23_0123, 0), mins_01);
                        __m256i iacc_row_min_3_01 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_23_0123, 170), mins_01);

                        __m256i iacc_row_min_0_23 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_01_0123, 85), mins_23);
                        __m256i iacc_row_min_1_23 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_01_0123, 255), mins_23);
                        __m256i iacc_row_min_2_23 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_23_0123, 85), mins_23);
                        __m256i iacc_row_min_3_23 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_23_0123, 255), mins_23);

                        __m256i iacc_row_min_0_45 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_01_4567, 0), mins_45);
                        __m256i iacc_row_min_1_45 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_01_4567, 170), mins_45);
                        __m256i iacc_row_min_2_45 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_23_4567, 0), mins_45);
                        __m256i iacc_row_min_3_45 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_23_4567, 170), mins_45);

                        __m256i iacc_row_min_0_67 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_01_4567, 85), mins_67);
                        __m256i iacc_row_min_1_67 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_01_4567, 255), mins_67);
                        __m256i iacc_row_min_2_67 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_23_4567, 85), mins_67);
                        __m256i iacc_row_min_3_67 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_23_4567, 255), mins_67);

                        __m256i iacc_row_min_0 = _mm256_add_epi32(_mm256_add_epi32(iacc_row_min_0_01, iacc_row_min_0_23), _mm256_add_epi32(iacc_row_min_0_45,iacc_row_min_0_67));
                        __m256i iacc_row_min_1 = _mm256_add_epi32(_mm256_add_epi32(iacc_row_min_1_01, iacc_row_min_1_23), _mm256_add_epi32(iacc_row_min_1_45,iacc_row_min_1_67));
                        __m256i iacc_row_min_2 = _mm256_add_epi32(_mm256_add_epi32(iacc_row_min_2_01, iacc_row_min_2_23), _mm256_add_epi32(iacc_row_min_2_45,iacc_row_min_2_67));
                        __m256i iacc_row_min_3 = _mm256_add_epi32(_mm256_add_epi32(iacc_row_min_3_01, iacc_row_min_3_23), _mm256_add_epi32(iacc_row_min_3_45,iacc_row_min_3_67));

                        acc_min_rows[rp * 4] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_0), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 0)), acc_min_rows[rp * 4]);
                        acc_min_rows[rp * 4 + 1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_1), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 85)), acc_min_rows[rp * 4 + 1]);
                        acc_min_rows[rp * 4 + 2] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_2), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_min_rows[rp * 4 + 2]);
                        acc_min_rows[rp * 4 + 3] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_3), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_min_rows[rp * 4 + 3]);

                    }
                }
            }
            // Store the accumulated values
            for (int i = 0; i < 16; i++) {
                _mm256_storeu_ps((float * )(s + ((y * 4 + i) * bs + x * 8)), _mm256_sub_ps(acc_rows[i], acc_min_rows[i]));

            }
        }
    }

    for (; y < nr / 4; y ++) {

        const block_q8_Kx4 * a_ptr = a_ptr_start + (y * nb);

        // Take group of eight block_q2_kx8 structures at each pass of the loop and perform dot product operation
        for (int64_t x = xstart; x < nc / 8; x++) {

            const block_q2_Kx8 * b_ptr = b_ptr_start + (x * b_nb);

            // Master FP accumulators
            __m256 acc_rows[4];
            for (int i = 0; i < 4; i++) {
                acc_rows[i] = _mm256_setzero_ps();
            }

            __m256 acc_min_rows[4];
            for (int i = 0; i < 4; i++) {
                acc_min_rows[i] = _mm256_setzero_ps();
            }

            for (int64_t b = 0; b < nb; b++) {
                // Delta values - Load the eight scale values of block_q2_kx8
                const __m256 col_scale_f32 = GGML_F32Cx8_LOAD(b_ptr[b].d);

                // dmin values - Load the eight dmin values of block_q2_kx8
                const __m256 col_dmin_f32 = GGML_F32Cx8_LOAD(b_ptr[b].dmin);

                // Loop to iterate over the sixteen sub blocks of a super block - eight sub blocks are processed per iteration
                for (int sb = 0; sb < QK_K / 128; sb++) {

                    // Load the eight block_q2_k for eight sub blocks quantized values interleaved with each other in chunks of eight bytes - B0,B1 ....B6,B7
                    const __m256i rhs_raw_mat_0123_0 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + sb * 256));
                    const __m256i rhs_raw_mat_4567_0 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 32 + sb * 256));
                    const __m256i rhs_raw_mat_0123_1 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 64 + sb * 256));
                    const __m256i rhs_raw_mat_4567_1 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 96 + sb * 256));
                    const __m256i rhs_raw_mat_0123_2 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 128 + sb * 256));
                    const __m256i rhs_raw_mat_4567_2 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 160 + sb * 256));
                    const __m256i rhs_raw_mat_0123_3 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 192 + sb * 256));
                    const __m256i rhs_raw_mat_4567_3 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 224 + sb * 256));

                    // Save the values in the following vectors in the formats B0B1B4B5, B2B3B6B7 for further processing and storing of values
                    //superblock    sub block   which part of sub block
                    const __m256i rhs_raw_mat_0145_0 = _mm256_blend_epi32(rhs_raw_mat_0123_0, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_0, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_0, requiredOrder), rhs_raw_mat_4567_0, 240);

                    const __m256i rhs_raw_mat_0145_1 = _mm256_blend_epi32(rhs_raw_mat_0123_1, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_1, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_1, requiredOrder), rhs_raw_mat_4567_1, 240);

                    const __m256i rhs_raw_mat_0145_2 = _mm256_blend_epi32(rhs_raw_mat_0123_2, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_2, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_2 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_2, requiredOrder), rhs_raw_mat_4567_2, 240);

                    const __m256i rhs_raw_mat_0145_3 = _mm256_blend_epi32(rhs_raw_mat_0123_3, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_3, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_3 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_3, requiredOrder), rhs_raw_mat_4567_3, 240);

                    // 2-bit -> 8-bit
                    // First sub block of the eight sub blocks processed in the iteration
                    const __m256i rhs_mat_0145_00 = _mm256_and_si256(rhs_raw_mat_0145_0, m3b); //B00(0-7) B01(0-7) B04(0-7) B05(0-7)
                    const __m256i rhs_mat_2367_00 = _mm256_and_si256(rhs_raw_mat_2367_0, m3b); //B02(0-7) B03(0-7) B06(0-7) B07(0-7)

                    const __m256i rhs_mat_0145_01 = _mm256_and_si256(rhs_raw_mat_0145_1, m3b); //B00(8-15) B01(8-15) B04(8-15) B05(8-15)
                    const __m256i rhs_mat_2367_01 = _mm256_and_si256(rhs_raw_mat_2367_1, m3b); //B02(8-15) B03(8-15) B06(8-15) B07(8-15)

                    // Second sub block of the eight sub blocks processed in the iteration
                    const __m256i rhs_mat_0145_10 = _mm256_and_si256(rhs_raw_mat_0145_2, m3b); //B10(0-7) B11(0-7) B14(0-7) B15(0-7)
                    const __m256i rhs_mat_2367_10 = _mm256_and_si256(rhs_raw_mat_2367_2, m3b); //B12(0-7) B13(0-7) B16(0-7) B17(0-7)

                    const __m256i rhs_mat_0145_11 = _mm256_and_si256(rhs_raw_mat_0145_3, m3b); //B10(8-15) B11(8-15) B14(8-15) B15(8-15)
                    const __m256i rhs_mat_2367_11 = _mm256_and_si256(rhs_raw_mat_2367_3, m3b); //B12(8-15) B13(8-15) B16(8-15) B17(8-15)

                    // Third sub block of the eight sub blocks processed in the iteration
                    const __m256i rhs_mat_0145_20 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_0, 2), m3b); //B20(0-7) B21(0-7) B24(0-7) B25(0-7)
                    const __m256i rhs_mat_2367_20 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_0, 2), m3b); //B22(0-7) B23(0-7) B26(0-7) B27(0-7)

                    const __m256i rhs_mat_0145_21 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_1, 2), m3b); //B20(8-15) B21(8-15) B24(8-15) B25(8-15)
                    const __m256i rhs_mat_2367_21 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_1, 2), m3b); //B22(8-15) B23(8-15) B26(8-15) B27(8-15)

                    // Fourth sub block of the eight sub blocks processed in the iteration
                    const __m256i rhs_mat_0145_30 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_2, 2), m3b); //B30(0-7) B31(0-7) B34(0-7) B35(0-7)
                    const __m256i rhs_mat_2367_30 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_2, 2), m3b); //B32(0-7) B33(0-7) B36(0-7) B37(0-7)

                    const __m256i rhs_mat_0145_31 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_3, 2), m3b); //B30(8-15) B31(8-15) B34(8-15) B35(8-15)
                    const __m256i rhs_mat_2367_31 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_3, 2), m3b); //B32(8-15) B33(8-15) B36(8-15) B37(8-15)

                    // Fifth sub block of the eight sub blocks processed in the iteration
                    const __m256i rhs_mat_0145_40 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_0, 4), m3b); //B40(0-7) B41(0-7) B44(0-7) B45(0-7)
                    const __m256i rhs_mat_2367_40 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_0, 4), m3b); //B42(0-7) B43(0-7) B46(0-7) B47(0-7)

                    const __m256i rhs_mat_0145_41 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_1, 4), m3b); //B40(8-15) B41(8-15) B44(8-15) B45(8-15)
                    const __m256i rhs_mat_2367_41 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_1, 4), m3b); //B42(8-15) B43(8-15) B46(8-15) B47(8-15)

                    // Sixth sub block of the eight sub blocks processed in the iteration
                    const __m256i rhs_mat_0145_50 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_2, 4), m3b); //B50(0-7) B51(0-7) B54(0-7) B55(0-7)
                    const __m256i rhs_mat_2367_50 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_2, 4), m3b); //B52(0-7) B53(0-7) B56(0-7) B57(0-7)

                    const __m256i rhs_mat_0145_51 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_3, 4), m3b); //B50(8-15) B51(8-15) B54(8-15) B55(8-15)
                    const __m256i rhs_mat_2367_51 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_3, 4), m3b); //B52(8-15) B53(8-15) B56(8-15) B57(8-15)

                    // Seventh sub block of the eight sub blocks processed in the iteration
                    const __m256i rhs_mat_0145_60 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_0, 6), m3b); //B60(0-7) B61(0-7) B64(0-7) B65(0-7)
                    const __m256i rhs_mat_2367_60 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_0, 6), m3b); //B62(0-7) B63(0-7) B66(0-7) B67(0-7)

                    const __m256i rhs_mat_0145_61 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_1, 6), m3b); //B60(8-15) B61(8-15) B64(8-15) B65(8-15)
                    const __m256i rhs_mat_2367_61 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_1, 6), m3b); //B62(8-15) B63(8-15) B66(8-15) B67(8-15)

                    // Eighth sub block of the eight sub blocks processed in the iteration
                    const __m256i rhs_mat_0145_70 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_2, 6), m3b); //B70(0-7) B71(0-7) B74(0-7) B75(0-7)
                    const __m256i rhs_mat_2367_70 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_2, 6), m3b); //B72(0-7) B73(0-7) B76(0-7) B77(0-7)

                    const __m256i rhs_mat_0145_71 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_3, 6), m3b); //B70(8-15) B71(8-15) B74(8-15) B75(8-15)
                    const __m256i rhs_mat_2367_71 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_3, 6), m3b); //B72(8-15) B73(8-15) B76(8-15) B77(8-15)

                    // Shuffle pattern one - right side input
                    const __m256i rhs_mat_0145_00_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_00, 136); //B00(0-3) B01(0-3) B00(0-3) B01(0-3) B04(0-3) B05(0-3) B04(0-3) B05(0-3)
                    const __m256i rhs_mat_2367_00_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_00, 136); //B02(0-3) B03(0-3) B02(0-3) B03(0-3) B06(0-3) B07(0-3) B06(0-3) B07(0-3)

                    const __m256i rhs_mat_0145_01_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_01, 136); //B00(8-11) B01(8-11) B00(8-11) B01(8-11) B04(8-11) B05(8-11) B04(8-11) B05(8-11)
                    const __m256i rhs_mat_2367_01_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_01, 136); //B02(8-11) B03(8-11) B02(8-11) B03(8-11) B06(8-11) B07(8-11) B06(8-11) B07(8-11)

                    const __m256i rhs_mat_0145_10_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_10, 136); //B10(0-3) B11(0-3) B10(0-3) B11(0-3) B14(0-3) B15(0-3) B14(0-3) B15(0-3)
                    const __m256i rhs_mat_2367_10_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_10, 136); //B12(0-3) B13(0-3) B12(0-3) B13(0-3) B16(0-3) B17(0-3) B16(0-3) B17(0-3)

                    const __m256i rhs_mat_0145_11_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_11, 136); //B10(8-11) B11(8-11) B10(8-11) B11(8-11) B14(8-11) B15(8-11) B14(8-11) B15(8-11)
                    const __m256i rhs_mat_2367_11_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_11, 136); //B12(8-11) B13(8-11) B12(8-11) B13(8-11) B16(8-11) B17(8-11) B16(8-11) B17(8-11)

                    const __m256i rhs_mat_0145_20_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_20, 136); //B20(0-3) B21(0-3) B20(0-3) B21(0-3) B24(0-3) B25(0-3) B24(0-3) B25(0-3)
                    const __m256i rhs_mat_2367_20_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_20, 136); //B22(0-3) B23(0-3) B22(0-3) B23(0-3) B26(0-3) B27(0-3) B26(0-3) B27(0-3)

                    const __m256i rhs_mat_0145_21_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_21, 136); //B20(8-11) B21(8-11) B20(8-11) B21(8-11) B24(8-11) B25(8-11) B24(8-11) B25(8-11)
                    const __m256i rhs_mat_2367_21_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_21, 136); //B22(8-11) B23(8-11) B22(8-11) B23(8-11) B26(8-11) B27(8-11) B26(8-11) B27(8-11)

                    const __m256i rhs_mat_0145_30_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_30, 136); //B30(0-3) B31(0-3) B30(0-3) B31(0-3) B34(0-3) B35(0-3) B34(0-3) B35(0-3)
                    const __m256i rhs_mat_2367_30_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_30, 136); //B32(0-3) B33(0-3) B32(0-3) B33(0-3) B36(0-3) B37(0-3) B36(0-3) B37(0-3)

                    const __m256i rhs_mat_0145_31_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_31, 136); //B30(8-11) B31(8-11) B30(8-11) B31(8-11) B34(8-11) B35(8-11) B34(8-11) B35(8-11
                    const __m256i rhs_mat_2367_31_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_31, 136); //B32(8-11) B33(8-11) B32(8-11) B33(8-11) B36(8-11) B37(8-11) B36(8-11) B37(8-11)

                    const __m256i rhs_mat_0145_40_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_40, 136); //B40(0-3) B41(0-3) B40(0-3) B41(0-3) B44(0-3) B45(0-3) B44(0-3) B45(0-3)
                    const __m256i rhs_mat_2367_40_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_40, 136); //B42(0-3) B43(0-3) B42(0-3) B43(0-3) B46(0-3) B47(0-3) B46(0-3) B47(0-3)

                    const __m256i rhs_mat_0145_41_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_41, 136); //B40(8-11) B41(8-11) B40(8-11) B41(8-11) B44(8-11) B45(8-11) B44(8-11) B45(8-11)
                    const __m256i rhs_mat_2367_41_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_41, 136); //B42(8-11) B43(8-11) B42(8-11) B43(8-11) B46(8-11) B47(8-11) B46(8-11) B47(8-11)

                    const __m256i rhs_mat_0145_50_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_50, 136); //B50(0-3) B51(0-3) B50(0-3) B51(0-3) B54(0-3) B55(0-3) B54(0-3) B55(0-3)
                    const __m256i rhs_mat_2367_50_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_50, 136); //B52(0-3) B53(0-3) B52(0-3) B53(0-3) B56(0-3) B57(0-3) B56(0-3) B57(0-3)

                    const __m256i rhs_mat_0145_51_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_51, 136); //B50(8-11) B51(8-11) B50(8-11) B51(8-11) B54(8-11) B55(8-11) B54(8-11) B55(8-11)
                    const __m256i rhs_mat_2367_51_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_51, 136); //B52(8-11) B53(8-11) B52(8-11) B53(8-11) B56(8-11) B57(8-11) B56(8-11) B57(8-11)

                    const __m256i rhs_mat_0145_60_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_60, 136); //B60(0-3) B61(0-3) B60(0-3) B61(0-3) B64(0-3) B65(0-3) B64(0-3) B65(0-3)
                    const __m256i rhs_mat_2367_60_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_60, 136); //B62(0-3) B63(0-3) B62(0-3) B63(0-3) B66(0-3) B67(0-3) B66(0-3) B67(0-3)

                    const __m256i rhs_mat_0145_61_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_61, 136); //B60(8-11) B61(8-11) B60(8-11) B61(8-11) B64(8-11) B65(8-11) B64(8-11) B65(8-11)
                    const __m256i rhs_mat_2367_61_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_61, 136); //B62(8-11) B63(8-11) B62(8-11) B63(8-11) B66(8-11) B67(8-11) B66(8-11) B67(8-11)

                    const __m256i rhs_mat_0145_70_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_70, 136); //B70(0-3) B71(0-3) B70(0-3) B71(0-3) B74(0-3) B75(0-3) B74(0-3) B75(0-3)
                    const __m256i rhs_mat_2367_70_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_70, 136); //B72(0-3) B73(0-3) B72(0-3) B73(0-3) B76(0-3) B77(0-3) B76(0-3) B77(0-3)

                    const __m256i rhs_mat_0145_71_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_71, 136); //B70(8-11) B71(8-11) B70(8-11) B71(8-11) B74(8-11) B75(8-11) B74(8-11) B75(8-11)
                    const __m256i rhs_mat_2367_71_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_71, 136); //B72(8-11) B73(8-11) B72(8-11) B73(8-11) B76(8-11) B77(8-11) B76(8-11) B77(8-11)


                    // Shuffle pattern two - right side input
                    const __m256i rhs_mat_0145_00_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_00, 221); //B00(4-7) B01(4-7) B00(4-7) B01(4-7) B04(4-7) B05(4-7) B04(4-7) B05(4-7)
                    const __m256i rhs_mat_2367_00_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_00, 221); //B02(4-7) B03(4-7) B02(4-7) B03(4-7) B06(4-7) B07(4-7) B06(4-7) B07(4-7)

                    const __m256i rhs_mat_0145_01_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_01, 221); //B00(12-15) B01(12-15) B00(12-15) B01(12-15) B04(12-15) B05(12-15) B04(12-15) B05(12-15)
                    const __m256i rhs_mat_2367_01_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_01, 221); //B02(12-15) B03(12-15) B02(12-15) B03(12-15) B06(12-15) B07(12-15) B06(12-15) B07(12-15)

                    const __m256i rhs_mat_0145_10_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_10, 221); //B10(4-7) B11(4-7) B10(4-7) B11(4-7) B14(4-7) B15(4-7) B14(4-7) B15(4-7)
                    const __m256i rhs_mat_2367_10_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_10, 221); //B12(4-7) B13(4-7) B12(4-7) B13(4-7) B16(4-7) B17(4-7) B16(4-7) B17(4-7)

                    const __m256i rhs_mat_0145_11_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_11, 221); //B10(12-15) B11(12-15) B10(12-15) B11(12-15) B14(12-15) B15(12-15) B14(12-15) B15(12-15)
                    const __m256i rhs_mat_2367_11_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_11, 221); //B12(12-15) B13(12-15) B12(12-15) B13(12-15) B16(12-15) B17(12-15) B16(12-15) B17(12-15)

                    const __m256i rhs_mat_0145_20_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_20, 221); //B20(4-7) B21(4-7) B20(4-7) B21(4-7) B24(4-7) B25(4-7) B24(4-7) B25(4-7)
                    const __m256i rhs_mat_2367_20_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_20, 221); //B22(4-7) B23(4-7) B22(4-7) B23(4-7) B26(4-7) B27(4-7) B26(4-7) B27(4-7)

                    const __m256i rhs_mat_0145_21_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_21, 221); //B20(12-15) B21(12-15) B20(12-15) B21(12-15) B24(12-15) B25(12-15) B24(12-15) B25(12-15)
                    const __m256i rhs_mat_2367_21_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_21, 221); //B22(12-15) B23(12-15) B22(12-15) B23(12-15) B26(12-15) B27(12-15) B26(12-15) B27(12-15)

                    const __m256i rhs_mat_0145_30_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_30, 221); //B30(4-7) B31(4-7) B30(4-7) B31(4-7) B34(4-7) B35(4-7) B34(4-7) B35(4-7)
                    const __m256i rhs_mat_2367_30_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_30, 221); //B32(4-7) B33(4-7) B32(4-7) B33(4-7) B36(4-7) B37(4-7) B36(4-7) B37(4-7)

                    const __m256i rhs_mat_0145_31_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_31, 221); //B30(12-15) B31(12-15) B30(12-15) B31(12-15) B34(12-15) B35(12-15) B34(12-15) B35(12-15)
                    const __m256i rhs_mat_2367_31_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_31, 221); //B32(12-15) B33(12-15) B32(12-15) B33(12-15) B36(12-15) B37(12-15) B36(12-15) B37(12-15)

                    const __m256i rhs_mat_0145_40_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_40, 221); //B40(4-7) B41(4-7) B40(4-7) B41(4-7) B44(4-7) B45(4-7) B44(4-7) B45(4-7)
                    const __m256i rhs_mat_2367_40_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_40, 221); //B42(4-7) B43(4-7) B42(4-7) B43(4-7) B46(4-7) B47(4-7) B46(4-7) B47(4-7)

                    const __m256i rhs_mat_0145_41_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_41, 221); //B40(12-15) B41(12-15) B40(12-15) B41(12-15) B44(12-15) B45(12-15) B44(12-15) B45(12-15)
                    const __m256i rhs_mat_2367_41_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_41, 221); //B42(12-15) B43(12-15) B42(12-15) B43(12-15) B46(12-15) B47(12-15) B46(12-15) B47(12-15)

                    const __m256i rhs_mat_0145_50_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_50, 221); //B50(4-7) B51(4-7) B50(4-7) B51(4-7) B54(4-7) B55(4-7) B54(4-7) B55(4-7)
                    const __m256i rhs_mat_2367_50_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_50, 221); //B52(4-7) B53(4-7) B52(4-7) B53(4-7) B56(4-7) B57(4-7) B56(4-7) B57(4-7)

                    const __m256i rhs_mat_0145_51_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_51, 221); //B50(12-15) B51(12-15) B50(12-15) B51(12-15) B54(12-15) B55(12-15) B54(12-15) B55(12-15)
                    const __m256i rhs_mat_2367_51_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_51, 221); //B52(12-15) B53(12-15) B52(12-15) B53(12-15) B56(12-15) B57(12-15) B56(12-15) B57(12-15)

                    const __m256i rhs_mat_0145_60_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_60, 221); //B60(4-7) B61(4-7) B60(4-7) B61(4-7) B64(4-7) B65(4-7) B64(4-7) B65(4-7)
                    const __m256i rhs_mat_2367_60_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_60, 221); //B62(4-7) B63(4-7) B62(4-7) B63(4-7) B66(4-7) B67(4-7) B66(4-7) B67(4-7)

                    const __m256i rhs_mat_0145_61_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_61, 221); //B60(12-15) B61(12-15) B60(12-15) B61(12-15) B64(12-15) B65(12-15) B64(12-15) B65(12-15)
                    const __m256i rhs_mat_2367_61_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_61, 221); //B62(12-15) B63(12-15) B62(12-15) B63(12-15) B66(12-15) B67(12-15) B66(12-15) B67(12-15)

                    const __m256i rhs_mat_0145_70_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_70, 221); //B70(4-7) B71(4-7) B70(4-7) B71(4-7) B74(4-7) B75(4-7) B74(4-7) B75(4-7)
                    const __m256i rhs_mat_2367_70_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_70, 221); //B72(4-7) B73(4-7) B72(4-7) B73(4-7) B76(4-7) B77(4-7) B76(4-7) B77(4-7)

                    const __m256i rhs_mat_0145_71_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_71, 221); //B70(12-15) B71(12-15) B70(12-15) B71(12-15) B74(12-15) B75(12-15) B74(12-15) B75(12-15)
                    const __m256i rhs_mat_2367_71_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_71, 221); //B72(12-15) B73(12-15) B72(12-15) B73(12-15) B76(12-15) B77(12-15) B76(12-15) B77(12-15)


                    //Scales and Mins of corresponding sub blocks from different Q2_K structures are stored together
                    //s00 m00  s01 m01   s10 m10  s11 m11  s20 m20  s21 m21   s30 m30  s31 m31  s40 m40  s41 m41   s50 m50  s51 m51  s60 m60  s61 m61   s70 m70  s71 m71

                    // Combine mins and scales for sub-blocks: 0-1, 2-3, 4-5, 6-7 in the sb loop
                    const __m128i mins_and_scales_01 = _mm_loadu_si128((const __m128i *)(b_ptr[b].scales + sb * 64));
                    const __m128i mins_and_scales_23 = _mm_loadu_si128((const __m128i *)(b_ptr[b].scales + 16 + sb * 64));
                    const __m128i mins_and_scales_45 = _mm_loadu_si128((const __m128i *)(b_ptr[b].scales + 32 + sb * 64));
                    const __m128i mins_and_scales_67 = _mm_loadu_si128((const __m128i *)(b_ptr[b].scales + 48 + sb * 64));

                    // Extract scales which is lower half from mins_and_scales
                    const __m128i scales_01 = _mm_and_si128(mins_and_scales_01, m4b_sse);
                    const __m128i scales_23 = _mm_and_si128(mins_and_scales_23, m4b_sse);
                    const __m128i scales_45 = _mm_and_si128(mins_and_scales_45, m4b_sse);
                    const __m128i scales_67 = _mm_and_si128(mins_and_scales_67, m4b_sse);

                    // Extract mins which is upper half from mins_and_scales
                    const __m256i mins_01 = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(mins_and_scales_01, 4), m4b_sse));
                    const __m256i mins_23 = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(mins_and_scales_23, 4), m4b_sse));
                    const __m256i mins_45 = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(mins_and_scales_45, 4), m4b_sse));
                    const __m256i mins_67 = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(mins_and_scales_67, 4), m4b_sse));

                    const __m256i scales_0 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(scales_01, scalesmask1_sse));
                    const __m256i scales_1 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(scales_01, scalesmask2_sse));

                    const __m256i scales_2 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(scales_23, scalesmask1_sse));
                    const __m256i scales_3 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(scales_23, scalesmask2_sse));

                    const __m256i scales_4 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(scales_45, scalesmask1_sse));
                    const __m256i scales_5 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(scales_45, scalesmask2_sse));

                    const __m256i scales_6 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(scales_67, scalesmask1_sse));
                    const __m256i scales_7 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(scales_67, scalesmask2_sse));

                    const __m256i scale_0145_0 = _mm256_shuffle_epi32(scales_0, 68);
                    const __m256i scale_2367_0 = _mm256_shuffle_epi32(scales_0, 238);

                    const __m256i scale_0145_1 = _mm256_shuffle_epi32(scales_1, 68);
                    const __m256i scale_2367_1 = _mm256_shuffle_epi32(scales_1, 238);

                    const __m256i scale_0145_2 = _mm256_shuffle_epi32(scales_2, 68);
                    const __m256i scale_2367_2 = _mm256_shuffle_epi32(scales_2, 238);

                    const __m256i scale_0145_3 = _mm256_shuffle_epi32(scales_3, 68);
                    const __m256i scale_2367_3 = _mm256_shuffle_epi32(scales_3, 238);

                    const __m256i scale_0145_4 = _mm256_shuffle_epi32(scales_4, 68);
                    const __m256i scale_2367_4 = _mm256_shuffle_epi32(scales_4, 238);

                    const __m256i scale_0145_5 = _mm256_shuffle_epi32(scales_5, 68);
                    const __m256i scale_2367_5 = _mm256_shuffle_epi32(scales_5, 238);

                    const __m256i scale_0145_6 = _mm256_shuffle_epi32(scales_6, 68);
                    const __m256i scale_2367_6 = _mm256_shuffle_epi32(scales_6, 238);

                    const __m256i scale_0145_7 = _mm256_shuffle_epi32(scales_7, 68);
                    const __m256i scale_2367_7 = _mm256_shuffle_epi32(scales_7, 238);

                    // Load the four block_q8_k quantized values interleaved with each other in chunks of eight bytes - A0,A1,A2,A3
                    // Loaded as set of 128 bit vectors and repeated into a 256 bit vector
                    __m256i lhs_mat_0123_00 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 512 * sb)));
                    __m256i lhs_mat_01_00 = _mm256_permute2f128_si256(lhs_mat_0123_00, lhs_mat_0123_00, 0);
                    __m256i lhs_mat_23_00 = _mm256_permute2f128_si256(lhs_mat_0123_00, lhs_mat_0123_00, 17);
                    __m256i lhs_mat_0123_01 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 32 + 512 * sb)));
                    __m256i lhs_mat_01_01 = _mm256_permute2f128_si256(lhs_mat_0123_01, lhs_mat_0123_01, 0);
                    __m256i lhs_mat_23_01 = _mm256_permute2f128_si256(lhs_mat_0123_01, lhs_mat_0123_01, 17);
                    __m256i lhs_mat_0123_10 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 64 + 512 * sb)));
                    __m256i lhs_mat_01_10 = _mm256_permute2f128_si256(lhs_mat_0123_10, lhs_mat_0123_10, 0);
                    __m256i lhs_mat_23_10 = _mm256_permute2f128_si256(lhs_mat_0123_10, lhs_mat_0123_10, 17);
                    __m256i lhs_mat_0123_11 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 96 + 512 * sb)));
                    __m256i lhs_mat_01_11 = _mm256_permute2f128_si256(lhs_mat_0123_11, lhs_mat_0123_11, 0);
                    __m256i lhs_mat_23_11 = _mm256_permute2f128_si256(lhs_mat_0123_11, lhs_mat_0123_11, 17);
                    __m256i lhs_mat_0123_20 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 128 + 512 * sb)));
                    __m256i lhs_mat_01_20 = _mm256_permute2f128_si256(lhs_mat_0123_20, lhs_mat_0123_20, 0);
                    __m256i lhs_mat_23_20 = _mm256_permute2f128_si256(lhs_mat_0123_20, lhs_mat_0123_20, 17);
                    __m256i lhs_mat_0123_21 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 160 + 512 * sb)));
                    __m256i lhs_mat_01_21 = _mm256_permute2f128_si256(lhs_mat_0123_21, lhs_mat_0123_21, 0);
                    __m256i lhs_mat_23_21 = _mm256_permute2f128_si256(lhs_mat_0123_21, lhs_mat_0123_21, 17);
                    __m256i lhs_mat_0123_30 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 192 + 512 * sb)));
                    __m256i lhs_mat_01_30 = _mm256_permute2f128_si256(lhs_mat_0123_30, lhs_mat_0123_30, 0);
                    __m256i lhs_mat_23_30 = _mm256_permute2f128_si256(lhs_mat_0123_30, lhs_mat_0123_30, 17);
                    __m256i lhs_mat_0123_31 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 224 + 512 * sb)));
                    __m256i lhs_mat_01_31 = _mm256_permute2f128_si256(lhs_mat_0123_31, lhs_mat_0123_31, 0);
                    __m256i lhs_mat_23_31 = _mm256_permute2f128_si256(lhs_mat_0123_31, lhs_mat_0123_31, 17);

                    __m256i lhs_mat_0123_40 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 256 + 512 * sb)));
                    __m256i lhs_mat_01_40 = _mm256_permute2f128_si256(lhs_mat_0123_40, lhs_mat_0123_40, 0);
                    __m256i lhs_mat_23_40 = _mm256_permute2f128_si256(lhs_mat_0123_40, lhs_mat_0123_40, 17);
                    __m256i lhs_mat_0123_41 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 288 + 512 * sb)));
                    __m256i lhs_mat_01_41 = _mm256_permute2f128_si256(lhs_mat_0123_41, lhs_mat_0123_41, 0);
                    __m256i lhs_mat_23_41 = _mm256_permute2f128_si256(lhs_mat_0123_41, lhs_mat_0123_41, 17);
                    __m256i lhs_mat_0123_50 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 320 + 512 * sb)));
                    __m256i lhs_mat_01_50 = _mm256_permute2f128_si256(lhs_mat_0123_50, lhs_mat_0123_50, 0);
                    __m256i lhs_mat_23_50 = _mm256_permute2f128_si256(lhs_mat_0123_50, lhs_mat_0123_50, 17);
                    __m256i lhs_mat_0123_51 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 352 + 512 * sb)));
                    __m256i lhs_mat_01_51 = _mm256_permute2f128_si256(lhs_mat_0123_51, lhs_mat_0123_51, 0);
                    __m256i lhs_mat_23_51 = _mm256_permute2f128_si256(lhs_mat_0123_51, lhs_mat_0123_51, 17);
                    __m256i lhs_mat_0123_60 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 384 + 512 * sb)));
                    __m256i lhs_mat_01_60 = _mm256_permute2f128_si256(lhs_mat_0123_60, lhs_mat_0123_60, 0);
                    __m256i lhs_mat_23_60 = _mm256_permute2f128_si256(lhs_mat_0123_60, lhs_mat_0123_60, 17);
                    __m256i lhs_mat_0123_61 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 416 + 512 * sb)));
                    __m256i lhs_mat_01_61 = _mm256_permute2f128_si256(lhs_mat_0123_61, lhs_mat_0123_61, 0);
                    __m256i lhs_mat_23_61 = _mm256_permute2f128_si256(lhs_mat_0123_61, lhs_mat_0123_61, 17);
                    __m256i lhs_mat_0123_70 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 448 + 512 * sb)));
                    __m256i lhs_mat_01_70 = _mm256_permute2f128_si256(lhs_mat_0123_70, lhs_mat_0123_70, 0);
                    __m256i lhs_mat_23_70 = _mm256_permute2f128_si256(lhs_mat_0123_70, lhs_mat_0123_70, 17);
                    __m256i lhs_mat_0123_71 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 480 + 512 * sb)));
                    __m256i lhs_mat_01_71 = _mm256_permute2f128_si256(lhs_mat_0123_71, lhs_mat_0123_71, 0);
                    __m256i lhs_mat_23_71 = _mm256_permute2f128_si256(lhs_mat_0123_71, lhs_mat_0123_71, 17);

                    // Bsums are loaded for the different Q8_K blocks
                    __m128i lhs_raw_bsums_01_0123 = _mm_loadu_si128((const __m128i *)((a_ptr[b].bsums + 32 * sb)));
                    __m128i lhs_raw_bsums_23_0123 = _mm_loadu_si128((const __m128i *)(a_ptr[b].bsums + 8 + 32 * sb));
                    __m128i lhs_raw_bsums_01_4567 = _mm_loadu_si128((const __m128i *)((a_ptr[b].bsums + 16 + 32 * sb)));
                    __m128i lhs_raw_bsums_23_4567 = _mm_loadu_si128((const __m128i *)(a_ptr[b].bsums + 24 + 32 * sb));

                    // Shuffle pattern one - left side input
                    const __m256i lhs_mat_01_00_sp1 = _mm256_shuffle_epi32(lhs_mat_01_00, 160); //A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3) A00(0-3) A01(0-3) A01(0-3)
                    const __m256i lhs_mat_23_00_sp1 = _mm256_shuffle_epi32(lhs_mat_23_00, 160); //A02(0-3) A03(0-3) A02(0-3) A03(0-3) A02(0-3) A03(0-3) A02(0-3) A03(0-3)

                    const __m256i lhs_mat_01_01_sp1 = _mm256_shuffle_epi32(lhs_mat_01_01, 160); //A00(8-11) A00(8-11) A01(8-11) A01(8-11) A00(8-11) A00(8-11) A01(8-11) A01(8-11)
                    const __m256i lhs_mat_23_01_sp1 = _mm256_shuffle_epi32(lhs_mat_23_01, 160); //A02(8-11) A03(8-11) A02(8-11) A03(8-11) A02(8-11) A03(8-11) A02(8-11) A03(8-11)

                    const __m256i lhs_mat_01_10_sp1 = _mm256_shuffle_epi32(lhs_mat_01_10, 160); //A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3) A10(0-3) A11(0-3) A11(0-3)
                    const __m256i lhs_mat_23_10_sp1 = _mm256_shuffle_epi32(lhs_mat_23_10, 160); //A12(0-3) A13(0-3) A12(0-3) A13(0-3) A12(0-3) A13(0-3) A12(0-3) A13(0-3)

                    const __m256i lhs_mat_01_11_sp1 = _mm256_shuffle_epi32(lhs_mat_01_11, 160); //A10(8-11) A10(8-11) A11(8-11) A11(8-11) A10(8-11) A10(8-11) A11(8-11) A11(8-11)
                    const __m256i lhs_mat_23_11_sp1 = _mm256_shuffle_epi32(lhs_mat_23_11, 160); //A12(8-11) A13(8-11) A12(8-11) A13(8-11) A12(8-11) A13(8-11) A12(8-11) A13(8-11)

                    const __m256i lhs_mat_01_20_sp1 = _mm256_shuffle_epi32(lhs_mat_01_20, 160); //A20(0-3) A20(0-3) A21(0-3) A21(0-3) A20(0-3) A20(0-3) A21(0-3) A21(0-3)
                    const __m256i lhs_mat_23_20_sp1 = _mm256_shuffle_epi32(lhs_mat_23_20, 160); //A22(0-3) A23(0-3) A22(0-3) A23(0-3) A22(0-3) A23(0-3) A22(0-3) A23(0-3)

                    const __m256i lhs_mat_01_21_sp1 = _mm256_shuffle_epi32(lhs_mat_01_21, 160); //A20(8-11) A20(8-11) A21(8-11) A21(8-11) A20(8-11) A20(8-11) A21(8-11) A21(8-11)
                    const __m256i lhs_mat_23_21_sp1 = _mm256_shuffle_epi32(lhs_mat_23_21, 160); //A22(8-11) A23(8-11) A22(8-11) A23(8-11) A22(8-11) A23(8-11) A22(8-11) A23(8-11)

                    const __m256i lhs_mat_01_30_sp1 = _mm256_shuffle_epi32(lhs_mat_01_30, 160); //A30(0-3) A30(0-3) A31(0-3) A31(0-3) A30(0-3) A30(0-3) A31(0-3) A31(0-3)
                    const __m256i lhs_mat_23_30_sp1 = _mm256_shuffle_epi32(lhs_mat_23_30, 160); //A32(0-3) A33(0-3) A32(0-3) A33(0-3) A32(0-3) A33(0-3) A32(0-3) A33(0-3)

                    const __m256i lhs_mat_01_31_sp1 = _mm256_shuffle_epi32(lhs_mat_01_31, 160); //A30(8-11) A30(8-11) A31(8-11) A31(8-11) A30(8-11) A30(8-11) A31(8-11) A31(8-11)
                    const __m256i lhs_mat_23_31_sp1 = _mm256_shuffle_epi32(lhs_mat_23_31, 160); //A32(8-11) A33(8-11) A32(8-11) A33(8-11) A32(8-11) A33(8-11) A32(8-11) A33(8-11)

                    const __m256i lhs_mat_01_40_sp1 = _mm256_shuffle_epi32(lhs_mat_01_40, 160); //A40(0-3) A40(0-3) A41(0-3) A41(0-3) A40(0-3) A40(0-3) A41(0-3) A41(0-3)
                    const __m256i lhs_mat_23_40_sp1 = _mm256_shuffle_epi32(lhs_mat_23_40, 160); //A42(0-3) A43(0-3) A42(0-3) A43(0-3) A42(0-3) A43(0-3) A42(0-3) A43(0-3)

                    const __m256i lhs_mat_01_41_sp1 = _mm256_shuffle_epi32(lhs_mat_01_41, 160); //A40(8-11) A40(8-11) A41(8-11) A41(8-11) A40(8-11) A40(8-11) A41(8-11) A41(8-11)
                    const __m256i lhs_mat_23_41_sp1 = _mm256_shuffle_epi32(lhs_mat_23_41, 160); //A42(8-11) A43(8-11) A42(8-11) A43(8-11) A42(8-11) A43(8-11) A42(8-11) A43(8-11)

                    const __m256i lhs_mat_01_50_sp1 = _mm256_shuffle_epi32(lhs_mat_01_50, 160); //A50(0-3) A50(0-3) A51(0-3) A51(0-3) A50(0-3) A50(0-3) A51(0-3) A51(0-3)
                    const __m256i lhs_mat_23_50_sp1 = _mm256_shuffle_epi32(lhs_mat_23_50, 160); //A52(0-3) A53(0-3) A52(0-3) A53(0-3) A52(0-3) A53(0-3) A52(0-3) A53(0-3)

                    const __m256i lhs_mat_01_51_sp1 = _mm256_shuffle_epi32(lhs_mat_01_51, 160); //A50(8-11) A50(8-11) A51(8-11) A51(8-11) A50(8-11) A50(8-11) A51(8-11) A51(8-11)
                    const __m256i lhs_mat_23_51_sp1 = _mm256_shuffle_epi32(lhs_mat_23_51, 160); //A52(8-11) A53(8-11) A52(8-11) A53(8-11) A52(8-11) A53(8-11) A52(8-11) A53(8-11)

                    const __m256i lhs_mat_01_60_sp1 = _mm256_shuffle_epi32(lhs_mat_01_60, 160); //A60(0-3) A60(0-3) A61(0-3) A61(0-3) A60(0-3) A60(0-3) A61(0-3) A61(0-3)
                    const __m256i lhs_mat_23_60_sp1 = _mm256_shuffle_epi32(lhs_mat_23_60, 160); //A62(0-3) A63(0-3) A62(0-3) A63(0-3) A62(0-3) A63(0-3) A62(0-3) A63(0-3)

                    const __m256i lhs_mat_01_61_sp1 = _mm256_shuffle_epi32(lhs_mat_01_61, 160); //A60(8-11) A60(8-11) A61(8-11) A61(8-11) A60(8-11) A60(8-11) A61(8-11) A61(8-11)
                    const __m256i lhs_mat_23_61_sp1 = _mm256_shuffle_epi32(lhs_mat_23_61, 160); //A62(8-11) A63(8-11) A62(8-11) A63(8-11) A62(8-11) A63(8-11) A62(8-11) A63(8-11)

                    const __m256i lhs_mat_01_70_sp1 = _mm256_shuffle_epi32(lhs_mat_01_70, 160); //A70(0-3) A70(0-3) A71(0-3) A71(0-3) A70(0-3) A70(0-3) A71(0-3) A71(0-3)
                    const __m256i lhs_mat_23_70_sp1 = _mm256_shuffle_epi32(lhs_mat_23_70, 160); //A72(0-3) A73(0-3) A72(0-3) A73(0-3) A72(0-3) A73(0-3) A72(0-3) A73(0-3)

                    const __m256i lhs_mat_01_71_sp1 = _mm256_shuffle_epi32(lhs_mat_01_71, 160); //A70(8-11) A70(8-11) A71(8-11) A71(8-11) A70(8-11) A70(8-11) A71(8-11) A71(8-11)
                    const __m256i lhs_mat_23_71_sp1 = _mm256_shuffle_epi32(lhs_mat_23_71, 160); //A72(8-11) A73(8-11) A72(8-11) A73(8-11) A72(8-11) A73(8-11) A72(8-11) A73(8-11)

                    // Shuffle pattern two- left side input
                    const __m256i lhs_mat_01_00_sp2 = _mm256_shuffle_epi32(lhs_mat_01_00, 245); //A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7) A00(4-7) A01(4-7) A01(4-7)
                    const __m256i lhs_mat_23_00_sp2 = _mm256_shuffle_epi32(lhs_mat_23_00, 245); //A02(4-7) A03(4-7) A02(4-7) A03(4-7) A02(4-7) A03(4-7) A02(4-7) A03(4-7)

                    const __m256i lhs_mat_01_01_sp2 = _mm256_shuffle_epi32(lhs_mat_01_01, 245); //A00(12-15) A00(12-15) A01(12-15) A01(12-15) A00(12-15) A00(12-15) A01(12-15) A01(12-15)
                    const __m256i lhs_mat_23_01_sp2 = _mm256_shuffle_epi32(lhs_mat_23_01, 245); //A02(12-15) A03(12-15) A02(12-15) A03(12-15) A02(12-15) A03(12-15) A02(12-15) A03(12-15)

                    const __m256i lhs_mat_01_10_sp2 = _mm256_shuffle_epi32(lhs_mat_01_10, 245); //A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7) A10(4-7) A11(4-7) A11(4-7)
                    const __m256i lhs_mat_23_10_sp2 = _mm256_shuffle_epi32(lhs_mat_23_10, 245); //A12(4-7) A13(4-7) A12(4-7) A13(4-7) A12(4-7) A13(4-7) A12(4-7) A13(4-7)

                    const __m256i lhs_mat_01_11_sp2 = _mm256_shuffle_epi32(lhs_mat_01_11, 245); //A10(12-15) A10(12-15) A11(12-15) A11(12-15) A10(12-15) A10(12-15) A11(12-15) A11(12-15)
                    const __m256i lhs_mat_23_11_sp2 = _mm256_shuffle_epi32(lhs_mat_23_11, 245); //A12(12-15) A13(12-15) A12(12-15) A13(12-15) A12(12-15) A13(12-15) A12(12-15) A13(12-15)

                    const __m256i lhs_mat_01_20_sp2 = _mm256_shuffle_epi32(lhs_mat_01_20, 245); //A20(4-7) A20(4-7) A21(4-7) A21(4-7) A20(4-7) A20(4-7) A21(4-7) A21(4-7)
                    const __m256i lhs_mat_23_20_sp2 = _mm256_shuffle_epi32(lhs_mat_23_20, 245); //A22(4-7) A23(4-7) A22(4-7) A23(4-7) A22(4-7) A23(4-7) A22(4-7) A23(4-7)

                    const __m256i lhs_mat_01_21_sp2 = _mm256_shuffle_epi32(lhs_mat_01_21, 245); //A20(12-15) A20(12-15) A21(12-15) A21(12-15) A20(12-15) A20(12-15) A21(12-15) A21(12-15)
                    const __m256i lhs_mat_23_21_sp2 = _mm256_shuffle_epi32(lhs_mat_23_21, 245); //A22(12-15) A23(12-15) A22(12-15) A23(12-15) A22(12-15) A23(12-15) A22(12-15) A23(12-15)

                    const __m256i lhs_mat_01_30_sp2 = _mm256_shuffle_epi32(lhs_mat_01_30, 245); //A30(4-7) A30(4-7) A31(4-7) A31(4-7) A30(4-7) A30(4-7) A31(4-7) A31(4-7)
                    const __m256i lhs_mat_23_30_sp2 = _mm256_shuffle_epi32(lhs_mat_23_30, 245); //A32(4-7) A33(4-7) A32(4-7) A33(4-7) A32(4-7) A33(4-7) A32(4-7) A33(4-7)

                    const __m256i lhs_mat_01_31_sp2 = _mm256_shuffle_epi32(lhs_mat_01_31, 245); //A30(12-15) A30(12-15) A31(12-15) A31(12-15) A30(12-15) A30(12-15) A31(12-15) A31(12-15)
                    const __m256i lhs_mat_23_31_sp2 = _mm256_shuffle_epi32(lhs_mat_23_31, 245); //A32(12-15) A33(12-15) A32(12-15) A33(12-15) A32(12-15) A33(12-15) A32(12-15) A33(12-15)

                    const __m256i lhs_mat_01_40_sp2 = _mm256_shuffle_epi32(lhs_mat_01_40, 245); //A40(4-7) A40(4-7) A41(4-7) A41(4-7) A40(4-7) A40(4-7) A41(4-7) A41(4-7)
                    const __m256i lhs_mat_23_40_sp2 = _mm256_shuffle_epi32(lhs_mat_23_40, 245); //A42(4-7) A43(4-7) A42(4-7) A43(4-7) A42(4-7) A43(4-7) A42(4-7) A43(4-7)

                    const __m256i lhs_mat_01_41_sp2 = _mm256_shuffle_epi32(lhs_mat_01_41, 245); //A40(12-15) A40(12-15) A41(12-15) A41(12-15) A40(12-15) A40(12-15) A41(12-15) A41(12-15)
                    const __m256i lhs_mat_23_41_sp2 = _mm256_shuffle_epi32(lhs_mat_23_41, 245); //A42(12-15) A43(12-15) A42(12-15) A43(12-15) A42(12-15) A43(12-15) A42(12-15) A43(12-15)

                    const __m256i lhs_mat_01_50_sp2 = _mm256_shuffle_epi32(lhs_mat_01_50, 245); //A50(4-7) A50(4-7) A51(4-7) A51(4-7) A50(4-7) A50(4-7) A51(4-7) A51(4-7)
                    const __m256i lhs_mat_23_50_sp2 = _mm256_shuffle_epi32(lhs_mat_23_50, 245); //A52(4-7) A53(4-7) A52(4-7) A53(4-7) A52(4-7) A53(4-7) A52(4-7) A53(4-7)

                    const __m256i lhs_mat_01_51_sp2 = _mm256_shuffle_epi32(lhs_mat_01_51, 245); //A50(12-15) A50(12-15) A51(12-15) A51(12-15) A50(12-15) A50(12-15) A51(12-15) A51(12-15)
                    const __m256i lhs_mat_23_51_sp2 = _mm256_shuffle_epi32(lhs_mat_23_51, 245); //A52(12-15) A53(12-15) A52(12-15) A53(12-15) A52(12-15) A53(12-15) A52(12-15) A53(12-15)

                    const __m256i lhs_mat_01_60_sp2 = _mm256_shuffle_epi32(lhs_mat_01_60, 245); //A60(4-7) A60(4-7) A61(4-7) A61(4-7) A60(4-7) A60(4-7) A61(4-7) A61(4-7)
                    const __m256i lhs_mat_23_60_sp2 = _mm256_shuffle_epi32(lhs_mat_23_60, 245); //A62(4-7) A63(4-7) A62(4-7) A63(4-7) A62(4-7) A63(4-7) A62(4-7) A63(4-7)

                    const __m256i lhs_mat_01_61_sp2 = _mm256_shuffle_epi32(lhs_mat_01_61, 245); //A60(12-15) A60(12-15) A61(12-15) A61(12-15) A60(12-15) A60(12-15) A61(12-15) A61(12-15)
                    const __m256i lhs_mat_23_61_sp2 = _mm256_shuffle_epi32(lhs_mat_23_61, 245); //A62(12-15) A63(12-15) A62(12-15) A63(12-15) A62(12-15) A63(12-15) A62(12-15) A63(12-15)

                    const __m256i lhs_mat_01_70_sp2 = _mm256_shuffle_epi32(lhs_mat_01_70, 245); //A70(4-7) A70(4-7) A71(4-7) A71(4-7) A70(4-7) A70(4-7) A71(4-7) A71(4-7)
                    const __m256i lhs_mat_23_70_sp2 = _mm256_shuffle_epi32(lhs_mat_23_70, 245); //A72(4-7) A73(4-7) A72(4-7) A73(4-7) A72(4-7) A73(4-7) A72(4-7) A73(4-7)

                    const __m256i lhs_mat_01_71_sp2 = _mm256_shuffle_epi32(lhs_mat_01_71, 245); //A70(12-15) A70(12-15) A71(12-15) A71(12-15) A70(12-15) A70(12-15) A71(12-15) A71(12-15)
                    const __m256i lhs_mat_23_71_sp2 = _mm256_shuffle_epi32(lhs_mat_23_71, 245); //A72(12-15) A73(12-15) A72(12-15) A73(12-15) A72(12-15) A73(12-15) A72(12-15) A73(12-15)

                    // The values arranged in shuffle patterns are operated with dot product operation within 32 bit lane i.e corresponding bytes and multiplied and added into 32 bit integers within 32 bit lane
                    __m256i iacc_mat_00_0_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_00_sp1, lhs_mat_01_00_sp1),_mm256_maddubs_epi16(rhs_mat_0145_01_sp1, lhs_mat_01_01_sp1));
                    __m256i iacc_mat_01_0_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_00_sp1, lhs_mat_01_00_sp1),_mm256_maddubs_epi16(rhs_mat_2367_01_sp1, lhs_mat_01_01_sp1));

                    __m256i iacc_mat_10_0_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_00_sp1, lhs_mat_23_00_sp1),_mm256_maddubs_epi16(rhs_mat_0145_01_sp1, lhs_mat_23_01_sp1));
                    __m256i iacc_mat_11_0_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_00_sp1, lhs_mat_23_00_sp1),_mm256_maddubs_epi16(rhs_mat_2367_01_sp1, lhs_mat_23_01_sp1));

                    __m256i iacc_mat_00_1_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_10_sp1, lhs_mat_01_10_sp1),_mm256_maddubs_epi16(rhs_mat_0145_11_sp1, lhs_mat_01_11_sp1));
                    __m256i iacc_mat_01_1_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_10_sp1, lhs_mat_01_10_sp1),_mm256_maddubs_epi16(rhs_mat_2367_11_sp1, lhs_mat_01_11_sp1));

                    __m256i iacc_mat_10_1_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_10_sp1, lhs_mat_23_10_sp1),_mm256_maddubs_epi16(rhs_mat_0145_11_sp1, lhs_mat_23_11_sp1));
                    __m256i iacc_mat_11_1_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_10_sp1, lhs_mat_23_10_sp1),_mm256_maddubs_epi16(rhs_mat_2367_11_sp1, lhs_mat_23_11_sp1));

                    __m256i iacc_mat_00_2_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_20_sp1, lhs_mat_01_20_sp1),_mm256_maddubs_epi16(rhs_mat_0145_21_sp1, lhs_mat_01_21_sp1));
                    __m256i iacc_mat_01_2_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_20_sp1, lhs_mat_01_20_sp1),_mm256_maddubs_epi16(rhs_mat_2367_21_sp1, lhs_mat_01_21_sp1));

                    __m256i iacc_mat_10_2_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_20_sp1, lhs_mat_23_20_sp1),_mm256_maddubs_epi16(rhs_mat_0145_21_sp1, lhs_mat_23_21_sp1));
                    __m256i iacc_mat_11_2_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_20_sp1, lhs_mat_23_20_sp1),_mm256_maddubs_epi16(rhs_mat_2367_21_sp1, lhs_mat_23_21_sp1));

                    __m256i iacc_mat_00_3_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_30_sp1, lhs_mat_01_30_sp1),_mm256_maddubs_epi16(rhs_mat_0145_31_sp1, lhs_mat_01_31_sp1));
                    __m256i iacc_mat_01_3_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_30_sp1, lhs_mat_01_30_sp1),_mm256_maddubs_epi16(rhs_mat_2367_31_sp1, lhs_mat_01_31_sp1));

                    __m256i iacc_mat_10_3_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_30_sp1, lhs_mat_23_30_sp1),_mm256_maddubs_epi16(rhs_mat_0145_31_sp1, lhs_mat_23_31_sp1));
                    __m256i iacc_mat_11_3_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_30_sp1, lhs_mat_23_30_sp1),_mm256_maddubs_epi16(rhs_mat_2367_31_sp1, lhs_mat_23_31_sp1));

                    __m256i iacc_mat_00_4_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_40_sp1, lhs_mat_01_40_sp1),_mm256_maddubs_epi16(rhs_mat_0145_41_sp1, lhs_mat_01_41_sp1));
                    __m256i iacc_mat_01_4_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_40_sp1, lhs_mat_01_40_sp1),_mm256_maddubs_epi16(rhs_mat_2367_41_sp1, lhs_mat_01_41_sp1));

                    __m256i iacc_mat_10_4_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_40_sp1, lhs_mat_23_40_sp1),_mm256_maddubs_epi16(rhs_mat_0145_41_sp1, lhs_mat_23_41_sp1));
                    __m256i iacc_mat_11_4_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_40_sp1, lhs_mat_23_40_sp1),_mm256_maddubs_epi16(rhs_mat_2367_41_sp1, lhs_mat_23_41_sp1));

                    __m256i iacc_mat_00_5_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_50_sp1, lhs_mat_01_50_sp1),_mm256_maddubs_epi16(rhs_mat_0145_51_sp1, lhs_mat_01_51_sp1));
                    __m256i iacc_mat_01_5_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_50_sp1, lhs_mat_01_50_sp1),_mm256_maddubs_epi16(rhs_mat_2367_51_sp1, lhs_mat_01_51_sp1));

                    __m256i iacc_mat_10_5_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_50_sp1, lhs_mat_23_50_sp1),_mm256_maddubs_epi16(rhs_mat_0145_51_sp1, lhs_mat_23_51_sp1));
                    __m256i iacc_mat_11_5_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_50_sp1, lhs_mat_23_50_sp1),_mm256_maddubs_epi16(rhs_mat_2367_51_sp1, lhs_mat_23_51_sp1));

                    __m256i iacc_mat_00_6_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_60_sp1, lhs_mat_01_60_sp1),_mm256_maddubs_epi16(rhs_mat_0145_61_sp1, lhs_mat_01_61_sp1));
                    __m256i iacc_mat_01_6_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_60_sp1, lhs_mat_01_60_sp1),_mm256_maddubs_epi16(rhs_mat_2367_61_sp1, lhs_mat_01_61_sp1));

                    __m256i iacc_mat_10_6_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_60_sp1, lhs_mat_23_60_sp1),_mm256_maddubs_epi16(rhs_mat_0145_61_sp1, lhs_mat_23_61_sp1));
                    __m256i iacc_mat_11_6_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_60_sp1, lhs_mat_23_60_sp1),_mm256_maddubs_epi16(rhs_mat_2367_61_sp1, lhs_mat_23_61_sp1));

                    __m256i iacc_mat_00_7_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_70_sp1, lhs_mat_01_70_sp1),_mm256_maddubs_epi16(rhs_mat_0145_71_sp1, lhs_mat_01_71_sp1));
                    __m256i iacc_mat_01_7_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_70_sp1, lhs_mat_01_70_sp1),_mm256_maddubs_epi16(rhs_mat_2367_71_sp1, lhs_mat_01_71_sp1));

                    __m256i iacc_mat_10_7_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_70_sp1, lhs_mat_23_70_sp1),_mm256_maddubs_epi16(rhs_mat_0145_71_sp1, lhs_mat_23_71_sp1));
                    __m256i iacc_mat_11_7_sp1 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_70_sp1, lhs_mat_23_70_sp1),_mm256_maddubs_epi16(rhs_mat_2367_71_sp1, lhs_mat_23_71_sp1));


                    __m256i iacc_mat_00_0_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_00_sp2, lhs_mat_01_00_sp2),_mm256_maddubs_epi16(rhs_mat_0145_01_sp2, lhs_mat_01_01_sp2));
                    __m256i iacc_mat_01_0_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_00_sp2, lhs_mat_01_00_sp2),_mm256_maddubs_epi16(rhs_mat_2367_01_sp2, lhs_mat_01_01_sp2));

                    __m256i iacc_mat_10_0_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_00_sp2, lhs_mat_23_00_sp2),_mm256_maddubs_epi16(rhs_mat_0145_01_sp2, lhs_mat_23_01_sp2));
                    __m256i iacc_mat_11_0_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_00_sp2, lhs_mat_23_00_sp2),_mm256_maddubs_epi16(rhs_mat_2367_01_sp2, lhs_mat_23_01_sp2));

                    __m256i iacc_mat_00_1_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_10_sp2, lhs_mat_01_10_sp2),_mm256_maddubs_epi16(rhs_mat_0145_11_sp2, lhs_mat_01_11_sp2));
                    __m256i iacc_mat_01_1_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_10_sp2, lhs_mat_01_10_sp2),_mm256_maddubs_epi16(rhs_mat_2367_11_sp2, lhs_mat_01_11_sp2));

                    __m256i iacc_mat_10_1_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_10_sp2, lhs_mat_23_10_sp2),_mm256_maddubs_epi16(rhs_mat_0145_11_sp2, lhs_mat_23_11_sp2));
                    __m256i iacc_mat_11_1_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_10_sp2, lhs_mat_23_10_sp2),_mm256_maddubs_epi16(rhs_mat_2367_11_sp2, lhs_mat_23_11_sp2));

                    __m256i iacc_mat_00_2_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_20_sp2, lhs_mat_01_20_sp2),_mm256_maddubs_epi16(rhs_mat_0145_21_sp2, lhs_mat_01_21_sp2));
                    __m256i iacc_mat_01_2_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_20_sp2, lhs_mat_01_20_sp2),_mm256_maddubs_epi16(rhs_mat_2367_21_sp2, lhs_mat_01_21_sp2));

                    __m256i iacc_mat_10_2_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_20_sp2, lhs_mat_23_20_sp2),_mm256_maddubs_epi16(rhs_mat_0145_21_sp2, lhs_mat_23_21_sp2));
                    __m256i iacc_mat_11_2_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_20_sp2, lhs_mat_23_20_sp2),_mm256_maddubs_epi16(rhs_mat_2367_21_sp2, lhs_mat_23_21_sp2));

                    __m256i iacc_mat_00_3_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_30_sp2, lhs_mat_01_30_sp2),_mm256_maddubs_epi16(rhs_mat_0145_31_sp2, lhs_mat_01_31_sp2));
                    __m256i iacc_mat_01_3_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_30_sp2, lhs_mat_01_30_sp2),_mm256_maddubs_epi16(rhs_mat_2367_31_sp2, lhs_mat_01_31_sp2));

                    __m256i iacc_mat_10_3_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_30_sp2, lhs_mat_23_30_sp2),_mm256_maddubs_epi16(rhs_mat_0145_31_sp2, lhs_mat_23_31_sp2));
                    __m256i iacc_mat_11_3_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_30_sp2, lhs_mat_23_30_sp2),_mm256_maddubs_epi16(rhs_mat_2367_31_sp2, lhs_mat_23_31_sp2));

                    __m256i iacc_mat_00_4_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_40_sp2, lhs_mat_01_40_sp2),_mm256_maddubs_epi16(rhs_mat_0145_41_sp2, lhs_mat_01_41_sp2));
                    __m256i iacc_mat_01_4_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_40_sp2, lhs_mat_01_40_sp2),_mm256_maddubs_epi16(rhs_mat_2367_41_sp2, lhs_mat_01_41_sp2));

                    __m256i iacc_mat_10_4_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_40_sp2, lhs_mat_23_40_sp2),_mm256_maddubs_epi16(rhs_mat_0145_41_sp2, lhs_mat_23_41_sp2));
                    __m256i iacc_mat_11_4_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_40_sp2, lhs_mat_23_40_sp2),_mm256_maddubs_epi16(rhs_mat_2367_41_sp2, lhs_mat_23_41_sp2));

                    __m256i iacc_mat_00_5_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_50_sp2, lhs_mat_01_50_sp2),_mm256_maddubs_epi16(rhs_mat_0145_51_sp2, lhs_mat_01_51_sp2));
                    __m256i iacc_mat_01_5_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_50_sp2, lhs_mat_01_50_sp2),_mm256_maddubs_epi16(rhs_mat_2367_51_sp2, lhs_mat_01_51_sp2));

                    __m256i iacc_mat_10_5_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_50_sp2, lhs_mat_23_50_sp2),_mm256_maddubs_epi16(rhs_mat_0145_51_sp2, lhs_mat_23_51_sp2));
                    __m256i iacc_mat_11_5_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_50_sp2, lhs_mat_23_50_sp2),_mm256_maddubs_epi16(rhs_mat_2367_51_sp2, lhs_mat_23_51_sp2));

                    __m256i iacc_mat_00_6_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_60_sp2, lhs_mat_01_60_sp2),_mm256_maddubs_epi16(rhs_mat_0145_61_sp2, lhs_mat_01_61_sp2));
                    __m256i iacc_mat_01_6_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_60_sp2, lhs_mat_01_60_sp2),_mm256_maddubs_epi16(rhs_mat_2367_61_sp2, lhs_mat_01_61_sp2));

                    __m256i iacc_mat_10_6_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_60_sp2, lhs_mat_23_60_sp2),_mm256_maddubs_epi16(rhs_mat_0145_61_sp2, lhs_mat_23_61_sp2));
                    __m256i iacc_mat_11_6_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_60_sp2, lhs_mat_23_60_sp2),_mm256_maddubs_epi16(rhs_mat_2367_61_sp2, lhs_mat_23_61_sp2));

                    __m256i iacc_mat_00_7_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_70_sp2, lhs_mat_01_70_sp2),_mm256_maddubs_epi16(rhs_mat_0145_71_sp2, lhs_mat_01_71_sp2));
                    __m256i iacc_mat_01_7_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_70_sp2, lhs_mat_01_70_sp2),_mm256_maddubs_epi16(rhs_mat_2367_71_sp2, lhs_mat_01_71_sp2));

                    __m256i iacc_mat_10_7_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_70_sp2, lhs_mat_23_70_sp2),_mm256_maddubs_epi16(rhs_mat_0145_71_sp2, lhs_mat_23_71_sp2));
                    __m256i iacc_mat_11_7_sp2 = _mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_70_sp2, lhs_mat_23_70_sp2),_mm256_maddubs_epi16(rhs_mat_2367_71_sp2, lhs_mat_23_71_sp2));

                    // Combine results from both shuffle patterns for each output block.
                    __m256i iacc_mat_00_0 = _mm256_add_epi16(iacc_mat_00_0_sp1, iacc_mat_00_0_sp2);
                    __m256i iacc_mat_01_0 = _mm256_add_epi16(iacc_mat_01_0_sp1, iacc_mat_01_0_sp2);
                    __m256i iacc_mat_10_0 = _mm256_add_epi16(iacc_mat_10_0_sp1, iacc_mat_10_0_sp2);
                    __m256i iacc_mat_11_0 = _mm256_add_epi16(iacc_mat_11_0_sp1, iacc_mat_11_0_sp2);

                    __m256i iacc_mat_00_1 = _mm256_add_epi16(iacc_mat_00_1_sp1, iacc_mat_00_1_sp2);
                    __m256i iacc_mat_01_1 = _mm256_add_epi16(iacc_mat_01_1_sp1, iacc_mat_01_1_sp2);
                    __m256i iacc_mat_10_1 = _mm256_add_epi16(iacc_mat_10_1_sp1, iacc_mat_10_1_sp2);
                    __m256i iacc_mat_11_1 = _mm256_add_epi16(iacc_mat_11_1_sp1, iacc_mat_11_1_sp2);

                    __m256i iacc_mat_00_2 = _mm256_add_epi16(iacc_mat_00_2_sp1, iacc_mat_00_2_sp2);
                    __m256i iacc_mat_01_2 = _mm256_add_epi16(iacc_mat_01_2_sp1, iacc_mat_01_2_sp2);
                    __m256i iacc_mat_10_2 = _mm256_add_epi16(iacc_mat_10_2_sp1, iacc_mat_10_2_sp2);
                    __m256i iacc_mat_11_2 = _mm256_add_epi16(iacc_mat_11_2_sp1, iacc_mat_11_2_sp2);

                    __m256i iacc_mat_00_3 = _mm256_add_epi16(iacc_mat_00_3_sp1, iacc_mat_00_3_sp2);
                    __m256i iacc_mat_01_3 = _mm256_add_epi16(iacc_mat_01_3_sp1, iacc_mat_01_3_sp2);
                    __m256i iacc_mat_10_3 = _mm256_add_epi16(iacc_mat_10_3_sp1, iacc_mat_10_3_sp2);
                    __m256i iacc_mat_11_3 = _mm256_add_epi16(iacc_mat_11_3_sp1, iacc_mat_11_3_sp2);

                    __m256i iacc_mat_00_4 = _mm256_add_epi16(iacc_mat_00_4_sp1, iacc_mat_00_4_sp2);
                    __m256i iacc_mat_01_4 = _mm256_add_epi16(iacc_mat_01_4_sp1, iacc_mat_01_4_sp2);
                    __m256i iacc_mat_10_4 = _mm256_add_epi16(iacc_mat_10_4_sp1, iacc_mat_10_4_sp2);
                    __m256i iacc_mat_11_4 = _mm256_add_epi16(iacc_mat_11_4_sp1, iacc_mat_11_4_sp2);

                    __m256i iacc_mat_00_5 = _mm256_add_epi16(iacc_mat_00_5_sp1, iacc_mat_00_5_sp2);
                    __m256i iacc_mat_01_5 = _mm256_add_epi16(iacc_mat_01_5_sp1, iacc_mat_01_5_sp2);
                    __m256i iacc_mat_10_5 = _mm256_add_epi16(iacc_mat_10_5_sp1, iacc_mat_10_5_sp2);
                    __m256i iacc_mat_11_5 = _mm256_add_epi16(iacc_mat_11_5_sp1, iacc_mat_11_5_sp2);

                    __m256i iacc_mat_00_6 = _mm256_add_epi16(iacc_mat_00_6_sp1, iacc_mat_00_6_sp2);
                    __m256i iacc_mat_01_6 = _mm256_add_epi16(iacc_mat_01_6_sp1, iacc_mat_01_6_sp2);
                    __m256i iacc_mat_10_6 = _mm256_add_epi16(iacc_mat_10_6_sp1, iacc_mat_10_6_sp2);
                    __m256i iacc_mat_11_6 = _mm256_add_epi16(iacc_mat_11_6_sp1, iacc_mat_11_6_sp2);

                    __m256i iacc_mat_00_7 = _mm256_add_epi16(iacc_mat_00_7_sp1, iacc_mat_00_7_sp2);
                    __m256i iacc_mat_01_7 = _mm256_add_epi16(iacc_mat_01_7_sp1, iacc_mat_01_7_sp2);
                    __m256i iacc_mat_10_7 = _mm256_add_epi16(iacc_mat_10_7_sp1, iacc_mat_10_7_sp2);
                    __m256i iacc_mat_11_7 = _mm256_add_epi16(iacc_mat_11_7_sp1, iacc_mat_11_7_sp2);

                    // Output of both shuffle patterns are added in order to sum dot product outputs of all 32 values in block
                    iacc_mat_00_0 = _mm256_madd_epi16(iacc_mat_00_0, scale_0145_0);
                    iacc_mat_01_0 = _mm256_madd_epi16(iacc_mat_01_0, scale_2367_0);
                    iacc_mat_10_0 = _mm256_madd_epi16(iacc_mat_10_0, scale_0145_0);
                    iacc_mat_11_0 = _mm256_madd_epi16(iacc_mat_11_0, scale_2367_0);

                    iacc_mat_00_1 = _mm256_madd_epi16(iacc_mat_00_1, scale_0145_1);
                    iacc_mat_01_1 = _mm256_madd_epi16(iacc_mat_01_1, scale_2367_1);
                    iacc_mat_10_1 = _mm256_madd_epi16(iacc_mat_10_1, scale_0145_1);
                    iacc_mat_11_1 = _mm256_madd_epi16(iacc_mat_11_1, scale_2367_1);

                    iacc_mat_00_2 = _mm256_madd_epi16(iacc_mat_00_2, scale_0145_2);
                    iacc_mat_01_2 = _mm256_madd_epi16(iacc_mat_01_2, scale_2367_2);
                    iacc_mat_10_2 = _mm256_madd_epi16(iacc_mat_10_2, scale_0145_2);
                    iacc_mat_11_2 = _mm256_madd_epi16(iacc_mat_11_2, scale_2367_2);

                    iacc_mat_00_3 = _mm256_madd_epi16(iacc_mat_00_3, scale_0145_3);
                    iacc_mat_01_3 = _mm256_madd_epi16(iacc_mat_01_3, scale_2367_3);
                    iacc_mat_10_3 = _mm256_madd_epi16(iacc_mat_10_3, scale_0145_3);
                    iacc_mat_11_3 = _mm256_madd_epi16(iacc_mat_11_3, scale_2367_3);

                    iacc_mat_00_4 = _mm256_madd_epi16(iacc_mat_00_4, scale_0145_4);
                    iacc_mat_01_4 = _mm256_madd_epi16(iacc_mat_01_4, scale_2367_4);
                    iacc_mat_10_4 = _mm256_madd_epi16(iacc_mat_10_4, scale_0145_4);
                    iacc_mat_11_4 = _mm256_madd_epi16(iacc_mat_11_4, scale_2367_4);

                    iacc_mat_00_5 = _mm256_madd_epi16(iacc_mat_00_5, scale_0145_5);
                    iacc_mat_01_5 = _mm256_madd_epi16(iacc_mat_01_5, scale_2367_5);
                    iacc_mat_10_5 = _mm256_madd_epi16(iacc_mat_10_5, scale_0145_5);
                    iacc_mat_11_5 = _mm256_madd_epi16(iacc_mat_11_5, scale_2367_5);

                    iacc_mat_00_6 = _mm256_madd_epi16(iacc_mat_00_6, scale_0145_6);
                    iacc_mat_01_6 = _mm256_madd_epi16(iacc_mat_01_6, scale_2367_6);
                    iacc_mat_10_6 = _mm256_madd_epi16(iacc_mat_10_6, scale_0145_6);
                    iacc_mat_11_6 = _mm256_madd_epi16(iacc_mat_11_6, scale_2367_6);

                    iacc_mat_00_7 = _mm256_madd_epi16(iacc_mat_00_7, scale_0145_7);
                    iacc_mat_01_7 = _mm256_madd_epi16(iacc_mat_01_7, scale_2367_7);
                    iacc_mat_10_7 = _mm256_madd_epi16(iacc_mat_10_7, scale_0145_7);
                    iacc_mat_11_7 = _mm256_madd_epi16(iacc_mat_11_7, scale_2367_7);

                    __m256i iacc_mat_00 = _mm256_add_epi32(_mm256_add_epi32(_mm256_add_epi32(iacc_mat_00_0, iacc_mat_00_1), _mm256_add_epi32(iacc_mat_00_2, iacc_mat_00_3)), _mm256_add_epi32(_mm256_add_epi32(iacc_mat_00_4, iacc_mat_00_5), _mm256_add_epi32(iacc_mat_00_6, iacc_mat_00_7)));
                    __m256i iacc_mat_01 = _mm256_add_epi32(_mm256_add_epi32(_mm256_add_epi32(iacc_mat_01_0, iacc_mat_01_1), _mm256_add_epi32(iacc_mat_01_2, iacc_mat_01_3)), _mm256_add_epi32(_mm256_add_epi32(iacc_mat_01_4, iacc_mat_01_5), _mm256_add_epi32(iacc_mat_01_6, iacc_mat_01_7)));
                    __m256i iacc_mat_10 = _mm256_add_epi32(_mm256_add_epi32(_mm256_add_epi32(iacc_mat_10_0, iacc_mat_10_1), _mm256_add_epi32(iacc_mat_10_2, iacc_mat_10_3)), _mm256_add_epi32(_mm256_add_epi32(iacc_mat_10_4, iacc_mat_10_5), _mm256_add_epi32(iacc_mat_10_6, iacc_mat_10_7)));
                    __m256i iacc_mat_11 = _mm256_add_epi32(_mm256_add_epi32(_mm256_add_epi32(iacc_mat_11_0, iacc_mat_11_1), _mm256_add_epi32(iacc_mat_11_2, iacc_mat_11_3)), _mm256_add_epi32(_mm256_add_epi32(iacc_mat_11_4, iacc_mat_11_5), _mm256_add_epi32(iacc_mat_11_6, iacc_mat_11_7)));

                    // Straighten out to make 4 row vectors
                    __m256i iacc_row_0 = _mm256_blend_epi32(iacc_mat_00, _mm256_shuffle_epi32(iacc_mat_01, 78), 204);
                    __m256i iacc_row_1 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_00, 78), iacc_mat_01, 204);
                    __m256i iacc_row_2 = _mm256_blend_epi32(iacc_mat_10, _mm256_shuffle_epi32(iacc_mat_11, 78), 204);
                    __m256i iacc_row_3 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_10, 78), iacc_mat_11, 204);

                    // Load the scale(d) values for all the 4 Q8_k blocks and repeat it across lanes
                    const __m128 row_scale_f32_sse = _mm_load_ps(a_ptr[b].d);
                    const __m256 row_scale_f32 = _mm256_set_m128(row_scale_f32_sse, row_scale_f32_sse);

                    // Multiply with appropriate scales and accumulate (for both d and dmin) below
                    acc_rows[0] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_0), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 0)), acc_rows[0]);
                    acc_rows[1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_1), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 85)), acc_rows[1]);
                    acc_rows[2] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_2), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_rows[2]);
                    acc_rows[3] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_3), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_rows[3]);

                    __m256i lhs_bsums_01_0123 = _mm256_inserti128_si256(_mm256_castsi128_si256(lhs_raw_bsums_01_0123), lhs_raw_bsums_01_0123, 1);
                    __m256i lhs_bsums_23_0123 = _mm256_inserti128_si256(_mm256_castsi128_si256(lhs_raw_bsums_23_0123), lhs_raw_bsums_23_0123, 1);
                    __m256i lhs_bsums_01_4567 = _mm256_inserti128_si256(_mm256_castsi128_si256(lhs_raw_bsums_01_4567), lhs_raw_bsums_01_4567, 1);
                    __m256i lhs_bsums_23_4567 = _mm256_inserti128_si256(_mm256_castsi128_si256(lhs_raw_bsums_23_4567), lhs_raw_bsums_23_4567, 1);

                    // Take two bsums from two Q8_Ks at a time and multiply with corresponding mins values from each Q2_K
                    __m256i iacc_row_min_0_01 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_01_0123, 0), mins_01);
                    __m256i iacc_row_min_1_01 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_01_0123, 170), mins_01);
                    __m256i iacc_row_min_2_01 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_23_0123, 0), mins_01);
                    __m256i iacc_row_min_3_01 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_23_0123, 170), mins_01);

                    __m256i iacc_row_min_0_23 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_01_0123, 85), mins_23);
                    __m256i iacc_row_min_1_23 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_01_0123, 255), mins_23);
                    __m256i iacc_row_min_2_23 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_23_0123, 85), mins_23);
                    __m256i iacc_row_min_3_23 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_23_0123, 255), mins_23);

                    __m256i iacc_row_min_0_45 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_01_4567, 0), mins_45);
                    __m256i iacc_row_min_1_45 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_01_4567, 170), mins_45);
                    __m256i iacc_row_min_2_45 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_23_4567, 0), mins_45);
                    __m256i iacc_row_min_3_45 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_23_4567, 170), mins_45);

                    __m256i iacc_row_min_0_67 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_01_4567, 85), mins_67);
                    __m256i iacc_row_min_1_67 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_01_4567, 255), mins_67);
                    __m256i iacc_row_min_2_67 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_23_4567, 85), mins_67);
                    __m256i iacc_row_min_3_67 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_23_4567, 255), mins_67);

                    __m256i iacc_row_min_0 = _mm256_add_epi32(_mm256_add_epi32(iacc_row_min_0_01, iacc_row_min_0_23), _mm256_add_epi32(iacc_row_min_0_45,iacc_row_min_0_67));
                    __m256i iacc_row_min_1 = _mm256_add_epi32(_mm256_add_epi32(iacc_row_min_1_01, iacc_row_min_1_23), _mm256_add_epi32(iacc_row_min_1_45,iacc_row_min_1_67));
                    __m256i iacc_row_min_2 = _mm256_add_epi32(_mm256_add_epi32(iacc_row_min_2_01, iacc_row_min_2_23), _mm256_add_epi32(iacc_row_min_2_45,iacc_row_min_2_67));
                    __m256i iacc_row_min_3 = _mm256_add_epi32(_mm256_add_epi32(iacc_row_min_3_01, iacc_row_min_3_23), _mm256_add_epi32(iacc_row_min_3_45,iacc_row_min_3_67));

                    acc_min_rows[0] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_0), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 0)), acc_min_rows[0]);
                    acc_min_rows[1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_1), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 85)), acc_min_rows[1]);
                    acc_min_rows[2] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_2), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_min_rows[2]);
                    acc_min_rows[3] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_3), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_min_rows[3]);
                }
            }
            // Store the accumulated values
            for (int i = 0; i < 4; i++) {
                _mm256_storeu_ps((float * )(s + ((y * 4 + i) * bs + x * 8)), _mm256_sub_ps(acc_rows[i], acc_min_rows[i]));
            }
        }
    }
#else

    ggml_gemm_q2_K_8x8_q8_K_generic(n, s, bs, vx, vy, nr, nc);


#endif
}

// =====================================================================================
// x16 kernels: 16 rows per 64-byte vector, AVX-512 VNNI, one row per int32 lane.
// =====================================================================================
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
#define GGML_X16_BC32(p) _mm512_set1_epi32(*(const int32_t *)(p))
#endif

void ggml_gemv_q4_K_x16_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
    GGML_ASSERT(nr == 1 && n % QK_K == 0 && nc % 16 == 0);
    GGML_UNUSED(bs);
    const int nb = n / QK_K; const __m512i m4 = _mm512_set1_epi8(0x0F);
    const block_q4_K_x16 * vxb = (const block_q4_K_x16 *) vx; const block_q8_K * vy8 = (const block_q8_K *) vy;
    for (int g = 0; g < nc / 16; g++) { const block_q4_K_x16 * bp = vxb + (size_t) g * nb; __m512 accf = _mm512_setzero_ps();
        for (int b = 0; b < nb; b++) { const block_q8_K * a = vy8 + b; const uint8_t * qs = bp[b].qs;
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


static void ggml_gemv_q5_K_x16_pair_q8_K(
        int n, float * s, size_t bs, const void * vx, const void * vy, int nc) {
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
    GGML_ASSERT(n % QK_K == 0 && nc % 16 == 0);
    const int nb = n / QK_K;
    const auto * weights = (const block_q5_K_x16 *) vx;
    const auto * activations = (const block_q8_K *) vy;
    const __m512i m4 = _mm512_set1_epi8(15);
    const __m512i m10 = _mm512_set1_epi8(16);
    for (int g = 0; g < nc / 16; ++g) {
        const block_q5_K_x16 * w = weights + g * nb;
        __m512 sum[2] = {_mm512_setzero_ps(), _mm512_setzero_ps()};
        for (int b = 0; b < nb; ++b) {
            const block_q8_K * a0 = activations + b;
            const block_q8_K * a1 = activations + nb + b;
            __m512i dots[2][8];
            for (int y = 0; y < 2; ++y) for (int sb = 0; sb < 8; ++sb) dots[y][sb] = _mm512_setzero_si512();
            for (int chunk = 0; chunk < 8; ++chunk) {
                const uint8_t * q = w[b].qsh + chunk * 320;
                const __m512i h = _mm512_loadu_si512(q + 256);
#define Q5_PAIR_DOT(sb, weights) { \
                const __m512i v = (weights); \
                dots[0][sb] = _mm512_dpbusd_epi32(dots[0][sb], v, GGML_X16_BC32(a0->qs + 32*(sb) + 4*chunk)); \
                dots[1][sb] = _mm512_dpbusd_epi32(dots[1][sb], v, GGML_X16_BC32(a1->qs + 32*(sb) + 4*chunk)); }
                const __m512i v0 = _mm512_loadu_si512(q);
                Q5_PAIR_DOT(0, _mm512_or_si512(_mm512_and_si512(v0, m4), _mm512_and_si512(_mm512_slli_epi16(h, 4), m10)))
                Q5_PAIR_DOT(1, _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(v0, 4), m4), _mm512_and_si512(_mm512_slli_epi16(h, 3), m10)))
                const __m512i v1 = _mm512_loadu_si512(q + 64);
                Q5_PAIR_DOT(2, _mm512_or_si512(_mm512_and_si512(v1, m4), _mm512_and_si512(_mm512_slli_epi16(h, 2), m10)))
                Q5_PAIR_DOT(3, _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(v1, 4), m4), _mm512_and_si512(_mm512_slli_epi16(h, 1), m10)))
                const __m512i v2 = _mm512_loadu_si512(q + 128);
                Q5_PAIR_DOT(4, _mm512_or_si512(_mm512_and_si512(v2, m4), _mm512_and_si512(h, m10)))
                Q5_PAIR_DOT(5, _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(v2, 4), m4), _mm512_and_si512(_mm512_srli_epi16(h, 1), m10)))
                const __m512i v3 = _mm512_loadu_si512(q + 192);
                Q5_PAIR_DOT(6, _mm512_or_si512(_mm512_and_si512(v3, m4), _mm512_and_si512(_mm512_srli_epi16(h, 2), m10)))
                Q5_PAIR_DOT(7, _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(v3, 4), m4), _mm512_and_si512(_mm512_srli_epi16(h, 3), m10)))
#undef Q5_PAIR_DOT
            }
            for (int y = 0; y < 2; ++y) {
                const block_q8_K * a = y == 0 ? a0 : a1;
                __m512i isum = _mm512_setzero_si512(), imin = _mm512_setzero_si512();
                for (int sb = 0; sb < 8; ++sb) {
                    const __m512i scale = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *) w[b].scales[sb]));
                    const __m512i min = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *) w[b].mins[sb]));
                    isum = _mm512_add_epi32(isum, _mm512_mullo_epi32(dots[y][sb], scale));
                    imin = _mm512_add_epi32(imin, _mm512_mullo_epi32(min, _mm512_set1_epi32(int(a->bsums[2*sb]) + a->bsums[2*sb+1])));
                }
                const __m512 ad = _mm512_set1_ps(a->d);
                const __m512 d = _mm512_mul_ps(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) w[b].d)), ad);
                const __m512 dmin = _mm512_mul_ps(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) w[b].dmin)), ad);
                sum[y] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(isum), d, sum[y]);
                sum[y] = _mm512_fnmadd_ps(_mm512_cvtepi32_ps(imin), dmin, sum[y]);
            }
        }
        _mm512_storeu_ps(s + g*16, sum[0]);
        _mm512_storeu_ps(s + bs + g*16, sum[1]);
    }
#else
    const size_t row_bytes = (n / QK_K) * sizeof(block_q8_K);
    ggml_gemv_q5_K_x16_q8_K_generic(n, s, 0, vx, vy, 1, nc);
    ggml_gemv_q5_K_x16_q8_K_generic(n, s + bs, 0, vx, (const char *) vy + row_bytes, 1, nc);
#endif
}

void ggml_gemv_q5_K_x16_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    if (nr == 2) {
        ggml_gemv_q5_K_x16_pair_q8_K(n, s, bs, vx, vy, nc);
        return;
    }

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
    GGML_ASSERT(nr == 1 && n % QK_K == 0 && nc % 16 == 0);
    GGML_UNUSED(bs);
    const int nb = n / QK_K; const __m512i m4 = _mm512_set1_epi8(0x0F); const __m512i m10 = _mm512_set1_epi8(0x10);
    const block_q5_K_x16 * vxb = (const block_q5_K_x16 *) vx; const block_q8_K * vy8 = (const block_q8_K *) vy;
    for (int g = 0; g < nc / 16; g++) { const block_q5_K_x16 * bp = vxb + (size_t) g * nb; __m512 accf = _mm512_setzero_ps();
        for (int b = 0; b < nb; b++) { const block_q8_K * a = vy8 + b; const uint8_t * q = bp[b].qsh; const int8_t * y = a->qs;
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

void ggml_gemv_q6_K_x16_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
    GGML_ASSERT(nr == 1 && n % QK_K == 0 && nc % 16 == 0);
    GGML_UNUSED(bs);
    const int nb = n / QK_K; const __m512i m4 = _mm512_set1_epi8(0x0F); const __m512i m30 = _mm512_set1_epi8(0x30);
    const block_q6_K_x16 * vxb = (const block_q6_K_x16 *) vx; const block_q8_K * vy8 = (const block_q8_K *) vy;
    for (int g = 0; g < nc / 16; g++) { const block_q6_K_x16 * bp = vxb + (size_t) g * nb; __m512 accf = _mm512_setzero_ps();
        for (int b = 0; b < nb; b++) { const block_q8_K * a = vy8 + b;
            __m512i iacc = _mm512_setzero_si512(), icorr = _mm512_setzero_si512();
            for (int h = 0; h < 2; h++) {
                const uint8_t * qh0 = bp[b].q + h * 1536; const int8_t * y = a->qs + 128 * h;
                __m512i C00 = _mm512_setzero_si512(), C01 = C00, C10 = C00, C11 = C00, C20 = C00, C21 = C00, C30 = C00, C31 = C00;
#define Q6STEP(i0, X0, X1, X2, X3) { const uint8_t * qi = qh0 + (i0) * 192; \
                const __m512i la = _mm512_loadu_si512((const void *)(qi)), lb = _mm512_loadu_si512((const void *)(qi + 64)), hv = _mm512_loadu_si512((const void *)(qi + 128)); \
                X0 = _mm512_dpbusd_epi32(X0, _mm512_or_si512(_mm512_and_si512(la, m4), _mm512_and_si512(_mm512_slli_epi16(hv, 4), m30)), GGML_X16_BC32(y + 4*(i0))); \
                X1 = _mm512_dpbusd_epi32(X1, _mm512_or_si512(_mm512_and_si512(lb, m4), _mm512_and_si512(_mm512_slli_epi16(hv, 2), m30)), GGML_X16_BC32(y + 32 + 4*(i0))); \
                X2 = _mm512_dpbusd_epi32(X2, _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(la, 4), m4), _mm512_and_si512(hv, m30)), GGML_X16_BC32(y + 64 + 4*(i0))); \
                X3 = _mm512_dpbusd_epi32(X3, _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(lb, 4), m4), _mm512_and_si512(_mm512_srli_epi16(hv, 2), m30)), GGML_X16_BC32(y + 96 + 4*(i0))); }
                Q6STEP(0, C00, C10, C20, C30) Q6STEP(1, C00, C10, C20, C30) Q6STEP(2, C00, C10, C20, C30) Q6STEP(3, C00, C10, C20, C30)
                Q6STEP(4, C01, C11, C21, C31) Q6STEP(5, C01, C11, C21, C31) Q6STEP(6, C01, C11, C21, C31) Q6STEP(7, C01, C11, C21, C31)
#undef Q6STEP
                const __m512i accs[8] = {C00, C01, C10, C11, C20, C21, C30, C31};
                for (int i = 0; i < 8; i++) {
                    const int sb = 8*h + i;
                    const __m512i sc = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *) bp[b].scales[sb]));
                    iacc  = _mm512_add_epi32(iacc,  _mm512_mullo_epi32(accs[i], sc));
                    icorr = _mm512_add_epi32(icorr, _mm512_mullo_epi32(sc, _mm512_set1_epi32((int32_t) a->bsums[sb])));
                }
            }
            const __m512i tot = _mm512_sub_epi32(iacc, _mm512_slli_epi32(icorr, 5));
            accf = _mm512_fmadd_ps(_mm512_cvtepi32_ps(tot), _mm512_mul_ps(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) bp[b].d)), _mm512_set1_ps(a->d)), accf);
        }
        _mm512_storeu_ps(s + g * 16, accf); }
    return;
#endif
    ggml_gemv_q6_K_x16_q8_K_generic(n, s, bs, vx, vy, nr, nc);
}

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
#include "qwen-q8-batch.h"
#endif

void ggml_gemv_q8_0_x16_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
    if (nr > 1) return qwen_q8_batch_dispatch(n, s, bs, vx, vy, nr, nc);
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
            for (int i0 = 0; i0 < 8; i0++) acc = _mm512_dpbusd_epi32(acc, _mm512_loadu_si512((const void *)(q0 + i0 * 64)), GGML_X16_BC32(y0 + 4*i0));
            acc = _mm512_sub_epi32(acc, _mm512_set1_epi32(128 * ysum[b]));
            accf = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), _mm512_mul_ps(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) bp[b].d)), _mm512_set1_ps(yd[b])), accf);
        }
        _mm512_storeu_ps(s + g * 16, accf); }
    return;
#endif
    ggml_gemv_q8_0_x16_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemv2_q4_K_x16_q8_K(int n, float * GGML_RESTRICT s0, float * GGML_RESTRICT s1, const void * GGML_RESTRICT vx0, const void * GGML_RESTRICT vx1, const void * GGML_RESTRICT vy, int nc) {
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
    GGML_ASSERT(n % QK_K == 0 && nc % 16 == 0);
    const int nb = n / QK_K; const __m512i m4 = _mm512_set1_epi8(0x0F);
    const block_q4_K_x16 * vg = (const block_q4_K_x16 *) vx0; const block_q4_K_x16 * vu = (const block_q4_K_x16 *) vx1;
    const block_q8_K * vy8 = (const block_q8_K *) vy;
    for (int g = 0; g < nc / 16; g++) {
        const block_q4_K_x16 * bg = vg + (size_t) g * nb; const block_q4_K_x16 * bu = vu + (size_t) g * nb;
        __m512 accfg = _mm512_setzero_ps(), accfu = _mm512_setzero_ps();
        for (int b = 0; b < nb; b++) { const block_q8_K * a = vy8 + b; const uint8_t * qsg = bg[b].qs; const uint8_t * qsu = bu[b].qs;
            __m512i iaccg = _mm512_setzero_si512(), iming = _mm512_setzero_si512(), iaccu = _mm512_setzero_si512(), iminu = _mm512_setzero_si512();
            for (int j = 0; j < 4; j++) {
                __m512i G0 = _mm512_setzero_si512(), G1 = G0, H0 = G0, H1 = G0, U0 = G0, U1 = G0, V0 = G0, V1 = G0;
                const uint8_t * qg = qsg + j * 512; const uint8_t * qu = qsu + j * 512; const int8_t * ya = a->qs + 64 * j; const int8_t * yb = ya + 32;
#define STEP2(i0, GG, HH, UU, VV) { const __m512i wg = _mm512_loadu_si512((const void *)(qg + (i0) * 64)); const __m512i wu = _mm512_loadu_si512((const void *)(qu + (i0) * 64)); \
                const __m512i ba = GGML_X16_BC32(ya + 4*(i0)); const __m512i bb = GGML_X16_BC32(yb + 4*(i0)); \
                GG = _mm512_dpbusd_epi32(GG, _mm512_and_si512(wg, m4), ba); HH = _mm512_dpbusd_epi32(HH, _mm512_and_si512(_mm512_srli_epi16(wg, 4), m4), bb); \
                UU = _mm512_dpbusd_epi32(UU, _mm512_and_si512(wu, m4), ba); VV = _mm512_dpbusd_epi32(VV, _mm512_and_si512(_mm512_srli_epi16(wu, 4), m4), bb); }
                STEP2(0,G0,H0,U0,V0) STEP2(1,G1,H1,U1,V1) STEP2(2,G0,H0,U0,V0) STEP2(3,G1,H1,U1,V1) STEP2(4,G0,H0,U0,V0) STEP2(5,G1,H1,U1,V1) STEP2(6,G0,H0,U0,V0) STEP2(7,G1,H1,U1,V1)
#undef STEP2
                const __m512i bsA = _mm512_set1_epi32((int32_t) a->bsums[4*j] + a->bsums[4*j+1]);
                const __m512i bsB = _mm512_set1_epi32((int32_t) a->bsums[4*j+2] + a->bsums[4*j+3]);
                iaccg = _mm512_add_epi32(iaccg, _mm512_mullo_epi32(_mm512_add_epi32(G0, G1), _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *) bg[b].scales[2*j]))));
                iaccg = _mm512_add_epi32(iaccg, _mm512_mullo_epi32(_mm512_add_epi32(H0, H1), _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *) bg[b].scales[2*j+1]))));
                iming = _mm512_add_epi32(iming, _mm512_mullo_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *) bg[b].mins[2*j])), bsA));
                iming = _mm512_add_epi32(iming, _mm512_mullo_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *) bg[b].mins[2*j+1])), bsB));
                iaccu = _mm512_add_epi32(iaccu, _mm512_mullo_epi32(_mm512_add_epi32(U0, U1), _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *) bu[b].scales[2*j]))));
                iaccu = _mm512_add_epi32(iaccu, _mm512_mullo_epi32(_mm512_add_epi32(V0, V1), _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *) bu[b].scales[2*j+1]))));
                iminu = _mm512_add_epi32(iminu, _mm512_mullo_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *) bu[b].mins[2*j])), bsA));
                iminu = _mm512_add_epi32(iminu, _mm512_mullo_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *) bu[b].mins[2*j+1])), bsB));
            }
            const __m512 ad = _mm512_set1_ps(a->d);
            accfg = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iaccg), _mm512_mul_ps(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) bg[b].d)), ad), accfg);
            accfg = _mm512_fnmadd_ps(_mm512_cvtepi32_ps(iming), _mm512_mul_ps(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) bg[b].dmin)), ad), accfg);
            accfu = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iaccu), _mm512_mul_ps(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) bu[b].d)), ad), accfu);
            accfu = _mm512_fnmadd_ps(_mm512_cvtepi32_ps(iminu), _mm512_mul_ps(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) bu[b].dmin)), ad), accfu);
        }
        _mm512_storeu_ps(s0 + g * 16, accfg); _mm512_storeu_ps(s1 + g * 16, accfu); }
    return;
#endif
    ggml_gemv_q4_K_x16_q8_K(n, s0, 0, vx0, vy, 1, nc);
    ggml_gemv_q4_K_x16_q8_K(n, s1, 0, vx1, vy, 1, nc);
}
