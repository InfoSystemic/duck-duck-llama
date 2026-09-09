#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml-backend-impl.h"

#include "ggml-impl.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include "simd-mappings.h"
#include "traits.h"
#include "vec.h"

#include "arch-fallback.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cstdio>  // for GGML_ASSERT
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include "repack.h"
#include "ggml-quants.h"

#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Woverlength-strings"
#endif

#define UNUSED GGML_UNUSED

static inline int nearest_int(float fval) {
    assert(fabsf(fval) <= 4194303.f);
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

// Functions to create the interleaved data layout formats

// interleave 4 block_q4_0s in blocks of blck_size_interleave
// returns an interleaved block_q4_0x4
// in the interleaved block_q4_0x4, place deltas for 4 block_q4_0 blocks
// first, then interleave quants from 4 block_q4_0s in blocks of blck_size_interleave
//
// - in                  : an array of block_q4_0 pointers
// - blck_size_interleave : the block_q4_0 quants bytes are interleaved in blocks of
//                         blck_size_interleave bytes
// - xor_mask            : the mask to convert the nibbles in block_q4_0 quants bytes
//                         from bias offset form to pure sign form (this saves subtract
//                         operations durin unpacking)
//

extern "C" {

#if defined __riscv_zvfh
void ggml_quantize_mat_q8_0_4x1_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    block_q8_0x4 * GGML_RESTRICT y = (block_q8_0x4 *) vy;

    // scalar
    const int blck_size_interleave = 1;
    float srcv[4][QK8_0];
    float id[4];

    for (int i = 0; i < nb; i++) {
        for (int row_iter = 0; row_iter < 4; row_iter++) {
            float amax = 0.0f; // absolute max

            for (int j = 0; j < QK8_0; j++) {
                srcv[row_iter][j] = x[row_iter * k + i * QK8_0 + j];
                amax = MAX(amax, fabsf(srcv[row_iter][j]));
            }

            const float d = amax / ((1 << 7) - 1);
            id[row_iter] = d ? 1.0f / d : 0.0f;

            y[i].d[row_iter] = GGML_CPU_FP32_TO_FP16(d);
        }

        for (int j = 0; j < QK8_0 * 4; j++) {
            int src_offset = (j / (4 * blck_size_interleave)) * blck_size_interleave;
            int src_id = (j % (4 * blck_size_interleave)) / blck_size_interleave;
            src_offset += (j % blck_size_interleave);

            float x0 = srcv[src_id][src_offset] * id[src_id];
            y[i].qs[j] = roundf(x0);
        }
    }
}

void ggml_quantize_mat_q8_K_4x1_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(QK_K == 256);
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    block_q8_Kx4 * GGML_RESTRICT y = (block_q8_Kx4 *) vy;

    const int blck_size_interleave = 1;
    float srcv[4][QK_K];
    float iscale[4];

    for (int i = 0; i < nb; i++) {
        for (int row_iter = 0; row_iter < 4; row_iter++) {
            float amax = 0.0f; // absolute max
            float max = 0;

            for (int j = 0; j < QK_K; j++) {
                srcv[row_iter][j] = x[row_iter * k + i * QK_K + j];
                // Update the maximum value of the corresponding super block
                if(amax < fabsf(srcv[row_iter][j])) {
                    amax = fabsf(srcv[row_iter][j]);
                    max = srcv[row_iter][j];
                }
            }

            iscale[row_iter] = amax ? -127.f/max : 0;
            y[i].d[row_iter] = amax ? 1/iscale[row_iter] : 0;
        }

        for (int j = 0; j < QK_K / 4; j++) {
            y[i].bsums[j] = 0;
        }
        for (int j = 0; j < QK_K * 4; j++) {
            int src_id = j % 4;
            int src_offset = j / 4;
            int index = ((j >> 6) << 2) + (j & 3);

            float x0 = srcv[src_id][src_offset] * iscale[src_id];
            y[i].qs[j] = nearest_int(x0);
            y[i].bsums[index] += y[i].qs[j];
        }
    }
}
#endif

void ggml_quantize_mat_q8_0_4x4_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    block_q8_0x4 * GGML_RESTRICT y = (block_q8_0x4 *) vy;

    // scalar
    const int blck_size_interleave = 4;
    float srcv[4][QK8_0];
    float id[4];

    for (int i = 0; i < nb; i++) {
        for (int row_iter = 0; row_iter < 4; row_iter++) {
            float amax = 0.0f; // absolute max

            for (int j = 0; j < QK8_0; j++) {
                srcv[row_iter][j] = x[row_iter * k + i * QK8_0 + j];
                amax = MAX(amax, fabsf(srcv[row_iter][j]));
            }

            const float d = amax / ((1 << 7) - 1);
            id[row_iter] = d ? 1.0f / d : 0.0f;

            y[i].d[row_iter] = GGML_CPU_FP32_TO_FP16(d);
        }

        for (int j = 0; j < QK8_0 * 4; j++) {
            int src_offset = (j / (4 * blck_size_interleave)) * blck_size_interleave;
            int src_id = (j % (4 * blck_size_interleave)) / blck_size_interleave;
            src_offset += (j % blck_size_interleave);

            float x0 = srcv[src_id][src_offset] * id[src_id];
            y[i].qs[j] = roundf(x0);
        }
    }
}

void ggml_quantize_mat_q8_0_4x8_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    block_q8_0x4 * GGML_RESTRICT y = (block_q8_0x4 *) vy;

    // scalar
    const int blck_size_interleave = 8;
    float srcv[4][QK8_0];
    float id[4];

    for (int i = 0; i < nb; i++) {
        for (int row_iter = 0; row_iter < 4; row_iter++) {
            float amax = 0.0f; // absolute max

            for (int j = 0; j < QK8_0; j++) {
                srcv[row_iter][j] = x[row_iter * k + i * QK8_0 + j];
                amax = MAX(amax, fabsf(srcv[row_iter][j]));
            }

            const float d = amax / ((1 << 7) - 1);
            id[row_iter] = d ? 1.0f / d : 0.0f;

            y[i].d[row_iter] = GGML_CPU_FP32_TO_FP16(d);
        }

        for (int j = 0; j < QK8_0 * 4; j++) {
            int src_offset = (j / (4 * blck_size_interleave)) * blck_size_interleave;
            int src_id = (j % (4 * blck_size_interleave)) / blck_size_interleave;
            src_offset += (j % blck_size_interleave);

            float x0 = srcv[src_id][src_offset] * id[src_id];
            y[i].qs[j] = roundf(x0);
        }
    }
}

void ggml_quantize_mat_q8_K_4x4_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(QK_K == 256);
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    block_q8_Kx4 * GGML_RESTRICT y = (block_q8_Kx4 *) vy;

    // scalar
    const int blck_size_interleave = 4;
    float srcv[4][QK_K];
    float iscale[4];

    for (int i = 0; i < nb; i++) {
        for (int row_iter = 0; row_iter < 4; row_iter++) {
            float amax = 0.0f; // absolute max
            float max = 0;

            for (int j = 0; j < QK_K; j++) {
                srcv[row_iter][j] = x[row_iter * k + i * QK_K + j];
                // Update the maximum value of the corresponding super block
                if(amax < fabsf(srcv[row_iter][j])) {
                    amax = fabsf(srcv[row_iter][j]);
                    max = srcv[row_iter][j];
                }
            }

            iscale[row_iter] = amax ? -127.f/max : 0;

            y[i].d[row_iter] = amax ? 1/iscale[row_iter] : 0;
        }

        for (int j = 0; j < QK_K / 4; j++) {
            y[i].bsums[j] = 0;
        }

        // Quants values are interleaved in sequence of four bytes from corresponding super blocks
        // Bsums values are interleaved in sequence of four bsums from each super block taken for interleaving
        // i.e first four bsums from the first super block, followed by first four bsums from second super block and so on
        for (int j = 0; j < QK_K * 4; j++) {
            int src_offset = (j / (4 * blck_size_interleave)) * blck_size_interleave;
            int src_id     = (j % (4 * blck_size_interleave)) / blck_size_interleave;
            src_offset += (j % blck_size_interleave);
            int index = (((j & 15) >> 2) << 2) + ((j >> 8) << 4) + ((j >> 6) & 3);

            float x0 = srcv[src_id][src_offset] * iscale[src_id];
            y[i].qs[j] = nearest_int(x0);
            y[i].bsums[index] += y[i].qs[j];
        }
    }
}

void ggml_quantize_mat_q8_K_4x8_generic(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(QK_K == 256);
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    block_q8_Kx4 * GGML_RESTRICT y = (block_q8_Kx4 *) vy;

    // scalar
    const int blck_size_interleave = 8;
    float srcv[4][QK_K];
    float iscale[4];

    for (int i = 0; i < nb; i++) {
        for (int row_iter = 0; row_iter < 4; row_iter++) {
            float amax = 0.0f; // absolute max
            float max = 0;

            for (int j = 0; j < QK_K; j++) {
                srcv[row_iter][j] = x[row_iter * k + i * QK_K + j];
                // Update the maximum value of the corresponding super block
                if(amax < fabsf(srcv[row_iter][j])) {
                    amax = fabsf(srcv[row_iter][j]);
                    max = srcv[row_iter][j];
                }
            }

            iscale[row_iter] = amax ? -127.f/max : 0;

            y[i].d[row_iter] = amax ? 1/iscale[row_iter] : 0;
        }

        for (int j = 0; j < QK_K / 4; j++) {
            y[i].bsums[j] = 0;
        }

        // Quants values are interleaved in sequence of eight bytes from corresponding super blocks
        // Bsums values are interleaved in sequence of four bsums from each super block taken for interleaving
        // i.e first four bsums from the first super block, followed by first four bsums from second super block and so on
        for (int j = 0; j < QK_K * 4; j++) {
            int src_offset = (j / (4 * blck_size_interleave)) * blck_size_interleave;
            int src_id     = (j % (4 * blck_size_interleave)) / blck_size_interleave;
            src_offset += (j % blck_size_interleave);
            int index = (((j & 31) >> 3) << 2) + ((j >> 8) << 4) + ((j >> 6) & 3);

            float x0 = srcv[src_id][src_offset] * iscale[src_id];
            y[i].qs[j] = nearest_int(x0);
            y[i].bsums[index] += y[i].qs[j];
        }
    }
}

} // extern "C"

template <int64_t INTER_SIZE, ggml_type PARAM_TYPE>
void ggml_quantize_mat_t(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t nrow, int64_t n_per_row);

template <> void ggml_quantize_mat_t<4, GGML_TYPE_Q8_0>(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t nrow, int64_t n_per_row) {
    assert(nrow == 4);
    UNUSED(nrow);
    ggml_quantize_mat_q8_0_4x4(x, vy, n_per_row);
}

template <> void ggml_quantize_mat_t<8, GGML_TYPE_Q8_0>(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t nrow, int64_t n_per_row) {
    assert(nrow == 4);
    UNUSED(nrow);
    ggml_quantize_mat_q8_0_4x8(x, vy, n_per_row);
}

template <> void ggml_quantize_mat_t<4, GGML_TYPE_Q8_K>(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t nrow, int64_t n_per_row) {
    assert(nrow == 4);
    UNUSED(nrow);
    ggml_quantize_mat_q8_K_4x4(x, vy, n_per_row);
}

template <> void ggml_quantize_mat_t<8, GGML_TYPE_Q8_K>(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t nrow, int64_t n_per_row) {
    assert(nrow == 4);
    UNUSED(nrow);
    ggml_quantize_mat_q8_K_4x8(x, vy, n_per_row);
}

#if defined __riscv_zvfh
template <> void ggml_quantize_mat_t<1, GGML_TYPE_Q8_0>(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t nrow, int64_t n_per_row) {
    assert(nrow == 4);
    UNUSED(nrow);
    ggml_quantize_mat_q8_0_4x1(x, vy, n_per_row);
}

template <> void ggml_quantize_mat_t<1, GGML_TYPE_Q8_K>(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t nrow, int64_t n_per_row) {
    assert(nrow == 4);
    UNUSED(nrow);
    ggml_quantize_mat_q8_K_4x1(x, vy, n_per_row);
}
#endif

template <int M, int N>
static void ggml_gemv_q6_K_NxM_q8_K_generic_impl(int                        n,
                                                 float * GGML_RESTRICT      s,
                                                 size_t                     bs,
                                                 const void * GGML_RESTRICT vx,
                                                 const void * GGML_RESTRICT vy,
                                                 int                        nr,
                                                 int                        nc) {
    constexpr int blocklen          = M;
    constexpr int ncols_interleaved = N;
    const int     qk                = QK_K;
    const int     nb                = n / qk;
    const int     blocks_per_half   = 64 / blocklen;

    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    float sumf[8];

    const block_q8_K * a_ptr = (const block_q8_K *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q6_Kx8 * b_ptr = (const block_q6_Kx8 *) vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0f;
        }

        for (int l = 0; l < nb; l++) {
            for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                const int base_l = (k / blocks_per_half) * 128 + (k % blocks_per_half) * blocklen;
                const int base_h = base_l + 64;

                const int scale_idx_l = base_l / 16;
                const int scale_idx_h = base_h / 16;

                const int qh_shift_l = ((base_l % 128) / 32) * 2;
                const int qh_shift_h = ((base_h % 128) / 32) * 2;

                const int qh_half_l = (base_l / 128) * 32;
                const int qh_half_h = (base_h / 128) * 32;

                for (int j = 0; j < ncols_interleaved; j++) {
                    const int8_t scale_l = b_ptr[l].scales[scale_idx_l * ncols_interleaved + j];
                    const int8_t scale_h = b_ptr[l].scales[scale_idx_h * ncols_interleaved + j];

                    int sumi_l = 0;
                    int sumi_h = 0;

                    for (int i = 0; i < blocklen; i++) {
                        const int ql_pos = k * ncols_interleaved * blocklen + j * blocklen + i;
                        const int l_4    = b_ptr[l].ql[ql_pos] & 0xF;
                        const int hi_4   = (b_ptr[l].ql[ql_pos] >> 4) & 0xF;

                        const int qh_idx_l    = qh_half_l + ((base_l + i) % 32);
                        const int qh_chunk_l  = qh_idx_l / blocklen;
                        const int qh_pos_l    = qh_idx_l % blocklen;
                        const int qh_offset_l = qh_chunk_l * (blocklen * ncols_interleaved) + j * blocklen + qh_pos_l;
                        const int hi_2_l      = (b_ptr[l].qh[qh_offset_l] >> qh_shift_l) & 0x3;

                        const int qh_idx_h    = qh_half_h + ((base_h + i) % 32);
                        const int qh_chunk_h  = qh_idx_h / blocklen;
                        const int qh_pos_h    = qh_idx_h % blocklen;
                        const int qh_offset_h = qh_chunk_h * (blocklen * ncols_interleaved) + j * blocklen + qh_pos_h;
                        const int hi_2_h      = (b_ptr[l].qh[qh_offset_h] >> qh_shift_h) & 0x3;

                        const int q_l = ((hi_2_l << 4) | l_4) - 32;
                        const int q_h = ((hi_2_h << 4) | hi_4) - 32;

                        const int8_t a_l = a_ptr[l].qs[base_l + i];
                        const int8_t a_h = a_ptr[l].qs[base_h + i];

                        sumi_l += q_l * a_l;
                        sumi_h += q_h * a_h;
                    }

                    sumf[j] +=
                        (sumi_l * scale_l + sumi_h * scale_h) * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d;
                }
            }
        }

        for (int j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j];
        }
    }
}

template <int M, int N>
static void ggml_gemm_q6_K_NxM_q8_K_generic_impl(int                        n,
                                                 float * GGML_RESTRICT      s,
                                                 size_t                     bs,
                                                 const void * GGML_RESTRICT vx,
                                                 const void * GGML_RESTRICT vy,
                                                 int                        nr,
                                                 int                        nc) {
    constexpr int blocklen          = M;
    constexpr int ncols_interleaved = N;
    const int     qk                = QK_K;
    const int     nb                = n / qk;
    const int     blocks_per_half   = 64 / blocklen;
    const int     q8_half_stride    = 512;
    const int     q8_low_high_step  = 256;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(bs);

    float sumf[4][8];

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_Kx4 * a_ptr = (const block_q8_Kx4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q6_Kx8 * b_ptr = (const block_q6_Kx8 *) vx + (x * nb);

            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0f;
                }
            }

            for (int l = 0; l < nb; l++) {
                for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                    const int base_l = (k / blocks_per_half) * 128 + (k % blocks_per_half) * blocklen;
                    const int base_h = base_l + 64;

                    const int scale_idx_l = base_l / 16;
                    const int scale_idx_h = base_h / 16;

                    const int qh_shift_l = ((base_l % 128) / 32) * 2;
                    const int qh_shift_h = ((base_h % 128) / 32) * 2;

                    const int qh_half_l = (base_l / 128) * 32;
                    const int qh_half_h = (base_h / 128) * 32;

                    const int q8_base = (k / blocks_per_half) * q8_half_stride + (k % blocks_per_half) * (blocklen * 4);

                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            const int8_t scale_l = b_ptr[l].scales[scale_idx_l * ncols_interleaved + j];
                            const int8_t scale_h = b_ptr[l].scales[scale_idx_h * ncols_interleaved + j];

                            int sumi_l = 0;
                            int sumi_h = 0;

                            for (int i = 0; i < blocklen; i++) {
                                const int ql_pos = k * ncols_interleaved * blocklen + j * blocklen + i;
                                const int l_4    = b_ptr[l].ql[ql_pos] & 0xF;
                                const int hi_4   = (b_ptr[l].ql[ql_pos] >> 4) & 0xF;

                                const int qh_idx_l   = qh_half_l + ((base_l + i) % 32);
                                const int qh_chunk_l = qh_idx_l / blocklen;
                                const int qh_pos_l   = qh_idx_l % blocklen;
                                const int qh_offset_l =
                                    qh_chunk_l * (blocklen * ncols_interleaved) + j * blocklen + qh_pos_l;
                                const int hi_2_l = (b_ptr[l].qh[qh_offset_l] >> qh_shift_l) & 0x3;

                                const int qh_idx_h   = qh_half_h + ((base_h + i) % 32);
                                const int qh_chunk_h = qh_idx_h / blocklen;
                                const int qh_pos_h   = qh_idx_h % blocklen;
                                const int qh_offset_h =
                                    qh_chunk_h * (blocklen * ncols_interleaved) + j * blocklen + qh_pos_h;
                                const int hi_2_h = (b_ptr[l].qh[qh_offset_h] >> qh_shift_h) & 0x3;

                                const int q_l = ((hi_2_l << 4) | l_4) - 32;
                                const int q_h = ((hi_2_h << 4) | hi_4) - 32;

                                const int8_t q8_l = a_ptr[l].qs[q8_base + m * blocklen + i];
                                const int8_t q8_h = a_ptr[l].qs[q8_base + m * blocklen + i + q8_low_high_step];

                                sumi_l += q_l * q8_l;
                                sumi_h += q_h * q8_h;
                            }

                            sumf[m][j] += (sumi_l * scale_l + sumi_h * scale_h) * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) *
                                          a_ptr[l].d[m];
                        }
                    }
                }
            }

            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}

template <int M, int N>
static void ggml_gemv_q5_K_NxM_q8_K_generic_impl(int                        n,
                                                 float * GGML_RESTRICT      s,
                                                 size_t                     bs,
                                                 const void * GGML_RESTRICT vx,
                                                 const void * GGML_RESTRICT vy,
                                                 int                        nr,
                                                 int                        nc) {
    constexpr int         blocklen          = M;
    constexpr int         ncols_interleaved = N;
    const int             qk                = QK_K;
    const int             nb                = n / qk;
    static const uint32_t kmask1            = 0x3f3f3f3f;
    static const uint32_t kmask2            = 0x0f0f0f0f;
    static const uint32_t kmask3            = 0x03030303;

    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    float    sumf[ncols_interleaved];
    float    sum_minf[ncols_interleaved];
    uint32_t utmp[32];
    int      sumi1;
    int      sumi2;
    int      sumi;

    const block_q8_K * a_ptr = (const block_q8_K *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q5_Kx8 * b_ptr = (const block_q5_Kx8 *) vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) {
            sumf[j]     = 0.0;
            sum_minf[j] = 0.0;
        }
        for (int l = 0; l < nb; l++) {
            for (int sb = 0; sb < 8; sb++) {
                memcpy(utmp + sb * 4, b_ptr[l].scales + sb * K_SCALE_SIZE, K_SCALE_SIZE);
                utmp[sb * 4 + 3]      = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                const uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                utmp[sb * 4 + 1]      = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                utmp[sb * 4 + 2]      = uaux_0;
                utmp[sb * 4 + 0] &= kmask1;
            }
            for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                constexpr int scale_stride = 32;
                uint8_t *     scales_0     = (uint8_t *) utmp + (k / (32 / blocklen)) * scale_stride;
                uint8_t *     scales_1     = (uint8_t *) utmp + (k / (32 / blocklen)) * scale_stride + 16;

                const int qh_shift = (k / (32 / blocklen)) * 2;
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi1 = 0;
                    sumi2 = 0;
                    sumi  = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int b_qs_offset = k * ncols_interleaved * blocklen + j * blocklen + i;

                        const int qh_idx      = (k * blocklen + i) % 32;
                        const int qh_chunk    = qh_idx / blocklen;
                        const int qh_pos      = qh_idx % blocklen;
                        const int b_qh_offset = qh_chunk * (blocklen * ncols_interleaved) + j * blocklen + qh_pos;

                        const uint8_t qh_val = b_ptr[l].qh[b_qh_offset];
                        const uint8_t h0     = (qh_val >> qh_shift) & 1;
                        const uint8_t h1     = (qh_val >> (qh_shift + 1)) & 1;

                        const int v0 = (int8_t) ((b_ptr[l].qs[b_qs_offset] & 0xF) | (h0 << 4));
                        const int v1 = (int8_t) ((b_ptr[l].qs[b_qs_offset] >> 4) | (h1 << 4));

                        const int q8_offset = (k / (32 / blocklen)) * 64 + (k % (32 / blocklen)) * blocklen + i;

                        sumi1 = (v0 * a_ptr[l].qs[q8_offset]);
                        sumi2 = (v1 * a_ptr[l].qs[q8_offset + 32]);
                        sumi1 = sumi1 * scales_0[j];
                        sumi2 = sumi2 * scales_1[j];
                        sumi += sumi1 + sumi2;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d;
                }
            }
            for (int sb = 0; sb < 8; sb++) {
                uint8_t * mins = (uint8_t *) utmp + 8 + sb * 16;
                for (int j = 0; j < ncols_interleaved; j++) {
                    sum_minf[j] += mins[j] * (a_ptr[l].bsums[sb * 2] + a_ptr[l].bsums[sb * 2 + 1]) *
                                   GGML_CPU_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d;
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j] - sum_minf[j];
        }
    }
}

template <int M, int N>
static void ggml_gemm_q5_K_NxM_q8_K_generic_impl(int                        n,
                                                 float * GGML_RESTRICT      s,
                                                 size_t                     bs,
                                                 const void * GGML_RESTRICT vx,
                                                 const void * GGML_RESTRICT vy,
                                                 int                        nr,
                                                 int                        nc) {
    constexpr int         blocklen          = M;
    constexpr int         ncols_interleaved = N;
    const int             qk                = QK_K;
    const int             nb                = n / qk;
    static const uint32_t kmask1            = 0x3f3f3f3f;
    static const uint32_t kmask2            = 0x0f0f0f0f;
    static const uint32_t kmask3            = 0x03030303;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    float    sumf[4][ncols_interleaved];
    float    sum_minf[4][ncols_interleaved];
    uint32_t utmp[32];
    int      sumi1;
    int      sumi2;
    int      sumi;

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_Kx4 * a_ptr = (const block_q8_Kx4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q5_Kx8 * b_ptr = (const block_q5_Kx8 *) vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j]     = 0.0;
                    sum_minf[m][j] = 0.0;
                }
            }
            for (int l = 0; l < nb; l++) {
                for (int sb = 0; sb < 8; sb++) {
                    memcpy(utmp + sb * 4, b_ptr[l].scales + sb * K_SCALE_SIZE, K_SCALE_SIZE);
                    utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                    utmp[sb * 4 + 1]      = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                    utmp[sb * 4 + 2]      = uaux_0;
                    utmp[sb * 4 + 0] &= kmask1;
                }
                for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                    constexpr int scale_stride = 32;
                    uint8_t *     scales_0     = (uint8_t *) utmp + (k / (32 / blocklen)) * scale_stride;
                    uint8_t *     scales_1     = (uint8_t *) utmp + (k / (32 / blocklen)) * scale_stride + 16;

                    const int qh_shift = (k / (32 / blocklen)) * 2;
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sumi1 = 0;
                            sumi2 = 0;
                            sumi  = 0;
                            for (int i = 0; i < blocklen; ++i) {
                                const int b_qs_offset = k * ncols_interleaved * blocklen + j * blocklen + i;

                                const int qh_idx   = (k * blocklen + i) % 32;
                                const int qh_chunk = qh_idx / blocklen;
                                const int qh_pos   = qh_idx % blocklen;
                                const int b_qh_offset =
                                    qh_chunk * (blocklen * ncols_interleaved) + j * blocklen + qh_pos;

                                const uint8_t qh_val = b_ptr[l].qh[b_qh_offset];
                                const uint8_t h0     = (qh_val >> qh_shift) & 1;
                                const uint8_t h1     = (qh_val >> (qh_shift + 1)) & 1;

                                const int v0 = (int8_t) ((b_ptr[l].qs[b_qs_offset] & 0xF) | (h0 << 4));
                                const int v1 = (int8_t) ((b_ptr[l].qs[b_qs_offset] >> 4) | (h1 << 4));

                                const int q8_offset = (k / (32 / blocklen)) * 256 +
                                                      (k % (32 / blocklen)) * 4 * blocklen + m * blocklen + i;

                                sumi1 = (v0 * a_ptr[l].qs[q8_offset]);
                                sumi2 = (v1 * a_ptr[l].qs[q8_offset + 128]);
                                sumi1 = sumi1 * scales_0[j];
                                sumi2 = sumi2 * scales_1[j];
                                sumi += sumi1 + sumi2;
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d[m];
                        }
                    }
                }
                for (int sb = 0; sb < 8; sb++) {
                    uint8_t * mins = (uint8_t *) utmp + 8 + sb * 16;
                    for (int m = 0; m < 4; m++) {
                        const int16_t * bsums = a_ptr[l].bsums + (sb * 8) + (m * 4) - ((sb % 2) * 6);
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sum_minf[m][j] += mins[j] * (bsums[0] + bsums[1]) *
                                              GGML_CPU_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d[m];
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j] - sum_minf[m][j];
                }
            }
        }
    }
}

extern "C" {

void ggml_gemv_q4_0_4x4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen = 4;

    assert(nr == 1);
    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    float sumf[4];
    int sumi;

    const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q4_0x4 * b_ptr = (const block_q4_0x4 *) vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        for (int l = 0; l < nb; l++) {
            for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                        const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2])) >> 4;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

void ggml_gemv_q4_0_4x8_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 4;
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

    float sumf[4];
    int sumi;

    const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q4_0x4 * b_ptr = (const block_q4_0x4 *) vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        for (int l = 0; l < nb; l++) {
            for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                        const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2])) >> 4;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

void ggml_gemv_q4_0_8x8_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
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

    float sumf[8];
    int sumi;

    const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q4_0x8 * b_ptr = (const block_q4_0x8 *) vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        for (int l = 0; l < nb; l++) {
            for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                        const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2])) >> 4;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

void ggml_gemv_q4_K_8x4_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 4;
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    assert (n % qk == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    float sumf[8];
    float sum_minf[8];
    uint32_t utmp[32];
    int sumi1;
    int sumi2;
    int sumi;

    const block_q8_K * a_ptr = (const block_q8_K *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q4_Kx8 * b_ptr = (const block_q4_Kx8 *) vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
            sum_minf[j] = 0.0;
        }
        for (int l = 0; l < nb; l++) {
            for (int sb = 0; sb < 8; sb++) {
                memcpy(utmp + sb * 4, b_ptr[l].scales + sb * 12, 12);
                utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                const uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                utmp[sb * 4 + 1] = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                utmp[sb * 4 + 2] = uaux_0;
                utmp[sb * 4 + 0] &= kmask1;
            }
            for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                uint8_t * scales_0 = (uint8_t *) utmp + (k / 8) * 32;
                uint8_t * scales_1 = (uint8_t *) utmp + (k / 8) * 32 + 16;
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi1 = 0;
                    sumi2 = 0;
                    sumi = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF);
                        const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4);
                        sumi1 = (v0 * a_ptr[l].qs[(k / 8) * 64 + (k % 8) * blocklen + i]);
                        sumi2 = (v1 * a_ptr[l].qs[(k / 8) * 64 + (k % 8) * blocklen + i + 32]);
                        sumi1 = sumi1 * scales_0[j];
                        sumi2 = sumi2 * scales_1[j];
                        sumi += sumi1 + sumi2;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d;
                }
            }
            for (int sb = 0; sb < 8; sb++) {
                uint8_t * mins = (uint8_t *) utmp + 8 + sb * 16;
                for (int j = 0; j < ncols_interleaved; j++) {
                    sum_minf[j] += mins[j] * (a_ptr[l].bsums[sb * 2] + a_ptr[l].bsums[sb * 2 + 1]) * GGML_CPU_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d;
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j] - sum_minf[j];
        }
    }
}

void ggml_gemv_q4_K_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    assert (n % qk == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    float sumf[8];
    float sum_minf[8];
    uint32_t utmp[32];
    int sumi1;
    int sumi2;
    int sumi;

    const block_q8_K * a_ptr = (const block_q8_K *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q4_Kx8 * b_ptr = (const block_q4_Kx8 *) vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
            sum_minf[j] = 0.0;
        }
        for (int l = 0; l < nb; l++) {
            for (int sb = 0; sb < 8; sb++) {
                memcpy(utmp + sb * 4, b_ptr[l].scales + sb * 12, 12);
                utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                const uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                utmp[sb * 4 + 1] = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                utmp[sb * 4 + 2] = uaux_0;
                utmp[sb * 4 + 0] &= kmask1;
            }
            for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                uint8_t *scales_0 = (uint8_t*) utmp + (k / 4) * 32;
                uint8_t *scales_1 = (uint8_t*) utmp + (k / 4) * 32 + 16;
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi1 = 0;
                    sumi2 = 0;
                    sumi = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF);
                        const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4);
                        sumi1 = (v0 * a_ptr[l].qs[(k >> 2) * 64 + (k % 4) * blocklen + i]);
                        sumi2 = (v1 * a_ptr[l].qs[(k >> 2) * 64 + (k % 4) * blocklen + i + 32]);
                        sumi1 = sumi1 * scales_0[j];
                        sumi2 = sumi2 * scales_1[j];
                        sumi += sumi1 + sumi2;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d;
                }
            }
            for (int sb = 0; sb < 8; sb++) {
                uint8_t *mins = (uint8_t*) utmp + 8 + sb * 16;
                for (int j = 0; j < ncols_interleaved; j++) {
                    sum_minf[j] += mins[j] * (a_ptr[l].bsums[sb * 2] + a_ptr[l].bsums[sb * 2 + 1]) * GGML_CPU_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d;
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j] - sum_minf[j];
        }
    }
}

void ggml_gemv_q2_K_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
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

    float sumf[8];
    float sum_minf[8];
    int sumi1,sumi2,sumi3,sumi4;
    int sumi;

    const block_q8_K * a_ptr = (const block_q8_K *)vy;
    for(int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q2_Kx8 * b_ptr = (const block_q2_Kx8 *) vx + (x * nb);
        for (int j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
            sum_minf[j] = 0.0;
        }
        for (int l = 0; l < nb; l++) {
            for (int k = 0; k < (qk / (4 * blocklen)); k++) {
                const uint8_t *scales_0 = b_ptr[l].scales + (k / 4) * 64 ;
                const uint8_t *scales_1 = b_ptr[l].scales + (k / 4) * 64 + 16;
                const uint8_t *scales_2 = b_ptr[l].scales + (k / 4) * 64 + 32;
                const uint8_t *scales_3 = b_ptr[l].scales + (k / 4) * 64 + 48;
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi1 = 0;
                    sumi2 = 0;
                    sumi3 = 0;
                    sumi4 = 0;
                    sumi = 0;
                    int offset = ((k / 2) % 2) + j * 2;
                    for (int i = 0; i < blocklen; ++i){
                        const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 3);
                        const int v1 = (int8_t) ((b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 2 ) & 3);
                        const int v2 = (int8_t) ((b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4 ) & 3);
                        const int v3 = (int8_t) ((b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 6 ) & 3);
                        sumi1 = (v0 * a_ptr[l].qs[(k >> 2) * 128 + (k % 4) * blocklen + i]);
                        sumi2 = (v1 * a_ptr[l].qs[(k >> 2) * 128 + (k % 4) * blocklen + i + 32]);
                        sumi3 = (v2 * a_ptr[l].qs[(k >> 2) * 128 + (k % 4) * blocklen + i + 64]);
                        sumi4 = (v3 * a_ptr[l].qs[(k >> 2) * 128 + (k % 4) * blocklen + i + 96]);

                        sumi1 = sumi1 * (scales_0[offset] & 0xF);
                        sumi2 = sumi2 * (scales_1[offset] & 0xF);
                        sumi3 = sumi3 * (scales_2[offset] & 0xF);
                        sumi4 = sumi4 * (scales_3[offset] & 0xF);
                        sumi += sumi1 + sumi2 + sumi3 + sumi4;
                    }
                    sumf[j] += sumi * GGML_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d;
                }
            }
            for(int sb = 0; sb < 8; sb++) {
                const uint8_t *mins = b_ptr[l].scales + sb * 16;
                for(int j = 0; j < ncols_interleaved; j++){
                    sum_minf[j] += ((mins[j * 2] >> 4) * a_ptr[l].bsums[sb * 2] + (mins[(j * 2)+ 1] >> 4) * a_ptr[l].bsums[sb * 2 + 1]) * GGML_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d;
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j] - sum_minf[j];
        }
    }
}

void ggml_gemv_iq2_xs_r8_q8_K_generic(
        int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy,
        int nr, int nc) {
    constexpr int ncols_interleaved = 8;
    constexpr int values_per_subblock = 16;
    constexpr int subblocks_per_block = QK_K / values_per_subblock;
    static const int8_t values[16] = {
        -43, -25, -8, 8, 25, 43, 0, 0,
          0,   0,  0, 0,  0,  0, 0, 0,
    };

    GGML_ASSERT(n % QK_K == 0);
    GGML_ASSERT(nc % ncols_interleaved == 0);

    const int nb = n / QK_K;
    const block_iq2_xs_r8 * b_ptr_start = (const block_iq2_xs_r8 *) vx;
    const block_q8_K * a_ptr_start = (const block_q8_K *) vy;

    for (int y = 0; y < nr; ++y) {
        const block_q8_K * a_ptr = a_ptr_start + y * nb;
        float * out = s + y * bs;

        for (int x = 0; x < nc / ncols_interleaved; ++x) {
            const block_iq2_xs_r8 * b_ptr = b_ptr_start + x * nb;
            float sumf[ncols_interleaved] = {};

            for (int b = 0; b < nb; ++b) {
                int32_t isum[ncols_interleaved] = {};

                for (int sb = 0; sb < subblocks_per_block; ++sb) {
                    const int ib32 = sb / 2;
                    const bool high_scale = (sb & 1) != 0;
                    const int8_t * aq = a_ptr[b].qs + sb * values_per_subblock;

                    for (int row = 0; row < ncols_interleaved; ++row) {
                        const uint8_t packed_scale = b_ptr[b].scales[ib32 * ncols_interleaved + row];
                        const int scale = 2 * (high_scale ? packed_scale >> 4 : packed_scale & 0x0f) + 1;
                        const uint8_t * packed_codes = b_ptr[b].qs +
                            (sb * 2 + row / 4) * 32;
                        int32_t dot = 0;
                        for (int k = 0; k < values_per_subblock; ++k) {
                            const int code_index = (row % 4) * values_per_subblock + k;
                            const uint8_t packed = packed_codes[code_index / 2];
                            const uint8_t code = (packed >> (4 * (code_index & 1))) & 0x0f;
                            dot += values[code] * aq[k];
                        }
                        isum[row] += scale * dot;
                    }
                }

                for (int row = 0; row < ncols_interleaved; ++row) {
                    sumf[row] += isum[row] *
                        (GGML_CPU_FP16_TO_FP32(b_ptr[b].d[row]) * a_ptr[b].d * 0.125f);
                }
            }

            for (int row = 0; row < ncols_interleaved; ++row) {
                out[x * ncols_interleaved + row] = sumf[row];
            }
        }
    }
}

void ggml_gemm_iq2_xs_r8_q8_K_generic(
        int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy,
        int nr, int nc) {
    ggml_gemv_iq2_xs_r8_q8_K_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemv_iq3_xxs_r8_q8_K_generic(
        int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy,
        int nr, int nc) {
    constexpr int ncols_interleaved = 8;
    constexpr int values_per_subblock = 16;
    constexpr int subblocks_per_block = QK_K / values_per_subblock;
    static const int8_t values[16] = {
        -62, -52, -44, -36, -28, -20, -12, -4,
          4,  12,  20,  28,  36,  44,  52, 62,
    };

    GGML_ASSERT(n % QK_K == 0);
    GGML_ASSERT(nc % ncols_interleaved == 0);

    const int nb = n / QK_K;
    const block_iq3_xxs_r8 * b_ptr_start = (const block_iq3_xxs_r8 *) vx;
    const block_q8_K * a_ptr_start = (const block_q8_K *) vy;

    for (int y = 0; y < nr; ++y) {
        const block_q8_K * a_ptr = a_ptr_start + y * nb;
        float * out = s + y * bs;

        for (int x = 0; x < nc / ncols_interleaved; ++x) {
            const block_iq3_xxs_r8 * b_ptr = b_ptr_start + x * nb;
            float sumf[ncols_interleaved] = {};

            for (int b = 0; b < nb; ++b) {
                int32_t isum[ncols_interleaved] = {};

                for (int sb = 0; sb < subblocks_per_block; ++sb) {
                    const int ib32 = sb / 2;
                    const int8_t * aq = a_ptr[b].qs + sb * values_per_subblock;

                    for (int row = 0; row < ncols_interleaved; ++row) {
                        const int scale = 2 * ((b_ptr[b].scales[ib32] >> (4 * row)) & 0x0f) + 1;
                        const uint8_t * packed_codes = b_ptr[b].qs +
                            (sb * 2 + row / 4) * 32;
                        int32_t dot = 0;
                        for (int k = 0; k < values_per_subblock; ++k) {
                            const int code_index = (row % 4) * values_per_subblock + k;
                            const uint8_t packed = packed_codes[code_index / 2];
                            const uint8_t code = (packed >> (4 * (code_index & 1))) & 0x0f;
                            dot += values[code] * aq[k];
                        }
                        isum[row] += scale * dot;
                    }
                }

                for (int row = 0; row < ncols_interleaved; ++row) {
                    sumf[row] += isum[row] *
                        (GGML_CPU_FP16_TO_FP32(b_ptr[b].d[row]) * a_ptr[b].d * 0.25f);
                }
            }

            for (int row = 0; row < ncols_interleaved; ++row) {
                out[x * ncols_interleaved + row] = sumf[row];
            }
        }
    }
}

void ggml_gemm_iq3_xxs_r8_q8_K_generic(
        int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy,
        int nr, int nc) {
    ggml_gemv_iq3_xxs_r8_q8_K_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemv_q4_K_r8_q8_K_generic(
        int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy,
        int nr, int nc) {
    constexpr int ncols_interleaved = 8;
    constexpr int values_per_subblock = 16;
    constexpr int subblocks_per_block = QK_K / values_per_subblock;

    GGML_ASSERT(n % QK_K == 0);
    GGML_ASSERT(nc % ncols_interleaved == 0);

    const int nb = n / QK_K;
    const block_q4_K_r8 * b_ptr_start = (const block_q4_K_r8 *) vx;
    const block_q8_K * a_ptr_start = (const block_q8_K *) vy;

    for (int y = 0; y < nr; ++y) {
        const block_q8_K * a_ptr = a_ptr_start + y * nb;
        float * out = s + y * bs;

        for (int x = 0; x < nc / ncols_interleaved; ++x) {
            const block_q4_K_r8 * b_ptr = b_ptr_start + x * nb;
            float sumf[ncols_interleaved] = {};

            for (int b = 0; b < nb; ++b) {
                int32_t isum[ncols_interleaved] = {};
                int32_t imin[ncols_interleaved] = {};

                for (int sb = 0; sb < subblocks_per_block; ++sb) {
                    const int ib32 = sb / 2;
                    const int8_t * aq = a_ptr[b].qs + sb * values_per_subblock;
                    const int bsum = a_ptr[b].bsums[sb];

                    for (int row = 0; row < ncols_interleaved; ++row) {
                        const int group = sb * 2 + row / 4;
                        const uint8_t * packed_low = b_ptr[b].ql + group * 32;
                        int32_t dot = 0;
                        for (int k = 0; k < values_per_subblock; ++k) {
                            const int code_index = (row % 4) * values_per_subblock + k;
                            const uint8_t packed = packed_low[code_index / 2];
                            const uint8_t weight = (packed >> (4 * (code_index & 1))) & 0x0f;
                            dot += weight * aq[k];
                        }
                        const int index = ib32 * ncols_interleaved + row;
                        isum[row] += b_ptr[b].scales[index] * dot;
                        imin[row] += b_ptr[b].mins[index] * bsum;
                    }
                }

                for (int row = 0; row < ncols_interleaved; ++row) {
                    const float activation_scale = a_ptr[b].d;
                    sumf[row] += activation_scale *
                        (GGML_CPU_FP16_TO_FP32(b_ptr[b].d[row]) * isum[row] -
                         GGML_CPU_FP16_TO_FP32(b_ptr[b].dmin[row]) * imin[row]);
                }
            }

            for (int row = 0; row < ncols_interleaved; ++row) {
                out[x * ncols_interleaved + row] = sumf[row];
            }
        }
    }
}

void ggml_gemm_q4_K_r8_q8_K_generic(
        int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy,
        int nr, int nc) {
    constexpr int ncols_interleaved = 8;
    constexpr int values_per_subblock = 16;
    constexpr int subblocks_per_block = QK_K / values_per_subblock;

    GGML_ASSERT(n % QK_K == 0);
    GGML_ASSERT(nr % 4 == 0);
    GGML_ASSERT(nc % ncols_interleaved == 0);

    const int nb = n / QK_K;
    const block_q4_K_r8 * b_ptr_start = (const block_q4_K_r8 *) vx;
    const block_q8_Kx4 * a_ptr_start = (const block_q8_Kx4 *) vy;

    for (int y = 0; y < nr / 4; ++y) {
        const block_q8_Kx4 * a_ptr = a_ptr_start + y * nb;

        for (int x = 0; x < nc / ncols_interleaved; ++x) {
            const block_q4_K_r8 * b_ptr = b_ptr_start + x * nb;
            float sumf[4][ncols_interleaved] = {};

            for (int b = 0; b < nb; ++b) {
                int32_t isum[4][ncols_interleaved] = {};
                int32_t imin[4][ncols_interleaved] = {};

                for (int sb = 0; sb < subblocks_per_block; ++sb) {
                    const int ib32 = sb / 2;
                    for (int m = 0; m < 4; ++m) {
                        const int bsum_index = (sb / 4) * 16 + m * 4 + sb % 4;
                        const int bsum = a_ptr[b].bsums[bsum_index];

                        for (int row = 0; row < ncols_interleaved; ++row) {
                            const int group = sb * 2 + row / 4;
                            const uint8_t * packed_low = b_ptr[b].ql + group * 32;
                            int32_t dot = 0;
                            for (int k = 0; k < values_per_subblock; ++k) {
                                const int p = sb * values_per_subblock + k;
                                const int q8_index = (p / 8) * 32 + m * 8 + p % 8;
                                const int code_index = (row % 4) * values_per_subblock + k;
                                const uint8_t packed = packed_low[code_index / 2];
                                const uint8_t weight = (packed >> (4 * (code_index & 1))) & 0x0f;
                                dot += weight * a_ptr[b].qs[q8_index];
                            }
                            const int index = ib32 * ncols_interleaved + row;
                            isum[m][row] += b_ptr[b].scales[index] * dot;
                            imin[m][row] += b_ptr[b].mins[index] * bsum;
                        }
                    }
                }

                for (int m = 0; m < 4; ++m) {
                    for (int row = 0; row < ncols_interleaved; ++row) {
                        sumf[m][row] += a_ptr[b].d[m] *
                            (GGML_CPU_FP16_TO_FP32(b_ptr[b].d[row]) * isum[m][row] -
                             GGML_CPU_FP16_TO_FP32(b_ptr[b].dmin[row]) * imin[m][row]);
                    }
                }
            }

            for (int m = 0; m < 4; ++m) {
                for (int row = 0; row < ncols_interleaved; ++row) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + row] = sumf[m][row];
                }
            }
        }
    }
}

void ggml_gemv_q5_K_r8_q8_K_generic(
        int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy,
        int nr, int nc) {
    constexpr int ncols_interleaved = 8;
    constexpr int values_per_subblock = 16;
    constexpr int subblocks_per_block = QK_K / values_per_subblock;

    GGML_ASSERT(n % QK_K == 0);
    GGML_ASSERT(nc % ncols_interleaved == 0);

    const int nb = n / QK_K;
    const block_q5_K_r8 * b_ptr_start = (const block_q5_K_r8 *) vx;
    const block_q8_K * a_ptr_start = (const block_q8_K *) vy;

    for (int y = 0; y < nr; ++y) {
        const block_q8_K * a_ptr = a_ptr_start + y * nb;
        float * out = s + y * bs;

        for (int x = 0; x < nc / ncols_interleaved; ++x) {
            const block_q5_K_r8 * b_ptr = b_ptr_start + x * nb;
            float sumf[ncols_interleaved] = {};

            for (int b = 0; b < nb; ++b) {
                int32_t isum[ncols_interleaved] = {};
                int32_t imin[ncols_interleaved] = {};

                for (int sb = 0; sb < subblocks_per_block; ++sb) {
                    const int ib32 = sb / 2;
                    const int8_t * aq = a_ptr[b].qs + sb * values_per_subblock;
                    const int bsum = a_ptr[b].bsums[sb];

                    for (int row = 0; row < ncols_interleaved; ++row) {
                        const int group = sb * 2 + row / 4;
                        const uint8_t * packed_low = b_ptr[b].ql + group * 32;
                        const uint64_t high_plane = b_ptr[b].qh[group];
                        int32_t dot = 0;
                        for (int k = 0; k < values_per_subblock; ++k) {
                            const int code_index = (row % 4) * values_per_subblock + k;
                            const uint8_t packed = packed_low[code_index / 2];
                            const uint8_t low = (packed >> (4 * (code_index & 1))) & 0x0f;
                            const uint8_t weight = low | (((high_plane >> code_index) & 1) << 4);
                            dot += weight * aq[k];
                        }
                        const int index = ib32 * ncols_interleaved + row;
                        isum[row] += b_ptr[b].scales[index] * dot;
                        imin[row] += b_ptr[b].mins[index] * bsum;
                    }
                }

                for (int row = 0; row < ncols_interleaved; ++row) {
                    const float activation_scale = a_ptr[b].d;
                    sumf[row] += activation_scale *
                        (GGML_CPU_FP16_TO_FP32(b_ptr[b].d[row]) * isum[row] -
                         GGML_CPU_FP16_TO_FP32(b_ptr[b].dmin[row]) * imin[row]);
                }
            }

            for (int row = 0; row < ncols_interleaved; ++row) {
                out[x * ncols_interleaved + row] = sumf[row];
            }
        }
    }
}

void ggml_gemm_q5_K_r8_q8_K_generic(
        int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy,
        int nr, int nc) {
    constexpr int ncols_interleaved = 8;
    constexpr int values_per_subblock = 16;
    constexpr int subblocks_per_block = QK_K / values_per_subblock;

    GGML_ASSERT(n % QK_K == 0);
    GGML_ASSERT(nr % 4 == 0);
    GGML_ASSERT(nc % ncols_interleaved == 0);

    const int nb = n / QK_K;
    const block_q5_K_r8 * b_ptr_start = (const block_q5_K_r8 *) vx;
    const block_q8_Kx4 * a_ptr_start = (const block_q8_Kx4 *) vy;

    for (int y = 0; y < nr / 4; ++y) {
        const block_q8_Kx4 * a_ptr = a_ptr_start + y * nb;

        for (int x = 0; x < nc / ncols_interleaved; ++x) {
            const block_q5_K_r8 * b_ptr = b_ptr_start + x * nb;
            float sumf[4][ncols_interleaved] = {};

            for (int b = 0; b < nb; ++b) {
                int32_t isum[4][ncols_interleaved] = {};
                int32_t imin[4][ncols_interleaved] = {};

                for (int sb = 0; sb < subblocks_per_block; ++sb) {
                    const int ib32 = sb / 2;
                    for (int m = 0; m < 4; ++m) {
                        const int bsum_index = (sb / 4) * 16 + m * 4 + sb % 4;
                        const int bsum = a_ptr[b].bsums[bsum_index];

                        for (int row = 0; row < ncols_interleaved; ++row) {
                            const int group = sb * 2 + row / 4;
                            const uint8_t * packed_low = b_ptr[b].ql + group * 32;
                            const uint64_t high_plane = b_ptr[b].qh[group];
                            int32_t dot = 0;
                            for (int k = 0; k < values_per_subblock; ++k) {
                                const int p = sb * values_per_subblock + k;
                                const int q8_index = (p / 8) * 32 + m * 8 + p % 8;
                                const int code_index = (row % 4) * values_per_subblock + k;
                                const uint8_t packed = packed_low[code_index / 2];
                                const uint8_t low = (packed >> (4 * (code_index & 1))) & 0x0f;
                                const uint8_t weight = low | (((high_plane >> code_index) & 1) << 4);
                                dot += weight * a_ptr[b].qs[q8_index];
                            }
                            const int index = ib32 * ncols_interleaved + row;
                            isum[m][row] += b_ptr[b].scales[index] * dot;
                            imin[m][row] += b_ptr[b].mins[index] * bsum;
                        }
                    }
                }

                for (int m = 0; m < 4; ++m) {
                    for (int row = 0; row < ncols_interleaved; ++row) {
                        sumf[m][row] += a_ptr[b].d[m] *
                            (GGML_CPU_FP16_TO_FP32(b_ptr[b].d[row]) * isum[m][row] -
                             GGML_CPU_FP16_TO_FP32(b_ptr[b].dmin[row]) * imin[m][row]);
                    }
                }
            }

            for (int m = 0; m < 4; ++m) {
                for (int row = 0; row < ncols_interleaved; ++row) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + row] = sumf[m][row];
                }
            }
        }
    }
}

void ggml_gemv_q5_K_8x4_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    ggml_gemv_q5_K_NxM_q8_K_generic_impl<4, 8>(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemv_q5_K_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    ggml_gemv_q5_K_NxM_q8_K_generic_impl<8, 8>(n, s, bs, vx, vy, nr, nc);
}


void ggml_gemv_q6_K_8x4_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    ggml_gemv_q6_K_NxM_q8_K_generic_impl<4, 8>(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemv_q6_K_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    ggml_gemv_q6_K_NxM_q8_K_generic_impl<8, 8>(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemv_iq4_nl_4x4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen = 4;

    assert(nr == 1);
    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    float sumf[4];
    int sumi;

    const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_iq4_nlx4 * b_ptr = (const block_iq4_nlx4 *) vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        for (int l = 0; l < nb; l++) {
            for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int v0 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                        const int v1 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2]));
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

void ggml_gemv_iq4_nl_8x8_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;

    assert(nr == 1);
    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    float sumf[8];
    int sumi;

    const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_iq4_nlx8 * b_ptr = (const block_iq4_nlx8 *) vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        for (int l = 0; l < nb; l++) {
            for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int v0 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                        const int v1 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2]));
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

void ggml_gemv_mxfp4_4x4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen = 4;

    assert(nr == 1);
    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    float sumf[4];
    int sumi;

    const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_mxfp4x4 * b_ptr = (const block_mxfp4x4 *) vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        for (int l = 0; l < nb; l++) {
            for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int v0 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                        const int v1 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2]));
                    }
                    sumf[j] += sumi * GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[l].e[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

void ggml_gemv_mxfp4_8x8_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;

    assert(nr == 1);
    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    float sumf[8];
    int sumi;

    const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_mxfp4x8 * b_ptr = (const block_mxfp4x8 *) vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        for (int l = 0; l < nb; l++) {
            for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int v0 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                        const int v1 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2]));
                    }
                    sumf[j] += sumi * GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[l].e[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

void ggml_gemv_q8_0_4x4_q8_0_generic(int                        n,
                                     float * GGML_RESTRICT      s,
                                     size_t                     bs,
                                     const void * GGML_RESTRICT vx,
                                     const void * GGML_RESTRICT vy,
                                     int                        nr,
                                     int                        nc) {
    const int qk                = QK8_0;
    const int nb                = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen          = 4;

    assert(nr == 1);
    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    float sumf[4];
    int   sumi;

    const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q8_0x4 * b_ptr = (const block_q8_0x4 *) vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
        }
        for (int l = 0; l < nb; l++) {
            for (int k = 0; k < (qk / blocklen); k++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int v0 = b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                        sumi += v0 * a_ptr[l].qs[k * blocklen + i];
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j];
        }
    }
}

void ggml_gemv_q8_0_4x8_q8_0_generic(int                        n,
                                     float * GGML_RESTRICT      s,
                                     size_t                     bs,
                                     const void * GGML_RESTRICT vx,
                                     const void * GGML_RESTRICT vy,
                                     int                        nr,
                                     int                        nc) {
    const int qk                = QK8_0;
    const int nb                = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen          = 8;

    assert(nr == 1);
    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    float sumf[4];
    int   sumi;

    const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q8_0x4 * b_ptr = (const block_q8_0x4 *) vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
        }
        for (int l = 0; l < nb; l++) {
            for (int k = 0; k < (qk / blocklen); k++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int v0 = b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                        sumi += v0 * a_ptr[l].qs[k * blocklen + i];
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j];
        }
    }
}

void ggml_gemv_q8_0_8x8_q8_0_generic(
        int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy,
        int nr, int nc) {
    constexpr int ncols_interleaved = 8;
    constexpr int values_per_subblock = 16;

    assert(n % QK8_0 == 0);
    assert(nc % ncols_interleaved == 0);

    const int nb = n / QK8_0;
    const block_q8_0x8 * b_ptr_start = (const block_q8_0x8 *) vx;
    const block_q8_0 * a_ptr_start = (const block_q8_0 *) vy;

    for (int y = 0; y < nr; ++y) {
        const block_q8_0 * a_ptr = a_ptr_start + y * nb;
        float * out = s + y * bs;

        for (int x = 0; x < nc / ncols_interleaved; ++x) {
            const block_q8_0x8 * b_ptr = b_ptr_start + x * nb;
            float sumf[ncols_interleaved] = {};

            for (int b = 0; b < nb; ++b) {
                int isum[ncols_interleaved] = {};
                for (int sb = 0; sb < QK8_0 / values_per_subblock; ++sb) {
                    for (int row = 0; row < ncols_interleaved; ++row) {
                        const int row_group = row / 4;
                        const int row_in_group = row % 4;
                        const int weight_offset =
                            (sb * 2 + row_group) * 64 + row_in_group * values_per_subblock;
                        for (int k = 0; k < values_per_subblock; ++k) {
                            const int weight = (int) (uint8_t) b_ptr[b].qs[weight_offset + k] - 128;
                            isum[row] += weight * a_ptr[b].qs[sb * values_per_subblock + k];
                        }
                    }
                }

                const float activation_scale = GGML_CPU_FP16_TO_FP32(a_ptr[b].d);
                for (int row = 0; row < ncols_interleaved; ++row) {
                    sumf[row] += isum[row] *
                        GGML_CPU_FP16_TO_FP32(b_ptr[b].d[row]) * activation_scale;
                }
            }

            for (int row = 0; row < ncols_interleaved; ++row) {
                out[x * ncols_interleaved + row] = sumf[row];
            }
        }
    }
}

// Only enable these for RISC-V.
#if defined __riscv_zvfh
void ggml_gemv_q4_0_16x1_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 16;
    const int blocklen = 1;

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

    float sumf[16];
    int sumi;

    const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q4_0x16 * b_ptr = (const block_q4_0x16 *) vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        for (int l = 0; l < nb; l++) {
            for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                        const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2])) >> 4;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

void ggml_gemv_q4_K_16x1_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_interleaved = 16;
    const int blocklen = 1;
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
    float sumf[16];
    float sum_minf[16];
    uint8_t scales[128];
    uint8_t mins[128];
    int sumi1;
    int sumi2;
    int sumi;
    const block_q8_K * a_ptr = (const block_q8_K *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q4_Kx16 * b_ptr = (const block_q4_Kx16 *) vx + (x * nb);
        for (int j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0f;
            sum_minf[j] = 0.0f;
        }
        for (int l = 0; l < nb; l++) {
            for (int i = 0; i < 128; i++) {
                scales[i] = b_ptr[l].scales[i] & 0x0F;
                mins[i] = b_ptr[l].scales[i] >> 4;
            }
            for (int i = 0; i < 64; i++) {
                scales[i] |= (b_ptr[l].scales[128 + i] & 0x03) << 4;
                mins[i] |= (b_ptr[l].scales[128 + i] & 0x0C) << 2;
                scales[i + 64] |= (b_ptr[l].scales[128 + i] & 0x30);
                mins[i + 64] |= (b_ptr[l].scales[128 + i] & 0xC0) >> 2;
            }
            for (int sb = 0; sb < 8; sb++) {
                uint8_t *min = &mins[sb * 16];
                for (int j = 0; j < ncols_interleaved; j++) {
                    sum_minf[j] += min[j] * (a_ptr[l].bsums[sb * 2] + a_ptr[l].bsums[sb * 2 + 1]) * GGML_CPU_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d;
                }
            }
            for (int sb = 0; sb < 8; sb += 2) {
                uint8_t *scales_0 = &scales[sb * 16];
                uint8_t *scales_1 = &scales[(sb + 1) * 16];
                for (int i = 0; i < QK4_0; i++) {
                    for (int j = 0; j < ncols_interleaved; j++) {
                        sumi1 = 0;
                        sumi2 = 0;
                        sumi = 0;
                        const int v0 = (int8_t) (b_ptr[l].qs[sb * 256 + i * 16 + j] & 0xF);
                        const int v1 = (int8_t) (b_ptr[l].qs[sb * 256 + i * 16 + j] >> 4);
                        sumi1 = (v0 * a_ptr[l].qs[sb * 32 + i]);
                        sumi2 = (v1 * a_ptr[l].qs[sb * 32 + 32 + i]);
                        sumi1 = sumi1 * scales_0[j];
                        sumi2 = sumi2 * scales_1[j];
                        sumi += sumi1 + sumi2;
                        sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d;
                    }
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j] - sum_minf[j];
        }
    }
}

void ggml_gemv_iq4_nl_16x1_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 16;
    const int blocklen = 1;

    assert(nr == 1);
    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    float sumf[16];
    int sumi;

    const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_iq4_nlx16 * b_ptr = (const block_iq4_nlx16 *) vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        for (int l = 0; l < nb; l++) {
            for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int v0 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                        const int v1 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2]));
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

void ggml_gemv_q8_0_16x1_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk                = QK8_0;
    const int nb                = n / qk;
    const int ncols_interleaved = 16;
    const int blocklen          = 1;

    assert(nr == 1);
    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(bs);
    UNUSED(nr);

    float sumf[16];
    int   sumi;

    const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q8_0x16 * b_ptr = (const block_q8_0x16 *) vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
        }
        for (int l = 0; l < nb; l++) {
            for (int k = 0; k < (qk / blocklen); k++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int v0 = b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                        sumi += v0 * a_ptr[l].qs[k * blocklen + i];
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j];
        }
    }
}

void ggml_gemv_q2_K_16x1_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    assert(n % QK_K == 0);
    assert(nr == 1);
    assert(nc % 16 == 0);

    UNUSED(bs);
    UNUSED(nr);

    const int nb = n / QK_K;
    const block_q2_Kx16 * x = (const block_q2_Kx16 *)vx;
    const block_q8_K    * y = (const block_q8_K *)vy;

    // Layout: Even-Low(0,2,4,6), Odd-Low(1,3,5,7), Even-High(8...), Odd-High(9...)
    const int sb_perm[16] = {
        0, 4, 1, 5, 2, 6, 3, 7,  // 0-7
        8, 12, 9, 13, 10, 14, 11, 15 // 8-15
    };

    for (int col_tile = 0; col_tile < nc; col_tile += 16) {
        const block_q2_Kx16 * x_ptr = x + (col_tile / 16) * nb;
        const block_q8_K    * y_ptr = y;

        float sumf[16] = {0};

        // Loop over K-blocks
        for (int k_block = 0; k_block < nb; ++k_block) {
            int32_t isum[16]  = {0};
            int32_t summs[16] = {0};

            const uint8_t * qs_rhs = x_ptr[k_block].qs;
            const uint8_t * sc_rhs = x_ptr[k_block].scales;
            const int8_t  * qs_lhs = y_ptr[k_block].qs;
            const int16_t * bs_lhs = y_ptr[k_block].bsums;

            // Iterate over sub-blocks 0..15
            for (int sb = 0; sb < 16; ++sb) {
                // Correction Term
                int16_t bsum = bs_lhs[sb];
                int scale_offset = sb_perm[sb] * 16;

                for (int col = 0; col < 16; ++col) {
                    uint8_t sc_val = sc_rhs[scale_offset + col];
                    summs[col] += bsum * (sc_val >> 4); // Min is high 4 bits
                }

                // Main Dot Product
                // Calculate base offsets for Q2 unpacking based on SB
                int byte_base;
                if (sb < 8) byte_base = (sb % 2 == 0) ? 0 : 16;
                else        byte_base = (sb % 2 == 0) ? 32 : 48;

                int shift = ((sb / 2) % 4) * 2;

                for (int col = 0; col < 16; ++col) {
                    uint8_t sc_val = sc_rhs[scale_offset + col];
                    int32_t d_sb = sc_val & 0xF; // Scale is low 4 bits

                    // Process 16 elements (l=0..15)
                    for (int l = 0; l < 16; ++l) {
                        // Q2: Interleaved by column. Byte `l` contains 4 k-values.
                        int qs_idx = (byte_base + l) * 16 + col;
                        uint8_t q2_val = (qs_rhs[qs_idx] >> shift) & 3;

                        // Q8: Linear access
                        int k = sb * 16 + l;
                        int8_t q8_val = qs_lhs[k];

                        isum[col] += q8_val * q2_val * d_sb;
                    }
                }
            }

            // Finalize K-Block
            for (int col = 0; col < 16; ++col) {
                float d_lhs = y_ptr[k_block].d;
                float d_rhs = GGML_FP16_TO_FP32(x_ptr[k_block].d[col]);
                float dm_rhs = GGML_FP16_TO_FP32(x_ptr[k_block].dmin[col]);

                float d_all = d_lhs * d_rhs;
                float d_min = d_lhs * dm_rhs;

                sumf[col] += (isum[col] * d_all) - (summs[col] * d_min);
            }
        }

        for (int col = 0; col < 16; ++col) {
            s[col_tile + col] = sumf[col];
        }
    }
}
#endif

void ggml_gemm_q4_0_4x4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen = 4;

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

    {
        float sumf[4][4];
        int sumi;

        for (int y = 0; y < nr / 4; y++) {
            const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + (y * nb);
            for (int x = 0; x < nc / ncols_interleaved; x++) {
                const block_q4_0x4 * b_ptr = (const block_q4_0x4 *) vx + (x * nb);
                for (int m = 0; m < 4; m++) {
                    for (int j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
                }
                for (int l = 0; l < nb; l++) {
                    for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                        for (int m = 0; m < 4; m++) {
                            for (int j = 0; j < ncols_interleaved; j++) {
                                sumi = 0;
                                for (int i = 0; i < blocklen; ++i) {
                                    const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                                    const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                                    sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                            (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4])) >> 4;
                                }
                                sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                            }
                        }
                    }
                }
                for (int m = 0; m < 4; m++) {
                    for (int j = 0; j < ncols_interleaved; j++)
                        s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}

void ggml_gemm_q4_0_4x8_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 4;
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

    float sumf[4][4];
    int sumi;

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q4_0x4 * b_ptr = (const block_q4_0x4 *) vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            for (int l = 0; l < nb; l++) {
                for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            for (int i = 0; i < blocklen; ++i) {
                                const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                                const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                        (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4])) >> 4;
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

void ggml_gemm_q4_0_8x8_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
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

    float sumf[4][8];
    int sumi;

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q4_0x8 * b_ptr = (const block_q4_0x8 *) vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            for (int l = 0; l < nb; l++) {
                for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            for (int i = 0; i < blocklen; ++i) {
                                const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                                const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                         (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4])) >> 4;
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

void ggml_gemm_q4_K_8x4_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 4;
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    assert (n % qk == 0);
    assert (nr % 4 == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

    float sumf[4][8];
    float sum_minf[4][8];
    uint32_t utmp[32];
    int sumi1;
    int sumi2;
    int sumi;

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_Kx4 * a_ptr = (const block_q8_Kx4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q4_Kx8 * b_ptr = (const block_q4_Kx8 *) vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                    sum_minf[m][j] = 0.0;
                }
            }
            for (int l = 0; l < nb; l++) {
                for (int sb = 0; sb < 8; sb++) {
                    memcpy(utmp + sb * 4, b_ptr[l].scales + sb * 12, 12);
                    utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                    utmp[sb * 4 + 1] = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                    utmp[sb * 4 + 2] = uaux_0;
                    utmp[sb * 4 + 0] &= kmask1;
                }
                for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                    uint8_t * scales_0 = (uint8_t *) utmp + (k / 8) * 32;
                    uint8_t * scales_1 = (uint8_t *) utmp + (k / 8) * 32 + 16;
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sumi1 = 0;
                            sumi2 = 0;
                            sumi = 0;
                            for (int i = 0; i < blocklen; ++i) {
                                const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF);
                                const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4);
                                sumi1 = (v0 * a_ptr[l].qs[(k / 8) * 256 + (k % 8) * 4 * blocklen + m * blocklen + i]);
                                sumi2 = (v1 * a_ptr[l].qs[(k / 8) * 256 + (k % 8) * 4 * blocklen + m * blocklen + i + 128]);
                                sumi1 = sumi1 * scales_0[j];
                                sumi2 = sumi2 * scales_1[j];
                                sumi += sumi1 + sumi2;
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d[m];
                        }
                    }
                }
                for (int sb = 0; sb < 8; sb++) {
                    uint8_t * mins = (uint8_t *) utmp + 8 + sb * 16;
                    for(int m = 0; m < 4; m++) {
                        const int16_t * bsums = a_ptr[l].bsums + (sb * 8) + (m * 4) - ((sb % 2) * 6);
                        for(int j = 0; j < ncols_interleaved; j++) {
                            sum_minf[m][j] += mins[j] * (bsums[0] + bsums[1]) * GGML_CPU_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d[m];
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j] - sum_minf[m][j];
                }
            }
        }
    }
}

void ggml_gemm_q4_K_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
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

    UNUSED(bs);

    float sumf[4][8];
    float sum_minf[4][8];
    uint32_t utmp[32];
    int sumi1;
    int sumi2;
    int sumi;

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_Kx4 * a_ptr = (const block_q8_Kx4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q4_Kx8 * b_ptr = (const block_q4_Kx8 *) vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                    sum_minf[m][j] = 0.0;
                }
            }
            for (int l = 0; l < nb; l++) {
                for (int sb = 0; sb < 8; sb++) {
                    memcpy(utmp + sb * 4, b_ptr[l].scales + sb * 12, 12);
                    utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                    utmp[sb * 4 + 1] = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                    utmp[sb * 4 + 2] = uaux_0;
                    utmp[sb * 4 + 0] &= kmask1;
                }
                for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                    uint8_t *scales_0 = (uint8_t*) utmp + (k / 4) * 32;
                    uint8_t *scales_1 = (uint8_t*) utmp + (k / 4) * 32 + 16;
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sumi1 = 0;
                            sumi2 = 0;
                            sumi = 0;
                            for (int i = 0; i < blocklen; ++i) {
                                const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF);
                                const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4);
                                sumi1 = (v0 * a_ptr[l].qs[(k >> 2) * 256 + (k % 4) * 4 * blocklen + m * blocklen + i]);
                                sumi2 = (v1 * a_ptr[l].qs[(k >> 2) * 256 + (k % 4) * 4 * blocklen + m * blocklen + i + 128]);
                                sumi1 = sumi1 * scales_0[j];
                                sumi2 = sumi2 * scales_1[j];
                                sumi += sumi1 + sumi2;
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d[m];
                        }
                    }
                }
                for (int sb = 0; sb < 8; sb++) {
                    uint8_t *mins = (uint8_t*) utmp + 8 + sb * 16;
                    for(int m = 0; m < 4; m++) {
                        const int16_t *bsums = a_ptr[l].bsums + (sb * 8) + (m * 4) - ((sb % 2) * 6);
                        for(int j = 0; j < ncols_interleaved; j++) {
                            sum_minf[m][j] += mins[j] * (bsums[0] + bsums[1]) * GGML_CPU_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d[m];
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j] - sum_minf[m][j];
                }
            }
        }
    }
}

void ggml_gemm_q2_K_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
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

    float sumf[4][8];
    float sum_minf[4][8];
    int sumi1, sumi2, sumi3, sumi4;
    int sumi;

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_Kx4 * a_ptr = (const block_q8_Kx4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q2_Kx8 * b_ptr = (const block_q2_Kx8 *) vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                    sum_minf[m][j] = 0.0;
                }
            }
            for (int l = 0; l < nb; l++) {
                for (int k = 0; k < (qk / (4 * blocklen)); k++) {

                    const uint8_t *scales_0 = b_ptr[l].scales + (k / 4) * 64 ;
                    const uint8_t *scales_1 = b_ptr[l].scales + (k / 4) * 64 + 16;
                    const uint8_t *scales_2 = b_ptr[l].scales + (k / 4) * 64 + 32;
                    const uint8_t *scales_3 = b_ptr[l].scales + (k / 4) * 64 + 48;
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sumi1 = 0;
                            sumi2 = 0;
                            sumi3 = 0;
                            sumi4 = 0;
                            sumi = 0;
                            int offset = ((k / 2) % 2) + j * 2;
                            for (int i = 0; i < blocklen; ++i){
                                const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 3);
                                const int v1 = (int8_t) ((b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 2 ) & 3);
                                const int v2 = (int8_t) ((b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4 ) & 3);
                                const int v3 = (int8_t) ((b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 6 ) & 3);
                                sumi1 = (v0 * a_ptr[l].qs[(k >> 2) * 512 + (k % 4) * 4 * blocklen + m * blocklen + i]);
                                sumi2 = (v1 * a_ptr[l].qs[(k >> 2) * 512  + (k % 4) * 4 * blocklen + m * blocklen + i + 128]);
                                sumi3 = (v2 * a_ptr[l].qs[(k >> 2) * 512  + (k % 4) * 4 * blocklen + m * blocklen + i + 256]);
                                sumi4 = (v3 * a_ptr[l].qs[(k >> 2) * 512  + (k % 4) * 4 * blocklen + m * blocklen + i + 384]);
                                sumi1 = sumi1 * (scales_0[offset] & 0xF);
                                sumi2 = sumi2 * (scales_1[offset] & 0xF);
                                sumi3 = sumi3 * (scales_2[offset] & 0xF);
                                sumi4 = sumi4 * (scales_3[offset] & 0xF);
                                sumi += sumi1 + sumi2 + sumi3 + sumi4;
                            }
                            sumf[m][j] += sumi * GGML_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d[m];
                        }
                    }
                }
                for(int sb = 0; sb < 8; sb++) {
                    const uint8_t *mins = b_ptr[l].scales + sb * 16;
                    for(int m = 0; m < 4; m++) {
                        const int16_t *bsums = a_ptr[l].bsums + (sb * 8) + (m * 4) - ((sb % 2) *  6);
                        for(int j = 0; j < ncols_interleaved; j++) {
                            int mins_prod = ((mins[j * 2] >> 4) * bsums[0] + (mins[(j * 2)+ 1] >> 4) * bsums[1]);
                            sum_minf[m][j] += (mins_prod) * GGML_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d[m];
                        }
                    }
                }
            }

            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j] - sum_minf[m][j];
                }
            }
        }
    }
}

void ggml_gemm_q5_K_8x4_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    ggml_gemm_q5_K_NxM_q8_K_generic_impl<4, 8>(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemm_q5_K_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    ggml_gemm_q5_K_NxM_q8_K_generic_impl<8, 8>(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemm_q6_K_8x4_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    ggml_gemm_q6_K_NxM_q8_K_generic_impl<4, 8>(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemm_q6_K_8x8_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
   ggml_gemm_q6_K_NxM_q8_K_generic_impl<8, 8>(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemm_iq4_nl_4x4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen = 4;

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

    {
        float sumf[4][4];
        int sumi;

        for (int y = 0; y < nr / 4; y++) {
            const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + (y * nb);
            for (int x = 0; x < nc / ncols_interleaved; x++) {
                const block_iq4_nlx4 * b_ptr = (const block_iq4_nlx4 *) vx + (x * nb);
                for (int m = 0; m < 4; m++) {
                    for (int j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
                }
                for (int l = 0; l < nb; l++) {
                    for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                        for (int m = 0; m < 4; m++) {
                            for (int j = 0; j < ncols_interleaved; j++) {
                                sumi = 0;
                                for (int i = 0; i < blocklen; ++i) {
                                    const int v0 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                                    const int v1 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                                    sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                            (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4]));
                                }
                                sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                            }
                        }
                    }
                }
                for (int m = 0; m < 4; m++) {
                    for (int j = 0; j < ncols_interleaved; j++)
                        s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}

void ggml_gemm_iq4_nl_8x8_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    float sumf[4][8];
    int sumi;

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_iq4_nlx8 * b_ptr = (const block_iq4_nlx8 *) vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            for (int l = 0; l < nb; l++) {
                for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            for (int i = 0; i < blocklen; ++i) {
                                const int v0 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                                const int v1 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                         (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4]));
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

void ggml_gemm_mxfp4_4x4_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen = 4;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    float sumf[4][4];
    int sumi;

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_mxfp4x4 * b_ptr = (const block_mxfp4x4 *) vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            for (int l = 0; l < nb; l++) {
                for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            for (int i = 0; i < blocklen; ++i) {
                                const int v0 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                                const int v1 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                         (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4]));
                            }
                            sumf[m][j] += sumi * GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[l].e[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

void ggml_gemm_mxfp4_8x8_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    float sumf[4][8];
    int sumi;

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_mxfp4x8 * b_ptr = (const block_mxfp4x8 *) vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            for (int l = 0; l < nb; l++) {
                for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            for (int i = 0; i < blocklen; ++i) {
                                const int v0 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                                const int v1 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                         (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4]));
                            }
                            sumf[m][j] += sumi * GGML_CPU_E8M0_TO_FP32_HALF(b_ptr[l].e[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

void ggml_gemm_q8_0_4x4_q8_0_generic(int                        n,
                                     float * GGML_RESTRICT      s,
                                     size_t                     bs,
                                     const void * GGML_RESTRICT vx,
                                     const void * GGML_RESTRICT vy,
                                     int                        nr,
                                     int                        nc) {
    const int qk                = QK8_0;
    const int nb                = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen          = 4;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    float sumf[4][4];
    int   sumi;

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q8_0x4 * b_ptr = (const block_q8_0x4 *) vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                }
            }
            for (int l = 0; l < nb; l++) {
                for (int k = 0; k < (qk / blocklen); k++) {
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            for (int i = 0; i < blocklen; ++i) {
                                const int v0 = b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                                sumi += v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i];
                            }
                            sumf[m][j] +=
                                sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}



void ggml_gemm_q8_0_4x8_q8_0_generic(int                        n,
                                     float * GGML_RESTRICT      s,
                                     size_t                     bs,
                                     const void * GGML_RESTRICT vx,
                                     const void * GGML_RESTRICT vy,
                                     int                        nr,
                                     int                        nc) {
    const int qk                = QK8_0;
    const int nb                = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen          = 8;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    float sumf[4][4];
    int   sumi;

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q8_0x4 * b_ptr = (const block_q8_0x4 *) vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                }
            }
            for (int l = 0; l < nb; l++) {
                for (int k = 0; k < (qk / blocklen); k++) {
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            for (int i = 0; i < blocklen; ++i) {
                                const int v0 = b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                                sumi += v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i];
                            }
                            sumf[m][j] +=
                                sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}

void ggml_gemm_q8_0_8x8_q8_0_generic(
        int n, float * GGML_RESTRICT s, size_t bs,
        const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy,
        int nr, int nc) {
    constexpr int ncols_interleaved = 8;
    constexpr int values_per_subblock = 16;

    assert(n % QK8_0 == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    const int nb = n / QK8_0;
    const block_q8_0x8 * b_ptr_start = (const block_q8_0x8 *) vx;
    const block_q8_0x4 * a_ptr_start = (const block_q8_0x4 *) vy;

    for (int y = 0; y < nr / 4; ++y) {
        const block_q8_0x4 * a_ptr = a_ptr_start + y * nb;
        for (int x = 0; x < nc / ncols_interleaved; ++x) {
            const block_q8_0x8 * b_ptr = b_ptr_start + x * nb;
            float sumf[4][ncols_interleaved] = {};

            for (int b = 0; b < nb; ++b) {
                for (int m = 0; m < 4; ++m) {
                    int isum[ncols_interleaved] = {};
                    for (int sb = 0; sb < QK8_0 / values_per_subblock; ++sb) {
                        for (int row = 0; row < ncols_interleaved; ++row) {
                            const int row_group = row / 4;
                            const int row_in_group = row % 4;
                            const int weight_offset =
                                (sb * 2 + row_group) * 64 + row_in_group * values_per_subblock;
                            for (int half = 0; half < 2; ++half) {
                                const int activation_offset =
                                    (sb * 2 + half) * 32 + m * 8;
                                for (int k = 0; k < 8; ++k) {
                                    const int weight =
                                        (int) (uint8_t) b_ptr[b].qs[weight_offset + half * 8 + k] - 128;
                                    isum[row] += weight * a_ptr[b].qs[activation_offset + k];
                                }
                            }
                        }
                    }

                    const float activation_scale = GGML_CPU_FP16_TO_FP32(a_ptr[b].d[m]);
                    for (int row = 0; row < ncols_interleaved; ++row) {
                        sumf[m][row] += isum[row] *
                            GGML_CPU_FP16_TO_FP32(b_ptr[b].d[row]) * activation_scale;
                    }
                }
            }

            for (int m = 0; m < 4; ++m) {
                for (int row = 0; row < ncols_interleaved; ++row) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + row] = sumf[m][row];
                }
            }
        }
    }
}

// Only enable these for RISC-V.
#if defined __riscv_zvfh
void ggml_gemm_q4_0_16x1_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 16;
    const int blocklen = 1;

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

    float sumf[4][16];
    int sumi;

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q4_0x16 * b_ptr = (const block_q4_0x16 *) vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            for (int l = 0; l < nb; l++) {
                for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            for (int i = 0; i < blocklen; ++i) {
                                const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                                const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                         (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4])) >> 4;
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

void ggml_gemm_q4_K_16x1_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_interleaved = 16;
    const int blocklen = 1;

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

    float sumf[4][16];
    float sum_minf[4][16];
    uint8_t scales[128];
    uint8_t mins[128];
    int sumi1;
    int sumi2;
    int sumi;

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_Kx4 * a_ptr = (const block_q8_Kx4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q4_Kx16 * b_ptr = (const block_q4_Kx16 *) vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                    sum_minf[m][j] = 0.0;
                }
            }
            for (int l = 0; l < nb; l++) {
                for (int i = 0; i < 128; i++) {
                    scales[i] = b_ptr[l].scales[i] & 0x0F;
                    mins[i] = b_ptr[l].scales[i] >> 4;
                }
                for (int i = 0; i < 64; i++) {
                    scales[i] |= (b_ptr[l].scales[128 + i] & 0x03) << 4;
                    mins[i] |= (b_ptr[l].scales[128 + i] & 0x0C) << 2;
                    scales[i + 64] |= (b_ptr[l].scales[128 + i] & 0x30);
                    mins[i + 64] |= (b_ptr[l].scales[128 + i] & 0xC0) >> 2;
                }

                for (int sb = 0; sb < 8; sb++) {
                    uint8_t *min = &mins[sb * 16];
                    for(int m = 0; m < 4; m++) {
                        const int16_t bsums = a_ptr[l].bsums[sb * 8 + m] + a_ptr[l].bsums[sb * 8 + m + 4];
                        for(int j = 0; j < ncols_interleaved; j++) {
                            sum_minf[m][j] += min[j] * bsums * GGML_CPU_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d[m];
                        }
                    }
                }

                for (int sb = 0; sb < 8; sb += 2) {
                    uint8_t *scales_0 = &scales[sb * 16];
                    uint8_t *scales_1 = &scales[(sb + 1) * 16];

                    for (int i = 0; i < QK4_0; i++) {
                        for (int m = 0; m < 4; m++) {
                            for (int j = 0; j < ncols_interleaved; j++) {
                                sumi1 = 0;
                                sumi2 = 0;
                                sumi = 0;

                                const int v0 = (int8_t) (b_ptr[l].qs[sb * 256 + i * 16 + j] & 0xF);
                                const int v1 = (int8_t) (b_ptr[l].qs[sb * 256 + i * 16 + j] >> 4);
                                sumi1 = (v0 * a_ptr[l].qs[sb * 4 * 32 + i * 4 + m]);
                                sumi2 = (v1 * a_ptr[l].qs[sb * 4 * 32 + 32 * 4 + i * 4 + m]);
                                sumi1 = sumi1 * scales_0[j];
                                sumi2 = sumi2 * scales_1[j];
                                sumi += sumi1 + sumi2;

                                sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d[m];
                            }
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j] - sum_minf[m][j];
                }
            }
        }
    }
}

void ggml_gemm_iq4_nl_16x1_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 16;
    const int blocklen = 1;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    float sumf[4][16];
    int sumi;

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_iq4_nlx16 * b_ptr = (const block_iq4_nlx16 *) vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            for (int l = 0; l < nb; l++) {
                for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            for (int i = 0; i < blocklen; ++i) {
                                const int v0 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                                const int v1 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                         (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + (qk / 2) * 4]));
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

void ggml_gemm_q8_0_16x1_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk                = QK8_0;
    const int nb                = n / qk;
    const int ncols_interleaved = 16;
    const int blocklen          = 1;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    float sumf[4][16];
    int   sumi;

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q8_0x16 * b_ptr = (const block_q8_0x16 *) vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                }
            }
            for (int l = 0; l < nb; l++) {
                for (int k = 0; k < (qk / blocklen); k++) {
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            for (int i = 0; i < blocklen; ++i) {
                                const int v0 = b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                                sumi += v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i];
                            }
                            sumf[m][j] +=
                                sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}


void ggml_gemm_q2_K_16x1_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    assert(n % QK_K == 0);
    assert(nr % 4 == 0);
    assert(nc % 16 == 0);
    const int nb = n / QK_K;
    const block_q2_Kx16 * x = (const block_q2_Kx16 *)vx;
    const block_q8_Kx4  * y = (const block_q8_Kx4 *)vy;

    const int sb_perm[16] = {
        0, 4, 1, 5, 2, 6, 3, 7,
        8, 12, 9, 13, 10, 14, 11, 15
    };

    // Iterate Rows in tiles of 4
    for (int row_tile = 0; row_tile < nr; row_tile += 4) {
        // Iterate Columns in tiles of 16
        for (int col_tile = 0; col_tile < nc; col_tile += 16) {

            const block_q2_Kx16 * x_ptr = x + (col_tile / 16) * nb;
            const block_q8_Kx4  * y_ptr = y + (row_tile / 4) * nb;

            float sumf[4][16];
            memset(sumf, 0, sizeof(sumf));

            for (int k_block = 0; k_block < nb; ++k_block) {
                int32_t isum[4][16];
                int32_t summs[4][16];
                memset(isum, 0, sizeof(isum));
                memset(summs, 0, sizeof(summs));

                const uint8_t * qs_rhs = x_ptr[k_block].qs;
                const uint8_t * sc_rhs = x_ptr[k_block].scales;
                const int8_t  * qs_lhs = y_ptr[k_block].qs;
                const int16_t * bs_lhs = y_ptr[k_block].bsums;

                for (int sb = 0; sb < 16; ++sb) {
                    int scale_offset = sb_perm[sb] * 16;

                    int byte_base;
                    if (sb < 8) byte_base = (sb % 2 == 0) ? 0 : 16;
                    else        byte_base = (sb % 2 == 0) ? 32 : 48;
                    int shift = ((sb / 2) % 4) * 2;

                    for (int col = 0; col < 16; ++col) {
                        uint8_t sc_val = sc_rhs[scale_offset + col];
                        int32_t d_sb = sc_val & 0xF;
                        int32_t m_sb = sc_val >> 4;

                        // Correction Term
                        for (int r = 0; r < 4; ++r) {
                            int bsum_idx = (sb / 4) * 16 + r * 4 + (sb % 4);
                            summs[r][col] += bs_lhs[bsum_idx] * m_sb;
                        }

                        // Main Dot Product
                        for (int l = 0; l < 16; ++l) {
                            int qs_idx = (byte_base + l) * 16 + col;
                            uint8_t q2_val = (qs_rhs[qs_idx] >> shift) & 3;

                            // Calculate Q8 index for this specific k and row
                            int k = sb * 16 + l;
                            int q8_idx = (k / 4) * 16 + (k % 4);

                            for (int r = 0; r < 4; ++r) {
                                // Add r*4 to jump to the correct row within the 4x4 chunk
                                int8_t q8_val = qs_lhs[q8_idx + r * 4];
                                isum[r][col] += q8_val * q2_val * d_sb;
                            }
                        }
                    }
                }

                // Finalize K-Block
                for (int col = 0; col < 16; ++col) {
                    float d_rhs = GGML_FP16_TO_FP32(x_ptr[k_block].d[col]);
                    float dm_rhs = GGML_FP16_TO_FP32(x_ptr[k_block].dmin[col]);

                    for (int r = 0; r < 4; ++r) {
                        float d_lhs = y_ptr[k_block].d[r];
                        float d_all = d_lhs * d_rhs;
                        float d_min = d_lhs * dm_rhs;
                        sumf[r][col] += (isum[r][col] * d_all) - (summs[r][col] * d_min);
                    }
                }
            }

            for (int r = 0; r < 4; ++r) {
                for (int col = 0; col < 16; ++col) {
                    s[(row_tile + r) * bs + (col_tile + col)] = sumf[r][col];
                }
            }
        }
    }
}
#endif

} // extern "C"

static block_q8_0x4 make_block_q8_0x4(block_q8_0 * in, unsigned int blck_size_interleave) {
    block_q8_0x4 out;

    for (int i = 0; i < 4; i++) {
        out.d[i] = in[i].d;
    }

    const int end = QK8_0 * 4 / blck_size_interleave;
    for (int i = 0; i < end; ++i) {
        int src_id     = i % 4;
        int src_offset = (i / 4) * blck_size_interleave;
        int dst_offset = i * blck_size_interleave;
        memcpy(&out.qs[dst_offset], &in[src_id].qs[src_offset], blck_size_interleave);
    }
    return out;
}

template <typename Function>
static void parallel_repack_groups(int64_t ngroups, Function && function);

// Eight Q8_0 rows arranged as four-row x 16-value VNNI tiles.  Q8_0 weights
// are signed, while Cascade Lake VNNI multiplies unsigned bytes by signed
// bytes.  Biasing every weight by 128 makes the packed input directly usable
// by VPDPBUSD; the GEMV kernel subtracts 128 * sum(q8 activation) exactly.
static block_q8_0x8 make_block_q8_0x8_vnni(const block_q8_0 * in) {
    block_q8_0x8 out = {};

    for (int row = 0; row < 8; ++row) {
        out.d[row] = in[row].d;
    }

    for (int half = 0; half < 2; ++half) {
        for (int row_group = 0; row_group < 2; ++row_group) {
            uint8_t * dst = (uint8_t *) out.qs + (half * 2 + row_group) * 64;
            for (int row = 0; row < 4; ++row) {
                for (int k = 0; k < 16; ++k) {
                    dst[row * 16 + k] =
                        ((const uint8_t *) in[row_group * 4 + row].qs)[half * 16 + k] ^ 0x80u;
                }
            }
        }
    }

    return out;
}

static int repack_q8_0_to_q8_0_8_vnni(
        struct ggml_tensor * t, const void * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q8_0);
    constexpr int nrows_interleaved = 8;

    const int64_t nrows = ggml_nrows(t);
    const int64_t nblocks = t->ne[0] / QK8_0;
    GGML_ASSERT(data_size == (size_t) nrows * nblocks * sizeof(block_q8_0));

    if (t->ne[0] % QK8_0 != 0 || t->ne[1] % nrows_interleaved != 0) {
        return -1;
    }

    if (const char * trace = std::getenv("GGML_CPU_Q8_0_REPACK_TRACE");
            trace != nullptr && strcmp(trace, "1") == 0) {
        fprintf(stderr, "q8_0_r8: repack ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]\n",
                t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
    }

    const block_q8_0 * src = (const block_q8_0 *) data;
    block_q8_0x8 * dst = (block_q8_0x8 *) t->data;
    const int64_t ngroups = nrows / nrows_interleaved;
    parallel_repack_groups(ngroups, [&](int64_t group) {
        const block_q8_0 * src_group = src + group * nrows_interleaved * nblocks;
        block_q8_0x8 * dst_group = dst + group * nblocks;
        block_q8_0 rows[nrows_interleaved];
        for (int64_t block = 0; block < nblocks; ++block) {
            for (int row = 0; row < nrows_interleaved; ++row) {
                rows[row] = src_group[block + row * nblocks];
            }
            dst_group[block] = make_block_q8_0x8_vnni(rows);
        }
    });

    return 0;
}

static block_q4_0x4 make_block_q4_0x4(block_q4_0 * in, int blck_size_interleave) {
    block_q4_0x4 out;

    for (int i = 0; i < 4; i++) {
        out.d[i] = in[i].d;
    }

    const int end = QK4_0 * 2 / blck_size_interleave;

    if (blck_size_interleave == 8) {
        const uint64_t xor_mask = 0x8888888888888888ULL;
        for (int i = 0; i < end; ++i) {
            int src_id = i % 4;
            int src_offset = (i / 4) * blck_size_interleave;
            int dst_offset = i * blck_size_interleave;

            uint64_t elems;
            // Using memcpy to avoid unaligned memory accesses
            memcpy(&elems, &in[src_id].qs[src_offset], sizeof(uint64_t));
            elems ^= xor_mask;
            memcpy(&out.qs[dst_offset], &elems, sizeof(uint64_t));
        }
    } else if (blck_size_interleave == 4) {
        const uint32_t xor_mask = 0x88888888;
        for (int i = 0; i < end; ++i) {
            int src_id = i % 4;
            int src_offset = (i / 4) * blck_size_interleave;
            int dst_offset = i * blck_size_interleave;

            uint32_t elems;
            memcpy(&elems, &in[src_id].qs[src_offset], sizeof(uint32_t));
            elems ^= xor_mask;
            memcpy(&out.qs[dst_offset], &elems, sizeof(uint32_t));
        }
    } else {
        GGML_ASSERT(false);
    }

    return out;
}

// interleave 8 block_q4_0s in blocks of blck_size_interleave
// returns an interleaved block_q4_0x8
// in the interleaved block_q4_0x8, place deltas for 8 block_q4_0 blocks
// first, then interleave quants from 8 block_q4_0s in blocks of blck_size_interleave
static block_q4_0x8 make_block_q4_0x8(block_q4_0 * in, unsigned int blck_size_interleave) {
    block_q4_0x8 out;

    for (int i = 0; i < 8; i++) {
        out.d[i] = in[i].d;
    }

    const int end = QK4_0 * 4 / blck_size_interleave;
    const uint64_t xor_mask = 0x8888888888888888ULL;

    for (int i = 0; i < end; ++i) {
        int src_id = i % 8;
        int src_offset = (i / 8) * blck_size_interleave;
        int dst_offset = i * blck_size_interleave;

        uint64_t elems;
        memcpy(&elems, &in[src_id].qs[src_offset], sizeof(uint64_t));
        elems ^= xor_mask;
        memcpy(&out.qs[dst_offset], &elems, sizeof(uint64_t));
    }

    return out;
}

static block_q4_0x16 make_block_q4_0x16(block_q4_0 * in, unsigned int blck_size_interleave) {
    block_q4_0x16 out;

    for (int i = 0; i < 16; i++) {
        out.d[i] = in[i].d;
    }

    const int end = QK4_0 * 8 / blck_size_interleave;

    if (blck_size_interleave == 1) {
        const uint8_t xor_mask = 0x88;
        for (int i = 0; i < end; ++i) {
            int src_id = i % 16;
            int src_offset = i / 16;
            int dst_offset = i;

            out.qs[dst_offset] = in[src_id].qs[src_offset] ^ xor_mask;
        }
    } else {
        GGML_ASSERT(false);
    }

    return out;
}

static block_q4_Kx8 make_block_q4_Kx8(block_q4_K * in, unsigned int blck_size_interleave) {
    block_q4_Kx8 out;
    //Delta(scale) and dmin values of the eight Q4_K structures are copied onto the output interleaved structure
    for (int i = 0; i < 8; i++) {
        out.d[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
    }

    for (int i = 0; i < 8; i++) {
        out.dmin[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
    }

    const int end = QK_K * 4 / blck_size_interleave;

    // Interleave Q4_K quants by taking 8 bytes at a time
    for (int i = 0; i < end; ++i) {
        int src_id = i % 8;
        int src_offset = (i / 8) * blck_size_interleave;
        int dst_offset = i * blck_size_interleave;

        // buffer large enough for the max interleave block size (8 bytes)
        uint64_t elems;
        memcpy(&elems, &in[src_id].qs[src_offset], blck_size_interleave);
        memcpy(&out.qs[dst_offset], &elems, blck_size_interleave);
    }

    // The below logic is designed so as to unpack and rearrange scales and mins values in Q4_K
    // Currently the Q4_K structure has 8 scales and 8 mins packed in 12 bytes ( 6 bits for each value)
    // The output Q4_Kx8 structure has 96 bytes
    // Every 12 byte is packed such that it contains scales and mins for corresponding sub blocks from Q4_K structure
    // For eg - First 12 bytes contains 8 scales and 8 mins - each of first sub block from different Q4_K structures
    uint8_t s[8], m[8];

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            s[j] = in[j].scales[i] & 63;
            m[j] = in[j].scales[i + 4] & 63;
        }

        out.scales[i * 12]      = (s[0] & 63) + ((s[4] & 48) << 2);
        out.scales[i * 12 + 1]  = (s[1] & 63) + ((s[5] & 48) << 2);
        out.scales[i * 12 + 2]  = (s[2] & 63) + ((s[6] & 48) << 2);
        out.scales[i * 12 + 3]  = (s[3] & 63) + ((s[7] & 48) << 2);
        out.scales[i * 12 + 4]  = (m[0] & 63) + ((m[4] & 48) << 2);
        out.scales[i * 12 + 5]  = (m[1] & 63) + ((m[5] & 48) << 2);
        out.scales[i * 12 + 6]  = (m[2] & 63) + ((m[6] & 48) << 2);
        out.scales[i * 12 + 7]  = (m[3] & 63) + ((m[7] & 48) << 2);
        out.scales[i * 12 + 8]  = (s[4] & 15) + ((m[4] & 15) << 4);
        out.scales[i * 12 + 9]  = (s[5] & 15) + ((m[5] & 15) << 4);
        out.scales[i * 12 + 10] = (s[6] & 15) + ((m[6] & 15) << 4);
        out.scales[i * 12 + 11] = (s[7] & 15) + ((m[7] & 15) << 4);

    }

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            s[j] = ((in[j].scales[i] & 192) >> 2) | (in[j].scales[i+8] & 15);
            m[j] = ((in[j].scales[i + 4] & 192) >> 2) | ((in[j].scales[i+8] & 240) >> 4);
        }

        out.scales[i * 12 + 48] = (s[0] & 63) + ((s[4] & 48) << 2);
        out.scales[i * 12 + 49] = (s[1] & 63) + ((s[5] & 48) << 2);
        out.scales[i * 12 + 50] = (s[2] & 63) + ((s[6] & 48) << 2);
        out.scales[i * 12 + 51] = (s[3] & 63) + ((s[7] & 48) << 2);
        out.scales[i * 12 + 52] = (m[0] & 63) + ((m[4] & 48) << 2);
        out.scales[i * 12 + 53] = (m[1] & 63) + ((m[5] & 48) << 2);
        out.scales[i * 12 + 54] = (m[2] & 63) + ((m[6] & 48) << 2);
        out.scales[i * 12 + 55] = (m[3] & 63) + ((m[7] & 48) << 2);
        out.scales[i * 12 + 56] = (s[4] & 15) + ((m[4] & 15) << 4);
        out.scales[i * 12 + 57] = (s[5] & 15) + ((m[5] & 15) << 4);
        out.scales[i * 12 + 58] = (s[6] & 15) + ((m[6] & 15) << 4);
        out.scales[i * 12 + 59] = (s[7] & 15) + ((m[7] & 15) << 4);

    }

    return out;
}

static block_q4_Kx16 make_block_q4_Kx16(block_q4_K * in, unsigned int blck_size_interleave) {
    block_q4_Kx16 out;
    //Delta(scale) and dmin values of the 16 Q4_K structures are copied onto the output interleaved structure
    for (int i = 0; i < 16; i++) {
        out.d[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
    }

    for (int i = 0; i < 16; i++) {
        out.dmin[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
    }

    const int end = QK_K * 8 / blck_size_interleave;

    if (blck_size_interleave == 1) {
        for (int i = 0; i < end; ++i) {
            int src_id = i % 16;
            int src_offset = i / 16;
            int dst_offset = i;

            out.qs[dst_offset] = in[src_id].qs[src_offset];
        }

        // RVV repacking.
        //
        // Extract sums and mins for all 8 sub-blocks for each block of Q4_K.
        uint8_t s[128], m[128];
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 16; j++) {
                s[i * 16 + j] = in[j].scales[i] & 63;
                m[i * 16 + j] = in[j].scales[i + 4] & 63;
            }
        }
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 16; j++) {
                s[64 + i * 16 + j] = ((in[j].scales[i] & 192) >> 2) | (in[j].scales[i+8] & 15);
                m[64 + i * 16 + j] = ((in[j].scales[i + 4] & 192) >> 2) | ((in[j].scales[i+8] & 240) >> 4);
            }
        }

        for (int i = 0; i < 128; i++) {
            out.scales[i] = (s[i] & 15) | ((m[i] & 15) << 4);
        }
        for (int i = 0; i < 64; i++) {
            out.scales[128 + i] = ((s[i] & 48) >> 4) | ((m[i] & 48) >> 2) | (s[64 + i] & 48) | ((m[64 + i] & 48) << 2);
        }
    } else {
        GGML_ASSERT(false);
    }

    return out;
}

static block_q2_Kx8 make_block_q2_Kx8(block_q2_K * in, unsigned int blck_size_interleave) {
    block_q2_Kx8 out;

    // Delta(scale) and dmin values of the eight Q2_K structures are copied onto the output interleaved structure
    for (int i = 0; i < 8; i++) {
        out.d[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
    }

    for (int i = 0; i < 8; i++) {
        out.dmin[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
    }

    const int end = QK_K * 2 / blck_size_interleave;

    // Interleave Q2_K quants by taking 8 bytes at a time
    for (int i = 0; i < end; ++i) {
        int src_id = i % 8;
        int src_offset = (i / 8) * blck_size_interleave;
        int dst_offset = i * blck_size_interleave;

        uint64_t elems;
        memcpy(&elems, &in[src_id].qs[src_offset], sizeof(uint64_t));
        memcpy(&out.qs[dst_offset], &elems, sizeof(uint64_t));
    }

    // The below logic is designed so as to unpack and rearrange scales and mins values in Q2_K
    // Currently the Q2_K structure has 16 scales and 16 mins packed in 16 bytes ( 4 bits for each value)
    // The output Q2_Kx8 structure has 128 bytes for storing scales and mins
    // Every 16 byte is packed such that it contains scales and mins for corresponding sub blocks from Q2_K structure
    // For eg - First 16 bytes contains 16 scales and 16 mins - each of first and second sub blocks from different Q2_K structures

    for (int i = 0; i < 128; i++) {
        // Index for selecting which q2k super block
        int src1 = (i % 16) / 2;
        // Index for selecting scale
        int src2 = ((i / 16) * 2) + (i % 2);

        out.scales[i] = in[src1].scales[src2];
    }
    return out;
}

static block_q5_Kx8 make_block_q5_Kx8(block_q5_K * in, unsigned int blck_size_interleave) {
    block_q5_Kx8 out;
    //Delta(scale) and dmin values of the eight Q5_K structures are copied onto the output interleaved structure
    for (int i = 0; i < 8; i++) {
        out.d[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
    }

    for (int i = 0; i < 8; i++) {
        out.dmin[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
    }

    const int end = QK_K * 4 / blck_size_interleave;

    // Interleave Q5_K quants by taking blck_size_interleave bytes at a time
    for (int i = 0; i < end; ++i) {
        int src_id     = i % 8;
        int src_offset = (i / 8) * blck_size_interleave;
        int dst_offset = i * blck_size_interleave;

        memcpy(&out.qs[dst_offset], &in[src_id].qs[src_offset], blck_size_interleave);
    }

    // Repeat for high bits with the same chunk size, since
    // the high bits are interleaved in Q5_K and the index is
    // qh_idx = (qs_idx % 32);
    // qh_val = qh[qh_idx] >> (qs_idx / 32);
    for (int i = 0; i < end / 4; ++i) {
        int src_id     = i % 8;
        int src_offset = (i / 8) * blck_size_interleave;
        int dst_offset = i * blck_size_interleave;

        memcpy(&out.qh[dst_offset], &in[src_id].qh[src_offset], blck_size_interleave);
    }

    // The below logic is copied over from Q4_K
    // The point is to unpack all the scales and mins for each sub block every time we load 12 bytes.
    // Currently the Q5_K structure has 8 scales and 8 mins packed in 12 bytes ( 6 bits for each value)
    // The output Q5_Kx8 structure has 96 bytes
    // Every 12 byte is packed such that it contains scales and mins for corresponding sub blocks from Q5_K structure
    // For eg - First 12 bytes contains 8 scales and 8 mins - each of first sub block from different Q5_K structures
    uint8_t s[8], m[8];

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            s[j] = in[j].scales[i] & 63;
            m[j] = in[j].scales[i + 4] & 63;
        }

        out.scales[i * 12]      = (s[0] & 63) + ((s[4] & 48) << 2);
        out.scales[i * 12 + 1]  = (s[1] & 63) + ((s[5] & 48) << 2);
        out.scales[i * 12 + 2]  = (s[2] & 63) + ((s[6] & 48) << 2);
        out.scales[i * 12 + 3]  = (s[3] & 63) + ((s[7] & 48) << 2);
        out.scales[i * 12 + 4]  = (m[0] & 63) + ((m[4] & 48) << 2);
        out.scales[i * 12 + 5]  = (m[1] & 63) + ((m[5] & 48) << 2);
        out.scales[i * 12 + 6]  = (m[2] & 63) + ((m[6] & 48) << 2);
        out.scales[i * 12 + 7]  = (m[3] & 63) + ((m[7] & 48) << 2);
        out.scales[i * 12 + 8]  = (s[4] & 15) + ((m[4] & 15) << 4);
        out.scales[i * 12 + 9]  = (s[5] & 15) + ((m[5] & 15) << 4);
        out.scales[i * 12 + 10] = (s[6] & 15) + ((m[6] & 15) << 4);
        out.scales[i * 12 + 11] = (s[7] & 15) + ((m[7] & 15) << 4);
    }

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            s[j] = ((in[j].scales[i] & 192) >> 2) | (in[j].scales[i + 8] & 15);
            m[j] = ((in[j].scales[i + 4] & 192) >> 2) | ((in[j].scales[i + 8] & 240) >> 4);
        }

        out.scales[i * 12 + 48] = (s[0] & 63) + ((s[4] & 48) << 2);
        out.scales[i * 12 + 49] = (s[1] & 63) + ((s[5] & 48) << 2);
        out.scales[i * 12 + 50] = (s[2] & 63) + ((s[6] & 48) << 2);
        out.scales[i * 12 + 51] = (s[3] & 63) + ((s[7] & 48) << 2);
        out.scales[i * 12 + 52] = (m[0] & 63) + ((m[4] & 48) << 2);
        out.scales[i * 12 + 53] = (m[1] & 63) + ((m[5] & 48) << 2);
        out.scales[i * 12 + 54] = (m[2] & 63) + ((m[6] & 48) << 2);
        out.scales[i * 12 + 55] = (m[3] & 63) + ((m[7] & 48) << 2);
        out.scales[i * 12 + 56] = (s[4] & 15) + ((m[4] & 15) << 4);
        out.scales[i * 12 + 57] = (s[5] & 15) + ((m[5] & 15) << 4);
        out.scales[i * 12 + 58] = (s[6] & 15) + ((m[6] & 15) << 4);
        out.scales[i * 12 + 59] = (s[7] & 15) + ((m[7] & 15) << 4);
    }

    return out;
}

static block_q6_Kx8 make_block_q6_Kx8(block_q6_K * in, unsigned int blck_size_interleave) {
    block_q6_Kx8  out;
    constexpr int n_blocks = 8;  // Kx8
    for (int i = 0; i < n_blocks; i++) {
        out.d[i] = in[i].d;
    }

    const int end_ls = QK_K * 4 / blck_size_interleave;
    // Interleave Q6_K quants by taking blck_size_interleave bytes at a time
    for (int i = 0; i < end_ls; ++i) {
        int src_id     = i % n_blocks;
        int src_offset = (i / n_blocks) * blck_size_interleave;
        int dst_offset = i * blck_size_interleave;

        uint64_t elem_ls;
        memcpy(&elem_ls, &in[src_id].ql[src_offset], blck_size_interleave);
        memcpy(&out.ql[dst_offset], &elem_ls, blck_size_interleave);
    }

    // Interleave high bits using same chunk size as low bits
    const int end_hs = end_ls / 2;
    for (int i = 0; i < end_hs; ++i) {
        int src_id     = i % n_blocks;
        int src_offset = (i / n_blocks) * blck_size_interleave;
        int dst_offset = i * blck_size_interleave;

        uint64_t elem_hs;
        memcpy(&elem_hs, &in[src_id].qh[src_offset], blck_size_interleave);
        memcpy(&out.qh[dst_offset], &elem_hs, blck_size_interleave);
    }

    // The below logic is designed so as to unpack and rearrange scales in Q6_K
    // The output Q6_Kx8 structure interleaves the 8 bit scales in the same fashion as the quants
    // Q6_K structure has an 8-bit scale per 16 elements -> 16 scales
    // scales: [0 bl0 0 bl1 ... 0 bl7][1 bl0 ... 1 bl7] ... [15 bl0 ... 15 bl7]  (bl = block)
    constexpr int n_scales = QK_K / 16;

    for (int i = 0; i < n_blocks; i++) {
        for (int j = 0; j < n_scales; j++) {
            out.scales[j * n_blocks + i] = in[i].scales[j];
        }
    }

    return out;
}

static block_q2_Kx16 make_block_q2_Kx16(const block_q2_K * in, unsigned int blck_size_interleave) {
    block_q2_Kx16 out;
    constexpr int N_COLS = 16;

    // 1. Copy Super-Scales (d) and Super-Mins (dmin)
    for (int i = 0; i < N_COLS; i++) {
        out.d[i]    = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
        out.dmin[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
    }

    // 2. Interleave Q2_K Data
    const int bytes_per_col = 64;
    const int total_bytes = N_COLS * bytes_per_col;
    const int end = total_bytes / blck_size_interleave;

    for (int i = 0; i < end; ++i) {
        int src_col_id = i % N_COLS;
        int src_offset = (i / N_COLS) * blck_size_interleave;
        int dst_offset = i * blck_size_interleave;
        memcpy(&out.qs[dst_offset], &in[src_col_id].qs[src_offset], blck_size_interleave);
    }

    // 3. Repack Scales into the Optimized "Sequential-Parallel" Layout
    int out_idx = 0;

    // Arrays define the sub-block order for each group
    const int even_low_sbs[]  = {0, 2, 4, 6};
    const int odd_low_sbs[]   = {1, 3, 5, 7};
    const int even_high_sbs[] = {8, 10, 12, 14};
    const int odd_high_sbs[]  = {9, 11, 13, 15};

    // Pack Group 1: Even-Low
    for (int sb : even_low_sbs) {
        for (int col = 0; col < N_COLS; col++) {
            out.scales[out_idx++] = in[col].scales[sb];
        }
    }

    // Pack Group 2: Odd-Low
    for (int sb : odd_low_sbs) {
        for (int col = 0; col < N_COLS; col++) {
            out.scales[out_idx++] = in[col].scales[sb];
        }
    }

    // Pack Group 3: Even-High
    for (int sb : even_high_sbs) {
        for (int col = 0; col < N_COLS; col++) {
            out.scales[out_idx++] = in[col].scales[sb];
        }
    }

    // Pack Group 4: Odd-High
    for (int sb : odd_high_sbs) {
        for (int col = 0; col < N_COLS; col++) {
            out.scales[out_idx++] = in[col].scales[sb];
        }
    }

    return out;
}

static int repack_q4_0_to_q4_0_4_bl(struct ggml_tensor * t, int interleave_block, const void * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(interleave_block == 4 || interleave_block == 8);
    constexpr int nrows_interleaved = 4;

    block_q4_0x4 * dst = (block_q4_0x4 *)t->data;
    const block_q4_0 * src = (const block_q4_0 *)data;
    block_q4_0 dst_tmp[4];
    int nrow = ggml_nrows(t);
    int nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_q4_0));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q4_0x4(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    GGML_UNUSED(data_size);
}

static int repack_q4_K_to_q4_K_8_bl(struct ggml_tensor * t, int interleave_block, const void * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_K);
    GGML_ASSERT(interleave_block == 8 || interleave_block == 4);
    constexpr int nrows_interleaved = 8;

    block_q4_Kx8 * dst = (block_q4_Kx8*)t->data;
    const block_q4_K * src = (const block_q4_K*) data;
    block_q4_K dst_tmp[8];
    int nrow = ggml_nrows(t);
    int nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_q4_K));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i  = 0; i < nrows_interleaved; i++ ) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q4_Kx8(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    GGML_UNUSED(data_size);
}

static int repack_q4_K_to_q4_K_16_bl(struct ggml_tensor * t, int interleave_block, const void * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_K);
    constexpr int nrows_interleaved = 16;

    block_q4_Kx16 * dst = (block_q4_Kx16*)t->data;
    const block_q4_K * src = (const block_q4_K*) data;
    block_q4_K dst_tmp[16];
    int nrow = ggml_nrows(t);
    int nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_q4_K));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i  = 0; i < nrows_interleaved; i++ ) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q4_Kx16(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    GGML_UNUSED(data_size);
}

static int repack_q2_K_to_q2_K_8_bl(struct ggml_tensor * t, int interleave_block, const void * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q2_K);
    GGML_ASSERT(interleave_block == 8);
    constexpr int nrows_interleaved = 8;

    block_q2_Kx8 * dst = (block_q2_Kx8*)t->data;
    const block_q2_K * src = (const block_q2_K*) data;
    block_q2_K dst_tmp[8];
    int nrow = ggml_nrows(t);
    int nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_q2_K));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q2_Kx8(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    GGML_UNUSED(data_size);
}

static int repack_q2_K_to_q2_K_16_bl(struct ggml_tensor * t, int interleave_block, const void * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q2_K);
    constexpr int nrows_interleaved = 16;

    block_q2_Kx16 * dst = (block_q2_Kx16*)t->data;
    const block_q2_K * src = (const block_q2_K*) data;

    block_q2_K dst_tmp[nrows_interleaved];

    int nrow = ggml_nrows(t);
    int nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_q2_K));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            // This loop gathers 16 separate blocks (one from each column)
            // that correspond to the same K-dimension chunk.
            for (int i  = 0; i < nrows_interleaved; i++ ) {
                dst_tmp[i] = src[x + i * nblocks];
            }

            *dst++ = make_block_q2_Kx16(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    GGML_UNUSED(data_size);
}

static uint8_t iq2_xs_r8_code(uint8_t magnitude, bool negative) {
    const int value = negative ? -(int) magnitude : (int) magnitude;
    switch (value) {
        case -43: return 0;
        case -25: return 1;
        case  -8: return 2;
        case   8: return 3;
        case  25: return 4;
        case  43: return 5;
        default: GGML_ABORT("unexpected IQ2_XS alphabet value: %d", value);
    }
}

template <typename Function>
static void parallel_repack_groups(int64_t ngroups, Function && function) {
    static const int load_threads = []() {
        const char * value = std::getenv("GGML_CPU_REPACK_LOAD_THREADS");
        return value == nullptr ? 1 : std::clamp(std::atoi(value), 1, 16);
    }();

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(load_threads) if(load_threads > 1 && ngroups >= load_threads)
    for (int64_t group = 0; group < ngroups; ++group) {
        function(group);
    }
#else
    if (load_threads <= 1 || ngroups < load_threads) {
        for (int64_t group = 0; group < ngroups; ++group) {
            function(group);
        }
        return;
    }

    std::vector<std::thread> workers;
    workers.reserve(load_threads);
    for (int thread = 0; thread < load_threads; ++thread) {
        workers.emplace_back([&, thread]() {
            for (int64_t group = thread; group < ngroups; group += load_threads) {
                function(group);
            }
        });
    }
    for (std::thread & worker : workers) {
        worker.join();
    }
#endif
}

static block_iq2_xs_r8 make_block_iq2_xs_r8(const block_iq2_xs * in) {
    block_iq2_xs_r8 out = {};
    constexpr int nrows_interleaved = 8;
    constexpr int values_per_subblock = 16;

    for (int row = 0; row < nrows_interleaved; ++row) {
        out.d[row] = in[row].d;
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            out.scales[ib32 * nrows_interleaved + row] = in[row].scales[ib32];
        }
    }

    for (int sb = 0; sb < QK_K / values_per_subblock; ++sb) {
        const int ib32 = sb / 2;
        const int offset32 = (sb & 1) * values_per_subblock;
        for (int row = 0; row < nrows_interleaved; ++row) {
            uint8_t * packed_codes = out.qs + (sb * 2 + row / 4) * 32;
            for (int k = 0; k < values_per_subblock; ++k) {
                const int pos32 = offset32 + k;
                const int grid_part = pos32 / 8;
                const int grid_pos = pos32 % 8;
                const uint16_t packed_grid = in[row].qs[4 * ib32 + grid_part];
                const uint8_t * grid = (const uint8_t *) (iq2xs_grid + (packed_grid & 511));
                const uint8_t signs = ksigns_iq2xs[packed_grid >> 9];
                const uint8_t code = iq2_xs_r8_code(
                    grid[grid_pos], (signs & kmask_iq2xs[grid_pos]) != 0);
                const int code_index = (row % 4) * values_per_subblock + k;
                if (code_index & 1) {
                    packed_codes[code_index / 2] |= code << 4;
                } else {
                    packed_codes[code_index / 2] = code;
                }
            }
        }
    }

    return out;
}

static int repack_iq2_xs_to_iq2_xs_r8(
        struct ggml_tensor * t, const void * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_IQ2_XS);
    constexpr int nrows_interleaved = 8;

    const int64_t nrows = ggml_nrows(t);
    const int64_t nblocks = t->ne[0] / QK_K;
    GGML_ASSERT(data_size == (size_t) nrows * nblocks * sizeof(block_iq2_xs));

    if (t->ne[0] % QK_K != 0 || t->ne[1] % nrows_interleaved != 0) {
        return -1;
    }

    const block_iq2_xs * src = (const block_iq2_xs *) data;
    block_iq2_xs_r8 * dst = (block_iq2_xs_r8 *) t->data;
    const int64_t ngroups = nrows / nrows_interleaved;
    parallel_repack_groups(ngroups, [&](int64_t group) {
        const block_iq2_xs * src_group =
            src + group * nrows_interleaved * nblocks;
        block_iq2_xs_r8 * dst_group = dst + group * nblocks;
        block_iq2_xs rows[nrows_interleaved];
        for (int64_t block = 0; block < nblocks; ++block) {
            for (int i = 0; i < nrows_interleaved; ++i) {
                rows[i] = src_group[block + i * nblocks];
            }
            dst_group[block] = make_block_iq2_xs_r8(rows);
        }
    });

    return 0;
}

static uint8_t iq3_xxs_r8_code(uint8_t magnitude, bool negative) {
    const int value = negative ? -(int) magnitude : (int) magnitude;
    switch (value) {
        case -62: return 0;
        case -52: return 1;
        case -44: return 2;
        case -36: return 3;
        case -28: return 4;
        case -20: return 5;
        case -12: return 6;
        case  -4: return 7;
        case   4: return 8;
        case  12: return 9;
        case  20: return 10;
        case  28: return 11;
        case  36: return 12;
        case  44: return 13;
        case  52: return 14;
        case  62: return 15;
        default: GGML_ABORT("unexpected IQ3_XXS alphabet value: %d", value);
    }
}

static block_iq3_xxs_r8 make_block_iq3_xxs_r8(const block_iq3_xxs * in) {
    block_iq3_xxs_r8 out = {};
    constexpr int nrows_interleaved = 8;
    constexpr int values_per_subblock = 16;

    for (int row = 0; row < nrows_interleaved; ++row) {
        out.d[row] = in[row].d;
    }

    for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
        for (int row = 0; row < nrows_interleaved; ++row) {
            uint32_t aux32;
            memcpy(&aux32, in[row].qs + QK_K / 4 + 4 * ib32, sizeof(aux32));
            out.scales[ib32] |= (aux32 >> 28) << (4 * row);

            for (int pos32 = 0; pos32 < 32; ++pos32) {
                const int l = pos32 / 8;
                const int pos8 = pos32 % 8;
                const int grid_index = ib32 * 8 + 2 * l + pos8 / 4;
                const uint8_t * grid = (const uint8_t *) (iq3xxs_grid + in[row].qs[grid_index]);
                const uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
                const uint8_t code = iq3_xxs_r8_code(
                    grid[pos8 % 4], (signs & kmask_iq2xs[pos8]) != 0);
                const int sb = ib32 * 2 + pos32 / values_per_subblock;
                const int k = pos32 % values_per_subblock;
                uint8_t * packed_codes = out.qs + (sb * 2 + row / 4) * 32;
                const int code_index = (row % 4) * values_per_subblock + k;
                if (code_index & 1) {
                    packed_codes[code_index / 2] |= code << 4;
                } else {
                    packed_codes[code_index / 2] = code;
                }
            }
        }
    }

    return out;
}

static int repack_iq3_xxs_to_iq3_xxs_r8(
        struct ggml_tensor * t, const void * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_IQ3_XXS);
    constexpr int nrows_interleaved = 8;

    const int64_t nrows = ggml_nrows(t);
    const int64_t nblocks = t->ne[0] / QK_K;
    GGML_ASSERT(data_size == (size_t) nrows * nblocks * sizeof(block_iq3_xxs));

    if (t->ne[0] % QK_K != 0 || t->ne[1] % nrows_interleaved != 0) {
        return -1;
    }

    const block_iq3_xxs * src = (const block_iq3_xxs *) data;
    block_iq3_xxs_r8 * dst = (block_iq3_xxs_r8 *) t->data;
    const int64_t ngroups = nrows / nrows_interleaved;
    parallel_repack_groups(ngroups, [&](int64_t group) {
        const block_iq3_xxs * src_group =
            src + group * nrows_interleaved * nblocks;
        block_iq3_xxs_r8 * dst_group = dst + group * nblocks;
        block_iq3_xxs rows[nrows_interleaved];
        for (int64_t block = 0; block < nblocks; ++block) {
            for (int i = 0; i < nrows_interleaved; ++i) {
                rows[i] = src_group[block + i * nblocks];
            }
            dst_group[block] = make_block_iq3_xxs_r8(rows);
        }
    });

    return 0;
}

// Four codes per output channel let each VNNI lane produce one row sum.
struct block_iq_r16 {
    ggml_half d[16];
    uint8_t scales[(QK_K / 32) * 16];
    uint8_t qs[(QK_K / 16) * 4 * 32];
};
static_assert(sizeof(block_iq_r16) == 2208, "wrong block_iq_r16 size/padding");

static bool iq_r16_paired_nibbles() {
    static const bool enabled = [] {
        const char * value = getenv("GGML_CPU_IQ_R16_NIBBLE2");
        return value && atoi(value) != 0;
    }();
    return enabled;
}

template <typename BLOC_TYPE>
static int repack_iq_to_r16(ggml_tensor * t, const void * data, size_t data_size) {
    const int64_t nrows = ggml_nrows(t);
    const int64_t nb = t->ne[0] / QK_K;
    GGML_ASSERT(data_size == nrows * nb * sizeof(BLOC_TYPE));
    if (t->ne[0] % QK_K != 0 || t->ne[1] % 16 != 0) {
        return -1;
    }
    const BLOC_TYPE * src = (const BLOC_TYPE *) data;
    block_iq_r16 * dst = (block_iq_r16 *) t->data;
    parallel_repack_groups(nrows / 16, [&](int64_t group) {
        for (int64_t b = 0; b < nb; ++b) {
            block_iq_r16 & out = dst[group * nb + b];
            memset(&out, 0, sizeof(out));
            for (int half = 0; half < 2; ++half) {
                BLOC_TYPE rows[8];
                for (int r = 0; r < 8; ++r) {
                    rows[r] = src[(group * 16 + half * 8 + r) * nb + b];
                }
                const auto r8 = [&]() {
                    if constexpr (std::is_same_v<BLOC_TYPE, block_iq3_xxs>) {
                        return make_block_iq3_xxs_r8(rows);
                    } else {
                        return make_block_iq2_xs_r8(rows);
                    }
                }();
                for (int r = 0; r < 8; ++r) {
                    const int row = half * 8 + r;
                    out.d[row] = r8.d[r];
                    for (int sb = 0; sb < QK_K / 32; ++sb) {
                        if constexpr (std::is_same_v<BLOC_TYPE, block_iq3_xxs>) {
                            const uint8_t scale = (r8.scales[sb] >> (4 * r)) & 15;
                            out.scales[sb * 16 + row] = scale | (scale << 4);
                        } else {
                            out.scales[sb * 16 + row] = r8.scales[sb * 8 + r];
                        }
                    }
                    for (int sb = 0; sb < QK_K / 16; ++sb) {
                        for (int k = 0; k < 16; k += 2) {
                            const int from = (sb * 2 + r / 4) * 32 + (r % 4) * 8 + k / 2;
                            const int to = (sb * 4 + k / 4) * 32 + row * 2 + (k % 4) / 2;
                            if (iq_r16_paired_nibbles()) {
                                const int idx = (sb * 2 + k / 8) * 64 + row * 4 + k % 4;
                                const int shift = ((k / 4) % 2) * 4;
                                out.qs[idx]     |= (r8.qs[from] & 15) << shift;
                                out.qs[idx + 1] |= (r8.qs[from] >> 4) << shift;
                            } else {
                                out.qs[to] = r8.qs[from];
                            }
                        }
                    }
                }
            }
        }
    });
    return 0;
}

template <bool IQ3, bool NIBBLE2>
static void ggml_gemv_iq_r16_q8_K_impl(
        int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    GGML_ASSERT(n % QK_K == 0 && nc % 16 == 0);
    const int nb = n / QK_K;
    const auto * weights = (const block_iq_r16 *) vx;
    const auto * activations = (const block_q8_K *) vy;
    constexpr float factor = IQ3 ? 0.25f : 0.125f;
    static const uint8_t values2[16] = {21, 39, 56, 72, 89, 107, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64};
    static const uint8_t values3[16] = {2, 12, 20, 28, 36, 44, 52, 60, 68, 76, 84, 92, 100, 108, 116, 126};
    const uint8_t * values = IQ3 ? values3 : values2;
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
    const __m512i lut = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *) values));
    const __m512i lo_mask = _mm512_set1_epi16(15);
    const __m512i hi_mask = _mm512_set1_epi16(0x0f00);
    const __m512i scale_mask = _mm512_set1_epi32(15);
    const __m512i one = _mm512_set1_epi32(1);
    for (int y = 0; y < nr; ++y) {
        const block_q8_K * a = activations + y * nb;
        for (int x = 0; x < nc / 16; ++x) {
            const block_iq_r16 * w = weights + x * nb;
            __m512 sum = _mm512_setzero_ps();
            for (int b = 0; b < nb; ++b) {
                __m512i isum = _mm512_setzero_si512();
                for (int sb = 0; sb < QK_K / 16; ++sb) {
                    __m512i dots = _mm512_setzero_si512();
                    if constexpr (NIBBLE2) {
                        const __m512i mask = _mm512_set1_epi8(15);
                        const __m512i p0 = _mm512_loadu_si512(w[b].qs + (sb * 2 + 0) * 64);
                        const __m512i p1 = _mm512_loadu_si512(w[b].qs + (sb * 2 + 1) * 64);
                        int32_t aq[4];
                        memcpy(aq, a[b].qs + sb * 16, sizeof(aq));
                        const __m512i d0 = _mm512_dpbusd_epi32(_mm512_setzero_si512(),
                            _mm512_shuffle_epi8(lut, _mm512_and_si512(p0, mask)), _mm512_set1_epi32(aq[0]));
                        const __m512i d1 = _mm512_dpbusd_epi32(_mm512_setzero_si512(),
                            _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16(p0, 4), mask)), _mm512_set1_epi32(aq[1]));
                        const __m512i d2 = _mm512_dpbusd_epi32(_mm512_setzero_si512(),
                            _mm512_shuffle_epi8(lut, _mm512_and_si512(p1, mask)), _mm512_set1_epi32(aq[2]));
                        const __m512i d3 = _mm512_dpbusd_epi32(_mm512_setzero_si512(),
                            _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16(p1, 4), mask)), _mm512_set1_epi32(aq[3]));
                        dots = _mm512_add_epi32(_mm512_add_epi32(d0, d1), _mm512_add_epi32(d2, d3));
                    } else {
                        for (int chunk = 0; chunk < 4; ++chunk) {
                            const __m512i packed = _mm512_cvtepu8_epi16(_mm256_loadu_si256(
                                (const __m256i *) (w[b].qs + (sb * 4 + chunk) * 32)));
                            const __m512i codes = _mm512_or_si512(
                                _mm512_and_si512(packed, lo_mask),
                                _mm512_and_si512(_mm512_slli_epi16(packed, 4), hi_mask));
                            int32_t aq;
                            memcpy(&aq, a[b].qs + sb * 16 + chunk * 4, sizeof(aq));
                            dots = _mm512_dpbusd_epi32(dots, _mm512_shuffle_epi8(lut, codes), _mm512_set1_epi32(aq));
                        }
                    }
                    dots = _mm512_sub_epi32(dots, _mm512_set1_epi32(64 * a[b].bsums[sb]));
                    __m512i scales = _mm512_cvtepu8_epi32(_mm_loadu_si128(
                        (const __m128i *) (w[b].scales + (sb / 2) * 16)));
                    if (sb & 1) {
                        scales = _mm512_srli_epi32(scales, 4);
                    }
                    scales = _mm512_add_epi32(_mm512_slli_epi32(_mm512_and_si512(scales, scale_mask), 1), one);
                    isum = _mm512_add_epi32(isum, _mm512_mullo_epi32(dots, scales));
                }
                const __m512 d = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) w[b].d));
                const __m512 scale = _mm512_mul_ps(d, _mm512_set1_ps(a[b].d * factor));
                sum = _mm512_fmadd_ps(_mm512_cvtepi32_ps(isum), scale, sum);
            }
            _mm512_storeu_ps(s + y * bs + x * 16, sum);
        }
    }
#else
    for (int y = 0; y < nr; ++y) {
        const block_q8_K * a = activations + y * nb;
        for (int x = 0; x < nc / 16; ++x) {
            const block_iq_r16 * w = weights + x * nb;
            for (int row = 0; row < 16; ++row) {
                float sum = 0.0f;
                for (int b = 0; b < nb; ++b) {
                    int32_t isum = 0;
                    for (int sb = 0; sb < QK_K / 16; ++sb) {
                        int32_t dots = 0;
                        for (int k = 0; k < 16; ++k) {
                            const int idx = NIBBLE2 ? (sb * 2 + k / 8) * 64 + row * 4 + k % 4
                                                    : (sb * 4 + k / 4) * 32 + row * 2 + (k % 4) / 2;
                            const int shift = NIBBLE2 ? 4 * ((k / 4) % 2) : 4 * (k % 2);
                            const uint8_t code = (w[b].qs[idx] >> shift) & 15;
                            dots += (int(values[code]) - 64) * int(a[b].qs[sb * 16 + k]);
                        }
                        const int scale = (w[b].scales[(sb / 2) * 16 + row] >> (4 * (sb % 2))) & 15;
                        isum += dots * (2 * scale + 1);
                    }
                    sum = std::fma(float(isum), GGML_FP16_TO_FP32(w[b].d[row]) * (a[b].d * factor), sum);
                }
                s[y * bs + x * 16 + row] = sum;
            }
        }
    }
#endif
}

template <bool IQ3>
static void ggml_gemv_iq_r16_q8_K(
        int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    if (iq_r16_paired_nibbles()) {
        ggml_gemv_iq_r16_q8_K_impl<IQ3, true>(n, s, bs, vx, vy, nr, nc);
    } else {
        ggml_gemv_iq_r16_q8_K_impl<IQ3, false>(n, s, bs, vx, vy, nr, nc);
    }
}


static bool iq_r16_batch3_enabled() {
    static const bool enabled = [] {
        const char * value = getenv("GGML_CPU_IQ_R16_BATCH3");
        return value && atoi(value) != 0;
    }();
    return enabled && iq_r16_paired_nibbles();
}

template <bool IQ3, int NR>
static void ggml_gemv_iq_r16_batch_impl(
        int n, float * s, size_t bs, const void * vx, const block_q8_K * const * a, int nc) {
    GGML_ASSERT(n % QK_K == 0 && nc % 16 == 0 && NR >= 2 && NR <= 3);
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
    const int nb = n / QK_K;
    const auto * weights = (const block_iq_r16 *) vx;
    constexpr float factor = IQ3 ? 0.25f : 0.125f;
    static const uint8_t values2[16] = {21, 39, 56, 72, 89, 107, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64};
    static const uint8_t values3[16] = {2, 12, 20, 28, 36, 44, 52, 60, 68, 76, 84, 92, 100, 108, 116, 126};
    const __m512i lut = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *) (IQ3 ? values3 : values2)));
    const __m512i mask = _mm512_set1_epi8(15);
    for (int x = 0; x < nc / 16; ++x) {
        const block_iq_r16 * w = weights + x * nb;
        __m512 sum[NR];
        for (int y = 0; y < NR; ++y) sum[y] = _mm512_setzero_ps();
        for (int b = 0; b < nb; ++b) {
            __m512i isum[NR];
            for (int y = 0; y < NR; ++y) isum[y] = _mm512_setzero_si512();
            for (int sb = 0; sb < QK_K / 16; ++sb) {
                const __m512i p0 = _mm512_loadu_si512(w[b].qs + (sb * 2 + 0) * 64);
                const __m512i p1 = _mm512_loadu_si512(w[b].qs + (sb * 2 + 1) * 64);
                const __m512i q0 = _mm512_shuffle_epi8(lut, _mm512_and_si512(p0, mask));
                const __m512i q1 = _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16(p0, 4), mask));
                const __m512i q2 = _mm512_shuffle_epi8(lut, _mm512_and_si512(p1, mask));
                const __m512i q3 = _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16(p1, 4), mask));
                __m512i scales = _mm512_cvtepu8_epi32(_mm_loadu_si128(
                    (const __m128i *) (w[b].scales + (sb / 2) * 16)));
                if (sb & 1) scales = _mm512_srli_epi32(scales, 4);
                scales = _mm512_add_epi32(_mm512_slli_epi32(_mm512_and_si512(scales, _mm512_set1_epi32(15)), 1),
                    _mm512_set1_epi32(1));
                for (int y = 0; y < NR; ++y) {
                    int32_t aq[4];
                    memcpy(aq, a[y][b].qs + sb * 16, sizeof(aq));
                    const __m512i d0 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), q0, _mm512_set1_epi32(aq[0]));
                    const __m512i d1 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), q1, _mm512_set1_epi32(aq[1]));
                    const __m512i d2 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), q2, _mm512_set1_epi32(aq[2]));
                    const __m512i d3 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), q3, _mm512_set1_epi32(aq[3]));
                    __m512i dots = _mm512_add_epi32(_mm512_add_epi32(d0, d1), _mm512_add_epi32(d2, d3));
                    dots = _mm512_sub_epi32(dots, _mm512_set1_epi32(64 * a[y][b].bsums[sb]));
                    isum[y] = _mm512_add_epi32(isum[y], _mm512_mullo_epi32(dots, scales));
                }
            }
            const __m512 d = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) w[b].d));
            for (int y = 0; y < NR; ++y) {
                const __m512 scale = _mm512_mul_ps(d, _mm512_set1_ps(a[y][b].d * factor));
                sum[y] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(isum[y]), scale, sum[y]);
            }
        }
        for (int y = 0; y < NR; ++y) _mm512_storeu_ps(s + y * bs + x * 16, sum[y]);
    }
#else
    for (int y = 0; y < NR; ++y) {
        ggml_gemv_iq_r16_q8_K<IQ3>(n, s + y * bs, bs, vx, a[y], 1, nc);
    }
#endif
}

template <bool IQ3>
static void ggml_gemv_iq_r16_batch(
        int n, float * s, size_t bs, const void * vx, const block_q8_K * const * a, int nr, int nc) {
    GGML_ASSERT(nr >= 1 && nr <= 3);
    if (nr == 3) ggml_gemv_iq_r16_batch_impl<IQ3, 3>(n, s, bs, vx, a, nc);
    else if (nr == 2) ggml_gemv_iq_r16_batch_impl<IQ3, 2>(n, s, bs, vx, a, nc);
    else ggml_gemv_iq_r16_q8_K<IQ3>(n, s, bs, vx, a[0], 1, nc);
}

static void qK_r8_get_scale_min(
        int j, const uint8_t * GGML_RESTRICT packed,
        uint8_t * GGML_RESTRICT scale, uint8_t * GGML_RESTRICT min) {
    if (j < 4) {
        *scale = packed[j] & 63;
        *min = packed[j + 4] & 63;
    } else {
        *scale = (packed[j + 4] & 0x0f) | ((packed[j - 4] >> 6) << 4);
        *min = (packed[j + 4] >> 4) | ((packed[j] >> 6) << 4);
    }
}

static block_q4_K_r8 make_block_q4_K_r8(const block_q4_K * in) {
    block_q4_K_r8 out = {};
    constexpr int nrows_interleaved = 8;
    constexpr int values_per_subblock = 16;

    for (int row = 0; row < nrows_interleaved; ++row) {
        out.d[row] = in[row].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
        out.dmin[row] = in[row].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            uint8_t scale;
            uint8_t min;
            qK_r8_get_scale_min(ib32, in[row].scales, &scale, &min);
            out.scales[ib32 * nrows_interleaved + row] = scale;
            out.mins[ib32 * nrows_interleaved + row] = min;
        }
    }

    for (int sb = 0; sb < QK_K / values_per_subblock; ++sb) {
        const int ib32 = sb / 2;
        const int offset32 = (sb & 1) * values_per_subblock;
        const int qs_base = (ib32 / 2) * 32 + offset32;
        for (int row = 0; row < nrows_interleaved; ++row) {
            const int row_group = row / 4;
            uint8_t * packed_low = out.ql + (sb * 2 + row_group) * 32;
            for (int k = 0; k < values_per_subblock; ++k) {
                const uint8_t packed = in[row].qs[qs_base + k];
                const uint8_t low = (ib32 & 1) ? packed >> 4 : packed & 0x0f;
                const int code_index = (row % 4) * values_per_subblock + k;
                if (code_index & 1) {
                    packed_low[code_index / 2] |= low << 4;
                } else {
                    packed_low[code_index / 2] = low;
                }
            }
        }
    }

    return out;
}

static int repack_q4_K_to_q4_K_r8(
        struct ggml_tensor * t, const void * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_K);
    constexpr int nrows_interleaved = 8;

    const int64_t nrows = ggml_nrows(t);
    const int64_t nblocks = t->ne[0] / QK_K;
    GGML_ASSERT(data_size == (size_t) nrows * nblocks * sizeof(block_q4_K));

    if (t->ne[0] % QK_K != 0 || t->ne[1] % nrows_interleaved != 0) {
        return -1;
    }

    if (const char * trace = std::getenv("GGML_CPU_Q4_K_REPACK_TRACE");
            trace != nullptr && strcmp(trace, "1") == 0) {
        fprintf(stderr, "q4_K_r8: repack ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]\n",
                t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
    }

    const block_q4_K * src = (const block_q4_K *) data;
    block_q4_K_r8 * dst = (block_q4_K_r8 *) t->data;
    const int64_t ngroups = nrows / nrows_interleaved;
    parallel_repack_groups(ngroups, [&](int64_t group) {
        const block_q4_K * src_group =
            src + group * nrows_interleaved * nblocks;
        block_q4_K_r8 * dst_group = dst + group * nblocks;
        block_q4_K rows[nrows_interleaved];
        for (int64_t block = 0; block < nblocks; ++block) {
            for (int row = 0; row < nrows_interleaved; ++row) {
                rows[row] = src_group[block + row * nblocks];
            }
            dst_group[block] = make_block_q4_K_r8(rows);
        }
    });

    return 0;
}

static block_q5_K_r8 make_block_q5_K_r8(const block_q5_K * in) {
    block_q5_K_r8 out = {};
    constexpr int nrows_interleaved = 8;
    constexpr int values_per_subblock = 16;

    for (int row = 0; row < nrows_interleaved; ++row) {
        out.d[row] = in[row].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
        out.dmin[row] = in[row].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            uint8_t scale;
            uint8_t min;
            qK_r8_get_scale_min(ib32, in[row].scales, &scale, &min);
            out.scales[ib32 * nrows_interleaved + row] = scale;
            out.mins[ib32 * nrows_interleaved + row] = min;
        }
    }

    for (int sb = 0; sb < QK_K / values_per_subblock; ++sb) {
        const int ib32 = sb / 2;
        const int offset32 = (sb & 1) * values_per_subblock;
        const int qs_base = (ib32 / 2) * 32 + offset32;
        for (int row = 0; row < nrows_interleaved; ++row) {
            const int row_group = row / 4;
            uint8_t * packed_low = out.ql + (sb * 2 + row_group) * 32;
            uint64_t * high_plane = out.qh + sb * 2 + row_group;
            for (int k = 0; k < values_per_subblock; ++k) {
                const uint8_t packed = in[row].qs[qs_base + k];
                const uint8_t low = (ib32 & 1) ? packed >> 4 : packed & 0x0f;
                const uint8_t high = (in[row].qh[offset32 + k] >> ib32) & 1;
                const int code_index = (row % 4) * values_per_subblock + k;
                if (code_index & 1) {
                    packed_low[code_index / 2] |= low << 4;
                } else {
                    packed_low[code_index / 2] = low;
                }
                *high_plane |= (uint64_t) high << code_index;
            }
        }
    }

    return out;
}

static int repack_q5_K_to_q5_K_r8(
        struct ggml_tensor * t, const void * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q5_K);
    constexpr int nrows_interleaved = 8;

    const int64_t nrows = ggml_nrows(t);
    const int64_t nblocks = t->ne[0] / QK_K;
    GGML_ASSERT(data_size == (size_t) nrows * nblocks * sizeof(block_q5_K));

    if (t->ne[0] % QK_K != 0 || t->ne[1] % nrows_interleaved != 0) {
        return -1;
    }

    if (const char * trace = std::getenv("GGML_CPU_Q5_K_REPACK_TRACE");
            trace != nullptr && strcmp(trace, "1") == 0) {
        fprintf(stderr, "q5_K_r8: repack ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]\n",
                t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
    }

    const block_q5_K * src = (const block_q5_K *) data;
    block_q5_K_r8 * dst = (block_q5_K_r8 *) t->data;
    const int64_t ngroups = nrows / nrows_interleaved;
    parallel_repack_groups(ngroups, [&](int64_t group) {
        const block_q5_K * src_group =
            src + group * nrows_interleaved * nblocks;
        block_q5_K_r8 * dst_group = dst + group * nblocks;
        block_q5_K rows[nrows_interleaved];
        for (int64_t block = 0; block < nblocks; ++block) {
            for (int row = 0; row < nrows_interleaved; ++row) {
                rows[row] = src_group[block + row * nblocks];
            }
            dst_group[block] = make_block_q5_K_r8(rows);
        }
    });

    return 0;
}

static int repack_q4_0_to_q4_0_16_bl(struct ggml_tensor * t, int interleave_block, const void * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_0);
    constexpr int nrows_interleaved = 16;

    block_q4_0x16 * dst = (block_q4_0x16*)t->data;
    const block_q4_0 * src = (const block_q4_0*) data;
    block_q4_0 dst_tmp[16];
    int nrow = ggml_nrows(t);
    int nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_q4_0));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i  = 0; i < nrows_interleaved; i++ ) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q4_0x16(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    GGML_UNUSED(data_size);
}

static int repack_q5_K_to_q5_K_8_bl(struct ggml_tensor *       t,
                                    int                        interleave_block,
                                    const void * GGML_RESTRICT data,
                                    size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q5_K);
    GGML_ASSERT(interleave_block == 4 || interleave_block == 8);
    constexpr int nrows_interleaved = 8;

    block_q5_Kx8 *     dst = (block_q5_Kx8 *) t->data;
    const block_q5_K * src = (const block_q5_K *) data;
    block_q5_K         dst_tmp[8];
    int                nrow    = ggml_nrows(t);
    int                nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_q5_K));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q5_Kx8(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;
}

static int repack_q6_K_to_q6_K_8_bl(struct ggml_tensor * t, int interleave_block, const void * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q6_K);
    GGML_ASSERT(interleave_block == 4 || interleave_block == 8);
    constexpr int nrows_interleaved = 8;

    block_q6_Kx8 * dst = (block_q6_Kx8 *)t->data;
    const block_q6_K * src = (const block_q6_K *) data;
    block_q6_K dst_tmp[8];
    int nrow = ggml_nrows(t);
    int nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_q6_K));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q6_Kx8(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;
}

static int repack_q4_0_to_q4_0_8_bl(struct ggml_tensor * t, int interleave_block, const void * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(interleave_block == 8);
    constexpr int nrows_interleaved = 8;

    block_q4_0x8 * dst = (block_q4_0x8*)t->data;
    const block_q4_0 * src = (const block_q4_0*) data;
    block_q4_0 dst_tmp[8];
    int nrow = ggml_nrows(t);
    int nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_q4_0));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i  = 0; i < nrows_interleaved; i++ ) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q4_0x8(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    GGML_UNUSED(data_size);
}

static int repack_q8_0_to_q8_0_4_bl(struct ggml_tensor *       t,
                                    int                        interleave_block,
                                    const void * GGML_RESTRICT data,
                                    size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(interleave_block == 4 || interleave_block == 8);
    constexpr int nrows_interleaved = 4;

    block_q8_0x4 *     dst = (block_q8_0x4 *) t->data;
    const block_q8_0 * src = (const block_q8_0 *) data;
    block_q8_0         dst_tmp[4];
    int                nrow    = ggml_nrows(t);
    int                nblocks = t->ne[0] / QK8_0;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_q8_0));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q8_0x4(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;
}

static block_q8_0x16 make_block_q8_0x16(block_q8_0 * in, unsigned int blck_size_interleave) {
    block_q8_0x16 out;

    for (int i = 0; i < 16; i++) {
        out.d[i] = in[i].d;
    }

    const int end = QK8_0 * 16 / blck_size_interleave;

    if (blck_size_interleave == 1) {
        for (int i = 0; i < end; ++i) {
            int src_id     = i % 16;
            int src_offset = i / 16;
            int dst_offset = i;
            out.qs[dst_offset] = in[src_id].qs[src_offset];
        }
    } else {
        GGML_ASSERT(false);
    }

    return out;
}

static int repack_q8_0_to_q8_0_16_bl(struct ggml_tensor *       t,
                                    int                        interleave_block,
                                    const void * GGML_RESTRICT data,
                                    size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q8_0);
    constexpr int nrows_interleaved = 16;

    block_q8_0x16 *     dst = (block_q8_0x16 *) t->data;
    const block_q8_0 * src = (const block_q8_0 *) data;
    block_q8_0         dst_tmp[16];
    int                nrow    = ggml_nrows(t);
    int                nblocks = t->ne[0] / QK8_0;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_q8_0));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q8_0x16(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;
}

static block_iq4_nlx4 make_block_iq4_nlx4(block_iq4_nl * in, unsigned int blck_size_interleave) {
    block_iq4_nlx4 out;

    for (int i = 0; i < 4; i++) {
        out.d[i] = in[i].d;
    }

    const int end = QK4_NL * 2 / blck_size_interleave;

    // TODO: this branch seems wrong
    //if (blck_size_interleave == 8) {
    //    for (int i = 0; i < end; ++i) {
    //        int src_id = i % 4;
    //        int src_offset = (i / 4) * blck_size_interleave;
    //        int dst_offset = i * blck_size_interleave;

    //        // Using memcpy to avoid unaligned memory accesses
    //        memcpy(&out.qs[dst_offset], &in[src_id].qs[src_offset], sizeof(uint64_t));
    //    }
    //} else
    if (blck_size_interleave == 4) {
        for (int i = 0; i < end; ++i) {
            int src_id = i % 4;
            int src_offset = (i / 4) * blck_size_interleave;
            int dst_offset = i * blck_size_interleave;

            memcpy(&out.qs[dst_offset], &in[src_id].qs[src_offset], sizeof(uint32_t));
        }
    } else {
        GGML_ASSERT(false);
    }

    return out;
}

static int repack_iq4_nl_to_iq4_nl_4_bl(struct ggml_tensor * t, int interleave_block, const void * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_IQ4_NL);
    GGML_ASSERT(interleave_block == 4);

    const block_iq4_nl   * src = (const block_iq4_nl   *)data;
          block_iq4_nlx4 * dst = (      block_iq4_nlx4 *)t->data;

    block_iq4_nl dst_tmp[4];

    int nrow = ggml_nrows(t);
    int nrows_interleaved = 4;
    int nblocks = t->ne[0] / QK4_NL;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_iq4_nl));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_iq4_nlx4(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    GGML_UNUSED(data_size);
}

static block_iq4_nlx8 make_block_iq4_nlx8(block_iq4_nl * in, unsigned int blck_size_interleave) {
    block_iq4_nlx8 out;

    for (int i = 0; i < 8; i++) {
        out.d[i] = in[i].d;
    }

    const int end = QK4_NL * 4 / blck_size_interleave;

    if (blck_size_interleave == 8) {
        for (int i = 0; i < end; ++i) {
            int src_id = i % 8;
            int src_offset = (i / 8) * blck_size_interleave;
            int dst_offset = i * blck_size_interleave;

            memcpy(&out.qs[dst_offset], &in[src_id].qs[src_offset], sizeof(uint64_t));
        }
    } else {
        GGML_ASSERT(false);
    }

    return out;
}

static int repack_iq4_nl_to_iq4_nl_8_bl(struct ggml_tensor * t, int interleave_block, const void * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_IQ4_NL);
    GGML_ASSERT(interleave_block == 8);

    const block_iq4_nl   * src = (const block_iq4_nl   *)data;
          block_iq4_nlx8 * dst = (      block_iq4_nlx8 *)t->data;

    block_iq4_nl dst_tmp[8];

    int nrow = ggml_nrows(t);
    int nrows_interleaved = 8;
    int nblocks = t->ne[0] / QK4_NL;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_iq4_nl));

    if (t->ne[1] % nrows_interleaved != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_iq4_nlx8(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    GGML_UNUSED(data_size);
}

static block_iq4_nlx16 make_block_iq4_nlx16(block_iq4_nl * in, unsigned int blck_size_interleave) {
    block_iq4_nlx16 out;

    for (int i = 0; i < 16; i++) {
        out.d[i] = in[i].d;
    }

    const int end = QK4_NL * 8 / blck_size_interleave;

    if (blck_size_interleave == 1) {
        for (int i = 0; i < end; ++i) {
            int src_id = i % 16;
            int src_offset = i / 16;
            int dst_offset = i;

            out.qs[dst_offset] = in[src_id].qs[src_offset];
        }
    } else {
        GGML_ASSERT(false);
    }

    return out;
}

static int repack_iq4_nl_to_iq4_nl_16_bl(struct ggml_tensor * t, int interleave_block, const void * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_IQ4_NL);
    GGML_ASSERT(interleave_block == 1);

    const block_iq4_nl    * src = (const block_iq4_nl   *)data;
          block_iq4_nlx16 * dst = (      block_iq4_nlx16 *)t->data;

    block_iq4_nl dst_tmp[16];

    int nrow = ggml_nrows(t);
    int nrows_interleaved = 16;
    int nblocks = t->ne[0] / QK4_NL;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_iq4_nl));

    if (t->ne[1] % nrows_interleaved != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_iq4_nlx16(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    GGML_UNUSED(data_size);
}

static block_mxfp4x4 make_block_mxfp4x4(block_mxfp4 * in, unsigned int blck_size_interleave) {
    block_mxfp4x4 out;

    for (int i = 0; i < 4; i++) {
        out.e[i] = in[i].e;
    }

    const int end = QK_MXFP4 * 2 / blck_size_interleave;

    if (blck_size_interleave == 4) {
        for (int i = 0; i < end; ++i) {
            int src_id = i % 4;
            int src_offset = (i / 4) * blck_size_interleave;
            int dst_offset = i * blck_size_interleave;

            memcpy(&out.qs[dst_offset], &in[src_id].qs[src_offset], sizeof(uint32_t));
        }
    } else {
        GGML_ASSERT(false);
    }

    return out;
}

static int repack_mxfp4_to_mxfp4_4_bl(struct ggml_tensor * t, int interleave_block, const void * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_MXFP4);
    GGML_ASSERT(interleave_block == 4);

    const block_mxfp4   * src = (const block_mxfp4   *)data;
          block_mxfp4x4 * dst = (      block_mxfp4x4 *)t->data;

    block_mxfp4 dst_tmp[4];

    int nrow = ggml_nrows(t);
    int nrows_interleaved = 4;
    int nblocks = t->ne[0] / QK_MXFP4;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_mxfp4));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_mxfp4x4(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    GGML_UNUSED(data_size);
}

static block_mxfp4x8 make_block_mxfp4x8(block_mxfp4 * in, unsigned int blck_size_interleave) {
    block_mxfp4x8 out;

    for (int i = 0; i < 8; i++) {
        out.e[i] = in[i].e;
    }

    const int end = QK_MXFP4 * 4 / blck_size_interleave;

    if (blck_size_interleave == 8) {
        for (int i = 0; i < end; ++i) {
            int src_id = i % 8;
            int src_offset = (i / 8) * blck_size_interleave;
            int dst_offset = i * blck_size_interleave;

            memcpy(&out.qs[dst_offset], &in[src_id].qs[src_offset], sizeof(uint64_t));
        }
    } else {
        GGML_ASSERT(false);
    }

    return out;
}

static int repack_mxfp4_to_mxfp4_8_bl(struct ggml_tensor * t, int interleave_block, const void * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_MXFP4);
    GGML_ASSERT(interleave_block == 8);

    const block_mxfp4   * src = (const block_mxfp4   *)data;
          block_mxfp4x8 * dst = (      block_mxfp4x8 *)t->data;

    block_mxfp4 dst_tmp[8];

    int nrow = ggml_nrows(t);
    int nrows_interleaved = 8;
    int nblocks = t->ne[0] / QK_MXFP4;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_mxfp4));

    if (t->ne[1] % nrows_interleaved != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_mxfp4x8(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    GGML_UNUSED(data_size);
}

namespace ggml::cpu::repack {
// repack
template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS>
int repack(struct ggml_tensor *, const void *, size_t);

// TODO: generalise.
template <> int repack<block_q4_0, 4, 4>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q4_0_to_q4_0_4_bl(t, 4, data, data_size);
}

template <> int repack<block_q4_0, 8, 4>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q4_0_to_q4_0_4_bl(t, 8, data, data_size);
}

template <> int repack<block_q4_0, 8, 8>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q4_0_to_q4_0_8_bl(t, 8, data, data_size);
}

template <> int repack<block_q4_K, 8, 8>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q4_K_to_q4_K_8_bl(t, 8, data, data_size);
}

template <> int repack<block_q4_K_r8_source, 8, 8>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q4_K_to_q4_K_r8(t, data, data_size);
}

template <> int repack<block_q4_K, 4, 8>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q4_K_to_q4_K_8_bl(t, 4, data, data_size);
}

template <> int repack<block_q2_K, 8, 8>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q2_K_to_q2_K_8_bl(t, 8, data, data_size);
}

template <> int repack<block_iq2_xs, 8, 16>(ggml_tensor * t, const void * data, size_t data_size) {
    return repack_iq_to_r16<block_iq2_xs>(t, data, data_size);
}

template <> int repack<block_iq2_xs, 8, 8>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_iq2_xs_to_iq2_xs_r8(t, data, data_size);
}

template <> int repack<block_iq3_xxs, 8, 16>(ggml_tensor * t, const void * data, size_t data_size) {
    return repack_iq_to_r16<block_iq3_xxs>(t, data, data_size);
}

template <> int repack<block_iq3_xxs, 8, 8>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_iq3_xxs_to_iq3_xxs_r8(t, data, data_size);
}

template <> int repack<block_q5_K_r8_source, 8, 8>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q5_K_to_q5_K_r8(t, data, data_size);
}

template <> int repack<block_q5_K, 4, 8>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q5_K_to_q5_K_8_bl(t, 4, data, data_size);
}

template <> int repack<block_q5_K, 8, 8>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q5_K_to_q5_K_8_bl(t, 8, data, data_size);
}

template <> int repack<block_q6_K, 4, 8>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q6_K_to_q6_K_8_bl(t, 4, data, data_size);
}

template <> int repack<block_q6_K, 8, 8>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q6_K_to_q6_K_8_bl(t, 8, data, data_size);
}

template <> int repack<block_iq4_nl, 4, 4>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_iq4_nl_to_iq4_nl_4_bl(t, 4, data, data_size);
}

// TODO: needs to be revisited
//template <> int repack<block_iq4_nl, 8, 4>(struct ggml_tensor * t, const void * data, size_t data_size) {
//    return repack_iq4_nl_to_iq4_nl_4_bl(t, 8, data, data_size);
//}

template <> int repack<block_iq4_nl, 8, 8>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_iq4_nl_to_iq4_nl_8_bl(t, 8, data, data_size);
}

template <> int repack<block_mxfp4, 4, 4>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_mxfp4_to_mxfp4_4_bl(t, 4, data, data_size);
}

template <> int repack<block_mxfp4, 8, 8>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_mxfp4_to_mxfp4_8_bl(t, 8, data, data_size);
}

template <> int repack<block_q8_0, 4, 4>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q8_0_to_q8_0_4_bl(t, 4, data, data_size);
}

template <> int repack<block_q8_0, 8, 4>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q8_0_to_q8_0_4_bl(t, 8, data, data_size);
}

template <> int repack<block_q8_0, 8, 8>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q8_0_to_q8_0_8_vnni(t, data, data_size);
}

#if defined __riscv_zvfh
template <> int repack<block_q4_0, 1, 16>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q4_0_to_q4_0_16_bl(t, 1, data, data_size);
}

template <> int repack<block_q4_K, 1, 16>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q4_K_to_q4_K_16_bl(t, 1, data, data_size);
}

template <> int repack<block_iq4_nl, 1, 16>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_iq4_nl_to_iq4_nl_16_bl(t, 1, data, data_size);
}

template <> int repack<block_q8_0, 1, 16>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q8_0_to_q8_0_16_bl(t, 1, data, data_size);
}

template <> int repack<block_q2_K, 1, 16>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q2_K_to_q2_K_16_bl(t, 1, data, data_size);
}
#endif

// gemv
template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS, ggml_type PARAM_TYPE>
void gemv(int, float *, size_t, const void *, const void *, int, int);

template <> void gemv<block_q4_0, 4, 4, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_q4_0_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_q4_0, 8, 4, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_q4_0_4x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_q4_0, 8, 8, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_q4_0_8x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <>
void gemv<block_q2_K, 8, 8, GGML_TYPE_Q8_K>(int          n,
                                            float *      s,
                                            size_t       bs,
                                            const void * vx,
                                            const void * vy,
                                            int          nr,
                                            int          nc) {
    ggml_gemv_q2_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_iq2_xs, 8, 16, GGML_TYPE_Q8_K>(
        int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_iq_r16_q8_K<false>(n, s, bs, vx, vy, nr, nc);
}

template <>
void gemv<block_iq2_xs, 8, 8, GGML_TYPE_Q8_K>(
        int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_iq2_xs_r8_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_iq3_xxs, 8, 16, GGML_TYPE_Q8_K>(
        int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_iq_r16_q8_K<true>(n, s, bs, vx, vy, nr, nc);
}

template <>
void gemv<block_iq3_xxs, 8, 8, GGML_TYPE_Q8_K>(
        int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_iq3_xxs_r8_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <>
void gemv<block_q5_K_r8_source, 8, 8, GGML_TYPE_Q8_K>(
        int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_q5_K_r8_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <>
void gemv<block_q4_K_r8_source, 8, 8, GGML_TYPE_Q8_K>(
        int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_q4_K_r8_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_q4_K, 4, 8, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_q4_K_8x4_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_q4_K, 8, 8, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_q4_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_q5_K, 4, 8, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_q5_K_8x4_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_q5_K, 8, 8, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_q5_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_q6_K, 4, 8, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_q6_K_8x4_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_q6_K, 8, 8, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_q6_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_iq4_nl, 4, 4, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_iq4_nl_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_iq4_nl, 8, 8, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_iq4_nl_8x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_mxfp4, 4, 4, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_mxfp4_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_mxfp4, 8, 8, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_mxfp4_8x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_q8_0, 4, 4, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_q8_0_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_q8_0, 8, 4, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_q8_0_4x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_q8_0, 8, 8, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_q8_0_8x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

#if defined __riscv_zvfh
template <> void gemv<block_q4_0, 1, 16, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_q4_0_16x1_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_q4_K, 1, 16, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_q4_K_16x1_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_iq4_nl, 1, 16, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_iq4_nl_16x1_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_q8_0, 1, 16, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_q8_0_16x1_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_q2_K, 1, 16, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_q2_K_16x1_q8_K(n, s, bs, vx, vy, nr, nc);
}
#endif

// gemm
template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS, ggml_type PARAM_TYPE>
void gemm(int, float *, size_t, const void *, const void *, int, int);

template <> void gemm<block_q4_0, 4, 4, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_q4_0_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_q4_0, 8, 4, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_q4_0_4x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <>
void gemm<block_q4_0, 8, 8, GGML_TYPE_Q8_0>(int          n,
                                            float *      s,
                                            size_t       bs,
                                            const void * vx,
                                            const void * vy,
                                            int          nr,
                                            int          nc) {
    ggml_gemm_q4_0_8x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_q2_K, 8, 8, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_q2_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_iq2_xs, 8, 16, GGML_TYPE_Q8_K>(
        int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_iq_r16_q8_K<false>(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_iq2_xs, 8, 8, GGML_TYPE_Q8_K>(
        int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_iq2_xs_r8_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_iq3_xxs, 8, 16, GGML_TYPE_Q8_K>(
        int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_iq_r16_q8_K<true>(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_iq3_xxs, 8, 8, GGML_TYPE_Q8_K>(
        int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_iq3_xxs_r8_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_q5_K_r8_source, 8, 8, GGML_TYPE_Q8_K>(
        int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_q5_K_r8_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_q4_K_r8_source, 8, 8, GGML_TYPE_Q8_K>(
        int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_q4_K_r8_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_q4_K, 4, 8, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_q4_K_8x4_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_q4_K, 8, 8, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_q4_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_q5_K, 4, 8, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_q5_K_8x4_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_q5_K, 8, 8, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_q5_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_q6_K, 4, 8, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_q6_K_8x4_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_q6_K, 8, 8, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_q6_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_iq4_nl, 4, 4, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_iq4_nl_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_iq4_nl, 8, 8, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_iq4_nl_8x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_mxfp4, 4, 4, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_mxfp4_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_mxfp4, 8, 8, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_mxfp4_8x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_q8_0, 4, 4, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_q8_0_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_q8_0, 8, 4, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_q8_0_4x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_q8_0, 8, 8, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_q8_0_8x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

#if defined __riscv_zvfh
template <> void gemm<block_q4_0, 1, 16, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_q4_0_16x1_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_q4_K, 1, 16, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_q4_K_16x1_q8_K(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_iq4_nl, 1, 16, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_iq4_nl_16x1_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_q8_0, 1, 16, GGML_TYPE_Q8_0>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_q8_0_16x1_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_q2_K, 1, 16, GGML_TYPE_Q8_K>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_q2_K_16x1_q8_K(n, s, bs, vx, vy, nr, nc);
}
#endif

class tensor_traits_base : public ggml::cpu::tensor_traits {
  public:
    virtual int repack(struct ggml_tensor * t, const void * data, size_t data_size) = 0;
    virtual size_t repacked_alloc_size(const struct ggml_tensor * t) const {
        return ggml_nbytes(t);
    }
};

template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS, ggml_type PARAM_TYPE> class tensor_traits : public tensor_traits_base {

    static constexpr bool expanded_iq2_xs =
        std::is_same_v<BLOC_TYPE, block_iq2_xs> && INTER_SIZE == 8 && (NB_COLS == 8 || NB_COLS == 16);
    static constexpr bool expanded_iq3_xxs =
        std::is_same_v<BLOC_TYPE, block_iq3_xxs> && INTER_SIZE == 8 && (NB_COLS == 8 || NB_COLS == 16);
    static constexpr bool expanded_q5_K =
        std::is_same_v<BLOC_TYPE, block_q5_K_r8_source> && INTER_SIZE == 8 && NB_COLS == 8;
    static constexpr bool expanded_q4_K =
        std::is_same_v<BLOC_TYPE, block_q4_K_r8_source> && INTER_SIZE == 8 && NB_COLS == 8;
    static constexpr bool expanded_iq = expanded_iq2_xs || expanded_iq3_xxs;
    static constexpr bool expanded_layout = expanded_iq || expanded_q4_K || expanded_q5_K;

    static bool validate_moe_down_weighted_sum() {
        static const bool enabled = []() {
            const char * value = getenv("GGML_CPU_MOE_DOWN_WEIGHTED_SUM_VALIDATE");
            return value != nullptr && strcmp(value, "1") == 0;
        }();
        return enabled;
    }

    static int moe_down_weighted_sum_tile_size() {
        static const int value = []() {
            const char * text = getenv("GGML_CPU_MOE_DOWN_WEIGHTED_SUM_TILE");
            if (text == nullptr || text[0] == '\0') {
                return 64;
            }
            const int parsed = atoi(text);
            return parsed >= NB_COLS && parsed <= 512 && parsed % NB_COLS == 0 ? parsed : 64;
        }();
        return value;
    }

    static int moe_single_token_threads() {
        static const int value = []() {
            const char * text = getenv("GGML_CPU_MOE_SINGLE_TOKEN_THREADS");
            if (text == nullptr || text[0] == '\0') {
                return 0;
            }
            const int parsed = atoi(text);
            return parsed >= 1 && parsed <= 256 ? parsed : 0;
        }();
        return value;
    }

    static size_t physical_row_group_size(const struct ggml_tensor * t) {
        if constexpr (expanded_iq && NB_COLS == 16) {
            return (t->ne[0] / QK_K) * sizeof(block_iq_r16);
        } else if constexpr (expanded_iq2_xs) {
            GGML_ASSERT(t->ne[0] % QK_K == 0);
            return (t->ne[0] / QK_K) * sizeof(block_iq2_xs_r8);
        } else if constexpr (expanded_iq3_xxs) {
            GGML_ASSERT(t->ne[0] % QK_K == 0);
            return (t->ne[0] / QK_K) * sizeof(block_iq3_xxs_r8);
        } else if constexpr (expanded_q4_K) {
            GGML_ASSERT(t->ne[0] % QK_K == 0);
            return (t->ne[0] / QK_K) * sizeof(block_q4_K_r8);
        } else if constexpr (expanded_q5_K) {
            GGML_ASSERT(t->ne[0] % QK_K == 0);
            return (t->ne[0] / QK_K) * sizeof(block_q5_K_r8);
        }
        return NB_COLS * t->nb[1];
    }

    static size_t physical_row_offset(const struct ggml_tensor * t, int64_t row) {
        if constexpr (expanded_layout) {
            GGML_ASSERT(row % NB_COLS == 0);
            return (row / NB_COLS) * physical_row_group_size(t);
        }
        return row * t->nb[1];
    }

    static size_t physical_expert_offset(const struct ggml_tensor * t, int64_t expert) {
        if constexpr (expanded_layout) {
            GGML_ASSERT(t->ne[1] % NB_COLS == 0);
            return expert * (t->ne[1] / NB_COLS) * physical_row_group_size(t);
        }
        return expert * t->nb[2];
    }

    size_t repacked_alloc_size(const struct ggml_tensor * t) const override {
        if constexpr (expanded_layout) {
            GGML_ASSERT(ggml_nrows(t) % NB_COLS == 0);
            return (ggml_nrows(t) / NB_COLS) * physical_row_group_size(t);
        }
        return ggml_nbytes(t);
    }

    bool work_size(int n_threads, const struct ggml_tensor * op, size_t & size) override {
        // not realy a GGML_TYPE_Q8_0 but same size.
        switch (op->op) {
            case GGML_OP_MUL_MAT:
                {
                    size = ggml_row_size(PARAM_TYPE, ggml_nelements(op->src[1]));
                    return true;
                }
            case GGML_OP_MUL_MAT_ID:
                {
                    size = ggml_row_size(PARAM_TYPE, ggml_nelements(op->src[1]));
                    size = GGML_PAD(size, sizeof(int64_t)); // + padding for next block.

                    const int64_t ne02 = op->src[0]->ne[2]; // n_as, n_expert
                    const int64_t ne12 = op->src[1]->ne[2]; // n_tokens

                    const size_t sizeof_mmid_row_mapping = sizeof(int64_t);

                    size += sizeof_mmid_row_mapping*ne02*(ne12 + 1);

                    if constexpr (expanded_iq3_xxs) {
                        if (validate_moe_down_weighted_sum()) {
                            size = GGML_PAD(size, alignof(float));
                            size += sizeof(float) * (op->ne[0] * op->ne[2] + n_threads);
                        }
                    }

                    return true;
                }
            default:
                // GGML_ABORT("fatal error");
                break;
        }
        return false;
    }

    bool compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) override {
        switch (op->op) {
            case GGML_OP_MUL_MAT:
                forward_mul_mat(params, op);
                return true;
            case GGML_OP_MUL_MAT_ID:
                forward_mul_mat_id(params, op);
                return true;
            default:
                // GGML_ABORT("fatal error");
                break;
        }
        return false;
    }

    bool compute_forward_mul_mat_swiglu(
            struct ggml_compute_params * params,
            struct ggml_tensor * gate,
            struct ggml_tensor * up,
            struct ggml_tensor * dst) override {
        if (gate->op != GGML_OP_MUL_MAT || up->op != GGML_OP_MUL_MAT ||
                gate->src[0]->extra != this || up->src[0]->extra != this) {
            return false;
        }

        forward_mul_mat_swiglu(params, gate, up, dst);
        return true;
    }

    void forward_mul_mat_swiglu(
            ggml_compute_params * params,
            ggml_tensor * gate,
            ggml_tensor * up,
            ggml_tensor * dst) {
        const ggml_tensor * gate_w = gate->src[0];
        const ggml_tensor * up_w = up->src[0];
        const ggml_tensor * src1 = gate->src[1];

        const int ith = params->ith;
        const int nth = params->nth;
        const int64_t k = gate_w->ne[0];

        GGML_ASSERT(up->src[1] == src1);
        GGML_ASSERT(gate_w->type == up_w->type);
        GGML_ASSERT(ggml_are_same_shape(gate_w, up_w));
        GGML_ASSERT(ggml_are_same_shape(gate, up));
        GGML_ASSERT(ggml_are_same_shape(gate, dst));
        GGML_ASSERT(src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
        GGML_ASSERT(src1->ne[0] == k);
        GGML_ASSERT(gate_w->ne[2] == 1 && gate_w->ne[3] == 1);
        GGML_ASSERT(src1->ne[3] == 1 && dst->ne[3] == 1);
        GGML_ASSERT(gate_w->nb[0] == ggml_type_size(gate_w->type));
        GGML_ASSERT(up_w->nb[0] == ggml_type_size(up_w->type));
        GGML_ASSERT(src1->nb[0] == sizeof(float) && dst->nb[0] == sizeof(float));

        char * wdata = static_cast<char *>(params->wdata);
        const size_t src1_row_size = ggml_row_size(PARAM_TYPE, k);
        const size_t src1_plane_size = src1_row_size * src1->ne[1];
        GGML_ASSERT(params->wsize >= src1_plane_size * src1->ne[2]);

        const ggml_from_float_t from_float = ggml_get_type_traits_cpu(PARAM_TYPE)->from_float;
        for (int64_t plane = 0; plane < src1->ne[2]; ++plane) {
            char * src_plane = (char *) src1->data + plane * src1->nb[2];
            char * q_plane = wdata + plane * src1_plane_size;

            const bool src1_rows_contiguous = src1->nb[1] == (size_t) k * sizeof(float);
            std::vector<float> row_tmp;
            if (!src1_rows_contiguous) {
                row_tmp.resize((size_t) 4 * k);
            }
            for (int64_t row = ith * 4; row < src1->ne[1] - src1->ne[1] % 4; row += nth * 4) {
                const float * rows_ptr = (const float *) (src_plane + row * src1->nb[1]);
                if (!src1_rows_contiguous) {
                    // the 4-row quantizer expects contiguous rows; gather strided rows (e.g. k-slice views)
                    for (int r = 0; r < 4; r++) {
                        memcpy(row_tmp.data() + (size_t) r * k, src_plane + (row + r) * src1->nb[1], (size_t) k * sizeof(float));
                    }
                    rows_ptr = row_tmp.data();
                }
                if constexpr (expanded_iq) {
                    for (int r = 0; r < 4; ++r) {
                        from_float(rows_ptr + r * k, q_plane + (row + r) * src1_row_size, k);
                    }
                } else {
                    ggml_quantize_mat_t<INTER_SIZE, PARAM_TYPE>(rows_ptr, q_plane + row * src1_row_size, 4, k);
                }
            }

            const int64_t rows4 = src1->ne[1] - src1->ne[1] % 4;
            for (int64_t row = rows4 + ith; row < src1->ne[1]; row += nth) {
                from_float(
                    (const float *) (src_plane + row * src1->nb[1]),
                    q_plane + row * src1_row_size, k);
            }
        }

        if (ith == 0) {
            ggml_threadpool_chunk_set(params->threadpool, nth);
        }
        ggml_barrier(params->threadpool);

        const int64_t nr0 = ggml_nrows(gate_w);
        int64_t nchunk0;
        {
            const int nth_scaled = nth * 4;
            const int64_t chunk_size = (nr0 + nth_scaled - 1) / nth_scaled;
            nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
        }

        const int64_t min_chunk_size = NB_COLS;
        if (nchunk0 > 0 && nr0 / nchunk0 < min_chunk_size && nr0 >= min_chunk_size) {
            nchunk0 = (nr0 + min_chunk_size - 1) / min_chunk_size;
        }
        if (nth == 1 || ((nchunk0 < nth || ggml_is_numa()) &&
                (nr0 + nth - 1) / nth >= min_chunk_size)) {
            nchunk0 = nth;
        }
        const int64_t max_nchunk = (nr0 + min_chunk_size - 1) / min_chunk_size;
        nchunk0 = std::min(nchunk0, max_nchunk);
        const int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
        const int64_t nchunk1 = src1->ne[2];

        int current_chunk = ith;
        while (current_chunk < nchunk0 * nchunk1) {
            const int64_t chunk0 = current_chunk % nchunk0;
            const int64_t chunk1 = current_chunk / nchunk0;
            int64_t row_start = dr0 * chunk0;
            int64_t row_end = std::min(row_start + dr0, nr0);

            row_start = GGML_PAD(row_start, NB_COLS);
            row_end = GGML_PAD(row_end, NB_COLS);
            row_end = std::min(row_end, nr0);

            if (row_start < row_end) {
                const int64_t src_start = chunk1 * src1->ne[1];
                const int64_t src_end = (chunk1 + 1) * src1->ne[1];

                forward_mul_mat_one_chunk(params, gate, row_start, row_end, src_start, src_end);
                forward_mul_mat_one_chunk(params, up, row_start, row_end, src_start, src_end);

                for (int64_t flat = src_start; flat < src_end; ++flat) {
                    const int64_t plane = flat / src1->ne[1];
                    const int64_t src_row = flat - plane * src1->ne[1];
                    const float * gate_row = (const float *) ((const char *) gate->data +
                        plane * gate->nb[2] + src_row * gate->nb[1]);
                    const float * up_row = (const float *) ((const char *) up->data +
                        plane * up->nb[2] + src_row * up->nb[1]);
                    float * dst_row = (float *) ((char *) dst->data +
                        plane * dst->nb[2] + src_row * dst->nb[1]);
                    ggml_vec_swiglu_f32(
                        row_end - row_start, dst_row + row_start,
                        gate_row + row_start, up_row + row_start);
                }
            }

            current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
        }
    }

    bool compute_forward_mul_mat_id_swiglu(
            struct ggml_compute_params * params,
            struct ggml_tensor * gate,
            struct ggml_tensor * up,
            struct ggml_tensor * dst) override {
        if (gate->op != GGML_OP_MUL_MAT_ID || up->op != GGML_OP_MUL_MAT_ID ||
                gate->src[0]->extra != this || up->src[0]->extra != this) {
            return false;
        }

        forward_mul_mat_id_swiglu(params, gate, up, dst);
        return true;
    }

    void forward_mul_mat_id_swiglu(
            ggml_compute_params * params,
            ggml_tensor * gate,
            ggml_tensor * up,
            ggml_tensor * dst) {
        const ggml_tensor * gate_w = gate->src[0];
        const ggml_tensor * up_w = up->src[0];
        const ggml_tensor * src1 = gate->src[1];
        const ggml_tensor * ids = gate->src[2];

        const int ith = params->ith;
        const int nth = params->nth;
        const int64_t k = gate_w->ne[0];
        const int64_t n_out = gate_w->ne[1];
        const int64_t n_experts = gate_w->ne[2];
        const int64_t n_used = ids->ne[0];
        const int64_t n_tokens = ids->ne[1];
        const int64_t n_src_rows = src1->ne[1];

        GGML_ASSERT(up->src[1] == src1 && up->src[2] == ids);
        GGML_ASSERT(gate_w->type == up_w->type);
        GGML_ASSERT(ggml_are_same_shape(gate_w, up_w));
        GGML_ASSERT(ggml_are_same_shape(gate, up));
        GGML_ASSERT(ggml_are_same_shape(gate, dst));
        GGML_ASSERT(src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
        GGML_ASSERT(src1->ne[0] == k && n_src_rows > 0 && src1->ne[2] == n_tokens);
        GGML_ASSERT(gate_w->ne[3] == 1 && src1->ne[3] == 1 && dst->ne[3] == 1);
        GGML_ASSERT(gate_w->nb[0] == ggml_type_size(gate_w->type));
        GGML_ASSERT(up_w->nb[0] == ggml_type_size(up_w->type));
        GGML_ASSERT(src1->nb[0] == sizeof(float) && dst->nb[0] == sizeof(float));

        const size_t src1_row_size = ggml_row_size(PARAM_TYPE, k);
        const size_t src1_plane_size = src1_row_size * n_src_rows;
        const size_t src1_size = src1_plane_size * n_tokens;

        struct row_mapping {
            int32_t i1;
            int32_t i2;
        };

        char * wdata = (char *) params->wdata;
        char * mapping_data = wdata + GGML_PAD(src1_size, sizeof(int64_t));
        int64_t * matrix_row_counts = (int64_t *) mapping_data;
        row_mapping * matrix_rows = (row_mapping *) (matrix_row_counts + n_experts);

        GGML_ASSERT(params->wsize >=
            GGML_PAD(src1_size, sizeof(int64_t)) +
            n_experts * (n_tokens + 1) * sizeof(int64_t));

        const ggml_from_float_t from_float = ggml_get_type_traits_cpu(PARAM_TYPE)->from_float;
        const int64_t quant_block = ggml_blck_size(PARAM_TYPE);
        for (int64_t token = 0; token < n_tokens; ++token) {
            for (int64_t src_row = 0; src_row < n_src_rows; ++src_row) {
                const int64_t block_start = (ith * (k / quant_block)) / nth;
                const int64_t block_end = ((ith + 1) * (k / quant_block)) / nth;
                from_float(
                    (const float *) ((const char *) src1->data +
                        token * src1->nb[2] + src_row * src1->nb[1] +
                        block_start * quant_block * src1->nb[0]),
                    wdata + token * src1_plane_size + src_row * src1_row_size +
                        block_start * ggml_type_size(PARAM_TYPE),
                    (block_end - block_start) * quant_block);
            }
        }

        if (ith == 0) {
            memset(matrix_row_counts, 0, n_experts * sizeof(int64_t));
            for (int64_t token = 0; token < n_tokens; ++token) {
                for (int64_t used = 0; used < n_used; ++used) {
                    const int32_t expert = *(const int32_t *) ((const char *) ids->data +
                        token * ids->nb[1] + used * ids->nb[0]);
                    GGML_ASSERT(expert >= 0 && expert < n_experts);
                    matrix_rows[expert * n_tokens + matrix_row_counts[expert]] = {
                        (int32_t) used, (int32_t) token
                    };
                    matrix_row_counts[expert] += 1;
                }
            }
        }

        ggml_barrier(params->threadpool);

        int active_nth = nth;
        if (n_tokens == 1) {
            const int requested = moe_single_token_threads();
            if (requested > 0) {
                active_nth = std::min(active_nth, requested);
            }
        }
        if (ith >= active_nth) {
            return;
        }

        int64_t row_start = (ith * n_out) / active_nth;
        int64_t row_end = ((ith + 1) * n_out) / active_nth;
        row_start = GGML_PAD(row_start, NB_COLS);
        row_end = GGML_PAD(row_end, NB_COLS);
        row_end = std::min(row_end, n_out);
        if (row_start >= row_end) {
            return;
        }

        for (int64_t expert = 0; expert < n_experts; ++expert) {
            const int64_t n_rows = matrix_row_counts[expert];
            if (n_rows == 0) {
                continue;
            }

            const char * gate_cur = (const char *) gate_w->data + physical_expert_offset(gate_w, expert);
            const char * up_cur = (const char *) up_w->data + physical_expert_offset(up_w, expert);

            if constexpr (expanded_iq && NB_COLS == 16) {
                if (iq_r16_batch3_enabled() && n_rows >= 2) {
                    constexpr int64_t tile_size = 64;
                    alignas(64) float gate_tmp[3][tile_size], up_tmp[3][tile_size];
                    for (int64_t row = 0; row < n_rows; row += 3) {
                        const int nr = std::min<int64_t>(3, n_rows - row);
                        const block_q8_K * activations[3];
                        float * outputs[3];
                        for (int y = 0; y < nr; ++y) {
                            const row_mapping mapping = matrix_rows[expert * n_tokens + row + y];
                            activations[y] = (const block_q8_K *) (wdata +
                                mapping.i2 * src1_plane_size + (mapping.i1 % n_src_rows) * src1_row_size);
                            outputs[y] = (float *) ((char *) dst->data +
                                mapping.i2 * dst->nb[2] + mapping.i1 * dst->nb[1]);
                        }
                        for (int64_t tile_start = row_start; tile_start < row_end; tile_start += tile_size) {
                            const int64_t tile_rows = std::min(tile_size, row_end - tile_start);
                            ggml_gemv_iq_r16_batch<expanded_iq3_xxs>(k, gate_tmp[0], tile_size,
                                gate_cur + physical_row_offset(gate_w, tile_start), activations, nr, tile_rows);
                            ggml_gemv_iq_r16_batch<expanded_iq3_xxs>(k, up_tmp[0], tile_size,
                                up_cur + physical_row_offset(up_w, tile_start), activations, nr, tile_rows);
                            for (int y = 0; y < nr; ++y) {
                                ggml_vec_swiglu_f32(tile_rows, outputs[y] + tile_start, gate_tmp[y], up_tmp[y]);
                            }
                        }
                    }
                    continue;
                }
            }
            for (int64_t row = 0; row < n_rows; ++row) {
                const row_mapping mapping = matrix_rows[expert * n_tokens + row];
                const char * src1_col = wdata +
                    mapping.i2 * src1_plane_size + (mapping.i1 % n_src_rows) * src1_row_size;
                float * out_dst = (float *) ((char *) dst->data +
                    mapping.i2 * dst->nb[2] + mapping.i1 * dst->nb[1]) + row_start;

                // The skipped MUL_MAT_ID outputs have no other consumers, so keep gate/up
                // results in L1-sized thread-local tiles instead of materializing both tensors.
                constexpr int64_t tile_size = 64;
                float gate_tmp[tile_size];
                float up_tmp[tile_size];
                for (int64_t tile_start = row_start; tile_start < row_end; tile_start += tile_size) {
                    const int64_t tile_end = std::min(tile_start + tile_size, row_end);
                    const int64_t tile_rows = tile_end - tile_start;
                    gemv<BLOC_TYPE, INTER_SIZE, NB_COLS, PARAM_TYPE>(
                        k, gate_tmp, tile_rows, gate_cur + physical_row_offset(gate_w, tile_start),
                        src1_col, 1, tile_rows);
                    gemv<BLOC_TYPE, INTER_SIZE, NB_COLS, PARAM_TYPE>(
                        k, up_tmp, tile_rows, up_cur + physical_row_offset(up_w, tile_start),
                        src1_col, 1, tile_rows);
                    ggml_vec_swiglu_f32(
                        tile_rows, out_dst + tile_start - row_start, gate_tmp, up_tmp);
                }
            }
        }
    }

    bool compute_forward_mul_mat_id_weighted_sum(
            struct ggml_compute_params * params,
            struct ggml_tensor * down,
            struct ggml_tensor * weights,
            struct ggml_tensor * dst) override {
        if constexpr (!expanded_iq3_xxs) {
            GGML_UNUSED(params);
            GGML_UNUSED(down);
            GGML_UNUSED(weights);
            GGML_UNUSED(dst);
            return false;
        } else {
            if (down->op != GGML_OP_MUL_MAT_ID || down->src[0]->extra != this ||
                    strstr(down->name, "ffn_moe_down") == nullptr ||
                    weights->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
                return false;
            }

            if (validate_moe_down_weighted_sum()) {
                validate_forward_mul_mat_id_weighted_sum(params, down, weights, dst);
            } else {
                forward_mul_mat_id_weighted_sum(params, down, weights, dst);
            }
            return true;
        }
    }

    void validate_forward_mul_mat_id_weighted_sum(
            ggml_compute_params * params,
            ggml_tensor * down,
            ggml_tensor * weights,
            ggml_tensor * dst) {
        const ggml_tensor * down_w = down->src[0];
        const ggml_tensor * src1 = down->src[1];

        const int ith = params->ith;
        const int nth = params->nth;
        const int64_t n_out = down_w->ne[1];
        const int64_t n_experts = down_w->ne[2];
        const int64_t n_tokens = down->ne[2];

        const size_t src1_size = ggml_row_size(PARAM_TYPE, ggml_nelements(src1));
        size_t reference_offset = GGML_PAD(src1_size, sizeof(int64_t));
        reference_offset += sizeof(int64_t) * n_experts * (n_tokens + 1);
        reference_offset = GGML_PAD(reference_offset, alignof(float));

        GGML_ASSERT(params->wsize >=
            reference_offset + sizeof(float) * (n_out * n_tokens + nth));

        float * reference = (float *) ((char *) params->wdata + reference_offset);
        float * thread_error = reference + n_out * n_tokens;

        forward_mul_mat_id(params, down);

        int64_t row_start = (ith * n_out) / nth;
        int64_t row_end = ((ith + 1) * n_out) / nth;
        row_start = GGML_PAD(row_start, NB_COLS);
        row_end = GGML_PAD(row_end, NB_COLS);
        row_end = std::min(row_end, n_out);

        for (int64_t token = 0; token < n_tokens; ++token) {
            float * ref = reference + token * n_out + row_start;
            const float * expert0 = (const float *) ((const char *) down->data +
                token * down->nb[2]) + row_start;
            const float weight0 = *(const float *) ((const char *) weights->data +
                token * weights->nb[2]);
            const int64_t n_rows = row_end - row_start;

            memcpy(ref, expert0, n_rows * sizeof(float));
            ggml_vec_scale_f32(n_rows, ref, weight0);
            for (int64_t used = 1; used < down->ne[1]; ++used) {
                const float * expert = (const float *) ((const char *) down->data +
                    token * down->nb[2] + used * down->nb[1]) + row_start;
                const float weight = *(const float *) ((const char *) weights->data +
                    token * weights->nb[2] + used * weights->nb[1]);
                ggml_vec_mad_f32(n_rows, ref, expert, weight);
            }
        }

        // The fused pass reuses the MUL_MAT_ID workspace, including the routing
        // table.  Wait until every worker has finished consuming the reference
        // table before thread 0 replaces it with the compact staged-ID layout.
        ggml_barrier(params->threadpool);

        forward_mul_mat_id_weighted_sum(params, down, weights, dst);

        float max_abs_error = 0.0f;
        for (int64_t token = 0; token < n_tokens; ++token) {
            const float * ref = reference + token * n_out;
            const float * actual = (const float *) ((const char *) dst->data + token * dst->nb[1]);
            for (int64_t row = row_start; row < row_end; ++row) {
                max_abs_error = std::max(max_abs_error, fabsf(actual[row] - ref[row]));
            }
        }
        thread_error[ith] = max_abs_error;
        ggml_barrier(params->threadpool);

        if (ith == 0) {
            for (int thread = 1; thread < nth; ++thread) {
                max_abs_error = std::max(max_abs_error, thread_error[thread]);
            }
            GGML_ASSERT(max_abs_error <= 1e-5f);
            static std::atomic<bool> logged { false };
            bool expected = false;
            if (logged.compare_exchange_strong(expected, true, std::memory_order_relaxed)) {
                GGML_LOG_INFO(
                    "MOE_DOWN_WEIGHTED_SUM_VALIDATE max_abs_error=%.9g n_out=%" PRId64
                    " n_used=%" PRId64 " n_tokens=%" PRId64 "\n",
                    max_abs_error, n_out, down->ne[1], n_tokens);
            }
        }
        ggml_barrier(params->threadpool);
    }

    void forward_mul_mat_id_weighted_sum(
            ggml_compute_params * params,
            ggml_tensor * down,
            ggml_tensor * weights,
            ggml_tensor * dst) {
        const ggml_tensor * down_w = down->src[0];
        const ggml_tensor * src1 = down->src[1];
        const ggml_tensor * ids = down->src[2];

        const int ith = params->ith;
        const int nth = params->nth;
        const int64_t k = down_w->ne[0];
        const int64_t n_out = down_w->ne[1];
        const int64_t n_experts = down_w->ne[2];
        const int64_t n_used = ids->ne[0];
        const int64_t n_tokens = ids->ne[1];
        const int64_t n_src_rows = src1->ne[1];

        GGML_ASSERT(src1->type == GGML_TYPE_F32 && weights->type == GGML_TYPE_F32);
        GGML_ASSERT(down->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
        GGML_ASSERT(src1->ne[0] == k && n_src_rows == n_used && src1->ne[2] == n_tokens);
        GGML_ASSERT(down_w->ne[3] == 1 && src1->ne[3] == 1 && down->ne[3] == 1);
        GGML_ASSERT(down->ne[0] == n_out && down->ne[1] == n_used && down->ne[2] == n_tokens);
        GGML_ASSERT(weights->ne[0] == 1 && weights->ne[1] == n_used &&
                    weights->ne[2] == n_tokens);
        GGML_ASSERT(dst->ne[0] == n_out && dst->ne[1] == n_tokens);
        GGML_ASSERT(down_w->nb[0] == ggml_type_size(down_w->type));
        GGML_ASSERT(src1->nb[0] == sizeof(float));
        GGML_ASSERT(weights->nb[0] == sizeof(float) && dst->nb[0] == sizeof(float));

        const size_t src1_row_size = ggml_row_size(PARAM_TYPE, k);
        const size_t src1_plane_size = src1_row_size * n_src_rows;
        const size_t src1_size = src1_plane_size * n_tokens;
        const size_t staged_ids_offset = GGML_PAD(src1_size, sizeof(int64_t));
        const size_t staged_ids_size = n_tokens * n_used * sizeof(int32_t);
        GGML_ASSERT(params->wsize >= staged_ids_offset + staged_ids_size);

        char * wdata = (char *) params->wdata;
        int32_t * staged_ids = (int32_t *) (wdata + staged_ids_offset);
        const ggml_from_float_t from_float = ggml_get_type_traits_cpu(PARAM_TYPE)->from_float;
        const int64_t quant_block = ggml_blck_size(PARAM_TYPE);
        GGML_ASSERT(k % quant_block == 0);
        for (int64_t token = 0; token < n_tokens; ++token) {
            for (int64_t used = 0; used < n_src_rows; ++used) {
                const int64_t block_start = (ith * (k / quant_block)) / nth;
                const int64_t block_end = ((ith + 1) * (k / quant_block)) / nth;
                from_float(
                    (const float *) ((const char *) src1->data +
                        token * src1->nb[2] + used * src1->nb[1] +
                        block_start * quant_block * src1->nb[0]),
                    wdata + token * src1_plane_size + used * src1_row_size +
                        block_start * ggml_type_size(PARAM_TYPE),
                    (block_end - block_start) * quant_block);
            }
        }

        // Match the established MUL_MAT_ID synchronization pattern: the routing IDs
        // are produced by an earlier graph node, so read them once on thread 0 and
        // publish a compact copy before every worker enters the expert loop.
        if (ith == 0) {
            for (int64_t token = 0; token < n_tokens; ++token) {
                for (int64_t used = 0; used < n_used; ++used) {
                    const int32_t expert = *(const int32_t *) ((const char *) ids->data +
                        token * ids->nb[1] + used * ids->nb[0]);
                    GGML_ASSERT(expert >= 0 && expert < n_experts);
                    staged_ids[token * n_used + used] = expert;
                }
            }
        }

        ggml_barrier(params->threadpool);

        int active_nth = nth;
        if (!validate_moe_down_weighted_sum() && n_tokens == 1) {
            const int requested = moe_single_token_threads();
            if (requested > 0) {
                active_nth = std::min(active_nth, requested);
            }
        }
        if (ith >= active_nth) {
            return;
        }

        int64_t row_start = (ith * n_out) / active_nth;
        int64_t row_end = ((ith + 1) * n_out) / active_nth;
        row_start = GGML_PAD(row_start, NB_COLS);
        row_end = GGML_PAD(row_end, NB_COLS);
        row_end = std::min(row_end, n_out);
        if (row_start >= row_end) {
            return;
        }

        const int64_t tile_size = moe_down_weighted_sum_tile_size();
        alignas(64) float tmp[512];

        for (int64_t token = 0; token < n_tokens; ++token) {
            float * out = (float *) ((char *) dst->data + token * dst->nb[1]);
            for (int64_t used = 0; used < n_used; ++used) {
                const int32_t expert = staged_ids[token * n_used + used];

                const char * down_cur =
                    (const char *) down_w->data + physical_expert_offset(down_w, expert);
                const char * src1_col =
                    wdata + token * src1_plane_size + used * src1_row_size;
                const float weight = *(const float *) ((const char *) weights->data +
                    token * weights->nb[2] + used * weights->nb[1]);

                for (int64_t tile_start = row_start; tile_start < row_end; tile_start += tile_size) {
                    const int64_t tile_end = std::min(tile_start + tile_size, row_end);
                    const int64_t tile_rows = tile_end - tile_start;
                    gemv<BLOC_TYPE, INTER_SIZE, NB_COLS, PARAM_TYPE>(
                        k, tmp, tile_rows,
                        down_cur + physical_row_offset(down_w, tile_start),
                        src1_col, 1, tile_rows);

                    float * out_tile = out + tile_start;
                    if (used == 0) {
                        memcpy(out_tile, tmp, tile_rows * sizeof(float));
                        ggml_vec_scale_f32(tile_rows, out_tile, weight);
                    } else {
                        ggml_vec_mad_f32(tile_rows, out_tile, tmp, weight);
                    }
                }
            }
        }
    }

    void forward_mul_mat_one_chunk(ggml_compute_params * params,
                                   ggml_tensor *         op,
                                   int64_t               src0_start,
                                   int64_t               src0_end,
                                   int64_t               src1_start,
                                   int64_t               src1_end) {
        const ggml_tensor * src0 = op->src[0];
        const ggml_tensor * src1 = op->src[1];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_BINARY_OP_LOCALS

        const size_t src1_col_stride = ggml_row_size(PARAM_TYPE, ne10);

        GGML_ASSERT(ne03 == 1 && ne13 == 1);
        GGML_ASSERT(ne12 % ne02 == 0);
        const int64_t r2 = ne12 / ne02;

        const int64_t i12 = src1_start / ne1;
        const int64_t i11 = src1_start - i12 * ne1;

        // Determine batch index
        const int64_t i02 = i12 / r2;

        const int64_t i1 = i11;
        const int64_t i2 = i12;

        const char * src0_ptr = (const char *) src0->data + physical_expert_offset(src0, i02);
        const char * src1_ptr = (const char *) params->wdata + (i11 + i12 * ne11) * src1_col_stride;
        char *       dst_ptr  = ((char *) dst->data + (i1 * nb1 + i2 * nb2));

        const int64_t nrows = src1_end - src1_start;
        const int64_t ncols = src0_end - src0_start;

        GGML_ASSERT(src1_ptr + src1_col_stride * nrows <= (const char *) params->wdata + params->wsize);

        // The Q8 VNNI kernel can share each packed weight load across the two
        // or three ordinary Q8 activation rows used by speculative
        // verification. Other repack formats retain their single-row GEMV
        // fallback because their GEMV entry points require nr == 1.
        if constexpr (std::is_same_v<BLOC_TYPE, block_q8_0> &&
                INTER_SIZE == 8 && NB_COLS == 8 && PARAM_TYPE == GGML_TYPE_Q8_0) {
            if (nrows > 0 && nrows <= 3) {
                gemv<BLOC_TYPE, INTER_SIZE, NB_COLS, PARAM_TYPE>(
                    ne00, (float *) dst_ptr + src0_start, nb1 / nb0,
                    src0_ptr + physical_row_offset(src0, src0_start), src1_ptr,
                    nrows, ncols);
                return;
            }
        }

        // If there are more than three rows in src1, use gemm; otherwise, use gemv.
        if (nrows > 3) {
            gemm<BLOC_TYPE, INTER_SIZE, NB_COLS, PARAM_TYPE>(ne00, (float *) (dst_ptr) + src0_start, nb1 / nb0,
                                                             src0_ptr + physical_row_offset(src0, src0_start), src1_ptr,
                                                             nrows - (nrows % 4), ncols);
        }
        for (int iter = nrows - (nrows % 4); iter < nrows; iter++) {
            gemv<BLOC_TYPE, INTER_SIZE, NB_COLS, PARAM_TYPE>(ne00, (float *) (dst_ptr + (iter * nb1)) + src0_start,
                                                             ne01, src0_ptr + physical_row_offset(src0, src0_start),
                                                             src1_ptr + (src1_col_stride * iter), 1 /* nrows */, ncols);
        }
    }

    void forward_mul_mat(ggml_compute_params * params, ggml_tensor * op) {
        const ggml_tensor * src0 = op->src[0];
        const ggml_tensor * src1 = op->src[1];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_BINARY_OP_LOCALS

        const int ith = params->ith;
        const int nth = params->nth;

        GGML_ASSERT(ne0 == ne01);
        GGML_ASSERT(ne1 == ne11);
        GGML_ASSERT(ne2 == ne12);
        GGML_ASSERT(ne3 == ne13);

        // dst cannot be transposed or permuted
        GGML_ASSERT(nb0 == sizeof(float));
        GGML_ASSERT(nb0 <= nb1);
        GGML_ASSERT(nb1 <= nb2);
        GGML_ASSERT(nb2 <= nb3);

        // TODO: General batched mul mat for 4D tensors
        // Currently only supports 3D tensors
        GGML_ASSERT(ne03 == 1);
        GGML_ASSERT(ne13 == 1);
        GGML_ASSERT(ne3 == 1);

        GGML_ASSERT(src1->type == GGML_TYPE_F32);

        GGML_ASSERT(ggml_n_dims(op->src[0]) == 2);
        // GGML_ASSERT(ggml_n_dims(op->src[1]) == 2);

        char *       wdata = static_cast<char *>(params->wdata);
        const size_t nbw1  = ggml_row_size(PARAM_TYPE, ne10);
        const size_t nbw2  = nbw1 * ne11;

        assert(params->wsize >= nbw2 * ne12);

        const ggml_from_float_t from_float = ggml_get_type_traits_cpu(PARAM_TYPE)->from_float;

        // INFO: Quantization is done in planes to avoid extra complexity in chunking.
        // Flattening dimensions not multiple of INTER_SIZE would require extra handling depending on how
        // the planes are broadcast.
        for (int64_t i12 = 0; i12 < ne12; i12++) {
            char * data_ptr  = (char *) src1->data + i12 * nb12;
            char * wdata_ptr = wdata + i12 * nbw2;

            const bool src1_rows_contiguous = nb11 == (size_t) ne10 * sizeof(float);
            std::vector<float> row_tmp;
            if (!src1_rows_contiguous) {
                row_tmp.resize((size_t) 4 * ne10);
            }
            for (int64_t i11 = ith * 4; i11 < ne11 - ne11 % 4; i11 += nth * 4) {
                const float * rows_ptr = (const float *) (data_ptr + i11 * nb11);
                if (!src1_rows_contiguous) {
                    // the 4-row quantizer expects contiguous rows; gather strided rows (e.g. k-slice views)
                    for (int r = 0; r < 4; r++) {
                        memcpy(row_tmp.data() + (size_t) r * ne10, data_ptr + (i11 + r) * nb11, (size_t) ne10 * sizeof(float));
                    }
                    rows_ptr = row_tmp.data();
                }
                if constexpr (expanded_iq) {
                    for (int r = 0; r < 4; ++r) {
                        from_float(rows_ptr + r * ne10, wdata_ptr + (i11 + r) * nbw1, ne10);
                    }
                } else {
                    ggml_quantize_mat_t<INTER_SIZE, PARAM_TYPE>(rows_ptr, (void *) (wdata_ptr + i11 * nbw1), 4, ne10);
                }
            }

            const int64_t i11_processed = ne11 - ne11 % 4;
            for (int64_t i11 = i11_processed + ith; i11 < ne11; i11 += nth) {
                from_float((float *) (data_ptr + i11 * nb11), (void *) (wdata_ptr + i11 * nbw1), ne10);
            }
        }

        // disable for NUMA
        const bool disable_chunking = ggml_is_numa();

        // 4x chunks per thread
        const int64_t nr0 = ggml_nrows(op->src[0]);

        int     nth_scaled  = nth * 4;
        int64_t chunk_size0 = (nr0 + nth_scaled - 1) / nth_scaled;
        int64_t nchunk0     = (nr0 + chunk_size0 - 1) / chunk_size0;

        // src1 is chunked only by full planes.
        // When we flatten we need to address dimensions not multiple of the q8 INTER_SIZE
        // to route them thorugh GEMV.
        // nchunk1 = ne12 also avoids messing the chunking for models with no 3d tensors
        // to avoid affecting their performance
        int64_t nchunk1 = ne12;

        // Ensure minimum chunk size to avoid alignment issues with high thread counts
        // Minimum chunk size should be at least NB_COLS to prevent overlapping chunks after alignment
        const int64_t min_chunk_size = NB_COLS;
        if (nchunk0 > 0 && (nr0 / nchunk0) < min_chunk_size && nr0 >= min_chunk_size) {
            nchunk0 = (nr0 + min_chunk_size - 1) / min_chunk_size;
        }

        int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
        // Only increase nchunk0 to nth if it won't make chunks too small
        if (nth == 1 || ((nchunk0 < nth || disable_chunking) && (nr0 + nth - 1) / nth >= min_chunk_size)) {
            nchunk0 = nth;
            dr0 = (nr0 + nchunk0 - 1) / nchunk0;
        }

        // Ensure nchunk doesn't exceed the number of rows divided by minimum chunk size
        // This prevents creating too many tiny chunks that could overlap after alignment
        const int64_t max_nchunk = (nr0 + min_chunk_size - 1) / min_chunk_size;
        nchunk0                  = MIN(nchunk0, max_nchunk);

        if (ith == 0) {
            // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
            ggml_threadpool_chunk_set(params->threadpool, nth);
        }

        ggml_barrier(params->threadpool);

        // The first chunk comes from our thread_id, the rest will get auto-assigned.
        int current_chunk = ith;

        while (current_chunk < nchunk0 * nchunk1) {
            const int64_t ith0 = current_chunk % nchunk0;
            const int64_t ith1 = current_chunk / nchunk0;

            int64_t src0_start = dr0 * ith0;
            int64_t src0_end   = MIN(src0_start + dr0, nr0);

            // full-plane range for src1
            int64_t src1_start = ith1 * ne11;
            int64_t src1_end = (ith1 + 1) * ne11;

            // Align boundaries to NB_COLS - round up to ensure all data is included
            // The chunk size limiting above ensures chunks are large enough to prevent overlaps
            src0_start = (src0_start % NB_COLS) ? src0_start + NB_COLS - (src0_start % NB_COLS) : src0_start;
            src0_end   = (src0_end % NB_COLS) ? src0_end + NB_COLS - (src0_end % NB_COLS) : src0_end;
            src0_end   = MIN(src0_end, ne01);

            // Make sure current plane is the last one before exiting
            if (src0_start >= src0_end) {
                current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
                continue;
            }

            forward_mul_mat_one_chunk(params, dst, src0_start, src0_end, src1_start, src1_end);

            current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
        }
    }

    void forward_mul_mat_id(ggml_compute_params * params, ggml_tensor * op) {
        const ggml_tensor * src0 = op->src[0];
        const ggml_tensor * src1 = op->src[1];
        const ggml_tensor * ids  = op->src[2];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_BINARY_OP_LOCALS

        const int ith = params->ith;
        const int nth = params->nth;

        const ggml_from_float_t from_float = ggml_get_type_traits_cpu(PARAM_TYPE)->from_float;

        // we don't support permuted src0 or src1
        GGML_ASSERT(nb00 == ggml_type_size(src0->type));
        GGML_ASSERT(nb10 == ggml_type_size(src1->type));

        // dst cannot be transposed or permuted
        GGML_ASSERT(nb0 == sizeof(float));
        GGML_ASSERT(nb0 <= nb1);
        GGML_ASSERT(nb1 <= nb2);
        GGML_ASSERT(nb2 <= nb3);

        GGML_ASSERT(ne03 == 1);
        GGML_ASSERT(ne13 == 1);
        GGML_ASSERT(ne3  == 1);

        GGML_ASSERT(src1->type == GGML_TYPE_F32);

        // row groups
        const int n_ids = ids->ne[0]; // n_expert_used
        const int n_as  = ne02;       // n_expert

        const size_t nbw1 = ggml_row_size(PARAM_TYPE, ne10);
        const size_t nbw2 = nbw1*ne11;
        const size_t nbw3 = nbw2*ne12;

        struct mmid_row_mapping {
            int32_t i1;
            int32_t i2;
        };

        GGML_ASSERT(params->wsize >=
                (GGML_PAD(nbw3, sizeof(int64_t)) +
                 n_as*(ne12 + 1)*sizeof(mmid_row_mapping))
                );

        auto * wdata          = (char *)params->wdata;
        auto * wdata_src1_end = (char *)wdata + GGML_PAD(nbw3, sizeof(int64_t));

        // total of [n_as][ne12 + 1] elements of type mmid_row_mapping (2*int32_t = int64_t)
        auto * matrix_row_counts = (int64_t *) (wdata_src1_end);                                        // [n_as]
        struct mmid_row_mapping * matrix_rows = (struct mmid_row_mapping *) (matrix_row_counts + n_as); // [n_as][ne12]

        // src1: float32 => param type
        for (int64_t i12 = 0; i12 < ne12; ++i12) {
            for (int64_t i11 = ith; i11 < ne11; i11 += nth) {
                from_float((float *)((char *) src1->data + i12 * nb12 + i11 * nb11),
                           (void *)               (wdata + i12 * nbw2 + i11 * nbw1),
                           ne10);
            }
        }

#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id) * ne12 + (i1)]

        if (ith == 0) {
            // initialize matrix_row_counts
            memset(matrix_row_counts, 0, n_as * sizeof(int64_t));

            // group rows by src0 matrix
            for (int32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
                for (int32_t id = 0; id < n_ids; ++id) {
                    const int32_t i02 =
                        *(const int32_t *) ((const char *) ids->data + iid1 * ids->nb[1] + id * ids->nb[0]);

                    GGML_ASSERT(i02 >= 0 && i02 < n_as);

                    MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = { id, iid1 };
                    matrix_row_counts[i02] += 1;
                }
            }
        }

        ggml_barrier(params->threadpool);

        // compute each matrix multiplication in sequence
        for (int cur_a = 0; cur_a < n_as; ++cur_a) {
            const int64_t cne1 = matrix_row_counts[cur_a];

            if (cne1 == 0) {
                continue;
            }

            const auto * src0_cur = (const char *) src0->data + physical_expert_offset(src0, cur_a);

            //const int64_t nr0 = ne01; // src0 rows
            const int64_t nr1 = cne1; // src1 rows

            int64_t src0_cur_start = (ith * ne01) / nth;
            int64_t src0_cur_end   = ((ith + 1) * ne01) / nth;

            // Align boundaries to NB_COLS - round up to ensure all data is included
            src0_cur_start = (src0_cur_start % NB_COLS) ? src0_cur_start + NB_COLS - (src0_cur_start % NB_COLS) : src0_cur_start;
            src0_cur_end   = (src0_cur_end   % NB_COLS) ? src0_cur_end   + NB_COLS - (src0_cur_end   % NB_COLS) : src0_cur_end;
            if (src0_cur_end > ne01) {
                src0_cur_end = ne01;
            }

            if (src0_cur_start >= src0_cur_end) {
                return;
            }


            if constexpr (expanded_iq && NB_COLS == 16) {
                if (iq_r16_batch3_enabled() && nr1 >= 2) {
                    constexpr int64_t tile_size = 64;
                    alignas(64) float tmp[3][tile_size];
                    for (int64_t row = 0; row < nr1; row += 3) {
                        const int nr = std::min<int64_t>(3, nr1 - row);
                        const block_q8_K * activations[3];
                        float * outputs[3];
                        for (int y = 0; y < nr; ++y) {
                            const mmid_row_mapping mapping = MMID_MATRIX_ROW(cur_a, row + y);
                            activations[y] = (const block_q8_K *) (wdata +
                                (mapping.i1 % ne11) * nbw1 + mapping.i2 * nbw2);
                            outputs[y] = (float *) ((char *) dst->data + mapping.i1 * nb1 + mapping.i2 * nb2);
                        }
                        for (int64_t tile_start = src0_cur_start; tile_start < src0_cur_end; tile_start += tile_size) {
                            const int64_t tile_rows = std::min(tile_size, src0_cur_end - tile_start);
                            ggml_gemv_iq_r16_batch<expanded_iq3_xxs>(ne00, tmp[0], tile_size,
                                src0_cur + physical_row_offset(src0, tile_start), activations, nr, tile_rows);
                            for (int y = 0; y < nr; ++y) {
                                memcpy(outputs[y] + tile_start, tmp[y], tile_rows * sizeof(float));
                            }
                        }
                    }
                    continue;
                }
            }
            for (int ir1 = 0; ir1 < nr1; ir1++) {
                struct mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, ir1);

                const int id = row_mapping.i1;  // selected expert index

                const int64_t i11 = id % ne11;
                const int64_t i12 = row_mapping.i2;  // row index in src1

                const int64_t i1 = id;               // selected expert index
                const int64_t i2 = i12;              // row

                const auto * src1_col = (const char *) wdata + (i11 * nbw1 + i12 * nbw2);

                gemv<BLOC_TYPE, INTER_SIZE, NB_COLS, PARAM_TYPE>(
                    ne00, (float *) ((char *) dst->data + (i1 * nb1 + i2 * nb2)) + src0_cur_start, ne01,
                    src0_cur + physical_row_offset(src0, src0_cur_start), src1_col, 1,
                    src0_cur_end - src0_cur_start);
            }
        }
#undef MMID_MATRIX_ROW
    }

    int repack(struct ggml_tensor * t, const void * data, size_t data_size) override {
        GGML_LOG_DEBUG("%s: repack tensor %s with %s_%dx%d\n", __func__, t->name, ggml_type_name(t->type),
                       (int) NB_COLS, (int) INTER_SIZE);
        return ggml::cpu::repack::repack<BLOC_TYPE, INTER_SIZE, NB_COLS>(t, data, data_size);
    }
};

}  // namespace ggml::cpu::repack


// =====================================================================================
// x16 layout: generic (scalar) kernels, packers, requantizing packers, and traits.
// =====================================================================================
static inline void ggml_x16_get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else { *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4); *m = (q[j+4] >> 4) | ((q[j-0] >> 6) << 4); }
}
// pack 16 native rows (row stride nb blocks) into nb x16 blocks
static void ggml_x16_pack_q4_K(const block_q4_K * in, int nb, block_q4_K_x16 * out) {
    for (int b = 0; b < nb; b++) { block_q4_K_x16 & o = out[b];
        for (int r = 0; r < 16; r++) { const block_q4_K & x = in[(size_t) r * nb + b];
            o.d[r] = x.GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d; o.dmin[r] = x.GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
            for (int sb = 0; sb < 8; sb++) { uint8_t sc, mn; ggml_x16_get_scale_min_k4(sb, x.scales, &sc, &mn); o.scales[sb][r] = sc; o.mins[sb][r] = mn; }
            for (int j = 0; j < 4; j++) for (int i0 = 0; i0 < 8; i0++) for (int t = 0; t < 4; t++) o.qs[((j*8 + i0)*16 + r)*4 + t] = x.qs[32*j + 4*i0 + t];
        } }
}
static void ggml_x16_pack_q5_K(const block_q5_K * in, int nb, block_q5_K_x16 * out) {
    for (int b = 0; b < nb; b++) { block_q5_K_x16 & o = out[b];
        for (int r = 0; r < 16; r++) { const block_q5_K & x = in[(size_t) r * nb + b];
            o.d[r] = x.GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d; o.dmin[r] = x.GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
            for (int sb = 0; sb < 8; sb++) { uint8_t sc, mn; ggml_x16_get_scale_min_k4(sb, x.scales, &sc, &mn); o.scales[sb][r] = sc; o.mins[sb][r] = mn; }
            for (int i0 = 0; i0 < 8; i0++) {
                for (int j = 0; j < 4; j++) for (int t = 0; t < 4; t++) o.qsh[i0*320 + j*64 + r*4 + t] = x.qs[32*j + 4*i0 + t];
                for (int t = 0; t < 4; t++) o.qsh[i0*320 + 256 + r*4 + t] = x.qh[4*i0 + t];
            }
        } }
}

struct block_q5_K_x16_bytes {
    ggml_half d[16], dmin[16];
    uint8_t scales[8][16], mins[8][16];
    uint8_t qs[8][8][64];
};
static_assert(sizeof(block_q5_K_x16_bytes) == 4416, "wrong byte-expanded Q5 block size");

static void ggml_x16_pack_q5_K_bytes(const block_q5_K * in, int nb, block_q5_K_x16_bytes * out) {
    std::vector<block_q5_K_x16> compact(nb);
    ggml_x16_pack_q5_K(in, nb, compact.data());
    for (int b = 0; b < nb; ++b) {
        memcpy(out[b].d, compact[b].d, sizeof(out[b].d));
        memcpy(out[b].dmin, compact[b].dmin, sizeof(out[b].dmin));
        memcpy(out[b].scales, compact[b].scales, sizeof(out[b].scales));
        memcpy(out[b].mins, compact[b].mins, sizeof(out[b].mins));
        for (int sb = 0; sb < 8; ++sb) for (int chunk = 0; chunk < 8; ++chunk) {
            const uint8_t * q = compact[b].qsh + chunk * 320;
            for (int i = 0; i < 64; ++i) {
                const uint8_t lo = (q[(sb / 2) * 64 + i] >> (4 * (sb & 1))) & 15;
                out[b].qs[sb][chunk][i] = lo | (((q[256 + i] >> sb) & 1) << 4);
            }
        }
    }
}

template <int NR>
static void ggml_gemv_q5_K_x16_bytes_impl(
        int n, float * s, size_t bs, const void * vx, const void * vy, int nc) {
    GGML_ASSERT(n % QK_K == 0 && nc % 16 == 0 && NR >= 1 && NR <= 3);
    const int nb = n / QK_K;
    const auto * weights = (const block_q5_K_x16_bytes *) vx;
    const auto * activations = (const block_q8_K *) vy;
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
    for (int g = 0; g < nc / 16; ++g) {
        const block_q5_K_x16_bytes * w = weights + g * nb;
        __m512 sum[NR];
        for (int y = 0; y < NR; ++y) sum[y] = _mm512_setzero_ps();
        for (int b = 0; b < nb; ++b) {
            __m512i isum[NR], imin[NR];
            for (int y = 0; y < NR; ++y) isum[y] = imin[y] = _mm512_setzero_si512();
            for (int sb = 0; sb < 8; ++sb) {
                __m512i dots[NR][4];
                for (int y = 0; y < NR; ++y) for (int part = 0; part < 4; ++part) dots[y][part] = _mm512_setzero_si512();
#if defined(__GNUC__)
#pragma GCC unroll 8
#endif
                for (int chunk = 0; chunk < 8; ++chunk) {
                    const __m512i q = _mm512_loadu_si512(w[b].qs[sb][chunk]);
                    for (int y = 0; y < NR; ++y) {
                        int32_t aq;
                        memcpy(&aq, activations[y * nb + b].qs + sb * 32 + chunk * 4, sizeof(aq));
                        dots[y][chunk & 3] = _mm512_dpbusd_epi32(dots[y][chunk & 3], q, _mm512_set1_epi32(aq));
                    }
                }
                const __m512i scale = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *) w[b].scales[sb]));
                const __m512i min = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *) w[b].mins[sb]));
                for (int y = 0; y < NR; ++y) {
                    const block_q8_K & a = activations[y * nb + b];
                    const __m512i dot = _mm512_add_epi32(_mm512_add_epi32(dots[y][0], dots[y][1]), _mm512_add_epi32(dots[y][2], dots[y][3]));
                    isum[y] = _mm512_add_epi32(isum[y], _mm512_mullo_epi32(dot, scale));
                    imin[y] = _mm512_add_epi32(imin[y], _mm512_mullo_epi32(min, _mm512_set1_epi32(int(a.bsums[2 * sb]) + a.bsums[2 * sb + 1])));
                }
            }
            const __m512 d = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) w[b].d));
            const __m512 dmin = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) w[b].dmin));
            for (int y = 0; y < NR; ++y) {
                const __m512 ad = _mm512_set1_ps(activations[y * nb + b].d);
                sum[y] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(isum[y]), _mm512_mul_ps(d, ad), sum[y]);
                sum[y] = _mm512_fnmadd_ps(_mm512_cvtepi32_ps(imin[y]), _mm512_mul_ps(dmin, ad), sum[y]);
            }
        }
        for (int y = 0; y < NR; ++y) _mm512_storeu_ps(s + y * bs + g * 16, sum[y]);
    }
#else
    for (int g = 0; g < nc / 16; ++g) for (int y = 0; y < NR; ++y) {
        const block_q5_K_x16_bytes * w = weights + g * nb;
        const block_q8_K * a = activations + y * nb;
        for (int row = 0; row < 16; ++row) {
            float sum = 0;
            for (int b = 0; b < nb; ++b) {
                int32_t isum = 0, imin = 0;
                for (int sb = 0; sb < 8; ++sb) {
                    int32_t dot = 0;
                    for (int k = 0; k < 32; ++k) dot += int(w[b].qs[sb][k / 4][row * 4 + k % 4]) * int(a[b].qs[sb * 32 + k]);
                    isum += dot * w[b].scales[sb][row];
                    imin += int(w[b].mins[sb][row]) * (int(a[b].bsums[2 * sb]) + a[b].bsums[2 * sb + 1]);
                }
                sum = std::fma(float(isum), GGML_FP16_TO_FP32(w[b].d[row]) * a[b].d, sum);
                sum = std::fma(-float(imin), GGML_FP16_TO_FP32(w[b].dmin[row]) * a[b].d, sum);
            }
            s[y * bs + g * 16 + row] = sum;
        }
    }
#endif
}

static void ggml_gemv_q5_K_x16_bytes_q8_K(
        int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    GGML_ASSERT(nr >= 1 && nr <= 3);
    if (nr == 1) {
        ggml_gemv_q5_K_x16_bytes_impl<1>(n, s, bs, vx, vy, nc);
    } else if (nr == 2) {
        ggml_gemv_q5_K_x16_bytes_impl<2>(n, s, bs, vx, vy, nc);
    } else {
        ggml_gemv_q5_K_x16_bytes_impl<3>(n, s, bs, vx, vy, nc);
    }
}

static void ggml_x16_pack_q6_K(const block_q6_K * in, int nb, block_q6_K_x16 * out) {
    for (int b = 0; b < nb; b++) { block_q6_K_x16 & o = out[b];
        for (int r = 0; r < 16; r++) { const block_q6_K & x = in[(size_t) r * nb + b];
            o.d[r] = x.d;
            for (int sb = 0; sb < 16; sb++) o.scales[sb][r] = x.scales[sb];
            for (int h = 0; h < 2; h++) for (int i0 = 0; i0 < 8; i0++) {
                uint8_t * dst = o.q + (h*8 + i0) * 192;
                for (int t = 0; t < 4; t++) { dst[r*4 + t] = x.ql[64*h + 4*i0 + t]; dst[64 + r*4 + t] = x.ql[64*h + 32 + 4*i0 + t]; dst[128 + r*4 + t] = x.qh[32*h + 4*i0 + t]; }
            }
        } }
}
static void ggml_x16_pack_q8_0(const block_q8_0 * in, int nb, block_q8_0_x16 * out) {
    for (int b = 0; b < nb; b++) { block_q8_0_x16 & o = out[b];
        for (int r = 0; r < 16; r++) { const block_q8_0 & x = in[(size_t) r * nb + b]; o.d[r] = x.d;
            for (int i0 = 0; i0 < 8; i0++) for (int t = 0; t < 4; t++) o.qs[(i0*16 + r)*4 + t] = (uint8_t)((int) x.qs[4*i0 + t] + 128);
        } }
}
// scalar reference kernels (used when AVX-512 VNNI is unavailable)
void ggml_gemv_q4_K_x16_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    GGML_ASSERT(nr == 1); GGML_UNUSED(bs);
    const int nb = n / QK_K; const block_q4_K_x16 * vxb = (const block_q4_K_x16 *) vx; const block_q8_K * y = (const block_q8_K *) vy;
    for (int g = 0; g < nc / 16; g++) { const block_q4_K_x16 * bp = vxb + (size_t) g * nb;
        for (int r = 0; r < 16; r++) { float sumf = 0.f;
            for (int b = 0; b < nb; b++) { int32_t isum = 0, imin = 0;
                for (int j = 0; j < 4; j++) for (int i0 = 0; i0 < 8; i0++) for (int t = 0; t < 4; t++) {
                    const uint8_t q = bp[b].qs[((j*8 + i0)*16 + r)*4 + t];
                    isum += (int)(q & 0xF) * y[b].qs[64*j + 4*i0 + t] * bp[b].scales[2*j][r];
                    isum += (int)(q >> 4) * y[b].qs[64*j + 32 + 4*i0 + t] * bp[b].scales[2*j+1][r];
                }
                for (int sb = 0; sb < 8; sb++) imin += (int) bp[b].mins[sb][r] * ((int) y[b].bsums[2*sb] + y[b].bsums[2*sb+1]);
                sumf += y[b].d * (GGML_CPU_FP16_TO_FP32(bp[b].d[r]) * isum - GGML_CPU_FP16_TO_FP32(bp[b].dmin[r]) * imin);
            }
            s[g*16 + r] = sumf; } }
}
void ggml_gemv_q5_K_x16_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    GGML_ASSERT(nr == 1); GGML_UNUSED(bs);
    const int nb = n / QK_K; const block_q5_K_x16 * vxb = (const block_q5_K_x16 *) vx; const block_q8_K * y = (const block_q8_K *) vy;
    for (int g = 0; g < nc / 16; g++) { const block_q5_K_x16 * bp = vxb + (size_t) g * nb;
        for (int r = 0; r < 16; r++) { float sumf = 0.f;
            for (int b = 0; b < nb; b++) { int32_t isum = 0, imin = 0;
                for (int i0 = 0; i0 < 8; i0++) for (int t = 0; t < 4; t++) {
                    const uint8_t h = bp[b].qsh[i0*320 + 256 + r*4 + t];
                    for (int j = 0; j < 4; j++) {
                        const uint8_t q = bp[b].qsh[i0*320 + j*64 + r*4 + t];
                        const int lo = (q & 0xF) | (((h >> (2*j)) & 1) << 4), hi = (q >> 4) | (((h >> (2*j+1)) & 1) << 4);
                        isum += lo * y[b].qs[64*j + 4*i0 + t] * bp[b].scales[2*j][r];
                        isum += hi * y[b].qs[64*j + 32 + 4*i0 + t] * bp[b].scales[2*j+1][r];
                    }
                }
                for (int sb = 0; sb < 8; sb++) imin += (int) bp[b].mins[sb][r] * ((int) y[b].bsums[2*sb] + y[b].bsums[2*sb+1]);
                sumf += y[b].d * (GGML_CPU_FP16_TO_FP32(bp[b].d[r]) * isum - GGML_CPU_FP16_TO_FP32(bp[b].dmin[r]) * imin);
            }
            s[g*16 + r] = sumf; } }
}
void ggml_gemv_q6_K_x16_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    GGML_ASSERT(nr == 1); GGML_UNUSED(bs);
    const int nb = n / QK_K; const block_q6_K_x16 * vxb = (const block_q6_K_x16 *) vx; const block_q8_K * y = (const block_q8_K *) vy;
    for (int g = 0; g < nc / 16; g++) { const block_q6_K_x16 * bp = vxb + (size_t) g * nb;
        for (int r = 0; r < 16; r++) { float sumf = 0.f;
            for (int b = 0; b < nb; b++) { int32_t isum = 0;
                for (int h = 0; h < 2; h++) for (int i0 = 0; i0 < 8; i0++) for (int t = 0; t < 4; t++) {
                    const uint8_t * qi = bp[b].q + (h*8 + i0) * 192;
                    const uint8_t la = qi[r*4 + t], lb = qi[64 + r*4 + t], hv = qi[128 + r*4 + t];
                    const int v0 = (la & 0xF) | ((hv & 3) << 4), v1 = (lb & 0xF) | (((hv >> 2) & 3) << 4);
                    const int v2 = (la >> 4) | (((hv >> 4) & 3) << 4), v3 = (lb >> 4) | (((hv >> 6) & 3) << 4);
                    const int p = 128*h + 4*i0 + t;
                    isum += (v0 - 32) * y[b].qs[p]      * bp[b].scales[(p) / 16][r];
                    isum += (v1 - 32) * y[b].qs[p + 32] * bp[b].scales[(p + 32) / 16][r];
                    isum += (v2 - 32) * y[b].qs[p + 64] * bp[b].scales[(p + 64) / 16][r];
                    isum += (v3 - 32) * y[b].qs[p + 96] * bp[b].scales[(p + 96) / 16][r];
                }
                sumf += y[b].d * GGML_CPU_FP16_TO_FP32(bp[b].d[r]) * isum;
            }
            s[g*16 + r] = sumf; } }
}
void ggml_gemv_q8_0_x16_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    GGML_ASSERT(nr == 1); GGML_UNUSED(bs);
    const int nb = n / QK8_0; const block_q8_0_x16 * vxb = (const block_q8_0_x16 *) vx; const block_q8_0 * y = (const block_q8_0 *) vy;
    for (int g = 0; g < nc / 16; g++) { const block_q8_0_x16 * bp = vxb + (size_t) g * nb;
        for (int r = 0; r < 16; r++) { float sumf = 0.f;
            for (int b = 0; b < nb; b++) { int32_t isum = 0;
                for (int i0 = 0; i0 < 8; i0++) for (int t = 0; t < 4; t++) isum += ((int) bp[b].qs[(i0*16 + r)*4 + t] - 128) * y[b].qs[4*i0 + t];
                sumf += GGML_CPU_FP16_TO_FP32(bp[b].d[r]) * GGML_CPU_FP16_TO_FP32(y[b].d) * isum;
            }
            s[g*16 + r] = sumf; } }
}

namespace ggml::cpu::repack {

typedef void (*ggml_x16_gemv_t)(int, float *, size_t, const void *, const void *, int, int);

struct x16_spec {
    ggml_type        src_type;    // type of the tensor in the GGUF
    ggml_type        dst_type;    // payload type inside the x16 layout
    int64_t          blck;        // QK_K or QK8_0
    size_t           x16_bytes;   // bytes of one x16 block (16 rows x blck weights)
    ggml_type        param_type;  // activation quantization type
    ggml_x16_gemv_t  gemv;
    const char *     name;
};

static void ggml_x16_pack_rows(const x16_spec & sp, const void * rows, int64_t k, char * dst) {
    // rows: 16 native rows of dst_type (contiguous); dst: nb x16 blocks
    const int nb = (int) (k / sp.blck);
    switch (sp.dst_type) {
        case GGML_TYPE_Q4_K: ggml_x16_pack_q4_K((const block_q4_K *) rows, nb, (block_q4_K_x16 *) dst); break;
        case GGML_TYPE_Q5_K:
            if (sp.x16_bytes == sizeof(block_q5_K_x16_bytes)) {
                ggml_x16_pack_q5_K_bytes((const block_q5_K *) rows, nb, (block_q5_K_x16_bytes *) dst);
            } else {
                ggml_x16_pack_q5_K((const block_q5_K *) rows, nb, (block_q5_K_x16 *) dst);
            }
            break;
        case GGML_TYPE_Q6_K: ggml_x16_pack_q6_K((const block_q6_K *) rows, nb, (block_q6_K_x16 *) dst); break;
        case GGML_TYPE_Q8_0: ggml_x16_pack_q8_0((const block_q8_0 *) rows, nb, (block_q8_0_x16 *) dst); break;
        default: GGML_ABORT("x16: unsupported dst type");
    }
}


// F32 router weights (ffn_gate_inp) stored as F16: halves the 6.3 MB read per layer per socket.
class tensor_traits_router_f16 : public tensor_traits_base {
  public:
    size_t repacked_alloc_size(const struct ggml_tensor * t) const override {
        return (size_t) ggml_nrows(t) * (size_t) t->ne[0] * sizeof(ggml_fp16_t);
    }
    int repack(struct ggml_tensor * t, const void * data, size_t data_size) override {
        const int64_t k = t->ne[0];
        const int64_t nrows = ggml_nrows(t);
        GGML_ASSERT(data_size >= (size_t) nrows * k * sizeof(float));
        ggml_fp16_t * dst = (ggml_fp16_t *) t->data;
        for (int64_t r = 0; r < nrows; r++) {
            ggml_fp32_to_fp16_row((const float *) data + r * k, dst + r * k, k);
        }
        return 0;
    }
    bool work_size(int n_threads, const struct ggml_tensor * op, size_t & size) override {
        GGML_UNUSED(n_threads);
        if (op->op == GGML_OP_MUL_MAT) {
            size = (size_t) ggml_nelements(op->src[1]) * sizeof(ggml_fp16_t);
            return true;
        }
        return false;
    }
    bool compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) override {
        if (op->op != GGML_OP_MUL_MAT) return false;
        const ggml_tensor * src0 = op->src[0];
        const ggml_tensor * src1 = op->src[1];
        ggml_tensor * dst = op;
        const int ith = params->ith, nth = params->nth;
        GGML_ASSERT(src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
        GGML_ASSERT(src0->ne[2] == 1 && src0->ne[3] == 1 && src0->ne[0] == src1->ne[0]);
        const int64_t k = src0->ne[0];
        const int64_t nr0 = src0->ne[1];
        const int64_t n1 = src1->ne[1] * src1->ne[2] * src1->ne[3];
        ggml_fp16_t * wdata = (ggml_fp16_t *) params->wdata;
        GGML_ASSERT(params->wsize >= (size_t) n1 * k * sizeof(ggml_fp16_t));
        for (int64_t i = ith; i < n1; i += nth) {
            const int64_t i1 = i % src1->ne[1];
            const int64_t i2 = (i / src1->ne[1]) % src1->ne[2];
            const int64_t i3 = i / (src1->ne[1] * src1->ne[2]);
            ggml_fp32_to_fp16_row((const float *) ((const char *) src1->data + i1 * src1->nb[1] + i2 * src1->nb[2] + i3 * src1->nb[3]), wdata + i * k, k);
        }
        ggml_barrier(params->threadpool);
        const ggml_fp16_t * w = (const ggml_fp16_t *) src0->data;
        const int64_t r0 = (ith * nr0) / nth;
        const int64_t r1 = ((ith + 1) * nr0) / nth;
        for (int64_t i = 0; i < n1; i++) {
            const int64_t i1 = i % dst->ne[1];
            const int64_t i2 = (i / dst->ne[1]) % dst->ne[2];
            const int64_t i3 = i / (dst->ne[1] * dst->ne[2]);
            float * out = (float *) ((char *) dst->data + i1 * dst->nb[1] + i2 * dst->nb[2] + i3 * dst->nb[3]);
            for (int64_t r = r0; r < r1; r++) {
                ggml_vec_dot_f16((int) k, out + r, 0, (ggml_fp16_t *) (w + r * k), 0, wdata + i * k, 0, 1);
            }
        }
        return true;
    }
};

class tensor_traits_x16 : public tensor_traits_base {
  public:
    explicit tensor_traits_x16(const x16_spec & spec) : sp(spec) {}

    size_t group_bytes(const struct ggml_tensor * t) const { return (size_t) (t->ne[0] / sp.blck) * sp.x16_bytes; }
    size_t row_offset(const struct ggml_tensor * t, int64_t row) const { GGML_ASSERT(row % 16 == 0); return (size_t) (row / 16) * group_bytes(t); }
    size_t expert_offset(const struct ggml_tensor * t, int64_t e) const { return (size_t) e * (size_t) (t->ne[1] / 16) * group_bytes(t); }

    size_t repacked_alloc_size(const struct ggml_tensor * t) const override {
        GGML_ASSERT(t->ne[0] % sp.blck == 0 && t->ne[1] % 16 == 0);
        return (size_t) (ggml_nrows(t) / 16) * group_bytes(t);
    }

    int repack(struct ggml_tensor * t, const void * data, size_t data_size) override {
        const int64_t k = t->ne[0];
        const int64_t nrows = ggml_nrows(t);
        GGML_ASSERT(nrows % 16 == 0 && k % sp.blck == 0);
        const size_t src_row_bytes = ggml_row_size(sp.src_type, k);
        GGML_ASSERT(data_size >= src_row_bytes * (size_t) nrows);
        const size_t dst_row_bytes = ggml_row_size(sp.dst_type, k);
        const size_t gb = group_bytes(t);
        char * dst = (char *) t->data;
        const int64_t n_groups = nrows / 16;
        const bool requant = sp.src_type != sp.dst_type;
        #pragma omp parallel
        {
            std::vector<float> tmpf;
            std::vector<char> tmpq;
            if (requant) { tmpf.resize(k); tmpq.resize(dst_row_bytes * 16); }
            #pragma omp for schedule(static)
            for (int64_t g = 0; g < n_groups; g++) {
                const char * src_rows = (const char *) data + (size_t) g * 16 * src_row_bytes;
                const void * rows = src_rows;
                if (requant) {
                    for (int r = 0; r < 16; r++) {
                        const char * srow = src_rows + (size_t) r * src_row_bytes;
                        switch (sp.src_type) {
                            case GGML_TYPE_Q8_0: dequantize_row_q8_0((const block_q8_0 *) srow, tmpf.data(), k); break;
                            case GGML_TYPE_F16:  ggml_fp16_to_fp32_row((const ggml_fp16_t *) srow, tmpf.data(), k); break;
                            case GGML_TYPE_F32:  memcpy(tmpf.data(), srow, k * sizeof(float)); break;
                            default: GGML_ABORT("x16: unsupported requant source type");
                        }
                        char * drow = tmpq.data() + (size_t) r * dst_row_bytes;
                        switch (sp.dst_type) {
                            case GGML_TYPE_Q4_K: quantize_row_q4_K_ref(tmpf.data(), (block_q4_K *) drow, k); break;
                            case GGML_TYPE_Q5_K: quantize_row_q5_K_ref(tmpf.data(), (block_q5_K *) drow, k); break;
                            case GGML_TYPE_Q6_K: quantize_row_q6_K_ref(tmpf.data(), (block_q6_K *) drow, k); break;
                            case GGML_TYPE_Q8_0: quantize_row_q8_0_ref(tmpf.data(), (block_q8_0 *) drow, k); break;
                            default: GGML_ABORT("x16: unsupported requant dst type");
                        }
                    }
                    rows = tmpq.data();
                }
                ggml_x16_pack_rows(sp, rows, k, dst + (size_t) g * gb);
            }
        }
        static std::atomic<int> announced{0};
        if (announced.fetch_add(1) == 0) {
            GGML_LOG_INFO("x16: repacking '%s' (%s) via %s%s\n", t->name, ggml_type_name(sp.src_type), sp.name, requant ? " (requantized)" : "");
        }
        return 0;
    }

    bool work_size(int n_threads, const struct ggml_tensor * op, size_t & size) override {
        GGML_UNUSED(n_threads);
        switch (op->op) {
            case GGML_OP_MUL_MAT:
                size = ggml_row_size(sp.param_type, ggml_nelements(op->src[1]));
                return true;
            case GGML_OP_MUL_MAT_ID: {
                size = ggml_row_size(sp.param_type, ggml_nelements(op->src[1]));
                size = GGML_PAD(size, sizeof(int64_t));
                const int64_t ne02 = op->src[0]->ne[2];
                const int64_t ne12 = op->src[1]->ne[2];
                size += sizeof(int64_t) * ne02 * (ne12 + 1);
                return true;
            }
            default: break;
        }
        return false;
    }

    bool compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) override {
        switch (op->op) {
            case GGML_OP_MUL_MAT:    forward_mul_mat(params, op);    return true;
            case GGML_OP_MUL_MAT_ID: forward_mul_mat_id(params, op); return true;
            default: break;
        }
        return false;
    }

    // quantize all src1 rows (planes flattened) into wdata; rows split across threads
    void quantize_src1(ggml_compute_params * params, const ggml_tensor * src1, char * wdata, size_t row_bytes) {
        const ggml_from_float_t from_float = ggml_get_type_traits_cpu(sp.param_type)->from_float;
        const int64_t n_rows = src1->ne[1] * src1->ne[2] * src1->ne[3];
        for (int64_t i = params->ith; i < n_rows; i += params->nth) {
            const int64_t i1 = i % src1->ne[1];
            const int64_t i2 = (i / src1->ne[1]) % src1->ne[2];
            const int64_t i3 = i / (src1->ne[1] * src1->ne[2]);
            from_float((const float *) ((const char *) src1->data + i1 * src1->nb[1] + i2 * src1->nb[2] + i3 * src1->nb[3]),
                       wdata + (size_t) i * row_bytes, src1->ne[0]);
        }
    }

    void forward_mul_mat(ggml_compute_params * params, ggml_tensor * op) {
        const ggml_tensor * src0 = op->src[0];
        const ggml_tensor * src1 = op->src[1];
        ggml_tensor * dst = op;
        const int ith = params->ith, nth = params->nth;
        GGML_ASSERT(src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
        GGML_ASSERT(src0->ne[0] == src1->ne[0]);
        GGML_ASSERT(src0->ne[3] == 1 && src1->ne[3] == 1);
        GGML_ASSERT(src1->nb[0] == sizeof(float) && dst->nb[0] == sizeof(float));
        const int64_t k = src0->ne[0];
        const int64_t nr0 = src0->ne[1];      // rows per weight plane
        const int64_t ne02 = src0->ne[2];     // weight planes (e.g. attention heads for k_b/v_b)
        const int64_t ne11 = src1->ne[1];
        const int64_t ne12 = src1->ne[2];
        GGML_ASSERT(ne12 % ne02 == 0);
        const int64_t r2 = ne12 / ne02;       // activation planes per weight plane (broadcast)
        const int64_t n1 = ne11 * ne12;
        char * wdata = (char *) params->wdata;
        const size_t row_bytes = ggml_row_size(sp.param_type, k);
        GGML_ASSERT(params->wsize >= row_bytes * (size_t) n1);
        quantize_src1(params, src1, wdata, row_bytes);
        if (ith == 0) {
            ggml_threadpool_chunk_set(params->threadpool, nth);
        }
        ggml_barrier(params->threadpool);
        // chunk weight rows in units of 16: at least 2 chunks per thread per plane group (small
        // matrices such as the 512-row shared expert must not leave half the socket idle), at most
        // 64 rows so a chunk stays L2-resident when several activation rows reuse it
        static const int64_t chunk_min = [] { const char * v = getenv("GGML_CPU_X16_CHUNK_MIN"); return v ? atoll(v) : 16; }();
        static const int64_t chunk_max = [] { const char * v = getenv("GGML_CPU_X16_CHUNK_MAX"); return v ? atoll(v) : 64; }();
        int64_t chunk = GGML_PAD((nr0 * ne02 + nth * 2 - 1) / (nth * 2), 16);
        if (chunk < chunk_min) chunk = chunk_min;
        if (chunk > chunk_max) chunk = chunk_max;
        if (chunk > nr0) chunk = GGML_PAD(nr0, 16);
        const int64_t nchunk = (nr0 + chunk - 1) / chunk;   // per plane
        const int64_t n_items = nchunk * ne02;
        int cur = ith;
        while (cur < n_items) {
            const int64_t p02 = cur / nchunk;
            const int64_t r0 = (cur % nchunk) * chunk;
            const int64_t r1 = std::min(r0 + chunk, nr0);
            const char * w = (const char *) src0->data + row_offset(src0, p02 * nr0 + r0);
            for (int64_t rr = 0; rr < r2; rr++) {
                const int64_t i12 = p02 * r2 + rr;
                static const bool pair_enabled = []() {
                    const char * value = getenv("GGML_CPU_X16_Q5_BATCH2");
                    return value != nullptr && atoi(value) != 0;
                }();
                static const bool triple_enabled = []() {
                    const char * value = getenv("GGML_CPU_X16_Q5_BYTES_BATCH3");
                    return value != nullptr && atoi(value) != 0;
                }();
                static const bool q8_batch = [] {
                    const char * value = getenv("GGML_CPU_X16_Q8_BATCH");
                    return value && atoi(value) == 1;
                }();
                for (int64_t i11 = 0; i11 < ne11;) {
                    const bool triple = triple_enabled && sp.gemv == ggml_gemv_q5_K_x16_bytes_q8_K && i11 + 2 < ne11;
                    const int nr = q8_batch && sp.dst_type == GGML_TYPE_Q8_0 ? (int) std::min<int64_t>(4, ne11 - i11) : triple ? 3 : pair_enabled && sp.dst_type == GGML_TYPE_Q5_K && i11 + 1 < ne11 ? 2 : 1;
                    const int64_t i = i12 * ne11 + i11;
                    float * out = (float *) ((char *) dst->data + i11 * dst->nb[1] + i12 * dst->nb[2]) + r0;
                    sp.gemv((int) k, out, dst->nb[1] / sizeof(float), w, wdata + (size_t) i * row_bytes, nr, (int) (r1 - r0));
                    i11 += nr;
                }
            }
            cur = ggml_threadpool_chunk_add(params->threadpool, 1);
        }
    }

    void forward_mul_mat_id(ggml_compute_params * params, ggml_tensor * op) {
        const ggml_tensor * src0 = op->src[0];
        const ggml_tensor * src1 = op->src[1];
        const ggml_tensor * ids  = op->src[2];
        ggml_tensor * dst = op;
        const int ith = params->ith, nth = params->nth;
        GGML_ASSERT(src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
        GGML_ASSERT(src0->ne[3] == 1 && src1->ne[3] == 1 && dst->ne[3] == 1);
        const int64_t k = src0->ne[0];
        const int64_t nr0 = src0->ne[1];
        const int n_ids = ids->ne[0];
        const int n_as = src0->ne[2];
        const int64_t ne11 = src1->ne[1], ne12 = src1->ne[2];
        const size_t row_bytes = ggml_row_size(sp.param_type, k);
        const size_t nbw2 = row_bytes * ne11;
        const size_t nbw3 = nbw2 * ne12;
        struct row_map { int32_t i1; int32_t i2; };
        char * wdata = (char *) params->wdata;
        int64_t * counts = (int64_t *) (wdata + GGML_PAD(nbw3, sizeof(int64_t)));
        row_map * rows = (row_map *) (counts + n_as);
        // quantize src1 (all planes)
        {
            const ggml_from_float_t from_float = ggml_get_type_traits_cpu(sp.param_type)->from_float;
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = ith; i11 < ne11; i11 += nth) {
                    from_float((const float *) ((const char *) src1->data + i12 * src1->nb[2] + i11 * src1->nb[1]),
                               wdata + i12 * nbw2 + i11 * row_bytes, k);
                }
            }
        }
        if (ith == 0) {
            memset(counts, 0, n_as * sizeof(int64_t));
            for (int32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
                for (int32_t id = 0; id < n_ids; ++id) {
                    const int32_t e = *(const int32_t *) ((const char *) ids->data + iid1 * ids->nb[1] + id * ids->nb[0]);
                    GGML_ASSERT(e >= 0 && e < n_as);
                    rows[e * ne12 + counts[e]] = { id, iid1 };
                    counts[e] += 1;
                }
            }
            ggml_threadpool_chunk_set(params->threadpool, nth);
        }
        ggml_barrier(params->threadpool);
        constexpr int64_t tile = 64;
        const int64_t n_tiles = (nr0 + tile - 1) / tile;
        int active[512]; int n_active = 0;
        GGML_ASSERT(n_as <= 512);
        for (int e = 0; e < n_as; ++e) {
            if (counts[e] > 0) active[n_active++] = e;
        }
        const int64_t n_items = (int64_t) n_active * n_tiles;
        int64_t item = ith;
        while (item < n_items) {
            const int64_t e = active[item / n_tiles];
            const int64_t t0 = (item % n_tiles) * tile;
            const int64_t tr = std::min(tile, nr0 - t0);
            const int64_t cnt = counts[e];
            const char * w = (const char *) src0->data + expert_offset(src0, e) + row_offset(src0, t0);
            for (int64_t ir = 0; ir < cnt; ir++) {
                const row_map m = rows[e * ne12 + ir];
                const int64_t i11 = m.i1 % ne11;
                const int64_t i12 = m.i2;
                const char * q = wdata + i11 * row_bytes + i12 * nbw2;
                float * out = (float *) ((char *) dst->data + m.i1 * dst->nb[1] + m.i2 * dst->nb[2]) + t0;
                sp.gemv((int) k, out, 0, w, q, 1, (int) tr);
            }
            item = ggml_threadpool_chunk_add(params->threadpool, 1);
        }
    }

    bool compute_forward_mul_mat_id_swiglu(struct ggml_compute_params * params, struct ggml_tensor * gate, struct ggml_tensor * up, struct ggml_tensor * dst) override {
        if (gate->op != GGML_OP_MUL_MAT_ID || up->op != GGML_OP_MUL_MAT_ID || gate->src[0]->extra != this || up->src[0]->extra != this) {
            return false;
        }
        const ggml_tensor * gate_w = gate->src[0];
        const ggml_tensor * up_w = up->src[0];
        const ggml_tensor * src1 = gate->src[1];
        const ggml_tensor * ids = gate->src[2];
        const int ith = params->ith, nth = params->nth;
        const int64_t k = gate_w->ne[0];
        const int64_t n_out = gate_w->ne[1];
        const int64_t n_experts = gate_w->ne[2];
        const int64_t n_used = ids->ne[0];
        const int64_t n_tokens = ids->ne[1];
        const int64_t n_src_rows = src1->ne[1];
        GGML_ASSERT(up->src[1] == src1 && up->src[2] == ids);
        GGML_ASSERT(ggml_are_same_shape(gate_w, up_w) && ggml_are_same_shape(gate, dst));
        GGML_ASSERT(src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && src1->ne[0] == k);
        const size_t row_bytes = ggml_row_size(sp.param_type, k);
        const size_t plane_bytes = row_bytes * n_src_rows;
        const size_t src1_bytes = plane_bytes * n_tokens;
        struct row_map { int32_t i1; int32_t i2; };
        char * wdata = (char *) params->wdata;
        int64_t * counts = (int64_t *) (wdata + GGML_PAD(src1_bytes, sizeof(int64_t)));
        row_map * rows = (row_map *) (counts + n_experts);
        GGML_ASSERT(params->wsize >= GGML_PAD(src1_bytes, sizeof(int64_t)) + n_experts * (n_tokens + 1) * sizeof(int64_t));
        const ggml_from_float_t from_float = ggml_get_type_traits_cpu(sp.param_type)->from_float;
        for (int64_t token = 0; token < n_tokens; ++token) {
            for (int64_t sr = ith; sr < n_src_rows; sr += nth) {
                from_float((const float *) ((const char *) src1->data + token * src1->nb[2] + sr * src1->nb[1]),
                           wdata + token * plane_bytes + sr * row_bytes, k);
            }
        }
        if (ith == 0) {
            memset(counts, 0, n_experts * sizeof(int64_t));
            for (int64_t token = 0; token < n_tokens; ++token) {
                for (int64_t u = 0; u < n_used; ++u) {
                    const int32_t e = *(const int32_t *) ((const char *) ids->data + token * ids->nb[1] + u * ids->nb[0]);
                    GGML_ASSERT(e >= 0 && e < n_experts);
                    rows[e * n_tokens + counts[e]] = { (int32_t) u, (int32_t) token };
                    counts[e] += 1;
                }
            }
            ggml_threadpool_chunk_set(params->threadpool, nth);
        }
        ggml_barrier(params->threadpool);
        // Dynamic work items (expert, 64-row tile): threads steal tiles, so a core that is
        // shared with another process does not stall the whole socket at the next barrier.
        constexpr int64_t tile = 64;
        const int64_t n_tiles = (n_out + tile - 1) / tile;
        int active[512]; int n_active = 0;
        GGML_ASSERT(n_experts <= 512);
        for (int64_t e = 0; e < n_experts; ++e) {
            if (counts[e] > 0) active[n_active++] = (int) e;
        }
        const int64_t n_items = (int64_t) n_active * n_tiles;
        static const bool dual_env = [] { const char * v = getenv("GGML_CPU_X16_DUAL"); return v && atoi(v) != 0; }();
        const bool dual_q4 = dual_env && sp.dst_type == GGML_TYPE_Q4_K && gate_w->type == up_w->type;
        float gate_tmp[tile]; float up_tmp[tile];
        int64_t item = ith;
        while (item < n_items) {
            const int64_t e = active[item / n_tiles];
            const int64_t t0 = (item % n_tiles) * tile;
            const int64_t tr = std::min(tile, n_out - t0);
            const int64_t cnt = counts[e];
            const char * gw = (const char *) gate_w->data + expert_offset(gate_w, e) + row_offset(gate_w, t0);
            const char * uw = (const char *) up_w->data + expert_offset(up_w, e) + row_offset(up_w, t0);
            for (int64_t ir = 0; ir < cnt; ir++) {
                const row_map m = rows[e * n_tokens + ir];
                const char * q = wdata + m.i2 * plane_bytes + (m.i1 % n_src_rows) * row_bytes;
                float * out = (float *) ((char *) dst->data + m.i2 * dst->nb[2] + m.i1 * dst->nb[1]);
                if (dual_q4) {
                    ggml_gemv2_q4_K_x16_q8_K((int) k, gate_tmp, up_tmp, gw, uw, q, (int) tr);
                } else {
                    sp.gemv((int) k, gate_tmp, 0, gw, q, 1, (int) tr);
                    sp.gemv((int) k, up_tmp,   0, uw, q, 1, (int) tr);
                }
                ggml_vec_swiglu_f32(tr, out + t0, gate_tmp, up_tmp);
            }
            item = ggml_threadpool_chunk_add(params->threadpool, 1);
        }
        return true;
    }

    bool compute_forward_mul_mat_swiglu(struct ggml_compute_params * params, struct ggml_tensor * gate, struct ggml_tensor * up, struct ggml_tensor * dst) override {
        if (gate->op != GGML_OP_MUL_MAT || up->op != GGML_OP_MUL_MAT || gate->src[0]->extra != this || up->src[0]->extra != this) {
            return false;
        }
        const ggml_tensor * gate_w = gate->src[0];
        const ggml_tensor * up_w = up->src[0];
        const ggml_tensor * src1 = gate->src[1];
        if (src1->ne[2] != 1 || src1->ne[3] != 1 || up->src[1] != src1) {
            return false;
        }
        const int ith = params->ith, nth = params->nth;
        const int64_t k = gate_w->ne[0];
        const int64_t nr0 = gate_w->ne[1];
        const int64_t n1 = src1->ne[1];
        GGML_ASSERT(ggml_are_same_shape(gate_w, up_w) && ggml_are_same_shape(gate, dst));
        GGML_ASSERT(src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && src1->ne[0] == k);
        char * wdata = (char *) params->wdata;
        const size_t row_bytes = ggml_row_size(sp.param_type, k);
        GGML_ASSERT(params->wsize >= row_bytes * (size_t) n1);
        quantize_src1(params, src1, wdata, row_bytes);
        if (ith == 0) {
            ggml_threadpool_chunk_set(params->threadpool, nth);
        }
        ggml_barrier(params->threadpool);
        static const int64_t chunk_min = [] { const char * v = getenv("GGML_CPU_X16_CHUNK_MIN"); return v ? atoll(v) : 16; }();
        static const int64_t chunk_max = [] { const char * v = getenv("GGML_CPU_X16_CHUNK_MAX"); return v ? atoll(v) : 64; }();
        int64_t chunk = GGML_PAD((nr0 + nth * 2 - 1) / (nth * 2), 16);
        if (chunk < chunk_min) chunk = chunk_min;
        if (chunk > chunk_max) chunk = chunk_max;
        const int64_t nchunk = (nr0 + chunk - 1) / chunk;
        static const bool q8_batch = [] {
            const char * value = getenv("GGML_CPU_X16_Q8_BATCH");
            return value && atoi(value) == 1;
        }();
        const int max_batch = q8_batch && sp.dst_type == GGML_TYPE_Q8_0 ? 4 : 1;
        std::vector<float> gtmp(chunk * max_batch), utmp(chunk * max_batch);
        int cur = ith;
        while (cur < nchunk) {
            const int64_t r0 = cur * chunk;
            const int64_t r1 = std::min(r0 + chunk, nr0);
            const char * gw = (const char *) gate_w->data + row_offset(gate_w, r0);
            const char * uw = (const char *) up_w->data + row_offset(up_w, r0);
            for (int64_t i = 0; i < n1;) {
                const int nr = (int) std::min<int64_t>(max_batch, n1 - i);
                sp.gemv((int) k, gtmp.data(), chunk, gw, wdata + (size_t) i * row_bytes, nr, (int) (r1 - r0));
                sp.gemv((int) k, utmp.data(), chunk, uw, wdata + (size_t) i * row_bytes, nr, (int) (r1 - r0));
                for (int t = 0; t < nr; ++t) {
                    float * out = (float *) ((char *) dst->data + (i + t) * dst->nb[1]) + r0;
                    ggml_vec_swiglu_f32(r1 - r0, out, gtmp.data() + t * chunk, utmp.data() + t * chunk);
                }
                i += nr;
            }
            cur = ggml_threadpool_chunk_add(params->threadpool, 1);
        }
        return true;
    }

  private:
    x16_spec sp;
};

}  // namespace ggml::cpu::repack

// selection helpers -------------------------------------------------------------------
static ggml_type ggml_x16_parse_type(const char * v) {
    if (v == nullptr) return GGML_TYPE_COUNT;
    if (strcmp(v, "q4_K") == 0 || strcmp(v, "q4_k") == 0) return GGML_TYPE_Q4_K;
    if (strcmp(v, "q5_K") == 0 || strcmp(v, "q5_k") == 0) return GGML_TYPE_Q5_K;
    if (strcmp(v, "q6_K") == 0 || strcmp(v, "q6_k") == 0) return GGML_TYPE_Q6_K;
    if (strcmp(v, "q8_0") == 0) return GGML_TYPE_Q8_0;
    return GGML_TYPE_COUNT;
}

static const ggml::cpu::tensor_traits * ggml_x16_select(const struct ggml_tensor * cur) {
    using namespace ggml::cpu::repack;
    static const x16_spec spec_q4_K   = { GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, QK_K,  sizeof(block_q4_K_x16), GGML_TYPE_Q8_K, ggml_gemv_q4_K_x16_q8_K, "x16 q4_K" };
    static const x16_spec spec_q5_K   = { GGML_TYPE_Q5_K, GGML_TYPE_Q5_K, QK_K,  sizeof(block_q5_K_x16), GGML_TYPE_Q8_K, ggml_gemv_q5_K_x16_q8_K, "x16 q5_K" };
    static const x16_spec spec_q6_K   = { GGML_TYPE_Q6_K, GGML_TYPE_Q6_K, QK_K,  sizeof(block_q6_K_x16), GGML_TYPE_Q8_K, ggml_gemv_q6_K_x16_q8_K, "x16 q6_K" };
    static const x16_spec spec_q8_0   = { GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, QK8_0, sizeof(block_q8_0_x16), GGML_TYPE_Q8_0, ggml_gemv_q8_0_x16_q8_0, "x16 q8_0" };
    static const x16_spec spec_80_q6  = { GGML_TYPE_Q8_0, GGML_TYPE_Q6_K, QK_K,  sizeof(block_q6_K_x16), GGML_TYPE_Q8_K, ggml_gemv_q6_K_x16_q8_K, "x16 q8_0->q6_K" };
    static const x16_spec spec_80_q5  = { GGML_TYPE_Q8_0, GGML_TYPE_Q5_K, QK_K,  sizeof(block_q5_K_x16), GGML_TYPE_Q8_K, ggml_gemv_q5_K_x16_q8_K, "x16 q8_0->q5_K" };
    static const x16_spec spec_80_q4  = { GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, QK_K,  sizeof(block_q4_K_x16), GGML_TYPE_Q8_K, ggml_gemv_q4_K_x16_q8_K, "x16 q8_0->q4_K" };
    static const tensor_traits_x16 t_q4_K(spec_q4_K), t_q5_K(spec_q5_K), t_q6_K(spec_q6_K), t_q8_0(spec_q8_0);
    static const x16_spec spec_q5_bytes = { GGML_TYPE_Q5_K, GGML_TYPE_Q5_K, QK_K, sizeof(block_q5_K_x16_bytes), GGML_TYPE_Q8_K, ggml_gemv_q5_K_x16_bytes_q8_K, "x16 q5_K bytes" };
    static const tensor_traits_x16 t_q5_bytes(spec_q5_bytes);
    static const bool q5_bytes = []() {
        const char * value = getenv("GGML_CPU_X16_Q5_BYTES");
        return value != nullptr && atoi(value) != 0;
    }();
    static const tensor_traits_x16 t_80_q6(spec_80_q6), t_80_q5(spec_80_q5), t_80_q4(spec_80_q4);

    static const bool have_vnni = ggml_cpu_has_avx512() && ggml_cpu_has_avx512_vnni();
    static const bool x16_q4_K = [] { const char * v = getenv("GGML_CPU_X16_Q4_K"); return v && atoi(v) != 0; }();
    static const bool x16_q5_K = [] { const char * v = getenv("GGML_CPU_X16_Q5_K"); return v && atoi(v) != 0; }();
    static const bool x16_q6_K = [] { const char * v = getenv("GGML_CPU_X16_Q6_K"); return v && atoi(v) != 0; }();
    static const bool x16_q8_0 = [] { const char * v = getenv("GGML_CPU_X16_Q8_0"); return v && atoi(v) != 0; }();
    static const ggml_type attn_rq   = ggml_x16_parse_type(getenv("GGML_CPU_ATTN_REQUANT"));
    static const ggml_type shexp_rq  = ggml_x16_parse_type(getenv("GGML_CPU_SHEXP_REQUANT"));
    static const ggml_type output_rq = ggml_x16_parse_type(getenv("GGML_CPU_OUTPUT_REQUANT"));
    static const ggml_type dense_rq  = ggml_x16_parse_type(getenv("GGML_CPU_DENSE_FFN_REQUANT"));

    static const tensor_traits_router_f16 t_router_f16;
    static const bool router_f16 = [] { const char * v = getenv("GGML_CPU_ROUTER_F16"); return v && atoi(v) != 0; }();
    if (router_f16 && cur->type == GGML_TYPE_F32 && ggml_n_dims(cur) == 2 && strstr(cur->name, "ffn_gate_inp.weight") != nullptr) {
        return &t_router_f16;
    }
    static const bool x16_attn3d = [] { const char * v = getenv("GGML_CPU_X16_ATTN3D"); return v && atoi(v) != 0; }();
    if (!have_vnni || cur->ne[1] % 16 != 0) {
        return nullptr;
    }
    const int nd = ggml_n_dims(cur);
    if (x16_attn3d && nd == 3 && cur->type == GGML_TYPE_Q8_0 && strstr(cur->name, ".attn_") != nullptr && cur->ne[0] % QK8_0 == 0) {
        return &t_q8_0;
    }
    const bool is_attn  = nd == 2 && strstr(cur->name, ".attn_") != nullptr;
    const bool is_shexp = nd == 2 && strstr(cur->name, "_shexp.weight") != nullptr;
    const bool is_out   = nd == 2 && strcmp(cur->name, "output.weight") == 0;
    const bool is_dense = nd == 2 && strncmp(cur->name, "blk.", 4) == 0 && strstr(cur->name, "_exps") == nullptr && strstr(cur->name, "_shexp") == nullptr &&
        (strstr(cur->name, ".ffn_gate.weight") || strstr(cur->name, ".ffn_up.weight") || strstr(cur->name, ".ffn_down.weight"));

    auto requant_for = [&](ggml_type target) -> const ggml::cpu::tensor_traits * {
        if (cur->ne[0] % QK_K != 0) return nullptr;
        switch (target) {
            case GGML_TYPE_Q6_K: return &t_80_q6;
            case GGML_TYPE_Q5_K: return &t_80_q5;
            case GGML_TYPE_Q4_K: return &t_80_q4;
            case GGML_TYPE_Q8_0: return &t_q8_0;
            default: return nullptr;
        }
    };
    if (cur->type == GGML_TYPE_Q8_0 && cur->ne[0] % QK8_0 == 0) {
        if (is_attn  && attn_rq   != GGML_TYPE_COUNT) { if (auto * t = requant_for(attn_rq))   return t; }
        if (is_shexp && shexp_rq  != GGML_TYPE_COUNT) { if (auto * t = requant_for(shexp_rq))  return t; }
        if (is_out   && output_rq != GGML_TYPE_COUNT) { if (auto * t = requant_for(output_rq)) return t; }
        if (is_dense && dense_rq  != GGML_TYPE_COUNT) { if (auto * t = requant_for(dense_rq))  return t; }
        if (x16_q8_0 && (is_attn || is_shexp || is_out || is_dense)) return &t_q8_0;
    }
    if (cur->type == GGML_TYPE_Q4_K && x16_q4_K && cur->ne[0] % QK_K == 0) return &t_q4_K;
    if (cur->type == GGML_TYPE_Q5_K && x16_q5_K && cur->ne[0] % QK_K == 0) return q5_bytes ? &t_q5_bytes : &t_q5_K;
    if (cur->type == GGML_TYPE_Q6_K && x16_q6_K && cur->ne[0] % QK_K == 0) return &t_q6_K;
    return nullptr;
}

static const ggml::cpu::tensor_traits * ggml_repack_get_optimal_repack_type(const struct ggml_tensor * cur) {
    static const bool iq_r16_enabled = []() {
        const char * value = getenv("GGML_CPU_IQ_R16_REPACK");
        return value != nullptr && strcmp(value, "1") == 0;
    }();

    // instance for Q4
    static const ggml::cpu::repack::tensor_traits<block_q4_0, 4, 4, GGML_TYPE_Q8_0> q4_0_4x4_q8_0;
    static const ggml::cpu::repack::tensor_traits<block_q4_0, 8, 4, GGML_TYPE_Q8_0> q4_0_4x8_q8_0;
    static const ggml::cpu::repack::tensor_traits<block_q4_0, 8, 8, GGML_TYPE_Q8_0> q4_0_8x8_q8_0;

    // instance for Q4_K
    static const ggml::cpu::repack::tensor_traits<block_q4_K, 4, 8, GGML_TYPE_Q8_K> q4_K_8x4_q8_K;
    static const ggml::cpu::repack::tensor_traits<block_q4_K, 8, 8, GGML_TYPE_Q8_K> q4_K_8x8_q8_K;
    static const ggml::cpu::repack::tensor_traits<block_q4_K_r8_source, 8, 8, GGML_TYPE_Q8_K> q4_K_r8_q8_K;

    // instance for Q5_K
    static const ggml::cpu::repack::tensor_traits<block_q5_K, 4, 8, GGML_TYPE_Q8_K> q5_K_8x4_q8_K;
    static const ggml::cpu::repack::tensor_traits<block_q5_K, 8, 8, GGML_TYPE_Q8_K> q5_K_8x8_q8_K;
    static const ggml::cpu::repack::tensor_traits<block_q5_K_r8_source, 8, 8, GGML_TYPE_Q8_K> q5_K_r8_q8_K;

    // instance for Q6_K
    static const ggml::cpu::repack::tensor_traits<block_q6_K, 4, 8, GGML_TYPE_Q8_K> q6_K_8x4_q8_K;
    static const ggml::cpu::repack::tensor_traits<block_q6_K, 8, 8, GGML_TYPE_Q8_K> q6_K_8x8_q8_K;

    // instance for Q2
    static const ggml::cpu::repack::tensor_traits<block_q2_K, 8, 8, GGML_TYPE_Q8_K> q2_K_8x8_q8_K;

    // Exact compact in-memory IQ2_XS.  This is opt-in because it increases the
    // tensor footprint from 2.3125 to 4.3125 bits per weight.  The extra bit
    // replaces three AVX-512 mask expansions with a packed-nibble decode.
    static const ggml::cpu::repack::tensor_traits<block_iq2_xs, 8, 8, GGML_TYPE_Q8_K> iq2_xs_r8_q8_K;
    static const ggml::cpu::repack::tensor_traits<block_iq2_xs, 8, 16, GGML_TYPE_Q8_K> iq2_xs_r16_q8_K;

    // Exact compact in-memory IQ3_XXS.  This is independently opt-in because
    // it increases the tensor footprint from 3.0625 to 4.1875 bits per weight.
    static const ggml::cpu::repack::tensor_traits<block_iq3_xxs, 8, 8, GGML_TYPE_Q8_K> iq3_xxs_r8_q8_K;
    static const ggml::cpu::repack::tensor_traits<block_iq3_xxs, 8, 16, GGML_TYPE_Q8_K> iq3_xxs_r16_q8_K;

    // instance for IQ4
    static const ggml::cpu::repack::tensor_traits<block_iq4_nl, 4, 4, GGML_TYPE_Q8_0> iq4_nl_4x4_q8_0;
    static const ggml::cpu::repack::tensor_traits<block_iq4_nl, 8, 8, GGML_TYPE_Q8_0> iq4_nl_8x8_q8_0;

    // instance for MXFP4
    static const ggml::cpu::repack::tensor_traits<block_mxfp4, 4, 4, GGML_TYPE_Q8_0> mxfp4_4x4_q8_0;
    static const ggml::cpu::repack::tensor_traits<block_mxfp4, 8, 8, GGML_TYPE_Q8_0> mxfp4_8x8_q8_0;

    // instance for Q8_0
    static const ggml::cpu::repack::tensor_traits<block_q8_0, 4, 4, GGML_TYPE_Q8_0> q8_0_4x4_q8_0;
    static const ggml::cpu::repack::tensor_traits<block_q8_0, 8, 4, GGML_TYPE_Q8_0> q8_0_4x8_q8_0;
    static const ggml::cpu::repack::tensor_traits<block_q8_0, 8, 8, GGML_TYPE_Q8_0> q8_0_8x8_q8_0;

    // instances for RISC-V
    //
    // These implement outer-product style matrix multiplication kernels with
    // an interleave of 1.
#if defined __riscv_zvfh
    static const ggml::cpu::repack::tensor_traits<block_q4_0, 1, 16, GGML_TYPE_Q8_0> q4_0_16x1_q8_0;
    static const ggml::cpu::repack::tensor_traits<block_q4_K, 1, 16, GGML_TYPE_Q8_K> q4_K_16x1_q8_K;
    static const ggml::cpu::repack::tensor_traits<block_iq4_nl, 1, 16, GGML_TYPE_Q8_0> iq4_nl_16x1_q8_0;
    static const ggml::cpu::repack::tensor_traits<block_q8_0, 1, 16, GGML_TYPE_Q8_0> q8_0_16x1_q8_0;
    static const ggml::cpu::repack::tensor_traits<block_q2_K, 1, 16, GGML_TYPE_Q8_K> q2_K_16x1_q8_K;
#endif

    if (const ggml::cpu::tensor_traits * x16 = ggml_x16_select(cur)) {
        return x16;
    }
    if (cur->type == GGML_TYPE_Q4_0) {
        if (ggml_cpu_has_avx2() || (ggml_cpu_has_sve() && ggml_cpu_has_matmul_int8() && ggml_cpu_get_sve_cnt() == QK8_0)) {
            if (cur->ne[1] % 8 == 0) {
                return &q4_0_8x8_q8_0;
            }
        }
        if (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
            if (cur->ne[1] % 4 == 0) {
                return &q4_0_4x8_q8_0;
            }
        }
        if (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            if (cur->ne[1] % 4 == 0) {
                return &q4_0_4x4_q8_0;
            }
        }
        if (ggml_cpu_has_riscv_v()) {
            #if defined __riscv_zvfh
            switch (__riscv_vlenb() * 8) {
                case 128:  { break; } // TODO
                case 256:  { if (cur->ne[1] % 16 == 0) { return &q4_0_16x1_q8_0; } break; }
                case 512:  { break; } // TODO
                case 1024: { break; } // TODO
                default:   { return nullptr; }
            }
            #endif
        }
    } else if (cur->type == GGML_TYPE_Q4_K) {
        static const bool x86_expanded_enabled = []() {
            const char * value = getenv("GGML_CPU_Q4_K_REPACK");
            return value != nullptr && strcmp(value, "1") == 0;
        }();
        if (x86_expanded_enabled && ggml_cpu_has_avx512() && ggml_cpu_has_avx512_vnni() &&
                cur->ne[0] % QK_K == 0 && cur->ne[1] % 8 == 0) {
            return &q4_K_r8_q8_K;
        }
        if (ggml_cpu_has_avx2()) {
            if (cur->ne[1] % 8 == 0) {
                return &q4_K_8x8_q8_K;
            }
        }
        if (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
            if (cur->ne[1] % 8 == 0) {
                return &q4_K_8x8_q8_K;
            }
        }
        if (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            if (cur->ne[1] % 8 == 0) {
                return &q4_K_8x4_q8_K;
            }
        }
        if (ggml_cpu_has_riscv_v()) {
            #if defined __riscv_zvfh
            switch (__riscv_vlenb() * 8) {
                case 128:  { break; } // TODO
                case 256:  { if (cur->ne[1] % 16 == 0) { return &q4_K_16x1_q8_K; } break; }
                case 512:  { break; } // TODO
                case 1024: { break; } // TODO
                default:   { return nullptr; }
            }
            #endif
        }
    } else if (cur->type == GGML_TYPE_Q2_K) {
        if (ggml_cpu_has_avx512()) {
            if (cur->ne[1] % 8 == 0) {
                return &q2_K_8x8_q8_K;
            }
        }
        if (ggml_cpu_has_riscv_v()) {
            #if defined __riscv_zvfh
            switch (__riscv_vlenb() * 8) {
                case 128:  { break; } // TODO
                case 256:  { if (cur->ne[1] % 16 == 0) { return &q2_K_16x1_q8_K; } break; }
                case 512:  { break; } // TODO
                case 1024: { break; } // TODO
                default:   { return nullptr; }
            }
            #endif
        }
    } else if (cur->type == GGML_TYPE_IQ2_XS) {
        static const bool enabled = []() {
            const char * value = getenv("GGML_CPU_IQ2_XS_REPACK");
            return value != nullptr && strcmp(value, "1") == 0;
        }();
        if (enabled && ggml_cpu_has_avx512() && ggml_cpu_has_avx512_vnni() &&
                cur->ne[0] % QK_K == 0 && cur->ne[1] % 8 == 0) {
            if (iq_r16_enabled && cur->ne[1] % 16 == 0) {
                return &iq2_xs_r16_q8_K;
            }
            return &iq2_xs_r8_q8_K;
        }
    } else if (cur->type == GGML_TYPE_IQ3_XXS) {
        static const bool enabled = []() {
            const char * value = getenv("GGML_CPU_IQ3_XXS_REPACK");
            return value != nullptr && strcmp(value, "1") == 0;
        }();
        if (enabled && ggml_cpu_has_avx512() && ggml_cpu_has_avx512_vnni() &&
                cur->ne[0] % QK_K == 0 && cur->ne[1] % 8 == 0) {
            if (iq_r16_enabled && cur->ne[1] % 16 == 0) {
                return &iq3_xxs_r16_q8_K;
            }
            return &iq3_xxs_r8_q8_K;
        }
    } else if (cur->type == GGML_TYPE_Q5_K) {
        static const bool x86_expanded_enabled = []() {
            const char * value = getenv("GGML_CPU_Q5_K_REPACK");
            return value != nullptr && strcmp(value, "1") == 0;
        }();
        static const bool x86_expanded_moe_down_enabled = []() {
            const char * value = getenv("GGML_CPU_Q5_K_REPACK_MOE_DOWN");
            return value == nullptr || strcmp(value, "0") != 0;
        }();
        const bool is_moe_down = strstr(cur->name, "ffn_down_exps") != nullptr;
        if (x86_expanded_enabled && (x86_expanded_moe_down_enabled || !is_moe_down) &&
                ggml_cpu_has_avx512() && ggml_cpu_has_avx512_vnni() &&
                cur->ne[0] % QK_K == 0 && cur->ne[1] % 8 == 0) {
            return &q5_K_r8_q8_K;
        }
        if (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
            if (cur->ne[1] % 8 == 0) {
                return &q5_K_8x8_q8_K;
            }
        }
        if (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            if (cur->ne[1] % 8 == 0) {
                return &q5_K_8x4_q8_K;
            }
        }
    } else if (cur->type == GGML_TYPE_Q6_K) {
        if (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
            if (cur->ne[1] % 8 == 0) {
                return &q6_K_8x8_q8_K;
            }
        }
        if (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            if (cur->ne[1] % 8 == 0) {
                return &q6_K_8x4_q8_K;
            }
        }
    } else if (cur->type == GGML_TYPE_IQ4_NL) {
        if (ggml_cpu_has_avx2()) {
            if (cur->ne[1] % 8 == 0) {
                return &iq4_nl_8x8_q8_0;
            }
        }
        if (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            if (cur->ne[1] % 4 == 0) {
                return &iq4_nl_4x4_q8_0;
            }
        }
        if (ggml_cpu_has_riscv_v()) {
            #if defined __riscv_zvfh
            switch (__riscv_vlenb() * 8) {
                case 128:  { break; } // TODO
                case 256:  { if (cur->ne[1] % 16 == 0) { return &iq4_nl_16x1_q8_0; } break; }
                case 512:  { break; } // TODO
                case 1024: { break; } // TODO
                default:   { return nullptr; }
            }
            #endif
        }
    } else if (cur->type == GGML_TYPE_MXFP4) {
        if (ggml_cpu_has_avx2()) {
            if (cur->ne[1] % 8 == 0) {
                return &mxfp4_8x8_q8_0;
            }
        }
        if (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            if (cur->ne[1] % 4 == 0) {
                return &mxfp4_4x4_q8_0;
            }
        }
    } else if (cur->type == GGML_TYPE_Q8_0) {
        static const bool x86_vnni_enabled = []() {
            const char * value = getenv("GGML_CPU_Q8_0_REPACK");
            return value != nullptr && strcmp(value, "1") == 0;
        }();
        static const bool x86_vnni_force = []() {
            const char * value = getenv("GGML_CPU_Q8_0_REPACK_FORCE");
            return value != nullptr && strcmp(value, "1") == 0;
        }();
        static const bool x86_vnni_ffn_enabled = []() {
            const char * value = getenv("GGML_CPU_Q8_0_REPACK_FFN");
            return value != nullptr && strcmp(value, "1") == 0;
        }();
        static const bool x86_vnni_output_enabled = []() {
            const char * value = getenv("GGML_CPU_Q8_0_REPACK_OUTPUT");
            return value != nullptr && strcmp(value, "1") == 0;
        }();
        // This layout is a decode-time optimization for dense attention
        // matrices, with separately gated FFN and output-head experiments.
        // Keeping the predicate narrow also prevents small SSM tensors and
        // embeddings from moving onto an extra-buffer path whose graph
        // operations do not consume the repacked representation.
        const bool eligible_matrix = ggml_n_dims(cur) == 2 && (
            strstr(cur->name, ".attn_") != nullptr ||
            (x86_vnni_ffn_enabled && strstr(cur->name, ".ffn_") != nullptr) ||
            (x86_vnni_output_enabled && strcmp(cur->name, "output.weight") == 0));
        if (x86_vnni_enabled && ggml_cpu_has_avx512() && ggml_cpu_has_avx512_vnni() &&
                (x86_vnni_force || eligible_matrix) &&
                cur->ne[0] % QK8_0 == 0 && cur->ne[1] % 8 == 0) {
            return &q8_0_8x8_q8_0;
        }
        if (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
            if (cur->ne[1] % 4 == 0) {
                return &q8_0_4x8_q8_0;
            }
        }
        if (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            if (cur->ne[1] % 4 == 0) {
                return &q8_0_4x4_q8_0;
            }
        }
        if (ggml_cpu_has_riscv_v()) {
            #if defined __riscv_zvfh
            switch (__riscv_vlenb() * 8) {
                case 128:  { break; } // TODO
                case 256:  { if (cur->ne[1] % 16 == 0) { return &q8_0_16x1_q8_0; } break; }
                case 512:  { break; } // TODO
                case 1024: { break; } // TODO
                default:   { return nullptr; }
            }
            #endif
        }
    }

    return nullptr;
}

static enum ggml_status ggml_backend_cpu_repack_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    tensor->extra = (void *) const_cast<ggml::cpu::tensor_traits *>(ggml_repack_get_optimal_repack_type(tensor));

    GGML_UNUSED(buffer);
    return GGML_STATUS_SUCCESS;
}

static bool ggml_backend_cpu_repack_keep_original() {
    static const bool enabled = []() {
        const char * value = getenv("GGML_CPU_REPACK_KEEP_ORIGINAL");
        return value != nullptr && strcmp(value, "1") == 0;
    }();
    return enabled;
}

static size_t ggml_backend_cpu_repack_tensor_data_size(const struct ggml_tensor * tensor) {
    const auto * traits = (const ggml::cpu::repack::tensor_traits_base *)
        ggml_repack_get_optimal_repack_type(tensor);
    return traits == nullptr ? ggml_nbytes(tensor) : traits->repacked_alloc_size(tensor);
}

static void ggml_backend_cpu_repack_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
                                                       const void * data, size_t offset, size_t size) {
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    auto tensor_traits = (ggml::cpu::repack::tensor_traits_base *) tensor->extra;
    if (ggml_backend_cpu_repack_keep_original()) {
        char * original = (char *) tensor->data + ggml_backend_cpu_repack_tensor_data_size(tensor);
        memcpy(original, data, size);
    }
    if (tensor_traits == nullptr) {
        // Extra buffer selection can be forced by a caller even when this
        // tensor has no optimized layout.  Its allocation size is already the
        // ordinary tensor size, so preserve the source layout instead of
        // dereferencing a missing repack trait.
        memcpy(tensor->data, data, size);
        return;
    }
    auto OK            = tensor_traits->repack(tensor, data, size);

    GGML_ASSERT(OK == 0);
    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_repack_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor,
                                                       void * data, size_t offset, size_t size) {
    GGML_ASSERT(ggml_backend_cpu_repack_keep_original());
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor));

    const char * original = (const char *) tensor->data + ggml_backend_cpu_repack_tensor_data_size(tensor);
    memcpy(data, original + offset, size);
    GGML_UNUSED(buffer);
}

static const char * ggml_backend_cpu_repack_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    auto * ctx = (ggml::cpu::repack::extra_buffer_type *) buft->context;
    return ctx->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_cpu_repack_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    auto * ctx = (ggml::cpu::repack::extra_buffer_type *) buft->context;
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(ctx->base_buft, size);

    if (buffer == nullptr) {
        return nullptr;
    }

    buffer->buft              = buft;
    buffer->iface.init_tensor = ggml_backend_cpu_repack_buffer_init_tensor;
    buffer->iface.set_tensor  = ggml_backend_cpu_repack_buffer_set_tensor;
    buffer->iface.get_tensor  = ggml_backend_cpu_repack_buffer_get_tensor;
    buffer->iface.cpy_tensor  = nullptr;
    return buffer;
}

static size_t ggml_backend_cpu_repack_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    auto * ctx = (ggml::cpu::repack::extra_buffer_type *) buft->context;
    return ggml_backend_buft_get_alignment(ctx->base_buft);
}

static size_t ggml_backend_cpu_repack_buffer_type_get_alloc_size(
        ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    GGML_UNUSED(buft);
    const size_t data_size = ggml_backend_cpu_repack_tensor_data_size(tensor);
    return data_size + (ggml_backend_cpu_repack_keep_original() ? ggml_nbytes(tensor) : 0);
}

namespace ggml::cpu::repack {
bool extra_buffer_type::supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    if (op->op == GGML_OP_MUL_MAT &&
        op->src[0]->buffer &&
        ggml_n_dims(op->src[0]) == 2 &&
        ggml_backend_cpu_is_repack_buffer_type(op->src[0]->buffer->buft) &&
        ggml_backend_buft_get_device(op->src[0]->buffer->buft) == dev &&
        ggml_repack_get_optimal_repack_type(op->src[0])) {
        if (op->src[1]->buffer && !ggml_backend_buft_is_host(op->src[1]->buffer->buft)) {
            ggml_backend_dev_t src1_dev = ggml_backend_buft_get_device(op->src[1]->buffer->buft);
            if (src1_dev == nullptr || ggml_backend_dev_type(src1_dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
                return false;
            }
        }
        if (op->src[1]->type == GGML_TYPE_F32) {
            return true;
        }
        //if (op->src[1]->type == GGML_TYPE_Q8_0) {
        //    return true;
        //}
        // may be possible if Q8_0 packed...
    } else if (op->op == GGML_OP_MUL_MAT_ID &&
               op->src[0]->buffer &&
               ggml_n_dims(op->src[0]) == 3 &&
               ggml_backend_cpu_is_repack_buffer_type(op->src[0]->buffer->buft) &&
               ggml_backend_buft_get_device(op->src[0]->buffer->buft) == dev &&
               ggml_repack_get_optimal_repack_type(op->src[0])) {
        if (op->src[1]->buffer && !ggml_backend_buft_is_host(op->src[1]->buffer->buft)) {
            ggml_backend_dev_t src1_dev = ggml_backend_buft_get_device(op->src[1]->buffer->buft);
            if (src1_dev == nullptr || ggml_backend_dev_type(src1_dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
                return false;
            }
        }
        if (op->src[1]->type == GGML_TYPE_F32) {
            return true;
        }
        //if (op->src[1]->type == GGML_TYPE_Q8_0) {
        //    return true;
        //}
    }
    return false;
}

ggml::cpu::tensor_traits * extra_buffer_type::get_tensor_traits(const struct ggml_tensor * op) {
    if (op->op == GGML_OP_MUL_MAT || op->op == GGML_OP_MUL_MAT_ID) {
        if (op->src[0]->buffer && ggml_backend_cpu_is_repack_buffer_type(op->src[0]->buffer->buft)) {
            return (ggml::cpu::tensor_traits *) op->src[0]->extra;
        }
    }
    return nullptr;
}
}  // namespace ggml::cpu::repack

namespace {
struct repack_buffer_type_bundle {
    std::unique_ptr<ggml::cpu::repack::extra_buffer_type> context;
    std::unique_ptr<ggml_backend_buffer_type> buft;
};

std::mutex & repack_buffer_types_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::vector<std::unique_ptr<repack_buffer_type_bundle>> & repack_buffer_types() {
    static std::vector<std::unique_ptr<repack_buffer_type_bundle>> bufts;
    return bufts;
}
}  // namespace

ggml_backend_buffer_type_t ggml_backend_cpu_repack_buffer_type_from(
        ggml_backend_buffer_type_t base_buft,
        ggml_backend_dev_t device,
        const char * name) {
    GGML_ASSERT(base_buft != nullptr);
    GGML_ASSERT(device != nullptr);
    GGML_ASSERT(name != nullptr);

    std::lock_guard<std::mutex> lock(repack_buffer_types_mutex());
    for (const auto & bundle : repack_buffer_types()) {
        if (bundle->context->base_buft == base_buft && ggml_backend_buft_get_device(bundle->buft.get()) == device) {
            return bundle->buft.get();
        }
    }

    auto bundle = std::make_unique<repack_buffer_type_bundle>();
    bundle->context = std::make_unique<ggml::cpu::repack::extra_buffer_type>(base_buft, name);
    bundle->buft = std::make_unique<ggml_backend_buffer_type>(ggml_backend_buffer_type {
        /* .iface    = */ {
                           /* .get_name         = */ ggml_backend_cpu_repack_buffer_type_get_name,
                           /* .alloc_buffer     = */ ggml_backend_cpu_repack_buffer_type_alloc_buffer,
                           /* .get_alignment    = */ ggml_backend_cpu_repack_buffer_type_get_alignment,
                           /* .get_max_size     = */ nullptr,  // defaults to SIZE_MAX
                           /* .get_alloc_size   = */ ggml_backend_cpu_repack_buffer_type_get_alloc_size,
                           /* .is_host          = */ nullptr,
                           },
        /* .device  = */ device,
        /* .context = */ bundle->context.get(),
    });

    ggml_backend_buffer_type_t result = bundle->buft.get();
    repack_buffer_types().push_back(std::move(bundle));
    return result;
}

bool ggml_backend_cpu_is_repack_buffer_type(ggml_backend_buffer_type_t buft) {
    // This is queried on the compute hot path. All repack buffer types use
    // this private allocator, so function identity is an O(1), lock-free tag.
    return buft != nullptr && buft->iface.alloc_buffer == ggml_backend_cpu_repack_buffer_type_alloc_buffer;
}

ggml_backend_buffer_type_t ggml_backend_cpu_repack_buffer_type(void) {
    return ggml_backend_cpu_repack_buffer_type_from(
        ggml_backend_cpu_buffer_type(),
        ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        "CPU_REPACK");
}
