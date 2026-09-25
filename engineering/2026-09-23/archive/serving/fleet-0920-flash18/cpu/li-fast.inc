#pragma once
// Lightning indexer pool score, restructured: the same arithmetic as ggml_compute_forward_lightning_indexer, in the same order,
// without 96 ggml_vec_dot_f32 calls per cached row.
//
// score[t][ik] = sum_h relu(q[h][t] . k[ik]) * w[h][t]  + mask[t][ik]
//
// The deployed kernel loops (stream, token, row, head) and calls ggml_vec_dot_f32 for every (head, token, row): each call reloads
// the row, runs the 4 x 16-lane FMA ladder, reduces, and returns through a function pointer. At 128K context that is 3.4 ms per
// layer for 0.84 GFLOP: ~10-13 GFLOP/s per core. Here each query vector is held in eight zmm registers and applied to a block of
// 16 cached rows (8 KB, L1-resident), so the 96 query vectors (48 KB) are streamed from L2 once per block instead of once per row;
// the per-dot arithmetic is exactly ggml_vec_dot_f32's AVX-512 path (GGML_F32_STEP 64, four accumulators, fma with a zero addend
// for the first half, pairwise reduce, _mm512_reduce_add_ps), the head sum is the same float chain in head order, and the mask
// add is last -- so the bytes are identical (cpu/test/li_fast_check.cpp compares both kernels bit for bit: 2.3x per core; the
// per-dot horizontal reduce is what remains, and a true GEMM would change the summation order).
//
// Scope: k F32 with 128 columns, q F32, dst F32, mask F16, contiguous rows, AVX-512F. Anything else falls back to the deployed
// kernel. Threads split the cached rows as before.
#include <immintrin.h>

static inline bool f18_li_fast_supported(const ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * w = dst->src[2];
    const ggml_tensor * m = dst->src[3];
    return ggml_cpu_has_avx512() &&
           q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F32 && w->type == GGML_TYPE_F32 && m->type == GGML_TYPE_F16 &&
           dst->type == GGML_TYPE_F32 && q->ne[0] == 128 && k->ne[0] == 128 && k->ne[1] == 1 && q->ne[2] <= 8 &&
           q->nb[0] == sizeof(float) && k->nb[0] == sizeof(float) && w->nb[0] == sizeof(float) && m->nb[0] == sizeof(ggml_fp16_t) &&
           dst->nb[0] == sizeof(float) && k->nb[2] == 128*sizeof(float) && q->nb[1] == 128*sizeof(float);
}

__attribute__((target("avx512f")))
static void f18_lightning_indexer_fast(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * w = dst->src[2];
    const ggml_tensor * m = dst->src[3];

    const int n_head   = q->ne[1];
    const int n_tokens = q->ne[2];
    const int n_stream = q->ne[3];
    const int n_kv     = k->ne[2];

    const int ith = params->ith, nth = params->nth;
    const int dr  = (n_kv + nth - 1)/nth;
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, n_kv);

    // Blocks of 16 cached rows (8 KB, L1-resident) see each query vector once: the 96 query vectors (48 KB) are streamed from L2
    // once per block instead of once per row. Per (token, row) the head terms are still added in head order, each dot product
    // still runs the same two-step four-accumulator ladder and the same reduce, and the mask is still added last.
    enum { BLOCK = 16 };
    for (int s = 0; s < n_stream; ++s) {
        for (int ik0 = ir0; ik0 < ir1; ik0 += BLOCK) {
            const int nb = MIN(BLOCK, ir1 - ik0);
            float score[8][BLOCK]; // n_tokens is at most 8 on this path (3 in production); larger batches take the deployed kernel
            for (int t = 0; t < n_tokens; ++t) for (int r = 0; r < nb; ++r) score[t][r] = 0.0f;
            for (int t = 0; t < n_tokens; ++t) {
                const float * w_row = (const float *) ((const char *) w->data + t*w->nb[1] + s*w->nb[3]);
                for (int h = 0; h < n_head; ++h) {
                    const float * q_row = (const float *) ((const char *) q->data + h*q->nb[1] + t*q->nb[2] + s*q->nb[3]);
                    __m512 qv[8];
                    for (int j = 0; j < 8; ++j) qv[j] = _mm512_loadu_ps(q_row + 16*j);
                    const float wh = w_row[h];
                    for (int r = 0; r < nb; ++r) {
                        const float * k_row = (const float *) ((const char *) k->data + (ik0 + r)*k->nb[2] + s*k->nb[3]);
                        // ggml_vec_dot_f32, n = 128: two GGML_F32_STEP iterations over four 16-lane accumulators
                        __m512 acc[4];
                        for (int j = 0; j < 4; ++j) acc[j] = _mm512_fmadd_ps(qv[j], _mm512_loadu_ps(k_row + 16*j), _mm512_setzero_ps());
                        for (int j = 0; j < 4; ++j) acc[j] = _mm512_fmadd_ps(qv[4 + j], _mm512_loadu_ps(k_row + 64 + 16*j), acc[j]);
                        // GGML_F32x16_REDUCE: offset 2 then 1, then the horizontal reduce
                        acc[0] = _mm512_add_ps(acc[0], acc[2]);
                        acc[1] = _mm512_add_ps(acc[1], acc[3]);
                        acc[0] = _mm512_add_ps(acc[0], acc[1]);
                        const float qk = (float) (ggml_float) _mm512_reduce_add_ps(acc[0]);
                        score[t][r] += MAX(qk, 0.0f) * wh;
                    }
                }
            }
            for (int t = 0; t < n_tokens; ++t) {
                const ggml_fp16_t * m_row = (const ggml_fp16_t *) ((const char *) m->data + t*m->nb[1] + (s % m->ne[3])*m->nb[3]);
                float * dst_row = (float *) ((char *) dst->data + t*dst->nb[1] + s*dst->nb[3]);
                for (int r = 0; r < nb; ++r) dst_row[ik0 + r] = score[t][r] + GGML_CPU_FP16_TO_FP32(m_row[ik0 + r]);
            }
        }
    }
}
