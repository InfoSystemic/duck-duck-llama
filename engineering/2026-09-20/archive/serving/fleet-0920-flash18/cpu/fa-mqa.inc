#pragma once
// Cell-split MQA attention for short query batches (GLM DSA/MLA decode and the MTP draft).
// One K/V head and one f16 mask row per query. Each thread takes a fixed share of the cache cells and computes partial
// softmax sums for all query rows with two GEMMs per tile; the partials are merged per row.
//
// The result for one query must not depend on the OTHER queries of the batch: under speculative decoding those are draft
// tokens, and a verified token must come out the same whatever was drafted next to it. So:
//  - a cell belongs to a worker by its index alone (blocks of 64 cells dealt round-robin), never by its rank in the visible set
//  - the softmax maximum is taken over the whole row first (pass A), so no running rescale depends on tile boundaries
//  - a row's weight sum is added sequentially in cell order; masked or padded cells add an exact 0.0 and change nothing
// Cells that only another query can see sit in the tiles as exact zeros, which is what keeps the GEMMs wide.

#include <vector>
#if defined(__AVX512BW__)
#include <immintrin.h>
#endif

static constexpr int64_t F18_FA_TILE     = 64;  // cells per tile
static constexpr int64_t F18_FA_MAX_Q    = 16;  // query tokens
static constexpr int64_t F18_FA_MAX_ROWS = 256; // query tokens * heads

struct f18_fa_scratch {
    std::vector<int32_t> cells;
    float * buf = nullptr;
    size_t  cap = 0;

    float * reserve(size_t n) {
        if (n > cap) {
            std::free(buf);
            void * p = nullptr;
            GGML_ASSERT(posix_memalign(&p, 64, n*sizeof(float)) == 0);
            buf = (float *) p;
            cap = n;
        }
        return buf;
    }
};

static thread_local f18_fa_scratch f18_fa_tls;

#ifdef F18_FA_TIMING
#include <x86intrin.h>
static thread_local uint64_t f18_fa_t[8] = {};
static thread_local uint64_t f18_fa_calls = 0;
#define F18_T(i) { const uint64_t now_ = __rdtsc(); f18_fa_t[i] += now_ - t_last_; t_last_ = now_; }
#else
#define F18_T(i)
#endif

#if defined(__AVX512F__)
// out[j] = column j of the 16x16 block with rows r[0..15]
static inline void f18_transpose16(const __m512 r[16], __m512 out[16]) {
    __m512 t[16], u[16], v[16];
    for (int i = 0; i < 8; ++i) {
        t[2*i]   = _mm512_unpacklo_ps(r[2*i], r[2*i+1]);
        t[2*i+1] = _mm512_unpackhi_ps(r[2*i], r[2*i+1]);
    }
    for (int i = 0; i < 4; ++i) {
        u[4*i+0] = _mm512_shuffle_ps(t[4*i+0], t[4*i+2], 0x44);
        u[4*i+1] = _mm512_shuffle_ps(t[4*i+0], t[4*i+2], 0xEE);
        u[4*i+2] = _mm512_shuffle_ps(t[4*i+1], t[4*i+3], 0x44);
        u[4*i+3] = _mm512_shuffle_ps(t[4*i+1], t[4*i+3], 0xEE);
    }
    for (int i = 0; i < 4; ++i) {
        v[i]    = _mm512_shuffle_f32x4(u[i],   u[i+4],  0x88);
        v[i+4]  = _mm512_shuffle_f32x4(u[i],   u[i+4],  0xDD);
        v[i+8]  = _mm512_shuffle_f32x4(u[i+8], u[i+12], 0x88);
        v[i+12] = _mm512_shuffle_f32x4(u[i+8], u[i+12], 0xDD);
    }
    for (int i = 0; i < 4; ++i) {
        out[i]    = _mm512_shuffle_f32x4(v[i],   v[i+8],  0x88);
        out[i+4]  = _mm512_shuffle_f32x4(v[i+4], v[i+12], 0x88);
        out[i+8]  = _mm512_shuffle_f32x4(v[i],   v[i+8],  0xDD);
        out[i+12] = _mm512_shuffle_f32x4(v[i+4], v[i+12], 0xDD);
    }
}
#endif

// Kt[d*Tw + tk] = Kr[tk*DK + d]
static void f18_fa_transpose(const float * Kr, float * Kt, int64_t Tw, int64_t DK) {
#if defined(__AVX512F__)
    if (Tw % 16 == 0 && DK % 16 == 0) {
        __m512 r[16], c[16];
        for (int64_t tk = 0; tk < Tw; tk += 16) {
            for (int64_t d = 0; d < DK; d += 16) {
                for (int i = 0; i < 16; ++i) {
                    r[i] = _mm512_loadu_ps(Kr + (tk + i)*DK + d);
                }
                f18_transpose16(r, c);
                for (int j = 0; j < 16; ++j) {
                    _mm512_storeu_ps(Kt + (d + j)*Tw + tk, c[j]);
                }
            }
        }
        return;
    }
#endif
    for (int64_t tk = 0; tk < Tw; ++tk) {
        for (int64_t d = 0; d < DK; ++d) {
            Kt[d*Tw + tk] = Kr[tk*DK + d];
        }
    }
}

static bool f18_fa_mqa_supported(const ggml_compute_params * params, const ggml_tensor * dst) {
    if (!(f18_features() & F18_FA_MQA) || params->use_ref) {
        return false;
    }

    const ggml_tensor * q     = dst->src[0];
    const ggml_tensor * k     = dst->src[1];
    const ggml_tensor * v     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    if (!mask || sinks) {
        return false;
    }
    if (q->type != GGML_TYPE_F32 || mask->type != GGML_TYPE_F16) {
        return false;
    }
    if (k->type != v->type || (k->type != GGML_TYPE_F16 && k->type != GGML_TYPE_F32)) {
        return false;
    }
    if (k->ne[2] != 1 || v->ne[2] != 1 || q->ne[3] != 1 || k->ne[3] != 1 || v->ne[3] != 1 || k->ne[1] != v->ne[1]) {
        return false;
    }
    if (mask->ne[2] != 1 || mask->ne[3] != 1 || mask->ne[0] < k->ne[1] || mask->ne[1] < q->ne[1]) {
        return false;
    }
    if (q->ne[1] > F18_FA_MAX_Q || q->ne[1]*q->ne[2] > F18_FA_MAX_ROWS || (q->ne[2] % 2) != 0) {
        return false;
    }

    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }

    // the work buffer carries one pointer per thread
    return params->wdata != nullptr && params->wsize >= params->nth*sizeof(float *);
}

static constexpr int64_t F18_FA_BLOCK = 64; // cells per round-robin block

static void f18_fa_mqa(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * q    = dst->src[0];
    const ggml_tensor * k    = dst->src[1];
    const ggml_tensor * v    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    const int64_t DK   = k->ne[0];
    const int64_t DV   = v->ne[0];
    const int64_t N    = q->ne[1];
    const int64_t H    = q->ne[2];
    const int64_t R    = N*H;
    const int64_t n_kv = k->ne[1];
    const int64_t T    = F18_FA_TILE;

    const int ith = params->ith;
    const int nth = params->nth;

    float scale = 1.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    const bool is_f16 = k->type == GGML_TYPE_F16;
    const bool v_is_k = v->data == k->data && v->nb[1] == k->nb[1] && DV == DK;

    auto & tls = f18_fa_tls;
#ifdef F18_FA_TIMING
    uint64_t t_last_ = __rdtsc();
#endif

    const uint16_t * mrow[F18_FA_MAX_Q];
    for (int64_t n = 0; n < N; ++n) {
        mrow[n] = (const uint16_t *) ((const char *) mask->data + n*mask->nb[1]);
    }

    // this worker's cells: its round-robin blocks, kept when at least one query can see the cell
    tls.cells.clear();
    for (int64_t b0 = (int64_t) ith*F18_FA_BLOCK; b0 < n_kv; b0 += (int64_t) nth*F18_FA_BLOCK) {
        const int64_t b1 = std::min(b0 + F18_FA_BLOCK, n_kv);
        int64_t c = b0;
#if defined(__AVX512BW__)
        const __m512i ninf = _mm512_set1_epi16((short) 0xFC00);
        for (; c + 32 <= b1; c += 32) {
            __mmask32 vis = 0;
            for (int64_t n = 0; n < N; ++n) {
                vis |= _mm512_cmpneq_epi16_mask(_mm512_loadu_si512((const void *) (mrow[n] + c)), ninf);
            }
            while (vis) {
                tls.cells.push_back((int32_t) (c + __builtin_ctz(vis)));
                vis &= vis - 1;
            }
        }
#endif
        for (; c < b1; ++c) {
            bool vis = false;
            for (int64_t n = 0; n < N; ++n) {
                vis |= mrow[n][c] != 0xFC00;
            }
            if (vis) {
                tls.cells.push_back((int32_t) c);
            }
        }
    }

    F18_T(0)
    const int64_t nL     = (int64_t) tls.cells.size();
    const int64_t nTiles = (nL + T - 1)/T;
    const int64_t nLp    = nTiles*T; // score storage, tile aligned

    // published block: [Mloc (R floats)] [S (R doubles)] [PV (R x DV floats)]; R is even here, so the doubles are aligned
    const size_t n_part = (size_t) R*(3 + DV);
    const size_t need   = (size_t) R*DK + (size_t) DK*T + (size_t) T*DK + (v_is_k ? 0 : (size_t) T*DV) + (size_t) R*nLp + (size_t) N*T + n_part + (size_t) DV + (size_t) R + 64;

    float * Qp   = tls.reserve(need);
    float * Kt   = Qp + R*DK;
    float * Kr   = Kt + DK*T;
    float * Vr   = v_is_k ? Kr : Kr + T*DK;
    float * SC   = Vr + T*DV;      // scores of all tiles, [tile][R x Tw]
    float * Mq   = SC + R*nLp;
    float * part = Mq + N*T;
    float * acc  = part + n_part;
    float * Mg   = acc + DV;       // row maxima over all workers

    float  * PM = part;
    double * PS = (double *) (part + R);
    float  * PV = part + 3*R;

    for (int64_t r = 0; r < R; ++r) {
        PM[r] = -INFINITY;
        PS[r] = 0.0;
    }
    memset(PV, 0, (size_t) R*DV*sizeof(float));

    ((float **) params->wdata)[ith] = part;

    // query rows: r = n*H + h, the order of the dst rows
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t h = 0; h < H; ++h) {
            memcpy(Qp + (n*H + h)*DK, (const char *) q->data + n*q->nb[1] + h*q->nb[2], DK*sizeof(float));
        }
    }

    auto load_tile = [&](int64_t t0, int64_t nt, int64_t Tw, bool with_mask) {
        for (int64_t tk = 0; tk < nt; ++tk) {
            const int64_t cell = tls.cells[t0 + tk];
            const char * k_data = (const char *) k->data + cell*k->nb[1];
            if (is_f16) {
                ggml_cpu_fp16_to_fp32((const ggml_fp16_t *) k_data, Kr + tk*DK, DK);
            } else {
                memcpy(Kr + tk*DK, k_data, DK*sizeof(float));
            }
            if (!v_is_k) {
                const char * v_data = (const char *) v->data + cell*v->nb[1];
                if (is_f16) {
                    ggml_cpu_fp16_to_fp32((const ggml_fp16_t *) v_data, Vr + tk*DV, DV);
                } else {
                    memcpy(Vr + tk*DV, v_data, DV*sizeof(float));
                }
            }
            if (with_mask) {
                for (int64_t n = 0; n < N; ++n) {
                    Mq[n*Tw + tk] = GGML_CPU_FP16_TO_FP32(mrow[n][cell]);
                }
            }
        }
        if (nt < Tw) {
            memset(Kr + nt*DK, 0, (size_t) (Tw - nt)*DK*sizeof(float));
            if (!v_is_k) {
                memset(Vr + nt*DV, 0, (size_t) (Tw - nt)*DV*sizeof(float));
            }
            if (with_mask) {
                for (int64_t n = 0; n < N; ++n) {
                    for (int64_t tk = nt; tk < Tw; ++tk) {
                        Mq[n*Tw + tk] = -INFINITY;
                    }
                }
            }
        }
    };

    F18_T(1)
    // pass A: masked scores of every tile, and the row maxima over this worker's cells
    for (int64_t it = 0; it < nTiles; ++it) {
        const int64_t t0 = it*T;
        const int64_t nt = std::min(T, nL - t0);
        const int64_t Tw = (nt + 15) & ~(int64_t) 15;
        float * KQ = SC + R*t0;

        load_tile(t0, nt, Tw, true);
        F18_T(2)
        f18_fa_transpose(Kr, Kt, Tw, DK);
        F18_T(3)
        memset(KQ, 0, (size_t) R*Tw*sizeof(float));
        simd_gemm(KQ, Qp, Kt, (int) R, (int) DK, (int) Tw);
        ggml_vec_scale_f32((int) (R*Tw), KQ, scale);
        F18_T(4)
        for (int64_t r = 0; r < R; ++r) {
            float * kq_row = KQ + r*Tw;
            ggml_vec_add_f32((int) Tw, kq_row, kq_row, Mq + (r/H)*Tw);
            float tile_max;
            ggml_vec_max_f32((int) Tw, &tile_max, kq_row);
            PM[r] = fmaxf(PM[r], tile_max);
        }
        F18_T(5)
    }

    ggml_barrier(params->threadpool);

    float ** parts = (float **) params->wdata;
    for (int64_t r = 0; r < R; ++r) {
        float m = -INFINITY;
        for (int p = 0; p < nth; ++p) {
            m = fmaxf(m, parts[p][r]);
        }
        Mg[r] = m;
    }

    // pass B: weights against the row maximum, weight sums in cell order, V accumulation
    for (int64_t it = 0; it < nTiles; ++it) {
        const int64_t t0 = it*T;
        const int64_t nt = std::min(T, nL - t0);
        const int64_t Tw = (nt + 15) & ~(int64_t) 15;
        float * KQ = SC + R*t0;

        load_tile(t0, nt, Tw, false);
        for (int64_t r = 0; r < R; ++r) {
            float * kq_row = KQ + r*Tw;
            if (Mg[r] == -INFINITY) {
                memset(kq_row, 0, (size_t) Tw*sizeof(float));
                continue;
            }
            ggml_vec_soft_max_f32((int) Tw, kq_row, kq_row, Mg[r]); // Tw is a multiple of 16: no scalar tail
            // one running double per row across all tiles: the sum cannot depend on where the tiles are cut
            double sum = PS[r];
            for (int64_t tk = 0; tk < Tw; ++tk) {
                sum += kq_row[tk];
            }
            PS[r] = sum;
        }
        simd_gemm(PV, KQ, Vr, (int) R, (int) Tw, (int) DV);
        F18_T(6)
    }

    ggml_barrier(params->threadpool);
    F18_T(7)
#ifdef F18_FA_TIMING
    if (ith == 0 && ++f18_fa_calls % 100 == 0) {
        fprintf(stderr, "F18_FA_TIMING calls=%llu scan=%llu setup=%llu pack=%llu transpose=%llu gemm1=%llu max=%llu passB=%llu barrier=%llu (kcycles/call)\n",
            (unsigned long long) f18_fa_calls, (unsigned long long) (f18_fa_t[0]/f18_fa_calls/1000), (unsigned long long) (f18_fa_t[1]/f18_fa_calls/1000),
            (unsigned long long) (f18_fa_t[2]/f18_fa_calls/1000), (unsigned long long) (f18_fa_t[3]/f18_fa_calls/1000), (unsigned long long) (f18_fa_t[4]/f18_fa_calls/1000),
            (unsigned long long) (f18_fa_t[5]/f18_fa_calls/1000), (unsigned long long) (f18_fa_t[6]/f18_fa_calls/1000), (unsigned long long) (f18_fa_t[7]/f18_fa_calls/1000));
    }
#endif

    for (int64_t r = ith; r < R; r += nth) {
        double S_final = 0.0;
        memset(acc, 0, (size_t) DV*sizeof(float));
        for (int p = 0; p < nth; ++p) {
            const float * pp = parts[p];
            S_final += ((const double *) (pp + R))[r];
            const float * V_chunk = pp + 3*R + r*DV;
            for (int64_t d = 0; d < DV; ++d) {
                acc[d] += V_chunk[d];
            }
        }
        if (S_final != 0.0) {
            ggml_vec_scale_f32((int) DV, acc, (float) (1.0/S_final));
        }
        memcpy((char *) dst->data + r*dst->nb[1], acc, (size_t) DV*sizeof(float));
    }
}
