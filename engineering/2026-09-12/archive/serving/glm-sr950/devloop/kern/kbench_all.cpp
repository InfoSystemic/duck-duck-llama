// 16-row interleaved AVX-512 VNNI gemv kernels for Q4_K / Q5_K / Q6_K / Q8_0, with reference checks and streaming benchmarks.
#include <immintrin.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <chrono>
#include <random>
#include <omp.h>
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"
extern "C" {
void quantize_row_q8_K(const float * x, void * y, int64_t k);
void quantize_row_q8_0(const float * x, void * y, int64_t k);
void ggml_vec_dot_q4_K_q8_K(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void ggml_vec_dot_q5_K_q8_K(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void ggml_vec_dot_q6_K_q8_K(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void ggml_vec_dot_q8_0_q8_0(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
}
static inline void get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else { *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4); *m = (q[j+4] >> 4) | ((q[j-0] >> 6) << 4); }
}
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
#define BC32(p) _mm512_set1_epi32(*(const int32_t *)(p))

// ============================== Q4_K ==============================
struct block_q4_K_x16 { ggml_half d[16]; ggml_half dmin[16]; uint8_t scales[8][16]; uint8_t mins[8][16]; uint8_t qs[2048]; };
static void repack_q4_K_x16(const block_q4_K * in, int nb, block_q4_K_x16 * out) {
    for (int b = 0; b < nb; b++) { block_q4_K_x16 & o = out[b];
        for (int r = 0; r < 16; r++) { const block_q4_K & x = in[r * nb + b];
            o.d[r] = x.data.data.d; o.dmin[r] = x.data.data.dmin;
            for (int sb = 0; sb < 8; sb++) { uint8_t sc, mn; get_scale_min_k4(sb, x.scales, &sc, &mn); o.scales[sb][r] = sc; o.mins[sb][r] = mn; }
            for (int j = 0; j < 4; j++) for (int i0 = 0; i0 < 8; i0++) for (int t = 0; t < 4; t++) o.qs[((j*8 + i0)*16 + r)*4 + t] = x.qs[32*j + 4*i0 + t];
        } }
}
static void gemv_q4_K_x16(int n, float * s, const block_q4_K_x16 * vx, const block_q8_K * vy, int nc) {
    const int nb = n / QK_K; const __m512i m4 = _mm512_set1_epi8(0x0F);
    for (int g = 0; g < nc / 16; g++) { const block_q4_K_x16 * bp = vx + (size_t) g * nb; __m512 accf = _mm512_setzero_ps();
        for (int b = 0; b < nb; b++) { const block_q8_K * a = vy + b; const uint8_t * qs = bp[b].qs;
            __m512i iacc = _mm512_setzero_si512(), imin = _mm512_setzero_si512();
            for (int j = 0; j < 4; j++) {
                __m512i A0 = _mm512_setzero_si512(), A1 = A0, B0 = A0, B1 = A0; const uint8_t * qj = qs + j * 512; const int8_t * ya = a->qs + 64 * j; const int8_t * yb = ya + 32;
#define STEP(i0, AA, BB) { const __m512i v = _mm512_loadu_si512((const void *)(qj + (i0) * 64)); \
                AA = _mm512_dpbusd_epi32(AA, _mm512_and_si512(v, m4), BC32(ya + 4*(i0))); \
                BB = _mm512_dpbusd_epi32(BB, _mm512_and_si512(_mm512_srli_epi16(v, 4), m4), BC32(yb + 4*(i0))); }
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
}
// ============================== Q5_K ==============================
// qsh[i0 (0..7)] = { v_j0[16 rows x 4 bytes], v_j1, v_j2, v_j3, h[16 rows x 4 bytes] } = 5 x 64 B, fully sequential per block
struct block_q5_K_x16 { ggml_half d[16]; ggml_half dmin[16]; uint8_t scales[8][16]; uint8_t mins[8][16]; uint8_t qsh[2560]; };
static void repack_q5_K_x16(const block_q5_K * in, int nb, block_q5_K_x16 * out) {
    for (int b = 0; b < nb; b++) { block_q5_K_x16 & o = out[b];
        for (int r = 0; r < 16; r++) { const block_q5_K & x = in[r * nb + b];
            o.d[r] = x.data.data.d; o.dmin[r] = x.data.data.dmin;
            for (int sb = 0; sb < 8; sb++) { uint8_t sc, mn; get_scale_min_k4(sb, x.scales, &sc, &mn); o.scales[sb][r] = sc; o.mins[sb][r] = mn; }
            for (int i0 = 0; i0 < 8; i0++) {
                for (int j = 0; j < 4; j++) for (int t = 0; t < 4; t++) o.qsh[i0*320 + j*64 + r*4 + t] = x.qs[32*j + 4*i0 + t];
                for (int t = 0; t < 4; t++) o.qsh[i0*320 + 256 + r*4 + t] = x.qh[4*i0 + t];
            }
        } }
}
static void gemv_q5_K_x16(int n, float * s, const block_q5_K_x16 * vx, const block_q8_K * vy, int nc) {
    const int nb = n / QK_K; const __m512i m4 = _mm512_set1_epi8(0x0F); const __m512i m10 = _mm512_set1_epi8(0x10);
    for (int g = 0; g < nc / 16; g++) { const block_q5_K_x16 * bp = vx + (size_t) g * nb; __m512 accf = _mm512_setzero_ps();
        for (int b = 0; b < nb; b++) { const block_q8_K * a = vy + b; const uint8_t * q = bp[b].qsh; const int8_t * y = a->qs;
            __m512i A0 = _mm512_setzero_si512(), B0 = A0, A1 = A0, B1 = A0, A2 = A0, B2 = A0, A3 = A0, B3 = A0;
#define Q5STEP(i0) { const uint8_t * qi = q + (i0) * 320; const __m512i h = _mm512_loadu_si512((const void *)(qi + 256)); \
            const __m512i v0 = _mm512_loadu_si512((const void *)(qi)), v1 = _mm512_loadu_si512((const void *)(qi + 64)); \
            const __m512i v2 = _mm512_loadu_si512((const void *)(qi + 128)), v3 = _mm512_loadu_si512((const void *)(qi + 192)); \
            A0 = _mm512_dpbusd_epi32(A0, _mm512_or_si512(_mm512_and_si512(v0, m4), _mm512_and_si512(_mm512_slli_epi16(h, 4), m10)), BC32(y + 4*(i0))); \
            B0 = _mm512_dpbusd_epi32(B0, _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(v0, 4), m4), _mm512_and_si512(_mm512_slli_epi16(h, 3), m10)), BC32(y + 32 + 4*(i0))); \
            A1 = _mm512_dpbusd_epi32(A1, _mm512_or_si512(_mm512_and_si512(v1, m4), _mm512_and_si512(_mm512_slli_epi16(h, 2), m10)), BC32(y + 64 + 4*(i0))); \
            B1 = _mm512_dpbusd_epi32(B1, _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(v1, 4), m4), _mm512_and_si512(_mm512_slli_epi16(h, 1), m10)), BC32(y + 96 + 4*(i0))); \
            A2 = _mm512_dpbusd_epi32(A2, _mm512_or_si512(_mm512_and_si512(v2, m4), _mm512_and_si512(h, m10)), BC32(y + 128 + 4*(i0))); \
            B2 = _mm512_dpbusd_epi32(B2, _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(v2, 4), m4), _mm512_and_si512(_mm512_srli_epi16(h, 1), m10)), BC32(y + 160 + 4*(i0))); \
            A3 = _mm512_dpbusd_epi32(A3, _mm512_or_si512(_mm512_and_si512(v3, m4), _mm512_and_si512(_mm512_srli_epi16(h, 2), m10)), BC32(y + 192 + 4*(i0))); \
            B3 = _mm512_dpbusd_epi32(B3, _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(v3, 4), m4), _mm512_and_si512(_mm512_srli_epi16(h, 3), m10)), BC32(y + 224 + 4*(i0))); }
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
}
// ============================== Q6_K ==============================
// q[h][i0 (0..7)] = { la[16x4], lb[16x4], hv[16x4] } = 3 x 64 B; la = ql[64h+4i0..], lb = ql[64h+32+4i0..], hv = qh[32h+4i0..]
struct block_q6_K_x16 { ggml_half d[16]; int8_t scales[16][16]; uint8_t q[3072]; };
static void repack_q6_K_x16(const block_q6_K * in, int nb, block_q6_K_x16 * out) {
    for (int b = 0; b < nb; b++) { block_q6_K_x16 & o = out[b];
        for (int r = 0; r < 16; r++) { const block_q6_K & x = in[r * nb + b];
            o.d[r] = x.d;
            for (int sb = 0; sb < 16; sb++) o.scales[sb][r] = x.scales[sb];
            for (int h = 0; h < 2; h++) for (int i0 = 0; i0 < 8; i0++) {
                uint8_t * dst = o.q + (h*8 + i0) * 192;
                for (int t = 0; t < 4; t++) { dst[r*4 + t] = x.ql[64*h + 4*i0 + t]; dst[64 + r*4 + t] = x.ql[64*h + 32 + 4*i0 + t]; dst[128 + r*4 + t] = x.qh[32*h + 4*i0 + t]; }
            }
        } }
}
static void gemv_q6_K_x16(int n, float * s, const block_q6_K_x16 * vx, const block_q8_K * vy, int nc) {
    const int nb = n / QK_K; const __m512i m4 = _mm512_set1_epi8(0x0F); const __m512i m30 = _mm512_set1_epi8(0x30);
    for (int g = 0; g < nc / 16; g++) { const block_q6_K_x16 * bp = vx + (size_t) g * nb; __m512 accf = _mm512_setzero_ps();
        for (int b = 0; b < nb; b++) { const block_q8_K * a = vy + b;
            __m512i iacc = _mm512_setzero_si512(), icorr = _mm512_setzero_si512();
            for (int h = 0; h < 2; h++) {
                const uint8_t * qh0 = bp[b].q + h * 1536; const int8_t * y = a->qs + 128 * h;
                __m512i C00 = _mm512_setzero_si512(), C01 = C00, C10 = C00, C11 = C00, C20 = C00, C21 = C00, C30 = C00, C31 = C00;
#define Q6STEP(i0, X0, X1, X2, X3) { const uint8_t * qi = qh0 + (i0) * 192; \
                const __m512i la = _mm512_loadu_si512((const void *)(qi)), lb = _mm512_loadu_si512((const void *)(qi + 64)), hv = _mm512_loadu_si512((const void *)(qi + 128)); \
                X0 = _mm512_dpbusd_epi32(X0, _mm512_or_si512(_mm512_and_si512(la, m4), _mm512_and_si512(_mm512_slli_epi16(hv, 4), m30)), BC32(y + 4*(i0))); \
                X1 = _mm512_dpbusd_epi32(X1, _mm512_or_si512(_mm512_and_si512(lb, m4), _mm512_and_si512(_mm512_slli_epi16(hv, 2), m30)), BC32(y + 32 + 4*(i0))); \
                X2 = _mm512_dpbusd_epi32(X2, _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(la, 4), m4), _mm512_and_si512(hv, m30)), BC32(y + 64 + 4*(i0))); \
                X3 = _mm512_dpbusd_epi32(X3, _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(lb, 4), m4), _mm512_and_si512(_mm512_srli_epi16(hv, 2), m30)), BC32(y + 96 + 4*(i0))); }
                Q6STEP(0, C00, C10, C20, C30) Q6STEP(1, C00, C10, C20, C30) Q6STEP(2, C00, C10, C20, C30) Q6STEP(3, C00, C10, C20, C30)
                Q6STEP(4, C01, C11, C21, C31) Q6STEP(5, C01, C11, C21, C31) Q6STEP(6, C01, C11, C21, C31) Q6STEP(7, C01, C11, C21, C31)
#undef Q6STEP
                const __m512i accs[8] = {C00, C01, C10, C11, C20, C21, C30, C31};   // sub-block order: 8h + 2q + half
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
}
// ============================== Q8_0 ==============================
struct block_q8_0_x16 { ggml_half d[16]; uint8_t qs[512]; };  // qs[i0][row][4] = (int8 + 128)
static void repack_q8_0_x16(const block_q8_0 * in, int nb, block_q8_0_x16 * out) {
    for (int b = 0; b < nb; b++) { block_q8_0_x16 & o = out[b];
        for (int r = 0; r < 16; r++) { const block_q8_0 & x = in[r * nb + b]; o.d[r] = x.d;
            for (int i0 = 0; i0 < 8; i0++) for (int t = 0; t < 4; t++) o.qs[(i0*16 + r)*4 + t] = (uint8_t)((int) x.qs[4*i0 + t] + 128);
        } }
}
static void gemv_q8_0_x16(int n, float * s, const block_q8_0_x16 * vx, const block_q8_0 * vy, const int32_t * ysum, int nc) {
    const int nb = n / QK8_0;
    for (int g = 0; g < nc / 16; g++) { const block_q8_0_x16 * bp = vx + (size_t) g * nb; __m512 accf = _mm512_setzero_ps();
        for (int b = 0; b < nb; b += 2) {
            __m512i acc0 = _mm512_setzero_si512(), acc1 = _mm512_setzero_si512();
            const uint8_t * q0 = bp[b].qs; const int8_t * y0 = vy[b].qs;
            for (int i0 = 0; i0 < 8; i0++) acc0 = _mm512_dpbusd_epi32(acc0, _mm512_loadu_si512((const void *)(q0 + i0 * 64)), BC32(y0 + 4*i0));
            acc0 = _mm512_sub_epi32(acc0, _mm512_set1_epi32(128 * ysum[b]));
            accf = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc0), _mm512_mul_ps(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) bp[b].d)), _mm512_set1_ps(ggml_fp16_to_fp32(vy[b].d))), accf);
            if (b + 1 < nb) {
                const uint8_t * q1 = bp[b+1].qs; const int8_t * y1 = vy[b+1].qs;
                for (int i0 = 0; i0 < 8; i0++) acc1 = _mm512_dpbusd_epi32(acc1, _mm512_loadu_si512((const void *)(q1 + i0 * 64)), BC32(y1 + 4*i0));
                acc1 = _mm512_sub_epi32(acc1, _mm512_set1_epi32(128 * ysum[b+1]));
                accf = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc1), _mm512_mul_ps(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) bp[b+1].d)), _mm512_set1_ps(ggml_fp16_to_fp32(vy[b+1].d))), accf);
            }
        }
        _mm512_storeu_ps(s + g * 16, accf); }
}

// ============================== harness ==============================
template <typename BLK, typename X16, typename ACT, typename QFN, typename RPFN, typename REFFN, typename KFN>
static void run_type(const char * name, int k, int rows_per_thread, int reps, int blk, QFN quant, RPFN repack, REFFN ref_dot, KFN kern, ACT * act, std::vector<float> & yf, std::mt19937 & rng) {
    const int nthreads = omp_get_max_threads(); const int nrows = rows_per_thread * nthreads; const int nb = k / blk;
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> xf(k); std::vector<BLK> native((size_t) nrows * nb);
    for (int r = 0; r < nrows; r++) { for (int i = 0; i < k; i++) xf[i] = nd(rng); quant(xf.data(), native.data() + (size_t) r * nb, k); }
    std::vector<X16> w((size_t) nrows / 16 * nb);
    #pragma omp parallel for
    for (int g = 0; g < nrows / 16; g++) repack(native.data() + (size_t) g * 16 * nb, nb, w.data() + (size_t) g * nb);
    std::vector<float> ref(nrows), out(nrows);
    for (int r = 0; r < nrows; r++) ref_dot(k, &ref[r], 0, native.data() + (size_t) r * nb, 0, act, 0, 1);
    kern(k, out.data(), w.data(), act, nrows);
    double maxe = 0, maxref = 0; for (int r = 0; r < nrows; r++) { maxe = fmax(maxe, fabs(out[r] - ref[r])); maxref = fmax(maxref, fabs(ref[r])); }
    double best = 1e9;
    for (int it = 0; it < reps; it++) { double t0 = now();
        #pragma omp parallel
        { int t = omp_get_thread_num(); kern(k, out.data() + (size_t) t * rows_per_thread, w.data() + (size_t) t * rows_per_thread / 16 * nb, act, rows_per_thread); }
        double dt = now() - t0; if (dt < best) best = dt; }
    const double bytes = (double) nrows / 16 * nb * sizeof(X16);
    double bestl2 = 1e9; const int small = 64;
    for (int it = 0; it < reps; it++) { double t0 = now();
        #pragma omp parallel
        { int t = omp_get_thread_num(); for (int q = 0; q < 100; q++) kern(k, out.data() + (size_t) t * rows_per_thread, w.data() + (size_t) t * rows_per_thread / 16 * nb, act, small); }
        double dt = now() - t0; if (dt < bestl2) bestl2 = dt; }
    const double bytesl2 = (double) small / 16 * nb * sizeof(X16) * 100 * nthreads;
    printf("%-6s bpw=%.3f max|ref|=%.2f maxerr=%.2e | stream %.1f GB/s (%.1f/core) | cache-resident %.1f GB/s (%.1f/core)\n",
        name, 8.0 * sizeof(X16) / 16 / blk, maxref, maxe, bytes / best / 1e9, bytes / best / 1e9 / nthreads, bytesl2 / bestl2 / 1e9, bytesl2 / bestl2 / 1e9 / nthreads);
}

int main(int argc, char ** argv) {
    const int k = argc > 1 ? atoi(argv[1]) : 6144; const int rows_per_thread = argc > 2 ? atoi(argv[2]) : 4096; const int reps = argc > 3 ? atoi(argv[3]) : 5;
    ggml_cpu_init();
    std::mt19937 rng(42); std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> yf(k); for (int i = 0; i < k; i++) yf[i] = nd(rng);
    std::vector<block_q8_K> y8k(k / QK_K); quantize_row_q8_K(yf.data(), y8k.data(), k);
    std::vector<block_q8_0> y80(k / QK8_0); quantize_row_q8_0(yf.data(), y80.data(), k);
    std::vector<int32_t> ysum(k / QK8_0); for (int b = 0; b < k / QK8_0; b++) { int s = 0; for (int i = 0; i < 32; i++) s += y80[b].qs[i]; ysum[b] = s; }
    printf("k=%d rows/thread=%d threads=%d\n", k, rows_per_thread, omp_get_max_threads());
    run_type<block_q4_K, block_q4_K_x16>("Q4_K", k, rows_per_thread, reps, QK_K, quantize_row_q4_K_ref, repack_q4_K_x16, ggml_vec_dot_q4_K_q8_K,
        [](int n, float * s, const block_q4_K_x16 * w, const block_q8_K * a, int nc) { gemv_q4_K_x16(n, s, w, a, nc); }, y8k.data(), yf, rng);
    run_type<block_q5_K, block_q5_K_x16>("Q5_K", k, rows_per_thread, reps, QK_K, quantize_row_q5_K_ref, repack_q5_K_x16, ggml_vec_dot_q5_K_q8_K,
        [](int n, float * s, const block_q5_K_x16 * w, const block_q8_K * a, int nc) { gemv_q5_K_x16(n, s, w, a, nc); }, y8k.data(), yf, rng);
    run_type<block_q6_K, block_q6_K_x16>("Q6_K", k, rows_per_thread, reps, QK_K, quantize_row_q6_K_ref, repack_q6_K_x16, ggml_vec_dot_q6_K_q8_K,
        [](int n, float * s, const block_q6_K_x16 * w, const block_q8_K * a, int nc) { gemv_q6_K_x16(n, s, w, a, nc); }, y8k.data(), yf, rng);
    const int32_t * ys = ysum.data();
    run_type<block_q8_0, block_q8_0_x16>("Q8_0", k, rows_per_thread, reps, QK8_0, quantize_row_q8_0_ref, repack_q8_0_x16, ggml_vec_dot_q8_0_q8_0,
        [ys](int n, float * s, const block_q8_0_x16 * w, const block_q8_0 * a, int nc) { gemv_q8_0_x16(n, s, w, a, ys, nc); }, y80.data(), yf, rng);
    return 0;
}
