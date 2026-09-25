// x16 Q8_0/MXFP4 prompt-batch GEMM epilogue prototypes vs the shipped kernel (copied verbatim).
//   ref : shipped kernel (separate ysum/yd arrays, zero-init + vpsubd)
//   A   : same arrays, but the accumulator starts at -mult*ysum (broadcast from a pre-multiplied array): no vpsubd
//   B   : activation blocks carry {qs[32], int32 bias, float yd} (40 B): one pointer per column
// usage: proto q8|mx <M> <K> <N> <NR> <CH> <threads> <iters>
#include <immintrin.h>
#include <omp.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <algorithm>

struct block_q8_0     { uint16_t d; int8_t qs[32]; };
struct block_q8_0_x16 { uint16_t d[16]; uint8_t qs[512]; };
struct block_mxfp4_x16 { uint8_t e[16]; uint8_t qs[256]; };
struct block_q8b { int8_t qs[32]; int32_t bias; float yd; };   // variant B activation block
static_assert(sizeof(block_q8b) == 40, "");
static const int8_t kvalues_mxfp4[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

static float f16_to_f32(uint16_t h) { return _cvtsh_ss(h); }
static uint16_t f16_bits(float f) { return _cvtss_sh(f, 0); }
static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static inline __m512i dp_m32(__m512i acc, __m512i w, const void * p) {
    __asm__("vpdpbusd %2%{1to16%}, %1, %0" : "+v"(acc) : "v"(w), "m"(*(const int32_t *) p));
    return acc;
}
static inline __m512 mxfp4_scales(const uint8_t * e) {
    const __m512i ev  = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *) e));
    const __m512i nrm = _mm512_slli_epi32(_mm512_sub_epi32(ev, _mm512_set1_epi32(1)), 23);
    const __m512i den = _mm512_sllv_epi32(_mm512_set1_epi32(0x00200000), ev);
    return _mm512_castsi512_ps(_mm512_mask_blend_epi32(_mm512_cmplt_epu32_mask(ev, _mm512_set1_epi32(2)), nrm, den));
}

// ---- ref (shipped) ----
template <int NR> __attribute__((noinline)) static void q8_ref(int nb, float * const * outs, const block_q8_0 * const * y, const block_q8_0_x16 * vxb, int nc, const int32_t * const * ysum, const float * const * yd) {
    for (int g = 0; g < nc / 16; g++) {
        const block_q8_0_x16 * bp = vxb + (size_t) g * nb;
        __m512 accf[NR];
        for (int j = 0; j < NR; j++) accf[j] = _mm512_setzero_ps();
        for (int b = 0; b < nb; b++) {
            __m512i acc[NR];
            for (int j = 0; j < NR; j++) acc[j] = _mm512_setzero_si512();
            const uint8_t * q0 = bp[b].qs;
            for (int i0 = 0; i0 < 8; i0++) {
                const __m512i w = _mm512_loadu_si512((const void *)(q0 + i0 * 64));
                for (int j = 0; j < NR; j++) acc[j] = dp_m32(acc[j], w, y[j][b].qs + 4*i0);
            }
            const __m512 dw = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) bp[b].d));
            for (int j = 0; j < NR; j++) {
                const __m512i a = _mm512_sub_epi32(acc[j], _mm512_set1_epi32(128 * ysum[j][b]));
                accf[j] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(a), _mm512_mul_ps(dw, _mm512_set1_ps(yd[j][b])), accf[j]);
            }
        }
        for (int j = 0; j < NR; j++) _mm512_storeu_ps(outs[j] + g * 16, accf[j]);
    }
}
// ---- A: pre-multiplied negated bias array, accumulator starts at it ----
template <int NR> __attribute__((noinline)) static void q8_A(int nb, float * const * outs, const block_q8_0 * const * y, const block_q8_0_x16 * vxb, int nc, const int32_t * const * ybias, const float * const * yd) {
    for (int g = 0; g < nc / 16; g++) {
        const block_q8_0_x16 * bp = vxb + (size_t) g * nb;
        __m512 accf[NR];
        for (int j = 0; j < NR; j++) accf[j] = _mm512_setzero_ps();
        for (int b = 0; b < nb; b++) {
            __m512i acc[NR];
            for (int j = 0; j < NR; j++) acc[j] = _mm512_set1_epi32(ybias[j][b]);
            const uint8_t * q0 = bp[b].qs;
            for (int i0 = 0; i0 < 8; i0++) {
                const __m512i w = _mm512_loadu_si512((const void *)(q0 + i0 * 64));
                for (int j = 0; j < NR; j++) acc[j] = dp_m32(acc[j], w, y[j][b].qs + 4*i0);
            }
            const __m512 dw = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) bp[b].d));
            for (int j = 0; j < NR; j++) accf[j] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[j]), _mm512_mul_ps(dw, _mm512_set1_ps(yd[j][b])), accf[j]);
        }
        for (int j = 0; j < NR; j++) _mm512_storeu_ps(outs[j] + g * 16, accf[j]);
    }
}
// ---- B: 40-byte activation blocks {qs, bias, yd} ----
template <int NR> __attribute__((noinline)) static void q8_B(int nb, float * const * outs, const block_q8b * const * y, const block_q8_0_x16 * vxb, int nc) {
    for (int g = 0; g < nc / 16; g++) {
        const block_q8_0_x16 * bp = vxb + (size_t) g * nb;
        __m512 accf[NR];
        for (int j = 0; j < NR; j++) accf[j] = _mm512_setzero_ps();
        for (int b = 0; b < nb; b++) {
            __m512i acc[NR];
            for (int j = 0; j < NR; j++) acc[j] = _mm512_set1_epi32(y[j][b].bias);
            const uint8_t * q0 = bp[b].qs;
            for (int i0 = 0; i0 < 8; i0++) {
                const __m512i w = _mm512_loadu_si512((const void *)(q0 + i0 * 64));
                for (int j = 0; j < NR; j++) acc[j] = dp_m32(acc[j], w, y[j][b].qs + 4*i0);
            }
            const __m512 dw = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) bp[b].d));
            for (int j = 0; j < NR; j++) accf[j] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[j]), _mm512_mul_ps(dw, _mm512_set1_ps(y[j][b].yd)), accf[j]);
        }
        for (int j = 0; j < NR; j++) _mm512_storeu_ps(outs[j] + g * 16, accf[j]);
    }
}
// ---- MXFP4 ----
#define MX_LUT \
    alignas(16) uint8_t lut16[16]; for (int i = 0; i < 16; i++) lut16[i] = (uint8_t) ((int) kvalues_mxfp4[i] + 12); \
    const __m512i lut = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *) lut16)); const __m512i m4 = _mm512_set1_epi8(0x0F);
template <int NR> __attribute__((noinline)) static void mx_ref(int nb, float * const * outs, const block_q8_0 * const * y, const block_mxfp4_x16 * vxb, int nc, const int32_t * const * ysum, const float * const * yd) {
    MX_LUT
    for (int g = 0; g < nc / 16; g++) {
        const block_mxfp4_x16 * bp = vxb + (size_t) g * nb;
        __m512 accf[NR];
        for (int j = 0; j < NR; j++) accf[j] = _mm512_setzero_ps();
        for (int b = 0; b < nb; b++) {
            __m512i acc[NR];
            for (int j = 0; j < NR; j++) acc[j] = _mm512_setzero_si512();
            const uint8_t * q0 = bp[b].qs;
            for (int i0 = 0; i0 < 4; i0++) {
                const __m512i w  = _mm512_loadu_si512((const void *)(q0 + i0 * 64));
                const __m512i lo = _mm512_shuffle_epi8(lut, _mm512_and_si512(w, m4));
                const __m512i hi = _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16(w, 4), m4));
                for (int j = 0; j < NR; j++) { acc[j] = dp_m32(acc[j], lo, y[j][b].qs + 4*i0); acc[j] = dp_m32(acc[j], hi, y[j][b].qs + 16 + 4*i0); }
            }
            const __m512 scv = mxfp4_scales(bp[b].e);
            for (int j = 0; j < NR; j++) {
                const __m512i a = _mm512_sub_epi32(acc[j], _mm512_set1_epi32(12 * ysum[j][b]));
                accf[j] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(a), _mm512_mul_ps(scv, _mm512_set1_ps(yd[j][b])), accf[j]);
            }
        }
        for (int j = 0; j < NR; j++) _mm512_storeu_ps(outs[j] + g * 16, accf[j]);
    }
}
template <int NR> __attribute__((noinline)) static void mx_B(int nb, float * const * outs, const block_q8b * const * y, const block_mxfp4_x16 * vxb, int nc) {
    MX_LUT
    for (int g = 0; g < nc / 16; g++) {
        const block_mxfp4_x16 * bp = vxb + (size_t) g * nb;
        __m512 accf[NR];
        for (int j = 0; j < NR; j++) accf[j] = _mm512_setzero_ps();
        for (int b = 0; b < nb; b++) {
            __m512i acc[NR];
            for (int j = 0; j < NR; j++) acc[j] = _mm512_set1_epi32(y[j][b].bias);
            const uint8_t * q0 = bp[b].qs;
            for (int i0 = 0; i0 < 4; i0++) {
                const __m512i w  = _mm512_loadu_si512((const void *)(q0 + i0 * 64));
                const __m512i lo = _mm512_shuffle_epi8(lut, _mm512_and_si512(w, m4));
                const __m512i hi = _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16(w, 4), m4));
                for (int j = 0; j < NR; j++) { acc[j] = dp_m32(acc[j], lo, y[j][b].qs + 4*i0); acc[j] = dp_m32(acc[j], hi, y[j][b].qs + 16 + 4*i0); }
            }
            const __m512 scv = mxfp4_scales(bp[b].e);
            for (int j = 0; j < NR; j++) accf[j] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[j]), _mm512_mul_ps(scv, _mm512_set1_ps(y[j][b].yd)), accf[j]);
        }
        for (int j = 0; j < NR; j++) _mm512_storeu_ps(outs[j] + g * 16, accf[j]);
    }
}


// ---- R2: two 16-row groups per pass; each activation dword is broadcast ONCE into a register and feeds 2 dpbusd
//      (register form, both vector ports); accumulators start at the pre-multiplied bias (no vpsubd). Per (row, col)
//      the float ops are the shipped kernel's, in the same order: bit-identical.
template <int NR> __attribute__((noinline)) static void q8_R2(int nb, float * const * outs, const block_q8_0 * const * y, const block_q8_0_x16 * vxb, int nc, const int32_t * const * ybias, const float * const * yd) {
    int g = 0;
    for (; g + 2 <= nc / 16; g += 2) {
        const block_q8_0_x16 * bp0 = vxb + (size_t) g * nb;
        const block_q8_0_x16 * bp1 = bp0 + nb;
        __m512 f0[NR], f1[NR];
        for (int j = 0; j < NR; j++) { f0[j] = _mm512_setzero_ps(); f1[j] = _mm512_setzero_ps(); }
        for (int b = 0; b < nb; b++) {
            __m512i a0[NR], a1[NR];
            for (int j = 0; j < NR; j++) { a0[j] = _mm512_set1_epi32(ybias[j][b]); a1[j] = a0[j]; }
            const uint8_t * q0 = bp0[b].qs; const uint8_t * q1 = bp1[b].qs;
            for (int i0 = 0; i0 < 8; i0++) {
                const __m512i w0 = _mm512_loadu_si512((const void *)(q0 + i0 * 64));
                const __m512i w1 = _mm512_loadu_si512((const void *)(q1 + i0 * 64));
                for (int j = 0; j < NR; j++) {
                    const __m512i yb = _mm512_set1_epi32(*(const int32_t *) (y[j][b].qs + 4*i0));
                    a0[j] = _mm512_dpbusd_epi32(a0[j], w0, yb);
                    a1[j] = _mm512_dpbusd_epi32(a1[j], w1, yb);
                }
            }
            const __m512 dw0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) bp0[b].d));
            const __m512 dw1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) bp1[b].d));
            for (int j = 0; j < NR; j++) {
                const __m512 ydj = _mm512_set1_ps(yd[j][b]);
                f0[j] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(a0[j]), _mm512_mul_ps(dw0, ydj), f0[j]);
                f1[j] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(a1[j]), _mm512_mul_ps(dw1, ydj), f1[j]);
            }
        }
        for (int j = 0; j < NR; j++) { _mm512_storeu_ps(outs[j] + g * 16, f0[j]); _mm512_storeu_ps(outs[j] + g * 16 + 16, f1[j]); }
    }
    for (; g < nc / 16; g++) {   // odd tail group: shipped single-group form
        const block_q8_0_x16 * bp = vxb + (size_t) g * nb;
        __m512 accf[NR];
        for (int j = 0; j < NR; j++) accf[j] = _mm512_setzero_ps();
        for (int b = 0; b < nb; b++) {
            __m512i acc[NR];
            for (int j = 0; j < NR; j++) acc[j] = _mm512_set1_epi32(ybias[j][b]);
            for (int i0 = 0; i0 < 8; i0++) {
                const __m512i w = _mm512_loadu_si512((const void *)(bp[b].qs + i0 * 64));
                for (int j = 0; j < NR; j++) acc[j] = _mm512_dpbusd_epi32(acc[j], w, _mm512_set1_epi32(*(const int32_t *) (y[j][b].qs + 4*i0)));
            }
            const __m512 dw = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) bp[b].d));
            for (int j = 0; j < NR; j++) accf[j] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[j]), _mm512_mul_ps(dw, _mm512_set1_ps(yd[j][b])), accf[j]);
        }
        for (int j = 0; j < NR; j++) _mm512_storeu_ps(outs[j] + g * 16, accf[j]);
    }
}
template <int NR> __attribute__((noinline)) static void mx_R2(int nb, float * const * outs, const block_q8_0 * const * y, const block_mxfp4_x16 * vxb, int nc, const int32_t * const * ybias, const float * const * yd) {
    MX_LUT
    for (int g = 0; g + 2 <= nc / 16; g += 2) {
        const block_mxfp4_x16 * bp0 = vxb + (size_t) g * nb;
        const block_mxfp4_x16 * bp1 = bp0 + nb;
        __m512 f0[NR], f1[NR];
        for (int j = 0; j < NR; j++) { f0[j] = _mm512_setzero_ps(); f1[j] = _mm512_setzero_ps(); }
        for (int b = 0; b < nb; b++) {
            __m512i a0[NR], a1[NR];
            for (int j = 0; j < NR; j++) { a0[j] = _mm512_set1_epi32(ybias[j][b]); a1[j] = a0[j]; }
            for (int i0 = 0; i0 < 4; i0++) {
                const __m512i w0 = _mm512_loadu_si512((const void *)(bp0[b].qs + i0 * 64));
                const __m512i w1 = _mm512_loadu_si512((const void *)(bp1[b].qs + i0 * 64));
                const __m512i lo0 = _mm512_shuffle_epi8(lut, _mm512_and_si512(w0, m4));
                const __m512i hi0 = _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16(w0, 4), m4));
                const __m512i lo1 = _mm512_shuffle_epi8(lut, _mm512_and_si512(w1, m4));
                const __m512i hi1 = _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16(w1, 4), m4));
                for (int j = 0; j < NR; j++) {
                    const __m512i ylo = _mm512_set1_epi32(*(const int32_t *) (y[j][b].qs + 4*i0));
                    const __m512i yhi = _mm512_set1_epi32(*(const int32_t *) (y[j][b].qs + 16 + 4*i0));
                    a0[j] = _mm512_dpbusd_epi32(a0[j], lo0, ylo); a1[j] = _mm512_dpbusd_epi32(a1[j], lo1, ylo);
                    a0[j] = _mm512_dpbusd_epi32(a0[j], hi0, yhi); a1[j] = _mm512_dpbusd_epi32(a1[j], hi1, yhi);
                }
            }
            const __m512 sc0 = mxfp4_scales(bp0[b].e), sc1 = mxfp4_scales(bp1[b].e);
            for (int j = 0; j < NR; j++) {
                const __m512 ydj = _mm512_set1_ps(yd[j][b]);
                f0[j] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(a0[j]), _mm512_mul_ps(sc0, ydj), f0[j]);
                f1[j] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(a1[j]), _mm512_mul_ps(sc1, ydj), f1[j]);
            }
        }
        for (int j = 0; j < NR; j++) { _mm512_storeu_ps(outs[j] + g * 16, f0[j]); _mm512_storeu_ps(outs[j] + g * 16 + 16, f1[j]); }
    }
}

// ---- R3: three 16-row groups per pass (each broadcast feeds 3 dpbusd) ----
template <int NR> __attribute__((noinline)) static void q8_R3(int nb, float * const * outs, const block_q8_0 * const * y, const block_q8_0_x16 * vxb, int nc, const int32_t * const * ybias, const float * const * yd) {
    int g = 0;
    for (; g + 3 <= nc / 16; g += 3) {
        const block_q8_0_x16 * bp[3] = { vxb + (size_t) g * nb, vxb + (size_t) (g + 1) * nb, vxb + (size_t) (g + 2) * nb };
        __m512 f[3][NR];
        for (int r = 0; r < 3; r++) for (int j = 0; j < NR; j++) f[r][j] = _mm512_setzero_ps();
        for (int b = 0; b < nb; b++) {
            __m512i a[3][NR];
            for (int j = 0; j < NR; j++) { a[0][j] = _mm512_set1_epi32(ybias[j][b]); a[1][j] = a[0][j]; a[2][j] = a[0][j]; }
            for (int i0 = 0; i0 < 8; i0++) {
                __m512i w[3];
                for (int r = 0; r < 3; r++) w[r] = _mm512_loadu_si512((const void *)(bp[r][b].qs + i0 * 64));
                for (int j = 0; j < NR; j++) {
                    const __m512i yb = _mm512_set1_epi32(*(const int32_t *) (y[j][b].qs + 4*i0));
                    for (int r = 0; r < 3; r++) a[r][j] = _mm512_dpbusd_epi32(a[r][j], w[r], yb);
                }
            }
            __m512 dw[3];
            for (int r = 0; r < 3; r++) dw[r] = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) bp[r][b].d));
            for (int j = 0; j < NR; j++) {
                const __m512 ydj = _mm512_set1_ps(yd[j][b]);
                for (int r = 0; r < 3; r++) f[r][j] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(a[r][j]), _mm512_mul_ps(dw[r], ydj), f[r][j]);
            }
        }
        for (int r = 0; r < 3; r++) for (int j = 0; j < NR; j++) _mm512_storeu_ps(outs[j] + (g + r) * 16, f[r][j]);
    }
    if (g < nc / 16) {   // remaining 1-2 groups: R2 on a sub-range
        float * o2[16]; for (int j = 0; j < NR; j++) o2[j] = outs[j] + g * 16;
        q8_R2<NR>(nb, o2, y, vxb + (size_t) g * nb, nc - g * 16, ybias, yd);
    }
}
template <int NR> __attribute__((noinline)) static void mx_R3(int nb, float * const * outs, const block_q8_0 * const * y, const block_mxfp4_x16 * vxb, int nc, const int32_t * const * ybias, const float * const * yd) {
    MX_LUT
    int g = 0;
    for (; g + 3 <= nc / 16; g += 3) {
        const block_mxfp4_x16 * bp[3] = { vxb + (size_t) g * nb, vxb + (size_t) (g + 1) * nb, vxb + (size_t) (g + 2) * nb };
        __m512 f[3][NR];
        for (int r = 0; r < 3; r++) for (int j = 0; j < NR; j++) f[r][j] = _mm512_setzero_ps();
        for (int b = 0; b < nb; b++) {
            __m512i a[3][NR];
            for (int j = 0; j < NR; j++) { a[0][j] = _mm512_set1_epi32(ybias[j][b]); a[1][j] = a[0][j]; a[2][j] = a[0][j]; }
            for (int i0 = 0; i0 < 4; i0++) {
                __m512i lo[3], hi[3];
                for (int r = 0; r < 3; r++) {
                    const __m512i w = _mm512_loadu_si512((const void *)(bp[r][b].qs + i0 * 64));
                    lo[r] = _mm512_shuffle_epi8(lut, _mm512_and_si512(w, m4));
                    hi[r] = _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16(w, 4), m4));
                }
                for (int j = 0; j < NR; j++) {
                    const __m512i ylo = _mm512_set1_epi32(*(const int32_t *) (y[j][b].qs + 4*i0));
                    const __m512i yhi = _mm512_set1_epi32(*(const int32_t *) (y[j][b].qs + 16 + 4*i0));
                    for (int r = 0; r < 3; r++) { a[r][j] = _mm512_dpbusd_epi32(a[r][j], lo[r], ylo); a[r][j] = _mm512_dpbusd_epi32(a[r][j], hi[r], yhi); }
                }
            }
            __m512 sc[3];
            for (int r = 0; r < 3; r++) sc[r] = mxfp4_scales(bp[r][b].e);
            for (int j = 0; j < NR; j++) {
                const __m512 ydj = _mm512_set1_ps(yd[j][b]);
                for (int r = 0; r < 3; r++) f[r][j] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(a[r][j]), _mm512_mul_ps(sc[r], ydj), f[r][j]);
            }
        }
        for (int r = 0; r < 3; r++) for (int j = 0; j < NR; j++) _mm512_storeu_ps(outs[j] + (g + r) * 16, f[r][j]);
    }
    if (g < nc / 16) {
        float * o2[16]; for (int j = 0; j < NR; j++) o2[j] = outs[j] + g * 16;
        mx_R2<NR>(nb, o2, y, vxb + (size_t) g * nb, nc - g * 16, ybias, yd);
    }
}

template <template <int> class K, typename... A> static void disp(int nr, A... a) {
    switch (nr) { case 1: K<1>::run(a...); break; case 2: K<2>::run(a...); break; case 3: K<3>::run(a...); break;
                  case 4: K<4>::run(a...); break; case 5: K<5>::run(a...); break; case 6: K<6>::run(a...); break;
                  case 8: K<8>::run(a...); break; case 9: K<9>::run(a...); break; case 10: K<10>::run(a...); break;
                  case 11: K<11>::run(a...); break; case 12: K<12>::run(a...); break; default: K<7>::run(a...); break; }
}
#define WRAP(name) template <int NR> struct W_##name { template <typename... A> static void run(A... a) { name<NR>(a...); } };
WRAP(q8_ref) WRAP(q8_A) WRAP(q8_B) WRAP(mx_ref) WRAP(mx_B) WRAP(q8_R2) WRAP(mx_R2) WRAP(q8_R3) WRAP(mx_R3)

int main(int argc, char ** argv) {
    const bool mx = strcmp(argv[1], "mx") == 0;
    const int M = atoi(argv[2]), K = atoi(argv[3]), N = atoi(argv[4]), NR = atoi(argv[5]), CH = atoi(argv[6]);
    const int nthr = atoi(argv[7]), iters = atoi(argv[8]);
    const int nb = K / 32; const int mult = mx ? 12 : 128;
    std::mt19937 rng(7);
    std::vector<block_q8_0_x16> Wq; std::vector<block_mxfp4_x16> Wm;
    if (mx) { Wm.resize((size_t) (M / 16) * nb); for (auto & b : Wm) { for (auto & e : b.e) e = (uint8_t) (120 + rng() % 10); for (auto & q : b.qs) q = (uint8_t) rng(); } }
    else    { Wq.resize((size_t) (M / 16) * nb); for (auto & b : Wq) { for (auto & d : b.d) d = f16_bits(0.01f + 0.001f * (rng() % 7)); for (auto & q : b.qs) q = (uint8_t) rng(); } }
    std::vector<block_q8_0> X((size_t) N * nb);
    for (auto & b : X) { b.d = f16_bits(0.02f + 0.001f * (rng() % 11)); for (auto & q : b.qs) q = (int8_t) (rng() % 255 - 127); }
    std::vector<int32_t> ysum((size_t) N * nb), ybias((size_t) N * nb); std::vector<float> yd((size_t) N * nb);
    std::vector<block_q8b> XB((size_t) N * nb);
    for (size_t i = 0; i < X.size(); i++) {
        int s = 0; for (int k = 0; k < 32; k++) s += X[i].qs[k];
        ysum[i] = s; ybias[i] = -mult * s; yd[i] = f16_to_f32(X[i].d);
        memcpy(XB[i].qs, X[i].qs, 32); XB[i].bias = -mult * s; XB[i].yd = yd[i];
    }
    const int nvar = mx ? 4 : 5; const int NR3 = getenv("NR3") ? atoi(getenv("NR3")) : 4; const int NR2 = getenv("NR2") ? atoi(getenv("NR2")) : 6;
    std::vector<std::vector<float>> C(nvar, std::vector<float>((size_t) N * M, 0.0f));
    const int nchunk = M / CH;
    std::vector<double> best(nvar, 1e30);
    for (int it = 0; it < iters + 1; it++) {
        for (int v = 0; v < nvar; v++) {
            const double t0 = now_ms();
            #pragma omp parallel for schedule(dynamic, 1) num_threads(nthr)
            for (int c = 0; c < nchunk; c++) {
                const int r0 = c * CH;
                const bool r2 = (!mx && v == 3) || (mx && v == 2);
                const bool r3 = (!mx && v == 4) || (mx && v == 3);
                const int nrmax = r3 ? NR3 : r2 ? NR2 : NR;
                for (int j0 = 0; j0 < N; ) {
                    const int nr = std::min(nrmax, N - j0);
                    float * outs[16]; const block_q8_0 * ys[16]; const int32_t * s1[16]; const int32_t * s2[16]; const float * ds[16]; const block_q8b * yb[16];
                    for (int j = 0; j < nr; j++) {
                        outs[j] = &C[v][(size_t) (j0 + j) * M + r0]; ys[j] = &X[(size_t) (j0 + j) * nb]; yb[j] = &XB[(size_t) (j0 + j) * nb];
                        s1[j] = &ysum[(size_t) (j0 + j) * nb]; s2[j] = &ybias[(size_t) (j0 + j) * nb]; ds[j] = &yd[(size_t) (j0 + j) * nb];
                    }
                    if (!mx) {
                        const block_q8_0_x16 * w = &Wq[(size_t) (r0 / 16) * nb];
                        if (v == 0) disp<W_q8_ref>(nr, nb, (float * const *) outs, (const block_q8_0 * const *) ys, w, CH, (const int32_t * const *) s1, (const float * const *) ds);
                        if (v == 1) disp<W_q8_A>(nr, nb, (float * const *) outs, (const block_q8_0 * const *) ys, w, CH, (const int32_t * const *) s2, (const float * const *) ds);
                        if (v == 2) disp<W_q8_B>(nr, nb, (float * const *) outs, (const block_q8b * const *) yb, w, CH);
                        if (v == 4) disp<W_q8_R3>(nr, nb, (float * const *) outs, (const block_q8_0 * const *) ys, w, CH, (const int32_t * const *) s2, (const float * const *) ds);
                        if (v == 3) disp<W_q8_R2>(nr, nb, (float * const *) outs, (const block_q8_0 * const *) ys, w, CH, (const int32_t * const *) s2, (const float * const *) ds);
                    } else {
                        const block_mxfp4_x16 * w = &Wm[(size_t) (r0 / 16) * nb];
                        if (v == 0) disp<W_mx_ref>(nr, nb, (float * const *) outs, (const block_q8_0 * const *) ys, w, CH, (const int32_t * const *) s1, (const float * const *) ds);
                        if (v == 1) disp<W_mx_B>(nr, nb, (float * const *) outs, (const block_q8b * const *) yb, w, CH);
                        if (v == 3) disp<W_mx_R3>(nr, nb, (float * const *) outs, (const block_q8_0 * const *) ys, w, CH, (const int32_t * const *) s2, (const float * const *) ds);
                        if (v == 2) disp<W_mx_R2>(nr, nb, (float * const *) outs, (const block_q8_0 * const *) ys, w, CH, (const int32_t * const *) s2, (const float * const *) ds);
                    }
                    j0 += nr;
                }
            }
            if (it > 0) best[v] = std::min(best[v], now_ms() - t0);
        }
    }
    const char * names[5] = {"ref", mx ? "B" : "A", mx ? "R2" : "B", mx ? "R3" : "R2", "R3"};
    for (int v = 0; v < nvar; v++) {
        double maxd = 0; for (size_t i = 0; i < C[0].size(); i++) maxd = std::max(maxd, (double) std::fabs(C[v][i] - C[0][i]));
        printf("%s %-3s M %d K %d N %d NR %d/%d CH %d: min %.3f ms  %.0f GOPS  max|d vs ref| %g\n", mx ? "mx" : "q8", names[v], M, K, N, NR, NR2, CH, best[v], 2.0 * M * N * K / (best[v] * 1e6), maxd);
    }
}
