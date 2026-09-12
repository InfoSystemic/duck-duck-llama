// Standalone micro-benchmark: 16-row interleaved AVX-512 VNNI Q4_K gemv vs ggml baselines.
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
void ggml_vec_dot_q4_K_q8_K(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void ggml_gemv_q4_K_8x8_q8_K(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc);
}

static inline float fp16_to_fp32(ggml_half h) { return ggml_fp16_to_fp32(h); }

// ---------------- 16-row interleaved layout ----------------
struct block_q4_K_x16 {
    ggml_half d[16];
    ggml_half dmin[16];
    uint8_t scales[8][16];
    uint8_t mins[8][16];
    uint8_t qs[2048];   // [j:4][i0:8][row:16][t:4]  <- native byte 32*j + 4*i0 + t of row
};
static_assert(sizeof(block_q4_K_x16) == 64 + 256 + 2048, "size");

static inline void get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else { *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4); *m = (q[j+4] >> 4) | ((q[j-0] >> 6) << 4); }
}

// in: 16 rows of nb blocks each (row-major native Q4_K); out: nb blocks of x16
static void repack_q4_K_x16(const block_q4_K * in, int nb, block_q4_K_x16 * out) {
    for (int b = 0; b < nb; b++) {
        block_q4_K_x16 & o = out[b];
        for (int r = 0; r < 16; r++) {
            const block_q4_K & x = in[r * nb + b];
            o.d[r] = x.GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
            o.dmin[r] = x.GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
            for (int sb = 0; sb < 8; sb++) {
                uint8_t sc, mn; get_scale_min_k4(sb, x.scales, &sc, &mn);
                o.scales[sb][r] = sc; o.mins[sb][r] = mn;
            }
            for (int j = 0; j < 4; j++) for (int i0 = 0; i0 < 8; i0++) for (int t = 0; t < 4; t++) {
                o.qs[((j*8 + i0)*16 + r)*4 + t] = x.qs[32*j + 4*i0 + t];
            }
        }
    }
}

// y: one q8_K activation row (nb blocks). x: nc/16 groups of nb blocks. s: nc outputs.
static void gemv_q4_K_x16_q8_K(int n, float * s, const block_q4_K_x16 * vx, const block_q8_K * vy, int nc) {
    const int nb = n / QK_K;
    const __m512i m4 = _mm512_set1_epi8(0x0F);
    for (int g = 0; g < nc / 16; g++) {
        const block_q4_K_x16 * bp = vx + (size_t) g * nb;
        __m512 accf = _mm512_setzero_ps();
        for (int b = 0; b < nb; b++) {
            const block_q8_K * a = vy + b;
            const uint8_t * qs = bp[b].qs;
            __m512i iacc = _mm512_setzero_si512();
            __m512i imin = _mm512_setzero_si512();
            for (int j = 0; j < 4; j++) {
                __m512i accA0 = _mm512_setzero_si512(), accA1 = _mm512_setzero_si512();
                __m512i accB0 = _mm512_setzero_si512(), accB1 = _mm512_setzero_si512();
                const uint8_t * qj = qs + j * 512;
                const int8_t * ya = a->qs + 64 * j;
                const int8_t * yb = ya + 32;
#define STEP(i0, AA, BB) { \
                const __m512i v  = _mm512_loadu_si512((const void *)(qj + (i0) * 64)); \
                const __m512i lo = _mm512_and_si512(v, m4); \
                const __m512i hi = _mm512_and_si512(_mm512_srli_epi16(v, 4), m4); \
                AA = _mm512_dpbusd_epi32(AA, lo, _mm512_set1_epi32(*(const int32_t *)(ya + 4*(i0)))); \
                BB = _mm512_dpbusd_epi32(BB, hi, _mm512_set1_epi32(*(const int32_t *)(yb + 4*(i0)))); }
                STEP(0, accA0, accB0) STEP(1, accA1, accB1) STEP(2, accA0, accB0) STEP(3, accA1, accB1)
                STEP(4, accA0, accB0) STEP(5, accA1, accB1) STEP(6, accA0, accB0) STEP(7, accA1, accB1)
#undef STEP
                const __m512i accA = _mm512_add_epi32(accA0, accA1);
                const __m512i accB = _mm512_add_epi32(accB0, accB1);
                const __m512i scA = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *) bp[b].scales[2*j]));
                const __m512i scB = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *) bp[b].scales[2*j+1]));
                iacc = _mm512_add_epi32(iacc, _mm512_mullo_epi32(accA, scA));
                iacc = _mm512_add_epi32(iacc, _mm512_mullo_epi32(accB, scB));
                const __m512i mA = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *) bp[b].mins[2*j]));
                const __m512i mB = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *) bp[b].mins[2*j+1]));
                const int32_t bsA = (int32_t) a->bsums[4*j] + a->bsums[4*j+1];
                const int32_t bsB = (int32_t) a->bsums[4*j+2] + a->bsums[4*j+3];
                imin = _mm512_add_epi32(imin, _mm512_mullo_epi32(mA, _mm512_set1_epi32(bsA)));
                imin = _mm512_add_epi32(imin, _mm512_mullo_epi32(mB, _mm512_set1_epi32(bsB)));
            }
            const __m512 ad = _mm512_set1_ps(a->d);
            const __m512 dv = _mm512_mul_ps(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) bp[b].d)), ad);
            const __m512 mv = _mm512_mul_ps(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) bp[b].dmin)), ad);
            accf = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc), dv, accf);
            accf = _mm512_fnmadd_ps(_mm512_cvtepi32_ps(imin), mv, accf);
        }
        _mm512_storeu_ps(s + g * 16, accf);
    }
}

// ---------------- upstream 8x8 layout (copy of make_block_q4_Kx8, interleave 8) ----------------
struct block_q4_Kx8 { ggml_half d[8]; ggml_half dmin[8]; uint8_t scales[96]; uint8_t qs[1024]; };
static block_q4_Kx8 make_block_q4_Kx8(const block_q4_K * in, unsigned blck) {
    block_q4_Kx8 out;
    for (int i = 0; i < 8; i++) out.d[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
    for (int i = 0; i < 8; i++) out.dmin[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
    const int end = QK_K * 4 / blck;
    for (int i = 0; i < end; ++i) {
        int src_id = i % 8; int src_offset = (i / 8) * blck; int dst_offset = i * blck;
        uint64_t elems; memcpy(&elems, &in[src_id].qs[src_offset], blck); memcpy(&out.qs[dst_offset], &elems, blck);
    }
    uint8_t s[8], m[8];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) { s[j] = in[j].scales[i] & 63; m[j] = in[j].scales[i + 4] & 63; }
        out.scales[i*12+0]=(s[0]&63)+((s[4]&48)<<2); out.scales[i*12+1]=(s[1]&63)+((s[5]&48)<<2);
        out.scales[i*12+2]=(s[2]&63)+((s[6]&48)<<2); out.scales[i*12+3]=(s[3]&63)+((s[7]&48)<<2);
        out.scales[i*12+4]=(m[0]&63)+((m[4]&48)<<2); out.scales[i*12+5]=(m[1]&63)+((m[5]&48)<<2);
        out.scales[i*12+6]=(m[2]&63)+((m[6]&48)<<2); out.scales[i*12+7]=(m[3]&63)+((m[7]&48)<<2);
        out.scales[i*12+8]=(s[4]&15)+((m[4]&15)<<4); out.scales[i*12+9]=(s[5]&15)+((m[5]&15)<<4);
        out.scales[i*12+10]=(s[6]&15)+((m[6]&15)<<4); out.scales[i*12+11]=(s[7]&15)+((m[7]&15)<<4);
    }
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            s[j] = ((in[j].scales[i] & 192) >> 2) | (in[j].scales[i+8] & 15);
            m[j] = ((in[j].scales[i + 4] & 192) >> 2) | ((in[j].scales[i+8] & 240) >> 4);
        }
        out.scales[i*12+48]=(s[0]&63)+((s[4]&48)<<2); out.scales[i*12+49]=(s[1]&63)+((s[5]&48)<<2);
        out.scales[i*12+50]=(s[2]&63)+((s[6]&48)<<2); out.scales[i*12+51]=(s[3]&63)+((s[7]&48)<<2);
        out.scales[i*12+52]=(m[0]&63)+((m[4]&48)<<2); out.scales[i*12+53]=(m[1]&63)+((m[5]&48)<<2);
        out.scales[i*12+54]=(m[2]&63)+((m[6]&48)<<2); out.scales[i*12+55]=(m[3]&63)+((m[7]&48)<<2);
        out.scales[i*12+56]=(s[4]&15)+((m[4]&15)<<4); out.scales[i*12+57]=(s[5]&15)+((m[5]&15)<<4);
        out.scales[i*12+58]=(s[6]&15)+((m[6]&15)<<4); out.scales[i*12+59]=(s[7]&15)+((m[7]&15)<<4);
    }
    return out;
}
static void repack_q4_K_8x8(const block_q4_K * in, int nb, block_q4_Kx8 * out) { // in: 8 rows
    block_q4_K tmp[8];
    for (int b = 0; b < nb; b++) { for (int r = 0; r < 8; r++) tmp[r] = in[r*nb + b]; out[b] = make_block_q4_Kx8(tmp, 8); }
}

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char ** argv) {
    const int k = argc > 1 ? atoi(argv[1]) : 6144;
    const int rows_per_thread = argc > 2 ? atoi(argv[2]) : 4096;  // per thread rows (streaming set)
    const int nthreads = omp_get_max_threads();
    const int nrows = rows_per_thread * nthreads;
    const int nb = k / QK_K;
    printf("k=%d rows=%d threads=%d  native bytes/row=%zu  x16 bytes/row=%zu\n", k, nrows, nthreads, (size_t) nb * sizeof(block_q4_K), (size_t) nb * sizeof(block_q4_K_x16) / 16);

    // random weights + activation
    std::mt19937 rng(42); std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> xf((size_t) k);
    std::vector<block_q4_K> native((size_t) nrows * nb);
    for (int r = 0; r < nrows; r++) {
        for (int i = 0; i < k; i++) xf[i] = nd(rng);
        quantize_row_q4_K_ref(xf.data(), native.data() + (size_t) r * nb, k);
    }
    std::vector<float> yf(k); for (int i = 0; i < k; i++) yf[i] = nd(rng);
    std::vector<block_q8_K> y8(nb); quantize_row_q8_K(yf.data(), y8.data(), k);

    // reference
    std::vector<float> ref(nrows), out16(nrows), out8(nrows);
    for (int r = 0; r < nrows; r++) ggml_vec_dot_q4_K_q8_K(k, &ref[r], 0, native.data() + (size_t) r * nb, 0, y8.data(), 0, 1);

    // repacks (per thread row range so each thread streams its own contiguous region)
    std::vector<block_q4_K_x16> w16((size_t) nrows / 16 * nb);
    std::vector<block_q4_Kx8>  w8((size_t) nrows / 8 * nb);
    #pragma omp parallel for
    for (int g = 0; g < nrows / 16; g++) repack_q4_K_x16(native.data() + (size_t) g * 16 * nb, nb, w16.data() + (size_t) g * nb);
    #pragma omp parallel for
    for (int g = 0; g < nrows / 8; g++) repack_q4_K_8x8(native.data() + (size_t) g * 8 * nb, nb, w8.data() + (size_t) g * nb);

    // correctness
    gemv_q4_K_x16_q8_K(k, out16.data(), w16.data(), y8.data(), nrows);
    ggml_gemv_q4_K_8x8_q8_K(k, out8.data(), nrows, w8.data(), y8.data(), 1, nrows);
    std::vector<float> wf(k), yq(k); dequantize_row_q8_K(y8.data(), yq.data(), k);
    std::vector<float> fref(nrows);
    for (int r = 0; r < nrows; r++) { dequantize_row_q4_K(native.data() + (size_t) r * nb, wf.data(), k); double acc = 0; for (int i = 0; i < k; i++) acc += (double) wf[i] * yq[i]; fref[r] = (float) acc; }
    printf("row0: vecdot=%.4f x16=%.4f 8x8=%.4f float=%.4f\n", ref[0], out16[0], out8[0], fref[0]);
    double maxef16 = 0, maxef8 = 0; for (int r = 0; r < nrows; r++) { maxef16 = fmax(maxef16, fabs(out16[r] - fref[r])); maxef8 = fmax(maxef8, fabs(out8[r] - fref[r])); }
    printf("vs float ref: max err x16=%.3e  8x8=%.3e\n", maxef16, maxef8);
    std::vector<float> wf(k), yq(k); dequantize_row_q8_K(y8.data(), yq.data(), k);
    std::vector<float> fref(nrows);
    for (int r = 0; r < nrows; r++) { dequantize_row_q4_K(native.data() + (size_t) r * nb, wf.data(), k); double acc = 0; for (int i = 0; i < k; i++) acc += (double) wf[i] * yq[i]; fref[r] = (float) acc; }
    printf("row0: vecdot=%.4f x16=%.4f 8x8=%.4f float=%.4f\n", ref[0], out16[0], out8[0], fref[0]);
    double maxef16 = 0, maxef8 = 0; for (int r = 0; r < nrows; r++) { maxef16 = fmax(maxef16, fabs(out16[r] - fref[r])); maxef8 = fmax(maxef8, fabs(out8[r] - fref[r])); }
    printf("vs float ref: max err x16=%.3e  8x8=%.3e\n", maxef16, maxef8);
    double maxe16 = 0, maxe8 = 0, maxref = 0;
    for (int r = 0; r < nrows; r++) { maxe16 = fmax(maxe16, fabs(out16[r] - ref[r])); maxe8 = fmax(maxe8, fabs(out8[r] - ref[r])); maxref = fmax(maxref, fabs(ref[r])); }
    printf("max|ref|=%.3f  max err x16=%.3e  max err 8x8=%.3e\n", maxref, maxe16, maxe8);

    // timing: each thread streams its own rows, repeat
    const int reps = argc > 3 ? atoi(argv[3]) : 10;
    const double bytes16 = (double) nrows / 16 * nb * sizeof(block_q4_K_x16);
    const double bytes8  = (double) nrows / 8 * nb * sizeof(block_q4_Kx8);
    const double bytesN  = (double) nrows * nb * sizeof(block_q4_K);
    auto bench = [&](const char * name, double bytes, auto fn) {
        double best = 1e9;
        for (int it = 0; it < reps; it++) {
            double t0 = now();
            #pragma omp parallel
            { int t = omp_get_thread_num(); fn(t); }
            double dt = now() - t0; if (dt < best) best = dt;
        }
        printf("%-28s best %.3f ms  -> %.1f GB/s aggregate (%.1f per thread)\n", name, best * 1e3, bytes / best / 1e9, bytes / best / 1e9 / nthreads);
    };
    bench("x16 VNNI gemv", bytes16, [&](int t) {
        gemv_q4_K_x16_q8_K(k, out16.data() + (size_t) t * rows_per_thread, w16.data() + (size_t) t * rows_per_thread / 16 * nb, y8.data(), rows_per_thread); });
    bench("upstream 8x8 AVX2 gemv", bytes8, [&](int t) {
        ggml_gemv_q4_K_8x8_q8_K(k, out8.data() + (size_t) t * rows_per_thread, nrows, w8.data() + (size_t) t * rows_per_thread / 8 * nb, y8.data(), 1, rows_per_thread); });
    bench("plain vec_dot AVX2", bytesN, [&](int t) {
        for (int r = t * rows_per_thread; r < (t + 1) * rows_per_thread; r++) ggml_vec_dot_q4_K_q8_K(k, &ref[r], 0, native.data() + (size_t) r * nb, 0, y8.data(), 0, 1); });
    // MoE-like chunked pattern: per thread, 16 calls of `chunk` rows, each from a different far-apart region
    {
        const int chunk = argc > 4 ? atoi(argv[4]) : 32; const int ncalls = 16;
        for (int variant = 0; variant < 2; variant++) {
            double best = 1e9;
            for (int it = 0; it < reps; it++) {
                double t0 = now();
                #pragma omp parallel
                { int t = omp_get_thread_num();
                  for (int c = 0; c < ncalls; c++) {
                      // region c: rows [c*nrows/ncalls, ...) ; this thread's slice inside region
                      int base = c * (nrows / ncalls) + t * chunk;
                      if (variant == 0) gemv_q4_K_x16_q8_K(k, out16.data() + base, w16.data() + (size_t) base / 16 * nb, y8.data(), chunk);
                      else ggml_gemv_q4_K_8x8_q8_K(k, out8.data() + base, nrows, w8.data() + (size_t) base / 8 * nb, y8.data(), 1, chunk);
                  } }
                double dt = now() - t0; if (dt < best) best = dt;
            }
            double bytes = (double) chunk * ncalls * nthreads * nb * (variant == 0 ? sizeof(block_q4_K_x16) / 16.0 : sizeof(block_q4_Kx8) / 8.0);
            printf("%-28s chunk=%d rows x %d calls: best %.3f ms -> %.1f GB/s aggregate\n", variant == 0 ? "x16 VNNI chunked" : "8x8 AVX2 chunked", chunk, ncalls, best * 1e3, bytes / best / 1e9);
        }
    }
    // MoE-like chunked pattern: per thread, 16 calls of `chunk` rows, each from a different far-apart region
    {
        const int chunk = argc > 4 ? atoi(argv[4]) : 32; const int ncalls = 16;
        for (int variant = 0; variant < 2; variant++) {
            double best = 1e9;
            for (int it = 0; it < reps; it++) {
                double t0 = now();
                #pragma omp parallel
                { int t = omp_get_thread_num();
                  for (int c = 0; c < ncalls; c++) {
                      int base = c * (nrows / ncalls) + t * chunk;
                      if (variant == 0) gemv_q4_K_x16_q8_K(k, out16.data() + base, w16.data() + (size_t) base / 16 * nb, y8.data(), chunk);
                      else ggml_gemv_q4_K_8x8_q8_K(k, out8.data() + base, nrows, w8.data() + (size_t) base / 8 * nb, y8.data(), 1, chunk);
                  } }
                double dt = now() - t0; if (dt < best) best = dt;
            }
            double bytes = (double) chunk * ncalls * nthreads * nb * (variant == 0 ? sizeof(block_q4_K_x16) / 16.0 : sizeof(block_q4_Kx8) / 8.0);
            printf("%-28s chunk=%d rows x %d calls: best %.3f ms -> %.1f GB/s aggregate\n", variant == 0 ? "x16 VNNI chunked" : "8x8 AVX2 chunked", chunk, ncalls, best * 1e3, bytes / best / 1e9);
        }
    }
    // L2-resident variant: small per-thread set (64 rows = 221KB) repeated
    {
        const int small = 64; double best = 1e9;
        for (int it = 0; it < reps; it++) {
            double t0 = now();
            #pragma omp parallel
            { int t = omp_get_thread_num(); for (int q = 0; q < 200; q++) gemv_q4_K_x16_q8_K(k, out16.data() + (size_t) t * rows_per_thread, w16.data() + (size_t) t * rows_per_thread / 16 * nb, y8.data(), small); }
            double dt = now() - t0; if (dt < best) best = dt;
        }
        double bytes = (double) small / 16 * nb * sizeof(block_q4_K_x16) * 200 * nthreads;
        printf("%-28s best %.3f ms  -> %.1f GB/s aggregate (%.1f per thread) [cache-resident compute bound]\n", "x16 VNNI gemv L2-resident", best * 1e3, bytes / best / 1e9, bytes / best / 1e9 / nthreads);
    }
    return 0;
}
