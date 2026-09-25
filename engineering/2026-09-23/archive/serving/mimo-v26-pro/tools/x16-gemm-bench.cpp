// x16-gemm-bench: time the x16 VNNI prompt-batch kernels on synthetic data at MiMo-V2.6-Pro's per-node shapes, called
// the way repack.cpp's batch paths call them (per-matmul activation prep, weight chunks, <= NRMAX columns per call),
// and check every output against the single-column GEMV.
//
//   dense: q8_0 weights [M x K], N tokens, weight chunks of CH rows (wqkv per node: M 6784, K 6144, N 512)
//   (activation prep = ggml_x16_q8_0_col_prep_bias: per-block bias -mult*sum, mult 128 for Q8_0 / 12 for MXFP4 weights;
//   NRMAX < 0 = even split at most |NRMAX| wide, as repack.cpp does: 5 / 6 for >= 32-row tiles, 10 / 11 for 16 rows)
//   moe:   E experts of mxfp4 [M x K] (gate per node: M 512), each seeing TOK tokens, 64-row tiles like the MoE path
//
// usage: x16-gemm-bench dense <M> <K> <N> <NRMAX> <CH> <threads> <iters>
//        x16-gemm-bench moe   <E> <M> <K> <TOK> <NRMAX> <threads> <iters>
#include <omp.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <algorithm>

extern "C" {
void ggml_gemm_ptrs_q8_0_x16_q8_0(int n, int nr, float * const * outs, const void * const * ys, const void * vx, int nc,
                                  const int32_t * const * ysums, const float * const * yds);
void ggml_gemm_ptrs_mxfp4_x16_q8_0(int n, int nr, float * const * outs, const void * const * ys, const void * vx, int nc,
                                   const int32_t * const * ysums, const float * const * yds);
void ggml_x16_q8_0_col_prep_bias(const void * y, int nb, int32_t mult, int32_t * ybias, float * yd);
void ggml_gemv_q8_0_x16_q8_0(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc);
void ggml_gemv_mxfp4_x16_q8_0(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc);
}

struct block_q8_0     { uint16_t d; int8_t qs[32]; };
struct block_q8_0_x16 { uint16_t d[16]; uint8_t qs[512]; };
struct block_mxfp4_x16 { uint8_t e[16]; uint8_t qs[256]; };
static_assert(sizeof(block_q8_0) == 34, "");
static_assert(sizeof(block_q8_0_x16) == 544, "");
static_assert(sizeof(block_mxfp4_x16) == 272, "");

static uint16_t f16_bits(float f) {   // positive normal values only (all this bench needs)
    uint32_t x; memcpy(&x, &f, 4);
    const uint32_t e = ((x >> 23) & 0xff) - 127 + 15, m = (x >> 13) & 0x3ff;
    return (uint16_t) ((e << 10) | m);
}

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "see header\n"); return 1; }
    const bool moe = strcmp(argv[1], "moe") == 0;
    std::mt19937 rng(7);
    if (!moe) {
        const int M = atoi(argv[2]), K = atoi(argv[3]), N = atoi(argv[4]), NRMAX = atoi(argv[5]), CH = atoi(argv[6]);
        const int nthr = atoi(argv[7]), iters = atoi(argv[8]);
        const int nb = K / 32;
        std::vector<block_q8_0_x16> W((size_t) (M / 16) * nb);
        for (auto & b : W) { for (auto & d : b.d) d = f16_bits(0.01f); for (auto & q : b.qs) q = (uint8_t) rng(); }
        std::vector<block_q8_0> X((size_t) N * nb);
        for (auto & b : X) { b.d = f16_bits(0.02f); for (auto & q : b.qs) q = (int8_t) (rng() % 255 - 127); }
        std::vector<int32_t> ysum((size_t) N * nb); std::vector<float> yd((size_t) N * nb);
        for (int j = 0; j < N; j++) ggml_x16_q8_0_col_prep_bias(&X[(size_t) j * nb], nb, 128, &ysum[(size_t) j * nb], &yd[(size_t) j * nb]);
        std::vector<float> C((size_t) N * M, 0.0f);
        const int nchunk = M / CH;
        std::vector<double> ts;
        for (int it = 0; it < iters + 1; it++) {
            const double t0 = now_ms();
            #pragma omp parallel for schedule(dynamic, 1) num_threads(nthr)
            for (int c = 0; c < nchunk; c++) {
                const int r0 = c * CH;
                const void * w = &W[(size_t) (r0 / 16) * nb];
                const int wmax = std::abs(NRMAX), ncall = (N + wmax - 1) / wmax;
                for (int ci = 0, j0 = 0; j0 < N; ci++) {
                    const int nr = NRMAX > 0 ? std::min(NRMAX, N - j0) : N / ncall + (ci < N % ncall ? 1 : 0);
                    float * outs[16]; const void * ys[16]; const int32_t * sums[16]; const float * ds[16];
                    for (int j = 0; j < nr; j++) {
                        outs[j] = &C[(size_t) (j0 + j) * M + r0]; ys[j] = &X[(size_t) (j0 + j) * nb];
                        sums[j] = &ysum[(size_t) (j0 + j) * nb]; ds[j] = &yd[(size_t) (j0 + j) * nb];
                    }
                    ggml_gemm_ptrs_q8_0_x16_q8_0(K, nr, outs, ys, w, CH, sums, ds);
                    j0 += nr;
                }
            }
            if (it > 0) ts.push_back(now_ms() - t0);
        }
        // check a sample of columns against the GEMV
        double maxd = 0; std::vector<float> ref(M);
        for (int j = 0; j < N; j += (N <= 64 ? 1 : 37)) {   // small N: check every column
            ggml_gemv_q8_0_x16_q8_0(K, ref.data(), 0, W.data(), &X[(size_t) j * nb], 1, M);
            for (int i = 0; i < M; i++) maxd = std::max(maxd, (double) std::fabs(ref[i] - C[(size_t) j * M + i]));
        }
        std::sort(ts.begin(), ts.end());
        printf("dense q8_0 M %d K %d N %d NRMAX %d CH %d thr %d: min %.2f ms median %.2f ms  %.0f GOPS  max|gemm-gemv| %g\n",
               M, K, N, NRMAX, CH, nthr, ts[0], ts[ts.size() / 2], 2.0 * M * N * K / (ts[0] * 1e6), maxd);
        return 0;
    }
    const int E = atoi(argv[2]), M = atoi(argv[3]), K = atoi(argv[4]), TOK = atoi(argv[5]), NRMAX = atoi(argv[6]);
    const int nthr = atoi(argv[7]), iters = atoi(argv[8]);
    const int nb = K / 32, TILE = argc > 9 ? atoi(argv[9]) : 64;
    std::vector<block_mxfp4_x16> W((size_t) E * (M / 16) * nb);
    for (auto & b : W) { for (auto & e : b.e) e = (uint8_t) (120 + rng() % 10); for (auto & q : b.qs) q = (uint8_t) rng(); }
    std::vector<block_q8_0> X((size_t) E * TOK * nb);
    for (auto & b : X) { b.d = f16_bits(0.02f); for (auto & q : b.qs) q = (int8_t) (rng() % 255 - 127); }
    std::vector<int32_t> ysum(X.size()); std::vector<float> yd(X.size());
    for (size_t j = 0; j < (size_t) E * TOK; j++) ggml_x16_q8_0_col_prep_bias(&X[j * nb], nb, 12, &ysum[j * nb], &yd[j * nb]);
    std::vector<float> C((size_t) E * TOK * M, 0.0f);
    const int ntile = M / TILE;
    std::vector<double> ts;
    for (int it = 0; it < iters + 1; it++) {
        const double t0 = now_ms();
        #pragma omp parallel for schedule(dynamic, 1) num_threads(nthr)
        for (int item = 0; item < E * ntile; item++) {
            const int e = item / ntile, t0r = (item % ntile) * TILE;
            const void * w = &W[((size_t) e * (M / 16) + t0r / 16) * nb];
            const int wmax = std::abs(NRMAX), ncall = (TOK + wmax - 1) / wmax;
            for (int ci = 0, j0 = 0; j0 < TOK; ci++) {
                const int nr = NRMAX > 0 ? std::min(NRMAX, TOK - j0) : TOK / ncall + (ci < TOK % ncall ? 1 : 0);
                float * outs[16]; const void * ys[16]; const int32_t * sums[16]; const float * ds[16];
                for (int j = 0; j < nr; j++) {
                    const size_t col = (size_t) e * TOK + j0 + j;
                    outs[j] = &C[col * M + t0r]; ys[j] = &X[col * nb]; sums[j] = &ysum[col * nb]; ds[j] = &yd[col * nb];
                }
                ggml_gemm_ptrs_mxfp4_x16_q8_0(K, nr, outs, ys, w, TILE, sums, ds);
                j0 += nr;
            }
        }
        if (it > 0) ts.push_back(now_ms() - t0);
    }
    double maxd = 0; std::vector<float> ref(M);
    for (int e = 0; e < E; e += 53) for (int t = TOK <= 16 ? 0 : TOK - 1; t < TOK; t++) {   // TOK <= 16: every column
        const size_t col = (size_t) e * TOK + t;
        ggml_gemv_mxfp4_x16_q8_0(K, ref.data(), 0, &W[(size_t) e * (M / 16) * nb], &X[col * nb], 1, M);
        for (int i = 0; i < M; i++) maxd = std::max(maxd, (double) std::fabs(ref[i] - C[col * M + i]));
    }
    std::sort(ts.begin(), ts.end());
    const double ops = 2.0 * E * TOK * (double) M * K;
    printf("moe mxfp4 E %d M %d K %d tok %d NRMAX %d thr %d: min %.2f ms median %.2f ms  %.0f GOPS  weights %.2f GB -> %.0f GB/s  max|gemm-gemv| %g\n",
           E, M, K, TOK, NRMAX, nthr, ts[0], ts[ts.size() / 2], ops / (ts[0] * 1e6),
           W.size() * 272.0 / 1e9, W.size() * 272.0 / 1e9 / (ts[0] / 1e3), maxd);
    return 0;
}
