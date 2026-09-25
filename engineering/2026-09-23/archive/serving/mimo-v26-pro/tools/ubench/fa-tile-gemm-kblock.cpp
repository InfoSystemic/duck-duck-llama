// fa_gqa_gemm (C[M x N] += A[M x K] B[K x N]) at the QK^T tile shape M=128 K=192 N=64 and the PV shape M=128 K=64 N=128:
// shipped (32-col panels, 12x2 ukernel over the full K) vs K-blocked (same FMA order per element, C stored/reloaded)
#include <immintrin.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
template <int RM, int RN>
static inline void uk(float * __restrict C, const float * __restrict A, const float * __restrict B, int K, int N, int lda) {
    __m512 acc[RM][RN];
    for (int i = 0; i < RM; i++) for (int r = 0; r < RN; r++) acc[i][r] = _mm512_loadu_ps(C + i * N + r * 16);
    for (int kk = 0; kk < K; kk++) {
        __m512 Bv[RN];
        for (int r = 0; r < RN; r++) Bv[r] = _mm512_loadu_ps(B + kk * N + r * 16);
        for (int i = 0; i < RM; i++) {
            __m512 p = _mm512_set1_ps(A[i * lda + kk]);
            for (int r = 0; r < RN; r++) acc[i][r] = _mm512_fmadd_ps(Bv[r], p, acc[i][r]);
        }
    }
    for (int i = 0; i < RM; i++) for (int r = 0; r < RN; r++) _mm512_storeu_ps(C + i * N + r * 16, acc[i][r]);
}
__attribute__((noinline)) void gemm_ref(float * C, const float * A, const float * B, int M, int K, int N) {
    for (int jj = 0; jj < N; jj += 32) {
        int ii = 0;
        for (; ii + 12 <= M; ii += 12) uk<12, 2>(C + ii * N + jj, A + ii * K, B + jj, K, N, K);
        for (; ii + 4 <= M; ii += 4) uk<4, 2>(C + ii * N + jj, A + ii * K, B + jj, K, N, K);
        for (; ii < M; ii++) uk<1, 2>(C + ii * N + jj, A + ii * K, B + jj, K, N, K);
    }
}
__attribute__((noinline)) void gemm_kb(float * C, const float * A, const float * B, int M, int K, int N, int KB) {
    for (int jj = 0; jj < N; jj += 32) {
        for (int k0 = 0; k0 < K; k0 += KB) {
            const int kn = std::min(KB, K - k0);
            int ii = 0;
            for (; ii + 12 <= M; ii += 12) uk<12, 2>(C + ii * N + jj, A + ii * K + k0, B + k0 * N + jj, kn, N, K);
            for (; ii + 4 <= M; ii += 4) uk<4, 2>(C + ii * N + jj, A + ii * K + k0, B + k0 * N + jj, kn, N, K);
            for (; ii < M; ii++) uk<1, 2>(C + ii * N + jj, A + ii * K + k0, B + k0 * N + jj, kn, N, K);
        }
    }
}
int main(int argc, char ** argv) {
    std::mt19937 rng(1); std::uniform_real_distribution<float> u(-1, 1);
    for (int shape = 0; shape < 2; shape++) {
        const int M = 128, K = shape == 0 ? 192 : 64, N = shape == 0 ? 64 : 128;
        std::vector<float> A(M * K), B(K * N), C1(M * N), C2(M * N);
        for (auto & x : A) x = u(rng); for (auto & x : B) x = u(rng);
        const long reps = 20000;
        for (int KB : {K, 96, 64, 48, 32}) {
            if (KB > K) continue;
            double best1 = 1e30, best2 = 1e30;
            for (int t = 0; t < 5; t++) {
                double t0 = now(); for (long r = 0; r < reps; r++) { memset(C1.data(), 0, C1.size() * 4); gemm_ref(C1.data(), A.data(), B.data(), M, K, N); } double t1 = now();
                for (long r = 0; r < reps; r++) { memset(C2.data(), 0, C2.size() * 4); gemm_kb(C2.data(), A.data(), B.data(), M, K, N, KB); } double t2 = now();
                best1 = std::min(best1, t1 - t0); best2 = std::min(best2, t2 - t1);
            }
            const double fl = 2.0 * M * K * N * reps;
            printf("M %d K %3d N %3d KB %3d: ref %.1f GFLOP/s  kblocked %.1f GFLOP/s  identical %s\n", M, K, N, KB, fl / best1 / 1e9, fl / best2 / 1e9,
                   memcmp(C1.data(), C2.data(), C1.size() * 4) == 0 ? "yes" : "NO");
        }
    }
}
