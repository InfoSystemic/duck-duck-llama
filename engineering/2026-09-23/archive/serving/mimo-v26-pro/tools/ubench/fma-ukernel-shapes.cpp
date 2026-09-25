#include <immintrin.h>
#include <chrono>
#include <cstdio>
#include <vector>
#include <cstring>
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
template <int RM, int RN>
__attribute__((noinline)) void uk(float * __restrict C, const float * __restrict A, const float * __restrict B, int K, int N, int lda) {
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
template <int RM, int RN> void run(const char * name, int K) {
    const int N = RN * 16;
    std::vector<float> A(RM * K, 0.001f), B(K * N, 0.002f), C(RM * N, 0.0f);
    const long reps = 200000000L / (RM * RN * K);
    double best = 1e30;
    for (int t = 0; t < 5; t++) { double t0 = now(); for (long r = 0; r < reps; r++) uk<RM, RN>(C.data(), A.data(), B.data(), K, N, K); best = std::min(best, now() - t0); }
    printf("%-6s K %3d: %.1f GFLOP/s (%.2f FMA/cycle @3.1)\n", name, K, 2.0 * RM * N * K * reps / best / 1e9, RM * RN * (double) K * reps / best / 1e9 / 3.1);
}
int main() {
    for (int K : {64, 192}) { run<12, 2>("12x2", K); run<8, 3>("8x3", K); run<6, 4>("6x4", K); run<4, 6>("4x6", K); run<14, 2>("14x2", K); }
}
