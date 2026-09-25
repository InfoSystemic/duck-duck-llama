// peak vpdpbusd zmm throughput per core: 12 independent accumulators, register operands, and with {1to16} memory bcst
#include <immintrin.h>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <omp.h>
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main(int argc, char ** argv) {
    const long iters = 50'000'000;
    alignas(64) int32_t ybuf[64]; for (int i = 0; i < 64; i++) ybuf[i] = i * 0x01010101;
    #pragma omp parallel
    {
        __m512i a0 = _mm512_set1_epi32(1), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0, a8 = a0, a9 = a0, a10 = a0, a11 = a0;
        const __m512i w = _mm512_set1_epi32(0x01020304), y = _mm512_set1_epi32(0x05060708);
        #pragma omp barrier
        double t0 = now();
        for (long i = 0; i < iters; i++) {
            __asm__ volatile(
                "vpdpbusd %[y], %[w], %[a0]\n vpdpbusd %[y], %[w], %[a1]\n vpdpbusd %[y], %[w], %[a2]\n vpdpbusd %[y], %[w], %[a3]\n"
                "vpdpbusd %[y], %[w], %[a4]\n vpdpbusd %[y], %[w], %[a5]\n vpdpbusd %[y], %[w], %[a6]\n vpdpbusd %[y], %[w], %[a7]\n"
                "vpdpbusd %[y], %[w], %[a8]\n vpdpbusd %[y], %[w], %[a9]\n vpdpbusd %[y], %[w], %[a10]\n vpdpbusd %[y], %[w], %[a11]\n"
                : [a0]"+v"(a0), [a1]"+v"(a1), [a2]"+v"(a2), [a3]"+v"(a3), [a4]"+v"(a4), [a5]"+v"(a5), [a6]"+v"(a6), [a7]"+v"(a7), [a8]"+v"(a8), [a9]"+v"(a9), [a10]"+v"(a10), [a11]"+v"(a11)
                : [w]"v"(w), [y]"v"(y));
        }
        double t1 = now();
        for (long i = 0; i < iters; i++) {
            __asm__ volatile(
                "vpdpbusd (%[p])%{1to16%}, %[w], %[a0]\n vpdpbusd 4(%[p])%{1to16%}, %[w], %[a1]\n vpdpbusd 8(%[p])%{1to16%}, %[w], %[a2]\n vpdpbusd 12(%[p])%{1to16%}, %[w], %[a3]\n"
                "vpdpbusd 16(%[p])%{1to16%}, %[w], %[a4]\n vpdpbusd 20(%[p])%{1to16%}, %[w], %[a5]\n vpdpbusd 24(%[p])%{1to16%}, %[w], %[a6]\n vpdpbusd 28(%[p])%{1to16%}, %[w], %[a7]\n"
                "vpdpbusd 32(%[p])%{1to16%}, %[w], %[a8]\n vpdpbusd 36(%[p])%{1to16%}, %[w], %[a9]\n vpdpbusd 40(%[p])%{1to16%}, %[w], %[a10]\n vpdpbusd 44(%[p])%{1to16%}, %[w], %[a11]\n"
                : [a0]"+v"(a0), [a1]"+v"(a1), [a2]"+v"(a2), [a3]"+v"(a3), [a4]"+v"(a4), [a5]"+v"(a5), [a6]"+v"(a6), [a7]"+v"(a7), [a8]"+v"(a8), [a9]"+v"(a9), [a10]"+v"(a10), [a11]"+v"(a11)
                : [w]"v"(w), [p]"r"(ybuf));
        }
        double t2 = now();
        // mix: 8 dpbusd + 1 vcvtdq2ps + 1 vfmadd per 8 (~ the kernel's ratio of p05 float work)
        __m512 f0 = _mm512_set1_ps(1.0f), f1 = f0; const __m512 s = _mm512_set1_ps(1e-9f);
        for (long i = 0; i < iters; i++) {
            __asm__ volatile(
                "vpdpbusd %[y], %[w], %[a0]\n vpdpbusd %[y], %[w], %[a1]\n vpdpbusd %[y], %[w], %[a2]\n vpdpbusd %[y], %[w], %[a3]\n"
                "vpdpbusd %[y], %[w], %[a4]\n vpdpbusd %[y], %[w], %[a5]\n vpdpbusd %[y], %[w], %[a6]\n vpdpbusd %[y], %[w], %[a7]\n"
                "vfmadd231ps %[s], %[s], %[f0]\n vfmadd231ps %[s], %[s], %[f1]\n vcvtdq2ps %[w], %%zmm31\n vpsubd %[w], %[y], %%zmm30\n"
                : [a0]"+v"(a0), [a1]"+v"(a1), [a2]"+v"(a2), [a3]"+v"(a3), [a4]"+v"(a4), [a5]"+v"(a5), [a6]"+v"(a6), [a7]"+v"(a7), [f0]"+v"(f0), [f1]"+v"(f1)
                : [w]"v"(w), [y]"v"(y), [s]"v"(s) : "zmm31", "zmm30");
        }
        double t3 = now();
        volatile int sink = _mm512_reduce_add_epi32(_mm512_add_epi32(_mm512_add_epi32(a0, a5), a11)) + (int) _mm512_reduce_add_ps(_mm512_add_ps(f0, f1)); (void) sink;
        #pragma omp critical
        if (omp_get_thread_num() == 0 || argc > 1)
            printf("thr %2d: reg %.3f G dpbusd/s | m32bcst %.3f G/s | mix 8dp+4 %.3f G dpbusd/s\n", omp_get_thread_num(),
                   12.0 * iters / (t1 - t0) / 1e9, 12.0 * iters / (t2 - t1) / 1e9, 8.0 * iters / (t3 - t2) / 1e9);
    }
}
