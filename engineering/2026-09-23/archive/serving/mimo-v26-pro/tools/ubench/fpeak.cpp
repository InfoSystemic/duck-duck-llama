// vfmadd231ps zmm throughput: register operands vs {1to16} memory broadcast (12 independent accumulators)
#include <immintrin.h>
#include <chrono>
#include <cstdio>
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main() {
    const long it = 50'000'000;
    alignas(64) float buf[64]; for (int i = 0; i < 64; i++) buf[i] = 1e-9f * i;
    __m512 a0 = _mm512_set1_ps(1), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0, a8 = a0, a9 = a0, a10 = a0, a11 = a0;
    const __m512 w = _mm512_set1_ps(1e-9f), y = _mm512_set1_ps(1e-9f);
    for (int rep = 0; rep < 2; rep++) {
    double t0 = now();
    for (long i = 0; i < it; i++)
        __asm__ volatile("vfmadd231ps %[y], %[w], %[a0]\n vfmadd231ps %[y], %[w], %[a1]\n vfmadd231ps %[y], %[w], %[a2]\n vfmadd231ps %[y], %[w], %[a3]\n"
                         "vfmadd231ps %[y], %[w], %[a4]\n vfmadd231ps %[y], %[w], %[a5]\n vfmadd231ps %[y], %[w], %[a6]\n vfmadd231ps %[y], %[w], %[a7]\n"
                         "vfmadd231ps %[y], %[w], %[a8]\n vfmadd231ps %[y], %[w], %[a9]\n vfmadd231ps %[y], %[w], %[a10]\n vfmadd231ps %[y], %[w], %[a11]\n"
            : [a0]"+v"(a0), [a1]"+v"(a1), [a2]"+v"(a2), [a3]"+v"(a3), [a4]"+v"(a4), [a5]"+v"(a5), [a6]"+v"(a6), [a7]"+v"(a7), [a8]"+v"(a8), [a9]"+v"(a9), [a10]"+v"(a10), [a11]"+v"(a11)
            : [w]"v"(w), [y]"v"(y));
    double t1 = now();
    for (long i = 0; i < it; i++)
        __asm__ volatile("vfmadd231ps (%[p])%{1to16%}, %[w], %[a0]\n vfmadd231ps 4(%[p])%{1to16%}, %[w], %[a1]\n vfmadd231ps 8(%[p])%{1to16%}, %[w], %[a2]\n vfmadd231ps 12(%[p])%{1to16%}, %[w], %[a3]\n"
                         "vfmadd231ps 16(%[p])%{1to16%}, %[w], %[a4]\n vfmadd231ps 20(%[p])%{1to16%}, %[w], %[a5]\n vfmadd231ps 24(%[p])%{1to16%}, %[w], %[a6]\n vfmadd231ps 28(%[p])%{1to16%}, %[w], %[a7]\n"
                         "vfmadd231ps 32(%[p])%{1to16%}, %[w], %[a8]\n vfmadd231ps 36(%[p])%{1to16%}, %[w], %[a9]\n vfmadd231ps 40(%[p])%{1to16%}, %[w], %[a10]\n vfmadd231ps 44(%[p])%{1to16%}, %[w], %[a11]\n"
            : [a0]"+v"(a0), [a1]"+v"(a1), [a2]"+v"(a2), [a3]"+v"(a3), [a4]"+v"(a4), [a5]"+v"(a5), [a6]"+v"(a6), [a7]"+v"(a7), [a8]"+v"(a8), [a9]"+v"(a9), [a10]"+v"(a10), [a11]"+v"(a11)
            : [w]"v"(w), [p]"r"(buf));
    double t2 = now();
    // explicit broadcast into a register, reused by 2 FMAs (what a 2-column-block microkernel does)
    for (long i = 0; i < it; i++)
        __asm__ volatile("vbroadcastss (%[p]), %%zmm30\n vfmadd231ps %%zmm30, %[w], %[a0]\n vfmadd231ps %%zmm30, %[y], %[a1]\n"
                         "vbroadcastss 4(%[p]), %%zmm30\n vfmadd231ps %%zmm30, %[w], %[a2]\n vfmadd231ps %%zmm30, %[y], %[a3]\n"
                         "vbroadcastss 8(%[p]), %%zmm30\n vfmadd231ps %%zmm30, %[w], %[a4]\n vfmadd231ps %%zmm30, %[y], %[a5]\n"
                         "vbroadcastss 12(%[p]), %%zmm30\n vfmadd231ps %%zmm30, %[w], %[a6]\n vfmadd231ps %%zmm30, %[y], %[a7]\n"
                         "vbroadcastss 16(%[p]), %%zmm30\n vfmadd231ps %%zmm30, %[w], %[a8]\n vfmadd231ps %%zmm30, %[y], %[a9]\n"
                         "vbroadcastss 20(%[p]), %%zmm30\n vfmadd231ps %%zmm30, %[w], %[a10]\n vfmadd231ps %%zmm30, %[y], %[a11]\n"
            : [a0]"+v"(a0), [a1]"+v"(a1), [a2]"+v"(a2), [a3]"+v"(a3), [a4]"+v"(a4), [a5]"+v"(a5), [a6]"+v"(a6), [a7]"+v"(a7), [a8]"+v"(a8), [a9]"+v"(a9), [a10]"+v"(a10), [a11]"+v"(a11)
            : [w]"v"(w), [y]"v"(y), [p]"r"(buf) : "zmm30");
    double t3 = now();
    printf("vfmadd231ps: reg %.2f G/s | m32bcst %.2f G/s | bcst-reg x2 %.2f G/s\n", 12.0 * it / (t1 - t0) / 1e9, 12.0 * it / (t2 - t1) / 1e9, 12.0 * it / (t3 - t2) / 1e9);
    }
    volatile float s = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(a0, a5), a11)); (void) s;
}
