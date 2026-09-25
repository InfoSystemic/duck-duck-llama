// actual core clock under the AVX-512 heavy license: a dependent vpaddd chain (latency 1) timed, with independent
// vpdpbusd mixed in to hold the heavy license
#include <immintrin.h>
#include <chrono>
#include <cstdio>
int main() {
    __m512i x = _mm512_set1_epi32(1), o = _mm512_set1_epi32(3), a0 = x, a1 = x, a2 = x, a3 = x;
    const long it = 100'000'000;
    for (int rep = 0; rep < 3; rep++) {
        auto t0 = std::chrono::steady_clock::now();
        for (long i = 0; i < it; i++) {
            __asm__ volatile(
                "vpaddd %[o], %[x], %[x]\n vpaddd %[o], %[x], %[x]\n vpaddd %[o], %[x], %[x]\n vpaddd %[o], %[x], %[x]\n"
                "vpdpbusd %[o], %[o], %[a0]\n"
                "vpaddd %[o], %[x], %[x]\n vpaddd %[o], %[x], %[x]\n vpaddd %[o], %[x], %[x]\n vpaddd %[o], %[x], %[x]\n"
                "vpdpbusd %[o], %[o], %[a1]\n"
                "vpaddd %[o], %[x], %[x]\n vpaddd %[o], %[x], %[x]\n vpaddd %[o], %[x], %[x]\n vpaddd %[o], %[x], %[x]\n"
                "vpdpbusd %[o], %[o], %[a2]\n"
                "vpaddd %[o], %[x], %[x]\n vpaddd %[o], %[x], %[x]\n vpaddd %[o], %[x], %[x]\n vpaddd %[o], %[x], %[x]\n"
                "vpdpbusd %[o], %[o], %[a3]\n"
                : [x]"+v"(x), [a0]"+v"(a0), [a1]"+v"(a1), [a2]"+v"(a2), [a3]"+v"(a3) : [o]"v"(o));
        }
        double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        printf("AVX-512 heavy clock: %.2f GHz\n", 16.0 * it / s / 1e9);
    }
    volatile int k = _mm512_reduce_add_epi32(_mm512_add_epi32(x, _mm512_add_epi32(a0, a3))); (void) k;
}
