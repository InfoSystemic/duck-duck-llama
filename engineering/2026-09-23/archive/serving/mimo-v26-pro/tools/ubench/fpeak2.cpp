
#include <immintrin.h>
#include <chrono>
#include <cstdio>
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
__attribute__((noinline)) void fma_loop(long it) { __asm__ volatile("vxorps %%zmm28, %%zmm28, %%zmm28\n vxorps %%zmm29, %%zmm29, %%zmm29" ::: "zmm28", "zmm29");
  for (long i = 0; i < it; i++) __asm__ volatile(
"vfmadd231ps %%zmm28, %%zmm29, %%zmm0\n"
"vfmadd231ps %%zmm28, %%zmm29, %%zmm1\n"
"vfmadd231ps %%zmm28, %%zmm29, %%zmm2\n"
"vfmadd231ps %%zmm28, %%zmm29, %%zmm3\n"
"vfmadd231ps %%zmm28, %%zmm29, %%zmm4\n"
"vfmadd231ps %%zmm28, %%zmm29, %%zmm5\n"
"vfmadd231ps %%zmm28, %%zmm29, %%zmm6\n"
"vfmadd231ps %%zmm28, %%zmm29, %%zmm7\n"
"vfmadd231ps %%zmm28, %%zmm29, %%zmm8\n"
"vfmadd231ps %%zmm28, %%zmm29, %%zmm9\n"
"vfmadd231ps %%zmm28, %%zmm29, %%zmm10\n"
"vfmadd231ps %%zmm28, %%zmm29, %%zmm11\n"
"vfmadd231ps %%zmm28, %%zmm29, %%zmm12\n"
"vfmadd231ps %%zmm28, %%zmm29, %%zmm13\n"
"vfmadd231ps %%zmm28, %%zmm29, %%zmm14\n"
"vfmadd231ps %%zmm28, %%zmm29, %%zmm15\n"
"vfmadd231ps %%zmm28, %%zmm29, %%zmm16\n"
"vfmadd231ps %%zmm28, %%zmm29, %%zmm17\n"
"vfmadd231ps %%zmm28, %%zmm29, %%zmm18\n"
"vfmadd231ps %%zmm28, %%zmm29, %%zmm19\n"
"vfmadd231ps %%zmm28, %%zmm29, %%zmm20\n"
"vfmadd231ps %%zmm28, %%zmm29, %%zmm21\n"
"vfmadd231ps %%zmm28, %%zmm29, %%zmm22\n"
"vfmadd231ps %%zmm28, %%zmm29, %%zmm23\n"
  ::: "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13", "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21", "zmm22", "zmm23", "zmm28", "zmm29"); }
__attribute__((noinline)) void dp_loop(long it) { for (long i = 0; i < it; i++) __asm__ volatile(
"vpdpbusd %%zmm28, %%zmm29, %%zmm0\n"
"vpdpbusd %%zmm28, %%zmm29, %%zmm1\n"
"vpdpbusd %%zmm28, %%zmm29, %%zmm2\n"
"vpdpbusd %%zmm28, %%zmm29, %%zmm3\n"
"vpdpbusd %%zmm28, %%zmm29, %%zmm4\n"
"vpdpbusd %%zmm28, %%zmm29, %%zmm5\n"
"vpdpbusd %%zmm28, %%zmm29, %%zmm6\n"
"vpdpbusd %%zmm28, %%zmm29, %%zmm7\n"
"vpdpbusd %%zmm28, %%zmm29, %%zmm8\n"
"vpdpbusd %%zmm28, %%zmm29, %%zmm9\n"
"vpdpbusd %%zmm28, %%zmm29, %%zmm10\n"
"vpdpbusd %%zmm28, %%zmm29, %%zmm11\n"
"vpdpbusd %%zmm28, %%zmm29, %%zmm12\n"
"vpdpbusd %%zmm28, %%zmm29, %%zmm13\n"
"vpdpbusd %%zmm28, %%zmm29, %%zmm14\n"
"vpdpbusd %%zmm28, %%zmm29, %%zmm15\n"
"vpdpbusd %%zmm28, %%zmm29, %%zmm16\n"
"vpdpbusd %%zmm28, %%zmm29, %%zmm17\n"
"vpdpbusd %%zmm28, %%zmm29, %%zmm18\n"
"vpdpbusd %%zmm28, %%zmm29, %%zmm19\n"
"vpdpbusd %%zmm28, %%zmm29, %%zmm20\n"
"vpdpbusd %%zmm28, %%zmm29, %%zmm21\n"
"vpdpbusd %%zmm28, %%zmm29, %%zmm22\n"
"vpdpbusd %%zmm28, %%zmm29, %%zmm23\n"
  ::: "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13", "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21", "zmm22", "zmm23", "zmm28", "zmm29"); }
__attribute__((noinline)) void chain(long it) { __asm__ volatile("vxorps %%zmm30, %%zmm30, %%zmm30" ::: "zmm30");
  for (long i = 0; i < it; i++) __asm__ volatile(
  "vaddps %%zmm30, %%zmm30, %%zmm30\n vaddps %%zmm30, %%zmm30, %%zmm30\n vaddps %%zmm30, %%zmm30, %%zmm30\n vaddps %%zmm30, %%zmm30, %%zmm30\n"
  "vfmadd231ps %%zmm28, %%zmm29, %%zmm0\n vfmadd231ps %%zmm28, %%zmm29, %%zmm1\n"
  ::: "zmm30", "zmm0", "zmm1", "zmm28", "zmm29"); }
int main() {
  for (int r = 0; r < 3; r++) {
    const long it = 20'000'000;
    double t0 = now(); fma_loop(it); double t1 = now(); dp_loop(it); double t2 = now(); chain(it * 4); double t3 = now();
    const double ghz = 16.0 * it * 4 / (t3 - t2) / 1e9;   // 4 dependent vaddps (latency 4) = 16 cycles per iteration
    printf("clock under FMA (vaddps chain): %.2f GHz | FMA x24: %.2f G/s = %.2f/cycle | VNNI x24: %.2f G/s = %.2f/cycle\n",
           ghz, 24.0 * it / (t1 - t0) / 1e9, 24.0 * it / (t1 - t0) / 1e9 / ghz, 24.0 * it / (t2 - t1) / 1e9, 24.0 * it / (t2 - t1) / 1e9 / ghz);
  }
}
