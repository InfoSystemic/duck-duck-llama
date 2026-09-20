#!/usr/bin/env python3
"""make_repack.py -- repack.f18.cpp: software prefetch in the Q5_K x16 expert kernel (GGML_F18_FEATURES bit 3, bit-exact).

Production profile, revision e: the routed experts' gate and up projections (Q4_K) stream at ~97 GB/s per socket, the machine's wall;
the down projection (Q5_K, 39 of 42 layers) reaches ~77 GB/s and costs 18.9 of the 43 ms the experts take. The compact Q5 kernel
walks a 2,880-byte super-block group by SUB-BLOCK: for each of 8 sub-blocks it visits all 8 chunks (stride 320 bytes) and re-reads
the high-bit line of every chunk, so first touches are strided and the hardware streamer does not follow them. Expert weights are
always cold (19 GB per cycle), so every first touch is a DRAM read. Prefetching the next group while the current one is
multiplied hides that latency. No arithmetic changes.
"""
from pathlib import Path
HERE = Path(__file__).resolve().parent
SRC = Path('/home/user/InfoSystemic/AI-Server/serving/fleet-0903/results/glm-flash-q8-r8-ordered-k-0908/private-cpu/repack.cpp')
s = SRC.read_text()
def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:70]); s = s.replace(old, new)

rep('''template <int NR, bool COMPACT = false, int NP = 4>
static void ggml_gemv_q5_K_x16_bytes_impl(''', '''// f18: bit 3 of GGML_F18_FEATURES / GGML_F18_CONTROL_FILE = prefetch the next Q5_K x16 block group; distance in groups from
// GGML_F18_Q5_PREFETCH_BLOCKS (default 1)
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
static inline bool f18_q5_prefetch_enabled() {
    static const uint32_t env_mask = [] {
        const char * v = getenv("GGML_F18_FEATURES");
        return v ? (uint32_t) strtoul(v, nullptr, 0) : 0u;
    }();
    static const uint32_t * control = [] () -> const uint32_t * {
        const char * path = getenv("GGML_F18_CONTROL_FILE");
        if (!path || !*path) return nullptr;
        const int fd = open(path, O_RDONLY | O_CLOEXEC);
        if (fd < 0) return nullptr;
        void * mapping = mmap(nullptr, sizeof(uint32_t), PROT_READ, MAP_SHARED, fd, 0);
        close(fd);
        return mapping == MAP_FAILED ? nullptr : (const uint32_t *) mapping;
    }();
    return ((control ? __atomic_load_n(control, __ATOMIC_ACQUIRE) : env_mask) & 8u) != 0;
}
static inline int f18_q5_prefetch_blocks() {
    static const int v = [] {
        const char * e = getenv("GGML_F18_Q5_PREFETCH_BLOCKS");
        const int x = e ? atoi(e) : 1;
        return x > 0 ? x : 1;
    }();
    return v;
}

template <int NR, bool COMPACT = false, int NP = 4>
static void ggml_gemv_q5_K_x16_bytes_impl(''')

rep('''#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
    for (int g = 0; g < nc / 16; ++g) {
        const block_type * w = weights + g * nb;
        __m512 sum[NR];
        for (int y = 0; y < NR; ++y) sum[y] = _mm512_setzero_ps();
        for (int b = 0; b < nb; ++b) {
            __m512i isum[NR], imin[NR];''', '''#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
    const bool f18_prefetch = f18_q5_prefetch_enabled();
    const int  f18_ahead    = f18_q5_prefetch_blocks();
    for (int g = 0; g < nc / 16; ++g) {
        const block_type * w = weights + g * nb;
        __m512 sum[NR];
        for (int y = 0; y < NR; ++y) sum[y] = _mm512_setzero_ps();
        for (int b = 0; b < nb; ++b) {
            if (f18_prefetch) {
                // the block group this kernel reaches f18_ahead iterations from now (groups are contiguous across g)
                const char * next = (const char *) (w + b + f18_ahead);
                for (size_t off = 0; off < sizeof(block_type); off += 64) {
                    _mm_prefetch(next + off, _MM_HINT_T0);
                }
            }
            __m512i isum[NR], imin[NR];''')
(HERE/'repack.f18.cpp').write_text(s)
print('wrote repack.f18.cpp')
