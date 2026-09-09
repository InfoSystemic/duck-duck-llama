#!/usr/bin/env python3
"""Prepare shared 32-value scale arithmetic for IQ2_XXS and IQ3_XXS r16."""
import argparse
import difflib
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--apply', action='store_true')
a = p.parse_args()
base = Path(__file__).resolve().parent
source = base.parents[1] / 'engines/llama.cpp-glm5n-goal-0904/ggml/src/ggml-cpu/repack.cpp'
old = source.read_text()
if 'GGML_CPU_IQ_R16_SCALE32' in old:
    raise SystemExit('Already applied')
kernel = '''static bool iq_r16_scale32_enabled() {
    static const bool enabled = [] {
        const char * value = getenv("GGML_CPU_IQ_R16_SCALE32");
        return value && atoi(value) == 1;
    }();
    return enabled && iq_r16_paired_nibbles();
}

template <bool IQ3, int NR>
static void ggml_gemv_iq_r16_scale32_impl(
        int n, float * s, size_t bs, const void * vx, const block_q8_K * const * a, int nc) {
    GGML_ASSERT(n % QK_K == 0 && nc % 16 == 0 && NR >= 1 && NR <= 3);
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
    const int nb = n / QK_K;
    const auto * weights = (const block_iq_r16 *) vx;
    constexpr float factor = IQ3 ? 0.25f : 0.125f;
    static const uint8_t values2[16] = {21, 39, 56, 72, 89, 107, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64};
    static const uint8_t values3[16] = {2, 12, 20, 28, 36, 44, 52, 60, 68, 76, 84, 92, 100, 108, 116, 126};
    const __m512i lut = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *) (IQ3 ? values3 : values2)));
    const __m512i mask = _mm512_set1_epi8(15);
    for (int x = 0; x < nc / 16; ++x) {
        const block_iq_r16 * w = weights + x * nb;
        __m512 sum[NR];
        for (int y = 0; y < NR; ++y) sum[y] = _mm512_setzero_ps();
        for (int b = 0; b < nb; ++b) {
            __m512i isum[NR];
            for (int y = 0; y < NR; ++y) isum[y] = _mm512_setzero_si512();
            for (int sb = 0; sb < QK_K / 32; ++sb) {
                __m512i dots[NR][4];
                for (int y = 0; y < NR; ++y) for (int part = 0; part < 4; ++part) dots[y][part] = _mm512_setzero_si512();
#if defined(__GNUC__)
#pragma GCC unroll 8
#endif
                for (int chunk = 0; chunk < 8; ++chunk) {
                    const __m512i packed = _mm512_loadu_si512(w[b].qs + (sb * 4 + chunk / 2) * 64);
                    const __m512i codes = (chunk & 1) ? _mm512_srli_epi16(packed, 4) : packed;
                    const __m512i q = _mm512_shuffle_epi8(lut, _mm512_and_si512(codes, mask));
                    for (int y = 0; y < NR; ++y) {
                        int32_t aq;
                        memcpy(&aq, a[y][b].qs + sb * 32 + chunk * 4, sizeof(aq));
                        dots[y][chunk & 3] = _mm512_dpbusd_epi32(dots[y][chunk & 3], q, _mm512_set1_epi32(aq));
                    }
                }
                __m512i scales = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *) (w[b].scales + sb * 16)));
                scales = _mm512_add_epi32(_mm512_slli_epi32(_mm512_and_si512(scales, _mm512_set1_epi32(15)), 1),
                    _mm512_set1_epi32(1));
                for (int y = 0; y < NR; ++y) {
                    __m512i dot = _mm512_add_epi32(_mm512_add_epi32(dots[y][0], dots[y][1]),
                        _mm512_add_epi32(dots[y][2], dots[y][3]));
                    const int bsum = int(a[y][b].bsums[2 * sb]) + a[y][b].bsums[2 * sb + 1];
                    dot = _mm512_sub_epi32(dot, _mm512_set1_epi32(64 * bsum));
                    isum[y] = _mm512_add_epi32(isum[y], _mm512_mullo_epi32(dot, scales));
                }
            }
            const __m512 d = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) w[b].d));
            for (int y = 0; y < NR; ++y) {
                const __m512 scale = _mm512_mul_ps(d, _mm512_set1_ps(a[y][b].d * factor));
                sum[y] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(isum[y]), scale, sum[y]);
            }
        }
        for (int y = 0; y < NR; ++y) _mm512_storeu_ps(s + y * bs + x * 16, sum[y]);
    }
#else
    for (int y = 0; y < NR; ++y) ggml_gemv_iq_r16_q8_K_impl<IQ3, true>(n, s + y * bs, bs, vx, a[y], 1, nc);
#endif
}

'''
anchor = 'template <bool IQ3>\nstatic void ggml_gemv_iq_r16_q8_K('
assert old.count(anchor) == 1
new = old.replace(anchor, kernel + 'template <bool IQ3, bool SHARED_SCALE = false>\nstatic void ggml_gemv_iq_r16_q8_K(', 1)
anchor = '''        int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    if (iq_r16_paired_nibbles()) {'''
assert new.count(anchor) == 1
new = new.replace(anchor, '''        int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    if constexpr (SHARED_SCALE) {
        if (iq_r16_scale32_enabled()) {
            for (int y = 0; y < nr; ++y) {
                const block_q8_K * rows[] = {(const block_q8_K *) vy + y * (n / QK_K)};
                ggml_gemv_iq_r16_scale32_impl<IQ3, 1>(n, s + y * bs, bs, vx, rows, nc);
            }
            return;
        }
    }
    if (iq_r16_paired_nibbles()) {''', 1)
anchor = 'template <bool IQ3>\nstatic void ggml_gemv_iq_r16_batch('
assert new.count(anchor) == 1
new = new.replace(anchor, 'template <bool IQ3, bool SHARED_SCALE = false>\nstatic void ggml_gemv_iq_r16_batch(', 1)
anchor = '''    if (nr == 3) ggml_gemv_iq_r16_batch_impl<IQ3, 3>(n, s, bs, vx, a, nc);'''
assert new.count(anchor) == 1
new = new.replace(anchor, '''    if constexpr (SHARED_SCALE) {
        if (iq_r16_scale32_enabled()) {
            if (nr == 3) ggml_gemv_iq_r16_scale32_impl<IQ3, 3>(n, s, bs, vx, a, nc);
            else if (nr == 2) ggml_gemv_iq_r16_scale32_impl<IQ3, 2>(n, s, bs, vx, a, nc);
            else ggml_gemv_iq_r16_scale32_impl<IQ3, 1>(n, s, bs, vx, a, nc);
            return;
        }
    }
    if (nr == 3) ggml_gemv_iq_r16_batch_impl<IQ3, 3>(n, s, bs, vx, a, nc);''', 1)
for operation in ['gemv', 'gemm']:
    for block, iq3 in [('block_iq2_xxs', 'false'), ('block_iq3_xxs', 'true')]:
        anchor = f'''template <> void {operation}<{block}, 8, 16, GGML_TYPE_Q8_K>(
        int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {{
    ggml_gemv_iq_r16_q8_K<{iq3}>(n, s, bs, vx, vy, nr, nc);'''
        assert new.count(anchor) == 1
        new = new.replace(anchor, anchor.replace(f'<{iq3}>', f'<{iq3}, true>'), 1)
anchor = 'ggml_gemv_iq_r16_batch<expanded_iq3_xxs>'
assert new.count(anchor) == 3
new = new.replace(anchor, 'ggml_gemv_iq_r16_batch<expanded_iq3_xxs, !std::is_same_v<BLOC_TYPE, block_iq2_xs>>')
relative = 'ggml/src/ggml-cpu/repack.cpp'
(base / 'glm-iq-scale32.patch').write_text(''.join(difflib.unified_diff(
    old.splitlines(True), new.splitlines(True), 'a/' + relative, 'b/' + relative)))
if a.apply:
    source.with_name(source.name + '.before-goal-iq-scale32').write_text(old)
    source.write_text(new)
print('Applied' if a.apply else 'Prepared only')
