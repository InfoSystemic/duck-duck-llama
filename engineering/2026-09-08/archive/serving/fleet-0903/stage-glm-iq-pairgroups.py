#!/usr/bin/env python3
"""Stage an exact two-output-group XXS kernel; never build or run a model."""
import argparse
import difflib
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--apply', action='store_true')
a = p.parse_args()
base = Path(__file__).resolve().parent
source = base.parents[1] / 'engines/llama.cpp-glm5n-goal-0904/ggml/src/ggml-cpu/repack.cpp'
old = source.read_text()
if 'GGML_CPU_IQ_R16_PAIRGROUPS' in old:
    raise SystemExit('Already applied')
kernel = r'''static bool iq_r16_pairgroups_enabled() {
    static const bool enabled = [] {
        const char * value = getenv("GGML_CPU_IQ_R16_PAIRGROUPS");
        return value && atoi(value) == 1;
    }();
    return enabled;
}

// Two independent output groups share each activation broadcast. The packed
// format and the floating-point accumulation order within each row are unchanged.
template <bool IQ3>
static void ggml_gemv_iq_r16_pairgroups(
        int n, float * s, size_t bs, const void * vx, const block_q8_K * a, int nc) {
    GGML_ASSERT(n % QK_K == 0 && nc % 16 == 0);
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
    const int nb = n / QK_K;
    const auto * weights = (const block_iq_r16 *) vx;
    constexpr float factor = IQ3 ? 0.25f : 0.125f;
    static const uint8_t values2[16] = {21, 39, 56, 72, 89, 107, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64};
    static const uint8_t values3[16] = {2, 12, 20, 28, 36, 44, 52, 60, 68, 76, 84, 92, 100, 108, 116, 126};
    const __m512i lut = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *) (IQ3 ? values3 : values2)));
    const __m512i mask = _mm512_set1_epi8(15);
    int x = 0;
    for (; x + 1 < nc / 16; x += 2) {
        const block_iq_r16 * w[2] = {weights + x * nb, weights + (x + 1) * nb};
        __m512 sum[2] = {_mm512_setzero_ps(), _mm512_setzero_ps()};
        for (int b = 0; b < nb; ++b) {
            __m512i isum[2] = {_mm512_setzero_si512(), _mm512_setzero_si512()};
            for (int sb = 0; sb < QK_K / 32; ++sb) {
                __m512i dots[2][2] = {{_mm512_setzero_si512(), _mm512_setzero_si512()},
                                     {_mm512_setzero_si512(), _mm512_setzero_si512()}};
#if defined(__GNUC__)
#pragma GCC unroll 8
#endif
                for (int chunk = 0; chunk < 8; ++chunk) {
                    int32_t aq;
                    memcpy(&aq, a[b].qs + sb * 32 + chunk * 4, sizeof(aq));
                    const __m512i activation = _mm512_set1_epi32(aq);
                    for (int group = 0; group < 2; ++group) {
                        const __m512i packed = _mm512_loadu_si512(w[group][b].qs + (sb * 4 + chunk / 2) * 64);
                        const __m512i codes = (chunk & 1) ? _mm512_srli_epi16(packed, 4) : packed;
                        const __m512i q = _mm512_shuffle_epi8(lut, _mm512_and_si512(codes, mask));
                        dots[group][chunk & 1] = _mm512_dpbusd_epi32(dots[group][chunk & 1], q, activation);
                    }
                }
                const int bsum = int(a[b].bsums[2 * sb]) + a[b].bsums[2 * sb + 1];
                const __m512i correction = _mm512_set1_epi32(64 * bsum);
                for (int group = 0; group < 2; ++group) {
                    __m512i scales = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *) (w[group][b].scales + sb * 16)));
                    scales = _mm512_add_epi32(_mm512_slli_epi32(_mm512_and_si512(scales, _mm512_set1_epi32(15)), 1),
                                             _mm512_set1_epi32(1));
                    __m512i dot = _mm512_sub_epi32(_mm512_add_epi32(dots[group][0], dots[group][1]), correction);
                    isum[group] = _mm512_add_epi32(isum[group], _mm512_mullo_epi32(dot, scales));
                }
            }
            const __m512 activation_scale = _mm512_set1_ps(a[b].d * factor);
            for (int group = 0; group < 2; ++group) {
                const __m512 d = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) w[group][b].d));
                const __m512 scale = _mm512_mul_ps(d, activation_scale);
                sum[group] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(isum[group]), scale, sum[group]);
            }
        }
        for (int group = 0; group < 2; ++group) _mm512_storeu_ps(s + (x + group) * 16, sum[group]);
    }
    if (x < nc / 16) {
        const block_q8_K * rows[] = {a};
        ggml_gemv_iq_r16_scale32_impl<IQ3, 1>(n, s + x * 16, bs, weights + x * nb, rows, 16);
    }
#else
    const block_q8_K * rows[] = {a};
    ggml_gemv_iq_r16_scale32_impl<IQ3, 1>(n, s, bs, vx, rows, nc);
#endif
}

'''
anchor = 'template <bool IQ3, bool SHARED_SCALE = false>\nstatic void ggml_gemv_iq_r16_q8_K('
assert old.count(anchor) == 1
new = old.replace(anchor, kernel + anchor, 1)
anchor = '                ggml_gemv_iq_r16_scale32_impl<IQ3, 1>(n, s + y * bs, bs, vx, rows, nc);'
assert new.count(anchor) == 1
new = new.replace(anchor, '''                if (iq_r16_pairgroups_enabled() && nc >= 32) {
                    ggml_gemv_iq_r16_pairgroups<IQ3>(n, s + y * bs, bs, vx, rows[0], nc);
                } else {
                    ggml_gemv_iq_r16_scale32_impl<IQ3, 1>(n, s + y * bs, bs, vx, rows, nc);
                }''', 1)
anchor = '            else ggml_gemv_iq_r16_scale32_impl<IQ3, 1>(n, s, bs, vx, a, nc);'
assert new.count(anchor) == 1
new = new.replace(anchor, '''            else if (iq_r16_pairgroups_enabled() && nc >= 32) ggml_gemv_iq_r16_pairgroups<IQ3>(n, s, bs, vx, a[0], nc);
            else ggml_gemv_iq_r16_scale32_impl<IQ3, 1>(n, s, bs, vx, a, nc);''', 1)
patch = ''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), fromfile=str(source), tofile=str(source)))
(base / 'glm-iq-pairgroups.patch').write_text(patch)
if a.apply:
    backup = source.with_suffix(source.suffix + '.before-goal-iq-pairgroups')
    assert not backup.exists()
    backup.write_text(old)
    source.write_text(new)
print('Applied' if a.apply else 'Staged', len(patch.splitlines()), 'patch lines')
