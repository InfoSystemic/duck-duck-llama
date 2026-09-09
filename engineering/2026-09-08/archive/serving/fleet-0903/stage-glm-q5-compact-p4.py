#!/usr/bin/env python3
"""Prepare a compact Q5 variant of the shared three-row kernel."""
import difflib
from pathlib import Path
import sys

base = Path(__file__).resolve().parent
path = base.parents[1] / 'engines/llama.cpp-glm5n-goal-0904/ggml/src/ggml-cpu/repack.cpp'
old = path.read_text()
assert 'GGML_CPU_X16_Q5_COMPACT_P4' not in old
start = old.index('template <int NR>\nstatic void ggml_gemv_q5_K_x16_bytes_impl(')
end = old.index('static void ggml_gemv_q5_K_x16_bytes_q8_K(', start)
kernel = old[start:end]
kernel = kernel.replace('template <int NR>', 'template <int NR, bool COMPACT = false>', 1)
kernel = kernel.replace('    const auto * weights = (const block_q5_K_x16_bytes *) vx;',
'''    using block_type = std::conditional_t<COMPACT, block_q5_K_x16, block_q5_K_x16_bytes>;
    const auto * weights = (const block_type *) vx;''', 1)
kernel = kernel.replace('const block_q5_K_x16_bytes * w =', 'const block_type * w =')
needle = '                    const __m512i q = _mm512_loadu_si512(w[b].qs[sb][chunk]);'
assert kernel.count(needle) == 1
kernel = kernel.replace(needle, '''                    __m512i q;
                    if constexpr (COMPACT) {
                        const uint8_t * packed = w[b].qsh + chunk * 320;
                        const __m512i low = _mm512_loadu_si512(packed + (sb / 2) * 64);
                        const __m512i high = _mm512_loadu_si512(packed + 256);
                        const __m512i lo4 = _mm512_and_si512(
                            _mm512_srl_epi16(low, _mm_cvtsi32_si128(4 * (sb & 1))), _mm512_set1_epi8(15));
                        const __m512i hi1 = _mm512_and_si512(
                            _mm512_srl_epi16(high, _mm_cvtsi32_si128(sb)), _mm512_set1_epi8(1));
                        q = _mm512_or_si512(lo4, _mm512_slli_epi16(hi1, 4));
                    } else {
                        q = _mm512_loadu_si512(w[b].qs[sb][chunk]);
                    }''')
needle = '                    for (int k = 0; k < 32; ++k) dot += int(w[b].qs[sb][k / 4][row * 4 + k % 4]) * int(a[b].qs[sb * 32 + k]);'
assert kernel.count(needle) == 1
kernel = kernel.replace(needle, '''                    for (int k = 0; k < 32; ++k) {
                        int q;
                        if constexpr (COMPACT) {
                            const uint8_t * packed = w[b].qsh + (k / 4) * 320;
                            const int offset = row * 4 + k % 4;
                            q = ((packed[(sb / 2) * 64 + offset] >> (4 * (sb & 1))) & 15) |
                                (((packed[256 + offset] >> sb) & 1) << 4);
                        } else {
                            q = w[b].qs[sb][k / 4][row * 4 + k % 4];
                        }
                        dot += q * int(a[b].qs[sb * 32 + k]);
                    }''')
wrapper = '''static void ggml_gemv_q5_K_x16_compact_p4_q8_K(
        int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    GGML_ASSERT(nr >= 1 && nr <= 3);
    if (nr == 3) ggml_gemv_q5_K_x16_bytes_impl<3, true>(n, s, bs, vx, vy, nc);
    else if (nr == 2) ggml_gemv_q5_K_x16_bytes_impl<2, true>(n, s, bs, vx, vy, nc);
    else ggml_gemv_q5_K_x16_bytes_impl<1, true>(n, s, bs, vx, vy, nc);
}

'''
new = old[:start] + kernel + wrapper + old[end:]
needle = '''                    const bool triple = triple_enabled && sp.gemv == ggml_gemv_q5_K_x16_bytes_q8_K && i11 + 2 < ne11;
                    const int nr = triple ? 3 : pair_enabled && sp.dst_type == GGML_TYPE_Q5_K && i11 + 1 < ne11 ? 2 : 1;'''
assert new.count(needle) == 1
new = new.replace(needle, '''                    const bool compact_batch = sp.gemv == ggml_gemv_q5_K_x16_compact_p4_q8_K;
                    const bool triple = (compact_batch || (triple_enabled && sp.gemv == ggml_gemv_q5_K_x16_bytes_q8_K)) && i11 + 2 < ne11;
                    const int nr = triple ? 3 : (pair_enabled || compact_batch) && sp.dst_type == GGML_TYPE_Q5_K && i11 + 1 < ne11 ? 2 : 1;''')
needle = '    static const tensor_traits_x16 t_q5_bytes(spec_q5_bytes);'
assert new.count(needle) == 1
new = new.replace(needle, needle + '''
    static const x16_spec spec_q5_compact_p4 = { GGML_TYPE_Q5_K, GGML_TYPE_Q5_K, QK_K, sizeof(block_q5_K_x16), GGML_TYPE_Q8_K, ggml_gemv_q5_K_x16_compact_p4_q8_K, "x16 q5_K compact p4" };
    static const tensor_traits_x16 t_q5_compact_p4(spec_q5_compact_p4);
    static const bool q5_compact_p4 = []() {
        const char * value = getenv("GGML_CPU_X16_Q5_COMPACT_P4");
        return value != nullptr && atoi(value) != 0;
    }();''')
needle = 'return q5_bytes ? &t_q5_bytes : &t_q5_K;'
assert new.count(needle) == 1
new = new.replace(needle, 'return q5_bytes ? &t_q5_bytes : q5_compact_p4 ? &t_q5_compact_p4 : &t_q5_K;')
patch = ''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
    fromfile='a/ggml/src/ggml-cpu/repack.cpp', tofile='b/ggml/src/ggml-cpu/repack.cpp'))
(base / 'glm-q5-compact-p4.patch').write_text(patch)
if '--apply' in sys.argv:
    backup = path.with_name(path.name + '.before-goal-q5-compact-p4')
    assert not backup.exists()
    backup.write_text(old)
    path.write_text(new)
print('Applied' if '--apply' in sys.argv else 'Prepared without applying', len(patch.splitlines()), 'patch lines')
