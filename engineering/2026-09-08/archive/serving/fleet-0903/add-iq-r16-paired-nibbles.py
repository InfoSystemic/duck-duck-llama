#!/usr/bin/env python3
from pathlib import Path
import difflib
import shutil

root = Path(__file__).resolve().parents[2]
path = root / 'engines/llama.cpp-glm5n-goal-0904/ggml/src/ggml-cpu/repack.cpp'
backup = path.with_name(path.name + '.before-goal-iq-r16-nibble2')
assert not backup.exists()
old = path.read_text()
text = old
marker = 'template <typename BLOC_TYPE>\nstatic int repack_iq_to_r16'
flag = '''static bool iq_r16_paired_nibbles() {
    static const bool enabled = [] {
        const char * value = getenv("GGML_CPU_IQ_R16_NIBBLE2");
        return value && atoi(value) != 0;
    }();
    return enabled;
}

'''
assert marker in text
text = text.replace(marker, flag + marker, 1)
marker = '                            out.qs[to] = r8.qs[from];'
replacement = '''                            if (iq_r16_paired_nibbles()) {
                                const int idx = (sb * 2 + k / 8) * 64 + row * 4 + k % 4;
                                const int shift = ((k / 4) % 2) * 4;
                                out.qs[idx]     |= (r8.qs[from] & 15) << shift;
                                out.qs[idx + 1] |= (r8.qs[from] >> 4) << shift;
                            } else {
                                out.qs[to] = r8.qs[from];
                            }'''
assert text.count(marker) == 1
text = text.replace(marker, replacement, 1)
marker = 'template <bool IQ3>\nstatic void ggml_gemv_iq_r16_q8_K('
text = text.replace(marker, 'template <bool IQ3, bool NIBBLE2>\nstatic void ggml_gemv_iq_r16_q8_K_impl(', 1)
start = text.index('                    __m512i dots = _mm512_setzero_si512();', text.index('static void ggml_gemv_iq_r16_q8_K_impl'))
end = text.index('                    dots = _mm512_sub_epi32', start)
legacy = text[start:end]
legacy = legacy.replace('                    __m512i dots = _mm512_setzero_si512();\n', '')
replacement = '''                    __m512i dots = _mm512_setzero_si512();
                    if constexpr (NIBBLE2) {
                        const __m512i mask = _mm512_set1_epi8(15);
                        const __m512i p0 = _mm512_loadu_si512(w[b].qs + (sb * 2 + 0) * 64);
                        const __m512i p1 = _mm512_loadu_si512(w[b].qs + (sb * 2 + 1) * 64);
                        int32_t aq[4];
                        memcpy(aq, a[b].qs + sb * 16, sizeof(aq));
                        const __m512i d0 = _mm512_dpbusd_epi32(_mm512_setzero_si512(),
                            _mm512_shuffle_epi8(lut, _mm512_and_si512(p0, mask)), _mm512_set1_epi32(aq[0]));
                        const __m512i d1 = _mm512_dpbusd_epi32(_mm512_setzero_si512(),
                            _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16(p0, 4), mask)), _mm512_set1_epi32(aq[1]));
                        const __m512i d2 = _mm512_dpbusd_epi32(_mm512_setzero_si512(),
                            _mm512_shuffle_epi8(lut, _mm512_and_si512(p1, mask)), _mm512_set1_epi32(aq[2]));
                        const __m512i d3 = _mm512_dpbusd_epi32(_mm512_setzero_si512(),
                            _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16(p1, 4), mask)), _mm512_set1_epi32(aq[3]));
                        dots = _mm512_add_epi32(_mm512_add_epi32(d0, d1), _mm512_add_epi32(d2, d3));
                    } else {
''' + ''.join('    ' + line for line in legacy.splitlines(True)) + '                    }\n'
text = text[:start] + replacement + text[end:]
marker = '''                            const uint8_t packed = w[b].qs[(sb * 4 + k / 4) * 32 + row * 2 + (k % 4) / 2];
                            const uint8_t code = (packed >> (4 * (k % 2))) & 15;'''
replacement = '''                            const int idx = NIBBLE2 ? (sb * 2 + k / 8) * 64 + row * 4 + k % 4
                                                    : (sb * 4 + k / 4) * 32 + row * 2 + (k % 4) / 2;
                            const int shift = NIBBLE2 ? 4 * ((k / 4) % 2) : 4 * (k % 2);
                            const uint8_t code = (w[b].qs[idx] >> shift) & 15;'''
assert marker in text
text = text.replace(marker, replacement, 1)
marker = 'static void qK_r8_get_scale_min('
wrapper = '''template <bool IQ3>
static void ggml_gemv_iq_r16_q8_K(
        int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    if (iq_r16_paired_nibbles()) {
        ggml_gemv_iq_r16_q8_K_impl<IQ3, true>(n, s, bs, vx, vy, nr, nc);
    } else {
        ggml_gemv_iq_r16_q8_K_impl<IQ3, false>(n, s, bs, vx, vy, nr, nc);
    }
}

'''
assert marker in text
text = text.replace(marker, wrapper + marker, 1)
shutil.copy2(path, backup)
path.write_text(text)
patch = Path(__file__).with_name('glm-iq-r16-nibble2.patch')
patch.write_text(''.join(difflib.unified_diff(old.splitlines(True), text.splitlines(True),
    fromfile='a/ggml/src/ggml-cpu/repack.cpp', tofile='b/ggml/src/ggml-cpu/repack.cpp')))
