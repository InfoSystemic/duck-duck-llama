#!/usr/bin/env python3
"""Prepare a two-accumulator variant of the compact three-row Q5 kernel."""
import argparse
import difflib
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--apply', action='store_true')
a = p.parse_args()
base = Path(__file__).resolve().parent
source = base.parents[1] / 'engines/llama.cpp-glm5n-goal-0904/ggml/src/ggml-cpu/repack.cpp'
old = source.read_text()
if 'GGML_CPU_X16_Q5_COMPACT_P2' in old:
    raise SystemExit('Already applied')
start = old.index('template <int NR, bool COMPACT = false>')
end = old.index('static void ggml_gemv_q5_K_x16_bytes_q8_K(', start)
block = old[start:end]
block = block.replace('template <int NR, bool COMPACT = false>',
                      'template <int NR, bool COMPACT = false, int NP = 4>')
block = block.replace('    const int nb = n / QK_K;',
                      '    static_assert(NP == 2 || NP == 4);\n    const int nb = n / QK_K;', 1)
block = block.replace('dots[NR][4]', 'dots[NR][NP]')
block = block.replace('part < 4', 'part < NP')
block = block.replace('dots[y][chunk & 3]', 'dots[y][chunk & (NP - 1)]')
block = block.replace('const __m512i dot = _mm512_add_epi32(_mm512_add_epi32(dots[y][0], dots[y][1]), _mm512_add_epi32(dots[y][2], dots[y][3]));',
'''__m512i dot = _mm512_add_epi32(dots[y][0], dots[y][1]);
                    if constexpr (NP == 4) dot = _mm512_add_epi32(dot, _mm512_add_epi32(dots[y][2], dots[y][3]));''')
block = block.replace('    if (nr == 3) ggml_gemv_q5_K_x16_bytes_impl<3, true>(n, s, bs, vx, vy, nc);',
'''    static const bool pair_accumulators = [] {
        const char * value = getenv("GGML_CPU_X16_Q5_COMPACT_P2");
        return value && atoi(value) == 1;
    }();
    if (nr == 3 && pair_accumulators) ggml_gemv_q5_K_x16_bytes_impl<3, true, 2>(n, s, bs, vx, vy, nc);
    else if (nr == 3) ggml_gemv_q5_K_x16_bytes_impl<3, true>(n, s, bs, vx, vy, nc);''')
assert 'dots[NR][NP]' in block and 'if constexpr (NP == 4)' in block and '<3, true, 2>' in block
new = old[:start] + block + old[end:]
relative = 'ggml/src/ggml-cpu/repack.cpp'
(base / 'glm-q5-compact-p2.patch').write_text(''.join(difflib.unified_diff(
    old.splitlines(True), new.splitlines(True), 'a/' + relative, 'b/' + relative)))
if a.apply:
    source.with_name(source.name + '.before-goal-q5-compact-p2').write_text(old)
    source.write_text(new)
print('Applied' if a.apply else 'Prepared only')
