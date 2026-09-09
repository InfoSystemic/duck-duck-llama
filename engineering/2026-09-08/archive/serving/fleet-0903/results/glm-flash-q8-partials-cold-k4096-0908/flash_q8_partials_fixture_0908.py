#!/usr/bin/env python3
"""Generate private NR=1 Q8 integer-chain controls from the measured source."""
from pathlib import Path

BASE = Path(__file__).resolve().parent
SOURCE = BASE / 'results/glm-flash-q8-batch-0908/private-cpu/repack-x86.cpp'
SOURCE_SHA = '133c8d2018a62cf063d76291160e1e0817d00b24c0d1eb879e88375af0fd746c'


def generate(destination, sha256):
    assert sha256(SOURCE) == SOURCE_SHA
    source = SOURCE.read_text()
    name = 'ggml_gemv_q8_0_x16_q8_0'
    start = source.index('void ' + name + '(')
    end = source.index('\nvoid ggml_gemv2_q4_K_x16_q8_K(', start)
    body = source[start:end]
    body = body.replace('void ' + name + '(',
        'template<int PARTS> __attribute__((noinline)) static void candidate_q8(', 1)
    dispatch = '    if (nr > 1) return qwen_q8_batch_dispatch(n, s, bs, vx, vy, nr, nc);\n'
    assert body.count(dispatch) == 1
    body = body.replace(dispatch, '    static_assert(PARTS == 1 || PARTS == 2 || PARTS == 4);\n')
    original = '            for (int i0 = 0; i0 < 8; i0++) acc = _mm512_dpbusd_epi32(acc, _mm512_loadu_si512((const void *)(q0 + i0 * 64)), GGML_X16_BC32(y0 + 4*i0));'
    assert body.count(original) == 1
    replacement = '''            if constexpr (PARTS == 1) {
ORIGINAL
            } else {
                __m512i partial[PARTS];
                for (int p = 0; p < PARTS; ++p) partial[p] = _mm512_setzero_si512();
                for (int i0 = 0; i0 < 8; ++i0) {
                    partial[i0 % PARTS] = _mm512_dpbusd_epi32(partial[i0 % PARTS],
                        _mm512_loadu_si512((const void *)(q0 + i0 * 64)), GGML_X16_BC32(y0 + 4*i0));
                }
                acc = partial[0];
                for (int p = 1; p < PARTS; ++p) acc = _mm512_add_epi32(acc, partial[p]);
            }'''.replace('ORIGINAL', original)
    body = body.replace(original, replacement)
    header = ('// NR=1 diagnostic, copied from source SHA256 ' + SOURCE_SHA + '\n'
              '// Only integer additions regroup; FP32 block accumulation order is retained.\n'
              '#pragma once\n#include <immintrin.h>\n'
              '#define GGML_X16_BC32(p) _mm512_set1_epi32(*(const int32_t *)(p))\n')
    path = destination / 'cold-q8-partials.h'
    path.write_text(header + body + '\n#undef GGML_X16_BC32\n')
    return dict(source=str(SOURCE), source_sha256=SOURCE_SHA,
                generated=str(path), generated_sha256=sha256(path))
