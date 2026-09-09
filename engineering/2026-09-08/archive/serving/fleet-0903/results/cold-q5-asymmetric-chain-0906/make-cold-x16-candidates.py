#!/usr/bin/env python3
"""Copy current Full arithmetic into a private fixture with optional prefetch."""
import hashlib
from pathlib import Path


def generate(engine, destination):
    path = Path(engine) / 'ggml/src/ggml-cpu/arch/x86/repack.cpp'
    source = path.read_text()
    blocks = []
    for kind, following in ((4, 5), (5, 6)):
        name = f'ggml_gemv_q{kind}_K_x16_q8_K'
        start = source.index(f'void {name}(')
        end = source.index(f'\nvoid ggml_gemv_q{following}_K_x16_q8_K(', start)
        body = source[start:end]
        body = body.replace(f'void {name}(', f'template<int AHEAD> static void candidate_q{kind}(', 1)
        marker = 'for (int b = 0; b < nb; b++) {'
        assert body.count(marker) == 1
        prefetch = '''
            if constexpr (AHEAD > 0) {
                if (g * nb + b + AHEAD < (nc / 16) * nb) {
                    const char * future = (const char *) (bp + b + AHEAD);
                    for (size_t offset = 0; offset < sizeof(*bp); offset += 64) {
                        _mm_prefetch(future + offset, _MM_HINT_T0);
                    }
                }
            }
'''
        body = body.replace(marker, marker + prefetch, 1)
        blocks.append(body)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    header = ('// Generated from unchanged Full source SHA256 ' + digest + '\n'
              '#pragma once\n#include <immintrin.h>\n'
              '#define GGML_X16_BC32(p) _mm512_set1_epi32(*(const int32_t *)(p))\n')
    header += '\n'.join(blocks) + '\n#undef GGML_X16_BC32\n'
    Path(destination).write_text(header)
    return dict(source=str(path), source_sha256=digest,
                generated_sha256=hashlib.sha256(header.encode()).hexdigest())


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('engine')
    parser.add_argument('destination')
    args = parser.parse_args()
    print(json.dumps(generate(args.engine, args.destination)))
