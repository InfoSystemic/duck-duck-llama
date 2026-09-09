#!/usr/bin/env python3
"""Check private Q8 instruction schedules against the installed arithmetic."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time
from model_measurement_guard import ModelMeasurementGuard
from qwen_split_trial import inference_snapshot

BASE = Path(__file__).resolve().parent
ENGINE = BASE.parents[1] / 'engines/llama.cpp-q4e-goal-0904'
OUT = BASE / 'results/qwen-q8-chains-0907'
SOURCE = ENGINE / 'ggml/src/ggml-cpu/arch/x86/repack.cpp'
LIB = ENGINE / 'validated-iq-batch3-bin'


def body_for_chains(source, chains):
    start = source.index('void ggml_gemv_q8_0_x16_q8_0(')
    end = source.index('\nvoid ggml_gemv2_q4_K_x16_q8_K(', start)
    body = source[start:end]
    if chains != 1:
        old = '''            __m512i acc = _mm512_setzero_si512();
            const uint8_t * q0 = bp[b].qs; const int8_t * y0 = vy8[b].qs;
            for (int i0 = 0; i0 < 8; i0++) acc = _mm512_dpbusd_epi32(acc, _mm512_loadu_si512((const void *)(q0 + i0 * 64)), GGML_X16_BC32(y0 + 4*i0));'''
        assert body.count(old) == 1
        lines = [f'            __m512i a{i} = _mm512_setzero_si512();' for i in range(chains)]
        lines += ['            const uint8_t * q0 = bp[b].qs; const int8_t * y0 = vy8[b].qs;']
        for i in range(8):
            a = i % chains
            lines += [f'            a{a} = _mm512_dpbusd_epi32(a{a}, _mm512_loadu_si512((const void *)(q0 + {i * 64})), GGML_X16_BC32(y0 + {4 * i}));']
        total = '_mm512_add_epi32(a0, a1)'
        if chains == 4:
            total = '_mm512_add_epi32(_mm512_add_epi32(a0, a1), _mm512_add_epi32(a2, a3))'
        lines += [f'            __m512i acc = {total};']
        body = body.replace(old, '\n'.join(lines))
    return body


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', default=OUT.name)
    parser.add_argument('--batch', action='store_true')
    parser.add_argument('--wide', action='store_true', help='Privately check batches up to eight rows')
    args = parser.parse_args()
    args.batch = args.batch or args.wide
    assert '/' not in args.label and args.label.startswith('qwen-q8-chains-')
    OUT = BASE / 'results' / args.label
    OUT.mkdir(exist_ok=True)
    assert not (OUT / 'result.json').exists(), 'Do not overwrite a completed check'
    original = SOURCE.read_text()
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    header = '#pragma once\n#include <immintrin.h>\n#define GGML_X16_BC32(p) _mm512_set1_epi32(*(const int32_t *)(p))\n'
    for chains in (1, 2, 4):
        header += body_for_chains(original, chains).replace('void ggml_gemv_q8_0_x16_q8_0(', f'static void candidate{chains}(', 1)
    header += '\n#undef GGML_X16_BC32\n'
    (OUT / 'q8-candidates.h').write_text(header)
    fixture_path = BASE / 'qwen-q8-chains-check.cpp'
    batch_header = BASE / 'qwen-q8-batch.h'
    if args.wide:
        wide_header = batch_header.read_text()
        marker = '        default: GGML_ASSERT(nr == 1);'
        assert wide_header.count(marker) == 1
        wide_header = wide_header.replace(marker, ''.join(
            f'        case {n}: return qwen_q8_batch<{n}>(n, s, bs, vx, vy, nr, nc);\n'
            for n in range(5, 9)) + marker)
        batch_header = OUT / 'qwen-q8-batch.h'
        batch_header.write_text(wide_header)
        fixture = fixture_path.read_text()
        fixture = fixture.replace('for (int tokens : {1, 2, 3, 4, 5})',
                                  'for (int tokens : {1, 2, 3, 4, 5, 6, 7, 8, 9})')
        fixture = fixture.replace('for (int batch : {2, 3, 4})',
                                  'for (int batch : {2, 3, 4, 5, 6, 7, 8})')
        fixture = fixture.replace('for (int tokens : {3, 4, 5})', 'for (int tokens : {4, 5, 6, 8, 9})')
        fixture = fixture.replace('mode == 2 ? 2 : 4', 'mode == 2 ? 4 : 8')
        fixture = fixture.replace('mode == 3 ? 4 : mode,', 'mode == 3 ? 8 : mode == 2 ? 4 : 1,')
        fixture_path = OUT / 'qwen-q8-chains-check.cpp'
        fixture_path.write_text(fixture)
    command = ['/usr/bin/c++', '-O3', '-DNDEBUG', '-std=c++17', '-march=native', '-fopenmp',
        '-I' + str(ENGINE / 'ggml/include'), '-I' + str(ENGINE / 'ggml/src'),
        '-I' + str(ENGINE / 'ggml/src/ggml-cpu'), '-I' + str(OUT),
        str(fixture_path), '-L' + str(LIB), '-Wl,-rpath,' + str(LIB),
        '-lggml-cpu', '-lggml-base', '-o', str(OUT / 'check')]
    if args.batch:
        command.insert(1, '-DQWEN_Q8_BATCH')
    with (OUT / 'build.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
    print('Q8 schedules compiled', flush=True)
    result = dict(started=time.time(), source_sha256=digest, command=command, passed=False,
                  batch=args.batch, wide=args.wide, fixture_sha256=hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
                  batch_header=str(batch_header), batch_header_sha256=hashlib.sha256(batch_header.read_bytes()).hexdigest())
    try:
        with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
            print('Waiting for any model transition to finish before the fixture', flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = json.loads((BASE / 'results/qwen-q6-trial-0907/state.json').read_text())
            pid = state['current']['pid']
            guard = ModelMeasurementGuard(pid, {pid: 18095}, inference_snapshot)
            guard.assert_idle()
            with (OUT / 'check.jsonl').open('w') as log:
                proc = subprocess.Popen([str(OUT / 'check')], stdout=log, stderr=subprocess.STDOUT)
                try:
                    deadline = time.monotonic() + 120
                    while proc.poll() is None:
                        guard.assert_idle()
                        assert time.monotonic() < deadline
                        time.sleep(0.5)
                    assert proc.returncode == 0, proc.returncode
                finally:
                    if proc.poll() is None:
                        proc.terminate()
                        proc.wait(timeout=10)
            guard.assert_idle()
        rows = [json.loads(line) for line in (OUT / 'check.jsonl').read_text().splitlines()]
        assert rows[0]['correctness_passed']
        assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == digest
        result.update(passed=True, rows=rows)
        print(json.dumps({'passed': True, 'correctness': rows[0], 'result': str(OUT / 'result.json')}), flush=True)
    finally:
        result['finished'] = time.time()
        (OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
