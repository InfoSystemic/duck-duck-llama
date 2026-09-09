#!/usr/bin/env python3
"""Compare existing Q8 kernels using private, renamed graph fixtures."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from model_measurement_guard import ModelMeasurementGuard
from qwen_split_trial import inference_snapshot, sha256

BASE = Path(__file__).resolve().parent
ENGINE = BASE.parents[1] / 'engines/llama.cpp-q4e-goal-0904'
OUT = BASE / 'results/qwen-q8-expert-x16-probe-0907'
BIN = ENGINE / 'validated-iq-batch3-bin'


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dense-extra', action='store_true')
    args = parser.parse_args()
    if args.dense_extra:
        OUT = BASE / 'results/qwen-q8-dense-extra-probe-0907'
    OUT.mkdir(exist_ok=False)
    manifest_path = BASE / 'results/qwen-q6-q8-fused-batch-0907/private-cpu/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    library = Path(manifest['library'])
    assert sha256(library) == manifest['library_sha256']
    state = json.loads((BASE / 'results/qwen-q6-trial-0907/state.json').read_text())
    pid = state['current']['pid']
    guard = ModelMeasurementGuard(pid, {pid: 18095}, inference_snapshot)
    guard.wait_idle(OUT / 'waiting-for-idle.json')
    source_path = BASE / 'iq2-repack-check.cpp'
    source = source_path.read_text()
    changes = {
        'const int experts = moe ? 12 : 1;': 'const int experts = moe ? 32 : 1;',
        'const int used = moe ? 8 : 1;': 'const int used = moe ? 10 : 1;',
        '"blk.0.ffn_gate_exps.weight"': '"blk.0.attn_output.weight"',
        '"blk.0.ffn_up_exps.weight"': '"blk.0.attn_up.weight"',
        'q8 ? std::vector<std::pair<int, int>>{{320, 512}, {10240, 320}}':
            'q8 ? std::vector<std::pair<int, int>>{{2560, 160}, {160, 2560}}',
        ': std::vector<int>{1, 4, 9};': ': std::vector<int>{1, 3, 4, 5};',
        '        if (work_sharing && moe == dense_work_sharing) continue;':
            '        if (!moe || (shape.first == 160 && fused)) continue;',
    }
    if args.dense_extra:
        changes['q8 ? std::vector<std::pair<int, int>>{{320, 512}, {10240, 320}}'] = (
            'q8 ? std::vector<std::pair<int, int>>{{1536, 2560}, {6144, 2560}, '
            '{10240, 64}, {10240, 128}, {64, 10240}, {128, 10240}, {2560, 10240}, {2560, 2560}}')
        changes['        if (work_sharing && moe == dense_work_sharing) continue;'] = '        if (moe || fused) continue;'
    for old, new in changes.items():
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    fixture = OUT / 'expert-check.cpp'
    fixture.write_text(source)
    result = dict(started=time.time(), passed=False, source_sha256=sha256(source_path),
                  library_sha256=sha256(library), fixture_sha256=sha256(fixture), runs=[],
                  dense_extra=args.dense_extra,
                  scope='Prototype only: renamed tensors exercise existing Q8 kernels on selected graph shapes. '
                        'This does not alter the model or its runtime selection policy.')
    def save():
        (OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    env = {k: v for k, v in os.environ.items() if not k.startswith(('GGML_', 'LLAMA_', 'OMP_', 'GOMP_', 'REPACK_TEST_'))}
    env.update(LD_LIBRARY_PATH=str(library.parent) + ':' + str(BIN),
               GGML_CPU_X16_Q8_0='1', GGML_CPU_X16_Q8_BATCH='1', REPACK_TEST_DOWN='1',
               REPACK_TEST_THREADS='15', REPACK_TEST_REPEATS='30', REPACK_TEST_TIMING_MEDIAN='1',
               REPACK_TEST_PERSISTENT_POOL='1', REPACK_TEST_PIN_POOL='1')
    def run(command, label):
        guard.assert_idle()
        log = OUT / (label + '.log')
        with log.open('w') as stream:
            proc = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 240
                while proc.poll() is None:
                    guard.assert_idle()
                    assert time.monotonic() < deadline, label
                    time.sleep(0.5)
                assert proc.returncode == 0, (label, proc.returncode)
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=10)
        print(json.dumps({'completed': label}), flush=True)
        return log
    try:
        command = ['c++', '-O3', '-std=c++17', '-march=native', '-fopenmp',
            *['-I' + str(ENGINE / p) for p in ('ggml/include', 'ggml/src', 'ggml/src/ggml-cpu')],
            str(fixture), '-L' + str(library.parent), '-L' + str(BIN),
            '-Wl,-rpath,' + env['LD_LIBRARY_PATH'], '-lggml', '-lggml-cpu', '-lggml-base',
            '-ldl', '-o', str(OUT / 'expert-check')]
        result['compile_command'] = command
        save()
        run(command, 'compile')
        for rep, enabled in enumerate(('0', '1', '1', '0')):
            env['GGML_CPU_X16_Q8_0' if args.dense_extra else 'GGML_CPU_X16_ATTN3D'] = enabled
            log = run(['taskset', '-c', '48-62', str(OUT / 'expert-check'), 'q8'],
                      f'run-{rep}-x16-{enabled}')
            rows = [dict(item.split('=', 1) for item in line.split()[1:])
                    for line in log.read_text().splitlines() if line.startswith('PASS ')]
            assert len(rows) == (32 if args.dense_extra else 12), len(rows)
            result['runs'].append(dict(x16=enabled == '1', rows=rows))
            save()
        assert sha256(source_path) == result['source_sha256']
        assert sha256(library) == result['library_sha256']
        result['passed'] = True
    finally:
        result['finished'] = time.time()
        save()


if __name__ == '__main__':
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
