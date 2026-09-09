#!/usr/bin/env python3
"""Check the real additional tensor names against the validated x16 path."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from model_measurement_guard import ModelMeasurementGuard
from qwen_split_trial import inference_snapshot, sha256

BASE = Path(__file__).resolve().parent
ROOT = BASE / 'results/qwen-q6-q8-dense-extra-0907'
OUT = ROOT / 'selector-check'


def main():
    OUT.mkdir(exist_ok=False)
    built = json.loads((ROOT / 'result.json').read_text())
    assert built['passed'] and built['dense_extra']
    library = Path(built['library'])
    assert sha256(library) == built['library_sha256']
    state = json.loads((BASE / 'results/qwen-q6-trial-0907/state.json').read_text())
    pid = state['current']['pid']
    guard = ModelMeasurementGuard(pid, {pid: 18095}, inference_snapshot)
    guard.assert_idle()
    fixture = (ROOT / 'graph-check.cpp').read_text()
    old = 'ggml_set_name(w, "blk.0.attn_output.weight");'
    assert fixture.count(old) == 1
    fixture = fixture.replace(old, '''ggml_set_name(w,
        k == 320 ? "blk.0.ssm_out.weight" :
        k == 640 ? "blk.0.ple_key.weight" :
        k == 1536 ? "blk.0.ple_value.weight" :
        k == 2560 ? "blk.0.hc_attn_down.weight" :
        k == 6144 ? "blk.0.hc_ffn_down.weight" : "blk.0.hc_head_down.weight");''')
    old = 'ggml_set_name(u, "blk.0.attn_up.weight");'
    assert fixture.count(old) == 1
    fixture = fixture.replace(old, 'ggml_set_name(u, "output_hc_down.weight");')
    source = OUT / 'graph-check.cpp'
    source.write_text(fixture)
    command = list(next(s['command'] for s in built['steps'] if s['label'] == 'graph-check-compile'))
    command[command.index(str(ROOT / 'graph-check.cpp'))] = str(source)
    command[command.index('-o') + 1] = str(OUT / 'graph-check')
    env = {k: v for k, v in os.environ.items() if not k.startswith(('GGML_', 'LLAMA_', 'OMP_', 'GOMP_', 'REPACK_TEST_'))}
    env.update(LD_LIBRARY_PATH=str(library.parent) + ':' + str(BASE.parents[1] / 'engines/llama.cpp-q4e-goal-0904/validated-iq-batch3-bin'),
               GGML_CPU_X16_Q8_0='1', GGML_CPU_X16_Q8_BATCH='1', GGML_CPU_X16_Q8_DENSE_EXTRA='1',
               REPACK_TEST_DENSE_WORK_SHARING='1', REPACK_TEST_SMALL_BATCHES='1', REPACK_TEST_REPEATS='1')
    result = dict(passed=False, started=time.time(), library_sha256=sha256(library),
                  fixture_sha256=sha256(source), compile_command=command, checks=[])
    def save():
        (OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    def run(command, label):
        guard.assert_idle()
        log = OUT / (label + '.log')
        with log.open('w') as stream:
            proc = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 240
                while proc.poll() is None:
                    guard.assert_idle()
                    assert time.monotonic() < deadline
                    time.sleep(0.5)
                assert proc.returncode == 0, (label, proc.returncode)
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=10)
        print(json.dumps({'completed': label}), flush=True)
        return log
    try:
        run(command, 'compile')
        for padded in (False, True):
            if padded: env['REPACK_TEST_PADDED'] = '1'
            name = 'padded' if padded else 'standard'
            log = run(['taskset', '-c', '48-62', str(OUT / 'graph-check'), 'q8'], name)
            def hashes(path):
                return [s.split(' hash=')[-1] for s in path.read_text().splitlines() if s.startswith('PASS ')]
            actual = hashes(log)
            expected = hashes(ROOT / f'graph-{name}-batch1.log')
            assert len(actual) == 216 and actual == expected
            result['checks'].append(dict(padded=padded, bit_exact_cases=len(actual)))
            save()
        assert sha256(library) == result['library_sha256']
        result['passed'] = True
    finally:
        result['finished'] = time.time()
        save()


if __name__ == '__main__':
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
