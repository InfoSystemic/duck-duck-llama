#!/usr/bin/env python3
"""Check clamped Q8 fusion with all 288 experts active during 64-token prefill."""
import array
import fcntl
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

from model_measurement_guard import ModelMeasurementGuard
from glm_flash_q8_trial import Manager, PORT
from qwen_split_trial import inference_snapshot, sha256

BASE = Path(__file__).resolve().parent
ENGINE = BASE.parents[1] / 'engines/llama.cpp-glm5n-goal-0904'
PINNED = ENGINE / 'validated-chunk16-bin'
OUT = BASE / 'results/glm-flash-q8-clamp-prefill-0908'


def main():
    parent_path = BASE / 'results/glm-flash-q8-clamp-0908/result.json'
    parent = json.loads(parent_path.read_text())
    assert parent['passed']
    library = Path(parent['library'])
    assert sha256(library) == parent['library_sha256']
    baseline = BASE / 'results/glm-flash-q8-experts-0908/private-cpu/libggml-cpu.so.0.22.0'
    assert sha256(baseline) == parent['parent_sha256']
    OUT.mkdir()
    manager = Manager()
    current = manager.validate_current()
    assert not current.get('op_profile_count')
    pid = current['pid']
    guard = ModelMeasurementGuard(pid, {pid: PORT}, inference_snapshot)
    guard.assert_idle()
    source_path = parent_path.parent / 'expert-check.cpp'
    source = source_path.read_text()
    replacements = {
        ': std::vector<int>{1, 3, 4};': ': std::vector<int>{64};',
        '(j + 31 * t + 250) % experts': '(j + 8 * t + 250) % experts',
        '        ggml_backend_tensor_set(ids, routes.data(), 0, ggml_nbytes(ids));':
            '''        std::vector<bool> touched(experts, false);
        for (int32_t id : routes) touched[id] = true;
        const int active = std::count(touched.begin(), touched.end(), true);
        if (experts != 288 || active != 288) std::abort();
        std::printf("ACTIVE_EXPERTS %d\\n", active);
        ggml_backend_tensor_set(ids, routes.data(), 0, ggml_nbytes(ids));''',
    }
    for old, new in replacements.items():
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    fixture, binary = OUT / 'prefill.cpp', OUT / 'prefill'
    fixture.write_text(source)
    result = dict(started=time.time(), passed=False, parent=str(parent_path), parent_sha256=sha256(parent_path),
                  library=str(library), library_sha256=sha256(library), baseline_sha256=sha256(baseline),
                  fixture_sha256=sha256(fixture), experts=288, tokens=64, runs=[], comparisons=[],
                  scope='Capacity and numerical component validation; no bandwidth or model-quality claim.')

    def save():
        (OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')

    def run(command, label, env=None):
        guard.assert_idle()
        path = OUT / (label + '.log')
        with path.open('w') as log:
            proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 600
                while proc.poll() is None:
                    guard.assert_idle()
                    assert time.monotonic() < deadline, label
                    time.sleep(.5)
                assert proc.returncode == 0, (label, proc.returncode, str(path))
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=10)
        print(json.dumps(dict(completed=label)), flush=True)
        return path.read_text()

    try:
        command = ['c++', '-O3', '-std=c++17', '-march=native', '-fopenmp']
        command += ['-I' + str(ENGINE / p) for p in ('ggml/include', 'ggml/src', 'ggml/src/ggml-cpu')]
        command += [str(fixture), '-L' + str(PINNED), '-Wl,-rpath,' + str(PINNED),
                    '-lggml', '-lggml-cpu', '-lggml-base', '-ldl', '-o', str(binary)]
        result['compile_command'] = command
        save()
        run(command, 'compile')
        env = {k: v for k, v in os.environ.items() if not k.startswith(('GGML_', 'LLAMA_', 'OMP_', 'GOMP_', 'REPACK_TEST_'))}
        env.update(GGML_CPU_X16_Q8_0='1', GGML_CPU_X16_Q8_BATCH='1', GGML_CPU_X16_CHUNK_MAX='16',
                   GGML_CPU_X16_Q8_EXPERTS='1', GGML_CPU_MOE_CLAMP_FUSION='1', REPACK_TEST_CLAMP='1',
                   REPACK_TEST_DOWN='1', REPACK_TEST_THREADS='15', REPACK_TEST_REPEATS='1',
                   REPACK_TEST_PERSISTENT_POOL='1', REPACK_TEST_PIN_POOL='1', REPACK_TEST_PADDED='1')
        for name, cpu, enabled in [('baseline', baseline, '0'), ('candidate', library, '1')]:
            directory = OUT / name
            directory.mkdir()
            env.update(LD_LIBRARY_PATH=str(cpu.parent) + ':' + str(PINNED),
                       GGML_CPU_X16_Q8_CLAMP_FUSION=enabled, REPACK_TEST_OUTPUT_DIR=str(directory))
            log = run(['taskset', '-c', '48-62', str(binary), 'q8'], name, env)
            mapped, = re.findall(r'^CPU_LIBRARY (.+)$', log, re.M)
            assert Path(mapped).resolve() == cpu.resolve()
            assert re.findall(r'^ACTIVE_EXPERTS (\d+)$', log, re.M) == ['288'] * 6
            rows = [dict(x.split('=', 1) for x in line.split()[1:]) for line in log.splitlines() if line.startswith('PASS ')]
            assert len(rows) == 3
            assert ('Q8_X16_CLAMP_FUSION_ACTIVE' in log) == (name == 'candidate')
            result['runs'].append(dict(name=name, rows=rows, all_288_active=True))
            save()
        for path in sorted((OUT / 'candidate').glob('*.f32')):
            a, b = array.array('f'), array.array('f')
            baseline_path = OUT / 'baseline' / path.name
            a.frombytes(baseline_path.read_bytes())
            b.frombytes(path.read_bytes())
            assert len(a) == len(b)
            maximum = 0.0
            for x, y in zip(a, b):
                error = abs(x - y) / (1 + abs(x))
                assert math.isfinite(error) and error <= 2e-4
                maximum = max(maximum, error)
            result['comparisons'].append(dict(case=path.stem, values=len(a), max_scaled_error=maximum,
                                              bit_exact=baseline_path.read_bytes() == path.read_bytes()))
        assert len(result['comparisons']) == 3
        assert sha256(parent_path) == result['parent_sha256']
        assert sha256(library) == result['library_sha256'] and sha256(baseline) == result['baseline_sha256']
        result['passed'] = True
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        result['finished'] = time.time()
        save()


if __name__ == '__main__':
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
