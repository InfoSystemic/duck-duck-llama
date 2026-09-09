#!/usr/bin/env python3
"""Compare isolated barrier costs while protecting the idle retained runtime."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from glm_flash_q8_trial import BASE, Manager, PORT
from model_measurement_guard import ModelMeasurementGuard
from qwen_split_trial import inference_snapshot, sha256

OUT = BASE / 'results/glm-flash-local-barrier-probe-0908b'


def main():
    os.umask(0o077)
    manager = Manager()
    current = manager.validate_current()
    guard = ModelMeasurementGuard(current['pid'], {current['pid']: PORT}, inference_snapshot)
    guard.assert_idle()
    assert current['ordered_k'] and current['drafts'] == 0
    sources = [Path(__file__), BASE / 'flash-local-barrier-0908.h',
               BASE / 'flash-local-barrier-check-0908.cpp', Path(current['cpu_library'])]
    inputs = {str(path): sha256(path) for path in sources}
    OUT.mkdir(exist_ok=False)
    for path in sources[:3]:
        (OUT / path.name).write_bytes(path.read_bytes())
    result = dict(started=time.time(), passed=False, current_pid=current['pid'],
                  cpu_sha256=current['cpu_sha256'], input_sha256=inputs, steps=[],
                  algorithm_source='https://www.cs.rochester.edu/research/synchronization/pseudocode/ss.html',
                  scope='Isolated synchronization and memory-publication checks. Synthetic timing is not model decode or IMC bandwidth.')
    def save():
        (OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    def run(command, label, env=None):
        manager.validate_current()
        guard.assert_idle()
        log = OUT / (label + '.log')
        with log.open('w') as stream:
            process = subprocess.Popen(command, cwd=BASE, env=env, stdout=stream, stderr=subprocess.STDOUT)
            result['owned_process'] = dict(pid=process.pid, label=label)
            save()
            try:
                deadline = time.monotonic() + 600
                while process.poll() is None:
                    guard.assert_idle()
                    assert time.monotonic() < deadline, label
                    time.sleep(.5)
                assert process.returncode == 0, (label, process.returncode)
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=10)
        result['steps'].append(dict(label=label, command=command, exit_code=process.returncode))
        save()
        print(json.dumps(dict(completed=label)), flush=True)
        return log.read_text()
    save()
    try:
        binary = OUT / 'local-barrier-check'
        run(['c++', '-O3', '-std=c++17', '-march=native', '-fopenmp',
             str(sources[2]), '-pthread', '-ldl', '-o', str(binary)], 'compile')
        result['binary_sha256'] = sha256(binary)
        environment = {k: v for k, v in os.environ.items() if not k.startswith(('GGML_', 'LLAMA_', 'OMP_', 'GOMP_'))}
        log = run(['taskset', '-c', '0-127', str(binary)], 'probe', environment)
        events = [json.loads(line) for line in log.splitlines() if line.startswith('{')]
        result['events'] = events
        startup, = [row for row in events if row['event'] == 'startup']
        assert startup['affinity_cpus'] == startup['omp_cpus'] == 128
        result['gomp_library'] = str(Path(startup['gomp_library']).resolve())
        result['gomp_sha256'] = sha256(result['gomp_library'])
        assert result['gomp_sha256'] == '135f3c8f006d2fe5e68e51281c7974cb991a03de3bfb3593d68d174dfcf854d1'
        correctness, = [row for row in events if row['event'] == 'correctness']
        assert correctness['passed'] and correctness['comparisons'] == 45
        timings = [row for row in events if row['event'] == 'timing']
        assert len(timings) == 60 and all(row['samples'] == 14 for row in timings)
        comparisons = []
        for newer in [row for row in timings if row['mode'] != 'openmp']:
            older, = [row for row in timings if row['mode'] == 'openmp' and
                      all(row[key] == newer[key] for key in ('teams', 'team', 'threads', 'workload'))]
            assert older['hash'] == newer['hash']
            comparisons.append(dict(teams=newer['teams'], team=newer['team'], workload=newer['workload'],
                                    mode=newer['mode'], baseline_ns=older['median_ns'], candidate_ns=newer['median_ns'],
                                    change_percent=100 * (newer['median_ns'] / older['median_ns'] - 1)))
        result['comparisons'] = comparisons
        assert events[-1] == dict(event='done', passed=True)
        assert all(sha256(path) == digest for path, digest in inputs.items())
        manager.validate_current()
        guard.assert_idle()
        result['passed'] = True
        print(json.dumps(dict(passed=True, correctness=correctness, comparisons=comparisons)), flush=True)
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
