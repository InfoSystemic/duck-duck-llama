#!/usr/bin/env python3
"""Probe parallel Q8 input work with the original floating accumulation order."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from glm_flash_q8_trial import BASE, Manager, PORT
from model_measurement_guard import ModelMeasurementGuard
from qwen_split_trial import inference_snapshot, sha256

OUT = BASE / 'results/glm-flash-q8-r8-ordered-k-probe-0908'
ENGINE = BASE.parents[1] / 'engines/llama.cpp-glm5n-goal-0904'


def main():
    os.umask(0o077)
    manager = Manager()
    current = manager.validate_current()
    guard = ModelMeasurementGuard(current['pid'], {current['pid']: PORT}, inference_snapshot)
    guard.assert_idle()
    assert current['rms_guard'] and current['drafts'] == 0
    cpu, pinned = Path(current['cpu_library']), Path(current['pinned_directory'])
    assert sha256(cpu) == current['cpu_sha256']
    fixture = BASE / 'flash-q8-r8-ordered-k-check-0908.cpp'
    header = BASE / 'flash-q8-r8-ordered-k-0908.h'
    sources = [Path(__file__), fixture, header, cpu,
               BASE / 'results/glm-flash-q8-clamp-0908/private-cpu/repack.cpp',
               BASE / 'results/glm-flash-q8-sum16-0908/private-cpu/repack-x86.cpp']
    sources += list((ENGINE / 'ggml/include').glob('*.h'))
    sources += list((ENGINE / 'ggml/src/ggml-cpu').glob('*.h'))
    inputs = {str(path): sha256(path) for path in sources}
    OUT.mkdir(exist_ok=False)
    for path in [Path(__file__), fixture, header, BASE / 'glm_flash_q8_trial.py']:
        (OUT / path.name).write_bytes(path.read_bytes())
    result = dict(started=time.time(), passed=False, current_pid=current['pid'],
                  cpu_sha256=current['cpu_sha256'], input_sha256=inputs, steps=[],
                  scope='One-socket component equivalence and timing. No full-model throughput or bandwidth claim.')
    def save():
        (OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    def run(command, label, env=None):
        manager.validate_current()
        guard.assert_idle()
        log = OUT / (label + '.log')
        with log.open('w') as stream:
            process = subprocess.Popen(command, cwd=BASE, env=env, stdout=stream, stderr=subprocess.STDOUT)
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
        result['steps'].append(dict(label=label, command=command))
        save()
        print(json.dumps(dict(completed=label)), flush=True)
        return log.read_text()
    save()
    try:
        binary = OUT / 'ordered-k-check'
        command = ['c++', '-O3', '-std=c++17', '-march=native', '-fopenmp']
        command += ['-I' + str(ENGINE / path) for path in ('include', 'ggml/include', 'ggml/src', 'ggml/src/ggml-cpu')]
        command += [str(fixture), str(cpu), str((pinned / 'libggml-base.so').resolve()), '-ldl', '-pthread', '-o', str(binary)]
        run(command, 'compile')
        result['binary_sha256'] = sha256(binary)
        environment = {k: v for k, v in os.environ.items() if not k.startswith(('GGML_', 'LLAMA_', 'OMP_', 'GOMP_'))}
        environment['LD_LIBRARY_PATH'] = str(cpu.parent) + ':' + str(pinned)
        output = OUT / 'outputs.bin'
        log = run(['taskset', '-c', '48-62', str(binary), str(output)], 'probe', environment)
        events = [json.loads(line) for line in log.splitlines() if line.startswith('{')]
        result['events'] = events
        assert Path(events[0]['path']).resolve() == cpu.resolve()
        correctness, = [row for row in events if row['event'] == 'correctness']
        assert correctness['passed'] and correctness['cases'] == 486
        assert output.stat().st_size == correctness['values'] * 4
        result['output_bytes'], result['output_sha256'] = output.stat().st_size, sha256(output)
        timings = [row for row in events if row['event'] == 'timing']
        assert len(timings) == 16 and all(row['samples'] == 18 for row in timings)
        comparisons = []
        for newer in [row for row in timings if row['ordered']]:
            older, = [row for row in timings if not row['ordered'] and
                      all(row[key] == newer[key] for key in ('k', 'nr', 'rotating', 'weight_bytes'))]
            assert older['hash'] == newer['hash']
            comparisons.append(dict(k=newer['k'], nr=newer['nr'], rotating=newer['rotating'],
                                    baseline_us=older['median_us'], candidate_us=newer['median_us'],
                                    change_percent=100 * (newer['median_us'] / older['median_us'] - 1)))
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
