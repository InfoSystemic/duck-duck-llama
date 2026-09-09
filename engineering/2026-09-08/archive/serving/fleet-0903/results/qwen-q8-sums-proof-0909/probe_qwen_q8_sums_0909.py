#!/usr/bin/env python3
"""Check the existing exact Q8 sum variants against Qwen's selected CPU library."""
import ast
import difflib
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from benchmark_qwen_q6 import host_cpu, wait_background
from model_measurement_guard import ModelMeasurementGuard, read_service
from qwen_split_trial import inference_snapshot, process_info, sha256

BASE = Path(__file__).resolve().parent
ENGINE = BASE.parents[1] / 'engines/llama.cpp-q4e-goal-0904'
PINNED = ENGINE / 'validated-iq-batch3-bin'
OUT = BASE / 'results/qwen-q8-sums-proof-0909'


def main():
    assert os.sched_getaffinity(0) == {127}
    assert process_info(1219506)['start'] == '103969952'
    parent_path = BASE / 'results/qwen-q6-q8-wide-batch-0907/private-cpu/manifest.json'
    parent = json.loads(parent_path.read_text())
    cpu = Path(parent['library'])
    source = cpu.parent / 'repack-x86.cpp'
    assert sha256(cpu) == parent['library_sha256'] == 'c79e19e38acfd6ada3f0637c136cf9cb10253127778cb214528b2f966fb6dfcf'
    assert all(sha256(path) == digest for path, digest in parent['private_source_sha256'].items())
    generator = BASE / 'probe_flash_q8_sums_0908.py'
    fixture = BASE / 'flash-q8-sums-check-0908.cpp'
    guard = ModelMeasurementGuard(1219506, {1219506:18095}, inference_snapshot)
    owned = None

    def interrupt(signum, frame):
        raise InterruptedError('Stop only this owned Q8 sum check')

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupt)
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guard.assert_idle()
        OUT.mkdir(exist_ok=False)
        # Reuse the reviewed transform, with Qwen's actual source as its input.
        tree = ast.parse(generator.read_text())
        function, = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'generate']
        namespace = dict(SOURCE=source, SOURCE_SHA=sha256(source), OUT=OUT, sha256=sha256, difflib=difflib)
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(generator), 'exec'), namespace)
        header = namespace['generate']()
        for path in (Path(__file__).resolve(), fixture, generator):
            (OUT / path.name).write_bytes(path.read_bytes())
        inputs = [Path(__file__).resolve(), fixture, generator, header, source, cpu, parent_path,
                  BASE / 'benchmark_qwen_q6.py', BASE / 'model_measurement_guard.py',
                  BASE / 'qwen_split_trial.py', (PINNED / 'libggml-base.so.0').resolve()]
        inputs += [path for root in ('ggml/include', 'ggml/src') for path in (ENGINE / root).rglob('*.h')]
        result = dict(started=time.time(), controller_pid=os.getpid(), passed=False, steps=[],
                      cpu_sha256=sha256(cpu), input_sha256={str(path):sha256(path) for path in inputs},
                      scope='Exact integer and FP32 checks against C79. Timings are one-core component evidence, not model throughput or bandwidth.')

        def save():
            (OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')

        def run(command, label, env=None, monitor_background=False):
            nonlocal owned
            guard.assert_idle()
            with (OUT / (label + '.log')).open('w') as log:
                owned = subprocess.Popen(command, cwd=BASE, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                result['owned'] = dict(pid=owned.pid, label=label)
                save()
                deadline = time.monotonic() + 600
                before, then = host_cpu(), time.monotonic()
                samples = []
                while owned.poll() is None:
                    guard.assert_idle()
                    now = time.monotonic()
                    assert now < deadline, label
                    if monitor_background and now - then >= 2:
                        after = host_cpu()
                        elapsed = time.monotonic() - then
                        cores = sum((ticks - before[pid][0]) / os.sysconf('SC_CLK_TCK') / elapsed
                                    for pid, (ticks, birth, _) in after.items()
                                    if pid not in (owned.pid, os.getpid()) and pid in before and before[pid][1] == birth)
                        samples.append(dict(seconds=elapsed, other_host_cores=cores))
                        before, then = after, time.monotonic()
                    time.sleep(.25)
            result['steps'].append(dict(label=label, command=command, exit_code=owned.returncode))
            if monitor_background:
                result['timing_background_samples'] = samples
                result['component_timing_qualified'] = bool(samples) and all(row['other_host_cores'] <= 4 for row in samples)
            save()
            assert owned.returncode == 0, label
            print(json.dumps(dict(completed=label)), flush=True)
            return (OUT / (label + '.log')).read_text()

        save()
        try:
            result['idle_gate'] = guard.wait_idle(OUT / 'idle.json', quiet_seconds=15)
            binary = OUT / 'q8-sums-check'
            command = ['/usr/bin/c++', '-O3', '-std=c++17', '-march=native', '-I' + str(OUT)]
            command += ['-I' + str(ENGINE / path) for path in ('include', 'ggml/include', 'ggml/src', 'ggml/src/ggml-cpu')]
            command += [str(fixture), str(cpu), str((PINNED / 'libggml-base.so.0').resolve()), '-ldl', '-pthread', '-o', str(binary)]
            run(command, 'compile')
            result['binary_sha256'] = sha256(binary)
            result['background_gate'] = wait_background(guard, 1219506, 4, OUT / 'background.json')
            environment = {k:v for k,v in os.environ.items() if not k.startswith(('GGML_', 'LLAMA_', 'OMP_', 'GOMP_'))}
            environment['LD_LIBRARY_PATH'] = str(cpu.parent) + ':' + str(PINNED)
            output = OUT / 'outputs.bin'
            log = run(['taskset', '-c', '48', str(binary), str(output)], 'probe', environment, True)
            events = [json.loads(line) for line in log.splitlines() if line.startswith('{')]
            assert Path(events[0]['path']).resolve() == cpu.resolve()
            check, = [row for row in events if row['event'] == 'correctness']
            assert check['passed'] and check['sum_cases'] == 18448 and check['matrix_cases'] == 240
            assert events[-1] == dict(event='done', passed=True)
            rows = [row for row in events if row['event'] == 'timing']
            assert len(rows) == 80 and all(row['samples'] == 34 for row in rows)
            assert output.stat().st_size == check['compared_values'] * 4
            comparisons = []
            for row in rows:
                if row['mode'] not in ('sad', 'inline16'):
                    continue
                references = [value for value in rows if all(value[key] == row[key] for key in ('k', 'nc', 'rotating'))
                              and value['mode'] in ('library', 'copy')]
                assert len(references) == 2 and all(value['hash'] == row['hash'] for value in references)
                comparisons.append(dict(k=row['k'], nc=row['nc'], rotating=row['rotating'], mode=row['mode'],
                                        speed_ratio={value['mode']:value['median_us'] / row['median_us'] for value in references}))
            assert all(sha256(path) == digest for path, digest in result['input_sha256'].items())
            result.update(passed=True, events=events, correctness=check, comparisons=comparisons,
                          output_bytes=output.stat().st_size, output_sha256=sha256(output))
            print(json.dumps(dict(passed=True, correctness=check,
                                  component_timing_qualified=result['component_timing_qualified'])), flush=True)
        except BaseException as error:
            result['error'] = repr(error)
            raise
        finally:
            if owned is not None and owned.poll() is None:
                os.killpg(owned.pid, signal.SIGTERM)
                owned.wait(timeout=10)
            result['finished'] = time.time()
            result['peer_preserved'] = process_info(1219506)['start'] == '103969952'
            result['peer_after'] = read_service(18095)
            save()


if __name__ == '__main__':
    main()
