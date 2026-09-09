#!/usr/bin/env python3
"""Check a copy-free Qwen residual combination through CPU and NUMA graphs."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from model_measurement_guard import ModelMeasurementGuard, read_service
from qwen_split_trial import inference_snapshot, process_info, sha256

BASE = Path(__file__).resolve().parent
ENGINE = BASE.parent.parent / 'engines/llama.cpp-q4e-goal-0904'
PINNED = ENGINE / 'validated-iq-batch3-bin'
OUT = BASE / 'results/qwen-hc-combine-proof-0908'


def main():
    assert os.sched_getaffinity(0) == {127}
    assert process_info(1219506)['start'] == '103969952'
    cpu = BASE / 'results/qwen-q6-q8-wide-batch-0907/private-cpu/libggml-cpu.so.0.22.0'
    assert sha256(cpu) == 'c79e19e38acfd6ada3f0637c136cf9cb10253127778cb214528b2f966fb6dfcf'
    source = BASE / 'check-qwen-hc-combine-0908.cpp'
    header = BASE / 'qwen-hc-combine-0908.h'
    preset = BASE / 'qwen-flash-20tps.json'
    inputs = [Path(__file__).resolve(), source, header, cpu, preset,
              BASE / 'model_measurement_guard.py', BASE / 'qwen_split_trial.py', PINNED / 'libggml-base.so.0']
    command = ['g++', '-O3', '-std=c++17', '-march=native', '-ffp-contract=off', '-fopenmp']
    command += ['-I' + str(ENGINE / path) for path in ('ggml/include', 'ggml/src', 'ggml/src/ggml-cpu')]
    command += [str(source), str(cpu), '-L' + str(PINNED), '-lggml', '-lggml-base', '-ldl', '-pthread', '-o', str(OUT / 'check')]
    result = dict(started=time.time(), controller_pid=os.getpid(), passed=False, steps=[], checks=[],
                  compile_command=command, input_sha256={str(p):sha256(p) for p in inputs},
                  scope='Graph correctness only. No runtime integration or performance claim.')
    guard = ModelMeasurementGuard(1219506, {1219506:18095}, inference_snapshot)
    owned = None

    def save():
        (OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')

    def interrupt(signum, frame):
        raise InterruptedError('Stop only the owned correctness check')

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupt)
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        OUT.mkdir(exist_ok=False)
        save()
        try:
            result['idle_gate'] = guard.wait_idle(OUT / 'idle.json', quiet_seconds=15)
            for p in inputs[:3]:
                (OUT / p.name).write_bytes(p.read_bytes())
            with (OUT / 'compile.log').open('w') as log:
                compiled = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=120)
            result['compile_exit'] = compiled.returncode
            assert compiled.returncode == 0
            environment = {k:v for k,v in os.environ.items() if not k.startswith(('GGML_', 'LLAMA_', 'OMP_', 'GOMP_'))}
            environment.update(json.loads(preset.read_text())['runtime_env'])
            environment['LD_LIBRARY_PATH'] = str(cpu.parent) + ':' + str(PINNED)
            for mode in ('cpu', 'numa'):
                trial = dict(environment)
                if mode == 'cpu':
                    trial = {k:v for k,v in trial.items() if not k.startswith('GGML_CPU_NUMA_')}
                cmd = ['taskset', '-c', '0-127', str(OUT / 'check'), mode]
                with (OUT / (mode + '.log')).open('w') as log:
                    owned = subprocess.Popen(cmd, env=trial, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    result['owned'] = dict(pid=owned.pid, mode=mode)
                    save()
                    deadline = time.monotonic() + 600
                    while owned.poll() is None:
                        guard.assert_idle()
                        assert time.monotonic() < deadline, mode
                        time.sleep(.25)
                result['steps'].append(dict(mode=mode, command=cmd, exit_code=owned.returncode))
                save()
                assert owned.returncode == 0, (mode, owned.returncode)
                rows = [json.loads(line) for line in (OUT / (mode + '.log')).read_text().splitlines() if line.startswith('{')]
                assert len(rows) == 1
                row = rows[0]
                assert row['passed'] and row['bit_exact'] and row['scalar_exact'] and row['inputs_preserved']
                assert row['mode'] == mode and row['cases'] == (288 if mode == 'cpu' else 96)
                assert Path(row['cpu_library']).resolve() == cpu.resolve()
                assert Path(row['base_library']).resolve() == (PINNED / 'libggml-base.so.0').resolve()
                result['checks'].append(row)
                save()
                print(json.dumps(dict(completed=mode, cases=row['cases'], outputs=row['outputs'])), flush=True)
            assert all(sha256(path) == digest for path,digest in result['input_sha256'].items())
            result.update(passed=True, binary_sha256=sha256(OUT / 'check'))
        except BaseException as error:
            result['error'] = repr(error)
            raise
        finally:
            if owned is not None and owned.poll() is None:
                owned.terminate(); owned.wait(timeout=10)
            result['finished'] = time.time()
            result['peer_preserved'] = process_info(1219506)['start'] == '103969952'
            result['peer_after'] = read_service(18095)
            save()
            print(json.dumps({key:result.get(key) for key in ('passed', 'error', 'peer_preserved')}), flush=True)


if __name__ == '__main__':
    main()
