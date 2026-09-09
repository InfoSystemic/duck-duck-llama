#!/usr/bin/env python3
"""Validate shared worker identity, concurrent contexts, and dispatch lifetime."""
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from glm_flash_q8_trial import memory_status
from model_measurement_guard import ModelMeasurementGuard
from qwen_split_trial import inference_snapshot, process_info, sha256
from validate_qwen_quantize_blocks_0909 import equal_files

BASE = Path(__file__).resolve().parent
ENGINE = BASE.parents[1] / 'engines/llama.cpp-q4e-goal-0904'
PINNED = ENGINE / 'validated-iq-batch3-bin'
OUT = BASE / 'results/qwen-shared-dispatch-lifecycle-0909'


def main():
    assert os.sched_getaffinity(0) == {127}
    assert process_info(1219506)['start'] == '103969952'
    manifest_path = BASE / 'results/qwen-shared-dispatch-build-0909b/private-runtime/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert all(manifest['baseline_objects'].values()) and all(manifest['baseline_libraries'].values())
    assert all(sha256(p) == h for p, h in manifest['input_sha256'].items())
    assert all(sha256(p) == h for p, h in manifest['private_source_sha256'].items())
    candidate = {k: Path(v['path']) for k, v in manifest['libraries'].items()}
    assert all(sha256(candidate[k]) == v['sha256'] for k, v in manifest['libraries'].items())
    parent = BASE / 'results/qwen-q6-q8-wide-batch-0907/private-cpu/libggml-cpu.so.0.22.0'
    parent_base = PINNED / 'libggml-base.so.0.22.0'
    assert sha256(parent) == 'c79e19e38acfd6ada3f0637c136cf9cb10253127778cb214528b2f966fb6dfcf'
    assert sha256(parent_base) == 'a8d6ff25ffab12b993c98f11c6d5b4146fc1dd5bd7bebb8c43303842cd959d36'
    preset = BASE / 'qwen-flash-20tps.json'
    fixture = BASE / 'check-qwen-shared-dispatch-lifecycle-0909.cpp'
    inputs = [Path(__file__).resolve(), manifest_path, parent, parent_base, preset, fixture, *candidate.values(),
              BASE / 'model_measurement_guard.py', BASE / 'qwen_split_trial.py',
              BASE / 'glm_flash_q8_trial.py', BASE / 'validate_qwen_quantize_blocks_0909.py']
    result = dict(started=time.time(), controller_pid=os.getpid(), passed=False, checks=[], steps=[],
                  input_sha256={str(p): sha256(p) for p in inputs}, model_loaded=False,
                  scope='Independent private-buffer Meta contexts: exact dyadic reference, alternating and concurrent submissions, 60-worker identity reuse, destruction during a peer graph, and final thread cleanup. No model-rate claim.')
    environment = {k: v for k, v in os.environ.items() if not k.startswith(
        ('GGML_', 'LLAMA_', 'OMP_', 'GOMP_', 'COLD_GRAPH_', 'REPACK_TEST_', 'LD_PRELOAD'))}
    environment.update(json.loads(preset.read_text())['runtime_env'])
    environment.update(GGML_CPU_NUMA_HUGEPAGES='0', GGML_CPU_X16_QUANTIZE_BLOCKS='0')
    guard = ModelMeasurementGuard(1219506, {1219506: 18095}, inference_snapshot)
    owned = None

    def save():
        (OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')

    def interrupt(signum, frame):
        raise InterruptedError('Release only the owned shared-dispatch lifecycle fixture')

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupt)

    def run(command, label, env):
        nonlocal owned
        guard.assert_idle()
        assert memory_status()['MemAvailable'] > 32 << 30
        with (OUT / (label + '.log')).open('w') as log:
            owned = subprocess.Popen(command, cwd=BASE, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
            result['owned'] = dict(pid=owned.pid, label=label)
            save()
            deadline = time.monotonic() + 300
            while owned.poll() is None:
                guard.assert_idle()
                assert time.monotonic() < deadline, label
                time.sleep(.25)
        result['steps'].append(dict(label=label, command=command, exit_code=owned.returncode))
        save()
        assert owned.returncode == 0, (label, owned.returncode)
        print(json.dumps(dict(completed=label)), flush=True)
        return (OUT / (label + '.log')).read_text()

    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        OUT.mkdir(exist_ok=False)
        save()
        try:
            result['idle_gate'] = guard.wait_idle(OUT / 'idle.json', quiet_seconds=15)
            binary = OUT / 'lifecycle-check'
            command = ['g++', '-O3', '-std=c++17', '-march=native', '-fopenmp',
                       '-I' + str(ENGINE / 'ggml/include'), str(fixture), '-L' + str(PINNED),
                       '-lggml', '-lggml-cpu', '-lggml-base', '-ldl', '-pthread', '-o', str(binary)]
            run(command, 'compile', environment)
            reference = None
            for arm in ('parent', 'off', 'on'):
                cpu = parent if arm == 'parent' else candidate['cpu']
                base = parent_base if arm == 'parent' else candidate['base']
                enabled = arm == 'on'
                output = OUT / (arm + '.f32')
                env = dict(environment, LD_LIBRARY_PATH=str(cpu.parent) + ':' + str(PINNED),
                           GGML_CPU_NUMA_SHARED_DISPATCH=str(int(enabled)))
                log = run(['taskset', '-c', '0-127', str(binary), str(int(enabled)), str(output)], arm, env)
                loaded_cpu, = re.findall(r'^LIFECYCLE_CPU_LIBRARY (.+)$', log, re.M)
                loaded_base, = re.findall(r'^LIFECYCLE_BASE_LIBRARY (.+)$', log, re.M)
                assert Path(loaded_cpu).resolve() == cpu.resolve()
                assert Path(loaded_base).resolve() == base.resolve()
                proof, = [json.loads(line) for line in log.splitlines() if line.startswith('{')]
                assert proof['passed'] and proof['runs'] == 24 and proof['exact_values'] == 209920
                assert proof['shared_workers'] == enabled and proof['workers_per_context'] == 60
                assert proof['concurrent_rounds'] == 8 and proof['destroyed_context_while_peer_workers_held']
                assert proof['final_threads'] == proof['baseline_threads']
                assert output.stat().st_size == proof['exact_values'] * 4
                creates = re.findall(r'SHARED_NUMA_DISPATCH create group=(\d+) ranks=4', log)
                attaches = re.findall(r'SHARED_NUMA_DISPATCH attach group=(\d+) ranks=4', log)
                destroys = re.findall(r'SHARED_NUMA_DISPATCH destroy group=(\d+)', log)
                if enabled:
                    assert len(creates) == 2 and len(set(creates)) == 2
                    assert attaches == [creates[0]] * 3 + [creates[1]] and destroys == creates
                    assert log.count('(caller dispatch)') == 16
                else:
                    assert not creates and not attaches and not destroys and '(caller dispatch)' not in log
                if reference is None:
                    reference = output
                else:
                    equal_files(reference, output)
                result['checks'].append(dict(arm=arm, proof=proof, full_output_matches_parent=True,
                                             output_sha256=sha256(output), bytes=output.stat().st_size,
                                             runtime_cpu_sha256=sha256(cpu), runtime_base_sha256=sha256(base),
                                             groups_created=creates, groups_attached=attaches, groups_destroyed=destroys))
                save()
            assert all(sha256(p) == h for p, h in result['input_sha256'].items())
            result.update(passed=True, bit_exact=True, binary_sha256=sha256(binary))
        except BaseException as error:
            result['error'] = repr(error)
            raise
        finally:
            if owned is not None and owned.poll() is None:
                os.killpg(owned.pid, signal.SIGTERM)
                owned.wait(timeout=10)
            result['finished'] = time.time()
            result['peer_preserved'] = process_info(1219506)['start'] == '103969952'
            save()
            print(json.dumps(dict(passed=result['passed'], arms=len(result['checks']), error=result.get('error'))), flush=True)


if __name__ == '__main__':
    main()
