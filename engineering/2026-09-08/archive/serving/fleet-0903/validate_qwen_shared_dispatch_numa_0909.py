#!/usr/bin/env python3
"""Compare pinned, disabled, and shared-dispatch four-socket expert graphs."""
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
OUT = BASE / 'results/qwen-shared-dispatch-numa-0909'


def main():
    assert os.sched_getaffinity(0) == {127}
    assert process_info(1219506)['start'] == '103969952'
    build_path = BASE / 'results/qwen-shared-dispatch-build-0909b/result.json'
    manifest_path = build_path.parent / 'private-runtime/manifest.json'
    build, manifest = [json.loads(p.read_text()) for p in (build_path, manifest_path)]
    assert build['passed'] and build['build_completed']
    assert all(build['baseline_objects'].values()) and all(build['baseline_libraries'].values())
    assert all(sha256(p) == h for p, h in manifest['input_sha256'].items())
    assert all(sha256(p) == h for p, h in manifest['private_source_sha256'].items())
    candidate = {k: Path(v['path']) for k, v in manifest['libraries'].items()}
    assert all(sha256(candidate[k]) == v['sha256'] for k, v in manifest['libraries'].items())
    parent = BASE / 'results/qwen-q6-q8-wide-batch-0907/private-cpu/libggml-cpu.so.0.22.0'
    parent_base = PINNED / 'libggml-base.so.0.22.0'
    llama = BASE / 'results/qwen-hc-norm-flat-build-0908/private-llama/libllama.so.0.3.0'
    assert sha256(parent) == build['parent_cpu_sha256']
    assert sha256(parent_base) == build['parent_base_sha256']
    assert sha256(llama) == 'd0b2321eae443255dd84a5ad98cda8d3d9bf5540a31891bdc5c03be91061d647'
    preset = BASE / 'qwen-flash-20tps.json'
    source_path = BASE / 'results/qwen-quantize-blocks-validation-0909/numa-check.cpp'
    source = source_path.read_text()
    marker = '    if (argc != 5) return 2;'
    assert source.count(marker) == 1
    source = source.replace(marker, '''    Dl_info base_runtime{};
    if (!dladdr(reinterpret_cast<void *>(ggml_backend_graph_compute), &base_runtime)) return 2;
    std::printf("SHARED_BASE_LIBRARY %s\\n", base_runtime.dli_fname);
''' + marker)
    inputs = [Path(__file__).resolve(), build_path, manifest_path, parent, parent_base, llama, preset,
              source_path, *candidate.values(), BASE / 'model_measurement_guard.py', BASE / 'qwen_split_trial.py',
              BASE / 'glm_flash_q8_trial.py', BASE / 'validate_qwen_quantize_blocks_0909.py']
    result = dict(started=time.time(), controller_pid=os.getpid(), passed=False, checks=[], steps=[],
                  input_sha256={str(p): sha256(p) for p in inputs}, model_loaded=False,
                  scope='512 experts, 10 selected, balanced four-NUMA Q6/Q8 graphs. Numeric references and complete output bytes across pinned, disabled, and enabled libraries. Component times do not establish model throughput.')
    environment = {k: v for k, v in os.environ.items() if not k.startswith(
        ('GGML_', 'LLAMA_', 'OMP_', 'GOMP_', 'COLD_GRAPH_', 'REPACK_TEST_', 'LD_PRELOAD'))}
    environment.update(json.loads(preset.read_text())['runtime_env'])
    environment.update(GGML_CPU_NUMA_HUGEPAGES='0', GGML_CPU_X16_QUANTIZE_BLOCKS='0')
    guard = ModelMeasurementGuard(1219506, {1219506: 18095}, inference_snapshot)
    owned = None

    def save():
        (OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')

    def interrupt(signum, frame):
        raise InterruptedError('Stop only the owned shared-dispatch graph fixture')

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupt)

    def run(command, label, env, input_text=None):
        nonlocal owned
        guard.assert_idle()
        assert memory_status()['MemAvailable'] > 32 << 30
        with (OUT / (label + '.log')).open('w') as log:
            owned = subprocess.Popen(command, cwd=BASE, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.PIPE if input_text else subprocess.DEVNULL, start_new_session=True)
            result['owned'] = dict(pid=owned.pid, label=label)
            save()
            if input_text:
                owned.stdin.write(input_text.encode())
                owned.stdin.close()
            deadline = time.monotonic() + 600
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
        fixture = OUT / 'numa-check.cpp'
        fixture.write_text(source)
        result['fixture_sha256'] = sha256(fixture)
        save()
        try:
            result['idle_gate'] = guard.wait_idle(OUT / 'idle.json', quiet_seconds=15)
            for kind in ('q6', 'q8'):
                binary = OUT / ('numa-' + kind)
                command = ['g++', '-O3', '-std=c++17', '-march=native', '-fopenmp', '-DQWEN_MOE_CHECK', '-DQWEN_' + kind.upper() + '_CHECK']
                command += ['-I' + str(ENGINE / p) for p in ('include', 'src', 'ggml/include', 'ggml/src', 'ggml/src/ggml-cpu')]
                command += [str(fixture), '-L' + str(PINNED), '-lllama', '-lggml', '-lggml-cpu', '-lggml-base', '-ldl', '-pthread', '-o', str(binary)]
                run(command, 'compile-' + kind, environment)
                cases = [(1, False), (3, False), (5, False), (64, False), (5, True)] if kind == 'q6' else [(1, False), (5, False), (5, True)]
                for tokens, unfused in cases:
                    reference = None
                    for arm in ('parent', 'off', 'on'):
                        cpu = parent if arm == 'parent' else candidate['cpu']
                        base = parent_base if arm == 'parent' else candidate['base']
                        enabled = arm == 'on'
                        label = f'{kind}-t{tokens}-unfused{int(unfused)}-{arm}'
                        output = OUT / (label + '.f32')
                        env = dict(environment, LD_LIBRARY_PATH=str(cpu.parent) + ':' + str(llama.parent) + ':' + str(PINNED),
                                   GGML_CPU_NUMA_SHARED_DISPATCH=str(int(enabled)), GGML_CPU_DISABLE_FUSION=str(int(unfused)),
                                   GGML_CPU_MOE_GATE_UP_FUSION=str(int(not unfused)), GGML_Q4E_EXPERT_EVEN_SPLIT='1',
                                   COLD_GRAPH_K_PER_SOCKET='2560', COLD_GRAPH_ROWS='640', COLD_GRAPH_MATRICES='4',
                                   COLD_GRAPH_TOKENS=str(tokens), COLD_GRAPH_OUTPUT_PATH=str(output))
                        log = run(['taskset', '-c', '0-127', str(binary), '32', '15', 'fixture', '3'], label, env, 'exit\n')
                        loaded_cpu, = re.findall(r'^QUANT_CPU_LIBRARY (.+)$', log, re.M)
                        loaded_base, = re.findall(r'^SHARED_BASE_LIBRARY (.+)$', log, re.M)
                        assert Path(loaded_cpu).resolve() == cpu.resolve()
                        assert Path(loaded_base).resolve() == base.resolve()
                        ready, = [json.loads(line) for line in log.splitlines() if line.startswith('{') and json.loads(line).get('event') == 'ready']
                        assert ready['experts'] == 512 and ready['used'] == 10 and ready['down_checked']
                        assert ready['weight_type'] == {'q6': 'q6_K', 'q8': 'q8_0'}[kind]
                        assert Path(ready['policy_library']).resolve() == llama.resolve()
                        assert output.stat().st_size == 3 * 4 * 10 * tokens * 2560 * 4
                        creates = re.findall(r'SHARED_NUMA_DISPATCH create group=(\d+) ranks=4', log)
                        attaches = re.findall(r'SHARED_NUMA_DISPATCH attach group=(\d+) ranks=4', log)
                        destroys = re.findall(r'SHARED_NUMA_DISPATCH destroy group=(\d+)', log)
                        if enabled:
                            assert len(creates) == 1 and creates == attaches == destroys
                            assert log.count('(caller dispatch)') == 4
                        else:
                            assert not creates and not attaches and not destroys and '(caller dispatch)' not in log
                        if reference is None:
                            reference = output
                        else:
                            equal_files(reference, output)
                        result['checks'].append(dict(label=label, kind=kind, tokens=tokens, unfused=unfused,
                                                     arm=arm, output_sha256=sha256(output), bytes=output.stat().st_size,
                                                     full_output_matches_parent=True, shared_dispatch_executed=enabled,
                                                     runtime_cpu_sha256=sha256(cpu), runtime_base_sha256=sha256(base), reference=ready))
                        save()
            assert all(sha256(p) == h for p, h in result['input_sha256'].items())
            assert sha256(fixture) == result['fixture_sha256']
            result.update(passed=True, bit_exact=True, arms=len(result['checks']))
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
