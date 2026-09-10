#!/usr/bin/env python3
"""Validate the integrated Q6 NR=1 path through expert and four-NUMA graphs."""
import fcntl
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import time

from benchmark_qwen_q6 import host_cpu, wait_background
from model_measurement_guard import ModelMeasurementGuard, read_service
from qwen_split_trial import inference_snapshot, process_info, runtime_environment, sha256
from run_qwen_q6_packed_single_benchmark_0908 import aggregate_cpu_seconds, children_cpu_seconds

BASE = Path(__file__).resolve().parent
ENGINE = BASE.parents[1] / 'engines/llama.cpp-q4e-goal-0904'
OUT = BASE / 'results/qwen-q6-single-validation-0910'

AUDIT = r'''
    Dl_info cpu_runtime{}, base_runtime{};
    if (!dladdr(reinterpret_cast<void *>(ggml_backend_cpu_init), &cpu_runtime)) std::abort();
    if (!dladdr(reinterpret_cast<void *>(ggml_backend_graph_compute), &base_runtime)) std::abort();
    std::printf("SINGLE_CPU_LIBRARY %s\n", cpu_runtime.dli_fname);
    std::printf("SINGLE_BASE_LIBRARY %s\n", base_runtime.dli_fname);
    std::atexit([] {
        using counter_fn = uint64_t (*)();
        const auto counter = reinterpret_cast<counter_fn>(dlsym(RTLD_DEFAULT, "ggml_cpu_qwen_q6_single_count"));
        std::printf("SINGLE_CALLS %llu\n", (unsigned long long) (counter ? counter() : 0));
    });
'''


def fixtures():
    small_path = BASE / 'check-qwen-q6-expert-tiles-0909b.cpp'
    small = small_path.read_text().replace('ggml_cpu_qwen_q6_moe_tile_count', 'ggml_cpu_qwen_q6_single_count')
    old = '    for (int i = 0; i < GGML_MAX_N_THREADS; ++i) pp.cpumask[i] = i < threads && CPU_ISSET(i,&available);'
    assert small.count(old) == 1
    small = small.replace(old, '''    int assigned = 0;
    for (int i = 0; i < GGML_MAX_N_THREADS; ++i) {
        pp.cpumask[i] = CPU_ISSET(i,&available) && assigned < threads;
        if (pp.cpumask[i]) ++assigned;
    }
    if (assigned != threads) std::abort();''')
    old = '(shift+j+t*(disjoint ? used : 3))%experts'
    assert small.count(old) == 1
    small = small.replace(old, '((shift+j+t*(disjoint ? used : 3))*73+19)%experts')
    numa_path = BASE / 'results/qwen-q6-512-expert-validation-0907/numeric.cpp'
    numa = numa_path.read_text()
    marker = 'int main(int argc, char ** argv) {'
    assert numa.count(marker) == 1
    numa = numa.replace(marker, marker + AUDIT)
    old = '(u + 10 * t + (m % 2 ? 490 : 250) + probe) % experts'
    assert numa.count(old) == 1
    numa = numa.replace(old, '(u + 2 * t + (m % 2 ? 508 : 250) + probe) % experts')
    return small_path, small, numa_path, numa


def main():
    assert os.sched_getaffinity(0) == {127} and not os.environ.get('LD_PRELOAD')
    assert process_info(1219506)['start'] == '103969952'
    build_path = BASE / 'results/qwen-q6-single-build-0910b/result.json'
    manifest_path = build_path.parent / 'private-cpu/manifest.json'
    parent_path = BASE / 'results/qwen-get-rows-runtime-0909/result.json'
    build, manifest, parent = [json.loads(path.read_text()) for path in (build_path, manifest_path, parent_path)]
    assert build['passed'] and build['finished'] and build['baseline_object_identical'] and build['baseline_library_identical']
    library = Path(build['library'])
    assert sha256(library) == build['library_sha256'] == manifest['library_sha256']
    assert sha256(parent['cpu']) == parent['cpu_sha256'] == build['parent_sha256']
    inputs = {}
    for record in (build, manifest, parent):
        for key in ('input_sha256', 'private_source_sha256', 'sources'):
            for path, digest in record.get(key, {}).items():
                assert sha256(path) == digest, path
                inputs[path] = digest
    small_path, small_text, numa_path, numa_text = fixtures()
    paths = [Path(__file__).resolve(), build_path, manifest_path, parent_path, small_path, numa_path,
             BASE / 'benchmark_qwen_q6.py', BASE / 'model_measurement_guard.py', BASE / 'qwen_split_trial.py',
             BASE / 'run_qwen_q6_packed_single_benchmark_0908.py']
    inputs.update({str(path): sha256(path) for path in paths})
    guard = ModelMeasurementGuard(1219506, {1219506: 18095}, inference_snapshot)
    owned = None
    environment = {key: value for key, value in os.environ.items()
                   if key not in runtime_environment(os.environ) and not key.startswith(('COLD_GRAPH_', 'REPACK_TEST_')) and key != 'LD_PRELOAD'}
    environment.update(parent['runtime_env'])
    runtime_dir = Path(parent['runtime_directory'])

    def interrupt(signum, frame):
        raise InterruptedError('Release only the owned Q6 validation process')

    def gate_timeout(signum, frame):
        raise TimeoutError('No qualified background window within 300 seconds')

    signal.signal(signal.SIGALRM, gate_timeout)
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupt)
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guard.assert_idle()
        OUT.mkdir(exist_ok=False)
        small, numa = OUT / 'expert-check.cpp', OUT / 'numa-check.cpp'
        small.write_text(small_text)
        numa.write_text(numa_text)
        result = dict(started=time.time(), passed=False, model_loaded=False, controller_pid=os.getpid(),
                      input_sha256=inputs, private_source_sha256={str(path): sha256(path) for path in (small, numa)},
                      cpu_sha256=sha256(library), parent_sha256=parent['cpu_sha256'], steps=[], checks=[], timings=[], numa_checks=[],
                      scope='Exact graph comparisons with 512 experts and ten routes, 1/4/15 workers, padding and extra consumers. Two-socket timing uses scattered synthetic route IDs and rotating weights. Four-NUMA graphs include Q6 gate/up, unchanged Q8 down and numeric references. No model throughput or IMC claim.')

        def save(stage=None):
            if stage:
                result['stage'] = stage
            (OUT / 'result.json').write_text(json.dumps(result, indent=2)+'\n')

        def run(command, label, env=None, input_text=None, timed=False):
            nonlocal owned
            guard.assert_idle()
            record = dict(label=label, command=command)
            if timed:
                save('waiting for CPU gate: '+label)
                signal.alarm(300)
                try:
                    record['background_gate'] = wait_background(guard, 1219506, 4, OUT / (label+'-background.json'))
                finally:
                    signal.alarm(0)
                before, aggregate_before, child_before = host_cpu(), aggregate_cpu_seconds(), children_cpu_seconds()
                started = time.monotonic()
            log_path = OUT / (label+'.log')
            with log_path.open('w') as log:
                owned = subprocess.Popen(command, cwd=BASE, env=env, stdout=log, stderr=subprocess.STDOUT,
                                         stdin=subprocess.PIPE if input_text else subprocess.DEVNULL, start_new_session=True)
                result['owned'] = dict(pid=owned.pid, label=label)
                save('running '+label)
                if input_text:
                    owned.stdin.write(input_text.encode())
                    owned.stdin.close()
                deadline = time.monotonic()+600
                while owned.poll() is None:
                    guard.assert_idle()
                    assert time.monotonic() < deadline, label
                    time.sleep(.2)
            if timed:
                elapsed = time.monotonic()-started
                aggregate = (aggregate_cpu_seconds()-aggregate_before-(children_cpu_seconds()-child_before))/elapsed
                loads = []
                for pid, value in host_cpu().items():
                    old = before.get(pid)
                    if pid == owned.pid or old is None or old[1] != value[1]:
                        continue
                    cores = (value[0]-old[0])/os.sysconf('SC_CLK_TCK')/elapsed
                    if cores > 0:
                        loads.append(dict(pid=pid, name=value[2], cores=cores))
                cores = max(aggregate, sum(row['cores'] for row in loads))
                record.update(other_host_cores=cores, background_within_gate=cores <= 4,
                              largest_background=sorted(loads, key=lambda row: -row['cores'])[:6])
            record.update(exit_code=owned.returncode, log_sha256=sha256(log_path))
            result['steps'].append(record)
            save()
            assert owned.returncode == 0, (label, owned.returncode)
            print(json.dumps(dict(completed=label, qualified=record.get('background_within_gate'))), flush=True)
            return log_path.read_text(), record

        def trial_env(expected, enabled, audit=True):
            return dict(environment, LD_LIBRARY_PATH=str(expected.parent)+':'+str(runtime_dir),
                        GGML_CPU_QWEN_Q6_PACKED_SINGLE=str(int(enabled)), GGML_CPU_QWEN_Q6_PACKED_SINGLE_AUDIT=str(int(audit)))

        def expert(label, enabled, is_parent=False, threads=15, socket=0, timed=False):
            expected = Path(parent['cpu']) if is_parent else library
            output = OUT / (label+'.bin')
            env = trial_env(expected, enabled, not timed)
            log, step = run(['taskset', '-c', f'{socket*16}-{socket*16+14}', str(OUT/'expert-check'),
                             '--timing' if timed else str(output), str(threads)], label, env, timed=timed)
            maps = re.findall(r'^CPU_LIBRARY (.+)$', log, re.M)
            assert len(maps) == 1 and Path(maps[0]).resolve() == expected.resolve()
            assert 'MOE_COUNTER_PRESENT '+str(int(not is_parent))+'\n' in log
            rows = [json.loads(line.removeprefix('MOE_CASE ')) for line in log.splitlines() if line.startswith('MOE_CASE ')]
            summary = [json.loads(line.removeprefix('MOE_SUMMARY ')) for line in log.splitlines() if line.startswith('MOE_SUMMARY ')]
            assert summary == [dict(cases=6 if timed else 30, failures=0, weights_preserved=True)]
            assert len(rows) == (6 if timed else 30) and all(row['passed'] for row in rows)
            assert all((row['selected_calls'] > 0) == (enabled and not timed) for row in rows)
            assert ('QWEN_Q6_PACKED_SINGLE nr=1 storage=unchanged' in log) == enabled
            record = dict(label=label, enabled=enabled, parent=is_parent, threads=threads, socket=socket, rows=rows)
            if timed:
                record.update(other_host_cores=step['other_host_cores'], background_within_gate=step['background_within_gate'])
            else:
                record.update(output_sha256=sha256(output), output_bytes=output.stat().st_size)
            return record

        save('compiling graph fixtures')
        try:
            flags = ['g++', '-O3', '-std=c++17', '-march=native', '-fopenmp', '-DGGML_USE_OPENMP']
            flags += ['-I'+str(ENGINE/path) for path in ('include', 'src', 'ggml/include', 'ggml/src', 'ggml/src/ggml-cpu')]
            links = ['-L'+str(runtime_dir), '-Wl,-rpath,'+str(runtime_dir), '-lllama', '-lggml', '-lggml-cpu', '-lggml-base', '-ldl', '-pthread']
            run([*flags, str(small), *links, '-o', str(OUT/'expert-check')], 'compile-expert', environment)
            run([*flags, '-DQWEN_MOE_CHECK', '-DQWEN_Q6_CHECK', str(numa), *links, '-o', str(OUT/'numa-check')], 'compile-numa', environment)
            result['binary_sha256'] = {name: sha256(OUT/name) for name in ('expert-check', 'numa-check')}
            result['idle_gate'] = guard.wait_idle(OUT/'idle.json', quiet_seconds=15)
            for label, enabled, is_parent, threads, socket in (
                    ('parent', False, True, 15, 0), ('off', False, False, 15, 0), ('on', True, False, 15, 0),
                    ('one-worker', True, False, 1, 0), ('four-workers', True, False, 4, 0),
                    ('socket2-parent', False, True, 15, 2), ('socket2-on', True, False, 15, 2)):
                row = expert(label, enabled, is_parent, threads, socket)
                if result['checks']:
                    assert row['output_sha256'] == result['checks'][0]['output_sha256']
                    assert row['output_bytes'] == result['checks'][0]['output_bytes']
                result['checks'].append(row)
                save()
            result['expert_bit_exact'] = True
            for socket in (0, 2):
                for index, enabled in enumerate((False, True, True, False)):
                    row = expert(f'timing-s{socket}-{index}', enabled, not enabled, socket=socket, timed=True)
                    result['timings'].append(row)
                    save()
            comparisons = []
            for socket in (0, 2):
                arms = [arm for arm in result['timings'] if arm['socket'] == socket]
                for i, first in enumerate(arms[0]['rows']):
                    rows = [arm['rows'][i] for arm in arms]
                    assert len({row['hash'] for row in rows}) == 1
                    before = statistics.mean(arm['rows'][i]['median_us'] for arm in arms if arm['parent'])
                    after = statistics.mean(arm['rows'][i]['median_us'] for arm in arms if arm['enabled'])
                    comparisons.append(dict(socket=socket, nr=first['nr'], rotating=first['rotating'],
                                            disjoint=first['disjoint'], parent_us=before, candidate_us=after,
                                            speed_ratio=before/after, qualified=all(arm['background_within_gate'] for arm in arms)))
            rotating = [row for row in comparisons if row['rotating']]
            ratio = math.exp(statistics.mean(math.log(row['speed_ratio']) for row in rotating))
            result.update(comparisons=comparisons, rotating_geomean_speed_ratio=ratio,
                          component_eligible=all(row['qualified'] for row in rotating) and ratio >= 1.02
                          and all(row['speed_ratio'] >= 1/1.05 for row in rotating))
            save()
            for tokens, unfused in ((1, False), (3, False), (5, False), (64, False), (5, True)):
                reference = None
                for enabled in (False, True):
                    expected = library if enabled else Path(parent['cpu'])
                    label = f'numa-t{tokens}-unfused{int(unfused)}-on{int(enabled)}'
                    output = OUT / (label+'.f32')
                    env = trial_env(expected, enabled)
                    env.update(GGML_CPU_DISABLE_FUSION=str(int(unfused)), GGML_CPU_MOE_GATE_UP_FUSION=str(int(not unfused)),
                               COLD_GRAPH_K_PER_SOCKET='2560', COLD_GRAPH_ROWS='640', COLD_GRAPH_MATRICES='4',
                               COLD_GRAPH_TOKENS=str(tokens), COLD_GRAPH_OUTPUT_PATH=str(output))
                    log, _ = run(['taskset', '-c', '0-127', str(OUT/'numa-check'), '32', '15', 'fixture', '3'], label, env, 'exit\n')
                    for kind, wanted in (('CPU', expected), ('BASE', Path(parent['base']))):
                        loaded, = re.findall(r'^SINGLE_'+kind+r'_LIBRARY (.+)$', log, re.M)
                        assert Path(loaded).resolve() == wanted.resolve()
                    calls, = re.findall(r'^SINGLE_CALLS (\d+)$', log, re.M)
                    assert (int(calls) > 0) == enabled
                    ready, = [json.loads(line) for line in log.splitlines() if line.startswith('{') and json.loads(line).get('event') == 'ready']
                    assert ready['experts'] == 512 and ready['used'] == 10 and ready['down_checked'] and ready['weight_type'] == 'q6_K'
                    assert Path(ready['policy_library']).resolve() == Path(parent['llama']).resolve()
                    assert output.stat().st_size == 3*4*10*tokens*2560*4
                    creates = re.findall(r'SHARED_NUMA_DISPATCH create group=(\d+) ranks=4', log)
                    attaches = re.findall(r'SHARED_NUMA_DISPATCH attach group=(\d+) ranks=4', log)
                    destroys = re.findall(r'SHARED_NUMA_DISPATCH destroy group=(\d+)', log)
                    assert len(creates) == 1 and creates == attaches == destroys and log.count('(caller dispatch)') == 4
                    digest = sha256(output)
                    if reference is None:
                        reference = digest
                    else:
                        assert digest == reference
                    result['numa_checks'].append(dict(label=label, tokens=tokens, unfused=unfused, enabled=enabled,
                                                     output_sha256=digest, output_bytes=output.stat().st_size,
                                                     selected_calls=int(calls), reference=ready))
                    save()
            assert all(sha256(path) == digest for path, digest in {**inputs, **result['private_source_sha256']}.items())
            guard.assert_idle()
            result.update(passed=True, bit_exact=True, model_test_eligible=result['component_eligible'])
        except BaseException as error:
            result['error'] = repr(error)
            raise
        finally:
            if owned is not None and owned.poll() is None:
                os.killpg(owned.pid, signal.SIGTERM)
                owned.wait(timeout=15)
            result.update(finished=time.time(), peer_preserved=process_info(1219506)['start'] == '103969952', peer_service=read_service(18095))
            save('finished')
            print(json.dumps(dict(passed=result['passed'], eligible=result.get('model_test_eligible'), error=result.get('error'))), flush=True)


if __name__ == '__main__':
    os.umask(0o077)
    main()
