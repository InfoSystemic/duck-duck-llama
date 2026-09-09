#!/usr/bin/env python3
"""Check private Qwen barrier outputs while preserving the resident service."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time

from glm_flash_q8_trial import memory_status
from model_measurement_guard import ModelMeasurementGuard, read_service
from qwen_split_trial import inference_snapshot, process_info, process_environment, runtime_environment, sha256

BASE = Path(__file__).resolve().parent
PINNED = BASE.parents[1] / 'engines/llama.cpp-q4e-goal-0904/validated-iq-batch3-bin'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('label')
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--guard-pid', type=int, required=True)
    parser.add_argument('--start-ticks', required=True)
    args = parser.parse_args()
    assert re.fullmatch(r'qwen-dissemination-[A-Za-z0-9_-]+', args.label)
    os.umask(0o077)
    out = BASE / 'results' / args.label
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        build_path, manifest_path = args.build / 'result.json', args.build / 'private-cpu/manifest.json'
        build, manifest = [json.loads(path.read_text()) for path in (build_path, manifest_path)]
        assert build['build_completed'] and not build.get('error')
        assert build['baseline_link_identical'] and build['unpatched_text_identical']
        assert all(sha256(path) == digest for path, digest in build['input_sha256'].items())
        assert all(sha256(path) == digest for path, digest in manifest['private_source_sha256'].items())
        assert all(sha256(args.build / (name + '-check')) == digest for name, digest in build['binary_sha256'].items())
        library = Path(build['library'])
        assert sha256(library) == build['library_sha256'] == manifest['library_sha256']
        parent = Path(json.loads(Path(manifest['parent_manifest']).read_text())['library'])
        assert sha256(parent) == build['parent_sha256'] == 'c79e19e38acfd6ada3f0637c136cf9cb10253127778cb214528b2f966fb6dfcf'
        preset_path = BASE / 'qwen-flash-20tps.json'
        preset = json.loads(preset_path.read_text())
        protected = process_info(args.guard_pid)
        assert protected['start'] == args.start_ticks
        assert runtime_environment(process_environment(args.guard_pid)) == preset['runtime_env']
        assert memory_status()['MemAvailable'] > 32 << 30
        guard = ModelMeasurementGuard(args.guard_pid, {args.guard_pid:18095}, inference_snapshot)
        out.mkdir(exist_ok=False)
        sources = {str(path):sha256(path) for path in [Path(__file__).resolve(), build_path, manifest_path,
                   preset_path, BASE / 'model_measurement_guard.py', BASE / 'qwen_split_trial.py']}
        result = dict(started=time.time(), passed=False, cpu_library=str(library), cpu_sha256=sha256(library),
                      protected_pid=args.guard_pid, protected_process=protected, source_sha256=sources,
                      checks=[], numa_checks=[], steps=[], model_loaded=False, model_gain_established=False,
                      scope='Component correctness only; all timing data excluded while unrelated host jobs are busy')
        def save():
            (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        def validate():
            guard.assert_idle()
            now = process_info(args.guard_pid)
            assert now['start'] == protected['start'] and now['command'] == protected['command']
        def run(command, label, env):
            validate()
            log = out / (label + '.log')
            with log.open('w') as stream:
                child = subprocess.Popen(command, cwd=BASE, env=env, stdout=stream, stderr=subprocess.STDOUT)
                result['owned_component'] = dict(pid=child.pid, label=label)
                save()
                try:
                    deadline = time.monotonic() + 600
                    while child.poll() is None:
                        guard.assert_idle()
                        assert time.monotonic() < deadline, label
                        time.sleep(.25)
                    assert child.returncode == 0, (label, child.returncode)
                finally:
                    if child.poll() is None:
                        child.terminate()
                        child.wait(timeout=10)
            result['steps'].append(dict(label=label, command=command, exit_code=child.returncode))
            save()
            print(json.dumps(dict(completed=label)), flush=True)
            return log.read_text()
        def calls(log, expected, enabled):
            loaded, = re.findall(r'^BARRIER_CPU_LIBRARY (.+)$', log, re.M)
            assert Path(loaded).resolve() == expected.resolve()
            count, = map(int, re.findall(r'^DISSEMINATION_CALLS (\d+)$', log, re.M))
            assert (count > 0) == enabled
            return count
        save()
        try:
            result['idle_gate'] = guard.wait_idle(out / 'waiting-for-idle.json', quiet_seconds=30)
            env = {k:v for k,v in os.environ.items() if not k.startswith(('GGML_', 'LLAMA_', 'OMP_', 'GOMP_', 'REPACK_TEST_', 'NUMA_REDUCE_TEST_'))}
            env.update(preset['runtime_env'])
            graph_env = {k:v for k,v in env.items() if not k.startswith('GGML_CPU_NUMA_')}
            graph_env.update(REPACK_TEST_DENSE_WORK_SHARING='1', REPACK_TEST_SMALL_BATCHES='1', REPACK_TEST_REPEATS='1')
            arms = [('parent', False, False, True), ('off', False, False, False), ('on', True, False, False),
                    ('parent-unfused', False, True, True), ('on-unfused', True, True, False)]
            for label, enabled, unfused, is_parent in arms:
                expected = parent if is_parent else library
                trial = dict(graph_env, LD_LIBRARY_PATH=str(expected.parent) + ':' + str(PINNED),
                             GGML_CPU_DISSEMINATION_BARRIER=str(int(enabled)), GGML_CPU_DISSEMINATION_AUDIT='1',
                             GGML_CPU_DISABLE_FUSION=str(int(unfused)))
                for kind in ('q8', 'q6'):
                    for padded in (False, True):
                        trial.pop('REPACK_TEST_PADDED', None)
                        if padded:
                            trial['REPACK_TEST_PADDED'] = '1'
                        name = f'{label}-{kind}-padded{int(padded)}'
                        log = run(['taskset', '-c', '0-127', str(args.build / (kind + '-check')), 'q8'], name, trial)
                        count = calls(log, expected, enabled)
                        hashes = re.findall(r'^PASS .* hash=([^\n]+)', log, re.M)
                        assert len(hashes) == 216, (name, len(hashes))
                        result['checks'].append(dict(label=label, kind=kind, padded=padded, unfused=unfused,
                                                     cases=216, hashes=hashes, barrier_calls=count, log_sha256=sha256(out / (name + '.log'))))
                        save()
            for kind in ('q8', 'q6'):
                for padded in (False, True):
                    for unfused, count in ((False, 3), (True, 2)):
                        group = [row for row in result['checks'] if row['kind'] == kind and row['padded'] == padded and row['unfused'] == unfused]
                        assert len(group) == count and all(row['hashes'] == group[0]['hashes'] for row in group)
            result['bit_exact'] = True
            for label, enabled, is_parent in [('parent', False, True), ('off', False, False), ('on', True, False)]:
                expected = parent if is_parent else library
                trial = dict(env, LD_LIBRARY_PATH=str(expected.parent) + ':' + str(PINNED),
                             GGML_CPU_DISSEMINATION_BARRIER=str(int(enabled)), GGML_CPU_DISSEMINATION_AUDIT='1')
                for name, extra, summary, length in [('threads', [str(out / ('threads-' + label + '.txt'))], 'cases=40 failures=0', 40),
                                                    ('reduce', ['--fused-graph'], 'Fused graph: 32 cases, 0 failures', 32)]:
                    log = run(['taskset', '-c', '0-127', str(args.build / (name + '-check')), *extra], name + '-' + label, trial)
                    count = calls(log, expected, enabled)
                    assert summary in log
                    hashes = re.findall(r' hash=([0-9a-f]+)$', log, re.M)
                    assert len(hashes) == length
                    result['numa_checks'].append(dict(label=label, name=name, hashes=hashes, cases=length, barrier_calls=count, passed=True))
                    save()
            for name in ('threads', 'reduce'):
                group = [row for row in result['numa_checks'] if row['name'] == name]
                assert len(group) == 3 and all(row['hashes'] == group[0]['hashes'] for row in group)
            assert all(sha256(path) == digest for path, digest in sources.items())
            assert all(sha256(path) == digest for path, digest in build['input_sha256'].items())
            assert all(sha256(path) == digest for path, digest in manifest['private_source_sha256'].items())
            validate()
            result.update(passed=True, service_after=read_service(18095))
            print(json.dumps(dict(passed=True, graph_arms=len(result['checks']),
                                  graph_case_executions=sum(row['cases'] for row in result['checks']),
                                  numa_arms=len(result['numa_checks']), model_loaded=False)), flush=True)
        except BaseException as error:
            result['error'] = repr(error)
            raise
        finally:
            result['finished'] = time.time()
            save()


if __name__ == '__main__':
    main()
