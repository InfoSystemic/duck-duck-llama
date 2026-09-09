#!/usr/bin/env python3
"""Validate compiled CPU graphs while preserving an independently served model."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time

from glm_flash_q8_trial import BASE, memory_status
from model_measurement_guard import ModelMeasurementGuard
from qwen_split_trial import inference_snapshot, process_info, sha256

BUILD = BASE / 'results/glm-flash-dissemination-0908b'
OUT = BASE / 'results/glm-flash-dissemination-component-0908'


def main(pid, port):
    os.umask(0o077)
    build_path = BUILD / 'result.json'
    manifest_path = BUILD / 'private-cpu/manifest.json'
    parent_path = BASE / 'results/glm-flash-q8-r8-ordered-k-post-model-0908.json'
    build, manifest, retained = [json.loads(p.read_text()) for p in (build_path, manifest_path, parent_path)]
    assert build['build_completed'] and not build.get('error') and build['mode'] == 'compile_only'
    assert all(sha256(p) == h for p, h in build['input_sha256'].items())
    assert all(sha256(p) == h for p, h in manifest['private_source_sha256'].items())
    assert all(sha256(BUILD / (name + '-check')) == h for name, h in build['binary_sha256'].items())
    assert retained['passed'] and retained['current']['cpu_sha256'] == build['parent_sha256']
    assert build['baseline_link_identical'] and build['unpatched_text_identical']
    library = Path(build['library'])
    assert sha256(library) == build['library_sha256'] == manifest['library_sha256']
    protected = process_info(pid)
    guard = ModelMeasurementGuard(pid, {pid: port}, inference_snapshot)
    assert memory_status()['MemAvailable'] > 32 << 30
    paths = [Path(__file__), build_path, manifest_path, parent_path,
             BASE / 'model_measurement_guard.py', BASE / 'qwen_split_trial.py']
    sources = {str(p): sha256(p) for p in paths}
    OUT.mkdir(exist_ok=False)
    (OUT / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    result = dict(started=time.time(), passed=False, cpu_library=str(library),
                  cpu_sha256=sha256(library), protected_process=protected, protected_pid=pid,
                  protected_port=port, source_sha256=sources, checks=[], numa_checks=[], timings=[], steps=[],
                  scope='Component outputs and timings during guarded idle windows. No Flash model is loaded and no whole-model decode or bandwidth result is claimed.')
    def save():
        (OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    def validate():
        guard.assert_idle()
        now = process_info(pid)
        assert now['start'] == protected['start'] and now['command'] == protected['command']
    def run(command, label, env):
        validate()
        log = OUT / (label + '.log')
        with log.open('w') as stream:
            process = subprocess.Popen(command, cwd=BASE, env=env, stdout=stream, stderr=subprocess.STDOUT)
            result['owned_component'] = dict(pid=process.pid, label=label)
            save()
            try:
                deadline = time.monotonic() + 600
                while process.poll() is None:
                    guard.assert_idle()
                    assert time.monotonic() < deadline, label
                    time.sleep(.25)
                assert process.returncode == 0, (label, process.returncode)
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=10)
        result['steps'].append(dict(label=label, command=command, exit_code=process.returncode))
        save()
        print(json.dumps(dict(completed=label)), flush=True)
        return log.read_text()
    def verified_calls(log, expected, enabled):
        loaded, = re.findall(r'^BARRIER_CPU_LIBRARY (.+)$', log, re.M)
        assert Path(loaded).resolve() == expected.resolve()
        calls, = map(int, re.findall(r'^DISSEMINATION_CALLS (\d+)$', log, re.M))
        assert (calls > 0) == enabled
        return calls
    save()
    try:
        result['idle_gate'] = guard.wait_idle(OUT / 'waiting-for-idle.json', quiet_seconds=30)
        env = {k: v for k, v in os.environ.items() if not k.startswith(('GGML_', 'LLAMA_', 'OMP_', 'GOMP_', 'REPACK_TEST_', 'NUMA_REDUCE_TEST_'))}
        env.update(retained['current']['runtime_env'])
        pinned = Path(retained['current']['pinned_directory'])
        parent_cpu = Path(retained['current']['cpu_library'])
        env['LD_LIBRARY_PATH'] = str(library.parent) + ':' + str(pinned)
        graph_env = {k: v for k, v in env.items() if not k.startswith('GGML_CPU_NUMA_')}
        outputs = []
        for label, enabled, no_fusion, is_parent in [('parent', False, False, True), ('off', False, False, False),
                                                    ('on', True, False, False), ('no-fusion', True, True, False)]:
            expected = parent_cpu if is_parent else library
            trial = dict(graph_env, LD_LIBRARY_PATH=str(expected.parent) + ':' + str(pinned),
                         GGML_CPU_DISSEMINATION_BARRIER=str(int(enabled)), GGML_CPU_DISSEMINATION_AUDIT='1',
                         GGML_CPU_DISABLE_FUSION=str(int(no_fusion)))
            output = OUT / (label + '.bin')
            log = run(['taskset', '-c', '0-127', str(BUILD / 'graph-check'), str(output)], 'graph-' + label, trial)
            calls = verified_calls(log, expected, enabled)
            assert 'SUMMARY cases=98 failures=0' in log
            rows = re.findall(r'^BARRIER_CASE id=(\d+) threads=(\d+) calls=(\d+)$', log, re.M)
            assert len(rows) == 98
            assert all((int(n) > 0) == (enabled and 1 < int(t) <= 64) for _, t, n in rows)
            outputs.append(output)
            result['checks'].append(dict(label=label, cases=98, samples_per_case=3, barrier_calls=calls,
                                         output_bytes=output.stat().st_size, output_sha256=sha256(output)))
            save()
        assert all(p.read_bytes() == outputs[0].read_bytes() for p in outputs[1:])
        result['bit_exact'] = True
        for label, enabled, is_parent in [('parent', False, True), ('off', False, False), ('on', True, False)]:
            expected = parent_cpu if is_parent else library
            trial = dict(env, LD_LIBRARY_PATH=str(expected.parent) + ':' + str(pinned),
                         GGML_CPU_DISSEMINATION_BARRIER=str(int(enabled)), GGML_CPU_DISSEMINATION_AUDIT='1')
            for name, args, summary in [('threads', [str(OUT / ('threads-' + label + '.txt'))], 'cases=40 failures=0'),
                                        ('reduce', ['--fused-graph'], 'Fused graph: 32 cases, 0 failures')]:
                log = run(['taskset', '-c', '0-127', str(BUILD / (name + '-check')), *args], name + '-' + label, trial)
                calls = verified_calls(log, expected, enabled)
                assert summary in log
                hashes = re.findall(r' hash=([0-9a-f]+)$', log, re.M)
                result['numa_checks'].append(dict(label=label, name=name, barrier_calls=calls, hashes=hashes, passed=True))
                save()
        for name, count in [('reduce', 32), ('threads', 40)]:
            hashes = [x['hashes'] for x in result['numa_checks'] if x['name'] == name]
            assert len(hashes) == 3 and all(len(x) == count and x == hashes[0] for x in hashes)
        for index, enabled in enumerate((False, True, True, False)):
            trial = dict(graph_env, GGML_CPU_DISSEMINATION_BARRIER=str(int(enabled)))
            log = run(['taskset', '-c', '0-127', str(BUILD / 'graph-check'), '--timing'], f'timing-{index}-{int(enabled)}', trial)
            rows = re.findall(r'^PASS .* tokens=(\d+) .* weighted=(\d+) .* ms=([\d.]+)$', log, re.M)
            assert len(rows) == 4
            result['timings'].append(dict(enabled=enabled, graph_ms={f'{w}-{t}': float(ms) for t, w, ms in rows}))
        assert all(sha256(p) == h for p, h in sources.items())
        assert all(sha256(p) == h for p, h in build['input_sha256'].items())
        assert all(sha256(p) == h for p, h in manifest['private_source_sha256'].items())
        validate()
        result['passed'] = True
        print(json.dumps(dict(passed=True, checks=result['checks'], numa_checks=result['numa_checks'], timings=result['timings'])), flush=True)
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        result['finished'] = time.time()
        save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--guard-pid', type=int, required=True)
    parser.add_argument('--guard-port', type=int, required=True)
    options = parser.parse_args()
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main(options.guard_pid, options.guard_port)
