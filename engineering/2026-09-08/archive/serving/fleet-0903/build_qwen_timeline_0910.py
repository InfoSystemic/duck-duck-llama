#!/usr/bin/env python3
"""Reproduce the corrected Qwen library and validate diagnostic-only timestamps."""
import difflib
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import time

from model_measurement_guard import ModelMeasurementGuard
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import inference_snapshot, runtime_environment, sha256
from qwen_timeline_transform_0910 import transform
from select_flash_q4_0910c import Manager

BASE = Path(__file__).resolve().parent
ENGINE = BASE.parents[1] / 'engines/llama.cpp-q4e-goal-0904'
OUT = BASE / 'results/qwen-timeline-build-0910'


def main():
    assert os.sched_getaffinity(0) == {127} and not OUT.exists()
    manager = Manager()
    peer = manager.validate_current()
    assert peer['quant'] == 'UD-Q4_K_XL' and set(inference_snapshot()) == {str(peer['pid'])}
    guard = ModelMeasurementGuard(peer['pid'], {peer['pid']: 18131}, inference_snapshot)
    guard.assert_idle()
    parent_path = BASE / 'results/qwen-get-rows-columns-build-0909/private-cpu/manifest.json'
    runtime_path = BASE / 'results/qwen-get-rows-runtime-0909/result.json'
    parent, runtime = [json.loads(p.read_text()) for p in (parent_path, runtime_path)]
    assert runtime['passed'] and sha256(runtime['cpu']) == parent['library_sha256'] == runtime['cpu_sha256']
    inputs = {}
    for record in (parent, runtime):
        for key in ('input_sha256', 'private_source_sha256', 'sources'):
            for path, value in record.get(key, {}).items():
                assert sha256(path) == value
                inputs[path] = value
    commands_path = ENGINE / 'build-goal/compile_commands.json'
    unit, = [row for row in json.loads(commands_path.read_text()) if row['file'].endswith('/ggml-cpu.c')]
    compile_command = shlex.split(unit['command'])
    source = Path(unit['file'])
    original_object = Path(unit['directory']) / compile_command[compile_command.index('-o') + 1]
    assert parent['link_command'].count(str(original_object)) == 1
    paths = [Path(__file__), BASE / 'qwen_timeline_transform_0910.py', parent_path, runtime_path,
             commands_path, source, original_object, BASE / 'model_measurement_guard.py']
    paths += [Path(v) for v in parent['link_command'] if v.endswith(('.o', '.a', '.so.0.22.0')) and Path(v).is_file()]
    inputs.update({str(p): sha256(p) for p in paths})
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guard.assert_idle()
        OUT.mkdir()
        private = OUT / 'private-cpu'
        private.mkdir()
        modified = private / 'ggml-cpu.c'
        original = source.read_text()
        changed = transform(original)
        modified.write_text(changed)
        (private / 'ggml-cpu.c.patch').write_text(''.join(difflib.unified_diff(
            original.splitlines(True), changed.splitlines(True), fromfile=str(source), tofile='private/ggml-cpu.c')))
        result = dict(started=time.time(), passed=False, controller_pid=os.getpid(), steps=[], checks=[],
            parent_cpu_sha256=runtime['cpu_sha256'], input_sha256=inputs,
            private_source_sha256={str(modified): sha256(modified)}, model_loaded=False,
            scope='Diagnostic only. Add monotonic graph and worker-zero stage timestamps to the existing opt-in profiler. Stage work-end can still include internal kernel waits. No arithmetic or scheduling optimization.')
        child = None

        def save():
            atomic_json(OUT / 'result.json', result)

        def cancel(*_):
            raise InterruptedError('Release only this diagnostic build/check')

        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, cancel)

        def run(command, label, environment=None):
            nonlocal child
            guard.assert_idle()
            log = OUT / (label + '.log')
            with log.open('w') as stream:
                child = subprocess.Popen(command, cwd=unit['directory'], env=environment, stdout=stream, stderr=subprocess.STDOUT)
                deadline = time.monotonic() + 600
                while child.poll() is None:
                    guard.assert_idle()
                    assert time.monotonic() < deadline
                    time.sleep(.25)
            result['steps'].append(dict(label=label, command=command, exit_code=child.returncode, log_sha256=sha256(log)))
            save()
            assert child.returncode == 0, label
            print(json.dumps(dict(completed=label)), flush=True)
            return log.read_text()

        save()
        try:
            baseline_object = OUT / 'baseline-ggml-cpu.c.o'
            baseline_compile = list(compile_command)
            baseline_compile[baseline_compile.index('-o') + 1] = str(baseline_object)
            run(baseline_compile, 'baseline-compile')
            assert sha256(baseline_object) == sha256(original_object)
            baseline_library = OUT / 'baseline-libggml-cpu.so.0.22.0'
            baseline_link = [str(baseline_object) if value == str(original_object) else value for value in parent['link_command']]
            baseline_link[baseline_link.index('-o') + 1] = str(baseline_library)
            run(baseline_link, 'baseline-link')
            assert sha256(baseline_library) == runtime['cpu_sha256']
            result.update(baseline_object_identical=True, baseline_library_identical=True)
            obj = private / 'ggml-cpu.c.o'
            candidate_compile = list(compile_command)
            candidate_compile[candidate_compile.index('-o') + 1] = str(obj)
            candidate_compile[candidate_compile.index('-c') + 1] = str(modified)
            run(candidate_compile, 'timeline-compile')
            library = private / 'libggml-cpu.so.0.22.0'
            link = [str(obj) if value == str(original_object) else value for value in parent['link_command']]
            link[link.index('-o') + 1] = str(library)
            run(link, 'timeline-link')
            (private / 'libggml-cpu.so').symlink_to(library.name)
            bundle = OUT / 'runtime'
            bundle.mkdir()
            for old in Path(runtime['server']).parent.iterdir():
                if old.is_file() or old.is_symlink():
                    target = library if old.name.startswith('libggml-cpu.so') else old.resolve()
                    (bundle / old.name).symlink_to(target)
            env = {k:v for k,v in os.environ.items() if k not in runtime_environment(os.environ)}
            env.update(runtime['runtime_env'])
            env.update(LD_LIBRARY_PATH=str(bundle), GGML_CPU_NUMA_DEVICES='0', GGML_CPU_PARALLEL_COPY='1')
            fixture = BASE / 'results/qwen-get-rows-columns-validation-0909/graph-check'
            result['input_sha256'][str(fixture)] = sha256(fixture)
            for mode in ('off', 'on'):
                current = dict(env)
                if mode == 'on':
                    current.update(GGML_CPU_OP_PROFILE='*', GGML_CPU_OP_PROFILE_COUNT='256')
                log = run(['taskset', '-c', '113-127', str(fixture)], 'graph-' + mode, current)
                assert 'Parallel copy: 46 cases, 0 failures' in log
                assert f'GATHER_LIBRARY {bundle}/libggml-cpu.so.0.22.0' in log or f'GATHER_LIBRARY {bundle}/libggml-cpu.so' in log
                headers = re.findall(r'CPU_OP_TIMELINE index=(\d+) cpu=(\d+) graph=(\S+) threads=(\d+) start_us=(\d+) end_us=(\d+)', log)
                nodes = re.findall(r"start_us=(\d+) work_end_us=(\d+) end_us=(\d+) dst_ne=\[", log)
                if mode == 'on':
                    assert len(headers) >= 46 and len(nodes) >= 46
                    assert all(int(start) <= int(end) for _,_,_,_,start,end in headers)
                    assert all(int(start) <= int(work) <= int(end) for start,work,end in nodes)
                    assert {int(h[3]) for h in headers} == {1, 15}
                else:
                    assert not headers and not nodes
                result['checks'].append(dict(mode=mode, graph_cases=46, failures=0, timeline_graphs=len(headers), timeline_nodes=len(nodes)))
            assert all(sha256(p) == h for p,h in inputs.items())
            manager.validate_current()
            result.update(passed=True, library=str(library), library_sha256=sha256(library),
                server=str(bundle / 'llama-server'), runtime_env=dict(runtime['runtime_env'], LD_LIBRARY_PATH=str(bundle)),
                base=runtime['base'], llama=runtime['llama'], server_sha256=runtime['server_sha256'],
                peer_preserved=True, model_gain_established=False)
        except BaseException as error:
            result['error'] = repr(error)
            raise
        finally:
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait(timeout=10)
            result['finished'] = time.time()
            save()


if __name__ == '__main__':
    os.umask(0o077)
    main()
