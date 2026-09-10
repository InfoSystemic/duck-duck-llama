#!/usr/bin/env python3
"""Validate ten-route Q6 scheduling; retain the previously checked HC implementation."""
import difflib
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from benchmark_qwen_q6 import wait_background
from model_measurement_guard import ModelMeasurementGuard, read_service
from qwen_decode_scheduling_transform_0909b import transform
from qwen_split_trial import inference_snapshot, process_info, runtime_environment, sha256
from validate_qwen_q6_expert_tiles_0909b import validate

BASE = Path(__file__).resolve().parent
ENGINE = BASE.parents[1] / 'engines/llama.cpp-q4e-goal-0904'
OUT = BASE / 'results/qwen-decode-scheduling-build-0909b'


def main():
    assert os.sched_getaffinity(0) == {127}
    assert process_info(1219506)['start'] == '103969952'
    paths = [BASE / 'results' / name for name in (
        'qwen-decode-scheduling-build-0909/result.json',
        'qwen-decode-scheduling-build-0909/private-cpu/manifest.json',
        'qwen-get-rows-runtime-0909/result.json',
        'qwen-decode-scheduling-recheck-0909/result.json',
        'qwen-hc-partial-pair-0909.json')]
    previous, manifest, parent, recheck, shape = [json.loads(path.read_text()) for path in paths]
    assert previous['passed'] and previous['finished'] and previous['bit_exact'] and previous['expert_bit_exact']
    assert previous['baseline_object_identical'] and previous['baseline_library_identical']
    assert recheck['passed'] and recheck['finished'] and recheck['model_test_eligible']
    assert recheck['input_sha256'][str(paths[0])] == sha256(paths[0])
    assert shape['passed'] and shape['actual_expert_used_count'] == 10 and shape['actual_expert_count'] == 512
    assert shape['target_q6_weight_shapes'] == [[2560, 160, 512]]
    assert sha256(previous['library']) == previous['library_sha256'] == '45f35424e9213a7cfe279f64f620f189184e9f653f0e14304c99bce0617ea363'
    assert parent['passed'] and parent['finished'] and sha256(parent['cpu']) == parent['cpu_sha256'] == previous['parent_sha256']
    inputs = {}
    for record in (previous, manifest, parent, recheck, shape):
        for key in ('input_sha256', 'private_source_sha256', 'sources'):
            for path, digest in record.get(key, {}).items():
                assert sha256(path) == digest, path
                inputs[path] = digest
    original = BASE / 'results/qwen-q6-q8-wide-batch-0907/private-cpu/repack.cpp'
    prior_source = Path(manifest['compile_command'][manifest['compile_command'].index('-c')+1])
    changed = transform(original.read_text())
    expected = prior_source.read_text().replace('experts != 512 || used != 8 || tokens < 1 || tokens > 8',
                                               'experts != 512 || used != 10 || tokens < 1 || tokens > 8')
    expected = expected.replace('QWEN_Q6_MOE_TILE rows=%lld\\n', 'QWEN_Q6_MOE_TILE rows=%lld routes=10 k=2560 nc=160\\n')
    assert changed == expected and changed != prior_source.read_text()
    paths += [Path(__file__).resolve(), original, prior_source,
              BASE / 'qwen_decode_scheduling_transform_0909b.py',
              BASE / 'validate_qwen_q6_expert_tiles_0909b.py',
              BASE / 'check-qwen-q6-expert-tiles-0909b.cpp']
    inputs.update({str(path): sha256(path) for path in paths})
    guard = ModelMeasurementGuard(1219506, {1219506: 18095}, inference_snapshot)
    owned = None

    def interrupted(signum, frame):
        raise InterruptedError('Release only the owned ten-route build or component test')

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupted)
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guard.assert_idle()
        OUT.mkdir(exist_ok=False)
        private = OUT / 'private-cpu'
        private.mkdir()
        source = private / 'repack.cpp'
        source.write_text(changed)
        header = private / 'qwen-q8-hc-ordered-k-0909.h'
        header.write_bytes((prior_source.parent / header.name).read_bytes())
        (private / 'repack.cpp.patch').write_text(''.join(difflib.unified_diff(
            original.read_text().splitlines(True), changed.splitlines(True), fromfile=str(original), tofile='private/repack.cpp')))
        result = dict(started=time.time(), passed=False, build_completed=False, model_loaded=False,
                      controller_pid=os.getpid(), input_sha256=inputs,
                      private_source_sha256={str(path): sha256(path) for path in (source, header)},
                      parent_sha256=parent['cpu_sha256'], implementation_parent_sha256=previous['library_sha256'],
                      route_guard_only=True, inherited_hc_validation=str(paths[0]), inherited_hc_recheck=str(paths[3]),
                      actual_expert_used_count=10, steps=[],
                      scope='Only the Q6 tile eligibility changes from eight to ten routes, plus its once-only execution marker. HC code, arithmetic, weights, base/llama/server libraries and defaults are unchanged. Ten-route graph correctness and rotating timings must pass before a model trial. Earlier HC correctness and retiming are inherited with verified source hashes.')

        def save():
            (OUT / 'result.json').write_text(json.dumps(result, indent=2)+'\n')

        def run(args, name, env=None):
            nonlocal owned
            guard.assert_idle()
            log_path = OUT / (name+'.log')
            with log_path.open('w') as log:
                owned = subprocess.Popen(args, cwd=ENGINE/'build-goal', env=env, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                result['owned'] = dict(pid=owned.pid, step=name)
                save()
                deadline = time.monotonic()+600
                while owned.poll() is None:
                    guard.assert_idle()
                    assert time.monotonic() < deadline, name
                    time.sleep(.25)
            result['steps'].append(dict(name=name, command=args, exit_code=owned.returncode, log_sha256=sha256(log_path)))
            save()
            assert owned.returncode == 0, name
            print(json.dumps(dict(completed=name)), flush=True)
            return log_path.read_text()

        save()
        try:
            obj = private / 'repack.cpp.o'
            command = list(manifest['compile_command'])
            command[command.index('-c')+1] = str(source)
            old_object = command[command.index('-o')+1]
            command[command.index('-o')+1] = str(obj)
            run(command, 'private-compile')
            library = private / 'libggml-cpu.so.0.22.0'
            link = [str(obj) if value == old_object else value for value in manifest['link_command']]
            link[link.index('-o')+1] = str(library)
            run(link, 'private-link')
            (private / 'libggml-cpu.so.0').symlink_to(library.name)
            (private / 'libggml-cpu.so').symlink_to('libggml-cpu.so.0')
            result.update(build_completed=True, library=str(library), library_sha256=sha256(library))
            new_manifest = dict(input_sha256=inputs, private_source_sha256=result['private_source_sha256'],
                                library=str(library), library_sha256=sha256(library),
                                compile_command=command, link_command=link,
                                parent_manifest=str(paths[1]), route_guard_only=True, scope=result['scope'])
            (private / 'manifest.json').write_text(json.dumps(new_manifest, indent=2)+'\n')
            runtime_dir = Path(parent['runtime_directory'])
            flags = ['/usr/bin/c++', '-O3', '-std=c++17', '-march=native', '-fopenmp', '-DGGML_USE_OPENMP']
            flags += ['-I'+str(ENGINE/path) for path in ('ggml/include', 'ggml/src', 'ggml/src/ggml-cpu')]
            flags += ['-I'+str(private)]
            links = ['-L'+str(runtime_dir), '-Wl,-rpath,'+str(runtime_dir), '-lggml', '-lggml-cpu', '-lggml-base', '-ldl']
            result['idle_gate'] = guard.wait_idle(OUT / 'waiting-for-idle.json')
            result['background_gate'] = wait_background(guard, 1219506, 4, OUT / 'background-wait.json')
            env = {key: value for key, value in os.environ.items() if key not in runtime_environment(os.environ) and key != 'LD_PRELOAD'}
            env.update(parent['runtime_env'])
            validate(result, save, run, OUT, private, dict(library=parent['cpu']), library, env, runtime_dir, flags, links)
            assert all(sha256(path) == digest for path, digest in {**inputs, **result['private_source_sha256']}.items())
            guard.assert_idle()
            result['passed'] = True
            print(json.dumps(dict(passed=True, cpu_sha256=result['library_sha256'],
                                  expert_eligibility=result['expert_eligibility'], expert_comparisons=result['expert_comparisons'])), flush=True)
        except BaseException as error:
            result['error'] = repr(error)
            raise
        finally:
            if owned is not None and owned.poll() is None:
                os.killpg(owned.pid, signal.SIGTERM)
                owned.wait(timeout=15)
            result.update(finished=time.time(), peer_preserved=process_info(1219506)['start'] == '103969952', peer_service=read_service(18095))
            save()


if __name__ == '__main__':
    os.umask(0o077)
    main()
