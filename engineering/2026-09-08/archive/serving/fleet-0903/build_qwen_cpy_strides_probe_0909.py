#!/usr/bin/env python3
"""Build and check a bounded copy-layout probe without changing model arithmetic."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from benchmark_qwen_q6 import wait_background
from model_measurement_guard import ModelMeasurementGuard
from qwen_split_trial import inference_snapshot, process_info, runtime_environment, sha256

BASE = Path(__file__).resolve().parent
ENGINE = BASE.parents[1] / 'engines/llama.cpp-q4e-goal-0904'
OUT = BASE / 'results/qwen-cpy-strides-probe-build-0909'


def main():
    assert os.sched_getaffinity(0) == {127}
    assert process_info(1219506)['start'] == '103969952'
    runtime_path = BASE / 'results/qwen-get-rows-runtime-0909/result.json'
    runtime = json.loads(runtime_path.read_text())
    assert runtime['passed'] and runtime['finished']
    assert sha256(runtime['cpu']) == runtime['cpu_sha256'] == '12c61b337736ca9210433f57c64ce7fffbf7e4b66920aba9a03f97eaef3fd9b7'
    assert all(sha256(path) == digest for path, digest in runtime['sources'].items())
    sources = [BASE / 'qwen-cpy-strides-probe-0909.cpp', BASE / 'check-qwen-cpy-strides-probe-0909.cpp']
    inputs = [Path(__file__).resolve(), *sources, runtime_path, Path(runtime['cpu']), Path(runtime['base']),
              BASE / 'benchmark_qwen_q6.py', BASE / 'model_measurement_guard.py', BASE / 'qwen_split_trial.py',
              ENGINE / 'ggml/src/ggml-cpu/ops.h', ENGINE / 'ggml/src/ggml-cpu/ggml-cpu-impl.h', ENGINE / 'ggml/include/ggml.h']
    guard = ModelMeasurementGuard(1219506, {1219506: 18095}, inference_snapshot)
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guard.assert_idle()
        OUT.mkdir(exist_ok=False)
        for path in sources:
            (OUT / path.name).write_bytes(path.read_bytes())
        result = dict(started=time.time(), passed=False, model_loaded=False, controller_pid=os.getpid(), steps=[], checks=[],
                      input_sha256={str(path): sha256(path) for path in inputs},
                      private_source_sha256={str(OUT / path.name): sha256(path) for path in sources},
                      scope='LD_PRELOAD diagnostic forwards CPY to the unchanged corrected CPU. It prints only tensor layout metadata after arming. Direct fixtures verify contiguous and strided copies, guard bytes, unchanged inputs, disabled logging, and the forwarded library. No performance result.')

        def save():
            (OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')

        def run(command, name, env=None):
            guard.assert_idle()
            checked = subprocess.run(command, cwd=OUT, env=env, capture_output=True, text=True, timeout=90)
            log = OUT / (name + '.log')
            log.write_text(checked.stdout + checked.stderr)
            result['steps'].append(dict(name=name, command=command, exit_code=checked.returncode, log_sha256=sha256(log)))
            save()
            assert checked.returncode == 0, name
            return checked.stdout + checked.stderr

        save()
        try:
            binary_dir = Path(runtime['runtime_directory'])
            prefix = ['/usr/bin/c++', '-O2', '-std=c++17', '-fPIC', '-fopenmp', '-DGGML_USE_OPENMP',
                      '-I' + str(ENGINE / 'ggml/include'), '-I' + str(ENGINE / 'ggml/src'), '-I' + str(ENGINE / 'ggml/src/ggml-cpu')]
            links = ['-L' + str(binary_dir), '-Wl,-rpath,' + str(binary_dir)]
            probe, fixture = OUT / 'libqwen-cpy-strides.so', OUT / 'cpy-probe-check'
            run([*prefix, '-shared', str(OUT / sources[0].name), *links, '-lggml-base', '-ldl', '-o', str(probe)], 'compile-probe')
            run([*prefix, str(OUT / sources[1].name), *links, '-lggml', '-lggml-cpu', '-lggml-base', '-o', str(fixture)], 'compile-check')
            result['idle_gate'] = guard.wait_idle(OUT / 'waiting-for-idle.json')
            result['background_gate'] = wait_background(guard, 1219506, 4, OUT / 'background-wait.json')
            env = {key: value for key, value in os.environ.items() if key not in runtime_environment(os.environ) and key != 'LD_PRELOAD'}
            env.update(runtime['runtime_env'])
            arm = OUT / 'copy.arm'
            env.update(LD_PRELOAD=str(probe), GGML_CPU_CPY_STRIDE_ARM_FILE=str(arm))
            for enabled in (False, True):
                if enabled:
                    arm.write_text('')
                text = run([str(fixture)], 'armed' if enabled else 'unarmed', env)
                summary, = [json.loads(line.removeprefix('CPY_PROBE_CHECK ')) for line in text.splitlines() if line.startswith('CPY_PROBE_CHECK ')]
                assert summary == dict(cases=3, exact=True, guards_preserved=True, inputs_preserved=True)
                traces = [json.loads(line.removeprefix('CPY_STRIDE_TRACE ')) for line in text.splitlines() if line.startswith('CPY_STRIDE_TRACE ')]
                assert len(traces) == (3 if enabled else 0)
                if enabled:
                    assert [(row['src_contiguous'], row['dst_contiguous']) for row in traces] == [(True, True), (False, True), (True, False)]
                    assert all(Path(row['forward_library']).resolve() == Path(runtime['cpu']).resolve() for row in traces)
                    assert all(row['ith'] == 0 and row['nth'] == 15 and row['same_shape'] for row in traces)
                result['checks'].append(dict(armed=enabled, **summary, traces=traces))
                save()
            arm.unlink()
            assert all(sha256(path) == digest for path, digest in {**result['input_sha256'], **result['private_source_sha256']}.items())
            guard.assert_idle()
            result.update(passed=True, library=str(probe), library_sha256=sha256(probe), fixture_sha256=sha256(fixture),
                          peer_preserved=process_info(1219506)['start'] == '103969952')
            print(json.dumps(dict(passed=True, library=str(probe), library_sha256=result['library_sha256'], direct_cases=6)), flush=True)
        except BaseException as error:
            result['error'] = repr(error)
            raise
        finally:
            result['finished'] = time.time()
            save()


if __name__ == '__main__':
    os.umask(0o077)
    main()
