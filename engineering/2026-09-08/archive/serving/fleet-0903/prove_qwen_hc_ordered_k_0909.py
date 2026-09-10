#!/usr/bin/env python3
"""Check ordered-K arithmetic against the unchanged Qwen HC projection runtime."""
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
OUT = BASE / 'results/qwen-hc-ordered-k-proof-0909'


def main():
    assert os.sched_getaffinity(0) == {127}
    assert process_info(1219506)['start'] == '103969952'
    runtime_path = BASE / 'results/qwen-get-rows-runtime-0909/result.json'
    runtime = json.loads(runtime_path.read_text())
    assert runtime['passed'] and runtime['finished']
    assert sha256(runtime['cpu']) == runtime['cpu_sha256'] == '12c61b337736ca9210433f57c64ce7fffbf7e4b66920aba9a03f97eaef3fd9b7'
    assert all(sha256(path) == digest for path, digest in runtime['sources'].items())
    sources = [BASE / 'flash-q8-r8-ordered-k-0908.h', BASE / 'check-qwen-hc-ordered-k-0909.cpp']
    headers = [ENGINE / ('ggml/' + name) for name in ('include/ggml.h', 'include/ggml-cpu.h',
               'src/ggml-cpu/repack.h', 'src/ggml-cpu/simd-mappings.h', 'src/ggml-cpu/ggml-cpu-impl.h')]
    inputs = [Path(__file__).resolve(), *sources, *headers, runtime_path, Path(runtime['cpu']), Path(runtime['base']),
              BASE / 'benchmark_qwen_q6.py', BASE / 'model_measurement_guard.py', BASE / 'qwen_split_trial.py']
    guard = ModelMeasurementGuard(1219506, {1219506: 18095}, inference_snapshot)
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guard.assert_idle()
        OUT.mkdir(exist_ok=False)
        header = sources[0].read_text().replace('flash_q8_r8_', 'qwen_q8_hc_')
        assert header.count('[3]') == 3
        header = header.replace('[3]', '[8]')
        header_path = OUT / 'qwen-q8-hc-ordered-k-0909.h'
        header_path.write_text(header)
        source = OUT / sources[1].name
        source.write_bytes(sources[1].read_bytes())
        result = dict(started=time.time(), passed=False, model_loaded=False, controller_pid=os.getpid(), steps=[],
                      input_sha256={str(path): sha256(path) for path in inputs},
                      private_source_sha256={str(path): sha256(path) for path in (source,header_path)},
                      scope='Arithmetic proof only. Extend the existing ordered-K header from three to eight activation rows. Compare 96 graph cases on the unchanged corrected Qwen CPU, including packed-four activation quantization, input preservation, and guards. No performance claim or model change.')

        def save():
            (OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')

        def run(command, name, env=None, timeout=120):
            guard.assert_idle()
            checked = subprocess.run(command, cwd=OUT, env=env, capture_output=True, text=True, timeout=timeout)
            log = OUT / (name + '.log')
            log.write_text(checked.stdout + checked.stderr)
            result['steps'].append(dict(name=name, command=command, exit_code=checked.returncode, log_sha256=sha256(log)))
            save()
            return checked

        save()
        try:
            binary_dir = Path(runtime['runtime_directory'])
            fixture = OUT / 'hc-ordered-k-proof'
            command = ['/usr/bin/c++', '-O3', '-std=c++17', '-march=native', '-fopenmp', '-DGGML_USE_OPENMP',
                       '-I' + str(ENGINE / 'ggml/include'), '-I' + str(ENGINE / 'ggml/src'),
                       '-I' + str(ENGINE / 'ggml/src/ggml-cpu'), str(source), '-L' + str(binary_dir),
                       '-Wl,-rpath,' + str(binary_dir), '-lggml', '-lggml-cpu', '-lggml-base', '-ldl', '-o', str(fixture)]
            assert run(command, 'compile').returncode == 0
            result['idle_gate'] = guard.wait_idle(OUT / 'waiting-for-idle.json')
            result['background_gate'] = wait_background(guard, 1219506, 4, OUT / 'background-wait.json')
            env = {key: value for key, value in os.environ.items() if key not in runtime_environment(os.environ) and key != 'LD_PRELOAD'}
            env.update(runtime['runtime_env'])
            checked = run(['taskset', '-c', '0-14', str(fixture)], 'proof', env)
            text = checked.stdout + checked.stderr
            rows = [json.loads(line.removeprefix('HC_PROOF ')) for line in text.splitlines() if line.startswith('HC_PROOF ')]
            result['cases'] = rows
            libraries = [line.removeprefix('CPU_LIBRARY ') for line in text.splitlines() if line.startswith('CPU_LIBRARY ')]
            result['runtime_library_correct'] = len(libraries) == 1 and Path(libraries[0]).resolve() == Path(runtime['cpu']).resolve()
            result['summary'] = [json.loads(line.removeprefix('HC_SUMMARY ')) for line in text.splitlines() if line.startswith('HC_SUMMARY ')]
            save()
            assert result['runtime_library_correct']
            assert checked.returncode == 0 and len(rows) == 96
            assert result['summary'] == [dict(cases=96, failures=0)]
            assert all(all(row[key] for key in ('exact','quant_exact','guards','inputs')) for row in rows)
            assert all(sha256(path) == digest for path, digest in {**result['input_sha256'], **result['private_source_sha256']}.items())
            guard.assert_idle()
            result.update(passed=True, fixture_sha256=sha256(fixture), peer_preserved=process_info(1219506)['start'] == '103969952')
            print(json.dumps(dict(passed=True, cases=len(rows), bit_exact=True, quant_exact=True)), flush=True)
        except BaseException as error:
            result['error'] = repr(error)
            raise
        finally:
            result['finished'] = time.time()
            save()


if __name__ == '__main__':
    os.umask(0o077)
    main()
