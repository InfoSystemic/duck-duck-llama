#!/usr/bin/env python3
"""Build, validate against the official CPU reference, then time native Engram hashing."""
import fcntl
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time

from benchmark_flash_q4_selected_0910 import background
from model_measurement_guard import ModelMeasurementGuard
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import inference_snapshot, sha256
from select_flash_q4_0910c import Manager

BASE = Path(__file__).resolve().parent
OUT = BASE / 'results/deepseek-v41-engram-hash-0910'
SETUP = BASE / 'results/deepseek-v41-hash-reference-setup-0910/result.json'


def main():
    assert os.sched_getaffinity(0) == {127} and not OUT.exists()
    setup = json.loads(SETUP.read_text())
    assert setup['passed']
    manager = Manager()
    peer = manager.validate_current()
    guard = ModelMeasurementGuard(peer['pid'], {peer['pid']: 18131}, inference_snapshot)
    cpp, checker = BASE / 'deepseek-v41-engram-hash-0910.cpp', BASE / 'check_deepseek_v41_engram_hash_0910.py'
    inputs = {str(p): sha256(p) for p in [Path(__file__), cpp, checker, SETUP, BASE / 'model_measurement_guard.py']}
    result = dict(passed=False, started=time.time(), input_sha256=inputs, selected_flash_pid=peer['pid'],
        model_loaded=False, runtime_promoted=False, steps=[], runs=[])
    owned = None
    def cancel(*_): raise InterruptedError('Stop only the owned Engram hash component process')
    for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGHUP]: signal.signal(sig, cancel)
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guard.assert_idle()
        OUT.mkdir()
        def save(): atomic_json(OUT / 'result.json', result)
        def run(command, label, timeout=300):
            nonlocal owned
            guard.assert_idle()
            log = OUT / (label + '.log')
            with log.open('w') as handle:
                owned = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
                    env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false'))
                deadline = time.monotonic() + timeout
                while owned.poll() is None:
                    guard.assert_idle()
                    assert time.monotonic() < deadline
                    time.sleep(.25)
            result['steps'].append(dict(label=label, command=command, exit_code=owned.returncode, log_sha256=sha256(log)))
            save()
            assert owned.returncode == 0, (label, owned.returncode)
            print(json.dumps(dict(completed=label)), flush=True)
            return log.read_text()
        save()
        try:
            library, benchmark = OUT / 'libdeepseek-v41-engram-hash.so', OUT / 'deepseek-v41-engram-hash-bench'
            flags = ['c++', '-O3', '-std=c++17', '-march=x86-64-v3', '-Wall', '-Wextra', '-Werror']
            run(flags + ['-shared', '-fPIC', str(cpp), '-o', str(library)], 'compile-library')
            run(flags + ['-DDEEPSEEK_V41_HASH_BENCH', str(cpp), '-o', str(benchmark)], 'compile-benchmark')
            result['library_sha256'], result['benchmark_sha256'] = sha256(library), sha256(benchmark)
            run(['taskset', '-c', '48', str(Path(setup['venv']) / 'bin/python'), str(checker), str(OUT), str(library)], 'official-reference-correctness')
            correctness = json.loads((OUT / 'correctness.json').read_text())
            assert correctness['passed']
            result['correctness_sha256'] = sha256(OUT / 'correctness.json')
            result['exact_hash_values_compared'] = correctness['exact_hash_values_compared']
            for tokens, repeats in [(1, 10000), (128, 1000), (4096, 30)]:
                expected = None
                for index, mode in enumerate([0, 1, 1, 0]):
                    before = background(peer['pid'])
                    timing = json.loads(run(['taskset', '-c', '48', str(benchmark), str(OUT / 'hash-config.bin'),
                        str(mode), str(tokens), str(repeats)], f'timing-{tokens}-{index}-{mode}'))
                    current = (timing['output_digest'], timing['guard'])
                    if expected is None: expected = current
                    assert current == expected
                    result['runs'].append(dict(index=index, mode=mode, tokens=tokens, timing=timing,
                        background_before=before, background_after=background(peer['pid'])))
                    save()
            summaries = []
            for tokens in [1, 128, 4096]:
                pairs = [[r['timing']['median_ns_per_call'] for r in result['runs'] if r['tokens'] == tokens and r['mode'] == mode] for mode in [0, 1]]
                baseline, candidate = [statistics.mean(values) for values in pairs]
                summaries.append(dict(tokens=tokens, baseline_ns_per_call=baseline, reciprocal_ns_per_call=candidate,
                    baseline_over_reciprocal=baseline / candidate, component_only=True))
            assert all(sha256(path) == digest for path, digest in inputs.items())
            manager.validate_current()
            guard.assert_idle()
            result.update(passed=True, summaries=summaries, selected_flash_preserved=True,
                limitations=['Standalone exact integer hash component; not yet integrated with the model graph or native gather.',
                    'CPU 48 only; timings include history writes and ID output, and record existing host load.',
                    'Hash time is a tiny component, not a model tok/s multiplier or an IMC bandwidth measurement.',
                    'Batch correctness uses one independent native state per sequence. Caller owns handle synchronization.'])
        except BaseException as error:
            result['error'] = repr(error)
            raise
        finally:
            if owned is not None and owned.poll() is None:
                os.killpg(owned.pid, signal.SIGTERM)
                try: owned.wait(timeout=15)
                except subprocess.TimeoutExpired: os.killpg(owned.pid, signal.SIGKILL); owned.wait(timeout=10)
            result['finished'] = time.time()
            save()


if __name__ == '__main__':
    os.umask(0o077)
    main()
