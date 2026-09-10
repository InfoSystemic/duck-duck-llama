#!/usr/bin/env python3
"""Build and validate integrated native hashing and sharded FP8 Engram lookup."""
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
OUT = BASE / 'results/deepseek-v41-engram-lookup-0910'
SETUP = BASE / 'results/deepseek-v41-hash-reference-setup-0910/result.json'


def main():
    assert os.sched_getaffinity(0) == {127} and not OUT.exists()
    setup = json.loads(SETUP.read_text())
    assert setup['passed']
    manager = Manager()
    peer = manager.validate_current()
    guard = ModelMeasurementGuard(peer['pid'], {peer['pid']: 18131}, inference_snapshot)
    cpp = BASE / 'deepseek-v41-engram-lookup-0910.cpp'
    header = BASE / 'deepseek-v41-engram-lookup-0910.h'
    checker = BASE / 'check_deepseek_v41_engram_lookup_0910.py'
    dependencies = [BASE / 'deepseek-v41-engram-hash-0910.cpp', BASE / 'deepseek-v41-engram-cpu-0910.cpp']
    inputs = {str(p): sha256(p) for p in [Path(__file__), cpp, header, checker, SETUP,
        BASE / 'model_measurement_guard.py', BASE / 'audit_deepseek_v41_tensors_0910.py',
        BASE / 'check_deepseek_v41_engram_hash_0910.py', *dependencies]}
    result = dict(passed=False, started=time.time(), input_sha256=inputs, selected_flash_pid=peer['pid'],
        model_loaded=False, runtime_promoted=False, steps=[], runs=[])
    owned = None
    def cancel(*_): raise InterruptedError('Stop only the owned Engram lookup component process')
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
            library = OUT / 'libdeepseek-v41-engram-lookup.so'
            flags = ['c++', '-O3', '-std=c++17', '-march=x86-64-v3', '-Wall', '-Wextra', '-Werror']
            run(flags + ['-shared', '-fPIC', str(cpp), *map(str, dependencies), '-o', str(library)], 'compile-library')
            result['library_sha256'] = sha256(library)
            run(['taskset', '-c', '48', str(Path(setup['venv']) / 'bin/python'), str(checker), str(OUT), str(library)],
                'official-reference-correctness', timeout=900)
            correctness = json.loads((OUT / 'correctness.json').read_text())
            assert correctness['passed']
            result['correctness_sha256'] = sha256(OUT / 'correctness.json')
            for key in ['exact_bf16_values_compared', 'exact_row_ids_compared', 'actual_pinned_rows', 'actual_retrieved_bytes']:
                result[key] = correctness[key]
            assert all(sha256(path) == digest for path, digest in inputs.items())
            manager.validate_current()
            guard.assert_idle()
            result.update(passed=True, selected_flash_preserved=True,
                limitations=['Integrated hash and native FP8 lookup component; projection/gate and full model graph remain pending.',
                    'Single CPU correctness with one or four logical row shards per layer; no NUMA scaling claim.',
                    'The actual 202.8 GB address spans are reserved without populating full tables; only bounded fixture rows are touched.',
                    'No model tok/s or IMC bandwidth measured. Native table precision is preserved.'])
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
