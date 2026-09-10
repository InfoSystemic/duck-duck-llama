#!/usr/bin/env python3
"""Guard the native Engram projection/gate build, bounded weights, and CPU oracle."""
import ast
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from model_measurement_guard import ModelMeasurementGuard
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import inference_snapshot, sha256
from select_flash_q4_0910c import Manager

BASE = Path(__file__).resolve().parent
OUT = BASE / 'results/deepseek-v41-engram-project-0910b'
PARENT = BASE / 'results/deepseek-v41-engram-project-0910'
SETUP = BASE / 'results/deepseek-v41-hash-reference-setup-0910/result.json'


def main():
    assert os.sched_getaffinity(0) == {127} and not OUT.exists()
    setup = json.loads(SETUP.read_text()); assert setup['passed']
    manager = Manager(); peer = manager.validate_current()
    guard = ModelMeasurementGuard(peer['pid'], {peer['pid']: 18131}, inference_snapshot)
    cpp = BASE / 'deepseek-v41-engram-project-0910.cpp'
    header = BASE / 'deepseek-v41-engram-project-0910.h'
    checker = BASE / 'check_deepseek_v41_engram_project_0910b.py'
    fetcher = BASE / 'fetch_deepseek_v41_engram_project_0910.py'
    for path in [checker, fetcher, Path(__file__)]: ast.parse(path.read_text())
    inputs = {str(p): sha256(p) for p in [Path(__file__), cpp, header, checker, fetcher, SETUP,
        BASE / 'model_measurement_guard.py', BASE / 'audit_deepseek_v41_tensors_0910.py',
        BASE / 'check_deepseek_v41_engram_lookup_0910.py', BASE / 'check_deepseek_v41_engram_hash_0910.py',
        BASE / 'results/deepseek-v41-engram-lookup-0910/result.json', BASE / 'select_flash_q4_0910c.py',
        PARENT / 'result.json', PARENT / 'weight-manifest.json', PARENT / 'libdeepseek-v41-engram-project.so']}
    result = dict(passed=False, started=time.time(), controller_pid=os.getpid(), input_sha256=inputs,
        selected_flash_pid=peer['pid'], selected_flash_start=peer['info']['start'],
        model_loaded=False, runtime_promoted=False, steps=[])
    owned = None
    def cancel(*_): raise InterruptedError('Stop only the owned DeepSeek projection component process')
    for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGHUP]: signal.signal(sig, cancel)
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guard.assert_idle(); OUT.mkdir()
        def save(): atomic_json(OUT / 'result.json', result)
        def run(command, label, timeout):
            nonlocal owned
            guard.assert_idle()
            log = OUT / (label + '.log')
            with log.open('w') as handle:
                owned = subprocess.Popen(command, cwd=BASE, stdout=handle, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, start_new_session=True,
                    env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false'))
                result['owned'] = dict(pid=owned.pid, label=label); save()
                deadline, report = time.monotonic() + timeout, 0
                while owned.poll() is None:
                    guard.assert_idle()
                    assert time.monotonic() < deadline, label
                    if time.monotonic() - report >= 30:
                        print(json.dumps(dict(running=label, pid=owned.pid, log=str(log))), flush=True)
                        report = time.monotonic()
                    time.sleep(.25)
            result['steps'].append(dict(label=label, command=command, exit_code=owned.returncode, log_sha256=sha256(log)))
            save()
            assert owned.returncode == 0, (label, owned.returncode)
            print(json.dumps(dict(completed=label)), flush=True)
        save()
        try:
            previous = json.loads((PARENT / 'result.json').read_text())
            assert not previous['passed'] and previous['steps'][0]['exit_code'] == 0
            library = PARENT / 'libdeepseek-v41-engram-project.so'
            assert sha256(library) == previous['library_sha256']
            assert sha256(cpp) == previous['input_sha256'][str(cpp)]
            assert sha256(header) == previous['input_sha256'][str(header)]
            weights = json.loads((PARENT / 'weight-manifest.json').read_text())
            assert weights['passed'] and all(sha256(r['file']) == r['sha256'] for r in weights['records'])
            (OUT / 'weight-manifest.json').write_bytes((PARENT / 'weight-manifest.json').read_bytes())
            result.update(library=str(library), library_sha256=sha256(library),
                parent_failed_run=str(PARENT / 'result.json'),
                harness_fix='Rename native wrapper parameters so invalid-pointer overrides reach the C API instead of colliding with Python arguments.',
                reused_native_library_and_downloads=True)
            save()
            run(['taskset', '-c', '48-51', str(Path(setup['venv']) / 'bin/python'), str(checker), str(OUT), str(library)],
                'reference-correctness', 1200)
            correctness = json.loads((OUT / 'correctness.json').read_text()); assert correctness['passed']
            assert all(sha256(path) == digest for path, digest in inputs.items())
            restored = manager.validate_current()
            assert restored['pid'] == peer['pid'] and restored['info']['start'] == peer['info']['start']
            guard.assert_idle()
            result.update(passed=True, selected_flash_preserved=True, correctness_sha256=sha256(OUT / 'correctness.json'),
                weight_manifest_sha256=sha256(OUT / 'weight-manifest.json'),
                limitations=correctness['limitations'])
        except BaseException as error:
            result['error'] = repr(error)
            raise
        finally:
            if owned is not None and owned.poll() is None:
                os.killpg(owned.pid, signal.SIGTERM)
                try: owned.wait(timeout=15)
                except subprocess.TimeoutExpired: os.killpg(owned.pid, signal.SIGKILL); owned.wait(timeout=10)
            result['finished'] = time.time(); save()


if __name__ == '__main__':
    os.umask(0o077)
    main()
