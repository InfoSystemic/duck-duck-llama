#!/usr/bin/env python3
"""Prepare isolated CPU-only reference dependencies and pinned tokenizer data."""
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import urllib.request

from model_measurement_guard import ModelMeasurementGuard
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import inference_snapshot, sha256
from select_flash_q4_0910c import Manager

BASE = Path(__file__).resolve().parent
OUT = BASE / 'results/deepseek-v41-hash-reference-setup-0910'
VENV = BASE.parents[1] / 'tools/deepseek-v41-cpu-reference-0910/venv'
REVISION = 'fb2764a5cf321eaa5070ca8f9e892818f477c16d'


def main():
    assert os.sched_getaffinity(0) == {127} and not OUT.exists() and not VENV.exists()
    assert shutil.disk_usage(VENV.parent.parent.parent).free > 8_000_000_000
    manager = Manager()
    peer = manager.validate_current()
    guard = ModelMeasurementGuard(peer['pid'], {peer['pid']: 18131}, inference_snapshot)
    result = dict(passed=False, started=time.time(), source_sha256=sha256(__file__),
        peer_pid=peer['pid'], venv=str(VENV), revision=REVISION, steps=[], retrieved=[])
    owned = None
    def cancel(*_): raise InterruptedError('Cancel only the owned dependency preparation')
    for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGHUP]: signal.signal(sig, cancel)
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guard.assert_idle()
        OUT.mkdir()
        def save(): atomic_json(OUT / 'result.json', result)
        def run(command, label, timeout=600):
            nonlocal owned
            guard.assert_idle()
            log = OUT / (label + '.log')
            with log.open('w') as handle:
                owned = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
                    env=dict(os.environ, PIP_CONFIG_FILE=os.devnull, PIP_NO_CACHE_DIR='1', OMP_NUM_THREADS='1',
                        MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false'))
                deadline = time.monotonic() + timeout
                while owned.poll() is None:
                    guard.assert_idle()
                    assert time.monotonic() < deadline
                    time.sleep(.5)
            result['steps'].append(dict(label=label, command=command, exit_code=owned.returncode, log_sha256=sha256(log)))
            save()
            assert owned.returncode == 0, (label, owned.returncode)
            print(json.dumps(dict(completed=label)), flush=True)
        save()
        try:
            run(['python3', '-m', 'venv', str(VENV)], 'create-venv')
            python = str(VENV / 'bin/python')
            run([python, '-m', 'pip', 'install', '--only-binary=:all:', '--index-url', 'https://pypi.org/simple',
                '--report', str(OUT / 'python-install-report.json'),
                'numpy==2.3.3', 'sympy==1.14.0', 'tokenizers==0.22.2'], 'install-small-dependencies')
            run([python, '-m', 'pip', 'install', '--only-binary=:all:', '--index-url', 'https://download.pytorch.org/whl/cpu',
                '--report', str(OUT / 'torch-install-report.json'), 'torch==2.10.0+cpu'], 'install-cpu-torch')
            for name, expected_bytes in [('tokenizer.json', 6367257), ('tokenizer_config.json', 801)]:
                guard.assert_idle()
                url = f'https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/{REVISION}/{name}'
                with urllib.request.urlopen(url, timeout=60) as response:
                    data = response.read(expected_bytes + 1)
                assert len(data) == expected_bytes
                json.loads(data)
                path = OUT / name
                path.write_bytes(data)
                result['retrieved'].append(dict(url=url, file=str(path), bytes=len(data), sha256=sha256(path)))
                save()
            code = "import importlib.metadata,json,torch;torch.set_num_threads(1);print(json.dumps({'versions':{p:importlib.metadata.version(p) for p in ['numpy','sympy','tokenizers','torch']},'cuda_version':torch.version.cuda,'cuda_available':torch.cuda.is_available(),'threads':torch.get_num_threads()}))"
            run([python, '-c', code], 'verify-cpu-reference')
            result['reference'] = json.loads((OUT / 'verify-cpu-reference.log').read_text())
            assert result['reference']['cuda_version'] is None and not result['reference']['cuda_available']
            manager.validate_current()
            guard.assert_idle()
            result.update(passed=True, selected_flash_preserved=True, no_full_model_downloaded=True)
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
