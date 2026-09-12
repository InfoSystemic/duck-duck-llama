#!/usr/bin/env python3
"""Guard an isolated component benchmark of the already correctness-checked kernel.

The controller must be pinned to CPU127. It acquires the shared inference trial
controller lock and checks both servers remain idle throughout its child lifetime.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import inference_snapshot, sha256, process_info
from select_flash_q4_0910c import Manager
from model_measurement_guard import ModelMeasurementGuard

BASE = Path(__file__).resolve().parent
OUT = BASE / 'results/deepseek-v41-native-vnni-goal-0910'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpus', default='48-63')
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    assert os.sched_getaffinity(0) == {127}
    assert 1 <= args.workers <= 64
    proof = json.loads((OUT / 'kernel-check.json').read_text())
    assert proof['passed'] and proof['fp32_reduction_changed']
    assert all(sha256(p) == h for p, h in proof['source_sha256'].items())
    assert not (OUT / 'benchmark-check.json').exists()
    peer = Manager().validate_current()
    guard = ModelMeasurementGuard(peer['pid'], {peer['pid']: 18131}, inference_snapshot)
    selected = json.loads((BASE / 'deepseek-v41-selected.json').read_text())
    assert process_info(selected['pid'])['start'] == selected['start']
    assert all(sha256(p) == h for p, h in selected['source_sha256'].items())
    request_record = Path(selected['request_record'])
    request_count = json.loads(request_record.read_text())['completed']
    def check_idle():
        guard.assert_idle()
        assert process_info(selected['pid'])['start'] == selected['start']
        with urllib.request.urlopen('http://127.0.0.1:18170/health', timeout=2) as response:
            assert not json.load(response)['busy']
        assert json.loads(request_record.read_text())['completed'] == request_count
    inputs = [Path(__file__), BASE / 'benchmark_deepseek_v41_native_vnni_goal_0910.py',
        OUT / 'libdeepseek-v41-native-vnni.so', BASE / 'deepseek-v41-native-vnni-goal-0910.cpp']
    result = dict(passed=False, started=time.time(), selected_flash_pid=peer['pid'],
        input_sha256={str(p): sha256(p) for p in inputs}, component_only=True,
        full_checkpoint_loaded=False, model_tok_s_measured=False, deepseek_pid=selected['pid'], deepseek_start=selected['start'])
    owned = None
    def cancel(*_):
        raise InterruptedError('Cancel owned native GEMM component fixture')
    for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGHUP]:
        signal.signal(sig, cancel)
    with (BASE / 'results/deepseek-v41-controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_idle()
        try:
            cmd = ['taskset', '-c', args.cpus, str(BASE.parents[1] / 'tools/deepseek-v41-cpu-reference-0910/venv/bin/python'),
                str(inputs[1])]
            with (OUT / 'benchmark.log').open('w') as log:
                owned = subprocess.Popen(cmd, cwd=BASE, stdout=log, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, start_new_session=True,
                    env=dict(os.environ, OMP_NUM_THREADS=str(args.workers), MKL_NUM_THREADS='1',
                        OMP_WAIT_POLICY='PASSIVE', TOKENIZERS_PARALLELISM='false'))
                deadline, report = time.monotonic() + 120, 0
                while owned.poll() is None:
                    check_idle()
                    assert time.monotonic() < deadline
                    if time.monotonic() - report > 30:
                        print(json.dumps(dict(pid=owned.pid)), flush=True)
                        report = time.monotonic()
                    time.sleep(.25)
            assert owned.returncode == 0, owned.returncode
            benchmark = json.loads((OUT / 'benchmark-check.json').read_text())
            assert benchmark['passed']
            assert all(sha256(p) == h for p, h in result['input_sha256'].items())
            current = Manager().validate_current()
            assert current['pid'] == peer['pid'] and current['info']['start'] == peer['info']['start']
            check_idle()
            result.update(passed=True, peer_preserved=True, deepseek_preserved=True, exact_values=sum(c['n'] for c in benchmark['cases']))
        except BaseException as error:
            result['error'] = repr(error)
            raise
        finally:
            if owned and owned.poll() is None:
                os.killpg(owned.pid, signal.SIGTERM)
                try:
                    owned.wait(30)
                except subprocess.TimeoutExpired:
                    os.killpg(owned.pid, signal.SIGKILL)
                    owned.wait(30)
            result['finished'] = time.time()
            atomic_json(OUT / 'benchmark-result.json', result)
            print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
