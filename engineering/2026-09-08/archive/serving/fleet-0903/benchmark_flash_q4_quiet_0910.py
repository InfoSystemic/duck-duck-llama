#!/usr/bin/env python3
"""Repeat selected Flash Q4/MTP2 measurements with a bounded background-CPU gate."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from benchmark_flash_q4_selected_0910 import background, read_measurement
from benchmark_qwen_q6 import wait_background
from model_measurement_guard import ModelMeasurementGuard
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import inference_snapshot, sha256
from select_flash_q4_0910c import BASE, PORT, SELECTED, Manager, mapped_libraries

OUT = BASE / 'results/flash-q4-quiet-mtp2-0910'


def main(lock_fd):
    assert not OUT.exists()
    manager = Manager()
    current = manager.validate_current()
    assert current['drafts'] == 2 and current['quant'] == 'UD-Q4_K_XL' and current['workers'] == 15
    assert set(inference_snapshot()) == {str(current['pid'])}
    sources = {str(p): sha256(p) for p in [Path(__file__), SELECTED, BASE / 'measure-model-bandwidth.py',
        BASE / 'benchmark_flash_q4_selected_0910.py', BASE / 'benchmark_qwen_q6.py',
        BASE / 'model_measurement_guard.py', BASE / 'guarded_inference_request.py',
        BASE / 'dram_bandwidth.py', BASE / 'inference_contention_guard.py', BASE / 'select_flash_q4_0910c.py']}
    libraries = {p: sha256(p) for p in mapped_libraries(current['pid'])}
    OUT.mkdir()
    result = dict(started=time.time(), passed=False, controller_pid=os.getpid(), current=current,
        source_sha256=sources, mapped_libraries=libraries, runs=[], model_started=False, runtime_promoted=False,
        policy='Two repeated prose/code pairs on the same loaded selected Q4/Q8-MTP2 process. '
            'Require 20 seconds at <=6 background CPU cores before each pair; record later load and qualify IMC independently. '
            'The server ignores request draft overrides; effective MTP2 is checked in the launch and returned token counts.')
    child = None
    def save(): atomic_json(OUT / 'result.json', result)
    def cancel(*_): raise InterruptedError('Release only the owned measurement request; preserve selected Flash')
    for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGHUP]: signal.signal(sig, cancel)
    def verify():
        observed = manager.validate_current()
        assert observed['pid'] == current['pid'] and observed['info']['start'] == current['info']['start']
        assert all(sha256(path) == digest for path, digest in sources.items())
        assert mapped_libraries(current['pid']) == set(libraries)
    save()
    try:
        for index in range(2):
            verify()
            guard = ModelMeasurementGuard(current['pid'], {current['pid']: PORT}, inference_snapshot)
            run = dict(index=index, started=time.time())
            result['runs'].append(run); save()
            run['idle_gate'] = guard.wait_idle(OUT / f'idle-{index}.json', quiet_seconds=15)
            run['background_gate'] = wait_background(guard, current['pid'], 6, OUT / f'background-{index}.json')
            run['background_before'] = background(current['pid'])
            guard.assert_idle(); verify()
            label = f'flash-q4-quiet-mtp2-measure-0910-{index}'
            command = [sys.executable, '-u', str(BASE / 'measure-model-bandwidth.py'), label,
                '--port', str(PORT), '--pid', str(current['pid']), '--alias', 'glm-flash-q4', '--drafts', '2',
                '--tokens', '512', '--request-timeout-seconds', '300', '--allowed-idle-pids', '', '--skip-idle-gate',
                '--bandwidth-target-gb-s', '250', '--bandwidth-capacity-gb-s', '380',
                '--chat-template-kwargs', '{"reasoning_effort":"max"}', '--check-reasoning-budget-tokens', '0']
            run['command'] = command; save()
            print(json.dumps(dict(measuring=label, background_cores=run['background_before']['background_cores'])), flush=True)
            child = subprocess.Popen(command, pass_fds=(lock_fd,))
            while child.poll() is None: time.sleep(.5)
            assert child.returncode == 0
            child = None
            path = BASE / 'results' / label / 'result.json'
            rows = read_measurement(path, current)
            run.update(rows=rows, measurement=str(path), measurement_sha256=sha256(path),
                background_after=background(current['pid']), finished=time.time())
            reference = result['runs'][0]['rows']
            run['outputs_and_counts_match'] = all(rows[k][field] == reference[k][field]
                for k in rows for field in ['output_sha256', 'generated_tokens', 'draft_tokens', 'accepted_draft_tokens'])
            save()
            assert run['outputs_and_counts_match'], 'Repeated Flash output or speculative counts changed'
            print(json.dumps(dict(completed_pair=index, rows=rows)), flush=True)
        verify()
        assert all(sha256(path) == digest for path, digest in libraries.items())
        result.update(passed=True, all_outputs_and_counts_match=True,
            all_adjacent_idle_qualifies=all(r['counters']['adjacent_idle_qualifies'] for run in result['runs'] for r in run['rows'].values()),
            target_240_reached=all(r['counters']['adjacent_idle_qualifies'] and r['adjusted_gb_s'] >= 240 for run in result['runs'] for r in run['rows'].values()),
            target_250_reached=all(r['qualified_over_250'] for run in result['runs'] for r in run['rows'].values()))
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try: child.wait(timeout=15)
            except subprocess.TimeoutExpired: child.kill(); child.wait(timeout=15)
        try:
            verify()
            result['selected_flash_preserved'] = True
        except BaseException as error:
            result.update(passed=False, selected_flash_preserved=False, preservation_error=repr(error))
        result['finished'] = time.time(); save()


if __name__ == '__main__':
    os.umask(0o077)
    assert os.sched_getaffinity(0) == {127}
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main(lock.fileno())
