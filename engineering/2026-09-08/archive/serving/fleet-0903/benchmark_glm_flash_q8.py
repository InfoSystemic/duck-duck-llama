#!/usr/bin/env python3
"""Measure isolated Flash Q8 decode with the shared guarded IMC harness."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from benchmark_qwen_q6 import wait_background
from glm_flash_q8_trial import Manager, OUT, PORT
from model_measurement_guard import ModelMeasurementGuard
from qwen_split_trial import inference_snapshot, sha256

BASE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('label')
    parser.add_argument('--tokens', type=int, default=512)
    parser.add_argument('--max-background-cores', type=float, default=5.0)
    parser.add_argument('--direct-answer', action='store_true',
                        help='Separately label a benchmark with the request reasoning budget set to zero')
    args = parser.parse_args()
    assert re.fullmatch(r'[A-Za-z0-9_-]+', args.label) and 128 <= args.tokens <= 2048
    os.umask(0o077)
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manager = Manager()
        current = manager.validate_current()
        assert not current.get('op_profile_count')
        assert not any('PROFILE' in key for key in current['runtime_env']), 'Disable operation profiling before measuring bandwidth'
        assert not manager.qwen.state.get('current') and manager.qwen.state['full_stopped']
        guard = ModelMeasurementGuard(current['pid'], {current['pid']: PORT}, inference_snapshot)
        guard.wait_idle(OUT / 'benchmark-waiting-for-idle.json', quiet_seconds=15)
        guard.assert_idle()
        background = wait_background(guard, current['pid'], args.max_background_cores, OUT / 'background-wait.json')
        path = BASE / 'results' / (args.label + '-controller.json')
        assert not path.exists()
        result = dict(started=time.time(), config=vars(args), current=current,
                      controller_pid=os.getpid(), background_gate=background, passed=False)
        def save():
            path.write_text(json.dumps(result, indent=2) + '\n')
        save()
        try:
            command = [sys.executable, '-u', str(BASE / 'measure-model-bandwidth.py'), args.label,
                '--port', str(PORT), '--pid', str(current['pid']), '--alias', 'glm-flash-q8-trial',
                '--drafts', str(current['drafts']), '--tokens', str(args.tokens),
                '--request-timeout-seconds', str(max(180, args.tokens / 3)),
                '--allowed-idle-pids', '', '--skip-idle-gate',
                '--bandwidth-target-gb-s', '285', '--bandwidth-capacity-gb-s', '380',
                '--chat-template-kwargs', json.dumps({'reasoning_effort': 'max'}),
                '--check-reasoning-budget-tokens', '0']
            if args.direct_answer:
                command += ['--reasoning-budget-tokens', '0']
            result['command'] = command
            save()
            completed = subprocess.run(command)
            assert completed.returncode == 0, ('Measurement failed', completed.returncode)
            measurement_path = BASE / 'results' / args.label / 'result.json'
            measurement = json.loads(measurement_path.read_text())
            assert not measurement.get('error') and measurement['input_integrity_verified']
            assert all(check['pass_check'] for check in measurement['checks'])
            assert measurement['server_command'] == current['command']
            manager.validate_current()
            result.update(passed=True, measurement=str(measurement_path), measurement_sha256=sha256(measurement_path),
                          rates=[x['timings']['predicted_per_second'] for x in measurement['measurements']],
                          adjusted_gb_s=[x['background_subtracted_gb_s'] for x in measurement['measurements']])
        except BaseException as error:
            result['error'] = repr(error)
            raise
        finally:
            result['finished'] = time.time()
            save()


if __name__ == '__main__':
    main()
