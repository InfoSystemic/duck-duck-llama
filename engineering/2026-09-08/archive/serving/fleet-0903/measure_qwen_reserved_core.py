#!/usr/bin/env python3
"""Run one bounded Q6 measurement with a temporary GPU-process CPU affinity."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from model_measurement_guard import ModelMeasurementGuard
from qwen_split_trial import inference_snapshot, process_info

BASE = Path(__file__).resolve().parent
TRIAL = BASE / 'results/qwen-q6-trial-0907'


def birth(path):
    return path.joinpath('stat').read_text().rsplit(')', 1)[1].split()[19]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('label')
    parser.add_argument('--chrome-pid', type=int, required=True)
    args = parser.parse_args()
    assert args.label.replace('-', '').replace('_', '').isalnum()
    record_path = BASE / 'results' / (args.label + '-affinity.json')
    assert not record_path.exists()
    proc = Path('/proc') / str(args.chrome_pid)
    assert proc.stat().st_uid == os.getuid() and os.readlink(proc / 'exe') == '/opt/google/chrome/chrome'
    assert b'--type=gpu-process' in proc.joinpath('cmdline').read_bytes().split(b'\0')
    original = list(range(8, 16))
    restricted = [15]
    process_birth = birth(proc)
    initial = {}
    for thread in proc.joinpath('task').iterdir():
        mask = sorted(os.sched_getaffinity(int(thread.name)))
        assert mask == original, ('Unexpected thread affinity', thread.name, mask)
        initial[thread.name] = dict(birth=birth(thread), affinity=mask)
    record = dict(started=time.time(), pid=args.chrome_pid, process_birth=process_birth,
                  original=original, temporary=restricted, initial_threads=initial,
                  applied=[], restored=[], skipped=[], completed=False)
    def save():
        temporary = record_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, indent=2) + '\n')
        temporary.replace(record_path)
    save()
    def interrupted(signum, _):
        raise InterruptedError('End the owned measurement and restore affinity: signal ' + str(signum))
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    child = None
    with (TRIAL / 'lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = json.loads((TRIAL / 'state.json').read_text())['current']
        pid = current['pid']
        assert current['q8_batch'] and current['drafts'] == 3 and not current['original']
        assert process_info(pid)['start'] == current['info']['start']
        assert int(Path(current['runtime_env']['GGML_CPU_NUMA_THREADS_FILE']).read_text()) == 15
        guard = ModelMeasurementGuard(pid, {pid: 18095}, inference_snapshot)
        guard.assert_idle()
        try:
            assert birth(proc) == process_birth
            # All existing threads have the same mask. Newly created GPU threads
            # therefore inherit the same original mask before this operation.
            for _ in range(3):
                for thread in proc.joinpath('task').iterdir():
                    tid = int(thread.name)
                    try:
                        mask = sorted(os.sched_getaffinity(tid))
                        if mask == restricted:
                            continue
                        assert mask == original
                        thread_birth = birth(thread)
                        os.sched_setaffinity(tid, restricted)
                        record['applied'].append(dict(tid=tid, birth=thread_birth))
                    except ProcessLookupError:
                        pass
                save()
            print(json.dumps({'temporarily_isolated_gpu_pid': args.chrome_pid,
                              'cpu': 15, 'model_workers_unchanged': pid}), flush=True)
            command = [sys.executable, '-u', str(BASE / 'measure-model-bandwidth.py'), args.label,
                '--port', '18095', '--pid', str(pid), '--alias', 'qwen-q6-trial', '--drafts', '3',
                '--tokens', '512', '--request-timeout-seconds', '90', '--allowed-idle-pids', '', '--skip-idle-gate']
            child = subprocess.Popen(command)
            code = child.wait(timeout=180)
            record['measurement_exit'] = code
            assert code == 0, ('Measurement failed', code)
            record['completed'] = True
        finally:
            # Stop only our HTTP client; its finally block releases the request.
            if child is not None and child.poll() is None:
                child.send_signal(signal.SIGINT)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    child.wait(timeout=10)
            if proc.exists() and birth(proc) == process_birth:
                for thread in proc.joinpath('task').iterdir():
                    try:
                        tid = int(thread.name)
                        mask = sorted(os.sched_getaffinity(tid))
                        if mask == restricted:
                            os.sched_setaffinity(tid, original)
                            assert sorted(os.sched_getaffinity(tid)) == original
                            record['restored'].append(tid)
                        else:
                            record['skipped'].append(dict(tid=tid, affinity=mask,
                                reason='Thread affinity was changed externally'))
                    except ProcessLookupError:
                        pass
            else:
                record['process_exited'] = True
            record['finished'] = time.time()
            save()
            print(json.dumps({'gpu_affinity_restored': len(record['restored']),
                              'process_exited': record.get('process_exited', False),
                              'record': str(record_path)}), flush=True)


if __name__ == '__main__':
    main()
