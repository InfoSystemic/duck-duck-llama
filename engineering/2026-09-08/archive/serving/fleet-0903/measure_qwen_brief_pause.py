#!/usr/bin/env python3
"""Prepared benchmark requiring approval to pause the separate Chrome process."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

from model_measurement_guard import ModelMeasurementGuard
from qwen_split_trial import inference_snapshot, process_info

BASE = Path(__file__).resolve().parent
TRIAL = BASE / 'results/qwen-q6-trial-0907'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('label')
    parser.add_argument('--chrome-pid', type=int, required=True)
    parser.add_argument('--approved', action='store_true')
    args = parser.parse_args()
    assert args.approved, 'This separate application pause has not been authorized'
    assert args.label.replace('-', '').replace('_', '').isalnum()
    record_path = BASE / 'results' / (args.label + '-pause.json')
    assert not record_path.exists()
    chrome = Path('/proc') / str(args.chrome_pid)
    assert chrome.stat().st_uid == os.getuid()
    assert os.readlink(chrome / 'exe') == '/opt/google/chrome/chrome'
    info = process_info(args.chrome_pid)
    assert info['state'] not in ('T', 't', 'Z'), 'Do not resume an already stopped process'
    chrome_fd = os.pidfd_open(args.chrome_pid)
    assert process_info(args.chrome_pid)['start'] == info['start']
    record = dict(started=time.time(), chrome_pid=args.chrome_pid, chrome_start=info['start'],
                  maximum_pause_seconds=90, completed=False)
    def save():
        temporary = record_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, indent=2) + '\n')
        temporary.replace(record_path)
    def interrupted(signum, _):
        raise InterruptedError('End the measurement and resume Chrome: ' + str(signum))
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    child, watchdog = None, None
    read_fd, write_fd = os.pipe()
    paused = False
    with (TRIAL / 'lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads((TRIAL / 'state.json').read_text())
        current = state['current']
        pid = current['pid']
        assert state['full_stopped'] and current['q8_batch'] and current['drafts'] == 3
        assert process_info(pid)['start'] == current['info']['start']
        assert int(Path(current['runtime_env']['GGML_CPU_NUMA_THREADS_FILE']).read_text()) == 15
        ModelMeasurementGuard(pid, {pid: 18095}, inference_snapshot).assert_idle()
        record['model_pid'] = pid
        save()
        try:
            # An independent watchdog resumes this exact process after 90 seconds
            # or as soon as the parent closes its pipe, including parent failure.
            watchdog = os.fork()
            if watchdog == 0:
                os.close(write_fd)
                for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                    signal.signal(sig, signal.SIG_IGN)
                select.select([read_fd], [], [], 90)
                try:
                    signal.pidfd_send_signal(chrome_fd, signal.SIGCONT)
                except ProcessLookupError:
                    pass
                os._exit(0)
            os.close(read_fd)
            signal.pidfd_send_signal(chrome_fd, signal.SIGSTOP)
            paused = True
            record['paused_at'] = time.time()
            save()
            print(json.dumps({'approved_pause_started': args.chrome_pid, 'maximum_seconds': 90}), flush=True)
            command = [sys.executable, '-u', str(BASE / 'measure-model-bandwidth.py'), args.label,
                '--port', '18095', '--pid', str(pid), '--alias', 'qwen-q6-trial', '--drafts', '3',
                '--tokens', '512', '--request-timeout-seconds', '70', '--allowed-idle-pids', '', '--skip-idle-gate']
            child = subprocess.Popen(command, close_fds=True)
            record['measurement_exit'] = child.wait(timeout=85)
            assert record['measurement_exit'] == 0
            record['completed'] = True
        finally:
            if paused:
                try:
                    signal.pidfd_send_signal(chrome_fd, signal.SIGCONT)
                    record['resume_sent_at'] = time.time()
                except ProcessLookupError:
                    record['chrome_exited'] = True
            os.close(write_fd)
            if child is not None and child.poll() is None:
                child.send_signal(signal.SIGINT)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    child.wait(timeout=10)
            if watchdog is not None:
                os.waitpid(watchdog, 0)
            os.close(chrome_fd)
            record['finished'] = time.time()
            save()
            print(json.dumps({'chrome_resume_sent': bool(record.get('resume_sent_at')),
                              'record': str(record_path)}), flush=True)


if __name__ == '__main__':
    main()
