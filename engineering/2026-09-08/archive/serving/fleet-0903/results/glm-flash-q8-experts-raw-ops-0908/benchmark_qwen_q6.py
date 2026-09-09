#!/usr/bin/env python3
"""Run the existing IMC benchmark on an idle Q6 server at a chosen thread count."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from model_measurement_guard import ModelMeasurementGuard
from qwen_split_trial import inference_snapshot, process_info, sha256

BASE = Path(__file__).resolve().parent
TRIAL = BASE / 'results/qwen-q6-trial-0907'


def set_threads(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(str(value) + '\n')
    temporary.replace(path)


def host_cpu():
    result = {}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
            result[int(path.name)] = (int(fields[11]) + int(fields[12]), fields[19],
                                      (path / 'comm').read_text().strip())
        except (OSError, ValueError, IndexError):
            pass
    return result


def wait_background(guard, pid, max_cores, status_path=None):
    if max_cores <= 0:
        return {'disabled': True}
    quiet = 0.0
    history = []
    while quiet < 20:
        guard.assert_idle()
        before = host_cpu()
        started = time.monotonic()
        time.sleep(5)
        after = host_cpu()
        elapsed = time.monotonic() - started
        loads = []
        for other, (ticks, birth, name) in after.items():
            old = before.get(other)
            if other == pid or old is None or old[1] != birth:
                continue
            cores = (ticks - old[0]) / os.sysconf('SC_CLK_TCK') / elapsed
            if cores > 0:
                loads.append(dict(pid=other, name=name, cores=cores))
        total = sum(x['cores'] for x in loads)
        quiet = quiet + elapsed if total <= max_cores else 0
        row = dict(time=time.time(), background_cores=total, quiet_seconds=quiet,
                   largest=sorted(loads, key=lambda x: -x['cores'])[:5])
        history.append(row)
        Path(status_path or TRIAL / 'background-wait.json').write_text(json.dumps(row, indent=2) + '\n')
        print(json.dumps({'waiting_for_background': True, 'cores': round(total, 2),
                          'quiet_seconds': round(quiet, 1)}), flush=True)
    return dict(max_cores=max_cores, samples=history)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('label')
    parser.add_argument('--threads', type=int, choices=range(1, 17), default=15)
    parser.add_argument('--tokens', type=int, default=512)
    parser.add_argument('--max-background-cores', type=float, default=4.0)
    args = parser.parse_args()
    assert re.fullmatch(r'[A-Za-z0-9_-]+', args.label)
    with (TRIAL / 'lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads((TRIAL / 'state.json').read_text())
        current = state['current']
        assert state['full_stopped'] and not current['original']
        assert current['info']['start'] == process_info(current['pid'])['start']
        assert current['command'] == process_info(current['pid'])['command']
        assert args.threads <= int(current['runtime_env']['GGML_CPU_NUMA_THREADS'])
        guard = ModelMeasurementGuard(current['pid'], {current['pid']: 18095}, inference_snapshot)
        guard.wait_idle(TRIAL / 'benchmark-waiting-for-idle.json')
        guard.assert_idle()
        background = wait_background(guard, current['pid'], args.max_background_cores)
        threads_path = Path(current['runtime_env']['GGML_CPU_NUMA_THREADS_FILE'])
        previous = int(threads_path.read_text())
        out = BASE / 'results' / (args.label + '-controller.json')
        assert not out.exists()
        result = dict(started=time.time(), config=vars(args), current=current,
                      previous_threads=previous, controller_pid=os.getpid(), background_gate=background)
        def save():
            out.write_text(json.dumps(result, indent=2) + '\n')
        save()
        try:
            set_threads(threads_path, args.threads)
            command = [sys.executable, '-u', str(BASE / 'measure-model-bandwidth.py'), args.label,
                '--port', '18095', '--pid', str(current['pid']), '--alias', 'qwen-q6-trial',
                '--drafts', str(current['drafts']), '--tokens', str(args.tokens),
                '--request-timeout-seconds', str(max(90, args.tokens / 4)),
                '--allowed-idle-pids', '', '--skip-idle-gate']
            completed = subprocess.run(command)
            assert completed.returncode == 0, ('Measurement failed', completed.returncode)
            path = BASE / 'results' / args.label / 'result.json'
            measurement = json.loads(path.read_text())
            assert not measurement.get('error') and measurement['input_integrity_verified']
            assert all(check['pass_check'] for check in measurement['checks'])
            result.update(passed=True, measurement=str(path), measurement_sha256=sha256(path),
                          rates=[entry['timings']['predicted_per_second'] for entry in measurement['measurements']])
        except BaseException as error:
            result['error'] = repr(error)
            raise
        finally:
            # Restore only this trial's control file, after the client has ended.
            set_threads(threads_path, previous)
            result['finished'] = time.time()
            save()


if __name__ == '__main__':
    main()
