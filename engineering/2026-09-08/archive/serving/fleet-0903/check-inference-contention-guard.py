#!/usr/bin/env python3
"""Exercise benchmark cancellation with fake server objects; signal no processes."""
import json
from pathlib import Path
import tempfile
import time

from inference_contention_guard import InferenceContentionGuard, activity, wait_for_idle

passed = []
samples, churn = activity({'111': (0, 'a', '18135'), '222': (20, 'b', '18091')},
                          {'111': (10000, 'a', '18135'), '222': (20, 'b', '18091')}, 2, 111)
assert samples == [dict(pid=222, port='18091', cpu_percent=0.0)] and not churn
passed.append('own CPU excluded')
samples, churn = activity({'222': (200, 'old', '18091')}, {'222': (1, 'new', '18091')}, 2, 111)
assert not samples and churn == ['222']
passed.append('PID reuse detected')

class FakeServer:
    pid = 111
    def __init__(self, graceful=False):
        self.graceful = graceful
        self.returncode = None
        self.calls = []
    def poll(self):
        return self.returncode
    def terminate(self):
        self.calls.append('terminate')
        if self.graceful:
            self.returncode = 0
    def kill(self):
        self.calls.append('kill')
        self.returncode = -9

with tempfile.TemporaryDirectory(prefix='flash-contention-guard-check-') as directory:
    root = Path(directory)
    for graceful in [False, True]:
        server = FakeServer(graceful)
        count = [0]
        def busy():
            count[0] += 1
            return {'111': (count[0] * 1000, 'own', '18135'), '222': (count[0] * 100, 'prod', '18091')}
        guard = InferenceContentionGuard(server, busy, root / f'busy-{graceful}.json', interval=0.005, grace=0.005)
        guard.start()
        guard.thread.join(timeout=1)
        assert not guard.thread.is_alive()
        info = guard.stop()
        assert server.calls == (['terminate'] if graceful else ['terminate', 'kill'])
        assert info['test_pid'] == 111 and info['other_inference'][0]['pid'] == 222
        assert info['kill_sent'] is not graceful
        assert json.loads((root / f'busy-{graceful}.json').read_text()) == info
        passed.append('graceful stop' if graceful else 'forced stop after grace')

    server = FakeServer()
    guard = InferenceContentionGuard(server, lambda: {'222': (10, 'prod', '18091')}, root / 'idle.json', interval=0.002)
    guard.start()
    time.sleep(0.012)
    assert guard.stop() is None and server.calls == [] and not (root / 'idle.json').exists()
    passed.append('idle production does not stop test')

    server = FakeServer(True)
    count = [0]
    def appeared():
        count[0] += 1
        return {} if count[0] == 1 else {'333': (0, 'new', '18136')}
    guard = InferenceContentionGuard(server, appeared, root / 'churn.json', interval=0.002, grace=0.002)
    guard.start()
    guard.thread.join(timeout=1)
    assert not guard.thread.is_alive()
    assert guard.stop()['process_churn'] == ['333'] and server.calls == ['terminate']
    passed.append('new inference process stops only test')

    status = wait_for_idle(lambda: {'222': (0, 'prod', '18091')}, root / 'quiet.json',
                           allowed_idle_pids=('222',), quiet_seconds=0.01, interval=0.002)
    assert status['quiet_seconds'] >= 0.01 and not status['foreign_pids']
    passed.append('sustained idle gate')

    class StopFixture(Exception):
        pass
    count = [0]
    def foreign():
        count[0] += 1
        if count[0] > 3:
            raise StopFixture()
        return {'333': (0, 'foreign', '18136')}
    try:
        wait_for_idle(foreign, root / 'foreign.json', allowed_idle_pids=('222',), quiet_seconds=0, interval=0.001)
        raise AssertionError('A foreign idle model must prevent another Flash load')
    except StopFixture:
        status = json.loads((root / 'foreign.json').read_text())
        assert status['foreign_pids'] == ['333'] and status['quiet_seconds'] == 0
    passed.append('foreign loaded model prevents launch')

print(json.dumps(dict(passed=len(passed), checks=passed, actual_process_signals=0), indent=2))
