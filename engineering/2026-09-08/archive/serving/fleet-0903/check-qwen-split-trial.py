#!/usr/bin/env python3
"""Exercise lifecycle failures with disposable HTTP processes, never models."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

import qwen_split_trial as lifecycle
from qwen_split_trial import ExactProcess, Trial, process_info, transaction

FIXTURE = r'''
import http.server,json,os,sys
class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self,*args): pass
    def do_GET(self):
        body=json.dumps(dict(status='ok',pid=os.getpid(),affinity=sorted(os.sched_getaffinity(0)),
                             policy=os.environ['TRIAL_FIXTURE_POLICY'])).encode()
        self.send_response(200);self.send_header('Content-Length',str(len(body)));self.end_headers()
        self.wfile.write(body)
http.server.HTTPServer(('127.0.0.1',int(sys.argv[1])),Handler).serve_forever()
'''


def unused_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def healthy(port):
    with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=1) as response:
        return json.load(response)


class Scenario:
    # Reuse the actual child-launch affinity and ownership-aware cleanup code.
    spawn_server = Trial.spawn_server
    stop_owned = Trial.stop_owned

    def __init__(self, directory, failure):
        self.out = Path(directory)
        self.state = dict(original_context={'cwd': directory})
        self.failure = failure
        self.events = []
        self.children = []
        self.original = None
        self.candidate = None
        self.restored = None
        self.original_port = unused_port()
        self.candidate_port = unused_port()
        self.peer_port = unused_port()
        self.old = self.launch(self.original_port, 'original', 'original')
        self.peer = self.launch(self.peer_port, 'peer', 'peer')
        self.original_info = process_info(self.old.pid)

    def record(self, phase, **values):
        self.events.append(dict(phase=phase, **values))

    def launch(self, port, name, policy):
        env = dict(os.environ, TRIAL_FIXTURE_POLICY=policy)
        proc = self.spawn_server([sys.executable, '-u', '-c', FIXTURE, str(port)], env, name)
        self.children.append(proc)
        deadline = time.monotonic() + 5
        while True:
            assert proc.poll() is None, 'Fixture exited before becoming ready'
            try:
                state = healthy(port)
                assert state['pid'] == proc.pid and state['policy'] == policy
                assert state['affinity'] == list(range(128)), 'Child inherited controller-only affinity'
                return proc
            except OSError:
                if time.monotonic() > deadline:
                    raise TimeoutError('Disposable HTTP process did not become ready')
                time.sleep(0.02)

    def fail(self, where):
        if self.failure == where:
            if where == 'cancel':
                raise KeyboardInterrupt('Simulated cancellation')
            raise RuntimeError('Simulated failure: ' + where)

    def measure_before(self):
        assert healthy(self.original_port)['policy'] == 'original'
        self.fail('before')

    def gate_before_stop(self):
        assert healthy(self.peer_port)['pid'] == self.peer.pid
        self.fail('gate')

    def stop_original(self):
        self.fail('before_termination')
        expected = dict(start_ticks=self.original_info['start'], exe=self.original_info['exe'])
        self.original = ExactProcess(self.old.pid, expected, {self.peer.pid})
        self.original.terminate()
        self.old.wait(timeout=5)
        self.fail('after_termination')

    def run_candidate(self):
        self.candidate = self.launch(self.candidate_port, 'candidate', 'candidate')
        self.fail('candidate_load')
        assert healthy(self.candidate_port)['policy'] == 'candidate'
        self.fail('candidate_measurement')
        self.fail('cancel')

    def stop_candidate(self):
        self.stop_owned(self.candidate)
        self.fail('cleanup')

    def restore_original(self):
        if self.original is None or not self.original.termination_sent:
            return
        assert self.original.exited()
        assert self.candidate is None or self.candidate.poll() is not None
        self.restored = self.launch(self.original_port, 'restored', 'original')
        assert healthy(self.original_port)['policy'] == 'original'

    def measure_after(self):
        assert self.restored is not None and healthy(self.original_port)['pid'] == self.restored.pid
        self.fail('after')

    def finish(self):
        if self.original is not None:
            self.original.close()
        # Test teardown owns every disposable process, including the fake peer.
        for proc in self.children:
            if proc.poll() is None:
                proc.terminate()
            proc.wait(timeout=5)
        assert all(proc.poll() is not None for proc in self.children)


def transaction_check(case):
    with tempfile.TemporaryDirectory(prefix='qwen-trial-check-') as directory:
        scenario = Scenario(directory, case)
        caught = None
        try:
            try:
                transaction(scenario)
            except (RuntimeError, KeyboardInterrupt) as error:
                caught = repr(error)
            assert bool(caught) == (case != 'normal'), (case, caught)
            peer = healthy(scenario.peer_port)
            assert peer['pid'] == scenario.peer.pid and scenario.peer.poll() is None
            restored_state = healthy(scenario.original_port)
            interrupted = case not in ('before', 'gate', 'before_termination')
            if interrupted:
                assert scenario.old.poll() is not None and scenario.restored.poll() is None
                assert restored_state['pid'] != scenario.old.pid
            else:
                assert scenario.old.poll() is None and scenario.restored is None
                assert restored_state['pid'] == scenario.old.pid
            assert restored_state['policy'] == 'original'
            assert scenario.candidate is None or scenario.candidate.poll() is not None
            return dict(case=case, passed=True, original_service_responsive=True,
                        original_restored=interrupted, peer_untouched=True)
        finally:
            scenario.finish()


def identity_checks():
    results = []
    with tempfile.TemporaryDirectory(prefix='qwen-trial-identity-') as directory:
        scenario = Scenario(directory, 'normal')
        try:
            info = process_info(scenario.old.pid)
            expected = dict(start_ticks=info['start'], exe=info['exe'])
            variants = (
                ('protected_pid', expected, {scenario.old.pid}),
                ('reused_pid', dict(expected, start_ticks='wrong-start'), set()),
                ('different_executable', dict(expected, exe='/wrong/executable'), set()))
            for name, identity, forbidden in variants:
                try:
                    handle = ExactProcess(scenario.old.pid, identity, forbidden)
                except RuntimeError:
                    pass
                else:
                    handle.close()
                    raise AssertionError('Unsafe process identity accepted: ' + name)
                assert healthy(scenario.original_port)['pid'] == scenario.old.pid
                results.append(dict(case=name, passed=True, process_survived=True))
            scenario.restored = scenario.old
            try:
                scenario.stop_owned(scenario.old)
            except AssertionError:
                pass
            else:
                raise AssertionError('Handed-off original service was stopped')
            assert healthy(scenario.original_port)['pid'] == scenario.old.pid
            results.append(dict(case='restoration_handoff', passed=True, process_survived=True))
        finally:
            scenario.finish()
    return results


def recovery_check(case):
    """Use the real restoration loop with disposable processes and injected monitoring faults."""
    with tempfile.TemporaryDirectory(prefix='qwen-trial-recovery-') as directory:
        scenario = Scenario(directory, 'normal')
        old_port_available = lifecycle.port_available
        old_full_pid = lifecycle.FULL_PID
        try:
            scenario.stop_original()
            scenario.plan = {'original_command': [sys.executable, '-u', '-c', FIXTURE, str(scenario.original_port)]}
            scenario.environment = dict(os.environ, TRIAL_FIXTURE_POLICY='original')
            scenario.wait_original_exit = lambda: scenario.old.wait(timeout=5)
            scenario.check_inputs = lambda: None
            calls = dict(quiet=0, ready=0)
            real_spawn = scenario.spawn_server
            def tracked_spawn(command, environment, name):
                proc = real_spawn(command, environment, name)
                scenario.children.append(proc)
                return proc
            scenario.spawn_server = tracked_spawn
            def quiet(ports, recovery=False):
                assert recovery and ports == {scenario.peer.pid: 18091}
                assert healthy(scenario.peer_port)['pid'] == scenario.peer.pid
                calls['quiet'] += 1
                if case == 'restore_idle_monitor_failure' and calls['quiet'] == 1:
                    raise OSError('Simulated idle monitor failure')
            scenario.quiet = quiet
            def ready(proc, port, recovery=False):
                assert recovery and port == 18095
                calls['ready'] += 1
                if calls['ready'] == 1:
                    if case == 'restore_peer_work':
                        raise lifecycle.PeerBusy('Simulated incoming peer work')
                    if case == 'restore_load_monitor_failure':
                        raise OSError('Simulated load monitor failure')
                deadline = time.monotonic() + 5
                while True:
                    assert proc.poll() is None
                    try:
                        assert healthy(scenario.original_port)['pid'] == proc.pid
                        return
                    except OSError:
                        if time.monotonic() > deadline:
                            raise TimeoutError('Restored fixture did not become healthy')
                        time.sleep(0.02)
            scenario.wait_ready = ready
            def verify(proc, candidate):
                assert not candidate
                assert healthy(scenario.original_port)['policy'] == 'original'
                assert proc.poll() is None
            scenario.verify_server = verify
            lifecycle.FULL_PID = scenario.peer.pid
            lifecycle.port_available = lambda port: port == 18095
            Trial.restore_original(scenario)
            assert healthy(scenario.original_port)['pid'] == scenario.restored.pid
            assert healthy(scenario.peer_port)['pid'] == scenario.peer.pid
            assert scenario.restored.poll() is None
            if case != 'restore_idle_monitor_failure':
                assert calls['ready'] == 2
                assert sum(x['phase'] == 'restore_load_yielded' for x in scenario.events) == 1
            else:
                assert calls['quiet'] == 2 and calls['ready'] == 1
            return dict(case=case, passed=True, original_service_responsive=True,
                        peer_untouched=True, production_restore_loop_exercised=True)
        finally:
            lifecycle.port_available = old_port_available
            lifecycle.FULL_PID = old_full_pid
            scenario.finish()


if __name__ == '__main__':
    assert os.sched_getaffinity(0) == {63}, 'Run this fixture controller on CPU63'
    cases = ('normal', 'before', 'gate', 'before_termination', 'after_termination',
             'candidate_load', 'candidate_measurement', 'cancel', 'cleanup', 'after')
    recovery_cases = ('restore_idle_monitor_failure', 'restore_peer_work', 'restore_load_monitor_failure')
    report = dict(cases=[transaction_check(case) for case in cases] + identity_checks()
                        + [recovery_check(case) for case in recovery_cases],
                  real_model_requests=0, real_model_signals=0,
                  controller_affinity=sorted(os.sched_getaffinity(0)))
    base = Path(__file__).resolve().parent
    report['source_sha256'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in [Path(__file__).resolve(), base / 'qwen_split_trial.py']}
    print(json.dumps(report, indent=2))
