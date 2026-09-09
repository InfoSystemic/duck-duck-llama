#!/usr/bin/env python3
"""Prepare, or explicitly execute, a temporary Qwen trial and restore the original.

Execution requires separate user authorization to interrupt the existing Qwen
service. The default invocation only checks the prepared configuration.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import select
import signal
import socket
import subprocess
import sys
import time
import urllib.request

from inference_contention_guard import activity
from model_measurement_guard import ModelMeasurementGuard, read_service

BASE = Path(__file__).resolve().parent
STAGING = BASE / 'results/qwen-even-split-model-trial-0906-staging'
FULL_PID = 4005448
ORIGINAL_PID = 2308651
TRIAL_PORT = 18155


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def process_info(pid):
    proc = Path('/proc') / str(pid)
    fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(start=fields[19], state=fields[0], exe=os.readlink(proc / 'exe'),
                command=[x.decode() for x in (proc / 'cmdline').read_bytes().split(b'\0') if x],
                cwd=os.readlink(proc / 'cwd'), affinity=sorted(os.sched_getaffinity(pid)))


def process_environment(pid):
    # Keep the full original environment only in memory, for exact restoration.
    return dict(x.decode().split('=', 1) for x in
                Path(f'/proc/{pid}/environ').read_bytes().split(b'\0') if b'=' in x)


def runtime_environment(environment):
    return {k: v for k, v in environment.items()
            if k.startswith(('GGML_', 'LLAMA_GRAPH_PHASE', 'LLAMA_MTP_', 'OMP_', 'GOMP_'))
            or k == 'LD_LIBRARY_PATH'}


def inference_snapshot(exclude=None):
    result = {}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or proc.name == str(exclude):
            continue
        try:
            stat = (proc / 'stat').read_text()
            name = stat[stat.find('(') + 1:stat.rfind(')')]
            if name != 'llama-server' and not name.startswith(('glm-mtp-head', 'qwen-mtp-head')):
                continue
            fields = stat[stat.rfind(')') + 2:].split()
            result[proc.name] = (int(fields[11]) + int(fields[12]), fields[19], name)
        except (OSError, ValueError, IndexError):
            pass
    return result


def port_available(port):
    with socket.socket() as sock:
        try:
            sock.bind(('127.0.0.1', port))
        except OSError:
            return False
    return True


def identity_matches(actual, expected):
    return actual['start'] == expected['start_ticks'] and actual['exe'] == expected['exe']


class ExactProcess:
    """A pidfd prevents sending a signal to a replacement that reused a PID."""
    def __init__(self, pid, expected, forbidden_pids, info_reader=process_info):
        if pid in forbidden_pids:
            raise RuntimeError('Refusing to signal a protected process')
        self.fd = os.pidfd_open(pid)
        try:
            if not identity_matches(info_reader(pid), expected):
                raise RuntimeError('Process identity changed before acquiring it')
        except BaseException:
            os.close(self.fd)
            raise
        self.termination_sent = False

    def terminate(self):
        signal.pidfd_send_signal(self.fd, signal.SIGTERM)
        self.termination_sent = True

    def exited(self):
        return bool(select.select([self.fd], [], [], 0)[0])

    def close(self):
        os.close(self.fd)


class PeerBusy(RuntimeError):
    pass


class PeerMonitor:
    def __init__(self, full_expected, exclude=None):
        if not identity_matches(process_info(FULL_PID), full_expected):
            raise RuntimeError('Protected Full identity changed')
        self.snapshot = lambda: inference_snapshot(exclude)
        self.guard = ModelMeasurementGuard(FULL_PID, {FULL_PID: 18091}, self.snapshot)
        self.previous, self.then = self.snapshot(), time.monotonic()

    def check(self):
        state, current = self.guard.inspect()
        now = time.monotonic()
        samples, churn = activity(self.previous, current, now - self.then)
        self.previous, self.then = current, now
        if state['busy'] or churn or any(x['cpu_percent'] > 20 for x in samples):
            raise PeerBusy(json.dumps(dict(state=state, activity=samples, churn=churn)))


def transaction(ops):
    """Always restore after our original-server termination, including failures."""
    ops.measure_before()
    ops.gate_before_stop()
    try:
        ops.stop_original()
        ops.run_candidate()
    finally:
        # Even cleanup failure must reach restoration. Restoration itself refuses
        # to load a second Flash model until the candidate has actually exited.
        try:
            ops.stop_candidate()
        finally:
            ops.restore_original()
    ops.measure_after()


class Trial:
    def __init__(self, label, tokens=512):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', label):
            raise ValueError('Invalid result label')
        self.label, self.tokens = label, tokens
        self.plan = json.loads((STAGING / 'plan.json').read_text())
        self.full_expected = self.plan['protected'][str(FULL_PID)]
        self.original_expected = self.plan['protected'][str(ORIGINAL_PID)]
        self.out = BASE / 'results' / label
        self.state = dict(label=label, started=time.time(), phases=[], candidate_deployed=False)
        self.original = None
        self.candidate = None
        self.measurement = None
        self.restored = None
        self.cancelled = False
        self.environment = None
        self.lock = None

    def record(self, phase, **values):
        self.state['phases'].append(dict(time=time.time(), phase=phase, **values))
        if self.out.exists():
            temporary = self.out / 'result.json.tmp'
            temporary.write_text(json.dumps(self.state, indent=2) + '\n')
            temporary.replace(self.out / 'result.json')
        print(json.dumps(dict(phase=phase, **values)), flush=True)

    def check_cancel(self):
        if self.cancelled:
            raise RuntimeError('Trial cancellation requested; original restoration takes precedence')

    def preflight(self):
        p = self.plan
        info = process_info(ORIGINAL_PID)
        assert identity_matches(info, self.original_expected)
        assert info['command'] == p['original_command']
        assert info['affinity'] == list(range(128))
        assert info['cwd'] == str(BASE.parent.parent)
        assert identity_matches(process_info(FULL_PID), self.full_expected)
        self.environment = process_environment(ORIGINAL_PID)
        assert runtime_environment(self.environment) == p['original_environment']
        assert not self.environment.get('LD_TRACE_LOADED_OBJECTS')
        changed = {key for key in set(p['candidate_environment']) | set(p['original_environment'])
                   if p['candidate_environment'].get(key) != p['original_environment'].get(key)}
        assert changed == {'LD_LIBRARY_PATH', 'GGML_Q4E_EXPERT_EVEN_SPLIT'}
        assert port_available(TRIAL_PORT)
        pinned = Path(p['original_command'][0]).parent
        for name, digest in p['baseline_binary_sha256'].items():
            assert sha256(pinned / name) == digest, name
        assert sha256(p['candidate_library']) == p['candidate_library_sha256']
        validation = json.loads(Path(p['validation_audit']).read_text())
        assert validation['total_valid_phases'] == 64
        assert validation['total_cross_layout_values'] == 4915200
        assert validation['private_policy']['sha256'] == p['candidate_library_sha256']
        assert set(inference_snapshot()) == {str(FULL_PID), str(ORIGINAL_PID)}
        sources = [Path(__file__), BASE / 'measure-model-bandwidth.py',
                   BASE / 'model_measurement_guard.py', BASE / 'guarded_inference_request.py',
                   BASE / 'dram_bandwidth.py', BASE / 'inference_contention_guard.py']
        self.state['source_sha256'] = {str(x): sha256(x) for x in sources}
        self.state['plan_sha256'] = sha256(STAGING / 'plan.json')
        self.state['original_context'] = info
        self.state['original_runtime_env'] = p['original_environment']
        return dict(ready=True, execute=False, full=read_service(18091), qwen=read_service(18095),
                    original_pid=ORIGINAL_PID, trial_port=TRIAL_PORT,
                    candidate_sha256=p['candidate_library_sha256'],
                    original_cwd=info['cwd'], original_affinity=info['affinity'],
                    source_sha256=self.state['source_sha256'])

    def check_inputs(self):
        for path, expected in self.state['source_sha256'].items():
            assert sha256(path) == expected, path
        assert sha256(STAGING / 'plan.json') == self.state['plan_sha256']
        p = self.plan
        pinned = Path(p['original_command'][0]).parent
        for name, expected in p['baseline_binary_sha256'].items():
            assert sha256(pinned / name) == expected, name
        assert sha256(p['candidate_library']) == p['candidate_library_sha256']

    def quiet(self, ports, recovery=False):
        guard = ModelMeasurementGuard(next(iter(ports)), ports, inference_snapshot)
        before, then = inference_snapshot(), time.monotonic()
        quiet_since, reported = None, 0
        while True:
            if not recovery:
                self.check_cancel()
            if not identity_matches(process_info(FULL_PID), self.full_expected):
                raise RuntimeError('Protected Full identity changed')
            state, current = guard.inspect()
            now = time.monotonic()
            samples, churn = activity(before, current, now - then)
            busy = state['busy'] or churn or any(x['cpu_percent'] > 1 for x in samples)
            quiet_since = None if busy else now if quiet_since is None else quiet_since
            seconds = 0 if quiet_since is None else now - quiet_since
            if now - reported >= 30 or seconds >= 60:
                self.record('waiting_for_idle', recovery=recovery, quiet_seconds=seconds,
                            state=state, activity=samples, churn=churn)
                reported = now
            if seconds >= 60:
                return
            before, then = current, now
            # Five-second CPU deltas avoid treating one scheduler tick as a
            # sustained one-percent load; request/load monitors use 0.5 seconds.
            time.sleep(5)

    def spawn_server(self, command, environment, name):
        # Apply affinity in the child through taskset, without constraining the
        # controller or changing the affinity of any unrelated process.
        launch = ['taskset', '-c', '0-127', *command]
        log = (self.out / (name + '.log')).open('w')
        try:
            proc = subprocess.Popen(launch, env=environment, cwd=self.state['original_context']['cwd'],
                                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        finally:
            log.close()
        self.record('server_started', role=name, pid=proc.pid, command=command)
        return proc

    def stop_owned(self, proc):
        if proc is None:
            return
        assert proc.pid not in (FULL_PID, ORIGINAL_PID)
        assert self.restored is None or proc.pid != self.restored.pid, 'Restored service is handed off'
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=20)
        self.record('owned_process_terminal', pid=proc.pid, exit_code=proc.returncode)

    def wait_ready(self, proc, port, recovery=False):
        monitor = PeerMonitor(self.full_expected, exclude=proc.pid)
        deadline = time.monotonic() + 900
        while True:
            if not recovery:
                self.check_cancel()
            monitor.check()
            assert proc.poll() is None, 'Server exited while loading'
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=1) as response:
                    healthy = json.load(response).get('status') == 'ok'
            except OSError:
                healthy = False
            if healthy:
                return
            if time.monotonic() > deadline:
                raise TimeoutError('Server load exceeded 900 seconds')
            time.sleep(0.5)

    def verify_server(self, proc, candidate):
        p = self.plan
        info = process_info(proc.pid)
        assert info['exe'] == p['original_command'][0]
        assert info['command'] == p['trial_command' if candidate else 'original_command']
        assert info['affinity'] == list(range(128))
        expected_env = p['candidate_environment' if candidate else 'original_environment']
        assert runtime_environment(process_environment(proc.pid)) == expected_env
        maps = Path(f'/proc/{proc.pid}/maps').read_text()
        expected_library = p['candidate_library'] if candidate else str(
            Path(p['original_command'][0]).parent / 'libllama.so.0.3.0')
        lines = [line for line in maps.splitlines() if 'libllama.so' in line]
        assert lines and all(line.endswith(expected_library) for line in lines)
        self.check_inputs()
        self.record('server_verified', pid=proc.pid, candidate=candidate,
                    info=info, library=expected_library, mappings=lines)

    def measure(self, pid, port, alias, phase, watch_candidate=False):
        label = self.label + '-' + phase
        command = [sys.executable, '-u', str(BASE / 'measure-model-bandwidth.py'), label,
                   '--pid', str(pid), '--port', str(port), '--alias', alias,
                   '--drafts', '2', '--tokens', str(self.tokens), '--allowed-idle-pids', str(FULL_PID)]
        with (self.out / (phase + '-measurement.log')).open('w') as log:
            self.measurement = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                                stdin=subprocess.DEVNULL, start_new_session=True)
        self.record('measurement_started', phase_name=phase, pid=self.measurement.pid)
        monitor = PeerMonitor(self.full_expected, exclude=pid) if watch_candidate else None
        try:
            while self.measurement.poll() is None:
                self.check_cancel()
                if monitor:
                    monitor.check()
                time.sleep(0.5)
            assert self.measurement.returncode == 0, 'Measurement failed; inspect its recorded result'
        finally:
            # Terminating this client closes its socket; it never signals a model.
            self.stop_owned(self.measurement)
            self.measurement = None
        path = BASE / 'results' / label / 'result.json'
        result = json.loads(path.read_text())
        assert result.get('finished') and not result.get('error') and result['input_integrity_verified']
        assert len(result['measurements']) == 2 and all(c['pass_check'] for c in result['checks'])
        self.record('measurement_completed', phase_name=phase, result=str(path), sha256=sha256(path))

    def measure_before(self):
        self.measure(ORIGINAL_PID, 18095, 'flash-next', 'before')

    def gate_before_stop(self):
        self.quiet({ORIGINAL_PID: 18095, FULL_PID: 18091})
        self.check_inputs()
        self.check_cancel()
        current = process_info(ORIGINAL_PID)
        assert all(current[key] == value for key, value in self.state['original_context'].items() if key != 'state')
        assert runtime_environment(process_environment(ORIGINAL_PID)) == self.plan['original_environment']

    def stop_original(self):
        self.original = ExactProcess(ORIGINAL_PID, self.original_expected, {FULL_PID})
        # Recheck queues immediately before the authorized service interruption.
        guard = ModelMeasurementGuard(ORIGINAL_PID, {ORIGINAL_PID: 18095, FULL_PID: 18091}, inference_snapshot)
        guard.assert_idle()
        self.check_cancel()
        self.original.terminate()
        self.record('original_termination_sent', pid=ORIGINAL_PID)
        self.wait_original_exit()
        self.record('original_terminal', pid=ORIGINAL_PID)

    def wait_original_exit(self):
        reported = 0
        while not self.original.exited():
            if time.monotonic() - reported >= 30:
                self.record('waiting_for_original_exit', pid=ORIGINAL_PID)
                reported = time.monotonic()
            time.sleep(0.5)

    def run_candidate(self):
        self.quiet({FULL_PID: 18091})
        self.check_inputs()
        self.check_cancel()
        assert port_available(TRIAL_PORT)
        env = dict(self.environment)
        env.update(self.plan['candidate_environment'])
        self.candidate = self.spawn_server(self.plan['trial_command'], env, 'candidate')
        self.wait_ready(self.candidate, TRIAL_PORT)
        self.verify_server(self.candidate, True)
        self.measure(self.candidate.pid, TRIAL_PORT, 'qwen-even-split-trial', 'candidate', True)

    def stop_candidate(self):
        self.stop_owned(self.candidate)

    def restore_original(self):
        if self.original is None or not self.original.termination_sent:
            return
        self.wait_original_exit()
        if self.candidate is not None and self.candidate.poll() is None:
            raise RuntimeError('Restoration deferred: the candidate is still alive')
        self.record('restoring_original')
        attempt = 0
        while True:
            try:
                self.quiet({FULL_PID: 18091}, recovery=True)
            except (OSError, RuntimeError, KeyError, ValueError) as error:
                self.record('restore_waiting_for_monitor', reason=repr(error))
                time.sleep(5)
                continue
            assert port_available(18095), 'Original port was acquired by another service'
            self.check_inputs()
            attempt += 1
            restore = self.spawn_server(self.plan['original_command'], self.environment, 'restore-' + str(attempt))
            try:
                self.wait_ready(restore, 18095, recovery=True)
                self.verify_server(restore, False)
            except (PeerBusy, OSError, RuntimeError) as error:
                self.stop_owned(restore)
                self.record('restore_load_yielded', reason=repr(error))
                continue
            except BaseException:
                self.stop_owned(restore)
                raise
            self.restored = restore  # Handoff: never stop this process again.
            self.state['restored_pid'] = restore.pid
            self.record('original_restored', pid=restore.pid, port=18095)
            return

    def measure_after(self):
        self.check_cancel()
        assert self.restored is not None
        self.measure(self.restored.pid, 18095, 'flash-next', 'after')

    def execute(self):
        self.preflight()
        self.lock = (STAGING / 'lifecycle.lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.out.mkdir(exist_ok=False)
        for name in self.state['source_sha256']:
            (self.out / Path(name).name).write_bytes(Path(name).read_bytes())
        (self.out / 'plan.json').write_bytes((STAGING / 'plan.json').read_bytes())
        previous_handlers = {}
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            previous_handlers[signum] = signal.signal(signum, lambda *_: setattr(self, 'cancelled', True))
        try:
            transaction(self)
            self.state['completed'] = True
        except BaseException as error:
            self.state['error'] = repr(error)
            raise
        finally:
            self.state['finished'] = time.time()
            self.record('terminal', restored_pid=self.state.get('restored_pid'),
                        completed=self.state.get('completed', False), error=self.state.get('error'))
            if self.original is not None:
                self.original.close()
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
            self.lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', default='qwen-even-split-model-trial-0906')
    parser.add_argument('--execute', action='store_true', help='Only after explicit user authorization for the Qwen interruption')
    parser.add_argument('--tokens', type=int, default=512)
    args = parser.parse_args()
    trial = Trial(args.label, args.tokens)
    if args.execute:
        trial.execute()
    else:
        print(json.dumps(trial.preflight(), indent=2))
