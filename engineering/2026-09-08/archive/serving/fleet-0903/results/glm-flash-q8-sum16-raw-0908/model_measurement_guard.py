"""Observe model identities and queues; never signal a model process."""
import json
import time
import urllib.request
from pathlib import Path

from inference_contention_guard import activity


def read_service(port):
    def get(endpoint):
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/{endpoint}', timeout=2) as response:
            return response.read().decode()
    slots = json.loads(get('slots'))
    queues = [float(line.split()[-1]) for line in get('metrics').splitlines()
              if not line.startswith('#') and 'requests_deferred' in line]
    if len(queues) != 1:
        raise RuntimeError('Missing queue metric')
    return dict(processing=sum(bool(slot['is_processing']) for slot in slots), queued=queues[0])


class ModelMeasurementGuard:
    def __init__(self, target_pid, ports, snapshot, service_reader=read_service):
        self.target = str(target_pid)
        self.ports = {str(pid): port for pid, port in ports.items()}
        self.snapshot = snapshot
        self.read_service = service_reader
        initial = snapshot()
        self.identities = {pid: initial[pid][1] for pid in self.ports}
        assert self.target in self.identities
        self.previous, self.previous_time = initial, time.monotonic()

    def inspect(self, allow_target=False):
        current = self.snapshot()
        for pid, identity in self.identities.items():
            if pid not in current or current[pid][1] != identity:
                raise RuntimeError(f'Model process identity changed: {pid}')
        foreign = sorted(set(current) - set(self.ports))
        services = {pid: self.read_service(port) for pid, port in self.ports.items()}
        busy = bool(foreign) or any(
            state['queued'] > 0 or state['processing'] > (1 if allow_target and pid == self.target else 0)
            for pid, state in services.items())
        return dict(servers=services, foreign_pids=foreign, busy=busy), current

    def assert_idle(self):
        state, _ = self.inspect()
        if state['busy']:
            raise RuntimeError(f'Model work is active or queued: {state}')
        return state

    def abort_reason(self):
        state, current = self.inspect(allow_target=True)
        now = time.monotonic()
        other, churn = activity(self.previous, current, now - self.previous_time, self.target)
        self.previous, self.previous_time = current, now
        if state['busy'] or churn or any(sample['cpu_percent'] > 20 for sample in other):
            return dict(reason='Other model work is active or queued; releasing only this request',
                        state=state, other_inference=other, churn=churn)
        return None

    def reset_activity(self):
        self.previous, self.previous_time = self.snapshot(), time.monotonic()

    def pause_idle(self, seconds):
        deadline = time.monotonic() + seconds
        while True:
            self.assert_idle()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(0.5, remaining))

    def wait_idle(self, status_path, quiet_seconds=60):
        previous, then = self.snapshot(), time.monotonic()
        quiet_since = None
        last_report = 0
        while True:
            time.sleep(5)
            state, current = self.inspect()
            now = time.monotonic()
            samples, churn = activity(previous, current, now - then)
            busy = state['busy'] or churn or any(sample['cpu_percent'] > 1 for sample in samples)
            quiet_since = None if busy else now if quiet_since is None else quiet_since
            quiet = 0 if quiet_since is None else now - quiet_since
            status = dict(time=time.time(), quiet_seconds=quiet, required_quiet_seconds=quiet_seconds,
                          state=state, other_inference=samples, churn=churn)
            Path(status_path).write_text(json.dumps(status, indent=2) + '\n')
            if quiet >= quiet_seconds:
                return status
            if now - last_report >= 30:
                print(json.dumps(dict(waiting_for_idle=True, quiet_seconds=quiet, busy=bool(busy))), flush=True)
                last_report = now
            previous, then = current, now
