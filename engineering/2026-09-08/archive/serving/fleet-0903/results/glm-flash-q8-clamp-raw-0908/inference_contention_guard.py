"""Keep private benchmark servers from competing with other inference."""
import json
import os
from pathlib import Path
import threading
import time


def activity(before, after, elapsed, own_pid=None):
    own_pid = str(own_pid) if own_pid is not None else None
    previous = {p: v for p, v in before.items() if p != own_pid}
    current = {p: v for p, v in after.items() if p != own_pid}
    churn = set(previous) ^ set(current)
    samples = []
    for pid, value in current.items():
        old = previous.get(pid)
        if old is None:
            continue
        if old[1] != value[1]:
            churn.add(pid)
            continue
        percent = 100 * max(0, value[0] - old[0]) / os.sysconf('SC_CLK_TCK') / max(elapsed, 1e-9)
        samples.append(dict(pid=int(pid), port=value[2], cpu_percent=round(percent, 2)))
    return samples, sorted(churn)


def wait_for_idle(snapshot, status_path, allowed_idle_pids=('4005448',), quiet_seconds=60, interval=5):
    allowed = set(allowed_idle_pids)
    before = snapshot()
    previous_time = time.monotonic()
    quiet_since = None
    last_report = 0
    while True:
        time.sleep(interval)
        now = time.monotonic()
        after = snapshot()
        samples, churn = activity(before, after, now - previous_time)
        foreign = sorted(set(after) - allowed)
        busy = foreign or churn or any(s['cpu_percent'] > 1 for s in samples)
        quiet_since = None if busy else now if quiet_since is None else quiet_since
        quiet = 0 if quiet_since is None else now - quiet_since
        status = dict(time=time.time(), other_inference=samples, foreign_pids=foreign,
                      process_churn=churn, quiet_seconds=quiet, required_quiet_seconds=quiet_seconds)
        Path(status_path).write_text(json.dumps(status, indent=2) + '\n')
        if not busy and quiet >= quiet_seconds:
            return status
        if now - last_report >= 30:
            print(f'Waiting for inference to be idle: quiet {quiet:.0f}/{quiet_seconds:.0f}s; foreign={foreign}; activity={samples}', flush=True)
            last_report = now
        before, previous_time = after, now


class InferenceContentionGuard:
    def __init__(self, server, snapshot, evidence_path, interval=2, threshold=100, consecutive=2, grace=5):
        self.server = server
        self.snapshot = snapshot
        self.evidence_path = Path(evidence_path)
        self.interval = interval
        self.threshold = threshold
        self.consecutive = consecutive
        self.grace = grace
        self.stop_event = threading.Event()
        self.info = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        return self.info

    def _record(self):
        self.evidence_path.write_text(json.dumps(self.info, indent=2) + '\n')

    def _run(self):
        before = self.snapshot()
        previous_time = time.monotonic()
        busy_count = 0
        while not self.stop_event.wait(self.interval):
            if self.server.poll() is not None:
                return
            now = time.monotonic()
            after = self.snapshot()
            samples, churn = activity(before, after, now - previous_time, self.server.pid)
            busy_count = busy_count + 1 if any(s['cpu_percent'] > self.threshold for s in samples) else 0
            if churn or busy_count >= self.consecutive:
                self.info = dict(time=time.time(), test_pid=self.server.pid,
                                 reason='Other inference became active or changed during this isolated test.',
                                 other_inference=samples, process_churn=churn,
                                 terminate_sent=False, kill_sent=False)
                if not self.stop_event.is_set() and self.server.poll() is None:
                    self.server.terminate()
                    self.info['terminate_sent'] = True
                self._record()
                print('Cancelling only this benchmark server due to competing inference.', flush=True)
                if not self.stop_event.wait(self.grace) and self.server.poll() is None:
                    self.server.kill()
                    self.info['kill_sent'] = True
                    self._record()
                return
            before, previous_time = after, now
