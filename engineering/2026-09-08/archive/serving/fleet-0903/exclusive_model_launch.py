"""Prevent independent whole-server launchers from loading over a resident model."""
import fcntl
import os
from pathlib import Path


class ModelLaunchConflict(RuntimeError):
    pass


def inference_processes(proc_root=Path('/proc')):
    found = []
    for process in proc_root.iterdir():
        if not process.name.isdigit():
            continue
        try:
            name = (process / 'comm').read_text().strip()
            if name != 'llama-server' and not name.startswith(('glm-mtp-head', 'qwen-mtp-head')):
                continue
            state = (process / 'stat').read_text().rsplit(')', 1)[1].split()[0]
            if state in ('Z', 'X'):
                continue
            args = [x.decode(errors='replace') for x in (process / 'cmdline').read_bytes().split(b'\0') if x]
            def option(flag):
                values = []
                for i, value in enumerate(args):
                    if value == flag and i + 1 < len(args):
                        values.append(args[i + 1])
                    elif value.startswith(flag + '='):
                        values.append(value.split('=', 1)[1])
                return values[-1] if values else None
            model = option('--model')
            found.append(dict(pid=int(process.name), name=name, port=option('--port'),
                              model=Path(model).name if model else None))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return sorted(found, key=lambda row: row['pid'])


class LaunchLease:
    def __init__(self, transaction, residency):
        self.transaction, self.residency = transaction, residency

    def close(self):
        self.residency.close()
        self.transaction.close()

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()


def acquire_exclusive_model_launch(base, proc_root=Path('/proc')):
    """Keep the returned lease alive until exec; never stop another process.

    The existing lifecycle transaction lock coordinates with tuning helpers and
    closes on exec. A separate residency lock survives exec for guarded public
    launchers, so two such launchers cannot both pass an empty process scan.
    Resident models started through older launchers are caught by the scan,
    including models still loading and not yet listening on their HTTP port.
    """
    results = Path(base) / 'results'
    transaction_path = results / 'qwen-q6-trial-0907/lifecycle.lock'
    residency_path = results / 'whole-server-public-launch.lock'
    transaction_path.parent.mkdir(parents=True, exist_ok=True)
    transaction = transaction_path.open('a')
    residency = None
    try:
        try:
            fcntl.flock(transaction, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ModelLaunchConflict('A model tuning or startup transaction is in progress; retry after it finishes.') from error
        peers = inference_processes(proc_root)
        if peers:
            description = '; '.join(f"PID {p['pid']}, port {p['port'] or 'not yet known'}, {p['model'] or p['name']}" for p in peers)
            raise ModelLaunchConflict('An inference process is already running: ' + description +
                                      '. Use the existing service or coordinate its shutdown before starting another whole-server model.')
        residency = residency_path.open('a')
        try:
            fcntl.flock(residency, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ModelLaunchConflict('Another guarded whole-server model is starting or still running.') from error
        os.set_inheritable(transaction.fileno(), False)
        os.set_inheritable(residency.fileno(), True)
        return LaunchLease(transaction, residency)
    except BaseException:
        if residency is not None:
            residency.close()
        transaction.close()
        raise
