#!/usr/bin/env python3
"""Move 12 GiB of owned Flash Q4 tmpfs pages from crowded node 2 to node 1."""
from collections import Counter
import ctypes
import fcntl
import json
import mmap
import os
from pathlib import Path
import signal
import time

from model_measurement_guard import ModelMeasurementGuard
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import inference_snapshot, sha256, process_info

BASE = Path(__file__).resolve().parent
OUT = BASE / 'results/glm-flash-q4-page-rebalance-0910'
LIMIT = 12 << 30
PAGE = os.sysconf('SC_PAGE_SIZE')


def nodes():
    result = {}
    for path in sorted(Path('/sys/devices/system/node').glob('node[0-9]*')):
        values = {}
        for line in (path / 'meminfo').read_text().splitlines():
            fields = line.split()
            key = fields[2].rstrip(':')
            if key in ('MemTotal', 'MemFree', 'FilePages', 'Shmem', 'SReclaimable', 'AnonPages'):
                values[key] = int(fields[3]) * 1024
        values['estimated_available'] = values['MemFree'] + max(0, values['FilePages'] - values['Shmem']) + values['SReclaimable']
        result[path.name] = values
    return result


def main():
    assert PAGE == 4096
    os.umask(0o077)
    state = json.loads((BASE / 'results/glm-flash-q4-download-0910/status.json').read_text())
    assert state['complete'] and len(state['verified']) == 6 and state['quant'] == 'UD-Q4_K_XL'
    previous = json.loads((BASE / 'results/glm-flash-q4-selection-0910b/result.json').read_text())
    assert previous['finished'] and previous['qwen_restored'] and not previous.get('restoration_error')
    pid = previous['restored_qwen_pid']
    assert process_info(pid)['start'] == previous['restored_qwen']['start']
    assert set(inference_snapshot()) == {str(pid)}
    guard = ModelMeasurementGuard(pid, {pid: 18095}, inference_snapshot)
    guard.assert_idle()
    def verify_sources():
        for record in state['verified'].values():
            st = Path(record['path']).stat()
            assert (st.st_size, st.st_ino, st.st_mtime_ns) == (record['bytes'],record['inode'],record['mtime_ns'])
    verify_sources()
    before = nodes()
    assert before['node1']['estimated_available'] > LIMIT + (32 << 30), before['node1']
    OUT.mkdir()
    (OUT / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    result = dict(started=time.time(), passed=False, from_node=2, to_node=1,
                  limit_bytes=LIMIT, moved_bytes=0, before=before, files=[],
                  scope='Only pages mapped from the owned, verified Flash tmpfs shards; no model-file byte writes')
    library = ctypes.CDLL('libnuma.so.1', use_errno=True)
    move = library.move_pages
    move.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.POINTER(ctypes.c_void_p),
                     ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    move.restype = ctypes.c_long
    cancelled = False
    def cancel(*_):
        nonlocal cancelled
        cancelled = True
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, cancel)
    save = lambda: atomic_json(OUT / 'result.json', result)
    order = [record for record in state['records'] if record['bytes'] > (32 << 30)]
    touched = []
    last_report = 0
    try:
        save()
        for sampled in order:
            if result['moved_bytes'] >= LIMIT:
                break
            record = next(r for r in state['records'] if r['path'] == sampled['path'])
            assert record['location'] == 'ram_volume'
            path = Path(record['path'])
            assert path.parent == Path('/home/kwebb/.local/share/ai-models/GLM-5.3-Flash-621d456e93e9/UD-Q4_K_XL')
            row = dict(path=str(path), moved_bytes=0, migration_status={}, bytes=record['bytes'], expected_sha256=record['sha256'])
            result['files'].append(row)
            with path.open('rb') as stream:
                mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_COPY)
                first = None
                try:
                    first = ctypes.c_char.from_buffer(mapping)
                    address = ctypes.addressof(first)
                    for offset in range(0, record['bytes'], 64 << 20):
                        if cancelled:
                            raise InterruptedError('Page placement cancelled; completed moves retain identical data')
                        if result['moved_bytes'] >= LIMIT:
                            break
                        guard.assert_idle()
                        assert nodes()['node1']['estimated_available'] > (32 << 30), 'Destination reserve reached'
                        positions = range(offset, min(offset + (64 << 20), record['bytes']), PAGE)
                        for position in positions:
                            mapping[position]
                        addresses = (ctypes.c_void_p * len(positions))(*(address + position for position in positions))
                        status = (ctypes.c_int * len(positions))(*([-999] * len(positions)))
                        code = move(0, len(positions), addresses, None, status, 0)
                        if code < 0:
                            raise OSError(ctypes.get_errno(), os.strerror(ctypes.get_errno()))
                        assert all(0 <= node <= 3 for node in status), dict(Counter(status))
                        indices = [i for i, node in enumerate(status) if node == 2]
                        indices = indices[:(LIMIT - result['moved_bytes']) // PAGE]
                        if not indices:
                            continue
                        selected = (ctypes.c_void_p * len(indices))(*(addresses[i] for i in indices))
                        destinations = (ctypes.c_int * len(indices))(*([1] * len(indices)))
                        migrated = (ctypes.c_int * len(indices))(*([-999] * len(indices)))
                        # MPOL_MF_MOVE (2) excludes pages mapped by other processes.
                        code = move(0, len(indices), selected, destinations, migrated, 2)
                        if code < 0:
                            raise OSError(ctypes.get_errno(), os.strerror(ctypes.get_errno()))
                        outcomes = Counter(migrated)
                        moved = outcomes.get(1, 0) * PAGE
                        row['moved_bytes'] += moved
                        result['moved_bytes'] += moved
                        row['migration_status'] = dict(Counter(row['migration_status']) + Counter({str(k): v for k, v in outcomes.items()}))
                        assert all(node == 1 or node == -16 for node in outcomes), dict(outcomes)
                        if time.monotonic() - last_report >= 10:
                            save()
                            print(json.dumps(dict(moved_bytes=result['moved_bytes'], limit_bytes=LIMIT, current_file=path.name)), flush=True)
                            last_report = time.monotonic()
                finally:
                    del first
                    mapping.close()
            if row['moved_bytes']:
                touched.append((path, row))
        assert result['moved_bytes'] == LIMIT, 'Insufficient movable owned pages on source node'
        result['after_migration'] = nodes()
        save()
        for path, row in touched:
            print(json.dumps(dict(verifying_unchanged_bytes=path.name)), flush=True)
            row['sha256_after'] = sha256(path)
            assert row['sha256_after'] == row['expected_sha256'], 'File bytes changed during page placement'
            save()
        verify_sources()
        result.update(passed=True, after=nodes())
        print(json.dumps(dict(passed=True, moved_bytes=result['moved_bytes'], files_verified=len(touched))), flush=True)
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        result['finished'] = time.time()
        save()


if __name__ == '__main__':
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lifecycle:
        fcntl.flock(lifecycle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
