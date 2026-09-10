#!/usr/bin/env python3
"""Complete the final Q4 shard with bounded parallel ranges and a full checksum."""
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import threading
import time
import urllib.request

from qwen_high_quant_trial import atomic_json
from qwen_split_trial import inference_snapshot, process_info, sha256
from stage_flash_q4_0910 import BASE, MANIFEST, OUT as DOWNLOAD, ROOT, memory_available

OUT = BASE / 'results/glm-flash-q4-final-shard-0910'
CHUNK = 256 << 20


def main():
    os.umask(0o077)
    assert os.sched_getaffinity(0) == {127}
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not OUT.exists()
        previous = json.loads((DOWNLOAD / 'status.json').read_text())
        assert previous['finished'] and not previous['complete'] and len(previous['verified']) == 5
        assert 'InterruptedError' in json.dumps(previous.get('errors', {}))
        assert previous['quant'] == 'UD-Q4_K_XL' and previous['revision'] == '621d456e93e926e4b52f85cff5f634358c1828f9'
        assert sha256(MANIFEST) == previous['manifest_sha256']
        assert sha256(BASE / 'stage_flash_q4_0910.py') == previous['script_sha256']
        peer = json.loads((BASE / 'results/flash-hugepages250-comparison-0910/result.json').read_text())
        pid = peer['restored_qwen_pid']
        assert process_info(pid)['start'] == peer['restored_qwen']['start']
        assert set(inference_snapshot()) == {str(pid)}
        for entry in previous['verified'].values():
            info = Path(entry['path']).stat()
            assert (info.st_size, info.st_ino, info.st_mtime_ns) == (entry['bytes'], entry['inode'], entry['mtime_ns'])
        missing = [row for row in previous['records'] if row['name'] not in previous['verified']]
        record, = missing
        assert Path(record['path']).parent == ROOT
        final = Path(record['path'])
        assert final.name == 'GLM-5.3-Flash-UD-Q4_K_XL-00005-of-00006.gguf' and not final.exists()
        partial = final.with_suffix('.gguf.part')
        parallel = final.with_suffix('.gguf.parallel.part')
        info = partial.lstat()
        assert stat.S_ISREG(info.st_mode) and 0 < info.st_size < record['bytes'] and not parallel.exists()
        seed = info.st_size
        remaining = record['bytes'] - seed
        assert memory_available() > remaining + (64 << 30)
        vfs = os.statvfs(ROOT)
        assert vfs.f_bavail * vfs.f_frsize > remaining + (8 << 30)
        OUT.mkdir()
        atomic_json(OUT / 'original-download-status.json', previous)
        plan = dict(time=time.time(), record=record, prefix_bytes=seed, prefix_inode=info.st_ino,
                    prefix_mtime_ns=info.st_mtime_ns, prefix_sha256=sha256(partial), chunk_bytes=CHUNK,
                    source_sha256=sha256(__file__), initial_memory_available=memory_available(),
                    previous_status_sha256=sha256(OUT / 'original-download-status.json'),
                    scope='Previously verified five shards are retained by identity. Only disjoint missing ranges '
                          'of the sixth shard are written, followed by a SHA-256 of its entire payload.')
        atomic_json(OUT / 'plan.json', plan)
        partial.rename(parallel)
        fd = os.open(parallel, os.O_RDWR | os.O_NOFOLLOW)
        assert os.fstat(fd).st_ino == info.st_ino
        cancel = threading.Event()
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, lambda *_: cancel.set())
        ranges = [(start, min(record['bytes'], start + CHUNK)) for start in range(seed, record['bytes'], CHUNK)]
        state = dict(started=time.time(), pid=os.getpid(), complete=False, finished_ranges=[], errors=[],
                     prefix_bytes=seed, expected_bytes=record['bytes'], total_ranges=len(ranges),
                     file=str(parallel), plan_sha256=sha256(OUT / 'plan.json'))
        mutex = threading.Lock()
        def persist():
            with mutex:
                state['updated'] = time.time()
                state['covered_bytes'] = seed + sum(row['end']-row['start'] for row in state['finished_ranges'])
                atomic_json(OUT / 'result.json', state)
                print(json.dumps({k:state[k] for k in ('updated','covered_bytes','expected_bytes','complete')}
                                 | {'finished_ranges':len(state['finished_ranges']),'total_ranges':len(ranges)}), flush=True)
        def fetch(bounds):
            start, end = bounds
            for attempt in range(5):
                if cancel.is_set():
                    raise InterruptedError('Parallel completion cancelled; range audit retained')
                digest = hashlib.sha256()
                offset = start
                url = (f'https://huggingface.co/{previous["repository"]}/resolve/{previous["revision"]}/'
                       f'{record["name"]}?download=true&q4_range={start}_{attempt}_{time.time_ns()}')
                try:
                    request = urllib.request.Request(url, headers={'Range':f'bytes={start}-{end-1}'})
                    with urllib.request.urlopen(request, timeout=60) as response:
                        assert response.status == 206
                        assert response.headers.get('Content-Range') == f'bytes {start}-{end-1}/{record["bytes"]}'
                        while data := response.read(min(8 << 20, end-offset) if offset < end else 1):
                            assert offset + len(data) <= end
                            if cancel.is_set():
                                raise InterruptedError('Parallel completion cancelled')
                            assert memory_available() > 64 << 30
                            written = os.pwrite(fd, data, offset)
                            assert written == len(data)
                            digest.update(data)
                            offset += len(data)
                    assert offset == end
                    with mutex:
                        state['finished_ranges'].append(dict(start=start,end=end,sha256=digest.hexdigest()))
                    return
                except (OSError, AssertionError) as error:
                    if attempt == 4:
                        raise
                    print(json.dumps(dict(retry_range=start,attempt=attempt+1,error_type=type(error).__name__)),flush=True)
                    if cancel.wait(min(10, 2**attempt)):
                        raise InterruptedError('Parallel completion cancelled')
        try:
            persist()
            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
                futures = [pool.submit(fetch,bounds) for bounds in ranges]
                pending = set(futures)
                while pending:
                    done,pending = concurrent.futures.wait(pending,timeout=15,return_when=concurrent.futures.FIRST_EXCEPTION)
                    for future in done:
                        try:
                            future.result()
                        except BaseException as error:
                            state['errors'].append(repr(error))
                            cancel.set()
                    persist()
            assert not state['errors'] and len(state['finished_ranges']) == len(ranges)
            assert sorted((r['start'],r['end']) for r in state['finished_ranges']) == ranges
            os.fsync(fd)
            assert os.fstat(fd).st_size == record['bytes']
            digest = sha256(parallel)
            assert digest == record['sha256'], 'Complete shard SHA-256 differs from publisher'
            parallel.rename(final)
            st = final.stat()
            completed = dict(path=str(final), logical_path=record['logical_path'], bytes=st.st_size,
                             sha256=digest,inode=st.st_ino,mtime_ns=st.st_mtime_ns,time=time.time(),location=record['location'])
            state.update(complete=True,final_sha256=digest,final_record=completed,finished=time.time())
            persist()
            # The first writer's incomplete status is preserved above. Complete the
            # common download manifest with explicit provenance for the final shard.
            final_status = dict(previous)
            final_status['verified'] = dict(previous['verified'], **{record['name']:completed})
            final_status.update(complete=True,errors={},finished=time.time(),updated=time.time(),
                downloaded_bytes=previous['expected_bytes'],memory_available_bytes=memory_available(),
                completion_script_sha256=sha256(__file__),completion_audit=str(OUT / 'result.json'),
                completion_audit_sha256=sha256(OUT / 'result.json'),
                resumed_from_status=str(OUT / 'original-download-status.json'))
            final_status.pop('error',None)
            atomic_json(DOWNLOAD / 'status.json',final_status)
            print(json.dumps(dict(all_six_shards_verified=True,bytes=final_status['downloaded_bytes'])),flush=True)
        except BaseException as error:
            state['error'] = repr(error)
            state['finished'] = time.time()
            persist()
            raise
        finally:
            os.close(fd)


if __name__ == '__main__':
    main()
