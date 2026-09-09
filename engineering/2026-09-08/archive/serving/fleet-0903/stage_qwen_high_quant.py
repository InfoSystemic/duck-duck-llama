#!/usr/bin/env python3
"""Resume the pinned Qwen Q6 download in-place and verify every shard."""
import concurrent.futures
import fcntl
import hashlib
import http.client
import json
import os
from pathlib import Path
import shutil
import threading
import time
import urllib.request
import urllib.error

BASE = Path(__file__).resolve().parent
MANIFEST = BASE / 'results/qwen-higher-quant-selection-0907.json'
OUT = BASE / 'results/qwen-q6-download-0907'
QUANT = 'UD-Q6_K_XL'


def main():
    manifest = json.loads(MANIFEST.read_text())
    candidate = manifest['candidates'][QUANT]
    destination = Path('/home/user/.local/share/ai-models') / (
        'Qwen3.8-Flash-Next-' + manifest['revision'][:12])
    destination.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    lock = (OUT / 'download.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = dict(repo=manifest['repo'], revision=manifest['revision'], quant=QUANT,
                 destination=str(destination), expected_bytes=candidate['bytes'],
                 started=time.time(), pid=os.getpid(), verified={}, errors={}, complete=False)
    progress_lock = threading.Lock()

    def downloaded_bytes():
        return sum(path.stat().st_size for path in destination.rglob('*')
                   if path.is_file() and (path.suffix in ('.gguf', '.incomplete', '.part')))

    remaining = candidate['bytes'] - min(candidate['bytes'], downloaded_bytes())
    free = shutil.disk_usage(destination).free
    assert free >= remaining + (20 << 30), (free, remaining, 'Need 20 GiB spare')

    def persist():
        with progress_lock:
            state['updated'] = time.time()
            state['downloaded_bytes'] = downloaded_bytes()
            state['free_bytes'] = shutil.disk_usage(destination).free
            tmp = OUT / 'status.json.tmp'
            tmp.write_text(json.dumps(state, indent=2) + '\n')
            tmp.replace(OUT / 'status.json')
            print(json.dumps({key: state[key] for key in
                              ('updated', 'downloaded_bytes', 'expected_bytes',
                               'free_bytes', 'complete')}
                             | {'verified_shards': len(state['verified'])}), flush=True)

    def download(record):
        path = destination / record['name']
        partial = path.with_suffix('.gguf.part')
        path.parent.mkdir(exist_ok=True)
        # Reuse partial data from the initial hub transfer. No second model copy.
        previous = list(destination.rglob('*.' + record['sha256'] + '.incomplete'))
        if not path.exists() and not partial.exists() and previous:
            assert len(previous) == 1
            previous[0].replace(partial)
        digest = hashlib.sha256()
        existing = path if path.exists() else partial
        offset = existing.stat().st_size if existing.exists() else 0
        assert offset <= record['bytes'], (existing, offset)
        if offset:
            with existing.open('rb') as stream:
                for block in iter(lambda: stream.read(8 << 20), b''):
                    digest.update(block)
        if not path.exists():
            with partial.open('ab') as output:
                for attempt in range(10):
                    if offset == record['bytes']:
                        break
                    url = ('https://huggingface.co/' + manifest['repo'] + '/resolve/'
                           + manifest['revision'] + '/' + record['name']
                           + '?download=true&q6_full=' + str(time.time_ns()))
                    request = urllib.request.Request(url, headers={'Range': f'bytes={offset}-'})
                    try:
                        with urllib.request.urlopen(request, timeout=120) as response:
                            expected_range = f'bytes {offset}-{record["bytes"]-1}/{record["bytes"]}'
                            if response.status == 206:
                                assert response.headers.get('Content-Range') == expected_range
                            else:
                                assert response.status == 200 and offset == 0
                            while block := response.read(8 << 20):
                                assert offset + len(block) <= record['bytes']
                                output.write(block)
                                digest.update(block)
                                offset += len(block)
                        output.flush()
                        if offset != record['bytes']:
                            raise OSError('Incomplete HTTP body')
                    except (OSError, urllib.error.URLError, http.client.HTTPException) as error:
                        output.flush()
                        print(json.dumps({'retry_shard': record['name'], 'attempt': attempt + 1,
                                          'offset': offset, 'error_type': type(error).__name__,
                                          'http_status': getattr(error, 'code', None)}), flush=True)
                        if attempt == 9:
                            raise
                        time.sleep(min(15, 2 ** attempt))
                assert offset == record['bytes'], ('Incomplete shard', offset)
        actual = digest.hexdigest()
        assert actual == record['sha256'], (path, 'SHA-256 mismatch', actual)
        if not path.exists():
            partial.replace(path)
        with progress_lock:
            state['verified'][record['name']] = dict(bytes=record['bytes'],
                sha256=actual, time=time.time(), inode=path.stat().st_ino,
                mtime_ns=path.stat().st_mtime_ns)
        print(json.dumps({'verified': record['name'], 'bytes': record['bytes']}), flush=True)

    try:
        persist()
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(download, record) for record in candidate['files']]
            pending = set(futures)
            while pending:
                done, pending = concurrent.futures.wait(pending, timeout=30,
                    return_when=concurrent.futures.FIRST_EXCEPTION)
                for future in done:
                    try:
                        future.result()
                    except BaseException as error:
                        record = candidate['files'][futures.index(future)]
                        state['errors'][record['name']] = repr(error)
                        print(json.dumps({'failed_shard': record['name'],
                                          'error_type': type(error).__name__}), flush=True)
                persist()
        assert not state['errors'], state['errors']
        assert len(state['verified']) == len(candidate['files'])
        state['complete'] = True
    except BaseException as error:
        state['error'] = repr(error)
        raise
    finally:
        state['finished'] = time.time()
        persist()
        lock.close()


if __name__ == '__main__':
    main()
