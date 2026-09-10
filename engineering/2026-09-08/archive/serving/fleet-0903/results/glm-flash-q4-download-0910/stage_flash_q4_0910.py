#!/usr/bin/env python3
"""Stage the user-selected Flash Q4 shards on a dedicated RAM-backed model volume."""
import concurrent.futures
import fcntl
import hashlib
import http.client
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import threading
import time
import urllib.error
import urllib.request

from qwen_split_trial import inference_snapshot, process_info
from model_measurement_guard import ModelMeasurementGuard

BASE = Path(__file__).resolve().parent
MANIFEST = BASE / 'results/glm-flash-q4-manifest-0910.json'
OUT = BASE / 'results/glm-flash-q4-download-0910'
MODEL = 'GLM-5.3-Flash-621d456e93e9'
ROOT = Path('/home/kwebb/.local/share/ai-models') / MODEL / 'UD-Q4_K_XL'
ROOTS = {'ram_volume': ROOT}
RESERVE = {'ram_volume': 8 << 30}


def ensure_volume():
    ROOT.mkdir(parents=True, exist_ok=True)
    assert ROOT.resolve() == ROOT and not ROOT.is_symlink()
    check = subprocess.run(['findmnt', '-rn', '-M', str(ROOT), '-o', 'FSTYPE,OPTIONS'],
                           capture_output=True, text=True)
    if check.returncode:
        assert check.returncode == 1 and not any(ROOT.iterdir())
        assert memory_available() > 199707321347 + (64 << 30)
        command = ['sudo', '-n', 'mount', '-t', 'tmpfs', '-o',
                   f'size=210G,nosuid,nodev,noexec,mode=700,uid={os.getuid()},gid={os.getgid()}',
                   'flash-q4-0910', str(ROOT)]
        subprocess.run(command, check=True, capture_output=True, text=True)
        check = subprocess.run(['findmnt', '-rn', '-M', str(ROOT), '-o', 'FSTYPE,OPTIONS'],
                               check=True, capture_output=True, text=True)
    assert check.stdout.startswith('tmpfs ')
    assert all(flag in check.stdout for flag in ('nosuid', 'nodev', 'noexec'))
    assert ROOT.stat().st_uid == os.getuid() and ROOT.stat().st_mode & 0o777 == 0o700
    return dict(path=str(ROOT), mount=check.stdout.strip(), volatile=True,
                purpose='Retain Q8 artifacts while staging the explicitly selected Q4 model; no global tmpfs policy change.')



def memory_available():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('MemAvailable is missing')


def main():
    os.umask(0o077)
    manifest = json.loads(MANIFEST.read_text())
    assert manifest['repository'] == 'unsloth/GLM-5.3-Flash-GGUF'
    assert manifest['revision'] == '621d456e93e926e4b52f85cff5f634358c1828f9'
    files = manifest['files']
    assert len(files) == 6 and sum(x['bytes'] for x in files) == 199707321347
    OUT.mkdir(exist_ok=True)
    cancel = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda *_: cancel.set())
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        handoff = json.loads((BASE / 'results/flash-hugepages250-comparison-0910/result.json').read_text())
        assert handoff['finished'] and handoff['qwen_restored'] and not handoff.get('restoration_error')
        peer = handoff['restored_qwen_pid']
        assert process_info(peer)['start'] == handoff['restored_qwen']['start']
        assert set(inference_snapshot()) == {str(peer)}
        ModelMeasurementGuard(peer, {peer: 18095}, inference_snapshot).assert_idle()
        volume = ensure_volume()
        for path in ROOTS.values():
            path.mkdir(parents=True, exist_ok=True)
            assert path.is_dir() and not path.is_symlink()
        records = []
        for i, item in enumerate(files, 1):
            name = Path(item['name']).name
            assert name == f'GLM-5.3-Flash-UD-Q4_K_XL-{i:05d}-of-00006.gguf'
            location = 'ram_volume'
            records.append(dict(item, location=location, path=str(ROOTS[location] / name),
                                logical_path=str(ROOT / name)))

        def size(record):
            path = Path(record['path'])
            partial = path.with_suffix('.gguf.part')
            assert not (path.exists() and partial.exists()), str(path)
            for current in (path, partial, path):
                try:
                    info = current.lstat()
                except FileNotFoundError:
                    continue
                assert stat.S_ISREG(info.st_mode), str(current)
                assert info.st_size <= record['bytes'], str(current)
                return info.st_size
            return 0

        def capacity():
            remaining = {name: 0 for name in ROOTS}
            for record in records:
                remaining[record['location']] += record['bytes'] - size(record)
            free = {name: shutil.disk_usage(path).free for name, path in ROOTS.items()}
            for name in ROOTS:
                if free[name] < remaining[name] + RESERVE[name]:
                    raise RuntimeError(f'Insufficient {name} capacity for remaining shards and reserve')
            available = memory_available()
            if available < remaining['ram_volume'] + (64 << 30):
                raise RuntimeError('Insufficient available RAM for remaining tmpfs shards and reserve')
            return dict(remaining_bytes=remaining, free_bytes=free, memory_available_bytes=available)

        initial_capacity = capacity()
        state = dict(repository=manifest['repository'], revision=manifest['revision'], quant='UD-Q4_K_XL',
                     destination=str(ROOT), records=records, expected_bytes=sum(r['bytes'] for r in records),
                     started=time.time(), pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
                     verified={}, errors={}, complete=False, temporary_storage=True, volume=volume, user_authorized_selection=True,
                     initial_capacity=initial_capacity, inference_before=inference_snapshot(),
                     manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
                     script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        (OUT / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
        (OUT / 'manifest.json').write_bytes(MANIFEST.read_bytes())
        state_lock = threading.Lock()

        def persist():
            with state_lock:
                state['updated'] = time.time()
                state['downloaded_bytes'] = sum(size(record) for record in records)
                state['free_bytes'] = {name: shutil.disk_usage(path).free for name, path in ROOTS.items()}
                state['memory_available_bytes'] = memory_available()
                temporary = OUT / 'status.json.tmp'
                temporary.write_text(json.dumps(state, indent=2) + '\n')
                temporary.replace(OUT / 'status.json')
                print(json.dumps({k: state[k] for k in ('updated', 'downloaded_bytes', 'expected_bytes', 'complete')}
                                 | {'verified_shards': len(state['verified'])}), flush=True)

        def download(record):
            path = Path(record['path'])
            partial = path.with_suffix('.gguf.part')
            digest = hashlib.sha256()
            existing = path if path.exists() else partial
            offset = size(record)
            if offset:
                with existing.open('rb') as stream:
                    for block in iter(lambda: stream.read(8 << 20), b''):
                        if cancel.is_set():
                            raise InterruptedError('Download cancelled; partial shards retained')
                        digest.update(block)
            if not path.exists():
                with partial.open('ab') as output:
                    for attempt in range(10):
                        if offset == record['bytes']:
                            break
                        if cancel.is_set():
                            raise InterruptedError('Download cancelled; partial shards retained')
                        url = (f'https://huggingface.co/{manifest["repository"]}/resolve/'
                               f'{manifest["revision"]}/{record["name"]}?download=true&flash_q4_0910={time.time_ns()}')
                        request = urllib.request.Request(url, headers={'Range': f'bytes={offset}-'})
                        try:
                            with urllib.request.urlopen(request, timeout=60) as response:
                                if response.status == 206:
                                    assert response.headers.get('Content-Range') == f'bytes {offset}-{record["bytes"]-1}/{record["bytes"]}'
                                else:
                                    assert response.status == 200 and offset == 0
                                next_check = offset
                                while block := response.read(8 << 20):
                                    if cancel.is_set():
                                        raise InterruptedError('Download cancelled; partial shards retained')
                                    if offset >= next_check:
                                        capacity()
                                        next_check = offset + (512 << 20)
                                    assert offset + len(block) <= record['bytes']
                                    assert output.write(block) == len(block)
                                    digest.update(block)
                                    offset += len(block)
                            output.flush()
                            if offset != record['bytes']:
                                raise OSError('Incomplete HTTP body')
                        except InterruptedError:
                            raise
                        except (OSError, urllib.error.URLError, http.client.HTTPException) as error:
                            output.flush()
                            print(json.dumps(dict(retry=record['name'], attempt=attempt + 1, offset=offset,
                                                  error_type=type(error).__name__, http_status=getattr(error, 'code', None))), flush=True)
                            if attempt == 9:
                                raise
                            if cancel.wait(min(15, 2 ** attempt)):
                                raise InterruptedError('Download cancelled; partial shards retained')
                    assert offset == record['bytes']
            assert digest.hexdigest() == record['sha256'], (str(path), 'SHA-256 mismatch')
            if not path.exists():
                partial.replace(path)
            logical = Path(record['logical_path'])
            if logical != path:
                if logical.is_symlink():
                    assert logical.readlink() == path
                else:
                    assert not logical.exists()
                    logical.symlink_to(path)
            stat = path.stat()
            with state_lock:
                state['verified'][record['name']] = dict(path=str(path), logical_path=str(logical),
                    bytes=stat.st_size, sha256=digest.hexdigest(), inode=stat.st_ino,
                    mtime_ns=stat.st_mtime_ns, time=time.time(), location=record['location'])
            print(json.dumps(dict(verified=record['name'], location=record['location'])), flush=True)

        try:
            persist()
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                futures = {pool.submit(download, record): record for record in records}
                pending = set(futures)
                while pending:
                    done, pending = concurrent.futures.wait(pending, timeout=25,
                        return_when=concurrent.futures.FIRST_EXCEPTION)
                    for future in done:
                        try:
                            future.result()
                        except BaseException as error:
                            with state_lock:
                                state['errors'][futures[future]['name']] = dict(type=type(error).__name__, message=str(error))
                            cancel.set()
                    persist()
            assert not state['errors'], 'One or more shards failed; see the recorded error'
            assert len(state['verified']) == len(records)
            state['complete'] = True
        except BaseException as error:
            state['error'] = repr(error)
            raise
        finally:
            state['finished'] = time.time()
            persist()


if __name__ == '__main__':
    main()
