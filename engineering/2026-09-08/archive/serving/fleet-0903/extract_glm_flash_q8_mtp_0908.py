#!/usr/bin/env python3
"""Extract and verify the matching, all-Q8 Flash MTP sidecar after staging."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parent
AI = BASE.parents[1]
EXTRACTOR = AI / 'serving/glm53-flash/extract-glm5next-mtp-gguf.py'
OUT = BASE / 'results/glm-flash-q8-mtp-0908'
MODEL = Path('/home/user/.local/share/ai-models/GLM-5.3-Flash-621d456e93e9')
DESTINATION = MODEL / 'MTP/GLM-5.3-Flash-MTP-Q8_0-621d456e93e9.gguf'
sys.path.insert(0, str(AI / 'engines/llama.cpp-glm53-flash/gguf-py'))

from qwen_split_trial import sha256
from qwen_high_quant_trial import atomic_json


def verified_sources():
    state = json.loads((BASE / 'results/glm-flash-q8-download-0908/status.json').read_text())
    manifest = json.loads((BASE / 'results/glm-flash-q8-manifest-0908.json').read_text())
    assert state['complete'] and not state['errors'] and len(state['verified']) == 8
    assert state['revision'] == manifest['revision'] == '621d456e93e926e4b52f85cff5f634358c1828f9'
    assert sha256(BASE / 'results/glm-flash-q8-manifest-0908.json') == state['manifest_sha256']
    for record in state['records']:
        expected = next(x for x in manifest['files'] if x['name'] == record['name'])
        verified = state['verified'][record['name']]
        path = Path(record['logical_path'])
        actual = path.stat()
        assert path.resolve() == Path(record['path'])
        assert actual.st_size == expected['bytes'] == verified['bytes']
        assert actual.st_ino == verified['inode'] and actual.st_mtime_ns == verified['mtime_ns']
        assert expected['sha256'] == verified['sha256']
    return state


def tensor_digest(tensor):
    return hashlib.sha256(memoryview(tensor.data).cast('B')).hexdigest()


def tensor_record(tensor):
    return dict(name=tensor.name, shape=[int(x) for x in tensor.shape],
                type=tensor.tensor_type.name, bytes=int(tensor.n_bytes), sha256=tensor_digest(tensor))


def main():
    import gguf

    os.umask(0o077)
    state = verified_sources()
    assert not DESTINATION.exists() and not DESTINATION.with_name(DESTINATION.name + '.partial').exists()
    OUT.mkdir()
    DESTINATION.parent.mkdir(exist_ok=True)
    result = dict(started=time.time(), passed=False, destination=str(DESTINATION),
                  revision=state['revision'], source_manifest_sha256=state['manifest_sha256'], tensors=[])
    command = [sys.executable, str(EXTRACTOR), state['records'][0]['logical_path'], str(DESTINATION)]
    result.update(command=command, extractor_sha256=sha256(EXTRACTOR), python_executable=sys.executable)
    (OUT / EXTRACTOR.name).write_bytes(EXTRACTOR.read_bytes())
    (OUT / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    save = lambda: atomic_json(OUT / 'result.json', result)
    proc = None
    try:
        selected = {}
        for record in state['records']:
            reader = gguf.GGUFReader(record['logical_path'], 'r')
            for tensor in reader.tensors:
                if tensor.name in ('output.weight', 'output_norm.weight', 'token_embd.weight') or tensor.name.startswith('blk.45.'):
                    assert tensor.name not in selected
                    selected[tensor.name] = tensor_record(tensor)
            del reader
        assert len(selected) == 32, len(selected)
        assert {r['type'] for r in selected.values()} == {'Q8_0', 'F32'}
        assert selected['output.weight']['type'] == selected['token_embd.weight']['type'] == 'Q8_0'
        result['tensors'] = list(selected.values())
        result['tensor_bytes'] = sum(x['bytes'] for x in selected.values())
        assert shutil.disk_usage(DESTINATION.parent).free > result['tensor_bytes'] + (5 << 30)
        save()
        print(json.dumps(dict(extracting=True, tensors=len(selected), tensor_bytes=result['tensor_bytes'])), flush=True)
        with (OUT / 'extract.log').open('w') as log:
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            assert proc.wait(timeout=600) == 0, 'MTP extraction failed; see extract.log'
        output = gguf.GGUFReader(str(DESTINATION), 'r')
        actual = {tensor.name: tensor_record(tensor) for tensor in output.tensors}
        assert actual == selected, 'Extracted tensor data differs from the pinned Q8 source'
        assert output.get_field('general.architecture').contents() == 'glm5next'
        assert int(output.get_field('glm5next.block_count').contents()) == 46
        assert int(output.get_field('glm5next.nextn_predict_layers').contents()) == 1
        del output
        verified_sources()
        assert sha256(EXTRACTOR) == result['extractor_sha256']
        info = DESTINATION.stat()
        result.update(passed=True, bytes=info.st_size, inode=info.st_ino, mtime_ns=info.st_mtime_ns,
                      sha256=sha256(DESTINATION), type_counts={kind: sum(r['type'] == kind for r in selected.values()) for kind in ('Q8_0', 'F32')})
        print(json.dumps({k: result[k] for k in ('passed', 'destination', 'bytes', 'sha256', 'type_counts')}), flush=True)
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=20)
        result['finished'] = time.time()
        save()


if __name__ == '__main__':
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, lambda *_: (_ for _ in ()).throw(InterruptedError('Extraction cancelled')))
    with (BASE / 'results/glm-flash-q8-download-0908/download.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
