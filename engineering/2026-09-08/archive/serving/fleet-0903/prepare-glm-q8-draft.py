#!/usr/bin/env python3
"""Fetch pinned Q8 NextN tensor ranges, then assemble a mixed-precision sidecar."""
import concurrent.futures
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / 'engines/llama.cpp-glm5n-goal-0904/gguf-py'))
import gguf

base = Path(__file__).resolve().parent
plan = json.loads((base / 'results/glm-mtp-q8-download-plan.json').read_text())
stage = Path(plan['staging_dir'])
source = stage / 'q8-source.sparse.gguf'
old_path = Path('/models/gguf/GLM-5.3-Flash/MTP/GLM-5.3-Flash-MTP-IQ2_XXS.gguf')
output = stage / 'GLM-5.3-Flash-MTP-Q8_0.gguf'
partial = output.with_suffix('.gguf.partial')
progress = stage / 'download-progress.jsonl'
if output.exists() or partial.exists():
    raise FileExistsError(f'Output or partial already exists: {output}')
old = gguf.GGUFReader(old_path)
src = gguf.GGUFReader(source)
old_t = {t.name: t for t in old.tensors}
new_t = {t.name: t for t in src.tensors if t.name.startswith('blk.45.')}
assert set(new_t) == {n for n in old_t if n.startswith('blk.45.')}
assert set(new_t) == {t['name'] for t in plan['selected']}
for name, tensor in new_t.items():
    assert list(tensor.shape) == list(old_t[name].shape), name

chunks = []
for item in plan['selected']:
    for offset in range(item['offset'], item['offset'] + item['size'], 16 * 1024**2):
        chunks.append((offset, min(16 * 1024**2, item['offset'] + item['size'] - offset)))
done = {}
if progress.exists():
    for line in progress.read_text().splitlines():
        item = json.loads(line)
        done[(item['offset'], item['size'])] = item
fd = os.open(source, os.O_RDWR)
for key, item in list(done.items()):
    if hashlib.sha256(os.pread(fd, key[1], key[0])).hexdigest() != item['sha256']:
        del done[key]

def fetch(chunk):
    offset, size = chunk
    end = offset + size - 1
    for attempt in range(6):
        try:
            url = plan['url'] + f'?download=true&flash_goal_range={offset}-{end}'
            request = urllib.request.Request(url, headers={'Range': f'bytes={offset}-{end}'})
            with urllib.request.urlopen(request, timeout=180) as response:
                content_range = response.headers.get('Content-Range')
                expected = f'bytes {offset}-{end}/{plan["source_size"]}'
                if response.status != 206 or content_range != expected:
                    raise ValueError((response.status, content_range, expected))
                data = response.read(size + 1)
            if len(data) != size:
                raise ValueError(f'Wrong response size: {len(data)} != {size}')
            view = memoryview(data)
            written = 0
            while written < size:
                n = os.pwrite(fd, view[written:], offset + written)
                if n <= 0:
                    raise OSError('short pwrite')
                written += n
            return dict(offset=offset, size=size, sha256=hashlib.sha256(data).hexdigest())
        except Exception as error:
            if attempt == 5:
                raise
            print(f'Retry offset={offset} attempt={attempt + 1}: {error}', flush=True)
            time.sleep(min(15, 2 ** attempt))

started = time.monotonic()
downloaded = sum(x['size'] for x in done.values())
total = sum(size for _, size in chunks)
print(f'Validated {len(new_t)} tensor shapes; fetching {total:,} bytes in {len(chunks)} ranges', flush=True)
with progress.open('a') as log, concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    futures = [pool.submit(fetch, chunk) for chunk in chunks if chunk not in done]
    last_report = 0
    for future in concurrent.futures.as_completed(futures):
        item = future.result()
        log.write(json.dumps(item) + '\n'); log.flush()
        downloaded += item['size']
        if time.monotonic() - last_report >= 10 or downloaded == total:
            print(f'Downloaded {downloaded:,}/{total:,} bytes in {time.monotonic()-started:.1f}s', flush=True)
            last_report = time.monotonic()
os.close(fd)

spec = importlib.util.spec_from_file_location('extract_mtp', root / 'serving/glm53-flash/extract-glm5next-mtp-gguf.py')
extract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extract)
writer = gguf.GGUFWriter(partial, arch='glm5next', endianess=old.endianess)
extract.copy_metadata(old, writer, 'GLM-5.3-Flash-MTP-Q8_0-with-original-shared-head')
writer.add_uint32('general.file_type', int(gguf.LlamaFileType.MOSTLY_Q8_0))
writer.add_string('flash_goal.draft_source_repo', plan['repo'])
writer.add_string('flash_goal.draft_source_revision', plan['revision'])
selected = [(new_t.get(t.name, t), src if t.name in new_t else old) for t in old.tensors]
for tensor, reader in selected:
    writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype,
                           tensor.data.nbytes, tensor.tensor_type)
writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_ti_data_to_file()
manifest = dict(repo=plan['repo'], revision=plan['revision'], url=plan['url'],
                shared_tensors_source=str(old_path), tensors=[])
for tensor, reader in selected:
    digest = hashlib.sha256(tensor.data).hexdigest()
    writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)
    manifest['tensors'].append(dict(name=tensor.name, type=int(tensor.tensor_type),
                                    shape=list(map(int, tensor.shape)), bytes=int(tensor.n_bytes),
                                    sha256=digest, source='Q8 range download' if tensor.name in new_t else str(old_path)))
    print(f'Wrote {tensor.name}: {tensor.n_bytes:,} bytes', flush=True)
writer.close()
os.replace(partial, output)
check = gguf.GGUFReader(output)
for tensor, expected in zip(check.tensors, manifest['tensors'], strict=True):
    assert tensor.name == expected['name']
    assert hashlib.sha256(tensor.data).hexdigest() == expected['sha256'], tensor.name
manifest.update(output=str(output), output_bytes=output.stat().st_size,
                output_sha256=hashlib.file_digest(output.open('rb'), 'sha256').hexdigest(),
                verified=True)
(base / 'results/glm-mtp-q8-artifact.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(f'Created and verified {output}: {output.stat().st_size:,} bytes', flush=True)
