#!/usr/bin/env python3
"""Retrieve only the two complete native Engram projection/gate tensor groups."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import time

from audit_deepseek_v41_tensors_0910 import read_range
from qwen_split_trial import sha256

BASE = Path(__file__).resolve().parent
AUDIT = BASE / 'results/deepseek-v41-tensor-audit-0910/result.json'


def main():
    out = Path(sys.argv[1])
    assert not (out / 'weight-manifest.json').exists()
    audit = json.loads(AUDIT.read_text())
    assert audit['passed'] and audit['revision'] == 'fb2764a5cf321eaa5070ca8f9e892818f477c16d'
    jobs = []
    headers = {}
    for shard in audit['shards']:
        hp = AUDIT.parent / 'headers' / (shard['file'] + '.json')
        assert sha256(hp) == shard['header_sha256']
        header = json.loads(hp.read_text())
        for layer in [1, 14]:
            for suffix, shape, dtype in [('q_weight', [4, 5120], 'BF16'), ('k_weight', [4, 5120], 'BF16'),
                    ('wkv.weight', [25600, 6144], 'F8_E4M3'), ('wkv.scale', [800, 192], 'F8_E8M0')]:
                name = f'layers.{layer}.engram.{suffix}'
                if name not in header: continue
                tensor = header[name]
                assert tensor['shape'] == shape and tensor['dtype'] == dtype
                begin, end = tensor['data_offsets']
                byte_count = end - begin
                assert byte_count == shape[0] * shape[1] * (2 if dtype == 'BF16' else 1)
                start = 8 + shard['header_bytes'] + begin
                jobs.append(dict(tensor=name, layer=layer, suffix=suffix, dtype=dtype, shape=shape,
                    bytes=byte_count, source_url=shard['source_url'], start=start, end=start+byte_count-1,
                    total=shard['size'], header_file=str(hp), header_sha256=shard['header_sha256'],
                    publisher_whole_shard_sha256=shard['publisher_sha256']))
                headers[str(hp)] = sha256(hp)
    assert len(jobs) == 8 and sum(j['bytes'] for j in jobs) == 315043840
    st = os.statvfs(out)
    assert st.f_bavail * st.f_frsize > sum(j['bytes'] for j in jobs) + (8 << 30)
    tensor_dir = out / 'weights'
    tensor_dir.mkdir(exist_ok=False)
    records = []
    # Each output file is created exclusively. Every response is bounded to 4 MiB;
    # read_range rejects an ignored Range before accepting any response bytes.
    for job in jobs:
        path = tensor_dir / (job['tensor'] + '.bin')
        chunks = [(offset, min(4 << 20, job['bytes'] - offset)) for offset in range(0, job['bytes'], 4 << 20)]
        def fetch(chunk):
            offset, size = chunk
            for attempt in range(3):
                try:
                    data = read_range(job['source_url'], job['start'] + offset, job['start'] + offset + size - 1, job['total'])
                    return offset, data
                except Exception as error:
                    if attempt == 2: raise RuntimeError('Bounded projection download failed: ' + type(error).__name__) from None
            raise AssertionError('unreachable')
        import hashlib
        spans = []
        with path.open('xb') as handle, ThreadPoolExecutor(max_workers=4) as pool:
            for offset, data in pool.map(fetch, chunks):
                assert handle.tell() == offset
                handle.write(data)
                spans.append(dict(start=job['start'] + offset, end=job['start'] + offset + len(data)-1,
                                  bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
                if len(spans) % 8 == 0:
                    print(json.dumps(dict(tensor=job['tensor'], downloaded=handle.tell(), total=job['bytes'])), flush=True)
        assert path.stat().st_size == job['bytes']
        records.append(dict(job, file=str(path), sha256=sha256(path), ranges=spans))
        print(json.dumps(dict(completed=job['tensor'], bytes=job['bytes'])), flush=True)
    result = dict(passed=True, finished=time.time(), revision=audit['revision'], records=records,
        retrieved_bytes=sum(j['bytes'] for j in records), whole_shard_hashes_verified=False,
        input_sha256={str(AUDIT): sha256(AUDIT), str(Path(__file__)): sha256(__file__),
            str(BASE / 'audit_deepseek_v41_tensors_0910.py'): sha256(BASE / 'audit_deepseek_v41_tensors_0910.py'), **headers},
        scope='Only complete Engram projection/gate tensors; native full checkpoint is not downloaded.')
    (out / 'weight-manifest.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    os.umask(0o077)
    main()
