#!/usr/bin/env python3
"""Inspect pinned Q8 headers while the full payload download proceeds."""
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import time
import urllib.request

base = Path(__file__).resolve().parent
out = base / 'results/glm-flash-q8-inventory-0908'
out.mkdir(exist_ok=False)
spec = importlib.util.spec_from_file_location('header_parser', base.parent / 'glm-sr950/remote-gguf-bytes-per-token.py')
parser = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parser)
manifest = json.loads((base / 'results/glm-flash-q8-download-0908/status.json').read_text())
tensors, headers = [], []
metadata = None
for record in manifest['records']:
    path = Path(record['path'])
    partial = path.with_suffix('.gguf.part')
    local = path if path.exists() else partial
    size = min(record['bytes'], 16 << 20 if metadata is None else 1 << 20)
    if local.exists() and local.stat().st_size >= size:
        with local.open('rb') as stream:
            data = stream.read(size)
        source = str(local)
    else:
        url = f'https://huggingface.co/{manifest["repository"]}/resolve/{manifest["revision"]}/{record["name"]}'
        request = urllib.request.Request(url + f'?download=true&header={time.time_ns()}',
                                         headers={'Range': f'bytes=0-{size - 1}'})
        with urllib.request.urlopen(request, timeout=60) as response:
            assert response.status == 206
            assert response.headers.get('Content-Range') == f'bytes 0-{size - 1}/{record["bytes"]}'
            data = response.read(size + 1)
        source = url
    assert len(data) == size
    version, meta, rows = parser.parse_header(data)
    if metadata is None:
        metadata = meta
    assert version == 3
    tensors.extend(rows)
    (out / (path.name + '.header')).write_bytes(data)
    headers.append(dict(shard=record['name'], bytes=len(data), tensors=len(rows), source=source,
                        header_sha256=hashlib.sha256(data).hexdigest(), full_sha256=record['sha256']))
arch = metadata['general.architecture']
layers = metadata[arch + '.block_count'] - metadata.get(arch + '.nextn_predict_layers', 0)
used = metadata[arch + '.expert_used_count']
assert arch == 'glm5next' and layers == 45 and used == 8
assert len({t[0] for t in tensors}) == len(tensors)
records = []
groups = defaultdict(lambda: dict(bytes=0, active_bytes=0, tensors=0))
for name, shape, kind in tensors:
    layer = re.match(r'blk\.(\d+)\.', name)
    if layer and int(layer[1]) >= layers:
        continue
    if not name.endswith('.weight'):
        continue
    total = parser.tensor_bytes(shape, kind)
    routed = '_exps.' in name and len(shape) == 3
    indexed = 'token_embd' in name
    active = 0 if indexed else total * used / shape[2] if routed else total
    category = 'indexed' if indexed else 'routed' if routed else 'dense_and_other'
    item = dict(name=name, shape=shape, type=parser.TYPE_INFO[kind][0], bytes=total,
                active_bytes=active, category=category)
    records.append(item)
    for group_name in (category, item['type']):
        group = groups[group_name]
        group['bytes'] += total
        group['active_bytes'] += active
        group['tensors'] += 1
gb = sum(r['active_bytes'] for r in records) / 1e9
result = dict(repository=manifest['repository'], revision=manifest['revision'], architecture=arch,
              target_layers=layers, experts=metadata[arch + '.expert_count'], active_experts=used,
              canonical_active_weight_gb_per_raw_token=gb,
              weight_only_tok_s_at_285_gb_s=285 / gb, weight_only_tok_s_at_380_gb_s=380 / gb,
              groups=groups, records=records, headers=headers,
              note='Header-based planning only. Excludes runtime replication, rereads, speculation, and activation/cache traffic. Full shard verification is separate and still downloading.')
(out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k not in ('records', 'headers')}, indent=2))
