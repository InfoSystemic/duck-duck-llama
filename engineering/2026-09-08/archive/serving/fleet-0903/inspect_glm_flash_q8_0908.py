#!/usr/bin/env python3
"""Fetch a pinned public file manifest and inspect space; do not download weights."""
import json
from pathlib import Path
import re
import shutil
import time
import urllib.request

base = Path(__file__).resolve().parent
out = base / 'results/glm-flash-q8-manifest-0908.json'
assert not out.exists()
repo = 'unsloth/GLM-5.3-Flash-GGUF'
revision = '621d456e93e926e4b52f85cff5f634358c1828f9'
url = f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true'
with urllib.request.urlopen(url, timeout=30) as response:
    data = response.read(16 * 1024 * 1024 + 1)
assert len(data) <= 16 * 1024 * 1024
manifest = json.loads(data)
assert manifest['sha'] == revision
files = []
for item in manifest['siblings']:
    if not item['rfilename'].startswith('Q8_0/') or not item['rfilename'].endswith('.gguf'):
        continue
    assert re.fullmatch(r'Q8_0/GLM-5\.3-Flash-Q8_0-\d{5}-of-00008\.gguf', item['rfilename'])
    digest = item['lfs']['sha256']
    assert re.fullmatch(r'[0-9a-f]{64}', digest)
    files.append(dict(name=item['rfilename'], bytes=item['size'], sha256=digest))
files.sort(key=lambda x: x['name'])
assert len(files) == 8 and sum(x['bytes'] for x in files) == 340981966112
memory = {k: int(v.split()[0]) * 1024 for k, v in
          (line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
          if k in ('MemTotal', 'MemAvailable', 'Shmem', 'SwapFree')}
result = dict(observed=time.time(), repository=repo, revision=revision, source=url,
              files=files, memory_bytes=memory,
              free_bytes={p: shutil.disk_usage(p).free for p in ('/home/kwebb', '/models', '/dev/shm')},
              downloaded_weights=False, near_lossless_quality_verified=False)
out.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
