#!/usr/bin/env python3
"""Inspect the pinned, user-selected Flash Q4 files and real host capacity."""
import json
from pathlib import Path
import re
import shutil
import time
import urllib.request

from glm_flash_q8_trial import memory_status, node_memory_status
from qwen_split_trial import inference_snapshot

BASE = Path(__file__).resolve().parent
OUT = BASE / 'results/glm-flash-q4-manifest-0910.json'


def main():
    assert not OUT.exists()
    repo = 'unsloth/GLM-5.3-Flash-GGUF'
    revision = '621d456e93e926e4b52f85cff5f634358c1828f9'
    url = f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true'
    with urllib.request.urlopen(url, timeout=30) as response:
        data = response.read((16 << 20) + 1)
    assert len(data) <= 16 << 20
    manifest = json.loads(data)
    assert manifest['sha'] == revision
    files = []
    for item in manifest['siblings']:
        if not item['rfilename'].startswith('UD-Q4_K_XL/') or not item['rfilename'].endswith('.gguf'):
            continue
        assert re.fullmatch(r'UD-Q4_K_XL/GLM-5\.3-Flash-UD-Q4_K_XL-\d{5}-of-00006\.gguf', item['rfilename'])
        digest = item['lfs']['sha256']
        assert re.fullmatch(r'[0-9a-f]{64}', digest)
        files.append(dict(name=item['rfilename'], bytes=item['size'], sha256=digest))
    files.sort(key=lambda x: x['name'])
    assert len(files) == 6 and sum(x['bytes'] for x in files) == 199707321347
    result = dict(observed=time.time(), repository=repo, revision=revision, quant='UD-Q4_K_XL',
                  source=url, files=files, memory_bytes=memory_status(), node_memory_bytes=node_memory_status(),
                  free_bytes={p: shutil.disk_usage(p).free for p in ('/home/kwebb', '/models', '/dev/shm')},
                  inference=inference_snapshot(), downloaded_weights=False,
                  user_selection='User explicitly instructed: switch GLM-5.3-Flash to Q4.',
                  qwen_selection='Keep current Q6 unless separately instructed; Q4 question is an assessment request.')
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'inference'}, indent=2))


if __name__ == '__main__':
    main()
