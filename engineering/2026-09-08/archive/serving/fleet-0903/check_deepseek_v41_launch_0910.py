#!/usr/bin/env python3
"""Refresh public V4.1 runtime/weight availability and local launch prerequisites."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.request
import urllib.parse
from select_flash_q4_0910c import Manager
from glm_flash_q8_trial import memory_status, node_memory_status
from qwen_split_trial import inference_snapshot, sha256

BASE = Path(__file__).resolve().parent
OUT = BASE / 'results/deepseek-v41-launch-preflight-0910'

def main():
    assert os.sched_getaffinity(0) == {127} and not OUT.exists()
    OUT.mkdir()
    sources = []
    def fetch(item):
        label, url = item
        path = OUT / label
        try:
            req = urllib.request.Request(url, headers={'User-Agent':'llama-llama-duck-v41-launch-preflight', 'Accept':'application/vnd.github+json' if 'api.github.com' in url else '*/*'})
            with urllib.request.urlopen(req, timeout=30) as response:
                raw = response.read((8 << 20)+1)
                assert len(raw) <= 8 << 20
                status = response.status
            path.write_bytes(raw)
            return dict(file=str(path), url=url, sha256=sha256(path), bytes=len(raw), status=status, fetched_at=time.time())
        except Exception as error:
            return dict(file=str(path), url=url, error_type=type(error).__name__, status=getattr(error,'code',None), fetched_at=time.time())
    jobs = [('official-metadata.json','https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4.1-Flash?blobs=true'),
        ('hf-search.json','https://huggingface.co/api/models?search=DeepSeek-V4.1&limit=100'),
        ('upstream-commit.json','https://api.github.com/repos/ggml-org/llama.cpp/commits?per_page=1'),
        ('upstream-issues.json','https://api.github.com/search/issues?q='+urllib.parse.quote('repo:ggml-org/llama.cpp "V4.1"')+'&per_page=50'),
        ('runtime-repositories.json','https://api.github.com/search/repositories?q='+urllib.parse.quote('DeepSeek-V4.1')+'&per_page=30')]
    with ThreadPoolExecutor(max_workers=4) as pool: sources.extend(pool.map(fetch,jobs))
    def data(label):
        p=OUT/label
        return json.loads(p.read_text()) if p.exists() else None
    candidates = data('hf-search.json') or []
    ids = sorted({x['id'] for x in candidates if 'gguf' in x['id'].lower()})
    ids = sorted(set(ids+['vcruz305/DeepSeek-V4.1-Flash-GGUF']))
    with ThreadPoolExecutor(max_workers=4) as pool:
        sources.extend(pool.map(fetch,[(f'quant-{i}.json','https://huggingface.co/api/models/'+model+'?blobs=true') for i,model in enumerate(ids)]))
    quant=[]
    for i, model in enumerate(ids):
        m=data(f'quant-{i}.json')
        if m:
            files=[dict(name=s['rfilename'],bytes=s.get('size',s.get('lfs',{}).get('size')),sha256=s.get('lfs',{}).get('sha256')) for s in m.get('siblings',[]) if s['rfilename'].lower().endswith('.gguf')]
            quant.append(dict(id=model,revision=m.get('sha'),gguf_files=files))
    upstream=data('upstream-commit.json')
    commit=upstream[0]['sha'] if upstream else None
    registrations=[]
    if commit:
        jobs=[('upstream-arch.cpp',f'https://raw.githubusercontent.com/ggml-org/llama.cpp/{commit}/src/llama-arch.cpp'),
            ('upstream-deepseek.py',f'https://raw.githubusercontent.com/ggml-org/llama.cpp/{commit}/conversion/deepseek.py')]
        with ThreadPoolExecutor(max_workers=2) as pool: sources.extend(pool.map(fetch,jobs))
        for name,_ in jobs:
            p=OUT/name
            if p.exists(): registrations.append(dict(file=str(p),sha256=sha256(p),identifiers=sorted(set(re.findall(r'DeepseekV41\w*|deepseek_v41\w*|deepseek41\w*|deepseek4_1\w*',p.read_text())))))
    peer=Manager().validate_current()
    official=data('official-metadata.json')
    weights=[] if not official else [s for s in official.get('siblings',[]) if s['rfilename'].endswith('.safetensors')]
    storage=[]
    for path in ['/home/user','/models']:
        v=os.statvfs(path);storage.append(dict(path=path,available_bytes=v.f_bavail*v.f_frsize))
    result=dict(passed=True,finished=time.time(),source_sha256=sha256(__file__),sources=sources,
        official_revision=None if not official else official.get('sha'),
        checkpoint_shards=len(weights),checkpoint_bytes=sum(s.get('size',s.get('lfs',{}).get('size',0)) for s in weights),
        hf_candidates=[x['id'] for x in candidates],quant_candidates=quant,upstream_commit=commit,registration_checks=registrations,
        issues=[] if not data('upstream-issues.json') else [dict(number=x['number'],title=x['title'],state=x['state'],url=x['html_url'],pull_request=x.get('pull_request')) for x in data('upstream-issues.json').get('items',[])],
        runtime_repositories=[] if not data('runtime-repositories.json') else [dict(name=x['full_name'],description=x.get('description'),url=x['html_url']) for x in data('runtime-repositories.json').get('items',[])],
        selected_flash_pid=peer['pid'],selected_flash_start=peer['info']['start'],storage=storage,memory=memory_status(),nodes=node_memory_status(),inference=inference_snapshot(),
        full_model_launched=False,scope='Read-only launch preflight. No process stopped or model weights removed.')
    (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['sources','nodes','inference']},indent=2),flush=True)

if __name__=='__main__':
    os.umask(0o077)
    main()
