#!/usr/bin/env python3
"""Inspect GGUF tensor storage without reading all tensor payloads."""
from collections import defaultdict
import glob
import json
from pathlib import Path
import re
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'engines/llama.cpp-q4e-goal-0904/gguf-py'))
from gguf import GGUFReader

stats=defaultdict(lambda: dict(stored_bytes=0, active_bytes=0, tensors=0, shapes=set()))
for file in sorted(glob.glob(sys.argv[1])):
    reader=GGUFReader(file, 'r')
    for tensor in reader.tensors:
        name=tensor.name
        if not name.endswith('.weight'):
            continue
        family=re.sub(r'blk\.\d+\.', 'blk.N.',name)
        key=(family,str(tensor.tensor_type))
        info=stats[key];info['stored_bytes']+=tensor.n_bytes;info['tensors']+=1
        shape=tuple(int(n) for n in tensor.shape);info['shapes'].add(shape)
        # Eight selected experts; embeddings are indexed rather than scanned.
        active=tensor.n_bytes * 8 / shape[2] if '_exps.' in name and len(shape)>2 else tensor.n_bytes
        if 'token_embd' in name:active=0
        info['active_bytes']+=active
out=[]
for (family,kind),v in stats.items():
    out.append(dict(family=family,type=kind,stored_gib=v['stored_bytes']/2**30,
                    active_mib=v['active_bytes']/2**20,tensors=v['tensors'],shapes=sorted(v['shapes'])))
out.sort(key=lambda x:-x['active_mib'])
Path(sys.argv[2]).write_text(json.dumps(out,indent=2))
print(json.dumps(out[:14],indent=2))
