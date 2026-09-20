#!/usr/bin/env python3
"""Rank matrix work in existing target graph samples; no new inference or benchmark."""
from collections import defaultdict
from hashlib import sha256
import json,re
from pathlib import Path
from summarize_optrace import HEADER,ROW
HERE=Path(__file__).resolve().parent
result={'scope':'Existing single-worker target graph samples; operation attribution, not end-to-end speed or a predicted optimization gain. Long sample predates bounded KV rollback. Matrix work is unchanged by cache-only MTP.', 'graphs':[]}
for filename in ['profile-kv-deployed-short-profile.optrace.log','profile-kpool-long-1.optrace.log']:
    path=HERE/filename;graphs=[];current={}
    for line in path.read_text().splitlines():
        if m:=HEADER.search(line):
            ix,cpu,ptr,nodes,total,bar=m.groups()
            g={'index':int(ix),'cpu':int(cpu),'nodes':int(nodes),'operations_ms':float(total),'barriers_ms':float(bar),'rows':[]}
            graphs.append(g);current[(cpu,ptr)]=g
        elif m:=ROW.search(line):
            cpu,ptr,node,op,ms,bar,name,typ,shape,src=m.groups()
            if (cpu,ptr) in current:current[(cpu,ptr)]['rows'].append(dict(op=op,ms=float(ms),type=typ,shape=shape,src=src,name=name))
    for g in graphs:
        if g['nodes']<1000:continue
        by_type=defaultdict(float);families=defaultdict(float);shapes=defaultdict(float);counts=defaultdict(int)
        for row in g.pop('rows'):
            if row['op'] not in ['MUL_MAT','MUL_MAT_ID']:continue
            key=f"{row['op']} / {row['type']}"
            by_type[key]+=row['ms'];counts[key]+=1
            family=re.sub(r'blk\.\d+\.', 'blk.#.',row['src'] or row['name'])
            families[(row['op'],row['type'],family)]+=row['ms']
            shapes[(row['op'],row['type'],row['shape'])]+=row['ms']
        g.update(source=str(path),source_sha256=sha256(path.read_bytes()).hexdigest(),
                 matrix_type_ms=dict(sorted(by_type.items(),key=lambda x:-x[1])),matrix_type_count=dict(counts),
                 families=[dict(op=k[0],type=k[1],family=k[2],ms=v) for k,v in sorted(families.items(),key=lambda x:-x[1])],
                 shapes=[dict(op=k[0],type=k[1],shape=k[2],ms=v) for k,v in sorted(shapes.items(),key=lambda x:-x[1])])
        result['graphs'].append(g)
        print(filename,json.dumps(g['matrix_type_ms']))
        for row in g['shapes'][:8]:print(row)
assert len(result['graphs'])==2
(HERE/'matrix-bottleneck-audit.json').write_text(json.dumps(result,indent=2)+'\n')
