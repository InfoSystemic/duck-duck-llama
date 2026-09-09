#!/usr/bin/env python3
"""Summarize the last captured large graph profile in a benchmark server log."""
from collections import defaultdict
import json
import gzip
from pathlib import Path
import re
import statistics
import sys
p=Path(sys.argv[1])
log_text=(p/'server.log').read_text() if (p/'server.log').exists() else gzip.open(p/'server.log.gz', 'rt').read()
lines=[line for line in log_text.splitlines() if 'CPU_OP_PROFILE' in line]
(p/'cpu-profile.log').write_text('\n'.join(lines)+'\n')
header=re.compile(r'CPU_OP_PROFILE index=(\d+) cpu=(\d+) graph=(\S+) nodes=(\d+) total=([\d.]+) ms')
node=re.compile(r"CPU_OP_PROFILE cpu=(\d+) graph=(\S+) node=(\d+) op=(\S+) time=([\d.]+) ms name='(.*?)' src0_type=(\S+) src0_ne=\[(.*?)\] src0_name='(.*?)'")
graphs=[]; current={}
for line in lines:
 m=header.search(line)
 if m:
  g=dict(graph_index=int(m[1]),cpu=int(m[2]),graph=m[3],nodes=int(m[4]),total_ms=float(m[5]),operations=[])
  graphs.append(g); current[(m[2],m[3])]=g
 m=node.search(line)
 if m and (m[1],m[2]) in current:
  current[(m[1],m[2])]['operations'].append(dict(node=int(m[3]),op=m[4],ms=float(m[5]),name=m[6],source_type=m[7],source_shape=m[8],source_name=m[9]))
if not graphs:
 raise SystemExit('No profiles')
max_nodes=max(g['nodes'] for g in graphs)
# A draft head can exceed 100 nodes. Select the latest full-sized graph,
# keeping the original capture index because graph pointers are recycled.
g=max((g for g in graphs if g['nodes']>=0.75*max_nodes),key=lambda g:g['graph_index'])
sums=defaultdict(float); names=defaultdict(float)
for row in g['operations']:
 sums[(row['op'],row['source_type'])]+=row['ms']
 names[re.sub(r'blk\.\d+\.', 'blk.N.',row['source_name'])]+=row['ms']
out={k:v for k,v in g.items() if k!='operations'}
out['selection']='latest graph with at least 75% of maximum captured node count'
out['maximum_captured_nodes']=max_nodes
out['captured_graphs']=[{k:v for k,v in item.items() if k!='operations'} for item in graphs]
out['parsed_operation_count']=len(g['operations'])
out['matmul_source_shapes']=sorted({row['source_shape'] for row in g['operations'] if row['op'] in ('MUL_MAT', 'MUL_MAT_ID')})
out['hc_pre_source_shapes']=sorted({row['source_shape'] for row in g['operations'] if row['op']=='DSV4_HC_PRE'})
out['operation_ms']=[dict(op=k[0],source_type=k[1],ms=round(v,3)) for k,v in sorted(sums.items(),key=lambda kv:-kv[1])]
out['source_ms']=[dict(source_name=k,ms=round(v,3)) for k,v in sorted(names.items(),key=lambda kv:-kv[1])]
grouped=defaultdict(list)
for item in graphs:
 if item['nodes'] < 0.75*max_nodes:
  continue
 signature=tuple(sorted({row['source_shape'] for row in item['operations'] if row['op']=='DSV4_HC_PRE'}))
 grouped[signature].append(item)
distributions=[]
for signature, items in grouped.items():
 totals=[item['total_ms'] for item in items]
 by_op=[]
 for item in items:
  values=defaultdict(float)
  for row in item['operations']:
   values[(row['op'],row['source_type'])]+=row['ms']
  by_op.append(values)
 keys=set().union(*(values.keys() for values in by_op))
 operations=[dict(op=key[0],source_type=key[1],
                  median_ms=round(statistics.median(values[key] for values in by_op),3)) for key in keys]
 distributions.append(dict(hc_pre_source_shapes=list(signature),count=len(items),
     graph_indices=[item['graph_index'] for item in items],
     total_ms=dict(median=statistics.median(totals),min=min(totals),max=max(totals)),
     operation_medians=sorted(operations,key=lambda row:-row['median_ms'])))
out['large_graph_distributions']=distributions
out['distribution_note']='Individual CPU graph captures grouped by HC input shapes; these are not end-to-end step times.'
(p/'decode-profile-summary.json').write_text(json.dumps(out,indent=2))
print(json.dumps({k:v[:12] if isinstance(v,list) else v for k,v in out.items()},indent=2))
