#!/usr/bin/env python3
"""opsum.py <optrace.log> [min_nodes] -- per-op totals for the LAST big graph in a CPU_OP_PROFILE trace"""
import re,sys,collections
lines=open(sys.argv[1]).read().splitlines(); minn=int(sys.argv[2]) if len(sys.argv)>2 else 500
graphs=[]; cur=None
for l in lines:
    m=re.search(r'CPU_OP_PROFILE index=(\d+) cpu=(\d+) graph=(\S+) nodes=(\d+) total=([\d.]+) ms barrier=([\d.]+)',l)
    if m:
        cur=dict(index=int(m[1]),cpu=int(m[2]),graph=m[3],nodes=int(m[4]),total=float(m[5]),barrier=float(m[6]),ops=[]); graphs.append(cur); continue
    m=re.search(r"CPU_OP_PROFILE cpu=(\d+) graph=(\S+) node=(\d+) op=(\S+) time=([\d.]+) ms bar=([\d.]+) ms name='([^']*)' src0_type=(\S+) src0_ne=\[([^\]]*)\]",l)
    if m and cur is not None and m[2]==cur['graph']:
        cur['ops'].append((m[4],float(m[5]),float(m[6]),m[7],m[8],m[9]))
big=[g for g in graphs if g['nodes']>=minn]
print(f"{len(graphs)} graphs, {len(big)} with >= {minn} nodes")
for g in big[-1:]:
    print(f"graph nodes={g['nodes']} cpu={g['cpu']} total={g['total']:.3f} barrier={g['barrier']:.3f}")
    agg=collections.defaultdict(lambda:[0,0.0,0.0])
    for op,t,b,name,ty,ne in g['ops']:
        key=op+':'+re.sub(r'-\d+$','',re.sub(r'node_\d+','node',name))
        a=agg[key]; a[0]+=1; a[1]+=t; a[2]+=b
    for k,(n,t,b) in sorted(agg.items(),key=lambda kv:-kv[1][1])[:int(sys.argv[3]) if len(sys.argv)>3 else 25]:
        print(f"  {k:48s} n={n:4d} time={t:8.3f} bar={b:7.3f}")
