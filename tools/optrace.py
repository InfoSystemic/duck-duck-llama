#!/usr/bin/env python3
"""Parse CPU_OP_PROFILE lines from a llama-server log into a node-family time table (per socket graph).
usage: optrace.py <server.log> [min_nodes=1000] [layer=N]  -> op totals, top families, and one layer's node sequence."""
import re, sys, collections
HEADER = re.compile(r"CPU_OP_PROFILE index=(\d+) cpu=(\d+) graph=(\S+) nodes=(\d+) total=([\d.]+) ms first='([^']*)' last='([^']*)'")
NODE = re.compile(r"CPU_OP_PROFILE cpu=(\d+) graph=(\S+) node=(\d+) op=(\S+) time=([\d.]+) ms name='([^']*)' src0_type=(\S+) src0_ne=\[([^]]+)\] src0_name='([^']*)'")
path = sys.argv[1]; min_nodes = int(sys.argv[2]) if len(sys.argv) > 2 else 1000; layer = int(sys.argv[3]) if len(sys.argv) > 3 else 1
graphs = {}; order = []
for line in open(path, errors='replace'):
    m = HEADER.search(line)
    if m:
        key = (m.group(2), m.group(3)); graphs[key] = dict(cpu=int(m.group(2)), nodes=int(m.group(4)), total=float(m.group(5)), first=m.group(6), last=m.group(7), list=[]); order.append(key); continue
    m = NODE.search(line)
    if m:
        key = (m.group(1), m.group(2))
        if key in graphs: graphs[key]['list'].append(dict(op=m.group(4), ms=float(m.group(5)), name=m.group(6), t=m.group(7), ne=m.group(8), src=m.group(9)))
big = [graphs[k] for k in order if graphs[k]['nodes'] >= min_nodes]
print(f"{len(graphs)} graphs parsed; {len(big)} with >= {min_nodes} nodes")
if not big: sys.exit(0)
g = big[0]; nodes = g['list']
per_op = collections.Counter(); per_ms = collections.Counter(); fam = {}
for n in nodes:
    per_op[n['op']] += 1; per_ms[n['op']] += n['ms']
    name = re.sub(r'-\d+$', '-#', n['name']); name = re.sub(r'\.\d+\.', '.#.', name)
    f = fam.setdefault((n['op'], name), dict(n=0, ms=0.0, src=n['src'][:30], ne=n['ne'])); f['n'] += 1; f['ms'] += n['ms']
print(f"graph cpu {g['cpu']}: {g['nodes']} nodes, {g['total']:.1f} ms ({g['first']} .. {g['last']}); mean over big graphs {sum(b['total'] for b in big)/len(big):.1f} ms")
print("ops: " + "  ".join(f"{op}:{c} ({per_ms[op]:.1f}ms, {per_ms[op]/c*1000:.0f}us)" for op, c in per_op.most_common(24)))
print("--- top families by time")
for (op, name), f in sorted(fam.items(), key=lambda x: -x[1]['ms'])[:30]:
    print(f"  {op:16s} {name:40s} n={f['n']:4d} {f['ms']:7.2f} ms {f['ms']/f['n']*1000:6.0f} us  {f['src']} [{f['ne']}]")
def layer_of(n):
    m = re.search(r'-(\d+)$', n['name']); return int(m.group(1)) if m else None
try:
    s = next(i for i, n in enumerate(nodes) if layer_of(n) == layer); e = next(i for i, n in enumerate(nodes) if layer_of(n) == layer + 1)
    print(f"--- layer {layer}: nodes {s}..{e} ({e-s} nodes, {sum(n['ms'] for n in nodes[s:e]):.2f} ms)")
    for i in range(s, e):
        n = nodes[i]; print(f"{i:5d} {n['op']:15s} {n['name'][:40]:40s} {n['ms']*1000:6.0f}us  {n['t'][:6]:6s} [{n['ne'][:20]:20s}] {n['src'][:34]}")
except StopIteration:
    pass
