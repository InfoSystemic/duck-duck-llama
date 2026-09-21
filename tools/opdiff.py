#!/usr/bin/env python3
"""opdiff.py <trace-a.log> <trace-b.log> [min_nodes] [top] -- where the time went between two CPU_OP_PROFILE traces.

Takes the LAST graph with >= min_nodes nodes in each trace (the verify graph on one device), sums time per op key
(op type + node name without digits) and per op type, and prints both tables sorted by the growth from a to b.
Same-device traces only: the profiler prints one graph per device and their node sets differ by the split.
"""
import re, sys, collections

def load(path, minn):
    graphs = []; cur = None
    for l in open(path, errors='replace'):
        m = re.search(r'CPU_OP_PROFILE index=(\d+) cpu=(\d+) graph=(\S+) nodes=(\d+) total=([\d.]+) ms barrier=([\d.]+)', l)
        if m:
            cur = dict(index=int(m[1]), cpu=int(m[2]), graph=m[3], nodes=int(m[4]), total=float(m[5]), barrier=float(m[6]), ops=[]); graphs.append(cur); continue
        m = re.search(r"CPU_OP_PROFILE cpu=(\d+) graph=(\S+) node=(\d+) op=(\S+) time=([\d.]+) ms bar=([\d.]+) ms name='([^']*)' src0_type=(\S+) src0_ne=\[([^\]]*)\]", l)
        if m and cur is not None and m[2] == cur['graph']:
            cur['ops'].append((m[4], float(m[5]), float(m[6]), m[7], m[8], m[9]))
    big = [g for g in graphs if g['nodes'] >= minn]
    if not big: sys.exit(f'{path}: no graph with >= {minn} nodes ({len(graphs)} graphs)')
    return big[-1]

def agg(g, by_key):
    out = collections.defaultdict(lambda: [0, 0.0, 0.0])
    for op, t, b, name, ty, ne in g['ops']:
        k = (op + ':' + re.sub(r'-\d+$', '', re.sub(r'node_\d+', 'node', re.sub(r'\d+', 'N', name)))) if by_key else op
        a = out[k]; a[0] += 1; a[1] += t; a[2] += b
    return out

def table(title, A, B, top):
    keys = set(A) | set(B)
    rows = sorted(keys, key=lambda k: -(B.get(k, [0, 0, 0])[1] - A.get(k, [0, 0, 0])[1]))
    print(f'\n{title}: {"key":50s} {"n_a":>5s} {"ms_a":>9s} {"n_b":>5s} {"ms_b":>9s} {"delta":>9s}')
    for k in rows[:top]:
        a = A.get(k, [0, 0.0, 0.0]); b = B.get(k, [0, 0.0, 0.0])
        print(f'  {k:50s} {a[0]:5d} {a[1]:9.3f} {b[0]:5d} {b[1]:9.3f} {b[1]-a[1]:+9.3f}')

def main():
    pa, pb = sys.argv[1], sys.argv[2]; minn = int(sys.argv[3]) if len(sys.argv) > 3 else 500; top = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    ga, gb = load(pa, minn), load(pb, minn)
    print(f"a: {pa} cpu={ga['cpu']} nodes={ga['nodes']} total={ga['total']:.2f} ms barrier={ga['barrier']:.2f}")
    print(f"b: {pb} cpu={gb['cpu']} nodes={gb['nodes']} total={gb['total']:.2f} ms barrier={gb['barrier']:.2f}")
    sa, sb = sum(t for _, t, *_ in ga['ops']), sum(t for _, t, *_ in gb['ops'])
    print(f'sum of node times a {sa:.2f} ms, b {sb:.2f} ms, growth {sb - sa:+.2f} ms')
    table('by op type', agg(ga, False), agg(gb, False), top)
    table('by op and name', agg(ga, True), agg(gb, True), top)

if __name__ == '__main__':
    main()
