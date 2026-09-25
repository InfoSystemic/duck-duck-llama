#!/usr/bin/env python3
# opsum.py <server.log> -- summarise GGML_CPU_OP_PROFILE records: per graph (one NUMA node's subgraph), then per op
# type and per tensor-name family, averaged over the profiled graphs of the dominant (largest) node count.
import re, sys, collections
recs, cur = [], None
for line in open(sys.argv[1], errors="replace"):
    m = re.search(r"CPU_OP_PROFILE index=(\d+) cpu=(-?\d+) graph=(\S+) nodes=(\d+) total=([\d.]+) ms", line)
    if m:
        cur = {"idx": int(m[1]), "cpu": int(m[2]), "nodes": int(m[4]), "total": float(m[5]), "ops": []}
        recs.append(cur); continue
    m = re.search(r"CPU_OP_PROFILE cpu=(-?\d+) graph=\S+ node=(\d+) op=(\S+) time=([\d.]+) ms name='([^']*)' src0_type=(\S+) src0_ne=\[([^\]]*)\]", line)
    if m and cur is not None:
        cur["ops"].append((m[3], float(m[4]), m[5], m[6], m[7]))
if not recs:
    sys.exit("no profile records")
bycount = collections.Counter(r["nodes"] for r in recs)
print("graphs by node count:", dict(bycount))
for n, _ in bycount.most_common():
    rs = [r for r in recs if r["nodes"] == n]
    print(f"\n== {len(rs)} graphs with {n} nodes, mean total {sum(r['total'] for r in rs)/len(rs):.2f} ms (cpus {sorted(set(r['cpu'] for r in rs))})")
    by_op, by_name = collections.defaultdict(float), collections.defaultdict(float)
    for r in rs:
        for op, t, name, st, ne in r["ops"]:
            by_op[op] += t / len(rs)
            fam = re.sub(r"-\d+", "-N", name)
            by_name[(op, fam, st)] += t / len(rs)
    tot = sum(by_op.values())
    for op, t in sorted(by_op.items(), key=lambda x: -x[1])[:12]:
        print(f"  {op:16s} {t:9.2f} ms  {100*t/tot:5.1f}%")
    print("  -- top tensor families")
    for (op, fam, st), t in sorted(by_name.items(), key=lambda x: -x[1])[:22]:
        print(f"  {op:14s} {st:8s} {fam[:44]:44s} {t:9.2f} ms  {100*t/tot:5.1f}%")
