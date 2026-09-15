#!/usr/bin/env python3
"""Sum the q4e engine's GRAPH_PHASE lines (LLAMA_GRAPH_PHASE_ARM_FILE) from a server log into per-MTP-cycle phase costs.
usage: phase.py <server.log> [bench3 json ...]  -> per-graph-class means (verify/draft), reuse rate, and if bench JSONs are
given, the wall per cycle vs the summed graph phases (the difference is sampling + server loop + state bookkeeping)."""
import re, sys, json, collections
L = re.compile(r"GRAPH_PHASE tokens=(\d+) nodes=(\d+) reused=(\d) apply=([\d.]+) prepare=([\d.]+) inputs=([\d.]+) compute=([\d.]+) total=([\d.]+) ms")
cls = collections.defaultdict(list)
for line in open(sys.argv[1], errors='replace'):
    m = L.search(line)
    if not m: continue
    tok, nodes, reused = int(m.group(1)), int(m.group(2)), int(m.group(3))
    k = 'verify' if nodes >= 1000 else ('draft' if nodes >= 100 else 'tiny')
    cls[k].append((tok, reused) + tuple(float(m.group(i)) for i in range(4, 9)))
for k in ('verify', 'draft', 'tiny'):
    rows = cls.get(k, [])
    if not rows: continue
    n = len(rows); avg = lambda i: sum(r[i] for r in rows) / n
    toks = collections.Counter(r[0] for r in rows)
    print(f"  {k:6s} n={n:5d} reused={sum(r[1] for r in rows)/n*100:5.1f}%  apply={avg(2):6.2f} prepare={avg(3):6.2f} inputs={avg(4):6.2f} compute={avg(5):7.2f} total={avg(6):7.2f} ms  tokens={dict(sorted(toks.items()))}")
    if k == 'verify':
        for t in sorted(toks):
            rr = [r for r in rows if r[0] == t]; m = len(rr)
            print(f"           tokens={t}: n={m:4d} reused={sum(r[1] for r in rr)/m*100:5.1f}% prepare={sum(r[3] for r in rr)/m:6.2f} compute={sum(r[5] for r in rr)/m:7.2f} total={sum(r[6] for r in rr)/m:7.2f}")
v = cls.get('verify', []); d = cls.get('draft', [])
if v:
    per_cycle = sum(r[6] for r in v) / len(v) + (sum(r[6] for r in d) / len(v) if d else 0)
    print(f"  graph phases per cycle (verify + drafts/verify): {per_cycle:.1f} ms  (drafts per verify {len(d)/len(v):.2f})")
    for f in sys.argv[2:]:
        try:
            t = json.load(open(f)).get('timings', {}); n = t['predicted_n']; acc = t.get('draft_n_accepted', 0) or 0
            cycles = n - acc; wall = t['predicted_ms'] / cycles
            print(f"  {f.split('/')[-1]}: wall/cycle={wall:.1f} ms  => outside-graph {wall - per_cycle:.1f} ms/cycle ({(wall - per_cycle)/wall*100:.0f}%)")
        except Exception as e: print("  ", f, e)
