#!/usr/bin/env python3
"""Fleet scoreboard, 2026-09-11. GB/s is RECOMPUTED from the raw IMC CSVs with the robust
longest-contiguous-run window (the TSVs carry the old, broken first-to-last window)."""
import glob, os, re, csv, statistics, sys
sys.path.insert(0,'/dev/shm/bwprobe')
from bwsum2 import series, decode_phase
CEIL=380.0
D=os.path.expanduser('~/InfoSystemic/AI-Server/serving/fleet-0911/results')
def gbs(tag):
    vals=[]
    for f in sorted(glob.glob(f'{D}/bw-{tag}-[123].csv')) or sorted(glob.glob(f'{D}/bw-{tag}.csv')):
        gb,dt=series(f)
        if not gb: continue
        a,z=decode_phase(gb,dt)
        w=gb[a:z+1]
        if w: vals.append(statistics.mean(w))
    return statistics.mean(vals) if vals else None
rows=[]
for f in sorted(glob.glob(D+'/*.tsv')):
    for r in csv.DictReader(open(f), delimiter='\t'):
        try: rows.append((r['tag'], float(r['tok_s'])))
        except (ValueError,KeyError,TypeError): pass
for f in [D+'/drive.out']:
    if not os.path.exists(f): continue
    for line in open(f):
        m=re.search(r'== (par\d+) C=(\d+): .*per-stream ([\d.]+)',line)
        if m: rows.append((m.group(1), float(m.group(3))*int(m.group(2))))
seen=set(); print(f"{'config':24s} {'tok/s':>8s} {'GB/s':>8s} {'% of 380':>9s}"); print('-'*54)
for n,t in rows:
    if n in seen: continue
    seen.add(n); g=gbs(n)
    print(f"{n:24s} {t:8.2f} {g if g else 0:8.1f} {(100*g/CEIL if g else 0):8.1f}%")
