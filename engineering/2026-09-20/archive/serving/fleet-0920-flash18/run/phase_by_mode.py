#!/usr/bin/env python3
"""phase_by_mode.py <port> <log> <mode>... -- per spec-control mode: cycle time and its graph phases (GRAPH_PHASE log, armed per run)"""
import json, struct, sys, time, urllib.request, re, os, statistics as st
port, log = sys.argv[1], sys.argv[2]; modes = [int(x, 0) for x in sys.argv[3:]]
ARM = os.environ.get('PHASE_ARM', '/dev/shm/f18-graph-phase.arm')
P = 'Implement a thread-safe LRU cache in C++ with O(1) get and put. Explain the data structures, then give the complete code with comments.'
KW = json.loads(os.environ.get('REQ', '{"temperature": 0, "seed": 1}'))
def setmode(m):
    with open('/dev/shm/f18-spec.u32', 'r+b') as f: f.write(struct.pack('<I', m))
    time.sleep(0.2)
def run(n):
    body = dict(prompt=P, n_predict=n, cache_prompt=True, stream=False); body.update(KW)
    r = json.load(urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{port}/completion', json.dumps(body).encode(), {'Content-Type': 'application/json'}), timeout=3600))
    return r['timings']
def ts(s):  # "8.50.549.767" -> ms
    a = [int(x) for x in s.split('.')]; 
    while len(a) < 4: a.insert(0, 0)
    return ((a[0]*60 + a[1])*1000 + a[2]) + a[3]/1000.0
for rep in range(int(os.environ.get('REPS', '2'))):
    for m in modes:
        setmode(m); run(8)
        pos = os.path.getsize(log); open(ARM, 'w').close()
        t = run(int(os.environ.get('NPRED', '260')))
        try: os.unlink(ARM)
        except FileNotFoundError: pass
        time.sleep(0.3)
        rows = []
        with open(log, 'rb') as f:
            f.seek(pos)
            for l in f.read().decode(errors='replace').splitlines():
                mm = re.match(r'(\S+) W GRAPH_PHASE tokens=(\d+) nodes=(\d+) reused=(\d) apply=([\d.]+) prepare=([\d.]+) inputs=([\d.]+) compute=([\d.]+) total=([\d.]+)', l)
                if mm: rows.append((ts(mm[1]), int(mm[2]), int(mm[3]), int(mm[4]), float(mm[6]), float(mm[8]), float(mm[9])))
        big = max(r[2] for r in rows)
        tgt = [r for r in rows if r[2] == big][3:]
        cyc = [b[0] - a[0] for a, b in zip(tgt, tgt[1:])]
        dft = [r for r in rows if r[2] != big][6:]
        n_c = max(1, len(tgt))
        print(f"mode {m:2d}: tg {t['predicted_per_second']:.3f}  cycle {st.median(cyc):.2f} ms | target total {st.median(r[6] for r in tgt):.2f} (compute {st.median(r[5] for r in tgt):.2f}, reused {sum(r[3] for r in tgt)}/{len(tgt)}) | "
              f"draft graphs/cycle {len(dft)/n_c:.2f} total/cycle {sum(r[6] for r in dft)/n_c:.2f} (compute {sum(r[5] for r in dft)/n_c:.2f}, prepare {sum(r[4] for r in dft)/n_c:.2f}) | "
              f"outside graphs {st.median(cyc) - st.median(r[6] for r in tgt) - sum(r[6] for r in dft)/n_c:.2f}", flush=True)
