#!/usr/bin/env python3
"""phase_timeline.py <port> <log> <mode>... -- per mode: median compute of each graph in the cycle and the gaps between graphs"""
import json, struct, sys, time, urllib.request, re, os, statistics as st
port, log = sys.argv[1], sys.argv[2]; modes = [int(x, 0) for x in sys.argv[3:]]
ARM = os.environ.get('PHASE_ARM', '/dev/shm/f18-graph-phase.arm')
P = 'Implement a thread-safe LRU cache in C++ with O(1) get and put. Explain the data structures, then give the complete code with comments.'
_src = open('/home/user/InfoSystemic/AI-Server/serving/fleet-0912-ctx/README.md').read()
P = {'short': P, 'mid': _src[:12000] + '\n\nSummarise the key findings above in five bullet points.\n', 'long': _src[:40000] + '\n\nSummarise the key findings above in five bullet points.\n'}[os.environ.get('PROMPT_KIND', 'short')]
KW = json.loads(os.environ.get('REQ', '{"temperature": 0, "seed": 1}'))
def setmode(m):
    with open('/dev/shm/f18-spec.u32', 'r+b') as f: f.write(struct.pack('<I', m))
    time.sleep(0.2)
def run(n):
    body = dict(prompt=P, n_predict=n, cache_prompt=True, stream=False); body.update(KW)
    return json.load(urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{port}/completion', json.dumps(body).encode(), {'Content-Type': 'application/json'}), timeout=3600))['timings']
def ts(s):
    a = [int(x) for x in s.split('.')]
    while len(a) < 4: a.insert(0, 0)
    return ((a[0]*60 + a[1])*1000 + a[2]) + a[3]/1000.0
for m in modes:
    setmode(m); run(8)
    pos = os.path.getsize(log); open(ARM, 'w').close()
    t = run(int(os.environ.get('NPRED', '260')))
    os.unlink(ARM); time.sleep(0.3)
    rows = []
    with open(log, 'rb') as f:
        f.seek(pos)
        for l in f.read().decode(errors='replace').splitlines():
            mm = re.match(r'(\S+) W GRAPH_PHASE tokens=(\d+) nodes=(\d+) reused=(\d) apply=([\d.]+) prepare=([\d.]+) inputs=([\d.]+) compute=([\d.]+) total=([\d.]+)', l)
            if mm: rows.append(dict(end=ts(mm[1]), tokens=int(mm[2]), nodes=int(mm[3]), compute=float(mm[8]), total=float(mm[9])))
    big = max(r['nodes'] for r in rows)
    # cycles: target graph followed by the draft graphs until the next target graph
    cycles = []; cur = None
    for r in rows:
        if r['nodes'] == big:
            if cur: cycles.append(cur)
            cur = [r]
        elif cur: cur.append(r)
    cycles = [c for c in cycles[3:] if len(c) == 3]
    med = lambda xs: st.median(xs)
    # log line is written at the END of the graph: start = end - total
    print(f"mode {m:2d}: tg {t['predicted_per_second']:.3f} | target {med([c[0]['compute'] for c in cycles]):.2f} | gap->draft1 {med([c[1]['end']-c[1]['total']-c[0]['end'] for c in cycles]):.2f} | draft1 {med([c[1]['compute'] for c in cycles]):.2f} (rows {cycles[0][1]['tokens']}) | gap->draft2 {med([c[2]['end']-c[2]['total']-c[1]['end'] for c in cycles]):.2f} | draft2 {med([c[2]['compute'] for c in cycles]):.2f} | n={len(cycles)}", flush=True)
