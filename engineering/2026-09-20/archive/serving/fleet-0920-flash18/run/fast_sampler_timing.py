#!/usr/bin/env python3
"""fast_sampler_timing.py <port> <log> -- per-mode decode rate and sampler timers (deltas of the cumulative trace) on a running server"""
import json, struct, sys, time, urllib.request, re
port, log = sys.argv[1], sys.argv[2]
P = 'Implement a thread-safe LRU cache in C++ with O(1) get and put. Explain the data structures, then give the complete code with comments.'
def setmode(m):
    with open('/dev/shm/f18-spec.u32', 'r+b') as f: f.write(struct.pack('<I', m))
    time.sleep(0.2)
def run(seed):
    body = dict(prompt=P, n_predict=520, cache_prompt=True, stream=False, temperature=0, seed=seed)
    r = json.load(urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{port}/completion', json.dumps(body).encode(), {'Content-Type': 'application/json'}), timeout=3600))
    t = r['timings']; return t['predicted_per_second'], t['predicted_n'], t['draft_n'], t['draft_n_accepted']
def trace():
    last = None
    for l in open(log, errors='replace'):
        m = re.search(r'F18_SAMPLER_TRACE samples=(\d+) per-sample ms: set_logits=([\d.]+) chain=([\d.]+) \| clones=(\d+) per-clone ms=([\d.]+)', l)
        if m: last = (int(m[1]), float(m[2]), float(m[3]), int(m[4]), float(m[5]))
    return last
res = {}
for rep in range(3):
    for m in (7, 39, 39, 7):
        setmode(m); t0 = trace(); r = run(1); t1 = trace()
        d = None
        if t0 and t1 and t1[0] > t0[0]:
            n = t1[0] - t0[0]; nc = max(1, t1[3] - t0[3])
            d = ((t1[0]*t1[1] - t0[0]*t0[1])/n, (t1[0]*t1[2] - t0[0]*t0[2])/n, (t1[3]*t1[4] - t0[3]*t0[4])/nc)
        res.setdefault(m, []).append((r[0], d))
        print(f"mode {m:2d}: tg {r[0]:.3f} tok/s n={r[1]} draft {r[3]}/{r[2]}  timers(set_logits, chain, clone) = {d and tuple(round(x,3) for x in d)}", flush=True)
for m, v in res.items():
    print(f"mode {m:2d}: mean tg {sum(x[0] for x in v)/len(v):.3f}")
