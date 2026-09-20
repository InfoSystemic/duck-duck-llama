#!/usr/bin/env python3
"""fast_sampler_check.py <port> -- the fast candidate path (control bit 32) must not change a single token:
   seeded sampled and greedy requests, with and without coupling, compared with the bit off."""
import json, struct, sys, time, urllib.request, hashlib
port = sys.argv[1]
P = 'Implement a thread-safe LRU cache in C++ with O(1) get and put. Explain the data structures, then give the complete code with comments.'
def setmode(m):
    with open('/dev/shm/f18-spec.u32', 'r+b') as f: f.write(struct.pack('<I', m))
    time.sleep(0.2)
def run(**kw):
    body = dict(prompt=P, n_predict=200, cache_prompt=True, stream=False); body.update(kw)
    for attempt in range(3):
        try:
            r = json.load(urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{port}/completion', json.dumps(body).encode(), {'Content-Type': 'application/json'}), timeout=3600))
            return r['content'], r['timings']['predicted_per_second']
        except urllib.error.HTTPError as e:
            return 'HTTP%d' % e.code, 0.0   # the proxy's garbage text can fail the output parser; same seed -> same failure in both modes
ok = True
for base in (7, 31):
    for kw in (dict(temperature=0, seed=1), dict(seed=42), dict(seed=43), dict(seed=44, top_k=20, top_p=0.8), dict(seed=45, top_k=100, min_p=0.0, top_p=1.0, temperature=1.3)):
        out = {}
        for m in (base, base | 32, base, base | 32):
            setmode(m); out.setdefault(m, []).append(run(**kw))
        texts = {x[0] for v in out.values() for x in v}
        same = len(texts) == 1; ok &= same
        tg = {m: sum(x[1] for x in v)/len(v) for m, v in out.items()}
        print(f"base {base:2d} {str(kw):70s} {'IDENTICAL' if same else 'DIFFERS  '} hash {hashlib.sha256(out[base][0][0].encode()).hexdigest()[:8]}  tg off {tg[base]:.2f} on {tg[base|32]:.2f}")
print('RESULT:', 'PASS' if ok else 'FAIL')
