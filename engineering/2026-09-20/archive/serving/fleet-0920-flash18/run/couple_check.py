#!/usr/bin/env python3
"""couple_check.py <port> -- functional checks of coupled sampling on a running server with the spec control file.
  modes (control word /dev/shm/f18-spec.u32): 7 = production draft loop, 15 = + verifier Gumbel pick, 31 = + drafter coupled.
  1. greedy request: text identical in modes 7 and 31 (coupling must not touch temp=0)
  2. seeded sampled request: text identical in modes 15 and 31 (the drafter cannot change the verifier's sample)
  3. unseeded sampled requests: acceptance per mode
"""
import json, struct, sys, time, urllib.request, hashlib
port = sys.argv[1]
src = open('/home/user/InfoSystemic/AI-Server/serving/fleet-0912-ctx/README.md').read()
PROMPTS = {
  'short': 'Implement a thread-safe LRU cache in C++ with O(1) get and put. Explain the data structures, then give the complete code with comments.',
  'mid':   src[:12000] + '\n\nSummarise the key findings above in five bullet points.\n',
}
def setmode(m):
    with open('/dev/shm/f18-spec.u32', 'r+b') as f: f.write(struct.pack('<I', m))
    time.sleep(0.2)
def run(prompt, n=160, **kw):
    body = dict(prompt=prompt, n_predict=n, cache_prompt=True, stream=False); body.update(kw)
    r = json.load(urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{port}/completion', json.dumps(body).encode(), {'Content-Type': 'application/json'}), timeout=3600))
    t = r['timings']
    return r['content'], t.get('draft_n'), t.get('draft_n_accepted'), t.get('predicted_per_second')
h = lambda s: hashlib.sha256(s.encode()).hexdigest()[:10]
ok = True
for name, p in PROMPTS.items():
    print(f'== {name}')
    res = {}
    for m in (7, 31):
        setmode(m); res[m] = run(p, temperature=0, seed=42)
        print(f'  greedy  mode {m:2d}: hash {h(res[m][0])} draft {res[m][2]}/{res[m][1]} tg {res[m][3]:.2f}')
    same = res[7][0] == res[31][0]; ok &= same
    print('  greedy 7 vs 31:', 'IDENTICAL' if same else 'DIFFERS')
    for seed in (42, 7, 1234):
        res = {}
        for m in (15, 31, 15, 31):
            setmode(m); r = run(p, seed=seed)
            res.setdefault(m, []).append(r)
            print(f'  sampled seed {seed} mode {m:2d}: hash {h(r[0])} draft {r[2]}/{r[1]} = {r[2]/max(1,r[1]):.3f} tg {r[3]:.2f}')
        texts = {x[0] for v in res.values() for x in v}
        same = len(texts) == 1; ok &= same
        print(f'  seed {seed}: verifier-only vs coupled drafter:', 'IDENTICAL' if same else 'DIFFERS')
    for m in (7, 15, 31):
        setmode(m); acc = dr = 0; texts = set()
        for k in range(4):
            r = run(p); dr += r[1]; acc += r[2]; texts.add(r[0])
        print(f'  unseeded mode {m:2d}: acceptance {acc}/{dr} = {acc/max(1,dr):.3f}, {len(texts)} distinct texts of 4')
print('RESULT:', 'PASS' if ok else 'FAIL')
