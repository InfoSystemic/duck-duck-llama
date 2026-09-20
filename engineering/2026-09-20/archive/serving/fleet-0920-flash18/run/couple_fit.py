#!/usr/bin/env python3
"""couple_fit.py <couple log> -- replay drafter policies offline against the verifier's actual draws.

The log (GGML_F18_COUPLE_LOG) has one "V" line per verifier pick (salt, stream, n, token, candidates after the chain) and one "D"
line per coupled draft pick (salt, step, n, token, n_keep, n, temp, top_p, min_p, top_k, the drafter's raw top-10).
Both sides add the same noise G(salt, n, token), so for any drafter setting the pick can be recomputed here and compared with the
token the verifier really emitted at (salt, n). Step-0 rows are always on-policy; step-1 rows only have a verifier pick when the
deployed drafter's step 0 was accepted, which is exactly the prefix the replay needs.
"""
import sys, math, collections
M = (1 << 64) - 1
def mix64(x):
    x = (x + 0x9E3779B97F4A7C15) & M
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & M
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & M
    return x ^ (x >> 31)
def gumbel(salt, n, tok):
    h = mix64((mix64((mix64(salt) + n) & M) + (tok & 0xFFFFFFFF)) & M)
    u = ((h >> 11) + 0.5) / 9007199254740992.0
    return -math.log(-math.log(u))

V, D = {}, []
TORN = [0]   # the two translation units log through separate FILE buffers, so a line that spans a flush can be torn: drop it
for line in open(sys.argv[1], errors='replace'):
    f = line.split()
    if not f: continue
    try:
        if f[0] == 'V':
            salt, stream, n, tok, size = int(f[1]), int(f[2]), int(f[3]), int(f[4]), int(f[5])
            if len(f) != 6 + 2*size or size < 1: TORN[0] += 1; continue
            cand = [(int(f[6 + 2*i]), float(f[7 + 2*i])) for i in range(size)]
            V[(salt, n)] = (tok, cand, stream)          # a grammar resample overwrites the first draw: the last line is what was emitted
        elif f[0] == 'D':
            salt, step, n, tok, n_keep, k = int(f[1]), int(f[2]), int(f[3]), int(f[4]), int(f[5]), int(f[6])
            temp, top_p, min_p, top_k = float(f[7]), float(f[8]), float(f[9]), int(f[10])
            if len(f) != 11 + 2*k or not (1 <= k <= 10) or step not in (0, 1): TORN[0] += 1; continue
            top = [(int(f[11 + 2*i]), float(f[12 + 2*i])) for i in range(k)]
            if not all(math.isfinite(l) for _, l in top) or any(top[i][1] < top[i + 1][1] for i in range(len(top) - 1)):
                odd = globals().setdefault('ODD', []); odd.append(line[:160]); continue
            D.append(dict(salt=salt, step=step, n=n, tok=tok, n_keep=n_keep, temp=temp, top_p=top_p, min_p=min_p, top_k=top_k, top=top))
    except (ValueError, IndexError):
        TORN[0] += 1; continue

def keep_count(top, top_k, top_p, min_p):
    """the deployed mimic of the verifier's truncation, on the untempered top-10"""
    pr = [math.exp(l - top[0][1]) for _, l in top]; s = sum(pr); n_keep = len(top)
    if top_k > 0: n_keep = min(n_keep, top_k)
    if top_p < 1.0:
        cum = 0.0
        for j in range(n_keep):
            cum += pr[j]/s
            if cum >= top_p: n_keep = j + 1; break
    if min_p > 0.0:
        for j in range(1, n_keep):
            if pr[j] < min_p*pr[0]: n_keep = j; break
    return n_keep

def pick(d, tau=1.0, top_p=None, min_p=None, greedy=False, bias_top=0.0):
    top = d['top']
    if greedy: return top[0][0]
    nk = keep_count(top, d['top_k'], d['top_p'] if top_p is None else top_p, d['min_p'] if min_p is None else min_p)
    inv = 1.0/(d['temp']*tau); best, bj = -1e300, 0
    for j in range(nk):
        v = (top[j][1] - top[0][1])*inv + gumbel(d['salt'], d['n'], top[j][0]) + (bias_top if j == 0 else 0.0)
        if v > best: best, bj = v, j
    return top[bj][0]

# a step-1 row is on-policy only if the step-0 draft of the SAME cycle was the token the verifier emitted; otherwise the verifier's
# pick at that position belongs to the next cycle and to a different prefix
last0 = {}
for d in D:
    if d['step'] == 0:
        last0[d['salt']] = d; d['on_policy'] = True
    else:
        m = last0.get(d['salt'])
        d['on_policy'] = bool(m and m['n'] == d['n'] - 1 and (m['salt'], m['n']) in V and V[(m['salt'], m['n'])][0] == m['tok'])
rows = [d for d in D if d['on_policy'] and (d['salt'], d['n']) in V]
print(TORN[0], 'torn or malformed lines dropped')
if globals().get('ODD'): print(len(ODD), 'draft rows with non-finite or unsorted logits skipped, e.g.', ODD[0])
print(f"{len(V)} verifier picks, {len(D)} draft picks, {len(rows)} joined ({sum(1 for d in rows if d['step']==0)} at step 0, {sum(1 for d in rows if d['step']==1)} at step 1)")
bad = sum(1 for d in D if pick(d) != d['tok'])
print(f"replay self-check: deployed policy reproduces the logged draft token in {len(D) - bad}/{len(D)} rows")
resampled = sum(1 for v in V.values() if v[2] != 0)
print(f"verifier picks that came from the grammar resample stream: {resampled}")

def report(name, fn):
    out = []
    for step in (0, 1):
        r = [d for d in rows if d['step'] == step]
        out.append(sum(1 for d in r if fn(d) == V[(d['salt'], d['n'])][0]) / max(1, len(r)))
    a1, a2 = out
    print(f"  {name:44s} step0 {a1:.4f}  step1 {a2:.4f}  tokens/cycle {1 + a1 + a1*a2:.4f}")
    return 1 + a1 + a1*a2

print("\ncoverage and baselines")
for step in (0, 1):
    r = [d for d in rows if d['step'] == step]
    cov = sum(1 for d in r if V[(d['salt'], d['n'])][0] in [t for t, _ in d['top']]) / max(1, len(r))
    cov1 = sum(1 for d in r if V[(d['salt'], d['n'])][0] == d['top'][0][0]) / max(1, len(r))
    single = sum(1 for d in r if len(V[(d['salt'], d['n'])][1]) == 1) / max(1, len(r))
    print(f"  step {step}: verifier token inside the drafter's top-10 {cov:.4f}; equals the drafter's top-1 {cov1:.4f}; verifier had a single candidate {single:.4f}  (n={len(r)})")
base = report('deployed: coupled, tau 1.0, mimic truncation', lambda d: pick(d))
report('greedy drafter', lambda d: pick(d, greedy=True))
print("\ndrafter temperature (relative to the verifier's)")
best = (base, 'tau 1.0')
for tau in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0):
    t = report(f'coupled, tau {tau}', lambda d, tau=tau: pick(d, tau=tau))
    if t > best[0]: best = (t, f'tau {tau}')
print("\ntruncation at the drafter")
for name, kw in (('no top_p, no min_p', dict(top_p=1.0, min_p=0.0)), ('min_p 0.05 only', dict(top_p=1.0)), ('min_p 0.02', dict(min_p=0.02)), ('min_p 0.1', dict(min_p=0.1)),
                 ('min_p 0.2', dict(min_p=0.2)), ('top_p 0.9', dict(top_p=0.9)), ('top_p 0.99', dict(top_p=0.99))):
    t = report(f'coupled, tau 1.0, {name}', lambda d, kw=kw: pick(d, **kw))
    if t > best[0]: best = (t, name)
print("\nbias toward the drafter's own top-1 (in logit units)")
for b in (0.25, 0.5, 1.0, 1.5, 2.0):
    t = report(f'coupled, tau 1.0, top-1 bias {b}', lambda d, b=b: pick(d, bias_top=b))
    if t > best[0]: best = (t, f'top-1 bias {b}')
print(f"\nbest: {best[1]} at {best[0]:.4f} tokens/cycle against {base:.4f} deployed ({100*(best[0]/base - 1):+.2f}%)")
