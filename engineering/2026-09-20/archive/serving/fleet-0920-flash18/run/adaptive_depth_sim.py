#!/usr/bin/env python3
"""adaptive_depth_sim.py <couple log> -- what would variable-length verification buy? Offline, from the window w7 log.

Per cycle the log gives the drafter's confidence in each draft token (softmax share of the chosen token among its kept top-10) and,
through the verifier's picks, whether each draft was accepted. Policy: drop the 2nd draft token when the confidence of the 1st (or the
product of both) is below a threshold. Cost model from measured phases (revision e, ms): fixed F, per verified row R, per draft pass D.
"""
import sys, math
F, R, D = float(sys.argv[2]) if len(sys.argv) > 2 else 66.0, float(sys.argv[3]) if len(sys.argv) > 3 else 16.5, float(sys.argv[4]) if len(sys.argv) > 4 else 4.6
V, cycles = {}, []
cur = None
for line in open(sys.argv[1], errors='replace'):
    f = line.split()
    try:
        if f[0] == 'V':
            if len(f) != 6 + 2*int(f[5]): continue
            V[(int(f[1]), int(f[3]))] = int(f[4])
        elif f[0] == 'D':
            k = int(f[6])
            if len(f) != 11 + 2*k: continue
            salt, step, n, tok, n_keep, temp = int(f[1]), int(f[2]), int(f[3]), int(f[4]), int(f[5]), float(f[7])
            top = [(int(f[11 + 2*i]), float(f[12 + 2*i])) for i in range(k)]
            if not all(math.isfinite(l) for _, l in top): continue
            z = sum(math.exp((l - top[0][1])/temp) for _, l in top[:n_keep])
            q = next((math.exp((l - top[0][1])/temp)/z for t, l in top[:n_keep] if t == tok), 0.0)
            if step == 0:
                cur = dict(salt=salt, n=n, tok0=tok, q0=q); cycles.append(cur)
            elif cur is not None and cur['salt'] == salt and cur['n'] == n - 1:
                cur.update(tok1=tok, q1=q)
    except (ValueError, IndexError):
        continue
rows = []
for c in cycles:
    if 'tok1' not in c or (c['salt'], c['n']) not in V: continue
    a0 = V[(c['salt'], c['n'])] == c['tok0']
    a1 = a0 and V.get((c['salt'], c['n'] + 1)) == c['tok1']
    rows.append((c['q0'], c['q1'], a0, a1))
N = len(rows)
base_tok = sum(1 + a0 + a1 for _, _, a0, a1 in rows)/N; base_ms = F + 3*R + 2*D
print(f"{N} cycles; depth 2 always: {base_tok:.3f} tokens per cycle, {base_ms:.1f} ms model cycle -> {1000*base_tok/base_ms:.2f} tok/s (model); acceptance step0 {sum(r[2] for r in rows)/N:.3f} step1|0 {sum(r[3] for r in rows)/max(1,sum(r[2] for r in rows)):.3f}")
print("confidence of draft 1 vs acceptance:")
for lo, hi in ((0, .3), (.3, .5), (.5, .7), (.7, .9), (.9, .99), (.99, 1.01)):
    g = [r for r in rows if lo <= r[0] < hi]
    if g: print(f"   q0 in [{lo:.2f},{hi:.2f}): {100*len(g)/N:5.1f}% of cycles, draft 1 accepted {sum(r[2] for r in g)/len(g):.3f}, both accepted {sum(r[3] for r in g)/len(g):.3f}")
print("policy: verify only draft 1 (2 rows, 1 draft pass... the 2nd pass is needed to KNOW q1, so gate on q0 only) when q0 < thr")
best = (1000*base_tok/base_ms, 'always 2')
for thr in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
    tok = ms = 0.0
    for q0, q1, a0, a1 in rows:
        if q0 < thr: tok += 1 + a0; ms += F + 2*R + 1*D
        else:        tok += 1 + a0 + a1; ms += F + 3*R + 2*D
    r = 1000*tok/ms; print(f"   thr {thr:.1f}: {tok/N:.3f} tokens per cycle, {ms/N:6.1f} ms -> {r:.2f} tok/s ({100*(r/(1000*base_tok/base_ms) - 1):+.2f}%)")
    if r > best[0]: best = (r, f'thr {thr}')
print(f"zero drafts when q0 < thr (1 row, 1 draft pass to know q0):")
for thr in (0.2, 0.3, 0.4):
    tok = ms = 0.0
    for q0, q1, a0, a1 in rows:
        if q0 < thr: tok += 1; ms += F + 1*R + 1*D
        else:        tok += 1 + a0 + a1; ms += F + 3*R + 2*D
    r = 1000*tok/ms; print(f"   thr {thr:.1f}: {tok/N:.3f} tokens per cycle, {ms/N:6.1f} ms -> {r:.2f} tok/s ({100*(r/(1000*base_tok/base_ms) - 1):+.2f}%)")
print('best gated policy:', best[1], f'{best[0]:.2f} tok/s (model)')
