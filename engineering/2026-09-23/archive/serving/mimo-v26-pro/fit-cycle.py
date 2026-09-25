#!/usr/bin/env python3
"""fit-cycle.py -- turn the measured draft-length curve into a byte model, then project what attention requant buys.

The curve gives (draft length, ms per verify cycle, tokens per cycle). The byte model says a cycle reads

    bytes(rows) = FIXED + EXPERTS_PER_ROW_SET * f(rows),   f(rows) = 384 * (1 - (1 - 8/384)^rows) / 8

because attention, the router, the output head and the one DFlash forward are read once per cycle regardless of the row
count, while each extra verify row lights up ~8 more of the 384 experts per layer. Fitting FIXED and the expert term to
the measured cycle times does two useful things at once:

  * it CHECKS the model -- if a single (FIXED, EXPERTS) pair reproduces every measured ms/cycle, the accounting is right
  * it makes the requant projection arithmetic instead of a guess: Q4_K halves the attention share of FIXED, so the new
    cycle time is (bytes - saving) / bytes of the old one, and the rate scales the same way at unchanged acceptance

Usage: fit-cycle.py [results/spec-detail-dflash-n*.txt ...]
"""
import glob, re, sys

ATTN_Q8 = 19.3          # GiB of Q8_0 attention weights, read once per verify cycle
ATTN_Q4 = 10.2          # the same tensors at Q4_K
EXPERTS_PER_TOKEN = 10.3  # GiB for the 8-of-384 experts one row activates, summed over 70 layers


def expert_factor(rows):
    return 384.0 * (1.0 - (1.0 - 8.0 / 384.0) ** rows) / 8.0


def read(path):
    txt = open(path).read()
    def grab(pat, cast=float):
        # MULTILINE matters: every pattern here is anchored to the start of a LINE, and without it "^" only matches at
        # the start of the whole file, so every field but the first silently comes back None.
        m = re.search(pat, txt, re.M)
        return cast(m.group(1)) if m else None
    n = int(re.search(r"-n(\d+)\.txt$", path).group(1))
    return {
        "n": n,
        "rows": n + 1,
        "tok_s": grab(r"^decode\s+([\d.]+)", ),
        "ms": grab(r"^ms/cycle\s+([\d.]+)"),
        "tpc": grab(r"^tokens/cycle\s+([\d.]+)"),
        "acc": grab(r"^draft acceptance \d+/\d+ = ([\d.]+)"),
    }


paths = sys.argv[1:] or sorted(glob.glob("results/spec-detail-dflash-n*.txt"))
rows = [read(p) for p in paths]
rows = [r for r in rows if r["ms"] and r["tok_s"]]
if len(rows) < 2:
    sys.exit("need at least two draft lengths; run the sweep first")
rows.sort(key=lambda r: r["n"])

# least squares on ms = A + B * f(rows): A is everything read once per cycle, B the per-expert-set cost
xs = [expert_factor(r["rows"]) for r in rows]
ys = [r["ms"] for r in rows]
n = len(xs)
mx, my = sum(xs) / n, sum(ys) / n
den = sum((x - mx) ** 2 for x in xs)
B = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0
A = my - B * mx

print(f"{'n':>3} {'rows':>5} {'tok/s':>7} {'tok/cyc':>8} {'acc':>6} {'ms/cyc':>8} {'fit':>8} {'err':>7}")
for r, x in zip(rows, xs):
    fit = A + B * x
    print(f"{r['n']:>3} {r['rows']:>5} {r['tok_s']:>7.2f} {r['tpc']:>8.3f} "
          f"{(r['acc'] if r['acc'] is not None else 0):>6.3f} {r['ms']:>8.1f} {fit:>8.1f} {100*(fit-r['ms'])/r['ms']:>6.1f}%")

print(f"\nfixed per cycle   {A:8.1f} ms   (attention + router + head + one draft forward)")
print(f"per expert set    {B:8.2f} ms   ({EXPERTS_PER_TOKEN:.1f} GiB of experts costs this much)")
gbps = EXPERTS_PER_TOKEN * 1.0737 / (B / 1000) if B > 0 else 0
print(f"implied expert bandwidth {gbps:.0f} GB/s = {100*gbps/381.6:.0f}% of this host's measured 381.6")

best = max(rows, key=lambda r: r["tok_s"])
print(f"\nbest measured: n={best['n']} at {best['tok_s']:.2f} tok/s "
      f"({best['tpc']:.3f} tokens/cycle, {best['ms']:.1f} ms/cycle)")

# Q4_K attention removes ATTN_Q8 - ATTN_Q4 GiB from the fixed part. Convert that to ms with the fitted expert rate,
# which is the only bandwidth number we have that came from these measurements rather than from a different model.
if B > 0:
    ms_per_gib = B / EXPERTS_PER_TOKEN
    saving_ms = (ATTN_Q8 - ATTN_Q4) * ms_per_gib
    for r in rows:
        proj = r["tpc"] / ((r["ms"] - saving_ms) / 1000) if r["ms"] > saving_ms else 0
        if r is best:
            print(f"projected with Q4_K attention at n={r['n']}: {proj:.2f} tok/s "
                  f"(-{saving_ms:.0f} ms of {r['ms']:.0f}, acceptance unchanged)")
    print("\nThat projection assumes the Q4_K kernel unpacks for free. On GLM-5.3-Flash the same swap measured SLOWER")
    print("because unpacking cost more than the bandwidth it saved -- so treat it as an upper bound and measure.")
