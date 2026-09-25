#!/usr/bin/env python3
"""Fit a speculative decoder's verify-cycle time to a byte model: what is read once per cycle, and what grows with rows.

For a mixture-of-experts target, one verify cycle reads everything outside the routed experts once (attention, router,
output head, one draft pass), while each verified row activates `used` of `experts` experts per layer, so r rows touch

    f(r) = experts * (1 - (1 - used/experts) ** r) / used        expert sets (distinct ones, in expectation)

and the model is  ms_per_cycle = A + B * f(rows),  rows = draft length + 1. If one (A, B) pair reproduces every measured
draft length, the accounting is right; B then prices one expert set, and --expert-gib turns it into a sustained bandwidth.
On MiMo-V2.6-Pro (384 experts, 8 used) this fit 99.2 + 36.83 f(rows) to 0.7% over draft lengths 2-7, and the expert term
ran at 79% of the measured memory wall, which says only tokens per cycle can still move the rate.

Input: tools/spec_detail.py outputs saved as <anything>-n<N>.txt, one per draft length N.
"""
import argparse, re, sys

GIB = 1.073741824   # GB per GiB


def expert_sets(rows, experts, used):
    return experts * (1.0 - (1.0 - used / experts) ** rows) / used


def read(path):
    text = open(path).read()

    def grab(pattern):
        m = re.search(pattern, text, re.M)   # every field is anchored at the start of a LINE
        return float(m.group(1)) if m else None

    m = re.search(r"-n(\d+)\.txt$", path)
    if not m:
        sys.exit(f"{path}: name must end in -n<draft length>.txt")
    n = int(m.group(1))
    return {"n": n, "rows": n + 1, "tok_s": grab(r"^decode\s+([\d.]+)"), "ms": grab(r"^ms/cycle\s+([\d.]+)"),
            "tpc": grab(r"^tokens/cycle\s+([\d.]+)"), "acc": grab(r"^draft acceptance \d+/\d+ = ([\d.]+)")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="spec_detail.py outputs named *-n<N>.txt")
    ap.add_argument("--experts", type=int, required=True, help="routed experts per layer")
    ap.add_argument("--used", type=int, required=True, help="experts activated per token")
    ap.add_argument("--expert-gib", type=float, help="GiB of expert weights one token activates, summed over layers")
    ap.add_argument("--ceiling-gbs", type=float, help="measured aggregate read bandwidth, GB/s")
    ap.add_argument("--fixed-saving-gib", type=float,
                    help="project the rate if this many GiB were removed from the once-per-cycle reads")
    args = ap.parse_args()

    rows = sorted((r for r in map(read, args.files) if r["ms"] and r["tok_s"]), key=lambda r: r["n"])
    if len(rows) < 2:
        sys.exit("need at least two draft lengths")
    xs = [expert_sets(r["rows"], args.experts, args.used) for r in rows]
    ys = [r["ms"] for r in rows]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0
    a = my - b * mx

    print(f"{'n':>3} {'rows':>5} {'tok/s':>7} {'tok/cyc':>8} {'acc':>6} {'ms/cyc':>8} {'fit':>8} {'err':>7}")
    for r, x in zip(rows, xs):
        fit = a + b * x
        acc = r["acc"] if r["acc"] is not None else 0.0
        print(f"{r['n']:>3} {r['rows']:>5} {r['tok_s']:>7.2f} {(r['tpc'] or 0):>8.3f} {acc:>6.3f} {r['ms']:>8.1f} "
              f"{fit:>8.1f} {100 * (fit - r['ms']) / r['ms']:>6.1f}%")
    print(f"\nread once per cycle  {a:8.1f} ms")
    print(f"per expert set       {b:8.2f} ms")
    if args.expert_gib and b > 0:
        gbps = args.expert_gib * GIB / (b / 1000)
        share = f" = {100 * gbps / args.ceiling_gbs:.0f}% of {args.ceiling_gbs:.1f}" if args.ceiling_gbs else ""
        print(f"expert bandwidth     {gbps:8.0f} GB/s{share}")
        if args.fixed_saving_gib:
            saving_ms = args.fixed_saving_gib * b / args.expert_gib   # priced at the fitted expert rate
            best = max(rows, key=lambda r: r["tok_s"])
            if best["tpc"] and best["ms"] > saving_ms:
                proj = best["tpc"] / ((best["ms"] - saving_ms) / 1000)
                print(f"projection at n={best['n']}: {proj:.2f} tok/s (-{saving_ms:.0f} ms of {best['ms']:.0f}, "
                      f"acceptance unchanged; an upper bound: a cheaper format can cost more to unpack than it saves)")


if __name__ == "__main__":
    main()
