#!/usr/bin/env python3
"""A speculative decode rate split into the two things that set it: cycles per second and tokens per cycle.

rate = (verify cycles per second) x (tokens emitted per cycle). The first is bandwidth and kernels; the second belongs to the
drafter. Reading them apart says whether a disappointing rate is a bytes problem or an acceptance problem, which have
completely different fixes. Save one output per draft length as <name>-n<N>.txt and tools/fit_cycle.py turns the set into a
cycle-cost model.

Reports, from one greedy request's server-side timings: decode rate, verify cycles, tokens per cycle, ms per cycle and draft
acceptance; with --gib-per-cycle, the implied sustained bandwidth, and with --ceiling-gbs, that as a share of the machine's
measured read bandwidth (tools/membw.c).
"""
import argparse, json, urllib.request

PROMPT = ("Explain in about 150 words why a speculative decoder's throughput is the product of its verify-cycle rate and the "
          "number of tokens accepted per cycle, and which of the two a memory-bandwidth limit constrains.")
GIB = 1.073741824   # GB per GiB


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--n-predict", type=int, default=192)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--gib-per-cycle", type=float, help="bytes one verify cycle reads, in GiB (model-specific)")
    ap.add_argument("--ceiling-gbs", type=float, help="measured aggregate read bandwidth of the machine, GB/s")
    ap.add_argument("--timeout", type=float, default=3600)
    args = ap.parse_args()

    body = {"prompt": args.prompt, "n_predict": args.n_predict, "temperature": 0, "seed": args.seed,
            "cache_prompt": False, "stream": False}
    req = urllib.request.Request(f"http://{args.host}:{args.port}/completion", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t = json.load(urllib.request.urlopen(req, timeout=args.timeout))["timings"]

    n = int(t.get("predicted_n", 0))
    ms = float(t.get("predicted_ms", 0.0))
    drafted = int(t.get("draft_n", 0) or 0)
    accepted = int(t.get("draft_n_accepted", 0) or 0)
    cycles = max(1, n - accepted)

    print(f"decode           {t.get('predicted_per_second', 0):.2f} tok/s  ({n} tokens in {ms / 1000:.1f} s)")
    print(f"prompt           {t.get('prompt_per_second', 0):.1f} tok/s ({t.get('prompt_n')} tokens)")
    print(f"verify cycles    {cycles}")
    print(f"tokens/cycle     {n / cycles:.3f}")
    print(f"ms/cycle         {ms / cycles:.1f}")
    if drafted:
        print(f"draft acceptance {accepted}/{drafted} = {accepted / drafted:.3f}")
    else:
        print("draft acceptance n/a (no drafts reported: speculation off or not engaged)")
    if args.gib_per_cycle and ms > 0:
        gbps = args.gib_per_cycle * GIB / (ms / cycles / 1000)
        share = f" = {100 * gbps / args.ceiling_gbs:.0f}% of {args.ceiling_gbs:.1f}" if args.ceiling_gbs else ""
        print(f"implied bandwidth {gbps:.0f} GB/s at {args.gib_per_cycle:.1f} GiB/cycle{share}")
    print()
    print("Tokens/cycle well below draft length + 1 is an acceptance problem. A low bandwidth share with healthy")
    print("tokens/cycle is a bytes or per-operation overhead problem instead.")


if __name__ == "__main__":
    main()
