#!/usr/bin/env python3
"""Draft acceptance by how predictable the text is: is a drafter broken, or only weak?

Low acceptance has two very different causes. A working drafter that meets hard text accepts nearly every draft on a
verbatim repetition and few on open prose. A drafter whose INPUT is wrong (a hidden state taken at the wrong point, the
wrong head, an empty embedding row, a wrong rotary setting) stays flat and low whatever the text. Running one server over
prompts that range from copying to open prose separates the two in about a minute, before any tuning.

On MiMo-V2.6-Pro this read 0.10-0.21 on every prompt for the MTP heads (not the model's drafter at all) and 1.000 on
verbatim text for the DFlash drafter once its six defects were fixed. Reports per prompt, from the server's own timings:
decode rate, tokens per verify cycle, drafted and accepted tokens. Needs a llama-server with speculative decoding on.
"""
import argparse, json, sys, urllib.request

CASES = [
    ("verbatim repeat",
     "Repeat this line exactly twenty times:\nthe quick brown fox jumps over the lazy dog\n"
     "the quick brown fox jumps over the lazy dog\nthe quick brown fox jumps over the lazy dog\n"
     "the quick brown fox jumps over the lazy dog\nthe quick brown fox jumps over the lazy dog\n"
     "the quick brown fox jumps over the lazy dog\nthe quick brown fox jumps over the lazy dog\n"),
    ("counting",
     "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35"),
    ("memorised list",
     "The capital of France is Paris. The capital of Germany is Berlin. The capital of Japan is"),
    ("code",
     "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n\n"
     "def factorial(n):\n"),
    ("open prose",
     "Write three sentences about why memory bandwidth limits CPU inference.\n\n"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--n-predict", type=int, default=96)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--json", help="also write the rows to this file")
    args = ap.parse_args()

    rows = []
    print(f"{'case':18s} {'tok/s':>7s} {'tok/cyc':>8s} {'drafted':>8s} {'accepted':>9s} {'rate':>6s}")
    for name, prompt in CASES:
        body = {"prompt": prompt, "n_predict": args.n_predict, "temperature": 0, "seed": args.seed,
                "cache_prompt": False, "stream": False}
        req = urllib.request.Request(f"http://{args.host}:{args.port}/completion", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        t = json.load(urllib.request.urlopen(req, timeout=args.timeout))["timings"]
        n = int(t.get("predicted_n", 0))
        drafted = int(t.get("draft_n", 0) or 0)
        accepted = int(t.get("draft_n_accepted", 0) or 0)
        cycles = max(1, n - accepted)          # every verify cycle emits its accepted drafts plus one token
        row = {"case": name, "tok_s": t.get("predicted_per_second", 0), "tokens_per_cycle": n / cycles,
               "drafted": drafted, "accepted": accepted, "acceptance": accepted / drafted if drafted else None}
        rows.append(row)
        print(f"{name:18s} {row['tok_s']:7.2f} {row['tokens_per_cycle']:8.3f} {drafted:8d} {accepted:9d} "
              f"{(row['acceptance'] or 0):6.3f}", flush=True)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=1)
    if not any(r["drafted"] for r in rows):
        print("\nno drafts reported: speculative decoding is off or never engaged", file=sys.stderr)
        return 1
    print("\nFlat, low acceptance even on 'verbatim repeat' points at the draft path's inputs, not at the drafter's quality.")
    print("A steep fall from verbatim to prose is the normal shape; tune the draft length or p_min for it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
