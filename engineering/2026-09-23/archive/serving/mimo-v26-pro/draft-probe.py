#!/usr/bin/env python3
"""draft-probe.py [port] -- is the MTP head broken, or just weak?

The depth-3 run measured 8-10% draft acceptance, which collapses throughput (4.6 tok/s against 7.9 with no speculation).
Two very different causes produce a low number and they need different fixes:

  * the head is FINE but the text is hard       -> acceptance rises sharply on near-deterministic continuations
  * the head's INPUT is wrong (hidden state,
    head selection, chaining)                    -> acceptance stays flat no matter how predictable the text is

So: run the same server over prompts whose next tokens range from "literally copying" to "open-ended prose", and print
acceptance for each. A trained next-token head should accept nearly every draft when the continuation is a verbatim
repetition; if it does not, the draft is not seeing what it was trained to see.

Reports per prompt: decode rate, tokens per verify cycle, and drafted/accepted.
"""
import json, sys, urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 18190
N = int(sys.argv[2]) if len(sys.argv) > 2 else 96

CASES = [
    ("verbatim repeat",
     "Repeat this line exactly twenty times:\nthe quick brown fox jumps over the lazy dog\n"
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

print(f"{'case':18s} {'tok/s':>7s} {'tok/cyc':>8s} {'drafted':>8s} {'accepted':>9s} {'rate':>6s}")
for name, prompt in CASES:
    body = {"prompt": prompt, "n_predict": N, "temperature": 0, "seed": 42, "cache_prompt": False, "stream": False}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t = json.load(urllib.request.urlopen(req, timeout=3600))["timings"]
    n = int(t.get("predicted_n", 0))
    d = int(t.get("draft_n", 0) or 0)
    a = int(t.get("draft_n_accepted", 0) or 0)
    cyc = max(1, n - a)
    print(f"{name:18s} {t.get('predicted_per_second', 0):7.2f} {n/cyc:8.3f} {d:8d} {a:9d} "
          f"{(a/d if d else 0):6.3f}", flush=True)

print()
print("If even 'verbatim repeat' accepts under ~40%, the MTP head is not receiving a correct hidden state or the wrong")
print("head is being used for the draft step -- a plumbing bug, not a weak head. Compare with GLM-5.3-Flash at 0.775.")
