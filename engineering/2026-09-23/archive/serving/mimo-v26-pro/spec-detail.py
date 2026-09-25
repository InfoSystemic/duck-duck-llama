#!/usr/bin/env python3
"""spec-detail.py [port] [n_predict] -- decode rate broken into the two things that set it: bytes per cycle and tokens per cycle.

A speculative decoder's rate is (cycles per second) x (tokens accepted per cycle). Only the second is a property of the
draft heads; the first is bandwidth. Reading them apart says whether a disappointing rate is a bytes problem or an
acceptance problem, and the two have completely different fixes.

Reports, from one request's server-side timings: tok/s, verify cycles, tokens per cycle, ms per cycle, draft acceptance, and
the implied sustained bandwidth given a bytes-per-cycle figure you supply (MiMo-V2.6-Pro at MTP depth 3 is ~59.3 GiB).
"""
import json, sys, urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 18190
N = int(sys.argv[2]) if len(sys.argv) > 2 else 192
GIB_PER_CYCLE = float(sys.argv[3]) if len(sys.argv) > 3 else 59.3

PROMPT = ("Explain in about 150 words why a speculative decoder's throughput is the product of its verify-cycle rate and the "
          "number of tokens accepted per cycle, and which of the two a memory-bandwidth limit constrains.")

body = {"prompt": PROMPT, "n_predict": N, "temperature": 0, "seed": 42, "cache_prompt": False, "stream": False}
req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", json.dumps(body).encode(),
                             {"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=3600))
t = r["timings"]

n = int(t.get("predicted_n", 0))
ms = float(t.get("predicted_ms", 0.0))
drafted = int(t.get("draft_n", 0) or 0)
accepted = int(t.get("draft_n_accepted", 0) or 0)
cycles = max(1, n - accepted)

print(f"decode           {t.get('predicted_per_second', 0):.2f} tok/s  ({n} tokens in {ms/1000:.1f} s)")
print(f"prompt           {t.get('prompt_per_second', 0):.1f} tok/s ({t.get('prompt_n')} tokens)")
print(f"verify cycles    {cycles}")
print(f"tokens/cycle     {n/cycles:.3f}")
print(f"ms/cycle         {ms/cycles:.1f}")
if drafted:
    print(f"draft acceptance {accepted}/{drafted} = {accepted/drafted:.3f}")
else:
    print("draft acceptance n/a (no drafts reported: speculation off or not engaged)")
gbps = GIB_PER_CYCLE * 1.0737 / (ms/cycles/1000)
print(f"implied bandwidth {gbps:.0f} GB/s at {GIB_PER_CYCLE:.1f} GiB/cycle = {100*gbps/381.6:.0f}% of this host's 381.6")
print()
print("Reading it: tokens/cycle well below the draft depth + 1 means an acceptance problem (better heads, lower depth).")
print("A low implied bandwidth percentage with healthy tokens/cycle means a bytes or op-overhead problem instead.")
