#!/usr/bin/env python3
"""lctx.py <port> [tag] [ctx...] -- decode rate as a function of CONTEXT LENGTH.

Every decode number in this fleet's record is taken at ~200 tokens of prompt, which is weights-only in practice. Agents run
at tens of thousands. On Qwen3.8-Flash-Next only 6 of 48 layers keep a growing KV cache (2 kv heads x 256 dim, f16 =
12,288 bytes per token across those layers), so the predicted penalty is ~3.2 GB/token of extra traffic at the full 262,144
context, on top of ~8.1 GB of weights -- about +40%. This measures it instead of predicting it.
"""
import json, sys, time, urllib.request

port = int(sys.argv[1]) if len(sys.argv) > 1 else 18083
tag  = sys.argv[2] if len(sys.argv) > 2 else "lctx"
ctxs = [int(x) for x in sys.argv[3:]] or [200, 8000, 32000, 100000]
N_PREDICT = 96

def post(path, body, timeout=3600):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))

# a filler that tokenises densely and does not repeat enough to feed the prompt cache or n-gram lookup
base = ("The quick brown fox jumps over the lazy dog near the riverbank at dawn while seventeen "
        "engineers debate cache coherency protocols and the merits of non-temporal stores. ")
print(f"  {tag}: decode rate vs context on :{port}")
print(f"  {'ctx tokens':>11} {'prompt tok/s':>13} {'decode tok/s':>13} {'ms/token':>9}")
for want in ctxs:
    words = max(1, int(want * 0.78))
    prompt = (base * (words // len(base.split()) + 2))
    body = {"prompt": prompt, "n_predict": N_PREDICT, "temperature": 0, "cache_prompt": False, "stream": False}
    try:
        t0 = time.time(); d = post("/completion", body); wall = time.time() - t0
    except Exception as e:
        print(f"  {want:>11,}  failed: {e}"); continue
    t = d.get("timings", {})
    pn, ps = t.get("prompt_n", 0), t.get("prompt_per_second") or 0
    dn, ds = t.get("predicted_n", 0), t.get("predicted_per_second") or 0
    print(f"  {pn:>11,} {ps:>13.1f} {ds:>13.2f} {1000/ds if ds else 0:>9.1f}   (wall {wall:.0f}s, generated {dn})")
