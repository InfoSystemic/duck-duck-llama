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

# 09-14: the old filler was `base * N`, i.e. PURE repetition, despite the comment claiming otherwise. Two consequences,
# both of which silently voided every number this tool produced: a repeated prompt makes the model emit EOS on the first
# token at temperature 0 (all three context points reported "generated 1" and 0.00 tok/s), and repetition feeds exactly
# the n-gram/prompt-cache paths the measurement is supposed to avoid. Deterministic non-repeating filler instead.
import random
# Common, overwhelmingly single-token words. The first version of this list used rare words (tessellate, obsidian,
# kestrel) which BPE splits into 2-4 tokens each, so every prompt overshot its target context by ~2x and a "32K" point
# was really ~58K -- 19 minutes of prefill instead of 8. Token count per word matters more than vocabulary richness
# here; randomised order already defeats the prompt cache and the n-gram draft.
_WORDS = ("the of and to in a is that it for was as with be by on not he this are or his from at which but have an "
          "they one you had we all her she there their when who will more no if out so said what up its about into "
          "than them can only other new some time very then how our two may these first also after most way even "
          "back any good work through where much before right well because those same day here take came").split()
def filler(n_words, seed=1234):
    r = random.Random(seed)
    return " ".join(r.choice(_WORDS) for _ in range(n_words))
# 09-14: warm the slot first. Window 54 called this cold and its 190-token point reported 19.72 tok/s at 31.4 tok/s
# prefill -- LOWER than its own 7,600-token point, which is impossible and was pure first-request warmup. Window 57's
# arm happened to warm up beforehand and got a clean 29.22 at the same context, which is how the artifact was caught.
try:
    post("/completion", {"prompt": "warm up the slot", "n_predict": 24, "temperature": 0,
                         "cache_prompt": False, "stream": False, "ignore_eos": True}, timeout=600)
except Exception as e:
    print(f"  warmup failed ({e}); treat the first context point as suspect", flush=True)
print(f"  {tag}: decode rate vs context on :{port}", flush=True)
print(f"  {'ctx tokens':>11} {'prompt tok/s':>13} {'decode tok/s':>13} {'ms/token':>9} {'draft acc':>9} {'tok/cycle':>9}", flush=True)
for want in ctxs:
    words = max(1, int(want * 0.95))   # ~1.05 tokens per common word, so targets land within ~5%
    prompt = filler(words)
    # ignore_eos is the whole point: without it the model stops after one token and the decode rate is a division by ~0
    body = {"prompt": prompt, "n_predict": N_PREDICT, "temperature": 0, "cache_prompt": False,
            "stream": False, "ignore_eos": True}
    try:
        t0 = time.time(); d = post("/completion", body); wall = time.time() - t0
    except Exception as e:
        print(f"  {want:>11,}  failed: {e}", flush=True); continue
    t = d.get("timings", {})
    pn, ps = t.get("prompt_n", 0), t.get("prompt_per_second") or 0
    dn, ds = t.get("predicted_n", 0), t.get("predicted_per_second") or 0
    # 09-14: without acceptance a decode-rate drop is AMBIGUOUS. Speculation converts one graph pass into several
    # tokens, so throughput falls either because the ops got slower OR because the draft stopped being accepted and
    # each cycle now yields fewer tokens. Those need completely different fixes, and only this number tells them apart.
    drn, dra = t.get("draft_n", 0), t.get("draft_n_accepted", 0)
    acc = 100*dra/drn if drn else 0
    warn = "  <-- generated < 8 tokens, decode rate is meaningless" if dn < 8 else ""
    tpc = dn/(drn/4.0) if drn else 0   # tokens per speculative cycle (n_max=4 drafts per cycle)
    print(f"  {pn:>11,} {ps:>13.1f} {ds:>13.2f} {1000/ds if ds else 0:>9.1f} {acc:>8.0f}% {tpc:>9.2f}"
          f"   (wall {wall:.0f}s, generated {dn}){warn}", flush=True)
