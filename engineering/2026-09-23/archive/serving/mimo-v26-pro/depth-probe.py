#!/usr/bin/env python3
"""depth-probe.py [port] [depths...] -- does MiMo's 256K context actually stay usable, or only fit in memory?

Two different questions, and they have different answers:

  * PREFILL cost is what you pay to get to a depth at all. At ~39 tok/s a 32K prompt is fourteen minutes, so this is
    usually the limit on how long a context is worth using, not memory.
  * DECODE cost at depth is what you pay per token once you are there. 63 of this model's 73 layers are
    sliding-window-128, so only 10 layers grow their KV with depth -- the decode slope should be gentle, unlike
    GLM-5.3-Flash where the DSA indexer walks the whole cache every token.

Measures both at each depth in ONE append-only session, so each step only prefills the tokens it adds, the same way a
real conversation does. Reports prefill tok/s for the new tokens, decode tok/s, and the draft acceptance at that depth
(a block drafter can behave differently deep in a context than at the start).
"""
import json, sys, time, urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 18190
DEPTHS = [int(x) for x in sys.argv[2:]] or [4096, 16384, 65536]
N_PREDICT = 64

FILLER = ("The memory system, not the arithmetic, is what limits large-model inference on a CPU. Every generated token "
          "requires reading the active weights from DRAM, so throughput is the ratio of available bandwidth to bytes "
          "touched per token. ")


def post(path, body, timeout=7200):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def n_tokens(text):
    return len(post("/tokenize", {"content": text})["tokens"])


print(f"{'depth':>8} {'prompt tok':>11} {'prefill t/s':>12} {'decode t/s':>11} {'tok/cyc':>8} {'accept':>7} {'wall s':>8}")
prompt = "You are reading a long technical document.\n\n"
reached = 0
for target in DEPTHS:
    # grow the prompt to the target depth; the prompt cache means only the new tokens are prefilled
    while reached < target:
        prompt += FILLER
        reached = n_tokens(prompt) if len(prompt) % 4096 < len(FILLER) else reached + len(FILLER) // 4
    prompt_n = n_tokens(prompt)
    q = prompt + "\n\nIn one sentence, what limits CPU inference throughput?\n"
    t0 = time.time()
    r = post("/completion", {"prompt": q, "n_predict": N_PREDICT, "temperature": 0, "seed": 42,
                             "cache_prompt": True, "stream": False})
    wall = time.time() - t0
    t = r["timings"]
    n = int(t.get("predicted_n", 0))
    a = int(t.get("draft_n_accepted", 0) or 0)
    d = int(t.get("draft_n", 0) or 0)
    cyc = max(1, n - a)
    print(f"{prompt_n:>8} {int(t.get('prompt_n', 0)):>11} {t.get('prompt_per_second', 0):>12.1f} "
          f"{t.get('predicted_per_second', 0):>11.2f} {n/cyc:>8.3f} {(a/d if d else 0):>7.3f} {wall:>8.1f}", flush=True)
    reached = prompt_n

print()
print("prompt tok is the NEW tokens prefilled at that step, not the whole context -- the cache carries the rest, which")
print("is what an append-only conversation actually pays. A decode rate that holds up across depths means the sliding")
print("window is doing its job; a falling one means the 10 full-attention layers dominate.")
