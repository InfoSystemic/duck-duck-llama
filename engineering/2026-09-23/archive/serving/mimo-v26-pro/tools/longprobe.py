#!/usr/bin/env python3
# longprobe.py PORT OUT.json [N_LINES] -- one greedy completion (top-5 logprobs) after a long synthetic prompt, so
# attention runs at depth; used to A/B kernels. The prompt is deterministic numbered prose, not pure repetition.
import json, sys, urllib.request
port, out = int(sys.argv[1]), sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 400
words = "river stone lantern orchard copper meadow signal harbor thistle ember quarry violet anchor saddle beacon".split()
lines = [f"Entry {i}: the {words[i % 15]} near the {words[(i * 7) % 15]} was logged at {(i * 37) % 1000} units." for i in range(n)]
prompt = "\n".join(lines) + "\nQuestion: which word appears in Entry 123 first? Answer:"
body = {"prompt": prompt, "n_predict": 24, "temperature": 0, "n_probs": 5, "cache_prompt": False, "return_tokens": True}
req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
r = json.loads(urllib.request.urlopen(req, timeout=7200).read())
probs = [[(t.get("id"), round(t.get("logprob", 0), 4)) for t in s.get("top_logprobs", [])] for s in r.get("completion_probabilities", [])]
json.dump([{"prompt": "long", "content": r.get("content"), "tokens": r.get("tokens"), "top5": probs, "timings": r.get("timings")}], open(out, "w"), indent=1)
t = r.get("timings", {})
print(f"long prompt {t.get('prompt_n')} tok: prefill {t.get('prompt_per_second', 0):.1f} tok/s, decode {t.get('predicted_per_second', 0):.2f} tok/s |", repr(r.get("content"))[:80])
