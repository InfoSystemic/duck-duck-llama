#!/usr/bin/env python3
# Greedy reference completions (+ top-5 logprobs) from a MiMo server, for engine parity checks.
#   golden.py PORT OUT.json [N_PREDICT]     compare with: golden.py --compare A.json B.json
import json, sys, urllib.request
PROMPTS = [
    "<|im_start|>user\nWrite a Python function that returns the n-th Fibonacci number iteratively.<|im_end|>\n<|im_start|>assistant\n<think></think>",
    "<|im_start|>user\nExplain in two sentences why the sky is blue.<|im_end|>\n<|im_start|>assistant\n<think></think>",
    "The capital of France is Paris. The capital of Germany is Berlin. The capital of Japan is",
]
def run(port, n):
    out = []
    for p in PROMPTS:
        body = {"prompt": p, "n_predict": n, "temperature": 0, "n_probs": 5, "cache_prompt": False, "return_tokens": True}
        req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=3600).read())
        probs = [[(t.get("id"), round(t.get("logprob", 0), 4)) for t in step.get("top_logprobs", [])] for step in r.get("completion_probabilities", [])]
        out.append({"prompt": p, "content": r.get("content"), "tokens": r.get("tokens"), "top5": probs, "timings": r.get("timings")})
        print(repr(r.get("content"))[:160], "|", round(r.get("timings", {}).get("predicted_per_second", 0), 2), "tok/s", flush=True)
    return out
if sys.argv[1] == "--compare":
    a, b = (json.load(open(f)) for f in sys.argv[2:4])
    for i, (x, y) in enumerate(zip(a, b)):
        n = min(len(x["tokens"]), len(y["tokens"]))
        div = next((j for j in range(n) if x["tokens"][j] != y["tokens"][j]), None)
        dmax = max((abs(p[1] - q[1]) for sx, sy in zip(x["top5"], y["top5"]) for p, q in zip(sx[:1], sy[:1]) if p[0] == q[0]), default=0.0)
        print(f"prompt {i}: {'IDENTICAL tokens' if div is None else f'first divergence at token {div}'} over {n}; max |dlogprob(top1)| {dmax:.4f}")
else:
    port, path = int(sys.argv[1]), sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    json.dump(run(port, n), open(path, "w"), indent=1)
