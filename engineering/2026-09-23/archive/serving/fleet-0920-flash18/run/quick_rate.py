#!/usr/bin/env python3
"""quick_rate.py --port P [--reps 3] [--n 192] [--sampled] -- decode rate and text hash of a llama-server instance on a fixed prompt.

Prompt: ~3K tokens of the fleet-0912 README followed by a writing task (the same shape as the Codex fixture: a few thousand
tokens of context, a 200-word answer). Greedy by default (temperature 0, seed 42) so the text hash must match across
configurations that claim bit-identical output; --sampled uses the server defaults (Codex traffic). Reports per repetition:
tok/s, ms per verify cycle (predicted_ms / (predicted_n - draft_n_accepted)), tokens per cycle, and the sha of the text.
The first repetition warms the prompt cache; the rate is reported for every repetition.
"""
import argparse, hashlib, json, sys, time, urllib.request
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port', type=int, required=True)
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--n', type=int, default=192)
    ap.add_argument('--sampled', action='store_true')
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    src = Path('/home/user/InfoSystemic/AI-Server/serving/fleet-0912-ctx/README.md').read_text()[:12000]
    prompt = src + ('\n\nWrite approximately 200 words explaining how an LRU cache uses a hash map and a doubly linked list, '
                    'including the get and put operations, capacity eviction, and a short concrete example.\n')
    body = dict(prompt=prompt, n_predict=a.n, cache_prompt=True, stream=False)
    if not a.sampled: body.update(temperature=0, seed=42)
    rows = []
    for r in range(a.reps):
        req = urllib.request.Request(f'http://127.0.0.1:{a.port}/completion', json.dumps(body).encode(), {'Content-Type': 'application/json'})
        t0 = time.time(); resp = json.load(urllib.request.urlopen(req, timeout=3600)); wall = time.time() - t0
        t = resp['timings']; n = int(t['predicted_n']); acc = int(t.get('draft_n_accepted', 0)); cyc = max(1, n - acc)
        h = hashlib.sha256(resp['content'].encode()).hexdigest()[:12]
        row = dict(rep=r, tok_s=round(t['predicted_per_second'], 2), ms_per_cycle=round(t['predicted_ms']/cyc, 2), tokens_per_cycle=round(n/cyc, 3),
                   prompt_n=int(t.get('prompt_n', 0)), prefill_tok_s=round(t.get('prompt_per_second', 0), 1), text_sha=h, wall_s=round(wall, 1))
        rows.append(row)
        print(f"{a.tag} rep {r}: {row['tok_s']:6.2f} tok/s | {row['ms_per_cycle']:6.2f} ms/cycle x {row['tokens_per_cycle']:.3f} tok/cycle | "
              f"prefill {row['prompt_n']} at {row['prefill_tok_s']} tok/s | text {h}", flush=True)
    print(json.dumps(dict(tag=a.tag, port=a.port, sampled=a.sampled, rows=rows)))

if __name__ == '__main__':
    main()
