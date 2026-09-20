#!/usr/bin/env python3
"""parity.py <port> <tag> [--ref tag] -- greedy completions with top-5 logprobs on short/mid/long prompts.
Saves results/parity-<tag>.json; with --ref compares text and logprobs against another tag."""
import json, sys, urllib.request, time, argparse
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument('port'); ap.add_argument('tag'); ap.add_argument('--ref'); ap.add_argument('--n', type=int, default=48)
ap.add_argument('--only', default='')
a = ap.parse_args()
W = Path('/home/user/InfoSystemic/AI-Server/serving/fleet-0920-flash18')
src = (Path('/home/user/InfoSystemic/AI-Server/serving/fleet-0912-ctx/README.md')).read_text()
prompts = {
  'short': 'Implement a thread-safe LRU cache in C++ with O(1) get and put. Explain the data structures, then give the complete code with comments.',
  'mid':   src[:12000] + '\n\nSummarise the key findings above in five bullet points.\n',
  'long':  src[:40000] + '\n\nSummarise the key findings above in five bullet points.\n',
}
if a.only: prompts = {k: v for k, v in prompts.items() if k in a.only.split(',')}
out = {}
for name, p in prompts.items():
    body = dict(prompt=p, n_predict=a.n, temperature=0, seed=42, cache_prompt=False, stream=False, n_probs=5)
    t0 = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{a.port}/completion', json.dumps(body).encode(), {'Content-Type': 'application/json'}), timeout=3600))
    t = r['timings']
    probs = r.get('completion_probabilities', [])
    out[name] = dict(content=r['content'], tokens=[x.get('id') for x in probs], logprob=[x.get('logprob') for x in probs],
                     top=[[(y.get('id'), y.get('logprob')) for y in x.get('top_logprobs', [])] for x in probs],
                     prompt_n=t.get('prompt_n'), pp=t.get('prompt_per_second'), tg=t.get('predicted_per_second'), draft_n=t.get('draft_n'), acc=t.get('draft_n_accepted'))
    print(f"{name}: prompt {t.get('prompt_n')} tok, pp {t.get('prompt_per_second'):.1f}, tg {t.get('predicted_per_second'):.2f} tok/s, draft {t.get('draft_n')}/{t.get('draft_n_accepted')}  [{time.time()-t0:.0f}s]", flush=True)
(W/'results'/f'parity-{a.tag}.json').write_text(json.dumps(out))
if a.ref:
    ref = json.loads((W/'results'/f'parity-{a.ref}.json').read_text())
    for name in out:
        A, B = out[name], ref[name]
        same = A['content'] == B['content']
        n = 0
        for x, y in zip(A['tokens'], B['tokens']):
            if x != y: break
            n += 1
        dmax = max([abs(x-y) for x, y in list(zip(A['logprob'], B['logprob']))[:n]] or [0])
        print(f"  {name}: text {'IDENTICAL' if same else 'DIFFERS'}; common prefix {n}/{len(B['tokens'])} tokens; max |dlogprob| over prefix {dmax:.2e}")
