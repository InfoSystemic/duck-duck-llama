#!/usr/bin/env python3
"""Try exact-history ngram proposals before MTP after the current dense-chunk run."""
import json
import os
from pathlib import Path
import subprocess
import time

base = Path(__file__).resolve().parent
previous = base / 'results/glm5n-goal-x16-chunk16-1024-t15'
while True:
    try:
        finished = json.loads((previous / 'result.json').read_text())
        if 'server_exit' in finished:
            if finished.get('error') or finished['server_exit'] != 0:
                raise SystemExit('Previous run did not finish successfully')
            break
    except (FileNotFoundError, ValueError):
        pass
    time.sleep(3)

def score(result):
    checks = [c for c in result['checks']
              if c['expected'] is None and c['reasoning_budget_tokens'] is None]
    if result.get('error') or any(c['throughput_contended'] for c in checks):
        return 0
    return min(c['response']['timings']['predicted_per_second'] for c in checks)

references = [finished]
for label in ['glm5n-goal-dynamic-iq-tiles-1024-t15', 'glm5n-goal-compact-p4-draft-sweep-1024-t15']:
    references.append(json.loads((base / 'results' / label / 'result.json').read_text()))
selected = max(references, key=score)
if score(selected) >= 20:
    print('A previous candidate reached the throughput target; skip this experiment.', flush=True)
    raise SystemExit(0)
(base / 'results/glm-ngram-reference.json').write_text(json.dumps({
    'selected': selected['config']['label'],
    'scores': {r['config']['label']: score(r) for r in references}}, indent=2) + '\n')
env = {k: v for k, v in os.environ.items()
       if not k.startswith(('GGML_', 'LLAMA_GRAPH_PHASE', 'LLAMA_MTP_DRAFT_N_FILE', 'OMP_', 'GOMP_'))}
env.update({k: v for k, v in selected['runtime_env'].items()
            if not k.endswith('_FILE') and not k.startswith('GGML_CPU_OP_PROFILE')})
cfg = dict(selected['config'])
cfg.update(label='glm5n-goal-ngram4-mtp2-1024-t15', port=18132,
           spec_type='ngram-simple,draft-mtp', ngram_n=4, ngram_m=4,
           draft_n=2, draft_sweep=None, thread_sweep=None)
args = ['python3', str(base / 'qwen-goal-case.py'), cfg.pop('label')]
for key, value in cfg.items():
    if isinstance(value, bool):
        if value:
            args.append('--' + key.replace('_', '-'))
    elif value is not None:
        args.extend(['--' + key.replace('_', '-'), str(value)])
raise SystemExit(subprocess.call(args, env=env))
