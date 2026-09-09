#!/usr/bin/env python3
"""Check the existing per-allocation huge-page option after the XXS full run."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

base = Path(__file__).resolve().parent
engine = base.parents[1] / 'engines/llama.cpp-glm5n-goal-0904'
p = argparse.ArgumentParser()
p.add_argument('--reference-label', default='glm5n-goal-iq-scale32-clean-1024-t15')
label = p.parse_args().reference_label
while True:
    try:
        current = json.loads((base / 'results' / label / 'result.json').read_text())
        if 'server_exit' in current:
            assert current['server_exit'] == 0 and not current.get('error')
            break
    except (FileNotFoundError, ValueError):
        pass
    time.sleep(3)
subprocess.run(['python3', str(base / 'validate-glm-full-reference.py'), label], check=True)
baseline = json.loads((base / 'results/glm5n-goal-x16-chunk16-1024-t15/result.json').read_text())
p2 = json.loads((base / 'results/glm5n-goal-q5-compact-p2-1024-t15/result.json').read_text())

def score(r):
    checks = [c for c in r['checks'] if c['expected'] is None and c['reasoning_budget_tokens'] is None]
    return 0 if any(c['throughput_contended'] for c in checks) else min(c['response']['timings']['predicted_per_second'] for c in checks)

selected = max([baseline, p2, current], key=score)
(base / 'results/glm-hugepages-reference.json').write_text(json.dumps({
    'selected': selected['config']['label'],
    'scores': {r['config']['label']: score(r) for r in [baseline, p2, current]}}, indent=2) + '\n')
if score(selected) >= 20:
    print('GLM meets the measured target; skip the huge-page trial.', flush=True)
    raise SystemExit(0)
env = {k: v for k, v in os.environ.items()
       if not k.startswith(('GGML_', 'LLAMA_GRAPH_PHASE', 'LLAMA_MTP_DRAFT_N_FILE', 'OMP_', 'GOMP_', 'REPACK_TEST_'))}
env.update({k: v for k, v in selected['runtime_env'].items()
            if not k.endswith('_FILE') and not k.startswith('GGML_CPU_OP_PROFILE')})
env['GGML_CPU_NUMA_HUGEPAGES'] = '1'
bindir = engine / ('build-goal/bin' if selected is current else
                   'q5-compact-p2-bin' if selected is p2 else 'validated-chunk16-bin')
cfg = dict(selected['config'])
draft = json.loads((base / 'results/glm-mtp-q8-persistent.json').read_text())
cfg.update(label='glm5n-goal-hugepages-1024-t15', port=18136, huge_pages=1,
           mtp=draft['destination'], binary=str(bindir / 'llama-server'))
args = ['python3', str(base / 'qwen-goal-case.py'), cfg.pop('label')]
for key, value in cfg.items():
    if isinstance(value, bool):
        if value:
            args.append('--' + key.replace('_', '-'))
    elif value is not None:
        args.extend(['--' + key.replace('_', '-'), str(value)])
raise SystemExit(subprocess.call(args, env=env))
