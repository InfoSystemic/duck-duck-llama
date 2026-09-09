#!/usr/bin/env python3
"""Reproduce the compact GLM case with a process-local OpenMP spin count."""
import argparse
import json
import os
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument('spin', type=int)
p.add_argument('--port', type=int, default=18128)
a = p.parse_args()
if a.spin < 0:
    p.error('spin must be nonnegative')
base = Path(__file__).resolve().parent
reference = base / 'results/glm5n-goal-compact-p4-draft-sweep-1024-t15'
old = json.loads((reference / 'config.json').read_text())
cfg = dict(old['config'])
cfg.update(label=f'glm5n-goal-compact-p4-spin{a.spin}-1024-t15',
           port=a.port, draft_n=2, draft_sweep=None, thread_sweep=None)
env = {key: value for key, value in os.environ.items()
       if not key.startswith(('GGML_', 'LLAMA_GRAPH_PHASE', 'LLAMA_MTP_DRAFT_N_FILE', 'OMP_', 'GOMP_'))}
env.update({key: value for key, value in old['runtime_env'].items()
            if not key.endswith('_FILE') and not key.startswith(('OMP_', 'GOMP_'))})
env['GOMP_SPINCOUNT'] = str(a.spin)
args = ['python3', str(base / 'qwen-goal-case.py'), cfg.pop('label')]
for key, value in cfg.items():
    if isinstance(value, bool):
        if value:
            args.append('--' + key.replace('_', '-'))
    elif value is not None:
        args.extend(['--' + key.replace('_', '-'), str(value)])
raise SystemExit(subprocess.call(args, env=env))
