#!/usr/bin/env python3
"""Retry the cancelled scale32 configuration through the guarded harness."""
import argparse
import json
import os
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--label', default='glm5n-goal-iq-scale32-clean-1024-t15')
p.add_argument('--port', type=int, default=18135)
a = p.parse_args()
base = Path(__file__).resolve().parent
previous = base / 'results/glm5n-goal-iq-scale32-1024-t15'
result = json.loads((previous / 'result.json').read_text())
assert result.get('server_exit') is not None and (previous / 'contention-cancellation.json').exists()
assert not (base / 'results' / a.label).exists(), 'Use a fresh label to preserve existing evidence'
env = {k: v for k, v in os.environ.items()
       if not k.startswith(('GGML_', 'LLAMA_GRAPH_PHASE', 'LLAMA_MTP_DRAFT_N_FILE', 'OMP_', 'GOMP_', 'REPACK_TEST_'))}
env.update({k: v for k, v in result['runtime_env'].items()
            if not k.endswith('_FILE') and not k.startswith('GGML_CPU_OP_PROFILE')})
cfg = dict(result['config'])
cfg.update(label=a.label, port=a.port)
args = ['python3', str(base / 'qwen-goal-case.py'), cfg.pop('label')]
for key, value in cfg.items():
    if isinstance(value, bool):
        if value:
            args.append('--' + key.replace('_', '-'))
    elif value is not None:
        args.extend(['--' + key.replace('_', '-'), str(value)])
raise SystemExit(subprocess.call(args, env=env))
