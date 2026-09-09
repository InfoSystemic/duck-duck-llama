#!/usr/bin/env python3
"""Check a GLM-specific small-operation cutoff using the verified chunk16 runtime."""
import json
import os
from pathlib import Path
import subprocess

base = Path(__file__).resolve().parent
selected = json.loads((base / 'results/glm5n-goal-x16-chunk16-1024-t15/result.json').read_text())
assert selected.get('server_exit') == 0 and not selected.get('error')
preserved = json.loads((base / 'results/glm-mtp-q8-persistent.json').read_text())
assert preserved['verified'] and Path(preserved['destination']).stat().st_size == preserved['bytes']
env = {k: v for k, v in os.environ.items()
       if not k.startswith(('GGML_', 'LLAMA_GRAPH_PHASE', 'LLAMA_MTP_DRAFT_N_FILE', 'OMP_', 'GOMP_'))}
env.update({k: v for k, v in selected['runtime_env'].items()
            if not k.endswith('_FILE') and not k.startswith('GGML_CPU_OP_PROFILE')})
env['GGML_CPU_SINGLE_TASK_MAX_ELEMENTS'] = '16384'
cfg = dict(selected['config'])
cfg.update(label='glm5n-goal-single16k-1024-t15', port=18133, mtp=preserved['destination'])
args = ['python3', str(base / 'qwen-goal-case.py'), cfg.pop('label')]
for key, value in cfg.items():
    if isinstance(value, bool):
        if value:
            args.append('--' + key.replace('_', '-'))
    elif value is not None:
        args.extend(['--' + key.replace('_', '-'), str(value)])
raise SystemExit(subprocess.call(args, env=env))
