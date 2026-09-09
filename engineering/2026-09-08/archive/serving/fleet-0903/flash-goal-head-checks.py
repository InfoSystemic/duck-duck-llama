#!/usr/bin/env python3
"""Sequential full MTP sidecar execution checks for the isolated Flash engines."""
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[2]
base = root / 'serving/fleet-0903'
label = sys.argv[1]
common = {key: '1' for key in (
    'GGML_CPU_NUMA_DEVICES', 'GGML_CPU_NUMA_THREADS', 'GGML_CPU_NUMA_REPACK',
    'GGML_CPU_NUMA_DIRECT_ALLREDUCE', 'GGML_CPU_NUMA_FUSED_REDUCE', 'GGML_CPU_NUMA_MERGE_REDUCE',
    'GGML_CPU_IQ2_XS_REPACK', 'GGML_CPU_IQ3_XXS_REPACK', 'GGML_CPU_Q5_K_REPACK',
    'GGML_CPU_X16_Q4_K', 'GGML_CPU_X16_Q5_K', 'GGML_CPU_X16_Q6_K', 'GGML_CPU_X16_Q8_0',
    'GGML_CPU_X16_ATTN3D', 'GGML_GLM5N_MLA_TP', 'GGML_GLM5N_ATTN_TP', 'GGML_GLM5N_SHEXP_TP')}
common.update(GGML_CPU_REPACK_LOAD_THREADS='16', GGML_Q4E_SPLIT='13')
models = [
    ('glm', 'glm5n', '/models/gguf/GLM-5.3-Flash/MTP/GLM-5.3-Flash-MTP-IQ2_XXS.gguf'),
    ('qwen', 'q4e', '/models/gguf/Qwen3.8-Flash-Next/MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf'),
]
for name, engine, model in models:
    if len(sys.argv) > 2 and name != sys.argv[2]:
        continue
    env = dict(os.environ, **common)
    env['LD_LIBRARY_PATH'] = str(root / f'engines/llama.cpp-{engine}-goal-0904/build-goal/bin')
    out = base / f'results/{name}-mtp-head-{label}'
    command = [str(base / f'{name}-mtp-head-check'), model, str(out.with_suffix('.f32')), '4']
    out.with_suffix('.config.json').write_text(json.dumps({'command': command, 'env': {k: v for k, v in env.items() if k.startswith('GGML_') or k == 'LD_LIBRARY_PATH'}}, indent=2))
    with out.with_suffix('.log').open('w') as log:
        completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    print(name, completed.returncode, flush=True)
    if completed.returncode:
        sys.exit(completed.returncode)
