#!/usr/bin/env python3
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np

base = Path(__file__).resolve().parent
results = base / 'results'
model = '/dev/shm/flash-goal-0904-mtp-q8/GLM-5.3-Flash-MTP-Q8_0.gguf'
cases = [
    ('glm-mtp-head-q8-draft-cpu', 'glm-mtp-head-hidden-fixed-cpu', model, {}),
    ('glm-mtp-head-q8-draft-numa', 'glm-mtp-head-hidden-fixed', model,
     {'GGML_CPU_Q8_0_REPACK': '1', 'GGML_CPU_Q8_0_REPACK_FORCE': '1', 'GGML_CPU_PARALLEL_COPY': '1'}),
    ('qwen-mtp-head-parallel-sigmoid', 'qwen-mtp-head-q8-force-hidden-checked-conditional', None,
     {'GGML_CPU_PARALLEL_SIGMOID': '1', 'GGML_CPU_PARALLEL_COPY': '1'}),
]
for label, source, new_model, extra in cases:
    config = json.loads((results / (source + '.config.json')).read_text())
    command = config['command'][:]
    if new_model:
        command[1] = new_model
    command[2] = str(results / (label + '.f32'))
    env = dict(os.environ, **config['env'])
    env.update(extra)
    prefix = ['numactl'] + config['numactl'] if config.get('numactl') else []
    config.update(command=command, env={k:v for k,v in env.items() if k.startswith('GGML_') or k == 'LD_LIBRARY_PATH'})
    (results / (label + '.config.json')).write_text(json.dumps(config, indent=2) + '\n')
    with (results / (label + '.log')).open('w') as log:
        run = subprocess.run(prefix + command, env=env, stdout=log, stderr=subprocess.STDOUT)
    print(label, run.returncode, flush=True)
    if run.returncode:
        raise SystemExit(run.returncode)

comparisons = [
    ('glm-mtp-head-q8-draft-numa', 'glm-mtp-head-q8-draft-cpu', 154880, 4096),
    ('qwen-mtp-head-parallel-sigmoid', 'qwen-mtp-head-q8-force-hidden-checked-conditional', 248320, 10240),
]
for label, reference, vocab, hidden in comparisons:
    a = np.fromfile(results / (label + '.f32'), np.float32).reshape(5, vocab + hidden)
    b = np.fromfile(results / (reference + '.f32'), np.float32).reshape(5, vocab + hidden)
    out = dict(label=label, reference=reference, finite=bool(np.isfinite(a).all() and np.isfinite(b).all()),
               logits_max_abs=float(np.abs(a[:, :vocab]-b[:, :vocab]).max()),
               hidden_max_abs=float(np.abs(a[:, vocab:]-b[:, vocab:]).max()),
               top1=a[:, :vocab].argmax(axis=1).tolist(), reference_top1=b[:, :vocab].argmax(axis=1).tolist(),
               hidden_norms=np.linalg.norm(a[:, vocab:], axis=1).tolist(), bit_identical=bool(np.array_equal(a,b)))
    (results / (label + '-compare.json')).write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out), flush=True)
