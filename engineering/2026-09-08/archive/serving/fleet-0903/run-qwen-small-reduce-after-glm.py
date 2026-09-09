#!/usr/bin/env python3
import json
import os
from pathlib import Path
import subprocess
import time
base=Path(__file__).resolve().parent
waiting=base/'results/glm5n-goal-reuse-mtp3-t8/result.json'
while True:
    try:
        if 'server_exit' in json.loads(waiting.read_text()):break
    except (FileNotFoundError,ValueError):pass
    time.sleep(3)
old=json.loads((base/'results/q4e-goal-rs-sigmoid-mtp3-t15/config.json').read_text())
cfg=old['config']
cfg.update(label='q4e-goal-rs-sigmoid-single-reduce-mtp3-t15',port=18107,
           profile_cpu=True,profile_phase=True,profile_answer_prefix=True)
env=dict(os.environ,**old['runtime_env'])
env['GGML_CPU_NUMA_FUSED_REDUCE_SINGLE_MAX_ELEMENTS']='65536'
args=['python3',str(base/'qwen-goal-case.py'),cfg.pop('label')]
for key,value in cfg.items():
    if isinstance(value,bool):
        if value:args.append('--'+key.replace('_','-'))
    elif value is not None:args.extend(['--'+key.replace('_','-'),str(value)])
print('GLM benchmark ended; starting Qwen with one worker per small reduction',flush=True)
raise SystemExit(subprocess.call(args,env=env))
