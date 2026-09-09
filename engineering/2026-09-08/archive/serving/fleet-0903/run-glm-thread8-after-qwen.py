#!/usr/bin/env python3
import json
import os
from pathlib import Path
import subprocess
import time
base=Path(__file__).resolve().parent
waiting=base/'results/q4e-goal-rs-sigmoid-mtp3-t15/result.json'
while True:
    try:
        if 'server_exit' in json.loads(waiting.read_text()):break
    except (FileNotFoundError,ValueError):pass
    time.sleep(3)
old=json.loads((base/'results/glm5n-goal-mtp-hidden-reuse-mtp3-t15/config.json').read_text())
cfg=old['config']
cfg.update(label='glm5n-goal-reuse-mtp3-t8',threads=8,port=18106,profile_cpu=True,profile_phase=True)
env=dict(os.environ,**old['runtime_env'])
env.update(GGML_CPU_PARALLEL_COPY='0',GGML_CPU_Q8_0_REPACK='0',GGML_CPU_Q8_0_REPACK_FORCE='0')
args=['python3',str(base/'qwen-goal-case.py'),cfg.pop('label')]
for key,value in cfg.items():
    if isinstance(value,bool):
        if value:args.append('--'+key.replace('_','-'))
    elif value is not None:args.extend(['--'+key.replace('_','-'),str(value)])
print('Qwen benchmark ended; starting the eight-worker GLM comparison',flush=True)
raise SystemExit(subprocess.call(args,env=env))
