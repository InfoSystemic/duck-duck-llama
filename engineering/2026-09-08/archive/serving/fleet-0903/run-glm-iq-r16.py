#!/usr/bin/env python3
import json
import os
from pathlib import Path
import subprocess

base=Path(__file__).resolve().parent
old=json.loads((base/'results/glm5n-goal-reuse-mtp3-t8/config.json').read_text())
cfg=old['config']
cfg.update(label='glm5n-goal-reuse-iq-r16-mtp3-t15',port=18108,threads=15,
           profile_cpu=True,profile_phase=True,profile_answer_prefix=True)
env=dict(os.environ,**old['runtime_env'],GGML_CPU_IQ_R16_REPACK='1')
args=['python3',str(base/'qwen-goal-case.py'),cfg.pop('label')]
for key,value in cfg.items():
    if isinstance(value,bool):
        if value:args.append('--'+key.replace('_','-'))
    elif value is not None:args.extend(['--'+key.replace('_','-'),str(value)])
raise SystemExit(subprocess.call(args,env=env))
