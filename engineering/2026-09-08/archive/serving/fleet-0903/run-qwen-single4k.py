#!/usr/bin/env python3
import json
import os
from pathlib import Path
import subprocess

base=Path(__file__).resolve().parent
old=json.loads((base/'results/q4e-goal-rs-iq-r16-q5-pair-mtp2-t15/config.json').read_text())
cfg=old['config']
cfg.update(label='q4e-goal-rs-iq-r16-q5-pair-mtp2-single4k-t15',port=18114)
env=dict(os.environ,**old['runtime_env'])
env['GGML_CPU_SINGLE_TASK_MAX_ELEMENTS']='4096'
args=['python3',str(base/'qwen-goal-case.py'),cfg.pop('label')]
for key,value in cfg.items():
    if isinstance(value,bool):
        if value:args.append('--'+key.replace('_','-'))
    elif value is not None:args.extend(['--'+key.replace('_','-'),str(value)])
raise SystemExit(subprocess.call(args,env=env))
