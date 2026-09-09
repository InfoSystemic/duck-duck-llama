#!/usr/bin/env python3
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

root=Path(__file__).resolve().parents[2]
base=root/'serving/fleet-0903'
waiting=base/'results/glm5n-goal-iq-r16-hc-copy-q5-pair-q8mtp2-t15/result.json'
while True:
    try:
        if 'server_exit' in json.loads(waiting.read_text()):break
    except (FileNotFoundError,ValueError):pass
    time.sleep(3)
choices=[]
for name in ['q4e-goal-rs-sigmoid-iq-r16-mtp3-t15','q4e-goal-rs-iq-r16-q5-pair-mtp2-t15']:
    directory=base/'results'/name
    result=json.loads((directory/'result.json').read_text())
    samples=[c for c in result['checks'] if c['expected'] is None and c.get('reasoning_budget_tokens') is None]
    if len(samples)==2 and all(not c['throughput_contended'] for c in samples) and all(c['pass'] for c in result['checks'] if c['expected'] is not None):
        choices.append((min(c['response']['timings']['predicted_per_second'] for c in samples),directory))
assert choices
score,reference=max(choices)
engine=root/'engines/llama.cpp-q4e-goal-0904'
snapshot=engine/'q5-pair-bin'
if not snapshot.exists():shutil.copytree(engine/'build-goal/bin',snapshot,symlinks=True)
old=json.loads((reference/'config.json').read_text())
cfg=old['config']
cfg.update(label='q4e-goal-best-iq-r16-hugepages-t15',port=18113,huge_pages=1)
env=dict(os.environ,**old['runtime_env'])
args=['python3',str(base/'qwen-goal-case.py'),cfg.pop('label')]
for key,value in cfg.items():
    if isinstance(value,bool):
        if value:args.append('--'+key.replace('_','-'))
    elif value is not None:args.extend(['--'+key.replace('_','-'),str(value)])
print('Huge-page comparison reference:',reference.name,'minimum workload rate:',score,flush=True)
raise SystemExit(subprocess.call(args,env=env))
