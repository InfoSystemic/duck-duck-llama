#!/usr/bin/env python3
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

root=Path(__file__).resolve().parents[2]
base=root/'serving/fleet-0903'
engine=root/'engines/llama.cpp-glm5n-goal-0904'
waiting=base/'results/q4e-goal-rs-sigmoid-iq-r16-mtp3-t15/result.json'
while True:
    try:
        if 'server_exit' in json.loads(waiting.read_text()):break
    except (FileNotFoundError,ValueError):pass
    time.sleep(3)
bindir=engine/'build-goal/bin'
snapshot=engine/'iq-r16-bin'
if not snapshot.exists():shutil.copytree(bindir,snapshot,symlinks=True)
def run(command,label,env=None):
    with (base/'results'/f'{label}.log').open('w') as f:
        p=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT)
    print(label,p.returncode,flush=True)
    if p.returncode:raise SystemExit(p.returncode)
run(['cmake','--build',str(engine/'build-goal'),'-j','16','--target','llama-server'],'glm-hc-post-vector-build')
command=['g++','-O2','-std=c++17']
command+=['-I'+str(engine/p) for p in ['include','ggml/include']]
command+=[str(base/'hc-post-vector-check.cpp'),'-L'+str(bindir),'-lggml','-lggml-cpu','-lggml-base','-pthread','-o',str(base/'hc-post-vector-check')]
run(command,'glm-hc-post-vector-check-build')
for enabled in [False,True]:
    env=dict(os.environ,LD_LIBRARY_PATH=str(bindir),GGML_CPU_HC_POST_VECTOR=str(int(enabled)))
    run([str(base/'hc-post-vector-check')],'glm-hc-post-vector-check-'+('on' if enabled else 'off'),env)
old=json.loads((base/'results/glm5n-goal-reuse-iq-r16-mtp3-t15/config.json').read_text())
cfg=old['config']
cfg.update(label='glm5n-goal-reuse-iq-r16-hc-mtp3-t15',port=18110)
env=dict(os.environ,**old['runtime_env'])
env['GGML_CPU_HC_POST_VECTOR']='1'
args=['python3',str(base/'qwen-goal-case.py'),cfg.pop('label')]
for key,value in cfg.items():
    if isinstance(value,bool):
        if value:args.append('--'+key.replace('_','-'))
    elif value is not None:args.extend(['--'+key.replace('_','-'),str(value)])
raise SystemExit(subprocess.call(args,env=env))
