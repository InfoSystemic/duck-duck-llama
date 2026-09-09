#!/usr/bin/env python3
import json
import os
from pathlib import Path
import subprocess
import time

root=Path(__file__).resolve().parents[2]
base=root/'serving/fleet-0903'
engine=root/'engines/llama.cpp-q4e-goal-0904'
waiting=base/'results/glm5n-goal-reuse-iq-r16-mtp3-t15/result.json'
while True:
    try:
        if 'server_exit' in json.loads(waiting.read_text()):break
    except (FileNotFoundError,ValueError):pass
    time.sleep(3)
def run(command,label,env=None):
    with (base/'results'/f'{label}.log').open('w') as f:
        p=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT)
    print(label,p.returncode,flush=True)
    if p.returncode:raise SystemExit(p.returncode)
run(['cmake','--build',str(engine/'build-goal'),'-j','16','--target','llama-server'],'q4e-iq-r16-build')
bindir=engine/'build-goal/bin'
command=['g++','-O2','-std=c++17']
command+=['-I'+str(engine/p) for p in ['include','ggml/include','ggml/src','ggml/src/ggml-cpu']]
command+=[str(base/'iq2-repack-check.cpp'),'-L'+str(bindir),'-lggml','-lggml-cpu','-lggml-base','-pthread','-o',str(base/'qwen-quant-repack-check')]
run(command,'q4e-iq-r16-check-build')
for padded in [False,True]:
    env=dict(os.environ,LD_LIBRARY_PATH=str(bindir),GGML_CPU_IQ_R16_REPACK='1')
    if padded:env.update(REPACK_TEST_PADDED='1',REPACK_TEST_DOWN='1')
    run([str(base/'qwen-quant-repack-check'),'iq-r16-qwen'],'q4e-iq-r16-check-'+('padded-down' if padded else 'standard'),env)
old=json.loads((base/'results/q4e-goal-rs-sigmoid-single-reduce-mtp3-t15/config.json').read_text())
cfg=old['config']
cfg.update(label='q4e-goal-rs-sigmoid-iq-r16-mtp3-t15',port=18109)
env=dict(os.environ,**old['runtime_env'])
env['GGML_CPU_IQ_R16_REPACK']='1'
args=['python3',str(base/'qwen-goal-case.py'),cfg.pop('label')]
for key,value in cfg.items():
    if isinstance(value,bool):
        if value:args.append('--'+key.replace('_','-'))
    elif value is not None:args.extend(['--'+key.replace('_','-'),str(value)])
raise SystemExit(subprocess.call(args,env=env))
