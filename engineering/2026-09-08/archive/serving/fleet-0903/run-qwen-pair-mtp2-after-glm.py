#!/usr/bin/env python3
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

root=Path(__file__).resolve().parents[2]
base=root/'serving/fleet-0903'
waiting=base/'results/glm5n-goal-reuse-iq-r16-hc-mtp3-t15/result.json'
while True:
    try:
        if 'server_exit' in json.loads(waiting.read_text()):break
    except (FileNotFoundError,ValueError):pass
    time.sleep(3)
def run(command,label,env=None):
    log=base/'results'/f'{label}.log'
    with log.open('w') as f:
        p=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT)
    print(label,p.returncode,flush=True)
    if p.returncode:raise SystemExit(p.returncode)
    return log.read_text()
engine=root/'engines/llama.cpp-q4e-goal-0904'
bindir=engine/'build-goal/bin'
snapshot=engine/'iq-r16-bin'
if not snapshot.exists():shutil.copytree(bindir,snapshot,symlinks=True)
run(['cmake','--build',str(engine/'build-goal'),'-j','16','--target','llama-server'],'q4e-q5-pair-build')
summaries=[]
for name in ['glm5n','q4e']:
    eng=root/f'engines/llama.cpp-{name}-goal-0904'
    binary_dir=eng/'build-goal/bin'
    output=base/f'{name}-q5-pair-check'
    command=['g++','-O2','-std=c++17']
    command+=['-I'+str(eng/p) for p in ['include','ggml/include','ggml/src','ggml/src/ggml-cpu']]
    command+=[str(base/'iq2-repack-check.cpp'),'-L'+str(binary_dir),'-lggml','-lggml-cpu','-lggml-base','-pthread','-o',str(output)]
    run(command,name+'-q5-pair-check-build')
    for padded in [False,True]:
        hashes=[]
        for enabled in [False,True]:
            env=dict(os.environ,LD_LIBRARY_PATH=str(binary_dir),GGML_CPU_X16_Q5_BATCH2=str(int(enabled)))
            if padded:env.update(REPACK_TEST_PADDED='1',REPACK_TEST_DOWN='1')
            label=f'{name}-q5-pair-check-'+('padded' if padded else 'standard')+'-'+('on' if enabled else 'off')
            log=run([str(output),'q5-pair'],label,env)
            hashes.append(re.findall(r'hash=([0-9a-f]+)',log))
        same=hashes[0]==hashes[1] and len(hashes[0])==72
        summaries.append(dict(engine=name,padded=padded,cases=len(hashes[0]),output_hashes_identical=same))
        print(summaries[-1],flush=True)
        if not same:raise SystemExit(1)
(base/'results/q5-pair-check-summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
old=json.loads((base/'results/q4e-goal-rs-sigmoid-iq-r16-mtp3-t15/config.json').read_text())
cfg=old['config']
cfg.update(label='q4e-goal-rs-iq-r16-q5-pair-mtp2-t15',port=18111,draft_n=2)
env=dict(os.environ,**old['runtime_env'])
env['GGML_CPU_X16_Q5_BATCH2']='1'
args=['python3',str(base/'qwen-goal-case.py'),cfg.pop('label')]
for key,value in cfg.items():
    if isinstance(value,bool):
        if value:args.append('--'+key.replace('_','-'))
    elif value is not None:args.extend(['--'+key.replace('_','-'),str(value)])
raise SystemExit(subprocess.call(args,env=env))
