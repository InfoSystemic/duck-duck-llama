#!/usr/bin/env python3
import json
import os
from pathlib import Path
import resource
import subprocess
import time

root=Path(__file__).resolve().parents[2]
base=root/'serving/fleet-0903'
waiting=base/'results/q4e-goal-best-iq-r16-hugepages-t15/result.json'
while True:
    try:
        if 'server_exit' in json.loads(waiting.read_text()):break
    except (FileNotFoundError,ValueError):pass
    time.sleep(3)
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
engine=root/'engines/llama.cpp-glm5n-goal-0904'
bindir=engine/'build-goal/bin'
command=['g++','-O2','-std=c++17']
command+=['-I'+str(engine/p) for p in ['include','ggml/include','ggml/src','ggml/src/ggml-cpu']]
command+=[str(base/'iq2-repack-check.cpp'),'-L'+str(bindir),'-lggml','-lggml-cpu','-lggml-base','-ldl','-pthread','-o',str(base/'iqk-comparison-check')]
subprocess.run(command,check=True)
env=dict(os.environ,LD_LIBRARY_PATH=str(bindir),GGML_CPU_IQ_R16_REPACK='1',
         IQK_BRIDGE_LIBRARY=str(root/'engines/ik_llama.cpp-flash-goal-0904/build-goal/ggml/src/libggml.so'))
for padded in [False,True]:
    case_env=dict(env)
    if padded:case_env.update(REPACK_TEST_PADDED='1',REPACK_TEST_DOWN='1')
    name='iqk-raw-kernel-compare-'+('padded-down' if padded else 'standard')
    with (base/'results'/f'{name}.log').open('w') as f:
        p=subprocess.run(['numactl','--physcpubind=0','--membind=0',str(base/'iqk-comparison-check'),'iqk-compare'],env=case_env,stdout=f,stderr=subprocess.STDOUT)
    print(name,p.returncode,flush=True)
    if p.returncode:raise SystemExit(p.returncode)
