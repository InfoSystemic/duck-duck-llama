#!/usr/bin/env python3
import json
import os
from pathlib import Path
import subprocess
import time

root=Path(__file__).resolve().parents[2]
base=root/'serving/fleet-0903'
engine=root/'engines/llama.cpp-q4e-goal-0904'
benchmark=base/'results/glm5n-goal-q8-draft-copy-mtp3-t15/result.json'
while True:
    if benchmark.exists():
        try:
            if 'server_exit' in json.loads(benchmark.read_text()):
                break
        except ValueError:
            pass
    time.sleep(3)
print('GLM benchmark ended; building Qwen rollback changes',flush=True)
def run(command, label, env=None):
    with (base/'results'/f'{label}.log').open('w') as log:
        result=subprocess.run(list(map(str,command)),env=env,stdout=log,stderr=subprocess.STDOUT)
    print(label,result.returncode,flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
run(['cmake','--build',engine/'build-goal','-j','16','--target','llama-server'],'q4e-rs-rollback-build')
bindir=engine/'build-goal/bin'
env=dict(os.environ,LD_LIBRARY_PATH=str(bindir),LLAMA_TEST_Q4E_ROLLBACK='1')
for source,output in [('test-llama-archs.cpp','qwen-rollback-fixture-gen'),('test-recurrent-state-rollback.cpp','qwen-rollback-check')]:
    command=['g++','-O2','-std=c++17']
    command += ['-I'+str(engine/p) for p in ['include','common','ggml/include','src','vendor']]
    command += [engine/'tests'/source,'-L'+str(bindir),'-Wl,-rpath,'+str(bindir),'-lllama-common',
                engine/'build-goal/common/libllama-common-base.a','-lllama','-lggml','-lggml-cpu','-lggml-base','-pthread','-o',base/output]
    run(command,output+'-build',env)
stage=Path('/dev/shm/flash-goal-0904-qwen-rollback-fixtures')
stage.mkdir(exist_ok=True)
run([base/'qwen-rollback-fixture-gen','--arch','qwen4exp','--seed','904','--out',stage],'qwen-rollback-fixture-generation',env)
run([root/'serving/litellm-proxy-venv/bin/python3',base/'add-qwen-rollback-fixture-ple.py'],'qwen-rollback-fixture-ple',env)
env.update(GGML_Q4E_RS_ROLLBACK='1',LLAMA_TEST_REQUIRE_RS='1')
for name in ['qwen4exp-moe.gguf','qwen4exp-ple-moe.gguf']:
    run([base/'qwen-rollback-check','-m',stage/name,'-c','256','-t','4','-ngl','0','--flash-attn','on'],name+'.cpu-rollback-check',env)
