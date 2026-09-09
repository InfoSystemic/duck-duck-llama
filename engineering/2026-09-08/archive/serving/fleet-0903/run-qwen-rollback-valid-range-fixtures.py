#!/usr/bin/env python3
import json
import os
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parents[2]
base=root/'serving/fleet-0903';engine=root/'engines/llama.cpp-q4e-goal-0904'
build=Path(os.environ.get('QWEN_ROLLBACK_BUILD_DIR',str(engine/'build-goal')))
bindir=build/'bin'
stage=Path('/dev/shm/flash-goal-0904-qwen-rollback-fixtures')
env=dict(os.environ,LD_LIBRARY_PATH=str(bindir))
cmd=['g++','-O2','-std=c++17']+['-I'+str(engine/p) for p in ['include','common','ggml/include','src','vendor']]
cmd += [str(engine/'tests/test-recurrent-state-rollback.cpp'),'-L'+str(bindir),'-Wl,-rpath,'+str(bindir),'-lllama-common',str(build/'common/libllama-common-base.a'),'-lllama','-lggml','-lggml-cpu','-lggml-base','-pthread','-o',str(base/'qwen-rollback-check')]
subprocess.run(cmd,env=env,check=True)
env.update(GGML_Q4E_RS_ROLLBACK='1',LLAMA_TEST_RS_VALID_RANGE='1',LLAMA_TEST_REQUIRE_RS='1',LLAMA_TEST_SPECULATIVE_RS='1',GGML_CPU_PARALLEL_SIGMOID='1',GGML_CPU_PARALLEL_COPY='1')
for mode in ['cpu','numa']:
    for model in ['qwen4exp-moe','qwen4exp-ple-moe']:
        cmd=[str(base/'qwen-rollback-check'),'-m',str(stage/(model+'-split-experts.gguf')),'-c','256','-t','2','--flash-attn','on','--fit','off']
        case_env=env.copy()
        if mode=='numa':
            case_env.update(GGML_CPU_NUMA_DEVICES='1',GGML_CPU_NUMA_THREADS='2',GGML_CPU_NUMA_REPACK='1',GGML_CPU_NUMA_DIRECT_ALLREDUCE='1',GGML_CPU_NUMA_FUSED_REDUCE='1',GGML_CPU_NUMA_MERGE_REDUCE='1',GGML_Q4E_SPLIT='13')
            cmd+=['-ngl','999','--device','CPU-NUMA0,CPU-NUMA1,CPU-NUMA2,CPU-NUMA3','--split-mode','tensor','--tensor-split','1,1,1,1']
        else:
            cmd+=['-ngl','0']
        label=os.environ.get('QWEN_ROLLBACK_LABEL_PREFIX','qwen-rs-valid-range')+f'-{model}-{mode}'
        config=dict(command=cmd,env={k:v for k,v in case_env.items() if k.startswith(('GGML_','LLAMA_TEST_')) or k=='LD_LIBRARY_PATH'})
        (base/'results'/(label+'.config.json')).write_text(json.dumps(config,indent=2)+'\n')
        with (base/'results'/(label+'.log')).open('w') as log:
            r=subprocess.run(cmd,env=case_env,stdout=log,stderr=subprocess.STDOUT)
        print(label,r.returncode,flush=True)
        if r.returncode:raise SystemExit(r.returncode)
