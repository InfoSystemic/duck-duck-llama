#!/usr/bin/env python3
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

root=Path(__file__).resolve().parents[2]
base=root/'serving/fleet-0903'
waiting=base/'results/q4e-goal-rs-iq-r16-q5-pair-mtp2-t15/result.json'
while True:
    try:
        if 'server_exit' in json.loads(waiting.read_text()):break
    except (FileNotFoundError,ValueError):pass
    time.sleep(3)
engine=root/'engines/llama.cpp-glm5n-goal-0904'
snapshot=engine/'hc-vector-bin'
if not snapshot.exists():shutil.copytree(engine/'build-goal/bin',snapshot,symlinks=True)
old=json.loads((base/'results/glm5n-goal-reuse-iq-r16-hc-mtp3-t15/config.json').read_text())
cfg=old['config']
cfg.update(label='glm5n-goal-iq-r16-hc-copy-q5-pair-q8mtp2-t15',port=18112,draft_n=2,bench_direct=True,
           mtp='/dev/shm/flash-goal-0904-mtp-q8/GLM-5.3-Flash-MTP-Q8_0.gguf')
env=dict(os.environ,**old['runtime_env'])
env.update(GGML_CPU_X16_Q5_BATCH2='1',GGML_CPU_PARALLEL_COPY='1',GGML_CPU_Q8_0_REPACK='1',
           GGML_CPU_Q8_0_REPACK_FORCE='1',GGML_CPU_NUMA_FUSED_REDUCE_SINGLE_MAX_ELEMENTS='65536')
args=['python3',str(base/'qwen-goal-case.py'),cfg.pop('label')]
for key,value in cfg.items():
    if isinstance(value,bool):
        if value:args.append('--'+key.replace('_','-'))
    elif value is not None:args.extend(['--'+key.replace('_','-'),str(value)])
raise SystemExit(subprocess.call(args,env=env))
