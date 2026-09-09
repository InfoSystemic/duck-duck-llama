#!/usr/bin/env python3
import json
import os
from pathlib import Path
import re
import subprocess

root=Path(__file__).resolve().parents[2]
base=root/'serving/fleet-0903'
engine=root/'engines/llama.cpp-glm5n-goal-0904'
bindir=engine/'build-goal/bin'
command=['g++','-O2','-std=c++17']
command+=['-I'+str(engine/p) for p in ['include','ggml/include','ggml/src','ggml/src/ggml-cpu']]
command+=[str(base/'iq2-repack-check.cpp'),'-L'+str(bindir),'-lggml','-lggml-cpu','-lggml-base','-pthread','-o',str(base/'quant-repack-check')]
subprocess.run(command,check=True)
summaries=[]
for padded in [False,True]:
    hashes=[]
    case_lines=[]
    for r16 in [False,True]:
        name=f'glm-iq-r{16 if r16 else 8}-check-'+('padded-down' if padded else 'standard')
        env=dict(os.environ,LD_LIBRARY_PATH=str(bindir),GGML_CPU_IQ_R16_REPACK=str(int(r16)))
        if padded:env.update(REPACK_TEST_PADDED='1',REPACK_TEST_DOWN='1')
        log=base/'results'/f'{name}.log'
        with log.open('w') as f:
            p=subprocess.run([base/'quant-repack-check','iq-r16'],env=env,stdout=f,stderr=subprocess.STDOUT)
        lines=log.read_text()
        print(name,p.returncode,lines.splitlines()[-1],flush=True)
        if p.returncode:raise SystemExit(p.returncode)
        hashes.append(re.findall(r'hash=([0-9a-f]+)',lines))
        case_lines.append([line for line in lines.splitlines() if line.startswith('PASS')])
    identical=hashes[0]==hashes[1] and len(hashes[0])==216
    different=[a.split(' max_abs=')[0] for a,b in zip(*case_lines) if a.split('hash=')[1]!=b.split('hash=')[1]]
    nonfused_identical=len(hashes[0])==216 and all('fused=1' in line for line in different)
    summaries.append(dict(padded_down=padded,cases=len(hashes[0]),r8_r16_output_hashes_identical=identical,
                          nonfused_output_hashes_identical=nonfused_identical,different_cases=different,
                          note='Fused SwiGLU chunk sizes change with NB_COLS; scalar expf and vector exp differ slightly. Every case passes the canonical operation tolerance.'))
    print(summaries[-1],flush=True)
(base/'results/glm-iq-r16-check-summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
if not all(x['nonfused_output_hashes_identical'] for x in summaries):raise SystemExit(1)
