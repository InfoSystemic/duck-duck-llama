#!/usr/bin/env python3
from pathlib import Path
import hashlib
import os
import subprocess
import sys
D=Path(__file__).resolve().parent
base='/home/user/InfoSystemic/AI-Server/serving/fleet-0912-ctx/glm-fix:/home/user/InfoSystemic/AI-Server/serving/fleet-0912-ctx/unary-lib:/home/user/InfoSystemic/AI-Server/serving/fleet-0903/results/glm-flash-q8-r8-ordered-k-0908/private-cpu:/home/user/InfoSystemic/AI-Server/engines/llama.cpp-glm5n-goal-0904/validated-chunk16-bin'
for name,on in [('production',0),('candidate-off',0),('candidate-on',1)]:
    env=dict(os.environ,LD_LIBRARY_PATH=(str(D/'build')+':' if name!='production' else '')+base,GGML_CPU_GLM_POOL_FUSION=str(on),GGML_CPU_SOFTMAX_POOL_FUSION='1',GGML_CPU_PARALLEL_UNARY='4096',GGML_CPU_SINGLE_TASK_MAX_ELEMENTS='4096',GGML_CPU_PARALLEL_COPY='1',OMP_NUM_THREADS=os.environ.get('TEST_THREADS','3'))
    with (D/f'test-{name}.log').open('w') as log:
        p=subprocess.run(['taskset','-c','124-127',str(D/'test_pool'),str(D/f'{name}.bin'),str(on),os.environ.get('TEST_THREADS','3')],env=env,stdout=log,stderr=subprocess.STDOUT)
    print(name,p.returncode)
    if p.returncode:
        print((D/f'test-{name}.log').read_text()[-5000:])
        sys.exit(p.returncode)
    log_text=(D/f'test-{name}.log').read_text()
    loaded_cpu=Path(next(line.split(' ',1)[1] for line in log_text.splitlines() if line.startswith('CPU_LIBRARY '))).resolve()
    loaded_base=Path(next(line.split(' ',1)[1] for line in log_text.splitlines() if line.startswith('BASE_LIBRARY '))).resolve()
    expected_cpu=(D.parent/'unary-lib/libggml-cpu.so.0' if name=='production' else D/'build/libggml-cpu.so.0').resolve()
    expected_base=(D.parent/'glm-fix/libggml-base.so.0').resolve()
    assert loaded_cpu==expected_cpu, (loaded_cpu,expected_cpu)
    assert loaded_base==expected_base, (loaded_base,expected_base)
    data=(D/f'{name}.bin').read_bytes()
    print(len(data),hashlib.sha256(data).hexdigest())
    if name=='production': reference=data
    elif data!=reference:
        first=next(i for i,(a,b) in enumerate(zip(reference,data)) if a!=b)
        print('MISMATCH first byte',first)
        sys.exit(3)
print('PASS: bit-identical production, candidate-off, candidate-on; expected fusion/fallback counts verified.')
