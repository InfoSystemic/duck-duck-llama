#!/usr/bin/env python3
from pathlib import Path
import hashlib,os,subprocess,json,datetime
D=Path(__file__).resolve().parent
base='/home/user/InfoSystemic/AI-Server/serving/fleet-0912-ctx/glm-fix:/home/user/InfoSystemic/AI-Server/serving/fleet-0912-ctx/unary-lib:/home/user/InfoSystemic/AI-Server/serving/fleet-0903/results/glm-flash-q8-r8-ordered-k-0908/private-cpu:/home/user/InfoSystemic/AI-Server/engines/llama.cpp-glm5n-goal-0904/validated-chunk16-bin'
threads=os.environ.get('TEST_THREADS','3')
records=[];references={}
for name,pool,copy in [('production',0,0),('both-off',0,0),('pool-only',1,0),('copy-only',0,1),('both-on',1,1)]:
    env=dict(os.environ,LD_LIBRARY_PATH=(str(D/'build')+':' if name!='production' else '')+base,GGML_CPU_GLM_POOL_FUSION=str(pool),GGML_CPU_CPY_FLAT=str(copy),GGML_CPU_SOFTMAX_POOL_FUSION='1',GGML_CPU_PARALLEL_UNARY='4096',GGML_CPU_PARALLEL_COPY='1',GGML_CPU_SINGLE_TASK_MAX_ELEMENTS='4096',OMP_NUM_THREADS=threads)
    for kind,enabled in [('pool',pool),('copy',copy)]:
        key=f'{kind}-{name}'
        p=subprocess.run(['taskset','-c','124-127',str(D/f'test_{kind}'),str(D/f'{key}.bin'),str(enabled),threads],env=env,capture_output=True,text=True)
        (D/f'test-{key}.log').write_text(p.stdout+p.stderr)
        if p.returncode:raise RuntimeError(p.stdout+p.stderr)
        loaded_cpu=Path(next(x.split(' ',1)[1] for x in p.stderr.splitlines() if x.startswith('CPU_LIBRARY '))).resolve()
        loaded_base=Path(next(x.split(' ',1)[1] for x in p.stderr.splitlines() if x.startswith('BASE_LIBRARY '))).resolve()
        assert loaded_cpu==(D.parent/'unary-lib/libggml-cpu.so.0' if name=='production' else D/'build/libggml-cpu.so.0').resolve()
        assert loaded_base==(D.parent/'glm-fix/libggml-base.so.0').resolve()
        data=(D/f'{key}.bin').read_bytes()
        if name=='production': references[kind]=data
        assert data==references[kind],key
        record={'kind':kind,'arm':name,'threads':int(threads),'bytes':len(data),'output_sha256':hashlib.sha256(data).hexdigest(),'result':'PASS (bitwise parity, expected path counts, mapped libraries)'}
        print(record,flush=True); records.append(record)
result={'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'library_sha256':hashlib.sha256((D/'build/libggml-cpu.so.0.22.0').read_bytes()).hexdigest(),'records':records,'full_model_test':'not run','performance':'not measured'}
(D/f'validation-threads-{threads}.json').write_text(json.dumps(result,indent=2)+'\n')
print('PASS independent switches in all four states; production parity for both harnesses')
