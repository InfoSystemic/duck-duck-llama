#!/usr/bin/env python3
from pathlib import Path
import datetime,hashlib,json,os,re,subprocess
D=Path(__file__).resolve().parent
base='/home/user/InfoSystemic/AI-Server/serving/fleet-0912-ctx/glm-fix:/home/user/InfoSystemic/AI-Server/serving/fleet-0912-ctx/unary-lib:/home/user/InfoSystemic/AI-Server/serving/fleet-0903/results/glm-flash-q8-r8-ordered-k-0908/private-cpu:/home/user/InfoSystemic/AI-Server/engines/llama.cpp-glm5n-goal-0904/validated-chunk16-bin'
records=[]
for threads in [1,3,4]:
    reference=None
    for name,on in [('production',0),('candidate-on',1),('production-repeat',0)]:
        env=dict(os.environ,LD_LIBRARY_PATH=(str(D/'build')+':' if on else '')+base,GGML_CPU_CPY_FLAT=str(on),GGML_CPU_PARALLEL_COPY='1',GGML_CPU_PARALLEL_UNARY='4096',GGML_CPU_SINGLE_TASK_MAX_ELEMENTS='4096',OMP_NUM_THREADS=str(threads))
        key=f'bench-t{threads}-{name}'
        p=subprocess.run(['taskset','-c','124-127',str(D/'test_copy'),str(D/f'{key}.bin'),str(on),str(threads),'bench'],env=env,capture_output=True,text=True)
        (D/f'{key}.log').write_text(p.stdout+p.stderr)
        if p.returncode: raise RuntimeError(p.stdout+p.stderr)
        loaded=Path(next(x.split(' ',1)[1] for x in p.stderr.splitlines() if x.startswith('CPU_LIBRARY '))).resolve()
        expected=(D/'build/libggml-cpu.so.0' if on else D.parent/'unary-lib/libggml-cpu.so.0').resolve()
        assert loaded==expected,(loaded,expected)
        data=(D/f'{key}.bin').read_bytes()
        if reference is None:reference=data
        assert data==reference,'microbench byte parity failed'
        line=next(x for x in p.stdout.splitlines() if x.startswith('BENCH '))
        values={k:float(v) for k,v in re.findall(r'(median_us|p10|p90)=([0-9.]+)',line)}
        record={'threads':threads,'arm':name,'output_sha256':hashlib.sha256(data).hexdigest(),**values}
        records.append(record); print(record,flush=True)
result={'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'affinity':'124-127','scope':'one CPU backend; actual 2MiB state-copy graph, input preparation excluded; 100 iterations, first10 discarded; not model throughput','library_sha256':hashlib.sha256((D/'build/libggml-cpu.so.0.22.0').read_bytes()).hexdigest(),'records':records}
(D/'microbench.json').write_text(json.dumps(result,indent=2)+'\n')
