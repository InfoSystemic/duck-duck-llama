#!/usr/bin/env python3
import hashlib,json,os,subprocess,time,urllib.request,sys
from pathlib import Path
p=Path(__file__).resolve().parent
cpu=p.parent/'glm-cpu-fast-0919'
base=p.parent/'glm-fix'
suite=sys.argv[1];assert suite in ('correctness','speed')
tag=sys.argv[2] if len(sys.argv)>2 else suite
assert tag.replace('-','').isalnum()
report=p/f'{tag}.run.json';assert not report.exists(),report
slots=json.load(urllib.request.urlopen('http://127.0.0.1:18131/slots',timeout=10))
assert all(not x['is_processing'] for x in slots),'Production is busy'
identities={str(x.resolve()):hashlib.sha256(x.read_bytes()).hexdigest() for x in [cpu/'libggml-cpu.so.0',base/'libggml-base.so.0',p/'check']}
assert identities[str((cpu/'libggml-cpu.so.0').resolve())]=='4793379e6894a9286168f79c4f323985388ec0b0ac65e54013dcd2957488af1f'
assert identities[str((base/'libggml-base.so.0').resolve())]=='598563d018482c8daa9fca62e8d5d9a53aa63e36cda9b9f978ffb972b9579b92'
env=os.environ.copy();env.update(LD_LIBRARY_PATH=f'{cpu}:{base}',GGML_CPU_X16_Q4_K='1',OMP_NUM_THREADS='1')
cmd=['taskset','-c','2',str(p/'check'),suite]
t=time.time()
with (p/f'{tag}.jsonl').open('w') as out,(p/f'{tag}.stderr').open('w') as err:
 r=subprocess.run(cmd,env=env,stdout=out,stderr=err,timeout=180)
state={'command':cmd,'identities':identities,'returncode':r.returncode,'elapsed_seconds':time.time()-t,'start_unix':t,'cpu':2,'single_threaded':True,'production_idle_before':True,'production_slots_after':json.load(urllib.request.urlopen('http://127.0.0.1:18131/slots',timeout=10))}
report.write_text(json.dumps(state,indent=2)+'\n')
print(json.dumps({k:state[k] for k in ['returncode','elapsed_seconds']}))
if r.returncode==0:
 for l in (p/f'{tag}.jsonl').read_text().splitlines():
  obj=json.loads(l);obj.pop('samples',None);print(json.dumps(obj))
raise SystemExit(r.returncode)
