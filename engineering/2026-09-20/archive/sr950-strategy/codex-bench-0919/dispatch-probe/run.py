#!/usr/bin/env python3
import argparse,hashlib,json,os,resource,subprocess,time,urllib.request
from pathlib import Path
P=Path(__file__).resolve().parent;B=P.parent
ap=argparse.ArgumentParser();ap.add_argument('tag');ap.add_argument('--workers',type=int,required=True,choices=[15,16]);ap.add_argument('--spin',type=int,required=True,choices=[0,50,300,20000]);ap.add_argument('--gomp-spin',type=int,choices=[0,100,1000,300000]);ap.add_argument('--omp-wait',choices=['ACTIVE','PASSIVE']);ap.add_argument('--simple-barrier',type=int,choices=[0,1],default=0);ap.add_argument('--reference-tag');ap.add_argument('--loader-team',type=int,choices=[0,128],default=0);ap.add_argument('--size',choices=['smoke','medium','long'],default='smoke');a=ap.parse_args()
assert a.tag.replace('-','').isalnum();out=P/(a.tag+'.json');assert not out.exists()
def slots():return json.load(urllib.request.urlopen('http://127.0.0.1:18131/slots',timeout=5))
s0=slots();assert not any(s['is_processing'] for s in s0)
runtime=json.loads((B/'q4batch-window-final-runtime.json').read_text())
env={k:v for k,v in os.environ.items() if not k.startswith(('GGML_','LLAMA_','OMP_','GOMP_','KMP_'))};env.update(runtime['environment'])
for k in list(env):
 if k.startswith('GGML_CPU_OP_PROFILE'):del env[k]
env['PROBE_LOADER_TEAM']=str(a.loader_team)
env['PROBE_OUTPUT_PREFIX']=str(P/a.tag)
env['GGML_CPU_OMP_SIMPLE_BARRIER']=str(a.simple_barrier)
if a.omp_wait is not None:env['OMP_WAIT_POLICY']=a.omp_wait
if a.gomp_spin is not None:env['GOMP_SPINCOUNT']=str(a.gomp_spin)
env['GGML_CPU_NUMA_THREADS']=str(a.workers);env['GGML_CPU_NUMA_DISPATCH_SPIN_US']=str(a.spin)
shapes={'smoke':['512','1024','2','1','5'],'medium':['4096','8192','32','2','100'],'long':['4096','8192','256','16','61']}
cmd=[str(P/'probe'),*shapes[a.size]]
identities={path:hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in runtime['mapped_libraries_sha256'] if 'libggml' in path}
assert all(digest==runtime['mapped_libraries_sha256'][path] for path,digest in identities.items())
def limits():resource.setrlimit(resource.RLIMIT_AS,(24*1024**3,24*1024**3));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
t=time.monotonic();report={'tag':a.tag,'workers':a.workers,'spin_us':a.spin,'gomp_spin':a.gomp_spin,'omp_wait':a.omp_wait,'simple_barrier':a.simple_barrier,'loader_team':a.loader_team,'size':a.size,'command':cmd,'production_changed':False,'library_identities':identities,'source_sha256':hashlib.sha256((P/'probe.cpp').read_bytes()).hexdigest(),'binary_sha256':hashlib.sha256((P/'probe').read_bytes()).hexdigest(),'virtual_memory_limit_gib':24,'rss_abort_limit_gib':6,'peak_vm_kib':0,'peak_rss_kib':0,'peak_threads':0}
proc=None
try:
 with (P/(a.tag+'.stdout.jsonl')).open('w') as stdout,(P/(a.tag+'.stderr.log')).open('w') as stderr:
  proc=subprocess.Popen(cmd,env=env,stdout=stdout,stderr=stderr,preexec_fn=limits)
  report['pid']=proc.pid
  while proc.poll() is None:
   time.sleep(1)
   assert time.monotonic()-t<180,'standalone timeout'
   status=Path('/proc')/str(proc.pid)/'status'
   if status.exists():
    memory={line.split(':')[0]:int(line.split()[1]) for line in status.read_text().splitlines() if line.startswith(('VmSize:','VmRSS:','Threads:'))}
    report['peak_threads']=max(report['peak_threads'],memory.get('Threads',0))
    report['peak_vm_kib']=max(report['peak_vm_kib'],memory.get('VmSize',0))
    report['peak_rss_kib']=max(report['peak_rss_kib'],memory.get('VmRSS',0))
    assert memory.get('VmRSS',0)<6*1024**2,'standalone resident-memory guard'
   assert not any(s['is_processing'] for s in slots()),'Production became busy; stop standalone test'
  report['returncode']=proc.returncode
 report['rows']=[json.loads(l) for l in (P/(a.tag+'.stdout.jsonl')).read_text().splitlines() if l.startswith('{')]
 assert proc.returncode==0,'standalone failed; inspect stderr'
 assert report['rows'][-1]['completed']
 report['output_sha256']={kind:hashlib.sha256((P/(a.tag+'.'+kind+'.bin')).read_bytes()).hexdigest() for kind in ['target','draft']}
 if a.reference_tag:
  assert a.reference_tag.replace('-','').isalnum()
  reference=json.loads((P/(a.reference_tag+'.json')).read_text())
  assert reference['completed'] and reference['size']==a.size
  assert all((P/(a.tag+'.'+kind+'.bin')).read_bytes()==(P/(a.reference_tag+'.'+kind+'.bin')).read_bytes() for kind in ['target','draft']),'cross-configuration output bytes changed'
  report['reference_bit_parity']=True;report['reference_tag']=a.reference_tag
 report['production_task_ids_before']=[s['id_task'] for s in s0]
 report['production_task_ids_after']=[s['id_task'] for s in slots()]
 assert report['production_task_ids_after']==report['production_task_ids_before'],'Production traffic overlapped'
 report['completed']=True
 print(json.dumps({k:v for k,v in report['rows'][-1].items() if k!='samples_ms'}),flush=True)
except BaseException as e:report['error']=repr(e);raise
finally:
 if proc and proc.poll() is None:
  proc.terminate()
  try:proc.wait(timeout=10)
  except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=10)
 report['elapsed_seconds']=time.monotonic()-t;out.write_text(json.dumps(report,indent=2)+'\n')
