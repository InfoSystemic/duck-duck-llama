#!/usr/bin/env python3
"""Private Q4/Q5 fusion benchmark with an expected 15-20 minute outage and automatic restore.
No production configuration changes or automatic promotion. No overlapping model loads.
"""
import argparse,hashlib,json,re,signal,subprocess,time
from pathlib import Path
from capture_runtime import snapshot
from pool_window import pid_for,wait_dead,wait_healthy,open_port,request
from thread_window import native,codex,idle
from summarize_optrace import HEADER,ROW

HERE=Path(__file__).resolve().parent
FLEET=Path('/home/user/InfoSystemic/AI-Server/serving/fleet-0912-ctx')
CAND=FLEET/'glm-kquant-clamp-0919/libggml-cpu.so.0.22.0'
PROD_LIB=FLEET/'glm-cpu-fast-0919/libggml-cpu.so.0.22.0'
LAUNCHER=FLEET/'launch-glm-flash-native.sh'
PRODUCTION='glm53-flash-production.service'
UNIT='glm53-flash-kquant-test-0919.service'
PORT=18141
REPORT=HERE/'kquant-window-report.json'
MANIFEST=HERE/'kquant-window-manifest.json'
SERVER_LOG=HERE/'kquant-candidate-server.log'

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj):Path(p).write_text(json.dumps(obj,indent=2)+'\n')
def run(cmd):return subprocess.run(cmd,check=True,cwd=HERE)
def profile_evidence(log):
 current={};graphs=[]
 for line in log.splitlines():
  if m:=HEADER.search(line):
   ix,cpu,ptr,n,total,bar=m.groups();g={'index':int(ix),'cpu':int(cpu),'nodes':int(n),'rows':[]}
   graphs.append(g);current[(cpu,ptr)]=g
  elif m:=ROW.search(line):
   cpu,ptr,node,op,ms,bar,name,typ,shape,src=m.groups()
   if (cpu,ptr) in current:current[(cpu,ptr)]['rows'].append({'node':int(node),'op':op,'name':name,'type':typ,'source':src,'ms':float(ms)})
 evidence=[]
 for graph in graphs:
  if graph['nodes']<1000:continue
  gates={};ups={};downs={}
  for row in graph['rows']:
   if row['op']!='MUL_MAT_ID':continue
   m=re.fullmatch(r'blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight',row['source'] or '')
   if m:
    layer,part=m.groups();{'gate':gates,'up':ups,'down':downs}[part][int(layer)]=row
  kquant={layer:r for layer,r in gates.items() if r['type'] in ['q4_K','q5_K']}
  evidence.append({'index':graph['index'],'cpu':graph['cpu'],'nodes':graph['nodes'],
   'q4_gate_layers':[k for k,r in kquant.items() if r['type']=='q4_K'],
   'q5_gate_layers':[k for k,r in kquant.items() if r['type']=='q5_K'],
   'separate_up_layers':sorted(ups),'down_layers':sorted(downs),
   'all_42_gate_layers_present':len(kquant)==42,
   'all_42_down_layers_present':len(downs)==42,
   'no_separate_expert_up':not ups})
 assert len(graphs)==8 and 'CPU_OP_PROFILE complete count=8' in log, 'Incomplete bounded profile'
 assert evidence, 'No target decode graph captured'
 assert any(e['all_42_gate_layers_present'] and e['all_42_down_layers_present'] and e['no_separate_expert_up'] for e in evidence), 'No complete target expert fusion evidence'
 return {'method':'Bounded operation trace; skipped fused up operations are absent while all 42 gate/down operations remain. Standalone counters separately verified exact dispatch.',
  'graphs':evidence,'timings_excluded_from_benchmarks':True}

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--production-pid',required=True,type=int);ap.add_argument('--check-only',action='store_true');a=ap.parse_args()
 manifest=json.loads(MANIFEST.read_text())
 for name,sha in manifest['reviewed_files'].items():assert digest(name)==sha, 'Changed reviewed file: '+name
 assert digest(CAND)==manifest['candidate_sha256'] and digest(PROD_LIB)==manifest['production_sha256']
 for suite in ['small','real','speed']:
  r=json.loads((CAND.parent/('validation-'+suite)/'report.json').read_text())
  assert r['completed'] and r['all_output_bit_equal'] and r['all_activation_counts_expected']
 assert pid_for(PRODUCTION)==a.production_pid and not open_port(PORT)
 original=snapshot(a.production_pid,18131);idle(18131)
 assert original['mapped_libraries_sha256'][str(PROD_LIB)]==manifest['production_sha256']
 assert original['environment']['GGML_CPU_NUMA_THREADS']=='15'
 assert 'GGML_CPU_KQUANT_CLAMP_FUSION' not in original['environment']
 if a.check_only:
  print(json.dumps({'plan_checked':True,'production_pid':a.production_pid,'production_changed':False,'candidate_sha256':manifest['candidate_sha256'],'expected_outage_minutes':[15,20]},indent=2));return
 assert not REPORT.exists() and not SERVER_LOG.exists(), 'Refuse to overwrite an experiment'
 report={'production_pid_before':a.production_pid,'candidate_sha256':manifest['candidate_sha256'],'promoted':False,
  'known_cache_consistency_issue_unresolved':True,'native_candidate':[],'codex_candidate':[]}
 save(HERE/'kquant-window-original.json',original)
 stopped=False
 def checkpoint():save(REPORT,report)
 def signal_stop(signum,frame):raise KeyboardInterrupt(f'Signal {signum}; restore production')
 signal.signal(signal.SIGTERM,signal_stop)
 try:
  report['native_before']=native(18131,'kquant-production-before');assert report['native_before']['all_parity'];checkpoint()
  cold_before=codex(18131,'kquant-production-cold-before',False)
  warm_before=codex(18131,'kquant-production-warm-before',True)
  report['codex_before']={'cold':cold_before,'warm':warm_before};checkpoint()
  idle(18131);assert pid_for(PRODUCTION)==a.production_pid
  assert snapshot(a.production_pid,18131)['proc_start_ticks']==original['proc_start_ticks']
  stopped=True;report['outage_started_epoch']=time.time();checkpoint()
  run(['systemctl','--user','stop',PRODUCTION]);wait_dead(a.production_pid)
  assert not open_port(18131)
  available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))
  assert available>380*1024*1024,'Insufficient headroom for private model'
  run(['systemd-run','--user','--unit='+UNIT,'--collect','--property=MemoryMax=380G','--property=MemorySwapMax=0',
   '--property=TimeoutStopSec=120','--property=RuntimeMaxSec=1800',
   '--property=StandardOutput=append:'+str(SERVER_LOG),'--property=StandardError=append:'+str(SERVER_LOG),
   '--setenv=PORT='+str(PORT),'--setenv=LIB_PREPEND='+str(CAND.parent),
   '--setenv=GGML_CPU_GLM_POOL_FUSION=1','--setenv=GGML_CPU_CPY_FLAT=1','--setenv=GGML_CPU_KQUANT_CLAMP_FUSION=1',str(LAUNCHER)])
  wait_healthy(PORT,UNIT)
  current=snapshot(pid_for(UNIT),PORT);save(HERE/'kquant-window-candidate.json',current)
  assert current['mapped_libraries_sha256'].get(str(CAND))==manifest['candidate_sha256']
  expected=list(original['command']);expected[expected.index('--port')+1]=str(PORT)
  assert current['command']==expected,'Unexpected command-line change'
  oldlibs={Path(k).name:v for k,v in original['mapped_libraries_sha256'].items() if 'libggml-cpu' not in k}
  newlibs={Path(k).name:v for k,v in current['mapped_libraries_sha256'].items() if 'libggml-cpu' not in k}
  assert oldlibs==newlibs,'Non-CPU library change'
  env=dict(original['environment']);env.update(GGML_CPU_KQUANT_CLAMP_FUSION='1')
  env['LD_LIBRARY_PATH']=str(CAND.parent)+':'+env['LD_LIBRARY_PATH'].split(':',1)[1]
  assert current['environment']==env,'Unexpected inference environment change'
  report['candidate_pid']=current['pid'];checkpoint()
  for i in range(2):
   row=native(PORT,f'kquant-candidate-{i}');report['native_candidate'].append(row);checkpoint()
   assert row['all_parity'],'Reject candidate: native output changed'
  cold=codex(PORT,'kquant-candidate-cold',False)
  assert cold['output_text']==cold_before['output_text'],'Reject candidate: cold Codex output changed'
  report['codex_candidate_cold']=cold;checkpoint()
  for i in range(2):
   row=codex(PORT,f'kquant-candidate-warm-{i}',True)
   report['codex_candidate'].append(row);checkpoint()
   assert row['output_text']==warm_before['output_text'],'Reject candidate: cached Codex output changed'
  run(['python3','-u',str(HERE/'stateful_gate.py'),'--endpoint',f'http://127.0.0.1:{PORT}',
   '--tag','kquant','--reference',str(HERE/'stateful-production-control.json')])
  stateful=json.loads((HERE/'stateful-kquant.json').read_text());assert stateful['regression_passed']
  report['stateful_regression_passed']=True;report['stateful_intrinsic_consistency']=stateful['passed'];checkpoint()
  idle(PORT);offset=SERVER_LOG.stat().st_size
  run(['python3','-u',str(HERE/'profile_once.py'),'--pid',str(current['pid']),'--port',str(PORT),'--tag','kquant'])
  text=SERVER_LOG.read_bytes()[offset:].decode(errors='replace')
  trace='\n'.join(line for line in text.splitlines() if 'CPU_OP_PROFILE' in line)+'\n'
  (HERE/'profile-kquant.optrace.log').write_text(trace)
  report['full_model_engagement']=profile_evidence(trace);checkpoint()
  run(['python3',str(HERE/'summarize_optrace.py'),str(HERE/'profile-kquant.optrace.log'),'--output',str(HERE/'profile-kquant.optrace-summary.json')])
  report['experiment_completed']=True;checkpoint()
 except BaseException as exc:report['error']=repr(exc);checkpoint();raise
 finally:
  if stopped:
   print('Stopping private candidate and restoring approved production',flush=True)
   candidate_pid=pid_for(UNIT);subprocess.run(['systemctl','--user','stop',UNIT],check=False)
   if candidate_pid:wait_dead(candidate_pid)
   assert not open_port(PORT),'Private model still live; refusing overlap'
   live=pid_for(PRODUCTION)
   if live and live!=a.production_pid:raise RuntimeError('Unexpected production process; manual identity check needed')
   if not live:
    assert not open_port(18131),'Unexpected production listener'
    available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))
    assert available>380*1024*1024,'Insufficient headroom for production restore'
    run(['systemctl','--user','start',PRODUCTION])
   wait_healthy(18131,PRODUCTION);restored=snapshot(pid_for(PRODUCTION),18131)
   assert restored['mapped_libraries_sha256']==original['mapped_libraries_sha256']
   assert restored['environment']==original['environment'] and restored['command']==original['command']
   save(HERE/'kquant-window-restored.json',restored)
   report.update(restored_healthy=True,restored_pid=restored['pid'],outage_seconds=time.time()-report['outage_started_epoch']);checkpoint()
   print('Approved production restored, PID',restored['pid'],flush=True)
  checkpoint()
 report['native_after']=native(18131,'kquant-production-after');assert report['native_after']['all_parity'];checkpoint()
 cold_after=codex(18131,'kquant-production-cold-after',False)
 warm_after=codex(18131,'kquant-production-warm-after',True)
 assert cold_after['output_text']==cold_before['output_text'] and warm_after['output_text']==warm_before['output_text']
 report['codex_after']={'cold':cold_after,'warm':warm_after}
 report['gate']='Completed with approved production restored; no promotion; pre-existing cache inconsistency retained'
 checkpoint();print(report['gate'],flush=True)
if __name__=='__main__':main()
