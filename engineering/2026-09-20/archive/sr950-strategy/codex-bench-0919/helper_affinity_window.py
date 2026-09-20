#!/usr/bin/env python3
"""Bounded live helper affinity comparison; no service restart or config writes."""
import argparse,hashlib,json,os,signal,subprocess,time,urllib.request
from pathlib import Path
from capture_runtime import snapshot
from scheduler_probe import thread_rows
B=Path(__file__).resolve().parent
REPORT=B/'helper-affinity-window.json'
PROBE=B/'appserver-scheduler-production15.json'
CATALOG=Path('/home/user/.codex-glm/model-catalogs/glm-5.3-flash.json')
def save(x):REPORT.write_text(json.dumps(x,indent=2)+'\n')
def start_ticks(tid):return Path(f'/proc/{tid}/stat').read_text().rsplit(')',1)[1].split()[19]
def idle():assert not any(x['is_processing'] for x in json.load(urllib.request.urlopen('http://127.0.0.1:18131/slots',timeout=10)))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--pid',required=True,type=int);ap.add_argument('--check-only',action='store_true');a=ap.parse_args()
 assert not REPORT.exists(),'Refuse to repeat completed or active experiment'
 probe=json.loads(PROBE.read_text());s=probe['scheduler'];assert s['pid']==a.pid and start_ticks(a.pid)==s['proc_start_ticks']
 assert not s['error'];idle();original=snapshot(a.pid,18131)
 assert original['mapped_libraries_sha256']==probe['runtime_before']['mapped_libraries_sha256']
 assert original['environment']==probe['runtime_before']['environment']
 assert original['command']==probe['runtime_before']['command']
 assert not Path('/dev/shm/flash-optrace.arm').exists()
 helpers=sorted(x['tid'] for x in s['threads'] if x['group']=='unbound' and x['system_seconds']>1 and x['tid']!=a.pid)
 assert len(helpers)==8,helpers
 targets=[a.pid]+helpers
 baseline={tid:{'affinity':sorted(os.sched_getaffinity(tid)),'start_ticks':start_ticks(tid)} for tid in targets}
 assert all(x['affinity']==list(range(128)) for x in baseline.values())
 spare={15,31,47,63}
 for x in s['threads']:
  if x['group']=='pinned':assert not (spare & set(os.sched_getaffinity(x['tid'])))
 for cpu in spare:
  # Confirm all four chosen logical CPUs are on distinct physical cores.
  assert Path(f'/sys/devices/system/cpu/cpu{cpu}/topology/core_id').exists()
 assert len({(Path(f'/sys/devices/system/cpu/cpu{c}/topology/physical_package_id').read_text().strip(),Path(f'/sys/devices/system/cpu/cpu{c}/topology/core_id').read_text().strip()) for c in spare})==4
 if a.check_only:
  print(json.dumps({'pid':a.pid,'targets':targets,'spare_cpus':sorted(spare),'production_changed':False,'plan_checked':True}));return
 report={'pid':a.pid,'no_restart':True,'no_config_changes':True,'original_runtime':original,'original_affinities':baseline,'spare_cpus':sorted(spare),'variants':[],'completed':False,'restoration_verified':False}
 save(report)
 def check_identity():assert start_ticks(a.pid)==s['proc_start_ticks']
 def restore():
  check_identity()
  for tid,v in baseline.items():
   assert start_ticks(tid)==v['start_ticks'],('Thread identity changed',tid)
   os.sched_setaffinity(tid,set(v['affinity']))
  assert all(sorted(os.sched_getaffinity(tid))==v['affinity'] for tid,v in baseline.items())
  report['restoration_verified']=True;save(report)
 def stop(signum,frame):raise KeyboardInterrupt(f'Signal {signum}: restore affinity')
 signal.signal(signal.SIGTERM,stop)
 try:
  reference=json.loads((B/'appserver-q4batch-production-warm-after.json').read_text())['output_text']
  for name,pinned in [('before',False),('spare0',True),('spare1',True),('after',False)]:
   idle();check_identity()
   for tid,v in baseline.items():
    assert start_ticks(tid)==v['start_ticks']
    os.sched_setaffinity(tid,spare if pinned else set(v['affinity']))
   report['current_variant']=name;report['temporarily_changed']=pinned;save(report)
   tag='helper-affinity-'+name;assert not (B/f'appserver-{tag}.json').exists()
   first=thread_rows(a.pid);start=time.monotonic()
   cmd=['python3','-u',str(B/'appserver_bench.py'),'--catalog',str(CATALOG),'--tag',tag,'--greedy','--warm']
   print('Start',name,'spare affinity',pinned,flush=True)
   with (B/f'{tag}.log').open('w') as out:
    proc=subprocess.Popen(cmd,cwd=B,stdout=out,stderr=subprocess.STDOUT)
    try:rc=proc.wait(timeout=180)
    finally:
     if proc.poll() is None:
      proc.terminate()
      try:proc.wait(timeout=10)
      except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=10)
   last=thread_rows(a.pid);elapsed=time.monotonic()-start
   actual_affinities={tid:sorted(os.sched_getaffinity(tid)) for tid in targets}
   assert all(mask==sorted(spare if pinned else set(baseline[tid]['affinity'])) for tid,mask in actual_affinities.items()), 'Affinity changed during the comparison'
   assert rc==0,(name,rc)
   result=json.loads((B/f'appserver-{tag}.json').read_text())
   assert result['output_text']==reference and result['marker_pass'],name+' output changed'
   assert result['server_metrics']['generated_tokens']==270 and result['server_metrics']['cached_prompt_tokens']==3994
   delta=[]
   for tid in targets:
    assert tid in first and tid in last
    x,y=first[tid],last[tid]
    delta.append({'tid':tid,'cpu_seconds':(y['run_ns']-x['run_ns'])/1e9,'runqueue_seconds':(y['wait_ns']-x['wait_ns'])/1e9,'system_seconds':(y['system_ticks']-x['system_ticks'])/os.sysconf('SC_CLK_TCK')})
   row={'name':name,'helper_affinity_spare':pinned,'result':result,'helper_deltas':delta,'observer_window_seconds':elapsed,'actual_affinities_after':actual_affinities}
   report['variants'].append(row);save(report)
   print('Completed',name,'decode tok/s',result['server_metrics']['decode_tokens_per_second'],'output exact',flush=True)
 except BaseException as exc:report['error']=repr(exc);save(report);raise
 finally:restore()
 idle();current=snapshot(a.pid,18131)
 for key in ['command','environment','mapped_libraries_sha256','cpu_affinity','proc_start_ticks']:assert current[key]==original[key],key
 report.update(completed=True,current_variant=None,temporarily_changed=False,final_runtime=current)
 save(report);print('All four comparisons complete; all original thread affinities restored.',flush=True)
if __name__=='__main__':main()
