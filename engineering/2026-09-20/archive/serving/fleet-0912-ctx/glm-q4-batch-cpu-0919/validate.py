#!/usr/bin/env python3
"""Run isolated candidate output/engagement checks; leaves production running."""
import argparse,array,hashlib,json,os,subprocess,time,urllib.request
from pathlib import Path
D=Path(__file__).resolve().parent
B=Path('/home/user/sr950-strategy/codex-bench-0919')
P=D.parent/'glm-cpu-fast-0919'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('suite',choices=['small','real','speed']);ap.add_argument('--check-existing',action='store_true');a=ap.parse_args()
 out=D/('validation-'+a.suite)
 if a.check_existing:assert out.exists()
 else:assert not out.exists();out.mkdir()
 runtime=json.loads((B/'cache-logits-promoted.json').read_text())['runtime']
 assert json.load(urllib.request.urlopen('http://127.0.0.1:18131/health',timeout=5))=={'status':'ok'}
 slots_before=json.load(urllib.request.urlopen('http://127.0.0.1:18131/slots',timeout=5))
 assert not any(r['is_processing'] for r in slots_before)
 assert not Path('/dev/shm/flash-optrace.arm').exists()
 base={k:v for k,v in os.environ.items() if not k.startswith(('GGML_','LLAMA_','OMP_','GOMP_','KMP_'))}
 base.update(runtime['environment'])
 for k in list(base):
  if k.startswith('GGML_CPU_OP_PROFILE'):del base[k]
 base['GGML_CPU_X16_Q4_BATCH_PROBE']='0' if a.suite=='speed' else '1'
 report={'suite':a.suite,'production_changed':False,'variants':{},'comparisons':{},'completed':False,'unchanged_fused_path':'Unclamped MoE gate/up fusion does not use the new ordinary-dispatch batching; one-thread graphs skip fusion and do use ordinary dispatch.','rechecking_saved_outputs':a.check_existing}
 path=out/'report.json'
 variants=[('production',P,False),('off',D,False),('on',D,True)] if a.suite!='speed' else [('off1',D,False),('on1',D,True),('off2',D,False),('on2',D,True)]
 try:
  for name,library,enabled in variants:
   dest=out/name
   if not a.check_existing:dest.mkdir()
   env=dict(base);env['LD_LIBRARY_PATH']=str(library)+':'+runtime['environment']['LD_LIBRARY_PATH']
   if enabled:env['GGML_CPU_X16_Q4_BATCH']='1'
   else:env.pop('GGML_CPU_X16_Q4_BATCH',None)
   cmd=['/usr/bin/taskset','-c','0-14',str(D/'expert_check'),str(dest),a.suite]
   print('Start',name,a.suite,flush=True);started=time.monotonic()
   rc=0
   if not a.check_existing:
    with (dest/'rows.jsonl').open('w') as stdout,(dest/'stderr.log').open('w') as stderr:
     proc=subprocess.Popen(cmd,env=env,stdout=stdout,stderr=stderr)
     while True:
      try:rc=proc.wait(timeout=30);break
      except subprocess.TimeoutExpired:print(name,'active',round(time.monotonic()-started),'seconds',flush=True)
   rows=[json.loads(line) for line in (dest/'rows.jsonl').read_text().splitlines() if line.startswith('{')]
   entry={'returncode':rc,'seconds':time.monotonic()-started,'command':cmd,'library_sha256':sha(library/'libggml-cpu.so.0.22.0'),'rows':rows[1:],'identity':rows[0] if rows else None}
   report['variants'][name]=entry;path.write_text(json.dumps(report,indent=2)+'\n')
   assert rc==0, f'{name} harness failure; see {dest}/stderr.log'
   assert Path(rows[0]['cpu_library']).resolve()==(library/'libggml-cpu.so.0.22.0').resolve(), 'wrong library'
   expected_cases=280 if a.suite=='small' else 80 if a.suite=='real' else 16
   assert len(rows)-1==expected_cases,(name,len(rows)-1,expected_cases)
   for row in rows[1:]:
    eligible=enabled and row['type']=='q4_K' and row['rows']%16==0 and (row['mode']!='unclamped' or row['threads']==1) and a.suite!='speed'
    counts=[0]*row['experts']
    for t in range(row['tokens']):
     selected=[]
     for j in range(8):
      e=0 if j==0 else row['experts']-1 if j==1 else (j*29+250+(t*31 if row['pattern'] else 0))%row['experts']
      while e in selected:e=(e+1)%row['experts']
      selected.append(e);counts[e]+=1
    expected=(sum(c//3+(c%3==2) for c in counts)*((row['rows']+63)//64)*2*(row['repeats']+1)) if eligible else 0
    assert row['batch_calls']==expected,(name,row['key'],row['batch_calls'],expected)
   print('Completed',name,'cases',len(rows)-1,'with expected activation counts',flush=True)
  reference=variants[0][0]
  refs={r['key']:r for r in report['variants'][reference]['rows']}
  all_equal=True
  for name,_,_ in variants[1:]:
   comparisons=[]
   for row in report['variants'][name]['rows']:
    ref=refs[row['key']]
    x=(out/reference/(row['key']+'.f32')).read_bytes();y=(out/name/(row['key']+'.f32')).read_bytes()
    equal=x==y and row['weight_hash']==ref['weight_hash'];all_equal &= equal
    comp={'key':row['key'],'bit_equal':equal,'time_ratio':row['median_ms']/ref['median_ms']}
    if not equal:
     xa=array.array('f');xa.frombytes(x);ya=array.array('f');ya.frombytes(y)
     comp['differing_values']=sum(u!=v for u,v in zip(xa,ya));comp['max_abs']=max(abs(u-v) for u,v in zip(xa,ya))
    comparisons.append(comp)
   report['comparisons'][name]=comparisons
  slots_after=json.load(urllib.request.urlopen('http://127.0.0.1:18131/slots',timeout=5))
  report.update(production_task_ids_before=[r['id_task'] for r in slots_before],production_task_ids_after=[r['id_task'] for r in slots_after],production_idle_after=not any(r['is_processing'] for r in slots_after))
  if a.suite=='speed' and not a.check_existing:assert report['production_idle_after'] and report['production_task_ids_before']==report['production_task_ids_after'], 'Production inference overlapped timing'
  report.update(completed=True,all_output_bit_equal=all_equal,all_activation_counts_expected=True)
  assert all_equal,'candidate differs from production; inspect report'
  print('PASS',a.suite,'all exact-output and engagement checks.',flush=True)
 except BaseException as exc:report['error']=repr(exc);raise
 finally:path.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
