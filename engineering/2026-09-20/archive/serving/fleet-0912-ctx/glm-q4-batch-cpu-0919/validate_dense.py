#!/usr/bin/env python3
"""Exact dense graph comparisons for the isolated candidate; no service changes."""
import hashlib,json,os,subprocess,time,urllib.request,sys
from pathlib import Path
D=Path(__file__).resolve().parent
B=Path('/home/user/sr950-strategy/codex-bench-0919')
P=D.parent/'glm-cpu-fast-0919'
recheck=sys.argv[1:]==['--check-existing'];assert not sys.argv[1:] or recheck
out=D/'validation-dense'
if recheck:assert out.exists()
else:assert not out.exists();out.mkdir()
assert not any(x['is_processing'] for x in json.load(urllib.request.urlopen('http://127.0.0.1:18131/slots',timeout=10)))
runtime=json.loads((B/'cache-logits-promoted.json').read_text())['runtime']
base={k:v for k,v in os.environ.items() if not k.startswith(('GGML_','LLAMA_','OMP_','GOMP_','KMP_'))};base.update(runtime['environment'])
for k in list(base):
 if k.startswith('GGML_CPU_OP_PROFILE'):del base[k]
base['GGML_CPU_X16_Q4_BATCH_PROBE']='1'
report={'production_changed':False,'variants':{},'comparisons':{},'completed':False,'rechecking_saved_outputs':recheck}
try:
 for name,library,on in [('production',P,False),('off',D,False),('on',D,True)]:
  dest=out/name
  if not recheck:dest.mkdir()
  env=dict(base);env['LD_LIBRARY_PATH']=str(library)+':'+runtime['environment']['LD_LIBRARY_PATH']
  env['GGML_CPU_X16_Q4_BATCH']='1' if on else '0'
  cmd=['taskset','-c','0-14',str(D/'dense_check'),str(dest)]
  print('Start',name,flush=True);t=time.monotonic()
  rc=0
  if not recheck:
   with (dest/'rows.jsonl').open('w') as stdout,(dest/'stderr.log').open('w') as stderr:
    proc=subprocess.Popen(cmd,env=env,stdout=stdout,stderr=stderr)
    while True:
     try:rc=proc.wait(timeout=30);break
     except subprocess.TimeoutExpired:print(name,'active',round(time.monotonic()-t),'seconds',flush=True)
  rows=[json.loads(l) for l in (dest/'rows.jsonl').read_text().splitlines()]
  report['variants'][name]={'returncode':rc,'seconds':time.monotonic()-t,'identity':rows[0] if rows else None,'rows':rows[1:],'library_sha256':hashlib.sha256((library/'libggml-cpu.so.0.22.0').read_bytes()).hexdigest()}
  (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
  assert rc==0,(name,rc)
  assert Path(rows[0]['cpu_library']).resolve()==(library/'libggml-cpu.so.0.22.0').resolve()
  assert len(rows)-1==864,(name,len(rows)-1)
  for row in rows[1:]:
   eligible=on and row['type']=='q4_K' and not (row['mode']=='fused' and row['planes']*row['broadcast']==1 and row['threads']>1)
   chunk=((row['rows']*row['planes']+row['threads']*2-1)//(row['threads']*2)+15)//16*16
   chunk=min(row['rows'],max(int(base.get('GGML_CPU_X16_CHUNK_MIN','16')),min(int(base.get('GGML_CPU_X16_CHUNK_MAX','64')),chunk)))
   expected=((row['rows']+chunk-1)//chunk)*row['planes']*row['broadcast']*(row['tokens']//3+(row['tokens']%3==2))*(2 if row['mode']=='fused' else 1)*(row['repeats']+1) if eligible else 0
   assert row['batch_calls']==expected,(name,row['key'],row['batch_calls'],expected)
  print('Completed',name,'864 exact engagement checks',flush=True)
 refs={r['key']:r for r in report['variants']['production']['rows']}
 for name in ['off','on']:
  checks=[]
  for row in report['variants'][name]['rows']:
   key=row['key'];ref=refs[key]
   same=(out/'production'/f'{key}.f32').read_bytes()==(out/name/f'{key}.f32').read_bytes() and row['weight_hash']==ref['weight_hash']
   checks.append({'key':key,'bit_equal':same})
  report['comparisons'][name]=checks
  assert all(x['bit_equal'] for x in checks),name+' output mismatch'
 report.update(completed=True,all_output_bit_equal=True,all_activation_counts_expected=True)
 print('PASS dense 864 cases: production/off/on exact.',flush=True)
except BaseException as e:report['error']=repr(e);raise
finally:(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
