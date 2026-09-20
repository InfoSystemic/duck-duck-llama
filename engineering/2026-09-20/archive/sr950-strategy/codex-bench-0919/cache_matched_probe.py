#!/usr/bin/env python3
"""Compare generated-state and prefilled-state cache append distributions on identical text."""
import argparse,hashlib,json,math,random,threading,time,urllib.request
from pathlib import Path
from capture_runtime import snapshot
HERE=Path(__file__).resolve().parent

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--pid',type=int,required=True);ap.add_argument('--tag',required=True);a=ap.parse_args()
 out=HERE/f'cache-matched-{a.tag}.json';assert not out.exists()
 runtime=snapshot(a.pid,18131)
 assert runtime['health']=={'status':'ok'} and not any(s['is_processing'] for s in runtime['slots'])
 assert not Path('/dev/shm/flash-optrace.arm').exists()
 old=json.loads((HERE/'cache-logits-promoted.json').read_text())
 assert runtime['mapped_libraries_sha256']==old['runtime']['mapped_libraries_sha256']
 report={'runtime':runtime,'purpose':__doc__,'cases':{},'completed':False}
 done=threading.Event()
 def heartbeat():
  started=time.monotonic()
  while not done.wait(30):print('Matched cache probe active',round(time.monotonic()-started),'seconds',flush=True)
 def comp(name,prompt,cache,slot,n):
  req=urllib.request.Request('http://127.0.0.1:18131/completion',data=json.dumps({'prompt':prompt,'n_predict':n,
   'temperature':0,'seed':42,'cache_prompt':cache,'id_slot':slot,'stream':False,'samplers':['temperature'],
   'n_probs':20,'post_sampling_probs':False}).encode(),headers={'Content-Type':'application/json'})
  started=time.monotonic()
  with urllib.request.urlopen(req,timeout=600) as r:result=json.load(r)
  assert result['id_slot']==slot
  row={k:v for k,v in result.items() if k not in ['prompt','generation_settings']}
  row['generation_settings']={k:result.get('generation_settings',{}).get(k) for k in ['temperature','seed','samplers','n_probs','post_sampling_probs']}
  row['wall_seconds']=time.monotonic()-started;row['input_text_sha256']=hashlib.sha256(prompt.encode()).hexdigest()
  report['cases'][name]=row;out.write_text(json.dumps(report,indent=2)+'\n')
  print(name,repr(row['content']),json.dumps(row['timings']),flush=True)
  return row
 rng=random.Random(7)
 words='alpha beta gamma delta memory bandwidth socket kernel scheduler page node thread barrier expert router token cache state vector matrix'.split()
 p='Summarize the following list of words and then continue the story:\n'+' '.join(rng.choice(words) for _ in range(3200))+'\nThe story begins:'
 continuation=old['cases']['mtp_initial']['content'];prefix=p+continuation;append=prefix+'\nThen suddenly'
 def distribution(row):return {v['id']:v for v in row['completion_probabilities'][0]['top_logprobs']}
 def comparison(left,right):
  l,r=distribution(left),distribution(right);common=sorted(set(l)&set(r))
  return {'first_token_equal':left['content']==right['content'],
   'top20_exactly_equal':left['completion_probabilities']==right['completion_probabilities'],
   'shared_top20_count':len(common),
   'largest_shared_probability_delta':max(abs(math.exp(l[k]['logprob'])-math.exp(r[k]['logprob'])) for k in common),
   'left_top':[{'token':v['token'],'probability':math.exp(v['logprob'])} for v in list(l.values())[:5]],
   'right_top':[{'token':v['token'],'probability':math.exp(v['logprob'])} for v in list(r.values())[:5]]}
 try:
  threading.Thread(target=heartbeat,daemon=True).start()
  teacher=comp('prefilled_fixed_continuation',prefix,False,0,1)
  assert teacher['timings'].get('draft_n',0)==0
  tc=comp('prefilled_append_cached',append,True,0,1)
  fresh=comp('append_fresh',append,False,1,1)
  generated=comp('generated_continuation',p,False,0,48)
  assert generated['content']==continuation,'Generated reference changed; matched experiment invalid'
  assert generated['timings'].get('draft_n',0)>0
  gc=comp('generated_append_cached',append,True,0,1)
  repeat=comp('append_fresh_repeat',append,False,1,1)
  assert tc['input_text_sha256']==gc['input_text_sha256']==fresh['input_text_sha256']==repeat['input_text_sha256']
  assert tc['timings']['cache_n']>0 and gc['timings']['cache_n']>0
  assert fresh['timings']['cache_n']==repeat['timings']['cache_n']==0
  report['same_cached_prefix_length']=tc['timings']['cache_n']==gc['timings']['cache_n']
  report['comparisons']={'prefilled_vs_fresh':comparison(tc,fresh),'generated_vs_fresh':comparison(gc,fresh),
   'generated_vs_prefilled':comparison(gc,tc),'fresh_repeat':comparison(fresh,repeat),
   'generated_vs_saved':comparison(gc,old['cases']['mtp_append_cached'])}
  final=snapshot(a.pid,18131);assert final['proc_start_ticks']==runtime['proc_start_ticks']
  assert final['mapped_libraries_sha256']==runtime['mapped_libraries_sha256']
  report['final_runtime']=final;report['completed']=True
 except BaseException as exc:report['error']=repr(exc);raise
 finally:done.set();out.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
