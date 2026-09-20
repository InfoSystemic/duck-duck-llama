#!/usr/bin/env python3
"""Controlled token-array prompts across cache and four-token pooling boundaries."""
import argparse,hashlib,json,math,random,threading,time,urllib.request
from pathlib import Path
from capture_runtime import snapshot
B=Path(__file__).resolve().parent

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--pid',type=int,required=True);ap.add_argument('--tag',required=True);ap.add_argument('--lengths',default='128,512');ap.add_argument('--offsets',default='8,7,6,5');a=ap.parse_args()
 assert a.tag.replace('-','').isalnum()
 lengths=[int(v) for v in a.lengths.split(',')];offsets=[int(v) for v in a.offsets.split(',')]
 assert 1<=len(lengths)<=6 and all(80<=v<=3269 for v in lengths)
 assert 1<=len(offsets)<=8 and all(1<=v<=32 for v in offsets)
 out=B/f'cache-boundary-{a.tag}.json';assert not out.exists()
 def request(path,body=None):
  req=urllib.request.Request('http://127.0.0.1:18131/'+path,data=json.dumps(body).encode() if body is not None else None,headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(req,timeout=300) as r:return json.load(r)
 def idle():
  assert request('health')=={'status':'ok'}
  assert not any(s['is_processing'] for s in request('slots')),'Another inference request is active'
  assert not Path('/dev/shm/flash-optrace.arm').exists()
 idle();runtime=snapshot(a.pid,18131);expected=json.loads((B/'dispatch-probe/final-runtime.json').read_text())
 for k in ['proc_start_ticks','command','environment','mapped_libraries_sha256','cpu_affinity']:assert runtime[k]==expected[k],k
 old=json.loads((B/'cache-logits-promoted.json').read_text())
 rng=random.Random(7);words='alpha beta gamma delta memory bandwidth socket kernel scheduler page node thread barrier expert router token cache state vector matrix'.split()
 original='Summarize the following list of words and then continue the story:\n'+' '.join(rng.choice(words) for _ in range(3200))+'\nThe story begins:'+old['cases']['mtp_initial']['content']+'\nThen suddenly'
 full=request('tokenize',{'content':original,'add_special':True})['tokens']
 assert len(full)==3269 and all(isinstance(t,int) for t in full),(len(full),type(full[0]))
 report={'purpose':__doc__,'runtime':runtime,'lengths':lengths,'offsets':offsets,'full_source_tokens':len(full),'retained_tail_tokens':64,'cases':{},'comparisons':{},'completed':False}
 def save():
  temporary=out.with_suffix('.tmp');temporary.write_text(json.dumps(report,indent=2)+'\n');temporary.replace(out)
 def compare(left,right):
  l=left['completion_probabilities'][0]['top_logprobs'];r=right['completion_probabilities'][0]['top_logprobs'];ld={v['id']:v for v in l};rd={v['id']:v for v in r};shared=ld.keys()&rd.keys()
  return {'first_token_equal':left['content']==right['content'],'top20_exact':left['completion_probabilities']==right['completion_probabilities'],'shared_top20_count':len(shared),'largest_shared_probability_delta':max([abs(math.exp(ld[k]['logprob'])-math.exp(rd[k]['logprob'])) for k in shared],default=None),'cached_prefix_tokens':left['timings']['cache_n'],'cached_prefix_mod4':left['timings']['cache_n']%4,'left_top':[{'token':v['token'],'probability':math.exp(v['logprob'])} for v in l[:3]],'right_top':[{'token':v['token'],'probability':math.exp(v['logprob'])} for v in r[:3]]}
 def comp(name,tokens,cache,slot):
  idle();t=time.monotonic()
  result=request('completion',{'prompt':tokens,'n_predict':1,'temperature':0,'seed':42,'cache_prompt':cache,'id_slot':slot,'stream':False,'samplers':['temperature'],'n_probs':20,'post_sampling_probs':False})
  row={k:v for k,v in result.items() if k not in ['prompt','generation_settings']};row['wall_seconds']=time.monotonic()-t;row['input_tokens']=len(tokens);row['input_token_sha256']=hashlib.sha256(json.dumps(tokens,separators=(',',':')).encode()).hexdigest()
  assert row['id_slot']==slot and row['timings']['predicted_n']==1 and row['timings'].get('draft_n',0)==0
  assert row['timings']['cache_n']+row['timings']['prompt_n']==len(tokens),('Unexpected tokenization',name,row['timings'],len(tokens))
  if not cache:assert row['timings']['cache_n']==0
  else:assert row['timings']['cache_n']>0
  report['cases'][name]=row;save()
  print(name,repr(row['content']),'cache',row['timings']['cache_n'],'new',row['timings']['prompt_n'],'seconds',round(row['wall_seconds'],3),flush=True)
  return row
 done=threading.Event()
 def heartbeat():
  t=time.monotonic()
  while not done.wait(30):print('Cache boundary probe active',round(time.monotonic()-t),'seconds',flush=True)
 try:
  save();threading.Thread(target=heartbeat,daemon=True).start()
  for n in lengths:
   tokens=full[:n-64]+full[-64:];assert len(tokens)==n
   report.setdefault('prompt_token_ids',{})[str(n)]=tokens
   fresh=comp(f'n{n}-fresh',tokens,False,1)
   for offset in offsets:
    prefix=n-offset
    comp(f'n{n}-prefix{prefix}',tokens[:prefix],False,0)
    cached=comp(f'n{n}-cached-prefix{prefix}',tokens,True,0)
    assert cached['input_token_sha256']==fresh['input_token_sha256']
    comparison=compare(cached,fresh);report['comparisons'][f'n{n}-prefix{prefix}']=comparison;save()
    print('comparison',n,prefix,json.dumps({k:v for k,v in comparison.items() if k not in ['left_top','right_top']}),flush=True)
   if n==lengths[0]:
    repeat=comp(f'n{n}-fresh-repeat',tokens,False,1);report['comparisons'][f'n{n}-fresh-repeat']=compare(fresh,repeat);save()
    assert fresh['completion_probabilities']==repeat['completion_probabilities'],'Fresh baseline is not reproducible'
  idle();final=snapshot(a.pid,18131)
  for k in ['proc_start_ticks','command','environment','mapped_libraries_sha256','cpu_affinity']:assert final[k]==runtime[k],k
  report['final_runtime']=final;report['completed']=True;save()
 except BaseException as e:report['error']=repr(e);save();raise
 finally:done.set();save()
if __name__=='__main__':main()
