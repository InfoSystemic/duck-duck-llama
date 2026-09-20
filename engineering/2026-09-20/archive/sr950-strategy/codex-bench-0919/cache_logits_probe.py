#!/usr/bin/env python3
"""Compare cached/fresh first-token distributions, with and without prior MTP generation."""
import argparse
import json
from pathlib import Path
import random
import threading
import time
import urllib.request
from capture_runtime import snapshot

HERE=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pid',type=int,required=True)
    ap.add_argument('--tag',required=True)
    args=ap.parse_args()
    path=HERE/f'cache-logits-{args.tag}.json'
    assert not path.exists()
    runtime=snapshot(args.pid,18131)
    assert runtime['health']=={'status':'ok'} and not any(s['is_processing'] for s in runtime['slots'])
    assert not Path('/dev/shm/flash-optrace.arm').exists()
    report={'runtime':runtime,'cases':{},'purpose':'Characterize existing cache divergence; do not equate greedy inequality with proven corruption.'}
    done=threading.Event()
    def heartbeat():
        started=time.monotonic()
        while not done.wait(30):print(f'Cache distribution probe active {time.monotonic()-started:.0f}s',flush=True)
    def comp(name,prompt,cache,slot,n):
        body={'prompt':prompt,'n_predict':n,'temperature':0,'seed':42,'cache_prompt':cache,
              'stream':False,'samplers':['temperature'],'id_slot':slot,
              'n_probs':20,'post_sampling_probs':False}
        req=urllib.request.Request('http://127.0.0.1:18131/completion',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        started=time.monotonic();result=json.load(urllib.request.urlopen(req,timeout=600))
        assert result['id_slot']==slot
        row={k:v for k,v in result.items() if k not in ['prompt','generation_settings']}
        row['generation_settings']={k:result.get('generation_settings',{}).get(k) for k in ['n_probs','post_sampling_probs','temperature','seed','samplers']}
        row['wall_seconds']=time.monotonic()-started
        report['cases'][name]=row
        path.write_text(json.dumps(report,indent=2)+'\n')
        print(name,repr(row.get('content')),json.dumps(row['timings']),flush=True)
        return row
    rng=random.Random(7)
    words='alpha beta gamma delta memory bandwidth socket kernel scheduler page node thread barrier expert router token cache state vector matrix'.split()
    base=' '.join(rng.choice(words) for _ in range(3200))
    p='Summarize the following list of words and then continue the story:\n'+base+'\nThe story begins:'
    try:
        threading.Thread(target=heartbeat,daemon=True).start()
        initial=comp('mtp_initial',p,False,0,48)
        assert initial['timings'].get('draft_n',0)>0, 'Initial generation did not exercise MTP'
        reference=json.loads((HERE/'stateful-combo.json').read_text())['cases']['initial']['content']
        report['initial_matches_saved_reference']=initial['content']==reference
        append=p+initial['content']+'\nThen suddenly'
        comp('mtp_append_cached',append,True,0,1)
        comp('mtp_append_fresh',append,False,1,1)
        # One sampled output token has not been fed back through MTP verification.
        # Append to the prompt itself, excluding that sampled token.
        prefill=comp('prefill_initial',p,False,0,1)
        assert prefill['timings'].get('draft_n',0)==0, 'Prefill control unexpectedly used MTP drafts'
        append=p+'\nThen suddenly'
        comp('prefill_append_cached',append,True,0,1)
        comp('prefill_append_fresh',append,False,1,1)
        for prefix in ['mtp','prefill']:
            cached=report['cases'][prefix+'_append_cached']
            fresh=report['cases'][prefix+'_append_fresh']
            report[prefix+'_first_token_equal']=cached['content']==fresh['content']
            assert cached['timings']['cache_n']>0 and fresh['timings']['cache_n']==0
        report['completed']=True
    except BaseException as exc:
        report['error']=repr(exc);raise
    finally:
        done.set();path.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
