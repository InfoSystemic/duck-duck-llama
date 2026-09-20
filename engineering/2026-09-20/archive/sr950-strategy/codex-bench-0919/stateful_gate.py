#!/usr/bin/env python3
"""Check warm append, checkpoint divergence, and two active streams against fresh decoding."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import threading
import time
import urllib.request

HERE = Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--endpoint', default='http://127.0.0.1:18131')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--reference', type=Path)
    args=ap.parse_args()
    def get(path):
        return json.load(urllib.request.urlopen(args.endpoint+path, timeout=10))
    assert not any(s['is_processing'] for s in get('/slots')), 'Other request active'
    report={'tag':args.tag,'cases':{},'checks':{}}
    def comp(name,prompt,cache,slot,n=48):
        body={'prompt':prompt,'n_predict':n,'temperature':0,'seed':42,'cache_prompt':cache,
              'stream':False,'samplers':['temperature'],'id_slot':slot}
        req=urllib.request.Request(args.endpoint+'/completion',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        started=time.monotonic()
        d=json.load(urllib.request.urlopen(req,timeout=600))
        row={'content':d['content'],'timings':d['timings'],'id_slot':d['id_slot'],'wall_seconds':time.monotonic()-started}
        report['cases'][name]=row
        print(name, json.dumps({k:row[k] for k in ('timings','id_slot','wall_seconds')}),flush=True)
        assert row['id_slot']==slot, 'Requested slot not honored'
        return row
    try:
        rng=random.Random(7)
        words='alpha beta gamma delta memory bandwidth socket kernel scheduler page node thread barrier expert router token cache state vector matrix'.split()
        base=' '.join(rng.choice(words) for _ in range(3200))
        prefix='Summarize the following list of words and then continue the story:\n'
        p_a=prefix+base+'\nThe story begins:'
        a=comp('initial',p_a,True,0)
        p_b=p_a+a['content']+'\nThen suddenly'
        b=comp('append_cached',p_b,True,0)
        f=comp('append_fresh',p_b,False,1)
        report['checks']['append_output_equal']=b['content']==f['content']
        report['checks']['append_cache_used']=b['timings'].get('cache_n',0)>0
        cut=int(len(base)*0.8)
        p_d=prefix+base[:cut]+' ZEBRA QUANTUM '+base[cut:]+'\nThe story begins:'
        d=comp('diverge_cached',p_d,True,0)
        f=comp('diverge_fresh',p_d,False,1)
        report['checks']['diverge_output_equal']=d['content']==f['content']
        report['checks']['diverge_cache_used']=d['timings'].get('cache_n',0)>0
        prompts=['Write a Python function to merge two sorted lists. Return complete code with comments.',
                 'Explain why a hash collision does not mean two keys are equal. Give a concrete example.']
        single=[comp('single_'+str(i),p,False,i,64) for i,p in enumerate(prompts)]
        barrier=threading.Barrier(2)
        def concurrent(i):
            barrier.wait(timeout=10)
            return comp('concurrent_'+str(i),prompts[i],False,i,64)
        both_active=False
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(concurrent,i) for i in range(2)]
            while not all(f.done() for f in futures):
                both_active |= sum(bool(s['is_processing']) for s in get('/slots'))==2
                time.sleep(.2)
            concurrent_rows=[f.result() for f in futures]
        report['checks']['two_slots_active']=both_active
        for i in range(2):
            report['checks']['concurrent_'+str(i)+'_equal']=single[i]['content']==concurrent_rows[i]['content']
        if args.reference:
            reference=json.loads(args.reference.read_text())
            for name,row in report['cases'].items():
                report['checks'][name+'_reference_equal']=row['content']==reference['cases'][name]['content']
        report['passed']=all(report['checks'].values())
        if args.reference:
            report['baseline_consistency_passed']=reference['passed']
            report['baseline_checks']=reference['checks']
            # Preserve failed intrinsic checks. A kernel regression gate compares
            # every cached/fresh/parallel output to the same production scenario.
            report['regression_passed']=all(report['checks'].get(name+'_reference_equal',False)
                                            for name in report['cases'])
            report['regression_passed'] &= all(report['checks'][k]==v for k,v in reference['checks'].items())
            assert report['regression_passed'], report['checks']
        else:
            assert report['passed'], report['checks']
    except BaseException as exc:
        report['error']=repr(exc)
        raise
    finally:
        (HERE/('stateful-'+args.tag+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks'],indent=2),flush=True)

if __name__=='__main__': main()
