#!/usr/bin/env python3
"""Matched, complete-model measurements of grouped experts and fused CPU operators."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import threading
os.sched_setaffinity(0,set(range(64)))
os.environ['OMP_NUM_THREADS']='64'
os.environ['MKL_NUM_THREADS']='64'
os.environ['OMP_PROC_BIND']='true'
os.environ['GOMP_CPU_AFFINITY']='0-63'
os.environ['OMP_WAIT_POLICY']='PASSIVE'
import torch
import deepseek_v41_server_0910 as server
from deepseek_v41_resident_store_0910 import ResidentStore, bind
from goal_runtime_0910 import Optimizations
from goal_numa_store_0910 import NumaResidentStore
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import sha256

BASE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cache',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    paths=[BASE/'goal_numa_store_0910.py',BASE/'goal_numa_0910.py',Path(__file__),BASE/'goal_runtime_0910.py',BASE/'goal_grouped_moe_0910.py',BASE/'goal_hc_0910.py',
        BASE/'goal_native_quant_0910.py',BASE/'goal_sparse_script_0910.py',BASE/'goal_quant_reuse_0910.py',
        BASE/'deepseek_v41_native_grouped_goal_0910.py',BASE/'results/deepseek-v41-native-grouped-goal-0910/libdeepseek-v41-native-grouped.so',
        BASE/'results/goal_native_quant_0910/libgoal_native_quant_0910.so',BASE/'results/goal_hc_0910/libgoal_hc_0910.so']
    result=dict(passed=False,source_sha256={str(p):sha256(p) for p in paths},runs=[],affinity=sorted(os.sched_getaffinity(0)),
        wait_policy=os.environ.get('OMP_WAIT_POLICY'),measurement='single-request decode forward passes after prefill')
    runtime=None
    try:
        server.ServingStore=NumaResidentStore;server.bind=bind
        runtime=server.Runtime(args);runtime.store.clone_common(runtime.model);opt=Optimizations(runtime)
        ids,count=runtime.prepare(dict(messages=[dict(role='user',content='Hi.')],max_tokens=16))
        golden=json.loads((BASE/'results/deepseek-v41-checkpoint-run-0910b/generation.json').read_text())['runs'][0]
        current=[];reference=[]
        runtime.model.register_forward_hook(lambda _m,_a,o:current.append(o[1].detach().float().numpy().copy()))
        full=dict(quant=True,reuse=True,grouped=True,hc=True,sparse=True)
        plan=[('warmup',dict(full,native_workers=64,torch_workers=64)),
              ('numa64',dict(full,native_workers=64,torch_workers=64)),
              ('numa32_warm',dict(full,native_workers=32,torch_workers=32)),
              ('numa32',dict(full,native_workers=32,torch_workers=32)),
              ('numa64_again',dict(full,native_workers=64,torch_workers=64))]
        for label,config in plan:
            opt.configure(config);replies=[];failures=[];current.clear()
            print(json.dumps(dict(running=label,config=config)),flush=True)
            def request_thread():
                try:
                    torch.set_num_threads(config.get('torch_workers',16))
                    replies.append(runtime.generate(ids,count,lambda _:None))
                except BaseException as error:failures.append(error)
            thread=threading.Thread(target=request_thread);thread.start();thread.join()
            if failures:raise failures[0]
            row=replies[0]
            hashes=[hashlib.sha256(x.tobytes()).hexdigest() for x in current]
            if not reference:reference.extend(current)
            row.update(label=label,config=config,decode_tok_s=(len(row['token_ids'])-1)/row['timings']['decode_seconds'],
                logits_sha256=hashes,exact_golden=hashes==[s['logits_sha256'] for s in golden['steps']],
                tokens_match=row['token_ids']==golden['token_ids'],logits_max_abs=max(float(abs(a-b).max()) for a,b in zip(reference,current)),
                calls=dict(grouped=opt.grouped.calls,quant=opt.quant.native_calls,hc=opt.hc.native_calls))
            row['numa']=runtime.store.stats()
            result['runs'].append(row);atomic_json(args.output/'result.json',result)
            print(json.dumps({k:row[k] for k in ['label','decode_tok_s','exact_golden','tokens_match','logits_max_abs','calls']}),flush=True)
            assert row['tokens_match'],(label,row['content'])
            if not label.endswith('_warm') and label!='warmup':assert row['timings']['downloaded_bytes']==0,label

        assert all(sha256(p)==h for p,h in result['source_sha256'].items())
        result['passed']=True
    except BaseException as error:result['error']=repr(error);raise
    finally:
        if runtime is not None:runtime.store.release_all();runtime.store.close()
        atomic_json(args.output/'result.json',result)

if __name__=='__main__':main()
