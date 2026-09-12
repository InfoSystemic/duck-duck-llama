#!/usr/bin/env python3
"""Measure exact fused decode with 16, 32, 48, and 64 native workers across four sockets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import threading
import torch
import deepseek_v41_server_0910 as server
from deepseek_v41_resident_store_0910 import ResidentStore, bind
from goal_runtime_0910 import Optimizations
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import sha256

BASE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cache',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    paths=[Path(__file__),BASE/'goal_runtime_0910.py',BASE/'goal_grouped_moe_0910.py',BASE/'goal_hc_0910.py',
        BASE/'goal_native_quant_0910.py',BASE/'goal_sparse_script_0910.py',BASE/'goal_quant_reuse_0910.py',
        BASE/'deepseek_v41_native_grouped_goal_0910.py',BASE/'results/deepseek-v41-native-grouped-goal-0910/libdeepseek-v41-native-grouped.so',
        BASE/'results/goal_native_quant_0910/libgoal_native_quant_0910.so',BASE/'results/goal_hc_0910/libgoal_hc_0910.so']
    result=dict(passed=False,source_sha256={str(p):sha256(p) for p in paths},runs=[],affinity=sorted(os.sched_getaffinity(0)),
        wait_policy=os.environ.get('OMP_WAIT_POLICY'),proc_bind=os.environ.get('OMP_PROC_BIND'),places=os.environ.get('OMP_PLACES'),requested_cpus=list(range(64)),requested_memory_policy='interleave:0-3',cached_weight_pages_relocated=False,measurement='single-request decode forward passes after prefill')
    runtime=None
    try:
        server.ServingStore=ResidentStore;server.bind=bind
        runtime=server.Runtime(args);opt=Optimizations(runtime)
        ids,count=runtime.prepare(dict(messages=[dict(role='user',content='Hi.')],max_tokens=16))
        golden=json.loads((BASE/'results/deepseek-v41-checkpoint-run-0910b/generation.json').read_text())['runs'][0]
        current=[];reference=[]
        runtime.model.register_forward_hook(lambda _m,_a,o:current.append(o[1].detach().float().numpy().copy()))
        full=dict(quant=True,reuse=True,grouped=True,hc=True,sparse=True)
        plan=[('warmup',full),('fused16_before',full),('fused32',dict(full,native_workers=32)),
              ('fused64',dict(full,native_workers=64)),('fused48',dict(full,native_workers=48)),
              ('fused64_repeat',dict(full,native_workers=64)),('fused32_repeat',dict(full,native_workers=32)),
              ('fused16_after',full)]
        for label,config in plan:
            opt.configure(config);replies=[];failures=[];current.clear()
            print(json.dumps(dict(running=label,config=config)),flush=True)
            def request_thread():
                try:
                    torch.set_num_threads(config.get('torch_workers',16))
                    row=runtime.generate(ids,count,lambda _:None)
                    affinities={}
                    for task in Path('/proc/self/task').iterdir():
                        try:
                            key=','.join(map(str,sorted(os.sched_getaffinity(int(task.name)))))
                            affinities[key]=affinities.get(key,0)+1
                        except ProcessLookupError:pass
                    row['thread_affinities']=affinities
                    row['numa_policy_counts']={}
                    for line in Path('/proc/self/numa_maps').read_text().splitlines():
                        policy=line.split()[1]
                        row['numa_policy_counts'][policy]=row['numa_policy_counts'].get(policy,0)+1
                    replies.append(row)
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
            result['runs'].append(row);atomic_json(args.output/'result.json',result)
            print(json.dumps({k:row[k] for k in ['label','decode_tok_s','exact_golden','tokens_match','logits_max_abs','calls']}),flush=True)
            assert row['tokens_match'],(label,row['content'])
            if not label.endswith('_warm') and label!='warmup':assert row['timings']['downloaded_bytes']==0,label
            if config.get('torch_workers',16)==16:assert row['exact_golden'],label
        assert all(sha256(p)==h for p,h in result['source_sha256'].items())
        result['passed']=True
        result['over_10_tok_s']=any(r['label']!='warmup' and r['decode_tok_s']>10 for r in result['runs'])
    except BaseException as error:result['error']=repr(error);raise
    finally:
        if runtime is not None:runtime.store.release_all();runtime.store.close()
        atomic_json(args.output/'result.json',result)

if __name__=='__main__':main()
