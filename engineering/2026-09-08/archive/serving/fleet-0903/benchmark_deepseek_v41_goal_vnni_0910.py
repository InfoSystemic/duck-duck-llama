#!/usr/bin/env python3
"""Matched, complete-model measurements of grouped experts and fused CPU operators."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import functools
from collections import defaultdict
import torch
import deepseek_v41_server_0910 as server
from deepseek_v41_resident_store_0910 import ResidentStore, bind
from goal_runtime_0910 import Optimizations
from goal_vnni_runtime_0910 import install as install_vnni
from deepseek_v41_native_grouped_vnni_goal_0910 import GroupedVnniGemm
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import sha256

BASE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cache',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    paths=[BASE/'goal_vnni_runtime_0910.py',BASE/'deepseek_v41_native_vnni_goal_0910.py',BASE/'deepseek_v41_native_grouped_vnni_goal_0910.py',BASE/'results/deepseek-v41-native-vnni-goal-0910/libdeepseek-v41-native-vnni.so',BASE/'results/deepseek-v41-native-grouped-vnni-goal-0910/libdeepseek-v41-native-grouped-vnni.so',Path(__file__),BASE/'goal_runtime_0910.py',BASE/'goal_grouped_moe_0910.py',BASE/'goal_hc_0910.py',
        BASE/'goal_native_quant_0910.py',BASE/'goal_sparse_script_0910.py',BASE/'goal_quant_reuse_0910.py',
        BASE/'deepseek_v41_native_grouped_goal_0910.py',BASE/'results/deepseek-v41-native-grouped-goal-0910/libdeepseek-v41-native-grouped.so',
        BASE/'results/goal_native_quant_0910/libgoal_native_quant_0910.so',BASE/'results/goal_hc_0910/libgoal_hc_0910.so']
    result=dict(passed=False,source_sha256={str(p):sha256(p) for p in paths},runs=[],affinity=sorted(os.sched_getaffinity(0)),
        wait_policy=os.environ.get('OMP_WAIT_POLICY'),measurement='single-request decode forward passes after prefill')
    runtime=None
    timers=defaultdict(lambda:[0,0.0]);phase=['loading']
    def instrument(owner,name,label):
        original=getattr(owner,name)
        @functools.wraps(original)
        def call(*args,**kwargs):
            began=time.perf_counter()
            try:return original(*args,**kwargs)
            finally:
                record=timers[phase[0]+':'+label];record[0]+=1;record[1]+=time.perf_counter()-began
        setattr(owner,name,call)
    from deepseek_v41_native_bridge_0910 import NativeGemm
    from deepseek_v41_native_grouped_goal_0910 import GroupedNativeGemm
    from goal_hc_0910 import NativeHC
    from goal_native_quant_0910 import NativeQuant
    from goal_sparse_script_0910 import SparseScript
    instrument(NativeGemm,'apply','base_gemm')
    instrument(GroupedNativeGemm,'apply','grouped_gemm')
    instrument(GroupedVnniGemm,'apply','grouped_vnni_gemm')
    instrument(NativeHC,'hc_split_sinkhorn','hc')
    instrument(NativeQuant,'act_quant','quant')
    instrument(SparseScript,'__call__','sparse')
    instrument(torch.nn.functional,'linear','torch_linear')
    instrument(torch,'einsum','einsum')
    try:
        server.ServingStore=ResidentStore;server.bind=bind
        runtime=server.Runtime(args);opt=Optimizations(runtime)
        ids,count=runtime.prepare(dict(messages=[dict(role='user',content='Hi.')],max_tokens=16))
        golden=json.loads((BASE/'results/deepseek-v41-checkpoint-run-0910b/generation.json').read_text())['runs'][0]
        current=[];reference=[]
        runtime.model.register_forward_pre_hook(lambda _m,a:phase.__setitem__(0,'prefill' if a[1]==0 else 'decode'))
        runtime.model.register_forward_hook(lambda _m,_a,o:current.append(o[1].detach().float().numpy().copy()))
        full=dict(quant=True,reuse=True,grouped=True,hc=True,sparse=True)
        opt.configure(full)
        vnni=install_vnni(runtime);vnni.enabled=False
        grouped_backend=GroupedVnniGemm(BASE/'results/deepseek-v41-native-grouped-vnni-goal-0910/libdeepseek-v41-native-grouped-vnni.so',vnni.get_packed,workers=16)
        plan=[('warmup',False,False),('baseline',False,False),('single_warm',True,False),
              ('single',True,False),('grouped_warm',True,True),('grouped',True,True),
              ('grouped_again',True,True),('baseline_after',False,True)]
        for label,enabled,use_grouped in plan:
            config=dict(full,vnni=enabled,grouped_vnni=use_grouped)
            vnni.enabled=enabled
            if use_grouped:vnni.set_grouped_backend(grouped_backend)
            replies=[];failures=[];current.clear();timers.clear()
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
            row['inclusive_timers']={k:dict(calls=v[0],seconds=v[1]) for k,v in timers.items()}
            row['vnni_stats']={k:getattr(vnni,k) for k in ['packed_bytes','cache_hits','cache_misses','pack_count','pack_seconds','cap_fallback_calls','dropped_entries']}
            result['runs'].append(row);atomic_json(args.output/'result.json',result)
            print(json.dumps({k:row[k] for k in ['label','decode_tok_s','exact_golden','tokens_match','logits_max_abs','calls']}),flush=True)
            assert row['tokens_match'],(label,row['content'])
            if not label.endswith('_warm') and label!='warmup':assert row['timings']['downloaded_bytes']==0,label
            if not enabled:assert row['exact_golden'],label
        assert all(sha256(p)==h for p,h in result['source_sha256'].items())
        result['passed']=True
    except BaseException as error:result['error']=repr(error);raise
    finally:
        if runtime is not None:runtime.store.release_all();runtime.store.close()
        atomic_json(args.output/'result.json',result)

if __name__=='__main__':main()
