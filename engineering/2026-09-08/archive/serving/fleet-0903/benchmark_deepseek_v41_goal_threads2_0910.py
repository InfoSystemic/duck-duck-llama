#!/usr/bin/env python3
"""Measure real cached generation across CPU thread counts in fresh HTTP-style threads."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import threading
import torch
from goal_quant_reuse_0910 import install as install_reuse
from goal_native_quant_0910 import NativeQuant
from check_deepseek_v41_native_gemm_goal_0910 import variant
import deepseek_v41_server_0910 as server
from deepseek_v41_resident_store_0910 import ResidentStore, bind
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import sha256

BASE = Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=dict(passed=False,source_sha256=sha256(__file__),runs=[],affinity=sorted(os.sched_getaffinity(0)),
        wait_policy=os.environ.get('OMP_WAIT_POLICY'),candidate_inputs={str(p):sha256(p) for p in [BASE/'goal_quant_reuse_0910.py',BASE/'goal_native_quant_0910.py',BASE/'results/goal_native_quant_0910/libgoal_native_quant_0910.so',BASE/'results/deepseek-v41-native-gemm-goal-0910/libdeepseek-v41-native-gemm.so']},measurement='single-request decode forward passes after prefill')
    runtime=None
    try:
        server.ServingStore=ResidentStore; server.bind=bind
        runtime=server.Runtime(args)
        reuse=install_reuse(runtime.module); reuse.enabled=False
        quant=NativeQuant(BASE/'results/goal_native_quant_0910/libgoal_native_quant_0910.so')
        original_quant=runtime.module.act_quant
        original_native=runtime.native
        candidate_native=variant(BASE/'results/deepseek-v41-native-gemm-goal-0910/libdeepseek-v41-native-gemm.so',1,16)
        ids,count=runtime.prepare(dict(messages=[dict(role='user',content='Hi.')],max_tokens=16))
        golden=json.loads((BASE/'results/deepseek-v41-checkpoint-run-0910b/generation.json').read_text())['runs'][0]
        current=[]; reference=[]
        def capture(_module,_args,output): current.append(output[1].detach().float().numpy().copy())
        runtime.model.register_forward_hook(capture)
        plan=[('warmup',16,False,False,False),('baseline',16,False,False,False),
              ('reuse',16,True,False,False),('native_quant',16,False,True,False),
              ('all_16',16,True,True,False),('all_8',8,True,True,False),('all_4',4,True,True,False),('all_2',2,True,True,False),('all_1',1,True,True,False),
              ('baseline_after',16,False,False,False)]
        plan=[item for p in plan for item in ([p] if p[0]=='warmup' else [(p[0]+'_warm',*p[1:]),p])]
        for label,workers,use_reuse,use_quant,use_gemm in plan:
            reuse.enabled=use_reuse
            runtime.module.act_quant=quant.act_quant if use_quant else original_quant
            runtime.native=candidate_native if use_gemm else original_native
            runtime.module.fp4_gemm=runtime.native.fp4; runtime.module.fp8_gemm=runtime.native.fp8
            replies=[]; failures=[]; current.clear()
            print(json.dumps(dict(running=label)),flush=True)
            def request_thread():
                try:
                    torch.set_num_threads(workers); runtime.native.workers=workers
                    replies.append(runtime.generate(ids,count,lambda _:None))
                except BaseException as error: failures.append(error)
            thread=threading.Thread(target=request_thread);thread.start();thread.join()
            if failures: raise failures[0]
            row=replies[0]
            row['token_ids_match']=row['token_ids']==golden['token_ids']
            hashes=[hashlib.sha256(x.tobytes()).hexdigest() for x in current]
            if not reference: reference.extend(current)
            row.update(label=label,workers=workers,reuse=use_reuse,native_quant=use_quant,new_gemm=use_gemm,decode_tok_s=(len(row['token_ids'])-1)/row['timings']['decode_seconds'],
                logits_sha256=hashes,exact_golden=hashes==[s['logits_sha256'] for s in golden['steps']],
                logits_max_abs=max(float(abs(a-b).max()) for a,b in zip(reference,current)))
            result['runs'].append(row);atomic_json(args.output/'result.json',result)
            assert row['token_ids_match'], (label,row['content'])
            if not label.endswith('_warm') and label!='warmup': assert row['timings']['downloaded_bytes']==0, label
            print(json.dumps({k:row[k] for k in ['label','decode_tok_s','exact_golden','logits_max_abs']}),flush=True)
        result['passed']=True
    except BaseException as error:
        result['error']=repr(error);raise
    finally:
        if runtime is not None:runtime.store.release_all();runtime.store.close()
        atomic_json(args.output/'result.json',result)

if __name__=='__main__':main()
