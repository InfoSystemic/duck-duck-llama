#!/usr/bin/env python3
"""Measure real cached generation across CPU thread counts in fresh HTTP-style threads."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import threading
import torch
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
        wait_policy=os.environ.get('OMP_WAIT_POLICY'),measurement='single-request decode forward passes after prefill')
    runtime=None
    try:
        server.ServingStore=ResidentStore; server.bind=bind
        runtime=server.Runtime(args)
        ids,count=runtime.prepare(dict(messages=[dict(role='user',content='Hi.')],max_tokens=16))
        golden=json.loads((BASE/'results/deepseek-v41-checkpoint-run-0910b/generation.json').read_text())['runs'][0]
        current=[]; reference=[]
        def capture(_module,_args,output): current.append(output[1].detach().float().numpy().copy())
        runtime.model.register_forward_hook(capture)
        for workers in [16,16,8,4,2,1,16]:
            label='warmup' if not result['runs'] else 'threads_'+str(workers)
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
            assert row['token_ids']==golden['token_ids'] and row['timings']['downloaded_bytes']==0
            hashes=[hashlib.sha256(x.tobytes()).hexdigest() for x in current]
            if not reference: reference.extend(current)
            row.update(label=label,workers=workers,decode_tok_s=(len(row['token_ids'])-1)/row['timings']['decode_seconds'],
                logits_sha256=hashes,exact_golden=hashes==[s['logits_sha256'] for s in golden['steps']],
                logits_max_abs=max(float(abs(a-b).max()) for a,b in zip(reference,current)))
            result['runs'].append(row);atomic_json(args.output/'result.json',result)
            print(json.dumps({k:row[k] for k in ['label','decode_tok_s','exact_golden','logits_max_abs']}),flush=True)
        result['passed']=True
    except BaseException as error:
        result['error']=repr(error);raise
    finally:
        if runtime is not None:runtime.store.release_all();runtime.store.close()
        atomic_json(args.output/'result.json',result)

if __name__=='__main__':main()
