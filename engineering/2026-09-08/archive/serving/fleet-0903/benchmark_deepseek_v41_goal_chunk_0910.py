#!/usr/bin/env python3
"""Perfect-draft verifier feasibility only; never served generation throughput.

All input proposals are taken from a separately verified greedy reference. The
measurement excludes a real draft model and cannot establish speculative speed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import threading
import time

import torch
import deepseek_v41_server_0910 as server
from deepseek_v41_resident_store_0910 import ResidentStore, bind
from goal_runtime_0910 import Optimizations
from goal_chunk_verify_0910 import ChunkVerifier
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import sha256

BASE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    proof=json.loads((BASE/'results/goal_chunk_verify_0910/fixture-check.json').read_text())
    assert proof['passed'] and proof['shared_runtime_exact']
    assert all(sha256(p)==h for p,h in proof['source_sha256'].items())
    sources=[Path(__file__),BASE/'goal_chunk_verify_0910.py',BASE/'goal_runtime_0910.py']
    result=dict(passed=False,source_sha256={str(p):sha256(p) for p in sources},runs=[],
                perfect_draft_verifier_only=True,real_draft_model_loaded=False,
                served_generation_tok_s_measured=False,
                excludes=['draft model time','proposal rejection and replay','network cold loads'])
    runtime=None
    try:
        server.ServingStore=ResidentStore;server.bind=bind
        runtime=server.Runtime(args)
        opt=Optimizations(runtime)
        opt.configure(dict(quant=True,reuse=True,grouped=True,hc=True,sparse=True))
        ids,count=runtime.prepare(dict(messages=[dict(role='user',content='Hi.')],max_tokens=16))
        golden=json.loads((BASE/'results/deepseek-v41-checkpoint-run-0910b/generation.json').read_text())['runs'][0]
        proposals=golden['token_ids'][:-1]
        references=[]
        verifier=ChunkVerifier(runtime.model,runtime.module)
        for label,width in [('warmup',1),('serial_before',1),('chunk3_warm',3),('chunk3',3),
                            ('chunk5_warm',5),('chunk5',5),('chunk8',8),('serial_after',1)]:
            print(json.dumps(dict(running=label,chunk_width=width)),flush=True)
            replies=[];failures=[]
            def request():
                try:
                    torch.set_num_threads(16)
                    with torch.inference_mode():
                        before=runtime.store.downloaded_bytes
                        began=time.perf_counter()
                        initial=runtime.model(torch.tensor([ids],dtype=torch.int64),0)
                        prefill_seconds=time.perf_counter()-began
                        logits=[initial[1].detach().clone()]
                        predicted=[int(initial[0].item())]
                        decode_seconds=0.;calls=[]
                        for at in range(0,len(proposals),width):
                            batch=proposals[at:at+width]
                            x=torch.tensor([batch],dtype=torch.int64)
                            began=time.perf_counter()
                            if len(batch)>1:
                                output=verifier.verify(x,len(ids)+at)
                                decoded=output[0][0].tolist()
                                values=[output[1][:,i].detach().clone() for i in range(len(batch))]
                                mode=verifier.last_mode
                            else:
                                output=runtime.model(x,len(ids)+at)
                                decoded=[int(output[0].item())]
                                values=[output[1].detach().clone()]
                                mode='single'
                            elapsed=time.perf_counter()-began
                            decode_seconds+=elapsed
                            calls.append(dict(width=len(batch),mode=mode,seconds=elapsed))
                            predicted.extend(decoded);logits.extend(values)
                        hashes=[hashlib.sha256(x.float().numpy().tobytes()).hexdigest() for x in logits]
                        if not references:references.extend(logits)
                        row=dict(label=label,chunk_width=width,prefill_seconds=prefill_seconds,
                                 decode_seconds=decode_seconds,verified_positions=len(proposals),
                                 ideal_verifier_positions_s=len(proposals)/decode_seconds,
                                 downloaded_bytes=runtime.store.downloaded_bytes-before,
                                 logits_sha256=hashes,predicted_ids=predicted,calls=calls,
                                 exact_golden=hashes==[s['logits_sha256'] for s in golden['steps']],
                                 tokens_match=predicted==golden['token_ids'],
                                 max_abs=max(float((a-b).abs().max()) for a,b in zip(logits,references)))
                        replies.append(row)
                except BaseException as error:failures.append(error)
            thread=threading.Thread(target=request);thread.start();thread.join()
            if failures:raise failures[0]
            row=replies[0];result['runs'].append(row)
            atomic_json(args.output/'result.json',result)
            print(json.dumps({k:row[k] for k in ['label','ideal_verifier_positions_s','exact_golden','tokens_match','max_abs','downloaded_bytes']}),flush=True)
            assert row['tokens_match'],label
            if width==1:assert row['exact_golden'],label
            if label!='warmup' and not label.endswith('_warm'):assert row['downloaded_bytes']==0,label
        verifier.uninstall()
        assert all(sha256(p)==h for p,h in result['source_sha256'].items())
        result['passed']=True
    except BaseException as error:
        result['error']=repr(error)
        raise
    finally:
        if runtime is not None:runtime.store.release_all();runtime.store.close()
        atomic_json(args.output/'result.json',result)


if __name__=='__main__':main()
