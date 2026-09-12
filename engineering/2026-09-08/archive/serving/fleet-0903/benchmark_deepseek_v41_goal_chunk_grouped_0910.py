#!/usr/bin/env python3
"""Grouped perfect-draft verifier feasibility only; never served generation throughput.

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
from goal_chunk_verify2_0910 import ChunkVerifier
from goal_exact16_runtime_0910 import install as install_exact16
from goal_chunk_grouped_moe_0910 import install as install_chunk_grouped
from deepseek_v41_ragged_int16_goal_0910 import RaggedExact16Gemm
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import sha256

BASE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    proof=json.loads((BASE/'results/goal_chunk_verify2_0910/fixture-check.json').read_text())
    assert proof['passed'] and proof['shared_runtime_exact'] and proof['hc_scope_restored']
    assert all(sha256(p)==h for p,h in proof['source_sha256'].items())
    selected=json.loads((BASE/'deepseek-v41-selected.json').read_text())
    protected=dict(selected['source_sha256'])
    assert all(sha256(p)==h for p,h in protected.items()), 'Selected source drift before trial'
    sources=[Path(__file__),BASE/'goal_chunk_verify_0910.py',BASE/'goal_chunk_verify2_0910.py',
             BASE/'goal_runtime_0910.py',BASE/'results/goal_chunk_verify2_0910/fixture-check.json']
    for name,key in [('results/goal_chunk_grouped_moe_0910/fixture-check.json','source_sha256'),
                     ('results/goal_chunk_grouped_moe_0910/native-fixture-check.json','source_sha256'),
                     ('results/deepseek-v41-grouped-int16-exact-goal-0910/runtime-check.json','sha256'),
                     ('results/deepseek-v41-ragged-int16-exact-goal-0910/kernel-check.json','source_sha256')]:
        extra=json.loads((BASE/name).read_text())
        assert extra['passed'] and all(sha256(p)==h for p,h in extra[key].items())
        sources.extend(Path(p) for p in extra[key])
    sources.extend(BASE/name for name in ['goal_exact16_runtime_0910.py','deepseek_v41_ragged_int16_goal_0910.py',
        'goal_chunk_grouped_moe_0910.py','deepseek_v41_grouped_int16_exact_goal_0910.py',
        'results/deepseek-v41-grouped-int16-exact-goal-0910/libdeepseek-v41-grouped-int16-exact.so',
        'results/deepseek-v41-ragged-int16-exact-goal-0910/libdeepseek-v41-ragged-int16-exact.so'])
    source_hashes=dict(proof['source_sha256'])
    source_hashes.update({str(p):sha256(p) for p in sources})
    result=dict(passed=False,source_sha256=source_hashes,protected_selected_sources=protected,runs=[],
                hc_per_token=True,candidate_chunk_router=True,torch_workers=16,native_workers=16,
                six_position_arm='five perfect draft proposals plus one bonus position',
                perfect_draft_verifier_only=True,real_draft_model_loaded=False,
                served_generation_tok_s_measured=False,
                excludes=['draft model time','proposal rejection and replay','network cold loads'])
    runtime=None;verifier=None;exact16=None;chunk_grouped=None
    try:
        server.ServingStore=ResidentStore;server.bind=bind
        runtime=server.Runtime(args)
        opt=Optimizations(runtime)
        opt.configure(dict(quant=True,reuse=True,grouped=True,hc=True,sparse=True,native_workers=16))
        ids,count=runtime.prepare(dict(messages=[dict(role='user',content='Hi.')],max_tokens=16))
        golden=json.loads((BASE/'results/deepseek-v41-checkpoint-run-0910b/generation.json').read_text())['runs'][0]
        proposals=golden['token_ids'][:-1]
        references=[]
        verifier=ChunkVerifier(runtime.model,runtime.module)
        exact16=install_exact16(runtime,opt)
        exact16.enabled=False
        ragged=RaggedExact16Gemm(BASE/'results/deepseek-v41-ragged-int16-exact-goal-0910/libdeepseek-v41-ragged-int16-exact.so',exact16.get_packed,16)
        chunk_grouped=install_chunk_grouped(runtime,ragged,lambda:verifier.active)
        original_hc=runtime.module.hc_split_sinkhorn
        for label,width,use_grouped in [('warmup',1,False),('serial_before',1,False),
                            ('chunk5_plain_warm',5,False),('chunk5_plain',5,False),
                            ('chunk5_grouped_warm',5,True),('chunk5_grouped_1',5,True),('chunk5_grouped_2',5,True),
                            ('chunk6_grouped_warm',6,True),('chunk6_grouped',6,True),
                            ('chunk8_grouped_warm',8,True),('chunk8_grouped',8,True),
                            ('serial_after',1,False)]:
            chunk_grouped.enabled=use_grouped
            pack_before=exact16.pack_count
            grouped_before=chunk_grouped.calls
            native_before=(ragged.native_calls,ragged.fallback_calls,exact16.cap_fallbacks)
            measured=label!='warmup' and not label.endswith('_warm')
            hc_before=(verifier.hc_batch_calls,verifier.hc_token_calls,opt.hc.native_calls,opt.hc.fallback_calls)
            print(json.dumps(dict(running=label,chunk_width=width)),flush=True)
            replies=[];failures=[]
            def request():
                try:
                    torch.set_num_threads(16)
                    assert runtime.native.workers==16
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
                        row=dict(label=label,measured=measured,chunk_width=width,grouped=use_grouped,prefill_seconds=prefill_seconds,
                                 decode_seconds=decode_seconds,verified_positions=len(proposals),
                                 ideal_verifier_positions_s=len(proposals)/decode_seconds,
                                 downloaded_bytes=runtime.store.downloaded_bytes-before,
                                 logits_sha256=hashes,predicted_ids=predicted,calls=calls,
                                 exact_golden=hashes==[s['logits_sha256'] for s in golden['steps']],
                                 tokens_match=predicted==golden['token_ids'],
                                 hc_calls_delta=dict(batch=verifier.hc_batch_calls-hc_before[0],
                                     token=verifier.hc_token_calls-hc_before[1],
                                     native=opt.hc.native_calls-hc_before[2],
                                     fallback=opt.hc.fallback_calls-hc_before[3]),
                                 max_abs=max(float((a-b).abs().max()) for a,b in zip(logits,references)))
                        replies.append(row)
                except BaseException as error:failures.append(error)
            thread=threading.Thread(target=request);thread.start();thread.join()
            if failures:raise failures[0]
            row=replies[0];row.update(packed_bytes=exact16.packed_bytes,pack_delta=exact16.pack_count-pack_before,chunk_grouped_calls=chunk_grouped.calls-grouped_before,
                ragged_native_calls=ragged.native_calls-native_before[0],ragged_fallback_calls=ragged.fallback_calls-native_before[1],
                cap_fallbacks=exact16.cap_fallbacks-native_before[2]);result['runs'].append(row)
            atomic_json(args.output/'result.json',result)
            print(json.dumps({k:row[k] for k in ['label','ideal_verifier_positions_s','exact_golden','tokens_match','max_abs','downloaded_bytes']}),flush=True)
            assert row['tokens_match'] and row['exact_golden'],label
            assert runtime.module.hc_split_sinkhorn is original_hc, 'Verifier HC wrapper leaked'
            if measured:assert row['downloaded_bytes']==0 and row['pack_delta']==0,label
            if use_grouped:assert row['chunk_grouped_calls']>0 and row['ragged_native_calls']>0 and row['ragged_fallback_calls']==row['cap_fallbacks']==0,label
            if width>1:assert row['hc_calls_delta']['batch']>0 and row['hc_calls_delta']['native']>0,label
            assert all(sha256(p)==h for p,h in protected.items()), 'Selected source drift during trial'
        for width,use_grouped in [(1,False),(5,False),(5,True),(6,True),(8,True)]:
            rows=[row for row in result['runs'] if row['measured'] and row['chunk_width']==width and row['grouped']==use_grouped]
            result['width'+str(width)+('_grouped' if use_grouped else '_plain')+'_ideal_positions_s']=(
                sum(row['verified_positions'] for row in rows)/sum(row['decode_seconds'] for row in rows))
        result['all_exact_golden']=all(row['exact_golden'] for row in result['runs'])
        result['zero_measured_downloads']=all(row['downloaded_bytes']==0 for row in result['runs'] if row['measured'])
        chunk_grouped.uninstall();chunk_grouped=None
        exact16.uninstall();exact16=None
        verifier.uninstall();verifier=None
        assert all(sha256(p)==h for p,h in result['source_sha256'].items())
        result['passed']=True
    except BaseException as error:
        result['error']=repr(error)
        raise
    finally:
        if chunk_grouped is not None:chunk_grouped.uninstall()
        if exact16 is not None:exact16.uninstall()
        if verifier is not None:verifier.uninstall()
        if runtime is not None:runtime.store.release_all();runtime.store.close()
        result['selected_sources_preserved']=all(sha256(p)==h for p,h in protected.items())
        atomic_json(args.output/'result.json',result)


if __name__=='__main__':main()
