#!/usr/bin/env python3
"""Execute the released V4.1 text backbone with demand-loaded native weights."""
import argparse
import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import types
import torch
from deepseek_v41_checkpoint_0910 import BASE,Catalog,Store,meta_model,bind_checkpoint,REVISION
from deepseek_v41_native_bridge_0910 import NativeGemm
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import sha256


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--inspect',action='store_true')
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--cache',type=Path)
    args=parser.parse_args();out=args.output
    torch.set_num_threads(16);torch.set_num_interop_threads(1);torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device('cpu')
    library=BASE/'results/deepseek-v41-native-gemm-0910/libdeepseek-v41-native-gemm.so'
    proof=json.loads((library.parent/'kernel-check.json').read_text());assert proof['passed']
    assert sha256(library)==proof['input_sha256'][str(library)]
    native=NativeGemm(library,16);native.install()
    m,config,tokenizer,model=meta_model()
    planned=bind_checkpoint(model,types.SimpleNamespace(catalog=Catalog()),dry_run=True)
    planned['args']=dataclasses.asdict(config)
    if args.inspect:
        atomic_json(out/'shape-check.json',planned);print(json.dumps(planned),flush=True);return
    assert args.cache
    result=dict(passed=False,started=time.time(),revision=REVISION,device='cpu',parameters_native_precision=True,
        all_checkpoint_bytes_resident=False,engram_rows_on_demand=True,experts_on_demand=True,
        released_model_generated_text=False,runs=[],full_checkpoint_loaded=False)
    store=Store(args.cache,out)
    try:
        result['backbone']=bind_checkpoint(model,store)
        result['ready_at']=time.time();atomic_json(out/'generation.json',result)
        ep=BASE/'results/deepseek-v41-cpu-source-0910/encoding/encoding.py'
        spec=importlib.util.spec_from_file_location('deepseek_v41_encoding_native_run',ep)
        encoder=importlib.util.module_from_spec(spec);spec.loader.exec_module(encoder)
        prompt=encoder.encode_messages([{'role':'user','content':'Hi.'}],thinking_mode='chat')
        ids=tokenizer.backend_tokenizer.encode(prompt,add_special_tokens=False).ids
        assert 1<len(ids)<32 and len(set(ids))>1
        eos=tokenizer.backend_tokenizer.token_to_id(encoder.eos_token)
        result.update(prompt=prompt,prompt_token_ids=ids,eos_token_id=eos)
        print(json.dumps(dict(real_checkpoint_ready=True,prompt_tokens=len(ids),cache_bytes=store.stored_bytes)),flush=True)
        for repeat in range(2):
            x=torch.tensor([ids],dtype=torch.int64);tokens=[];steps=[];position=0
            for step in range(24):
                before_bytes,before_network=store.downloaded_bytes,store.network_seconds
                start=time.perf_counter()
                output,logits,_=model(x,position)
                elapsed=time.perf_counter()-start
                assert logits.shape==(1,config.vocab_size) and torch.isfinite(logits).all()
                token=int(output.item());tokens.append(token)
                steps.append(dict(position=position,input_tokens=x.shape[1],seconds_including_downloads=elapsed,
                    downloaded_bytes=store.downloaded_bytes-before_bytes,
                    range_fetch_seconds=store.network_seconds-before_network,finite_logits=True,
                    logits_sha256=hashlib.sha256(logits.numpy().tobytes()).hexdigest(),token_id=token))
                text=tokenizer.backend_tokenizer.decode(tokens,skip_special_tokens=False)
                print(json.dumps(dict(repeat=repeat,generated_tokens=len(tokens),token_id=token,text=text,
                    seconds_including_downloads=elapsed,downloaded_bytes=steps[-1]['downloaded_bytes'],cache_bytes=store.stored_bytes)),flush=True)
                partial=dict(result,current_repeat=repeat,current_tokens=tokens,current_text=text,current_steps=steps,
                    released_model_generated_text=True)
                atomic_json(out/'generation-progress.json',partial)
                if token==eos:break
                position+=x.shape[1];x=torch.tensor([[token]],dtype=torch.int64)
            result['runs'].append(dict(token_ids=tokens,text=text,steps=steps,reached_eos=tokens[-1]==eos))
            result['released_model_generated_text']=True;atomic_json(out/'generation.json',result)
        assert result['runs'][0]['token_ids']==result['runs'][1]['token_ids'],'Greedy repeat changed tokens'
        assert [s['logits_sha256'] for s in result['runs'][0]['steps']]==[s['logits_sha256'] for s in result['runs'][1]['steps']],'Fresh request reset changed logits'
        assert all(s['downloaded_bytes']==0 for s in result['runs'][1]['steps']),'Warm repeat fetched additional weights'
        result.update(passed=True,exact_repeated_logits=True,downloaded_bytes=store.downloaded_bytes,
            cache_bytes=store.stored_bytes,engram_cached_rows=len(store.row_cache),finished=time.time(),
            scope='Real released 40-layer text backbone, native checkpoint weights and real Engram rows. '
                  'First request includes range downloads; second is cached. Short greeting is a bring-up check, not a quality evaluation. '
                  'No vision, DSpark, full expert residency, serving endpoint, or controlled bandwidth measurement.')
    except BaseException as e:result['error']=repr(e);result['finished']=time.time();raise
    finally:
        store.close();atomic_json(out/'generation.json',result)


if __name__=='__main__':main()
