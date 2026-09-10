#!/usr/bin/env python3
"""Resume the same native checkpoint graph with coalesced, rate-aware downloads."""
import json
import run_deepseek_v41_checkpoint_0910 as runner
from deepseek_v41_checkpoint_transport_0910b import CoalescedStore

original_bind=runner.bind_checkpoint


def bind(model,store,progress=print,dry_run=False):
    result=original_bind(model,store,progress,dry_run)
    if not dry_run:
        for layer in model.layers:
            lid=layer.layer_id
            def prefetch(module,inputs,output,_lid=lid):
                selected=sorted(set(output[1].flatten().tolist()))
                for start in range(0,len(selected),16):
                    names=[f'layers.{_lid}.ffn.experts.{i}.{w}.{kind}' for i in selected[start:start+16]
                        for w in ['w1','w2','w3'] for kind in ['weight','scale']]
                    store.ensure(names)
            layer.ffn.gate.register_forward_hook(prefetch)
    return result


if __name__=='__main__':
    runner.Store=CoalescedStore;runner.bind_checkpoint=bind;runner.main()
