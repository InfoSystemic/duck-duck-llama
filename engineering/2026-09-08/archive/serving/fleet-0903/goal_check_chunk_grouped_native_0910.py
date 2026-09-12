#!/usr/bin/env python3
"""One-core variable-token routing parity; no released model or timing."""
import json
from pathlib import Path
import types
import torch
import deepseek_v41_cpu_reference_0910 as cpu
from deepseek_v41_native_bridge_0910 import NativeGemm
from goal_chunk_grouped_moe_0910 import install
from goal_check_grouped_moe_0910 import FixtureStore
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import sha256
BASE=Path(__file__).resolve().parent

class ReferenceRagged:
    def __init__(self,native):self.native=native;self.workers=1;self.calls=[]
    def apply(self,tasks):
        self.calls.append([t[1].shape[0] for t in tasks])
        return [self.native.apply(*task) for task in tasks]

def fixture(full_width=False,experts=8):
    base = Path(__file__).resolve().parent
    native = NativeGemm(base / 'results/deepseek-v41-native-gemm-0910b/libdeepseek-v41-native-gemm.so', 1)
    native.install()
    m = cpu.load_official_model('goal_chunk_grouped_moe_fixture')
    args = m.ModelArgs(**json.loads((cpu.OFFICIAL / 'config.json').read_text()))
    args.dim, args.moe_inter_dim = (5120, 2304) if full_width else (128, 96)
    args.n_routed_experts, args.n_activated_experts = experts, 6
    args.vision_n_layers = 0
    moe = m.MoE(0, args).eval()
    with torch.no_grad():
        for name, p in moe.named_parameters():
            if p.dtype == torch.float8_e8m0fnu:
                p.view(torch.uint8).random_(118, 123)
            elif p.dtype == torch.float4_e2m1fn_x2:
                p.view(torch.uint8).random_(0, 256)
            elif p.dtype == torch.float8_e4m3fn:
                p.copy_((torch.randn(p.shape, dtype=torch.float32) * .7).to(p.dtype))
            else:
                p.copy_((torch.randn(p.shape, dtype=torch.float32) * .03).to(p.dtype))
    store = FixtureStore()
    wrapper_calls = [0]
    for index, expert in enumerate(moe.experts):
        original = expert.forward
        prefix = f'layers.0.ffn.experts.{index}.'

        def resident(this, x, weights=None, _original=original, _prefix=prefix):
            wrapper_calls[0] += 1
            store.activate(this, _prefix)
            return _original(x, weights)

        expert.forward = types.MethodType(resident, expert)
    gate_calls = []
    moe.gate.register_forward_hook(lambda module, inputs, output: gate_calls.append(output[1].clone()))
    runtime = types.SimpleNamespace(module=m, model=types.SimpleNamespace(layers=[types.SimpleNamespace(layer_id=0, ffn=moe)]),
                                    store=store, native=native)
    from goal_native_quant_0910 import NativeQuant
    NativeQuant(BASE/'results/goal_native_quant_0910/libgoal_native_quant_0910.so').install(m)
    active=[True]
    from deepseek_v41_grouped_int16_exact_goal_0910 import GroupedExact16Gemm
    from deepseek_v41_ragged_int16_goal_0910 import RaggedExact16Gemm
    packer=GroupedExact16Gemm(BASE/'results/deepseek-v41-grouped-int16-exact-goal-0910/libdeepseek-v41-grouped-int16-exact.so',workers=1)
    cache={}
    def provider(b,bs):
        key=(b.data_ptr(),bs.data_ptr())
        if key not in cache:cache[key]=packer.pack(b,bs)
        return cache[key]
    backend=RaggedExact16Gemm(BASE/'results/deepseek-v41-ragged-int16-exact-goal-0910/libdeepseek-v41-ragged-int16-exact.so',provider,workers=1)
    task_sizes=[];apply_original=backend.apply
    def apply(tasks):
        task_sizes.append([t[1].shape[0] for t in tasks])
        return apply_original(tasks)
    backend.apply=apply
    candidate=install(runtime,backend,lambda:active[0])
    rows=[]
    with torch.inference_mode():
        for length,factor in [(2,.1),(3,1.),(5,10.),(8,100.)]:
            x=torch.randn(1,length,args.dim).bfloat16()*factor
            candidate.enabled=False
            store.activation_order.clear()
            expected=moe(x)
            activations=list(store.activation_order)
            candidate.enabled=True
            store.activation_order.clear();store.residents.clear();store.ensure_calls.clear()
            before=cpu.COUNTS['act_quant'];gate_before=len(gate_calls);wrapper_before=wrapper_calls[0]
            actual=moe(x)
            assert torch.equal(actual,expected),(length,float((actual-expected).abs().max()))
            assert activations==store.activation_order
            assert len(gate_calls)==gate_before+1 and wrapper_calls[0]==wrapper_before
            assert cpu.COUNTS['act_quant']==before+2
            assert len(store.ensure_calls)==1
            assert sum(task_sizes[-2])==2*length*7
            assert sum(task_sizes[-1])==length*7
            rows.append(dict(length=length,factor=factor,exact=True,tasks=[len(task_sizes[-2]),len(task_sizes[-1])]))
        calls=candidate.calls
        active[0]=False
        assert torch.equal(moe(x),expected)
        assert calls==candidate.calls
        active[0]=True
        y=x[:,:1].contiguous()
        candidate.enabled=False;reference=moe(y);candidate.enabled=True
        assert torch.equal(moe(y),reference) and calls==candidate.calls
        candidate.uninstall()
        assert torch.equal(moe(x),expected)
    return dict(full_width=full_width,experts=experts,rows=rows,exact=True,cold_group_protected=True,fallbacks=True)

def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.bfloat16);torch.manual_seed(4109940)
    sources=[Path(__file__),BASE/'goal_chunk_grouped_moe_0910.py',BASE/'deepseek_v41_ragged_int16_goal_0910.py',
        BASE/'results/deepseek-v41-ragged-int16-exact-goal-0910/libdeepseek-v41-ragged-int16-exact.so']
    hashes={str(p):sha256(p) for p in sources}
    cases=[fixture(False),fixture(True),fixture(False,64)]
    assert all(sha256(p)==h for p,h in hashes.items())
    out=BASE/'results/goal_chunk_grouped_moe_0910';out.mkdir(exist_ok=True)
    result=dict(passed=True,cases=cases,source_sha256=hashes,full_checkpoint_loaded=False,model_tok_s_measured=False,backend='ragged exact16 native grouped projections')
    atomic_json(out/'native-fixture-check.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
