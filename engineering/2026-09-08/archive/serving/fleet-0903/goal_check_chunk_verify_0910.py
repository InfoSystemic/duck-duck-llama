#!/usr/bin/env python3
"""Reduced-width 40-layer causal chunk and rollback correctness fixtures."""
import json
from pathlib import Path
import torch
import deepseek_v41_cpu_reference_0910 as cpu
from check_deepseek_v41_engram_hash_0910 import TokenizerAdapter
from deepseek_v41_native_bridge_0910 import NativeGemm
from goal_chunk_verify_0910 import ChunkVerifier
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import sha256
BASE=Path(__file__).resolve().parent

def main():
    paths=[Path(__file__),BASE/'goal_chunk_verify_0910.py',BASE/'goal_hc_0910.py',
           BASE/'goal_native_quant_0910.py',BASE/'goal_sparse_script_0910.py',
           BASE/'results/goal_hc_0910/libgoal_hc_0910.so',
           BASE/'results/goal_native_quant_0910/libgoal_native_quant_0910.so']
    hashes={str(p):sha256(p) for p in paths}
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.set_default_device('cpu'); torch.set_default_dtype(torch.bfloat16); torch.manual_seed(4100910)
    native=NativeGemm(BASE/'results/deepseek-v41-native-gemm-0910b/libdeepseek-v41-native-gemm.so',1);native.install()
    m = cpu.load_official_model('deepseek_chunk_fixture')
    config_path = cpu.OFFICIAL / 'config.json'
    published = json.loads(config_path.read_text())
    args = m.ModelArgs(**published)
    # Keep all 40 layers, ownership patterns, Engram layer locations, and the
    # hierarchical indexer. Reduce matrix widths and expert count for a bounded fixture.
    args.max_batch_size, args.max_seq_len = 1, 32
    args.temperature = 0
    args.dim, args.moe_inter_dim = 64, 64
    args.n_heads, args.head_dim, args.rope_head_dim = 4, 32, 16
    args.q_lora_rank, args.o_lora_rank, args.o_groups = 32, 32, 1
    args.n_routed_experts, args.n_activated_experts = 4, 2
    args.index_n_heads, args.index_head_dim, args.index_topk = 2, 32, 4
    args.window_size = 8
    args.candidate_topk_blocks, args.candidate_block_size = 2, 4
    args.original_seq_len = 0
    args.engram_vocab_size, args.engram_head_dim = 17, 32
    args.dspark_block_size, args.n_mtp_layers, args.dspark_target_layer_ids = 0, 0, ()
    args.vision_n_layers = 0
    layout = m.EngramLayout.from_args(args)
    args.engram_num_embeddings = tuple(sum(p for group in layer for p in group) for layer in layout.primes)
    model = m.Transformer(args, TokenizerAdapter()).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dtype == torch.float8_e8m0fnu:
                p.view(torch.uint8).fill_(120)
            elif p.dtype == torch.float4_e2m1fn_x2:
                p.view(torch.uint8).random_(0, 256)
            elif p.dtype == torch.float8_e4m3fn:
                p.copy_((torch.randn(p.shape, dtype=torch.float32) * .7).to(p.dtype))
            elif 'norm.weight' in name or name.endswith(('q_weight', 'k_weight')):
                p.fill_(1)
            elif name.endswith(('hc_attn_scale', 'hc_ffn_scale')):
                p.fill_(.1)
            else:
                p.copy_((torch.randn(p.shape, dtype=torch.float32) * .03).to(p.dtype))
    assert len(model.layers) == 40
    assert [x.layer_id for x in model.layers if x.engram is not None] == [1, 14]
    assert [x.layer_id for x in model.layers if x.attn.is_kv_source] == [2, 8, 14, 20]
    assert [x.layer_id for x in model.layers if x.attn.is_index_source] == [2, 8, 14, 20, 24, 28, 32, 36]
    from goal_hc_0910 import NativeHC
    from goal_native_quant_0910 import NativeQuant
    from goal_sparse_script_0910 import install as install_sparse
    NativeHC(BASE/'results/goal_hc_0910/libgoal_hc_0910.so').install(m)
    NativeQuant(BASE/'results/goal_native_quant_0910/libgoal_native_quant_0910.so').install(m)
    install_sparse(m)
    verifier=ChunkVerifier(model,m)
    cases=[]
    out=BASE/'results/goal_chunk_verify_0910'
    out.mkdir(exist_ok=True)
    with torch.inference_mode():
        for topk,start,length in [(t,s,n) for t in (32,4) for s,n in [(1,2),(6,3),(7,6),(8,4),(15,6),(24,7),(7,8),(24,8)]]:
            for layer in model.layers:
                if layer.attn.indexer is not None:layer.attn.indexer.index_topk=topk
            torch.manual_seed(9041+start)
            prompt=torch.randint(1,1000,(1,start),dtype=torch.int64)
            draft=torch.randint(1,1000,(1,length),dtype=torch.int64)
            model(prompt,0)
            initial=verifier.snapshot()
            reference=[]
            for i in range(length):
                reference.append(model(draft[:,i:i+1],start+i)[1])
            expected=torch.stack(reference,1)
            sequential=verifier.snapshot()
            shared_expected={k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in vars(m.shared_attn).items()}
            initial.restore()
            token,actual,_=verifier.verify(draft,start)
            different=int((actual!=expected).sum())
            error=float((actual-expected).abs().max())
            cache_differences=[]
            for (a,ref),(_,_) in zip(sequential.buffers,initial.buffers):
                if not torch.equal(a,ref):cache_differences.append(dict(shape=list(a.shape),count=int((a!=ref).sum())))
            cases.append(dict(topk=topk,start=start,length=length,mode=verifier.last_mode,exact_logits=torch.equal(actual,expected),different_logits=different,max_abs=error,tokens_equal=torch.equal(token,expected.argmax(-1)),cache_differences=cache_differences))
            print(json.dumps(cases[-1]),flush=True)
            atomic_json(out/'fixture-check.json',dict(passed=False,cases=cases))
            assert torch.equal(actual,expected),{k:v for k,v in cases[-1].items() if k!='cache_differences'}
            assert not cache_differences,cases[-1]
            assert set(vars(m.shared_attn))==set(shared_expected)
            assert all(torch.equal(getattr(m.shared_attn,k),v) if isinstance(v,torch.Tensor)
                       else getattr(m.shared_attn,k) is v for k,v in shared_expected.items())
            assert all(getattr(m.shared_attn,k).data_ptr()==v.data_ptr()
                       for k,v in sequential.shared_values.items() if k in ('index_k','compress_kv'))
            initial.restore()
            verifier.verify(draft,start,rollback=True)
            assert all(torch.equal(a,b) for a,b in initial.buffers)
            assert all(getattr(m.shared_attn,k) is v for k,v in initial.shared_values.items())
        verifier.uninstall()
    assert all(sha256(p)==h for p,h in hashes.items())
    atomic_json(out/'fixture-check.json',dict(passed=True,cases=cases,rollback_exact=True,shared_runtime_exact=True,full_checkpoint_loaded=False,model_tok_s_measured=False,source_sha256=hashes))

if __name__=='__main__':main()
