#!/usr/bin/env python3
"""Run the official 40-layer control flow with small, synthetic CPU weights.

This is a graph-integration fixture. It does not run the released checkpoint and
must not be reported as DeepSeek model generation, quality, or throughput.
"""
import dataclasses
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

import deepseek_v41_cpu_reference_0910 as cpu
from check_deepseek_v41_engram_hash_0910 import TokenizerAdapter
from qwen_split_trial import sha256

BASE = Path(__file__).resolve().parent


def main():
    out = Path(sys.argv[1])
    assert not (out / 'graph-check.json').exists()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.set_default_device('cpu'); torch.set_default_dtype(torch.bfloat16); torch.manual_seed(4100910)
    m = cpu.load_official_model()
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
    prompt = torch.tensor([[42, 121, 9123, 109, 23, 652, 11111]], dtype=torch.int64)
    checks = []
    with torch.inference_mode():
        before = dict(cpu.COUNTS)
        token, logits, hidden = model(prompt, 0)
        assert token.shape == (1,) and logits.shape == (1, args.vocab_size) and hidden is None
        assert torch.isfinite(logits).all()
        initial = logits.clone()
        checks.append(dict(phase='prefill', positions=prompt.shape[1], finite=True, logits_sha256=hashlib.sha256(logits.numpy().tobytes()).hexdigest()))
        for position, fixed_token in enumerate([77, 86, 900, 16], prompt.shape[1]):
            x = torch.tensor([[fixed_token]], dtype=torch.int64)
            token, logits, hidden = model(x, position)
            assert token.shape == (1,) and torch.isfinite(logits).all() and hidden is None
            checks.append(dict(phase='decode', position=position, finite=True,
                logits_sha256=hashlib.sha256(logits.numpy().tobytes()).hexdigest()))
        token, reset_logits, hidden = model(prompt, 0)
        assert torch.equal(reset_logits, initial), float((reset_logits-initial).abs().max())
        counts = {k: cpu.COUNTS[k]-before[k] for k in before}
        assert all(v > 0 for v in counts.values()), counts
        assert counts['sparse_attn'] == 40 * 6 and counts['hc_split_sinkhorn'] == 80 * 6
        assert m.shared_attn.candidates is not None
        assert m.shared_attn.topk_idxs is not None and m.shared_attn.index_k is not None
    paths = [Path(__file__), Path(cpu.__file__), cpu.OFFICIAL/'model.py', cpu.OFFICIAL/'kernel.py',
        cpu.OFFICIAL/'vision.py', cpu.OFFICIAL/'engram.py', config_path, cpu.EXTRA/'image_processor.py',
        BASE/'check_deepseek_v41_engram_hash_0910.py']
    result = dict(passed=True, finished=time.time(), layers=40, device='cpu', checks=checks,
        kernel_calls=counts, fixture_args=dataclasses.asdict(args), exact_reset_after_decode=True,
        engram_layers=[1,14], kv_source_layers=[2,8,14,20], index_source_layers=[2,8,14,20,24,28,32,36],
        hierarchical_candidates_executed=True,
        parameter_storage_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
        input_sha256={str(p):sha256(p) for p in paths}, full_checkpoint_loaded=False,
        released_model_generated_text=False, model_tok_s_measured=False, vision_executed=False, dspark_executed=False,
        scope='Full 40-layer official text control flow with reduced-width synthetic weights. Validates CPU graph plumbing and fresh-request reset only; no model accuracy or performance claim.')
    (out/'graph-check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['passed','layers','kernel_calls','exact_reset_after_decode','full_checkpoint_loaded','released_model_generated_text']}),flush=True)


if __name__ == '__main__':
    main()
