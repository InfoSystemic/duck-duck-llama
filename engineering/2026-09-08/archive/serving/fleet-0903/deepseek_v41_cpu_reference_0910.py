#!/usr/bin/env python3
"""CPU compatibility layer for the pinned official V4.1 graph.

These kernels retain the publisher's FP8/FP4 block arithmetic and BF16 rounding
points using ordinary CPU PyTorch operations. They are a bring-up path, not an
optimized serving engine. The official graph and its caches are imported intact.
No checkpoint is implicitly downloaded or loaded by this module.
"""
import ast
import importlib.util
from pathlib import Path
import sys
import types

import torch

BASE = Path(__file__).resolve().parent
OFFICIAL = BASE / 'results/deepseek-v41-intake-0910/official/inference'
EXTRA = BASE / 'results/deepseek-v41-cpu-source-0910/inference'
COUNTS = dict(act_quant=0, fp4_act_quant=0, fp8_gemm=0, fp4_gemm=0, sparse_attn=0, hc_split_sinkhorn=0)


def _cpu(*values):
    assert all(value.device.type == 'cpu' for value in values)


def _round_scale(amax, inverse):
    ratio = (amax.float() * inverse).contiguous()
    bits = ratio.view(torch.int32)
    codes = ((bits >> 23) & 255) + ((bits & 0x7fffff) != 0).int()
    assert ((codes >= 0) & (codes < 255)).all()
    return codes.to(torch.uint8).view(torch.float8_e8m0fnu)


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    _cpu(x); COUNTS['act_quant'] += 1
    assert x.shape[-1] % block_size == 0 and block_size in (32, 128)
    blocks = x.float().unflatten(-1, (-1, block_size))
    amax = blocks.abs().amax(-1).clamp_min(1e-4)
    scales = _round_scale(amax, 1 / 448).to(scale_dtype) if scale_fmt is not None else (amax * (1 / 448)).to(scale_dtype)
    quant = (blocks / scales.float()[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    if inplace:
        x.copy_((quant.float() * scales.float()[..., None]).flatten(-2).to(x.dtype))
        return x
    return quant.flatten(-2).contiguous(), scales.contiguous()


def _fp4_codes(x):
    value = x.float().abs().clamp_max(6)
    midpoints = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.], dtype=torch.float32)
    index = torch.bucketize(value.contiguous(), midpoints)
    tie = (index < 7) & (value == midpoints[index.clamp_max(6)])
    index += (tie & ((index & 1) != 0)).long()
    return (index | (torch.signbit(x).long() << 3)).to(torch.uint8)


def _fp4_values(codes):
    values = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.], dtype=torch.float32)
    return values[codes.long()]


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    _cpu(x); COUNTS['fp4_act_quant'] += 1
    assert x.shape[-1] % block_size == 0 and block_size in (16, 32)
    blocks = x.float().unflatten(-1, (-1, block_size))
    if scale_dtype == torch.float8_e4m3fn:
        scales = (blocks.abs().amax(-1).clamp_min(6 * 2**-9) / 6).to(scale_dtype)
    else:
        assert scale_dtype == torch.float8_e8m0fnu
        scales = _round_scale(blocks.abs().amax(-1).clamp_min(6 * 2**-126), 1 / 6)
    codes = _fp4_codes((blocks / scales.float()[..., None]).clamp(-6, 6))
    if inplace:
        x.copy_((_fp4_values(codes) * scales.float()[..., None]).flatten(-2).to(x.dtype))
        return x
    codes = codes.flatten(-2)
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous()
    return packed.view(torch.float4_e2m1fn_x2), scales.contiguous()


def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, block_size=128):
    _cpu(a, a_s, b, b_s); COUNTS['fp8_gemm'] += 1
    assert a.dtype == b.dtype == torch.float8_e4m3fn and block_size in (32, 128)
    k, n = a.shape[-1], b.shape[0]
    assert b.ndim == 2 and b.shape[1] == k and k % block_size == 0
    assert b_s.shape == ((n + block_size-1)//block_size, k//block_size)
    aa, bb = a.reshape(-1, k).float(), b.float()
    sa, sb = a_s.reshape(-1, k//block_size).float(), b_s.float()
    output = torch.zeros(aa.shape[0], n, dtype=torch.float32)
    for block, begin in enumerate(range(0, k, block_size)):
        partial = aa[:, begin:begin+block_size] @ bb[:, begin:begin+block_size].T
        output += (partial * sa[:, block, None]) * sb[:, block].repeat_interleave(block_size)[:n][None]
    return output.reshape(*a.shape[:-1], n).to(torch.get_default_dtype())


def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, act_block_size=128):
    _cpu(a, a_s, b, b_s); COUNTS['fp4_gemm'] += 1
    assert a.dtype == torch.float8_e4m3fn and b.dtype == torch.float4_e2m1fn_x2
    k, n = a.shape[-1], b.shape[0]
    assert b.ndim == 2 and b.shape[1] * 2 == k and k % act_block_size == 0
    assert act_block_size in (32, 128) and b_s.shape == (n, k//32)
    aa, sa, sb = a.reshape(-1, k).float(), a_s.reshape(-1, k//act_block_size).float(), b_s.float()
    packed = b.view(torch.uint8)
    output = torch.zeros(aa.shape[0], n, dtype=torch.float32)
    for block, begin in enumerate(range(0, k, 32)):
        raw = packed[:, begin//2:(begin+32)//2]
        codes = torch.stack([raw & 15, raw >> 4], dim=-1).flatten(-2)
        partial = aa[:, begin:begin+32] @ _fp4_values(codes).T
        output += (partial * sa[:, begin//act_block_size, None]) * sb[:, block][None]
    return output.reshape(*a.shape[:-1], n).to(torch.get_default_dtype())


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    _cpu(mixes, hc_scale, hc_base); COUNTS['hc_split_sinkhorn'] += 1
    h = hc_mult
    assert mixes.dtype == hc_scale.dtype == hc_base.dtype == torch.float32
    assert mixes.shape[-1] == (h+2)*h and sinkhorn_iters >= 1
    pre = torch.sigmoid(mixes[..., :h] * hc_scale[0] + hc_base[:h]) + eps
    post = 2 * torch.sigmoid(mixes[..., h:2*h] * hc_scale[1] + hc_base[h:2*h])
    comb = (mixes[..., 2*h:] * hc_scale[2] + hc_base[2*h:]).unflatten(-1, (h, h))
    comb = torch.exp(comb - comb.amax(-1, keepdim=True))
    comb = comb / comb.sum(-1, keepdim=True) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    _cpu(q, kv, attn_sink, topk_idxs); COUNTS['sparse_attn'] += 1
    assert q.dtype == kv.dtype == torch.bfloat16
    batch, seq, heads, dim = q.shape
    assert kv.shape[0] == batch and kv.shape[-1] == dim and attn_sink.shape == (heads,)
    assert topk_idxs.shape[:2] == (batch, seq) and ((topk_idxs >= -1) & (topk_idxs < kv.shape[1])).all()
    output = torch.empty_like(q)
    # Mirror the publisher's 64-position online softmax, including BF16 rounding
    # of unnormalised probabilities before the value matmul and the sink denominator.
    for b in range(batch):
        for token in range(seq):
            maximum = torch.full((heads,), -1e30, dtype=torch.float32)
            denominator = torch.zeros(heads, dtype=torch.float32)
            numerator = torch.zeros(heads, dim, dtype=torch.float32)
            ids = topk_idxs[b, token]
            for begin in range(0, ids.numel(), 64):
                chunk = ids[begin:begin+64].long()
                valid = chunk >= 0
                values = kv[b, chunk.clamp_min(0)].float().masked_fill(~valid[:, None], 0)
                scores = (q[b, token].float() @ values.T).masked_fill(~valid[None], -torch.inf) * softmax_scale
                next_max = torch.maximum(maximum, scores.amax(-1))
                correction = torch.exp(maximum - next_max)
                probs = torch.exp(scores - next_max[:, None])
                denominator = denominator * correction + probs.sum(-1)
                numerator = numerator * correction[:, None] + probs.to(torch.bfloat16).float() @ values
                maximum = next_max
            denominator += torch.exp(attn_sink - maximum)
            output[b, token] = (numerator / denominator[:, None]).to(q.dtype)
    return output


def load_official_model(module_name='deepseek_v41_cpu_model'):
    """Load the pinned model with CPU kernels; no model object/weights allocated.

    Each imported model module owns its publisher-defined shared-attention state.
    The caller must serialize use of a model module and enforce valid cache positions.
    Image preprocessing is intentionally not installed by this text bring-up loader.
    """
    processor = types.ModuleType('image_processor')
    tree = ast.parse((EXTRA / 'image_processor.py').read_text())
    definitions = [n for n in tree.body if isinstance(n, ast.Assign) and any(
        isinstance(x, ast.Name) and x.id in {'TEXT', 'IMAGE_START', 'IMAGE', 'IMAGE_NEW_LINE', 'IMAGE_END'}
        for target in n.targets for x in ast.walk(target))]
    assert len(definitions) == 2
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(EXTRA / 'image_processor.py'), 'exec'), processor.__dict__)
    previous = {name: sys.modules.get(name) for name in ['kernel', 'image_processor', 'engram', 'vision']}
    old_path = list(sys.path)
    try:
        sys.path.insert(0, str(OFFICIAL))
        sys.modules['kernel'] = sys.modules[__name__]
        sys.modules['image_processor'] = processor
        sys.modules.pop('engram', None); sys.modules.pop('vision', None)
        spec = importlib.util.spec_from_file_location(module_name, OFFICIAL / 'model.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = old_path
        for name, value in previous.items():
            if value is None: sys.modules.pop(name, None)
            else: sys.modules[name] = value
