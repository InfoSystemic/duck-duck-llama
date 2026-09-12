#!/usr/bin/env python3
"""Full-width HC/RMSNorm batch-vs-token rounding audit, Torch16 on one CPU.

No checkpoint is loaded and no performance rate is measured. The publisher's
methods are imported unchanged; only F.linear is split exactly as the frozen
chunk verifier does. Every intermediate is compared bitwise to token-serial.
"""
import hashlib
import json
import os
from pathlib import Path
import types

import torch
import torch.nn.functional as F

import deepseek_v41_cpu_reference_0910 as cpu
from goal_hc_0910 import NativeHC


BASE = Path(__file__).resolve().parent


def parity(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    raw_actual = actual.contiguous().view(torch.uint8).reshape(actual.numel(), actual.element_size())
    raw_expected = expected.contiguous().view(torch.uint8).reshape(expected.numel(), expected.element_size())
    different = (raw_actual != raw_expected).any(-1)
    indices = different.nonzero().flatten()
    first = int(indices[0]) if indices.numel() else None
    row = dict(exact_bits=first is None, different_elements=int(different.sum()),
               elements=actual.numel(), dtype=str(actual.dtype), shape=list(actual.shape),
               max_abs=float((actual.float() - expected.float()).abs().max()), first_flat_index=first)
    if first is not None:
        row.update(actual=float(actual.flatten()[first]), expected=float(expected.flatten()[first]))
    return row


def concatenate_traces(traces):
    return {key: torch.cat([row[key] for row in traces], dim=1) for key in traces[0]}


def check_traces(actual, expected):
    details = {key: parity(value, expected[key]) for key, value in actual.items()}
    first = next((key for key, value in details.items() if not value['exact_bits']), None)
    return dict(exact_bits=first is None, first_different_operator=first, operators=details)


def main():
    assert os.sched_getaffinity(0) == {127}, 'Run under taskset -c 127; do not consume other cores'
    torch.set_num_threads(16)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.bfloat16)
    model_module = cpu.load_official_model('deepseek_hc_batch16_fixture')
    native_hc = NativeHC(BASE / 'results/goal_hc_0910/libgoal_hc_0910.so')
    native_hc.install(model_module)
    block = types.SimpleNamespace(norm_eps=1e-6, hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6)
    original_linear = F.linear

    def strict_linear(x, weight, bias=None):
        if x.numel() == x.shape[-1]:
            return original_linear(x, weight, bias)
        flat = x.reshape(-1, x.shape[-1])
        return torch.cat([original_linear(flat[i:i+1], weight, bias)
                          for i in range(flat.shape[0])], 0).reshape(*x.shape[:-1], weight.shape[0])

    def hc_trace(x, weight, scale, base, sublayer):
        flat = x.flatten(2).float()
        square = flat.square()
        mean = square.mean(-1, keepdim=True)
        rsqrt = torch.rsqrt(mean + block.norm_eps)
        projected = strict_linear(flat, weight)
        mixes = projected * rsqrt
        pre, post, comb = model_module.hc_split_sinkhorn(mixes, scale, base, 4, 20, 1e-6)
        actual = model_module.Block.hc_mixes(block, x, weight, scale, base)
        assert all(parity(a, b)['exact_bits'] for a, b in zip(actual, (pre, post, comb)))
        return dict(hc_square=square, hc_mean=mean, hc_rsqrt=rsqrt, hc_strict_projection=projected,
                    hc_mixes_before_sinkhorn=mixes, hc_pre_mix=pre, hc_post_mix=post, hc_comb_mix=comb,
                    hc_pre=model_module.Block.hc_pre(block, x, pre),
                    hc_post=model_module.Block.hc_post(block, sublayer, x, post, comb))

    def norm_trace(norm, x):
        floating = x.float()
        square = floating.square()
        mean = square.mean(-1, keepdim=True)
        rsqrt = torch.rsqrt(mean + norm.eps)
        normalized = floating * rsqrt
        weighted = norm.weight * normalized
        cast = weighted.to(x.dtype)
        assert parity(norm(x), cast)['exact_bits']
        return dict(rms_square=square, rms_mean=mean, rms_rsqrt=rsqrt,
                    rms_normalized=normalized, rms_weighted=weighted, rms_output=cast)

    cases = []
    paths = [Path(__file__), BASE / 'goal_hc_0910.py', cpu.OFFICIAL / 'model.py',
             BASE / 'deepseek_v41_cpu_reference_0910.py',
             BASE / 'goal_chunk_verify_0910.py', BASE / 'results/goal_hc_0910/libgoal_hc_0910.so']
    source_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    F.linear = strict_linear
    try:
        with torch.inference_mode():
            for seed, distribution in [(4100940, 'normal'), (4100941, 'dynamic')]:
                torch.manual_seed(seed)
                weight = torch.randn(24, 20480, dtype=torch.float32) * 0.01
                scale = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float32)
                base = torch.randn(24, dtype=torch.float32) * 0.1
                for length in (2, 3, 5, 8):
                    x = torch.randn(1, length, 4, 5120, dtype=torch.float32)
                    if distribution == 'dynamic':
                        exponents = torch.randint(-10, 11, x.shape)
                        x = x * torch.pow(torch.tensor(2.0), exponents)
                    x = x.bfloat16()
                    sublayer = torch.randn(1, length, 5120, dtype=torch.float32).bfloat16()
                    serial = concatenate_traces([hc_trace(x[:, i:i+1], weight, scale, base, sublayer[:, i:i+1])
                                                 for i in range(length)])
                    batch = hc_trace(x, weight, scale, base, sublayer)
                    row = dict(component='HC', distribution=distribution, sequence=length,
                               dim=5120, hc_mult=4, **check_traces(batch, serial))
                    # Isolate pre/post reduction shapes from any coefficient drift.
                    common_pre = model_module.Block.hc_pre(block, x, serial['hc_pre_mix'])
                    common_post = model_module.Block.hc_post(block, sublayer, x,
                                                            serial['hc_post_mix'], serial['hc_comb_mix'])
                    row['shared_coefficient_hc_pre'] = parity(common_pre, serial['hc_pre'])
                    row['shared_coefficient_hc_post'] = parity(common_post, serial['hc_post'])
                    # Isolate normalization drift from scalar-vs-vector Sinkhorn.
                    token_sinkhorn = [model_module.hc_split_sinkhorn(batch['hc_mixes_before_sinkhorn'][:, i:i+1].contiguous(),
                                                                     scale, base, 4, 20, 1e-6)
                                      for i in range(length)]
                    row['per_token_sinkhorn'] = {name: parity(torch.cat([value[index] for value in token_sinkhorn], 1), serial[name])
                        for index, name in enumerate(['hc_pre_mix', 'hc_post_mix', 'hc_comb_mix'])}
                    cases.append(row)
                    print(json.dumps({key: row[key] for key in ['component', 'distribution', 'sequence', 'first_different_operator']}), flush=True)
                    for dim in (512, 1280, 5120, 8192):
                        norm = model_module.RMSNorm(dim).eval()
                        norm.weight.copy_((1 + torch.randn(dim, dtype=torch.float32) * .1).bfloat16())
                        z = torch.randn(1, length, dim, dtype=torch.float32)
                        if distribution == 'dynamic':
                            z *= torch.pow(torch.tensor(2.0), torch.randint(-10, 11, z.shape))
                        z = z.bfloat16()
                        expected = concatenate_traces([norm_trace(norm, z[:, i:i+1]) for i in range(length)])
                        actual = norm_trace(norm, z)
                        cases.append(dict(component='RMSNorm', distribution=distribution, sequence=length,
                                          dim=dim, **check_traces(actual, expected)))
    finally:
        F.linear = original_linear
    assert source_hashes == {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    result = dict(completed=True, all_exact_bits=all(row['exact_bits'] for row in cases),
                  torch_threads=torch.get_num_threads(), affinity=sorted(os.sched_getaffinity(0)),
                  no_full_checkpoint=True, no_performance_measurement=True, source_sha256=source_hashes,
                  native_hc_calls=native_hc.native_calls, native_hc_fallback_calls=native_hc.fallback_calls,
                  first_differences=[{key: row[key] for key in ['component', 'distribution', 'sequence', 'dim', 'first_different_operator']}
                                     for row in cases if not row['exact_bits']], cases=cases)
    out = BASE / 'results/goal-hc-batch16-0910'
    out.mkdir(exist_ok=True)
    (out / 'check.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'cases'}, indent=2))


if __name__ == '__main__':
    main()
