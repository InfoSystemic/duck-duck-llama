#!/usr/bin/env python3
"""Correctness-only exact-input VNNI experiment; numerical differences are reported.

Checks lossless nibble packing, metadata, exact-integer block oracle, random and
real matrix shapes, special scale codes, NaN propagation and hostile BF16 ties.
It does not load checkpoint tensors or measure model/kernel performance.
"""
import hashlib
import json
from pathlib import Path
import torch
import deepseek_v41_cpu_reference_0910 as cpu
from deepseek_v41_native_bridge_0910 import NativeGemm
from deepseek_v41_native_vnni_goal_0910 import NativeVnniGemm
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import sha256

BASE = Path(__file__).resolve().parent
OUT = BASE / 'results/deepseek-v41-native-vnni-goal-0910'


def packed_codes(codes):
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous().view(torch.float4_e2m1fn_x2)


def integer_oracle(a, asc, b, bsc):
    m, k, n = a.numel() // a.shape[-1], a.shape[-1], b.shape[0]
    aa = a.float().reshape(m, k)
    nan = aa.isnan()
    aa = torch.where(nan, 0, aa).mul(512).to(torch.int64)
    raw = b.view(torch.uint8)
    codes = torch.stack([raw & 15, raw >> 4], -1).flatten(-2)
    bb = (cpu._fp4_values(codes) * 2).to(torch.int64)
    sa, sb = asc.float().reshape(m, k // 32), bsc.float()
    output = torch.zeros(m, n, dtype=torch.float32)
    for block in range(k // 32):
        begin = block * 32
        exact = (aa[:, None, begin:begin + 32] * bb[None, :, begin:begin + 32]).sum(-1)
        assert int(exact.abs().max()) <= 88080384
        partial = exact.to(torch.float32) * (1 / 1024)
        partial = torch.where(nan[:, begin:begin + 32].any(-1)[:, None], float('nan'), partial)
        output += (partial * sa[:, block, None]) * sb[None, :, block]
    return output.to(torch.bfloat16).reshape(*a.shape[:-1], n)


def same(a, b):
    return bool(torch.equal(a.isnan(), b.isnan()) and torch.equal(a[~a.isnan()].view(torch.int16), b[~b.isnan()].view(torch.int16)))


def differences(reference, actual):
    assert torch.equal(reference.isnan(), actual.isnan())
    finite = reference.isfinite() & actual.isfinite()
    rf, af = reference[finite].float(), actual[finite].float()
    error = (rf - af).abs()
    ru, au = reference[finite].view(torch.int16).to(torch.int32) & 65535, actual[finite].view(torch.int16).to(torch.int32) & 65535
    rorder = torch.where((ru & 32768) != 0, (~ru) & 65535, ru | 32768)
    aorder = torch.where((au & 32768) != 0, (~au) & 65535, au | 32768)
    return dict(values=reference.numel(), finite_values=int(finite.sum()),
        bf16_bit_differences=int((reference.view(torch.int16) != actual.view(torch.int16)).sum()),
        finite_value_differences=int((rf != af).sum()), max_abs_error=float(error.max()) if error.numel() else 0,
        max_relative_error=float((error / rf.abs().clamp_min(1e-30)).max()) if error.numel() else 0,
        max_bf16_ulp_distance=int((rorder - aorder).abs().max()) if error.numel() else 0,
        zero_reference_differences=int(((rf == 0) & (af != 0)).sum()),
        nonfinite_class_differences=int((reference.isinf() != actual.isinf()).sum()))


def check_pack(native, b, bs):
    packed = native.pack(b, bs)
    n, k, tiles, blocks = b.shape[0], b.shape[1] * 2, (b.shape[0] + 15) // 16, b.shape[1] // 16
    reconstructed = packed.packed.reshape(tiles, blocks, 8, 16, 2).permute(0, 3, 1, 2, 4).reshape(tiles * 16, k // 2)
    assert torch.equal(reconstructed[:n], b.view(torch.uint8))
    assert (reconstructed[n:] == 0).all()
    reconstructed_scale = packed.scales.permute(0, 2, 1).reshape(tiles * 16, blocks)
    assert torch.equal(reconstructed_scale[:n], bs.view(torch.uint8))
    codes = torch.stack([reconstructed & 15, reconstructed >> 4], -1).flatten(-2)
    expected_sum = (cpu._fp4_values(codes) * 2).to(torch.int32).reshape(tiles, 16, blocks, 32).sum(-1).permute(0, 2, 1)
    assert torch.equal(packed.sums.to(torch.int32), expected_sum)
    assert int(packed.sums.abs().max()) <= 384
    return packed


def hostile():
    a = torch.zeros((2, 32), dtype=torch.uint8)
    codes = torch.zeros((1, 32), dtype=torch.uint8)
    for position in range(0, 16, 2):
        a[:, position] = 126  # +448
        codes[0, position] = 7  # +6
    for position in [1, 3, 5, 7]:
        a[:, position] = 126
        codes[0, position] = 15  # -6
    a[:, 9] = 96  # +32
    codes[0, 9] = 2  # +1
    a[:, 16] = 1  # +1/512
    codes[0, 16] = 1  # +1/2
    a[1] ^= 128  # Symmetric negative case, including signed zeros.
    return a.view(torch.float8_e4m3fn), torch.full((2, 1), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu), packed_codes(codes), torch.full((1, 1), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_flush_denormal(False)
    torch.manual_seed(411910)
    library = OUT / 'libdeepseek-v41-native-vnni.so'
    baseline_path = BASE / 'results/deepseek-v41-native-gemm-0910b/libdeepseek-v41-native-gemm.so'
    native, baseline = NativeVnniGemm(library, 1), NativeGemm(baseline_path, 1)
    sources = [Path(__file__), BASE / 'deepseek-v41-native-vnni-goal-0910.cpp',
        BASE / 'deepseek-v41-native-gemm-0910b.cpp', BASE / 'deepseek_v41_native_vnni_goal_0910.py', library, baseline_path]
    hashes = {str(p): sha256(p) for p in sources}
    cases = []
    def check(label, a, asc, b, bs):
        inputs = [hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest() for t in [a, asc, b, bs]]
        packed = check_pack(native, b, bs)
        actual = native.apply_packed(a, asc, packed)
        oracle = integer_oracle(a, asc, b, bs)
        assert same(actual, oracle), (label, int((actual != oracle).sum()))
        reference = baseline.apply(4, a, asc, b, bs)
        metrics = differences(reference, actual)
        assert inputs == [hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest() for t in [a, asc, b, bs]]
        case = dict(label=label, shape=[a.numel() // a.shape[-1], b.shape[0], a.shape[-1]],
            integer_oracle_exact=True, native_nibble_bits_preserved=True, metadata_exact=True,
            source_inputs_preserved=True, native_weight_bytes=b.numel(), packed_weight_bytes=packed.packed.numel(),
            sum_metadata_bytes=packed.sums.numel() * 2, scale_bytes=packed.scales.numel(), **metrics)
        if label == 'hostile_bf16_ties':
            case.update(baseline_values=reference.flatten().float().tolist(), candidate_values=actual.flatten().float().tolist())
            assert reference.flatten().float().tolist() == [10752.0, -10752.0]
            assert actual.flatten().float().tolist() == [10816.0, -10816.0]
        cases.append(case)
        print(json.dumps(case), flush=True)
    for m, n, k in [(1, 33, 32), (3, 129, 128), (1, 2304, 5120), (1, 5120, 2304), (6, 2304, 5120)]:
        a, asc = cpu.act_quant(torch.randn(m, k, dtype=torch.float32).bfloat16(), 32, 'ue8m0', torch.float8_e8m0fnu)
        b = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
        bs = torch.randint(118, 130, (n, k // 32), dtype=torch.uint8).view(torch.float8_e8m0fnu)
        check('random_real_shape', a, asc, b, bs)
    check('hostile_bf16_ties', *hostile())
    a = torch.arange(256, dtype=torch.uint8).reshape(8, 32)
    a[(a & 127) == 127] = 126
    b = torch.randint(0, 256, (33, 16), dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
    asc = torch.tensor([0, 1, 63, 126, 127, 190, 253, 254], dtype=torch.uint8).reshape(8, 1).view(torch.float8_e8m0fnu)
    bs = torch.tensor([0, 1, 63, 126, 127, 128, 190, 253, 254, 255, 127] * 3, dtype=torch.uint8).reshape(33, 1).view(torch.float8_e8m0fnu)
    check('all_finite_fp8_and_scale_edges', a.view(torch.float8_e4m3fn), asc, b, bs)
    a = torch.tensor([127, 255] + [0] * 30, dtype=torch.uint8).reshape(1, 32).view(torch.float8_e4m3fn)
    check('nan_activation', a, torch.full((1, 1), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu), b,
        torch.full((33, 1), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu))
    # FP8 weight fallback must retain baseline bits as well.
    a, asc = cpu.act_quant(torch.randn(1, 128).bfloat16(), 32, 'ue8m0', torch.float8_e8m0fnu)
    b8 = torch.randn(33, 128).to(torch.float8_e4m3fn)
    bs8 = torch.full((2, 4), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    assert same(native.apply(8, a, asc, b8, bs8), baseline.apply(8, a, asc, b8, bs8))
    assert all(sha256(p) == h for p, h in hashes.items())
    result = dict(passed=True, workers=1, cases=cases, source_sha256=hashes,
        integer_oracle_values=sum(c['values'] for c in cases), fp8_fallback_exact=True,
        fp32_reduction_changed=True, baseline_bit_differences=sum(c['bf16_bit_differences'] for c in cases),
        full_checkpoint_loaded=False, model_tok_s_measured=False, component_only=True,
        limitation='Experimental: preserves exact native values, but exact integer within-block summation differs from baseline FP32 tree; no model quality or throughput conclusion.')
    atomic_json(OUT / 'kernel-check.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ['cases', 'source_sha256']}), flush=True)


if __name__ == '__main__':
    main()
