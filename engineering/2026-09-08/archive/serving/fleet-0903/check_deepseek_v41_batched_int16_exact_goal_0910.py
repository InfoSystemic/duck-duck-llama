#!/usr/bin/env python3
"""One-core batched exact16 parity, eligibility/fallback, and immutable-source checks."""
import ctypes
import hashlib
import json
from pathlib import Path
import torch
import deepseek_v41_cpu_reference_0910 as cpu
from deepseek_v41_native_bridge_0910 import NativeGemm
from deepseek_v41_grouped_int16_exact_goal_0910 import GroupedExact16Gemm
from deepseek_v41_batched_int16_exact_goal_0910 import BatchedExact16Gemm
from check_deepseek_v41_native_vnni_goal_0910 import hostile
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import sha256

BASE = Path(__file__).resolve().parent
OUT = BASE / 'results/deepseek-v41-batched-int16-exact-goal-0910'


def digest(t): return hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()
def bits_equal(a, b): return torch.equal(a.view(torch.int16), b.view(torch.int16))


def main():
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.set_flush_denormal(False); torch.manual_seed(419125)
    lib = OUT / 'libdeepseek-v41-batched-int16-exact.so'
    native = BatchedExact16Gemm(lib, 1)
    baseline = NativeGemm(BASE / 'results/deepseek-v41-native-gemm-0910b/libdeepseek-v41-native-gemm.so', 1)
    grouped = GroupedExact16Gemm(BASE / 'results/deepseek-v41-grouped-int16-exact-goal-0910/libdeepseek-v41-grouped-int16-exact.so', workers=1)
    sources = [Path(__file__), lib, BASE / 'deepseek-v41-batched-int16-exact-goal-0910.cpp',
        BASE / 'deepseek-v41-grouped-int16-exact-goal-0910.cpp', BASE / 'deepseek-v41-native-gemm-0910b.cpp',
        BASE / 'deepseek_v41_batched_int16_exact_goal_0910.py']
    hashes = {str(p): sha256(p) for p in sources}
    cases = []
    def check(label, a, asc, b, bs):
        original = grouped.pack(b, bs)
        packed = native.bind_packed(b, bs, original)
        assert packed.packed.data_ptr() == original.packed.data_ptr()
        tensors = [a, asc, b, bs, packed.packed, packed.scales]
        before = [digest(t) for t in tensors]
        expected = baseline.apply(4, a, asc, b, bs)
        records = []
        for workers in [1, 4]:
            native.workers = workers
            for preallocated in [False, True]:
                output = torch.full_like(expected, float('nan')) if preallocated else None
                actual = native.apply_packed(a, asc, packed, output)
                assert bits_equal(actual, expected), (label, workers, int((actual != expected).sum()))
                if preallocated: assert actual.data_ptr() == output.data_ptr()
                records.append(dict(workers=workers, exact_values=actual.numel(), eligible=native.last_eligible_blocks, fallback=native.last_fallback_blocks))
        assert before == [digest(t) for t in tensors]
        case = dict(label=label, shape=[a.numel() // a.shape[-1], b.shape[0], a.shape[-1]], checks=records,
            source_and_packed_buffers_unchanged=True, existing_pack_storage_reused=True)
        cases.append(case)
        print(json.dumps({k:v for k,v in case.items() if k != 'checks'}), flush=True)
    for m in range(1, 9):
        a, asc = cpu.act_quant(torch.randn(m, 96).bfloat16(), 32, 'ue8m0', torch.float8_e8m0fnu)
        if m > 1: a.view(torch.uint8)[0, 1] = 1
        b = torch.randint(0, 256, (33, 48), dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
        bs = torch.randint(118, 130, (33, 3), dtype=torch.uint8).view(torch.float8_e8m0fnu)
        check('all_batch_sizes_mixed_eligibility', a, asc, b, bs)
    for m, n, k in [(2, 2304, 5120), (4, 5120, 2304), (8, 2304, 5120), (8, 5120, 2304)]:
        a, asc = cpu.act_quant(torch.randn(m, k).bfloat16(), 32, 'ue8m0', torch.float8_e8m0fnu)
        b = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
        bs = torch.randint(118, 130, (n, k // 32), dtype=torch.uint8).view(torch.float8_e8m0fnu)
        check('real_expert_shape', a, asc, b, bs)
    a, asc, b, bs = hostile()
    a = torch.cat([a, torch.zeros(6, 32, dtype=torch.uint8).view(torch.float8_e4m3fn)], 0)
    a.view(torch.uint8)[2, 0] = 127
    a.view(torch.uint8)[3, 0] = 255
    asc = torch.tensor([127, 127, 127, 127, 0, 254, 255, 1], dtype=torch.uint8)[:, None].view(torch.float8_e8m0fnu)
    b = b.expand(33, -1).contiguous()
    bs = torch.tensor([127] * 30 + [0, 254, 255], dtype=torch.uint8)[:, None].view(torch.float8_e8m0fnu)
    check('hostile_ties_nan_activations_scale_edges', a, asc, b, bs)
    packed = native.pack(b, bs)
    actual = native.apply_packed(a, asc, packed)
    assert actual[:2, 0].float().tolist() == [10752.0, -10752.0]
    # Mode8 and oversized batches remain the original baseline function.
    a8, s8 = cpu.act_quant(torch.randn(9, 32).bfloat16(), 32, 'ue8m0', torch.float8_e8m0fnu)
    w8 = torch.randn(33, 32).to(torch.float8_e4m3fn)
    sw8 = torch.full((2, 1), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    assert bits_equal(native.apply(8, a8, s8, w8, sw8), baseline.apply(8, a8, s8, w8, sw8))
    assert bits_equal(native.apply(4, a8, s8, b, bs), baseline.apply(4, a8, s8, b, bs))
    assert all(sha256(p) == h for p, h in hashes.items())
    result = dict(passed=True, cases=cases, source_sha256=hashes, exact_values=sum(c['exact_values'] for case in cases for c in case['checks']),
        batches=list(range(1, 9)), workers=[1, 4], cpu_affinity=[0], hostile_values=[10752.0, -10752.0],
        fp8_baseline_unchanged=True, oversized_batch_baseline_fallback=True, full_checkpoint_loaded=False, model_tok_s_measured=False)
    atomic_json(OUT / 'kernel-check.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['cases','source_sha256']}), flush=True)


if __name__ == '__main__': main()
