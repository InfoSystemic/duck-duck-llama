#!/usr/bin/env python3
"""Validate exact VNNI selection/fallback and FP4 matrix outputs at native precision."""
import ctypes
import json
from pathlib import Path
import sys
import time
import torch
import check_deepseek_v41_native_gemm_0910b as earlier
import deepseek_v41_cpu_reference_0910 as cpu
from deepseek_v41_native_bridge_0910 import NativeGemm
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import sha256

BASE = Path(__file__).resolve().parent


def main():
    out = Path(sys.argv[1]); earlier.main()
    candidate_path = out / 'libdeepseek-v41-native-gemm.so'
    baseline_path = BASE / 'results/deepseek-v41-native-gemm-0910b/libdeepseek-v41-native-gemm.so'
    baseline = NativeGemm(baseline_path, 16); candidate = NativeGemm(candidate_path, 16)
    count = candidate.library.ds41_integer_blocks
    count.argtypes = [ctypes.c_void_p, ctypes.c_int]; count.restype = ctypes.c_int
    torch.set_num_threads(1); torch.manual_seed(411002)
    cases = []
    result = dict(passed=False, cases=cases, exact_product_bound=28672 * 12,
        exact_32_product_sum_bound=32 * 28672 * 12, exact_integer_limit=2**24,
        source_sha256={str(p): sha256(p) for p in [Path(__file__), candidate_path, baseline_path,
                                                 BASE / 'deepseek-v41-native-gemm-0910d.cpp']},
        component_only=True, full_checkpoint_loaded=False)
    inputs = [torch.arange(256, dtype=torch.uint8)[:, None].expand(256, 32).contiguous().view(torch.float8_e4m3fn),
              torch.tensor([126, 254, 8, 136] * 8, dtype=torch.uint8)[None].view(torch.float8_e4m3fn),
              torch.tensor(list(range(32)), dtype=torch.uint8)[None].view(torch.float8_e4m3fn)]
    for m, k in [(1, 5120), (1, 2304), (6, 5120)]:
        x = torch.randn(m, k, dtype=torch.float32).bfloat16()
        inputs.append(cpu.act_quant(x, 32, 'ue8m0', torch.float8_e8m0fnu)[0])
    for index, a in enumerate(inputs):
        m, k = a.shape; n = 67 if index < 3 else 5120 if k == 2304 else 2304
        scale = torch.full((m, k // 32), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)
        w = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
        ws = torch.randint(118, 130, (n, k // 32), dtype=torch.uint8).view(torch.float8_e8m0fnu)
        scaled = a.float().reshape(-1, 32) * 64
        eligible = int((torch.isfinite(scaled) & (scaled == scaled.trunc())).all(-1).sum())
        assert count(a.data_ptr(), a.numel()) == eligible
        expected = baseline.apply(4, a, scale, w, ws)
        for workers in [1, 4, 16]:
            candidate.workers = workers
            actual = candidate.apply(4, a, scale, w, ws)
            assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)), (index, workers)
        candidate.workers = 16
        row = dict(index=index, m=m, n=n, k=k, integer_blocks=eligible,
                   total_blocks=a.numel() // 32, exact_values=expected.numel() * 3)
        if index >= 3:
            arms = []
            for name, native in [('baseline', baseline), ('vnni', candidate), ('vnni', candidate), ('baseline', baseline)]:
                for _ in range(100):
                    native.apply(4, a, scale, w, ws)
                began = time.perf_counter()
                for _ in range(100):
                    native.apply(4, a, scale, w, ws)
                arms.append(dict(path=name, seconds_per_call=(time.perf_counter() - began) / 100))
            row['timings'] = arms
            row['speedup'] = (arms[0]['seconds_per_call'] + arms[3]['seconds_per_call']) / (arms[1]['seconds_per_call'] + arms[2]['seconds_per_call'])
            row['controls_stable_within_5_percent'] = max(arms[0]['seconds_per_call'], arms[3]['seconds_per_call']) / min(arms[0]['seconds_per_call'], arms[3]['seconds_per_call']) <= 1.05
        cases.append(row); atomic_json(out / 'integer-check.json', result)
        print(json.dumps(row), flush=True)
    assert all(sha256(p) == h for p, h in result['source_sha256'].items())
    result.update(passed=True, exact_values=sum(c['exact_values'] for c in cases),
                  nan_and_signed_zero_bits_match=True, mixed_eligibility_fallback_tested=True)
    atomic_json(out / 'integer-check.json', result)


if __name__ == '__main__':
    main()
