#!/usr/bin/env python3
"""Warm component A/B only; packing cost is separately reported, never model tok/s."""
import json
from pathlib import Path
import time
import torch
import deepseek_v41_cpu_reference_0910 as cpu
from deepseek_v41_native_bridge_0910 import NativeGemm
from deepseek_v41_native_vnni_goal_0910 import NativeVnniGemm
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import sha256

BASE = Path(__file__).resolve().parent
OUT = BASE / 'results/deepseek-v41-native-vnni-goal-0910'


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(419113)
    baseline_path = BASE / 'results/deepseek-v41-native-gemm-0910b/libdeepseek-v41-native-gemm.so'
    library = OUT / 'libdeepseek-v41-native-vnni.so'
    proof = json.loads((OUT / 'kernel-check.json').read_text())
    assert proof['passed'] and all(sha256(p) == h for p, h in proof['source_sha256'].items())
    native, baseline = NativeVnniGemm(library, 1), NativeGemm(baseline_path, 1)
    cases = []
    result = dict(passed=False, component_only=True, full_checkpoint_loaded=False,
        model_tok_s_measured=False, matrices_hot_repeated=True, repeats=128, cases=cases,
        source_sha256={str(p): sha256(p) for p in [Path(__file__), library, baseline_path,
            BASE / 'deepseek_v41_native_vnni_goal_0910.py', BASE / 'deepseek-v41-native-vnni-goal-0910.cpp']})
    for n, k in [(2304, 5120), (5120, 2304)]:
        a, asc = cpu.act_quant(torch.randn(1, k).bfloat16(), 32, 'ue8m0', torch.float8_e8m0fnu)
        b = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
        bs = torch.randint(118, 130, (n, k // 32), dtype=torch.uint8).view(torch.float8_e8m0fnu)
        for workers in [1, 4, 8, 16]:
            native.workers = baseline.workers = workers
            began = time.perf_counter()
            packed = native.pack(b, bs)
            pack_seconds = time.perf_counter() - began
            reference = baseline.apply(4, a, asc, b, bs)
            candidate = native.apply_packed(a, asc, packed)
            # This sample's BF16 equality is reported, not required by the experimental algorithm.
            differences = int((reference.view(torch.int16) != candidate.view(torch.int16)).sum())
            arms = []
            for name in ['baseline', 'vnni', 'vnni', 'baseline']:
                fn = (lambda: baseline.apply(4, a, asc, b, bs)) if name == 'baseline' else (lambda: native.apply_packed(a, asc, packed))
                for _ in range(3):
                    fn()
                began = time.perf_counter()
                for _ in range(128):
                    fn()
                arms.append(dict(variant=name, seconds_per_call=(time.perf_counter() - began) / 128))
            base_seconds = (arms[0]['seconds_per_call'] + arms[3]['seconds_per_call']) / 2
            candidate_seconds = (arms[1]['seconds_per_call'] + arms[2]['seconds_per_call']) / 2
            case = dict(n=n, k=k, workers=workers, pack_seconds=pack_seconds,
                native_weight_bytes=b.numel(), native_scale_bytes=bs.numel(),
                packed_weight_bytes=packed.packed.numel(), sum_metadata_bytes=packed.sums.numel() * 2,
                packed_scale_bytes=packed.scales.numel(), packed_total_bytes=packed.storage_bytes,
                extra_resident_bytes_with_original_retained=packed.storage_bytes,
                bf16_bit_differences=differences, arms=arms, baseline_seconds=base_seconds,
                vnni_seconds=candidate_seconds, speedup=base_seconds / candidate_seconds)
            cases.append(case)
            atomic_json(OUT / 'benchmark-check.json', result)
            print(json.dumps({key: value for key, value in case.items() if key != 'arms'}), flush=True)
    assert all(sha256(p) == h for p, h in result['source_sha256'].items())
    result['passed'] = True
    atomic_json(OUT / 'benchmark-check.json', result)


if __name__ == '__main__':
    main()
