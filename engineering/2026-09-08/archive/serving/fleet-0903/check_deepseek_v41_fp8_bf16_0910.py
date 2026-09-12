#!/usr/bin/env python3
"""Prove FP8 expansion/reduction parity and measure bounded component A/B/B/A."""
import json
from pathlib import Path
import sys
import threading
import time

import torch
import deepseek_v41_cpu_reference_0910 as cpu
from check_deepseek_v41_native_gemm_0910 import tree_oracle
from deepseek_v41_fp8_bf16_0910 import ExpandedFP8
from deepseek_v41_native_bridge_0910 import NativeGemm
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import sha256

BASE = Path(__file__).resolve().parent


def main():
    out = Path(sys.argv[1])
    library = out / 'libdeepseek-v41-fp8-bf16.so'
    baseline_path = BASE / 'results/deepseek-v41-native-gemm-0910b/libdeepseek-v41-native-gemm.so'
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(411013)
    candidate, baseline = ExpandedFP8(library, 1), NativeGemm(baseline_path, 1)
    paths = [Path(__file__), library, baseline_path, BASE / 'deepseek_v41_fp8_bf16_0910.py',
             BASE / 'deepseek-v41-native-fp8-bf16-0910.cpp', BASE / 'deepseek-v41-native-gemm-0910b.cpp']
    result = dict(passed=False, component_only=True, model_tok_s_measured=False,
                  expanded_weight_bytes_per_native_byte=2, source_sha256={str(p): sha256(p) for p in paths},
                  cases=[], benchmarks=[], exact_values=0)
    try:
        codes = torch.arange(256, dtype=torch.uint8).reshape(8, 32).view(torch.float8_e4m3fn)
        packed = candidate.pack(codes)
        finite = torch.isfinite(codes.float())
        assert torch.equal(packed.float()[finite], codes.float()[finite])
        assert torch.equal(torch.isnan(packed.float()), torch.isnan(codes.float()))
        assert torch.equal(torch.signbit(packed.float()[finite]), torch.signbit(codes.float()[finite]))
        result['all_fp8_codes_preserved'] = True

        shapes = [(1, 33, 32), (3, 67, 96), (6, 129, 128), (1, 512, 5120),
                  (1, 1280, 5120), (1, 2304, 5120), (1, 32768, 1280),
                  (1, 5120, 8192), (6, 512, 5120)]
        for m, n, k in shapes:
            a, asc = cpu.act_quant(torch.randn(m, k).bfloat16(), 32, 'ue8m0', torch.float8_e8m0fnu)
            weight = (torch.randn(n, k) * 2).to(torch.float8_e4m3fn)
            scale = torch.randint(118, 130, ((n + 31) // 32, k // 32), dtype=torch.uint8).view(torch.float8_e8m0fnu)
            packed = candidate.pack(weight)
            reference = baseline.apply(8, a, asc, weight, scale)
            if n < 130:
                assert torch.equal(reference, tree_oracle(8, a, asc, weight, scale))
            for workers in (1, 16):
                candidate.workers = workers
                for tile in (1, 2, 4):
                    candidate.tile = tile
                    actual = candidate.apply_packed(a, asc, packed, scale)
                    assert torch.equal(actual.view(torch.int16), reference.view(torch.int16)), (m, n, k, workers, tile)
                    result['exact_values'] += actual.numel()
            result['cases'].append(dict(m=m, n=n, k=k, workers=[1, 16], tiles=[1, 2, 4], exact=True))
            atomic_json(out / 'kernel-check.json', result)

        # Include every FP8 code (NaN and signed zero included), all E8M0 codes,
        # and rows crossing FP8 block-scale boundaries.
        a = codes[:1].clone()
        weight = codes.repeat(5, 1)
        scale = torch.tensor([[127], [0]], dtype=torch.uint8).view(torch.float8_e8m0fnu)
        packed = candidate.pack(weight)
        candidate.workers = 1
        for code in range(256):
            a.view(torch.uint8).fill_(code)
            asc = torch.tensor([[code]], dtype=torch.uint8).view(torch.float8_e8m0fnu)
            reference = baseline.apply(8, a, asc, weight, scale)
            actual = candidate.apply_packed(a, asc, packed, scale)
            assert torch.equal(actual.view(torch.int16), reference.view(torch.int16)), code
            result['exact_values'] += actual.numel()
        result['hostile_formats_exact'] = True

        # Cache byte cap and ownership are independently checked before reuse.
        small = weight.clone()
        candidate.cap_bytes = small.numel() * 2
        assert candidate.get_packed(small) is candidate.get_packed(small)
        assert candidate.get_packed(small.clone()) is None
        assert candidate.packed_bytes == candidate.cap_bytes
        candidate.clear()
        assert candidate.packed_bytes == 0 and not candidate.cache
        result['cache_cap_checked'] = True

        for n, k in [(512, 5120), (1280, 5120), (32768, 1280), (5120, 8192)]:
            a, asc = cpu.act_quant(torch.randn(1, k).bfloat16(), 32, 'ue8m0', torch.float8_e8m0fnu)
            weight = torch.randn(n, k).to(torch.float8_e4m3fn)
            scale = torch.full(((n + 31) // 32, k // 32), 125, dtype=torch.uint8).view(torch.float8_e8m0fnu)
            started = time.perf_counter()
            packed = candidate.pack(weight)
            pack_seconds = time.perf_counter() - started
            candidate.workers = baseline.workers = 16
            for tile in (1, 2, 4):
                candidate.tile = tile
                row = dict(n=n, k=k, workers=16, tile=tile, pack_seconds=pack_seconds,
                           source_weight_bytes=weight.numel(), expanded_bytes=packed.numel() * 2,
                           repeated_hot_matrices=True, arms=[])
                errors = []

                def request_thread():
                    try:
                        torch.set_num_threads(16)
                        for name in ('baseline', 'expanded', 'expanded', 'baseline'):
                            fn = (lambda: baseline.apply(8, a, asc, weight, scale)) if name == 'baseline' else (
                                lambda: candidate.apply_packed(a, asc, packed, scale))
                            for _ in range(10):
                                fn()
                            began = time.perf_counter()
                            for _ in range(128):
                                fn()
                            row['arms'].append(dict(variant=name, seconds_per_call=(time.perf_counter() - began) / 128))
                    except BaseException as error:
                        errors.append(error)

                thread = threading.Thread(target=request_thread)
                thread.start(); thread.join()
                if errors:
                    raise errors[0]
                arms = [v['seconds_per_call'] for v in row['arms']]
                row.update(speedup=(arms[0] + arms[3]) / (arms[1] + arms[2]),
                           control_ratio=max(arms[0], arms[3]) / min(arms[0], arms[3]))
                result['benchmarks'].append(row)
                atomic_json(out / 'kernel-check.json', result)
                print(json.dumps({k:v for k,v in row.items() if k != 'arms'}), flush=True)
        assert all(sha256(p) == h for p, h in result['source_sha256'].items())
        result['passed'] = True
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        atomic_json(out / 'kernel-check.json', result)


if __name__ == '__main__':
    main()
