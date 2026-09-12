#!/usr/bin/env python3
"""Check row-tiled native kernels exactly, then compare matched matrix timings."""
import json
from pathlib import Path
import statistics
import sys
import time
import torch
import check_deepseek_v41_native_gemm_0910b as earlier
import deepseek_v41_cpu_reference_0910 as cpu
from deepseek_v41_native_bridge_0910 import NativeGemm
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import sha256

BASE = Path(__file__).resolve().parent


def variant(path, tile, workers=16):
    native = NativeGemm(path, workers)
    if tile:
        signature = native.fn.argtypes
        native.fn = getattr(native.library, 'ds41_gemm_t' + str(tile))
        native.fn.argtypes = signature; native.fn.restype = __import__('ctypes').c_int
    return native


def main():
    out = Path(sys.argv[1])
    earlier.main()  # Independent reduction oracle, FP8 format, and reduced official graph.
    candidate = out / 'libdeepseek-v41-native-gemm.so'
    baseline = BASE / 'results/deepseek-v41-native-gemm-0910b/libdeepseek-v41-native-gemm.so'
    natives = {tile: variant(candidate if tile else baseline, tile) for tile in [0, 1, 2, 4, 8]}
    torch.manual_seed(410912)
    cases = []
    shapes = [(4, 1, 33, 32), (4, 3, 129, 128), (4, 1, 2304, 5120), (4, 6, 2304, 5120),
              (4, 1, 5120, 2304), (8, 1, 33, 32), (8, 3, 129, 128), (8, 1, 1280, 5120),
              (8, 1, 32768, 1280), (8, 1, 5120, 8192), (8, 1, 2304, 5120),
              (8, 1, 5120, 2304), (8, 1, 25600, 6144)]
    result = dict(passed=False, cases=cases, source_sha256={str(p): sha256(p) for p in
        [Path(__file__), candidate, baseline, BASE / 'deepseek-v41-native-gemm-0910c.cpp']},
        component_only=True, hot_repeated_matrix_timings=True, full_checkpoint_loaded=False)
    for mode, m, n, k in shapes:
        x = torch.randn(m, k, dtype=torch.float32).bfloat16()
        a, asc = cpu.act_quant(x, 32, 'ue8m0', torch.float8_e8m0fnu)
        raw = torch.randint(0, 256, (n, k // 2 if mode == 4 else k), dtype=torch.uint8)
        if mode == 8:
            raw[(raw & 127) == 127] = 126
        b = raw.view(torch.float4_e2m1fn_x2 if mode == 4 else torch.float8_e4m3fn)
        bs = torch.randint(118, 130, (n if mode == 4 else (n + 31) // 32, k // 32), dtype=torch.uint8).view(torch.float8_e8m0fnu)
        reference = natives[0].apply(mode, a, asc, b, bs)
        checks = []
        for tile in [1, 2, 4, 8]:
            for workers in [1, 4, 16]:
                native = natives[tile]; native.workers = workers
                output = native.apply(mode, a, asc, b, bs)
                assert torch.equal(reference, output), (mode, m, n, k, tile, workers)
                checks.append(dict(tile=tile, workers=workers, exact_values=output.numel()))
            native.workers = 16
        timings = []
        if n >= 1280:
            repeats = max(5, min(60, (256 << 20) // raw.numel()))
            for tile in [1, 2, 4, 8]:
                arms = []
                for which in [0, tile, tile, 0]:
                    native = natives[which]
                    native.apply(mode, a, asc, b, bs)
                    began = time.perf_counter()
                    for _ in range(repeats):
                        native.apply(mode, a, asc, b, bs)
                    arms.append(dict(tile=which, seconds_per_call=(time.perf_counter() - began) / repeats))
                control = (arms[0]['seconds_per_call'] + arms[3]['seconds_per_call']) / 2
                optimized = (arms[1]['seconds_per_call'] + arms[2]['seconds_per_call']) / 2
                timings.append(dict(tile=tile, repeats=repeats, arms=arms, speedup=control / optimized))
        cases.append(dict(mode=mode, m=m, n=n, k=k, checks=checks, timings=timings))
        atomic_json(out / 'tile-check.json', result)
        print(json.dumps(dict(shape=[mode, m, n, k], speedups={t['tile']: round(t['speedup'], 3) for t in timings})), flush=True)
    assert all(sha256(p) == h for p, h in result['source_sha256'].items())
    result.update(passed=True, exact_values=sum(x['exact_values'] for case in cases for x in case['checks']))
    atomic_json(out / 'tile-check.json', result)


if __name__ == '__main__':
    main()
