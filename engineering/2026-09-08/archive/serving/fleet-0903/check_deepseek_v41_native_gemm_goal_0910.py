#!/usr/bin/env python3
"""Exact decoding, block reduction and graph checks for the register-LUT candidate.

Correctness only by default. --benchmark records isolated hot-matrix timings;
these component timings are never checkpoint generation throughput.
"""
import argparse
import ctypes
import json
import math
from pathlib import Path
import statistics
import struct
import sys
import time
import torch
import deepseek_v41_cpu_reference_0910 as cpu
from check_deepseek_v41_native_gemm_0910 import tree_oracle
from deepseek_v41_native_bridge_0910 import NativeGemm
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import sha256

BASE = Path(__file__).resolve().parent


def variant(path, tile, workers):
    native = NativeGemm(path, workers)
    if tile:
        signature = native.fn.argtypes
        native.fn = getattr(native.library, 'ds41_gemm_t' + str(tile))
        native.fn.argtypes = signature
        native.fn.restype = ctypes.c_int
    return native


def equal_bits(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--graph', action='store_true')
    args = parser.parse_args()
    assert 1 <= args.workers <= 64
    out = args.out
    candidate = out / 'libdeepseek-v41-native-gemm.so'
    baseline = BASE / 'results/deepseek-v41-native-gemm-0910b/libdeepseek-v41-native-gemm.so'
    inputs = [Path(__file__), candidate, baseline, BASE / 'deepseek-v41-native-gemm-goal-0910.cpp']
    hashes = {str(p): sha256(p) for p in inputs}
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(410912)
    natives = {tile: variant(candidate if tile else baseline, tile, args.workers) for tile in [0, 1, 2, 4, 8]}
    fn = natives[4].library.ds41_decode8
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    fn.restype = ctypes.c_int
    codes = (ctypes.c_uint8 * 256)(*range(256))
    values = (ctypes.c_float * 256)()
    assert fn(codes, values, 256) == 0
    for c, v in enumerate(values):
        mag = c & 127
        if mag == 127:
            assert math.isnan(v)
            continue
        e, f = mag >> 3, mag & 7
        expected = math.ldexp(f, -9) if e == 0 else math.ldexp(8 + f, e - 10)
        if c & 128:
            expected = -expected
        assert struct.pack('<f', v) == struct.pack('<f', expected), (c, v, expected)
    cases = []
    shapes = [(4, 1, 33, 32), (4, 3, 129, 128), (4, 1, 2304, 5120),
              (4, 6, 2304, 5120), (4, 1, 5120, 2304), (8, 1, 33, 32),
              (8, 3, 129, 128), (8, 1, 1280, 5120), (8, 1, 32768, 1280),
              (8, 1, 5120, 8192), (8, 1, 2304, 5120), (8, 1, 5120, 2304),
              (8, 1, 25600, 6144)]
    result = dict(passed=False, cases=cases, source_sha256=hashes, workers=args.workers,
        fp8_decode_exact_finite_codes=254, fp8_decode_nan_codes=2, signed_zero_preserved=True,
        component_only=True, full_checkpoint_loaded=False, hot_repeated_matrix_timings=args.benchmark)
    for mode, m, n, k in shapes:
        x = torch.randn(m, k, dtype=torch.float32).bfloat16()
        a, asc = cpu.act_quant(x, 32, 'ue8m0', torch.float8_e8m0fnu)
        raw = torch.randint(0, 256, (n, k // 2 if mode == 4 else k), dtype=torch.uint8)
        if mode == 8:
            raw[(raw & 127) == 127] = 126
        b = raw.view(torch.float4_e2m1fn_x2 if mode == 4 else torch.float8_e4m3fn)
        bs = torch.randint(118, 130, (n if mode == 4 else (n + 31) // 32, k // 32), dtype=torch.uint8).view(torch.float8_e8m0fnu)
        reference = natives[0].apply(mode, a, asc, b, bs)
        oracle_checked = n < 130
        if oracle_checked:
            assert equal_bits(reference, tree_oracle(mode, a, asc, b, bs))
        checks = []
        for tile in [1, 2, 4, 8]:
            for workers in sorted(set([1, args.workers])):
                native = natives[tile]
                native.workers = workers
                output = native.apply(mode, a, asc, b, bs)
                assert equal_bits(reference, output), (mode, m, n, k, tile, workers, int((reference != output).sum()))
                checks.append(dict(tile=tile, workers=workers, exact_values=output.numel()))
            native.workers = args.workers
        timings = []
        if args.benchmark and n >= 1280:
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
        cases.append(dict(mode=mode, m=m, n=n, k=k, checks=checks, timings=timings, independent_tree=oracle_checked))
        atomic_json(out / ('benchmark-check.json' if args.benchmark else 'kernel-check.json'), result)
        print(json.dumps(dict(shape=[mode, m, n, k], exact=True, speedups={t['tile']: round(t['speedup'], 3) for t in timings})), flush=True)
    if args.graph:
        import check_deepseek_v41_cpu_graph_0910 as graph
        natives[4].install()
        original_argv = sys.argv
        original_setter = torch.set_num_interop_threads
        try:
            sys.argv = [str(Path(graph.__file__)), str(out)]
            torch.set_num_interop_threads = lambda n: None if n == torch.get_num_interop_threads() == 1 else (_ for _ in ()).throw(AssertionError(n))
            graph.main()
        finally:
            sys.argv = original_argv
            torch.set_num_interop_threads = original_setter
        current = json.loads((out / 'graph-check.json').read_text())
        prior = json.loads((baseline.parent / 'graph-check.json').read_text())
        assert prior['checks'] == current['checks']
        result['official_synthetic_graph_logits_unchanged'] = True
    assert all(sha256(p) == h for p, h in hashes.items())
    result.update(passed=True, exact_values=sum(x['exact_values'] for case in cases for x in case['checks']))
    atomic_json(out / ('benchmark-check.json' if args.benchmark else 'kernel-check.json'), result)


if __name__ == '__main__':
    main()
