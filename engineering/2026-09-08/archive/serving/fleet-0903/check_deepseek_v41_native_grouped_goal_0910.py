#!/usr/bin/env python3
"""Correctness-only grouped projection checks against original b native outputs."""
import ctypes
import json
from pathlib import Path
import torch
import deepseek_v41_cpu_reference_0910 as cpu
from deepseek_v41_native_bridge_0910 import NativeGemm
from deepseek_v41_native_grouped_goal_0910 import GroupedNativeGemm, Descriptor
from check_deepseek_v41_native_gemm_0910 import tree_oracle
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import sha256

BASE = Path(__file__).resolve().parent
OUT = BASE / 'results/deepseek-v41-native-grouped-goal-0910'


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(419103)
    baseline_path = BASE / 'results/deepseek-v41-native-gemm-0910b/libdeepseek-v41-native-gemm.so'
    baseline = NativeGemm(baseline_path, 1)
    grouped = [GroupedNativeGemm(OUT / name, 4) for name in
        ['libdeepseek-v41-native-grouped.so', 'libdeepseek-v41-native-grouped-lut.so']]
    sources = [Path(__file__), BASE / 'deepseek-v41-native-grouped-goal-0910.cpp',
        BASE / 'deepseek-v41-native-gemm-goal-0910.cpp', BASE / 'deepseek_v41_native_grouped_goal_0910.py',
        baseline_path] + [Path(g.library._name) for g in grouped]
    hashes = {str(p): sha256(p) for p in sources}
    cases = []
    shapes = [(1, 33, 32, False), (3, 129, 128, True), (7, 65, 96, False),
              (14, 2304, 5120, True), (7, 5120, 2304, False)]
    for count, n, k, shared in shapes:
        activations = []
        for _ in range(1 if shared else count):
            x = torch.randn(1, k, dtype=torch.float32).bfloat16()
            activations.append(cpu.act_quant(x, 32, 'ue8m0', torch.float8_e8m0fnu))
        tasks = []
        for index in range(count):
            a, asc = activations[0 if shared else index]
            mode = 8 if index == count - 1 or (count == 14 and index == 12) else 4
            raw = torch.randint(0, 256, (n, k // 2 if mode == 4 else k), dtype=torch.uint8)
            if mode == 8:
                raw[(raw & 127) == 127] = 126
            b = raw.view(torch.float4_e2m1fn_x2 if mode == 4 else torch.float8_e4m3fn)
            bs = torch.randint(118, 130, (n if mode == 4 else (n + 31) // 32, k // 32),
                dtype=torch.uint8).view(torch.float8_e8m0fnu)
            tasks.append((mode, a, asc, b, bs))
        expected = torch.cat([baseline.apply(*task) for task in tasks], 0)
        if n < 130:
            oracle = torch.cat([tree_oracle(*task) for task in tasks], 0)
            assert torch.equal(expected.view(torch.int16), oracle.view(torch.int16))
        input_hashes = [[sha256_bytes(tensor) for tensor in task[1:]] for task in tasks]
        checked = 0
        for native in grouped:
            for workers in [1, 4]:
                native.workers = workers
                for preallocated in [False, True]:
                    output = torch.full_like(expected, float('nan')) if preallocated else None
                    actual = native.apply(tasks, output)
                    if preallocated:
                        assert actual.data_ptr() == output.data_ptr()
                    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16)), (count, n, k, workers, native.library._name)
                    checked += expected.numel()
        assert input_hashes == [[sha256_bytes(tensor) for tensor in task[1:]] for task in tasks]
        cases.append(dict(tasks=count, n=n, k=k, shared_activations=shared,
            modes=[task[0] for task in tasks], exact_values=checked, input_buffers_preserved=True))
        print(json.dumps(cases[-1]), flush=True)
    d = (Descriptor * 1)()
    for native in grouped:
        assert native.fn(d, 0, 1) == -1
        assert native.fn(d, 65, 1) == -1
        assert native.fn(d, 1, 0) == -1
        assert native.fn(d, 1, 65) == -1
        assert native.fn(d, 1, 1) == -1
    assert all(sha256(p) == h for p, h in hashes.items())
    result = dict(passed=True, cases=cases, source_sha256=hashes,
        exact_values=sum(c['exact_values'] for c in cases), workers=[1, 4],
        decoding_variants=['baseline_b', 'register_lut'], descriptor_rejections=10,
        full_checkpoint_loaded=False, model_tok_s_measured=False, component_only=True)
    atomic_json(OUT / 'kernel-check.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ['cases', 'source_sha256']}), flush=True)


def sha256_bytes(tensor):
    import hashlib
    return hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()


if __name__ == '__main__':
    main()
