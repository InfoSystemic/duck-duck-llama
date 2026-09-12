#!/usr/bin/env python3
"""Bounded grouped-VNNI parity, exact-tree oracle, ownership and whole-call fallback checks."""
import gc
import hashlib
import json
import os
import time
from pathlib import Path
import weakref
import torch
import deepseek_v41_cpu_reference_0910 as cpu
from deepseek_v41_native_bridge_0910 import NativeGemm
from deepseek_v41_lattice16_0910 import GroupedLattice16
from deepseek_v41_native_grouped_goal_0910 import GroupedNativeGemm
from check_deepseek_v41_native_gemm_0910 import tree_oracle
from deepseek_v41_native_vnni_goal_0910 import NativeVnniGemm
from check_deepseek_v41_native_vnni_goal_0910 import integer_oracle, hostile, same
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import sha256

BASE = Path(__file__).resolve().parent
OUT = BASE / 'results/deepseek-v41-lattice16-0910'


def digest(t):
    return hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_flush_denormal(False)
    torch.manual_seed(419117)
    lib = OUT / 'libdeepseek-v41-lattice16.so'
    single = NativeVnniGemm(BASE / 'results/deepseek-v41-native-vnni-goal-0910/libdeepseek-v41-native-vnni.so', 1)
    baseline = NativeGemm(BASE / 'results/deepseek-v41-native-gemm-0910b/libdeepseek-v41-native-gemm.so', 1)
    sources = [Path(__file__), lib, BASE / 'deepseek-v41-native-lattice16-0910.cpp', BASE / 'deepseek_v41_lattice16_0910.py', BASE / 'deepseek-v41-native-grouped-vnni-goal-0910.cpp',
        BASE / 'deepseek-v41-native-vnni-goal-0910.cpp', BASE / 'deepseek-v41-native-gemm-0910b.cpp',
        BASE / 'deepseek_v41_native_grouped_vnni_goal_0910.py', BASE / 'deepseek_v41_native_vnni_goal_0910.py']
    hashes = {str(p): sha256(p) for p in sources}
    cases = []; benchmarks = []
    oracle_values = 0
    for count, n, k, shared in [(3, 33, 32, True), (7, 65, 96, False), (14, 2304, 5120, True), (7, 5120, 2304, False)]:
        activations = [cpu.act_quant(torch.randn(1, k).bfloat16(), 32, 'ue8m0', torch.float8_e8m0fnu)
            for _ in range(1 if shared else count)]
        tasks, cache = [], {}
        for index in range(count):
            mode = 8 if index == count - 1 or (count == 14 and index == 12) else 4
            a, asc = activations[0 if shared else index]
            raw = torch.randint(0, 256, (n, k // 2 if mode == 4 else k), dtype=torch.uint8)
            if mode == 8:
                raw[(raw & 127) == 127] = 126
            b = raw.view(torch.float4_e2m1fn_x2 if mode == 4 else torch.float8_e4m3fn)
            bs = torch.randint(118, 130, (n if mode == 4 else (n + 31) // 32, k // 32), dtype=torch.uint8).view(torch.float8_e8m0fnu)
            tasks.append((mode, a, asc, b, bs))
            if mode == 4:
                cache[(b.data_ptr(), bs.data_ptr())] = single.pack(b, bs)
        def provider(b, bs):
            return cache[(b.data_ptr(), bs.data_ptr())]
        grouped = GroupedLattice16(lib, provider, workers=1)
        expected = torch.cat([baseline.apply(mode, a, asc, b, bs)
            for mode, a, asc, b, bs in tasks], 0)
        for index, (mode, a, asc, b, bs) in enumerate(tasks):
            if mode == 4 and (n < 130 or index == 0):
                oracle = tree_oracle(4, a, asc, b, bs)
                assert same(expected[index:index + 1], oracle)
                oracle_values += oracle.numel()
        tensors = [t for task in tasks for t in task[1:]] + [t for packed in cache.values() for t in (packed.packed, packed.sums, packed.scales)]
        before = [digest(t) for t in tensors]
        checks = []
        for workers in [1, 4, 16]:
            grouped.workers = workers
            for preallocated in [False, True]:
                output = torch.full_like(expected, float('nan')) if preallocated else None
                actual = grouped.apply(tasks, output)
                assert same(actual, expected), (count, n, k, workers)
                if preallocated:
                    assert output.data_ptr() == actual.data_ptr()
                checks.append(dict(workers=workers, preallocated=preallocated, exact_values=actual.numel()))
        assert before == [digest(t) for t in tensors]
        cases.append(dict(tasks=count, n=n, k=k, shared_activation=shared, checks=checks,
            source_and_packed_buffers_preserved=True, native_calls=grouped.native_calls, fallback_calls=grouped.fallback_calls))
        print(json.dumps({key: value for key, value in cases[-1].items() if key != 'checks'}), flush=True)
        if n > 1000:
            grouped.workers = 16
            control = GroupedNativeGemm(BASE / 'results/deepseek-v41-native-grouped-goal-0910/libdeepseek-v41-native-grouped.so', 16)
            arms = []
            for name in ['baseline', 'lattice16', 'lattice16', 'baseline']:
                fn = control.apply if name == 'baseline' else grouped.apply
                for _ in range(5): fn(tasks)
                began = time.perf_counter()
                for _ in range(64): fn(tasks)
                arms.append(dict(variant=name, seconds_per_call=(time.perf_counter()-began)/64))
            durations = [r['seconds_per_call'] for r in arms]
            benchmark = dict(n=n,k=k,tasks=count,workers=16,arms=arms,component_only=True,
                eligible_blocks=[grouped.eligible_blocks(a) for a, _ in activations],
                total_blocks=[a.numel()//32 for a, _ in activations],
                speedup=(durations[0]+durations[3])/(durations[1]+durations[2]),
                control_ratio=max(durations[0],durations[3])/min(durations[0],durations[3]))
            benchmarks.append(benchmark); print(json.dumps(benchmark),flush=True)

    # A second provider miss must route the ENTIRE call through baseline, even
    # when the first task has a packed tensor and distinguishable rounding.
    a, asc, b, bs = hostile()
    tasks = [(4, a[i:i + 1], asc[i:i + 1], b, bs) for i in range(2)]
    refs, calls = [], [0]
    def capped(b, bs):
        calls[0] += 1
        if calls[0] == 2:
            return None
        value = single.pack(b, bs)
        refs.append(weakref.ref(value))
        return value
    capped_grouped = GroupedLattice16(lib, capped, workers=1)
    output = capped_grouped.apply(tasks)
    assert output.flatten().float().tolist() == [10752.0, -10752.0]
    assert capped_grouped.fallback_calls == 1 and capped_grouped.native_calls == 0
    gc.collect()
    assert all(ref() is None for ref in refs), 'Fallback bridge retained packed tensors'
    refs.clear()
    def ephemeral(b, bs):
        value = single.pack(b, bs)
        refs.append(weakref.ref(value))
        return value
    grouped = GroupedLattice16(lib, ephemeral, workers=1)
    output = grouped.apply(tasks)
    assert output.flatten().float().tolist() == [10752.0, -10752.0]
    gc.collect()
    assert all(ref() is None for ref in refs), 'Native bridge retained packed tensors'
    assert all(sha256(p) == h for p, h in hashes.items())
    result = dict(passed=True, cases=cases, benchmarks=benchmarks, source_sha256=hashes,
        exact_values=sum(check['exact_values'] for case in cases for check in case['checks']),
        independent_tree_oracle_values=oracle_values, workers=[1, 4, 16], cpu_affinity=sorted(os.sched_getaffinity(0)),
        whole_call_cap_fallback_exact=True, no_packed_tensor_retention=True,
        hostile_native_values=[10752.0, -10752.0], hostile_fallback_values=[10752.0, -10752.0],
        fp32_tree_preserved=True, full_checkpoint_loaded=False, model_tok_s_measured=False, component_only=True)
    atomic_json(OUT / 'kernel-check.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ['cases', 'source_sha256']}), flush=True)


if __name__ == '__main__':
    main()
