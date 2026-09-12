#!/usr/bin/env python3
"""Bounded one-core ragged ABI/bridge parity; no model or performance test."""
import ctypes
import hashlib
import json
import os
from pathlib import Path

import torch
import deepseek_v41_cpu_reference_0910 as cpu
from deepseek_v41_native_bridge_0910 import NativeGemm
from deepseek_v41_grouped_int16_exact_goal_0910 import GroupedExact16Gemm
from deepseek_v41_ragged_int16_goal_0910 import Descriptor, RaggedExact16Gemm
from check_deepseek_v41_native_vnni_goal_0910 import hostile
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import sha256

BASE = Path(__file__).resolve().parent
OUT = BASE / 'results/deepseek-v41-ragged-int16-exact-goal-0910'


def digest(t):
    return hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_flush_denormal(False)
    torch.manual_seed(419126)
    lib = OUT / 'libdeepseek-v41-ragged-int16-exact.so'
    baseline = NativeGemm(BASE / 'results/deepseek-v41-native-gemm-0910b/libdeepseek-v41-native-gemm.so', 1)
    packer = GroupedExact16Gemm(BASE / 'results/deepseek-v41-grouped-int16-exact-goal-0910/libdeepseek-v41-grouped-int16-exact.so', workers=1)
    cache = {}
    def provider(b, bs):
        key = (b.data_ptr(), bs.data_ptr())
        if key not in cache:
            cache[key] = packer.pack(b, bs)
        return cache[key]
    native = RaggedExact16Gemm(lib, provider, workers=1, fallback=baseline)
    sources = [Path(__file__), lib, BASE / 'deepseek-v41-ragged-int16-exact-goal-0910.cpp',
        BASE / 'deepseek-v41-batched-int16-exact-goal-0910.cpp',
        BASE / 'deepseek-v41-grouped-int16-exact-goal-0910.cpp', BASE / 'deepseek-v41-native-gemm-0910b.cpp',
        BASE / 'deepseek_v41_ragged_int16_goal_0910.py']
    hashes = {str(p): sha256(p) for p in sources}
    cases = []

    def activation(m, k, tiny=False):
        a, asc = cpu.act_quant(torch.randn(m, k).bfloat16(), 32, 'ue8m0', torch.float8_e8m0fnu)
        if tiny:
            a.view(torch.uint8)[0, 1] = 1
        return a, asc

    def weight(mode, n, k):
        b = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8).view(torch.float4_e2m1fn_x2) if mode == 4 else torch.randn(n, k).to(torch.float8_e4m3fn)
        bs = torch.randint(118, 130, (n if mode == 4 else (n + 31) // 32, k // 32), dtype=torch.uint8).view(torch.float8_e8m0fnu)
        return b, bs

    def check(label, tasks):
        cache.clear()
        for mode, _, _, b, bs in tasks:
            if mode == 4:
                provider(b, bs)
        unique = {}
        for _, a, asc, b, bs in tasks:
            for t in (a, asc, b, bs):
                unique[(t.data_ptr(), t.numel())] = t
        for p in cache.values():
            for t in (p.packed, p.scales):
                unique[(t.data_ptr(), t.numel())] = t
        tensors = list(unique.values())
        before = [digest(t) for t in tensors]
        expected = [baseline.apply(*task) for task in tasks]
        groups = {}
        for mode, a, asc, _, _ in tasks:
            if mode == 4:
                groups[(a.data_ptr(), asc.data_ptr(), tuple(a.shape))] = a
        eligible = fallback = 0
        for a in groups.values():
            q = a.float() * 64
            good = (torch.isfinite(q) & (q == q.trunc())).reshape(-1, 32).all(dim=1)
            eligible += int(good.sum())
            fallback += good.numel() - int(good.sum())
        records = []
        for workers in [1, 4]:
            native.workers = workers
            old = (native.eligible_blocks, native.fallback_blocks)
            actual = native.apply(tasks)
            assert len(actual) == len(expected)
            assert all(torch.equal(a.view(torch.int16), b.view(torch.int16)) for a, b in zip(actual, expected)), label
            assert (native.eligible_blocks - old[0], native.fallback_blocks - old[1]) == (eligible, fallback)
            records.append(dict(workers=workers, exact_values=sum(a.numel() for a in actual), unique_eligible_blocks=eligible, unique_fallback_blocks=fallback))
        assert before == [digest(t) for t in tensors]
        case = dict(label=label, task_count=len(tasks), batches=[int(t[1].shape[0]) for t in tasks],
            n=tasks[0][3].shape[0], k=tasks[0][1].shape[1], checks=records,
            source_and_packed_buffers_unchanged=True, activation_pointer_dedup_verified=True)
        cases.append(case)
        print(json.dumps({k:v for k,v in case.items() if k != 'checks'}), flush=True)

    tasks = []
    for m in range(1, 9):
        a, asc = activation(m, 96, tiny=True)
        b4, bs4 = weight(4, 33, 96)
        b8, bs8 = weight(8, 33, 96)
        # FP8 appears first: subsequent FP4 tasks must mark the shared decode.
        tasks.extend([(8, a, asc, b8, bs8), (4, a, asc, b4, bs4), (4, a, asc, b4, bs4)])
    check('mixed_modes_all_m_shared_activation_and_weight_pointers', tasks)

    a, asc = activation(8, 32)
    b4, bs4 = weight(4, 17, 32)
    b8, bs8 = weight(8, 17, 32)
    check('maximum_128_tasks', [(8, a, asc, b8, bs8), (4, a, asc, b4, bs4)] * 64)

    for label, n, k, projections in [('real_gate_up', 2304, 5120, 2), ('real_down', 5120, 2304, 1)]:
        tasks = []
        for m in [1, 2, 3, 4, 6, 8]:
            a, asc = activation(m, k, tiny=(m == 3))
            for _ in range(projections):
                b, bs = weight(4, n, k)
                tasks.append((4, a, asc, b, bs))
        a, asc = activation(8, k)
        for _ in range(projections):
            b, bs = weight(8, n, k)
            tasks.append((8, a, asc, b, bs))
        check(label, tasks)

    a, asc, b4, bs4 = hostile()
    a = torch.cat([a, torch.zeros(6, 32, dtype=torch.uint8).view(torch.float8_e4m3fn)], dim=0)
    a.view(torch.uint8)[2, 0] = 127
    a.view(torch.uint8)[3, 0] = 255
    a.view(torch.uint8)[4, 0] = 1
    asc = torch.tensor([127, 127, 127, 127, 0, 254, 255, 1], dtype=torch.uint8)[:, None].view(torch.float8_e8m0fnu)
    b4 = b4.expand(33, -1).contiguous()
    bs4 = torch.tensor([127] * 30 + [0, 254, 255], dtype=torch.uint8)[:, None].view(torch.float8_e8m0fnu)
    b8 = torch.randint(0, 256, (33, 32), dtype=torch.uint8).view(torch.float8_e4m3fn)
    bs8 = torch.tensor([127, 255], dtype=torch.uint8)[:, None].view(torch.float8_e8m0fnu)
    check('hostile_rounding_nan_tiny_and_scale_edges', [(4, a, asc, b4, bs4), (8, a, asc, b8, bs8)])
    assert native.apply([(4, a, asc, b4, bs4)])[0][:2, 0].float().tolist() == [10752.0, -10752.0]

    # All raw activation byte codes, including signed zeros, subnormals and NaNs.
    for first in range(0, 256, 8):
        raw = torch.arange(first, first + 8, dtype=torch.uint8)[:, None].expand(-1, 32).contiguous()
        a = raw.view(torch.float8_e4m3fn)
        asc = torch.full((8, 1), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)
        check('all_fp8_codes_%03d' % first, [(4, a, asc, b4, bs4)])

    # Whole-call fallback when the externally bounded provider rejects any pack.
    fallback_native = RaggedExact16Gemm(lib, lambda b, bs: None, workers=1, fallback=baseline)
    tasks = [(8, a, asc, b8, bs8), (4, a, asc, b4, bs4)]
    actual = fallback_native.apply(tasks)
    assert all(torch.equal(x.view(torch.int16), baseline.apply(*t).view(torch.int16)) for x, t in zip(actual, tasks))
    assert fallback_native.fallback_calls == 1 and fallback_native.native_calls == 0

    # Invalid descriptors are rejected before touching caller-owned output.
    packed = provider(b4, bs4)
    output = torch.full((8, 33), 123, dtype=torch.bfloat16)
    valid = Descriptor(4, 8, 33, 32, a.data_ptr(), asc.data_ptr(), b4.data_ptr(), bs4.data_ptr(), packed.packed.data_ptr(), packed.scales.data_ptr(), output.data_ptr())
    stats = (ctypes.c_uint64 * 2)()
    rejected = 0
    for field, value in [('mode', 3), ('m', 0), ('m', 9), ('n', 0), ('k', 31), ('a', None), ('packed', None)]:
        bad = Descriptor.from_buffer_copy(valid)
        setattr(bad, field, value)
        assert native.fn(ctypes.pointer(bad), 1, 1, stats) == -1
        rejected += 1
    for count, workers in [(0, 1), (129, 1), (1, 0), (1, 65)]:
        assert native.fn(ctypes.pointer(valid), count, workers, stats) == -1
        rejected += 1
    for field in ['n', 'k']:
        descs = (Descriptor * 2)(valid, valid)
        setattr(descs[1], field, getattr(valid, field) + (1 if field == 'n' else 32))
        assert native.fn(descs, 2, 1, stats) == -1
        rejected += 1
    assert torch.all(output == 123)
    assert all(sha256(p) == h for p, h in hashes.items())
    result = dict(passed=True, cases=cases, source_sha256=hashes,
        exact_values=sum(c['exact_values'] for case in cases for c in case['checks']),
        workers=[1, 4], cpu_affinity=sorted(os.sched_getaffinity(0)), descriptor_bytes=ctypes.sizeof(Descriptor),
        invalid_descriptors_rejected=rejected, whole_call_cap_fallback=True, all_256_fp8_codes=True,
        original_fp8_arithmetic_unchanged=True, hostile_values=[10752.0, -10752.0],
        full_checkpoint_loaded=False, model_tok_s_measured=False)
    atomic_json(OUT / 'kernel-check.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['cases', 'source_sha256']}), flush=True)


if __name__ == '__main__':
    main()
