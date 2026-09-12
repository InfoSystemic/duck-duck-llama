#!/usr/bin/env python3
"""Tiny CPU32 fixture: exact pack bytes, physical nodes, budgets and lifetime."""
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import weakref

import torch
import deepseek_v41_cpu_reference_0910 as cpu
from deepseek_v41_native_bridge_0910 import NativeGemm
from deepseek_v41_native_grouped_goal_0910 import GroupedNativeGemm
from deepseek_v41_ragged_int16_goal_0910 import RaggedExact16Gemm
from deepseek_v41_grouped_int16_exact_goal_0910 import PackedFP4Exact16
from goal_exact16_runtime_0910 import install
import goal_exact16_packed_numa_0910 as helper
from goal_numa_0910 import PAGE_SIZE, query_page_nodes
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import sha256

BASE = Path(__file__).resolve().parent
OUT = BASE / 'results/goal-exact16-packed-numa-0910'


def digest(tensor):
    return hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()


def expected_locations(metadata):
    pages = []
    bounds = metadata['actual_byte_boundaries']
    for node, start, end in zip(metadata['nodes'], bounds, bounds[1:]):
        pages.extend([node] * ((end - start) // PAGE_SIZE))
    return pages


def main():
    assert os.sched_getaffinity(0) == {32}, 'Run this bounded fixture with taskset -c 32'
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(419127)
    baseline = NativeGemm(BASE / 'results/deepseek-v41-native-gemm-0910b/libdeepseek-v41-native-gemm.so', 1)
    grouped = SimpleNamespace(enabled=True, backend=GroupedNativeGemm(
        BASE / 'results/deepseek-v41-native-grouped-goal-0910/libdeepseek-v41-native-grouped.so', 1))
    runtime = SimpleNamespace(native=baseline, store=SimpleNamespace(residents={}, release=lambda prefix: None))
    candidate = install(runtime, SimpleNamespace(grouped=grouped), cap_bytes=4 << 20)
    sources = [Path(__file__), BASE / 'goal_exact16_packed_numa_0910.py', BASE / 'goal_numa_0910.py',
        BASE / 'goal_exact16_runtime_0910.py', BASE / 'deepseek_v41_grouped_int16_exact_goal_0910.py',
        BASE / 'deepseek-v41-grouped-int16-exact-goal-0910.cpp', BASE / 'deepseek-v41-ragged-int16-exact-goal-0910.cpp',
        BASE / 'results/deepseek-v41-ragged-int16-exact-goal-0910/libdeepseek-v41-ragged-int16-exact.so']
    hashes = {str(p): sha256(p) for p in sources}
    placement = helper.Exact16NumaProvider(candidate)
    ragged = RaggedExact16Gemm(BASE / 'results/deepseek-v41-ragged-int16-exact-goal-0910/libdeepseek-v41-ragged-int16-exact.so', candidate.get_packed, workers=1, fallback=baseline)
    records = []
    exact_values = 0

    def weight(n, k):
        b = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
        bs = torch.randint(119, 129, (n, k // 32), dtype=torch.uint8).view(torch.float8_e8m0fnu)
        return b, bs

    for n, k in [(256, 640), (256, 2048)]:
        b, bs = weight(n, k)
        # Borrow and hash the original owned pack without retaining its storage.
        initial = placement.original_get_packed(b, bs)
        object_id = id(initial)
        source_hashes = [digest(b), digest(bs)]
        packed_hashes = [digest(initial.packed), digest(initial.scales)]
        old_refs = [weakref.ref(initial.packed), weakref.ref(initial.scales)]
        source_ptrs = [b.data_ptr(), bs.data_ptr()]
        data_bytes = initial.storage_bytes
        del initial
        packed = candidate.get_packed(b, bs)
        assert isinstance(packed, PackedFP4Exact16) and id(packed) == object_id
        assert packed.storage_bytes == data_bytes
        assert [digest(packed.packed), digest(packed.scales)] == packed_hashes
        assert [digest(b), digest(bs)] == source_hashes
        assert source_ptrs == [b.data_ptr(), bs.data_ptr()]
        gc.collect()
        assert all(reference() is None for reference in old_refs)
        metadata = placement.ledger[placement._key(b, bs)]
        actual_weight = query_page_nodes(packed.packed)
        actual_scale = query_page_nodes(packed.scales)
        assert actual_weight == expected_locations(metadata['packed'])
        assert actual_scale == expected_locations(metadata['scales'])
        assert set(actual_weight) == {0, 1, 2, 3}
        assert metadata['packed']['boundary_policy'] == 'strict'
        assert metadata['scales']['boundary_policy'] == 'nearest_page'
        assert all(value % PAGE_SIZE == 0 for value in metadata['packed']['requested_byte_boundaries'])
        before_count = placement.relocated_entries
        pointers = packed.packed.data_ptr(), packed.scales.data_ptr()
        assert candidate.get_packed(b, bs) is packed
        # Storage aliases are accepted by the original provider's pointer key.
        assert candidate.get_packed(b.view_as(b), bs.view_as(bs)) is packed
        assert pointers == (packed.packed.data_ptr(), packed.scales.data_ptr())
        assert placement.relocated_entries == before_count
        a, asc = cpu.act_quant(torch.randn(5, k).bfloat16(), 32, 'ue8m0', torch.float8_e8m0fnu)
        a.view(torch.uint8)[0, 0] = 1
        actual = ragged.apply([(4, a, asc, b, bs)])[0]
        expected = baseline.apply(4, a, asc, b, bs)
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
        exact_values += actual.numel()
        current_refs = [weakref.ref(packed), weakref.ref(packed.packed), weakref.ref(packed.scales)]
        record = dict(shape=[n, k], native_bytes_unchanged=True, pack_bytes_unchanged=True,
            class_and_object_identity_preserved=True, old_packed_tensors_released=True,
            weight_pages_by_node=dict(Counter(actual_weight)), scale_pages_by_node=dict(Counter(actual_scale)),
            placement_metadata=metadata, exact_values=actual.numel())
        candidate.drop_sources({bs.data_ptr()})
        assert not candidate.cache and not placement.ledger
        assert candidate.packed_bytes == placement.padding_bytes == placement.relocated_data_bytes == 0
        del packed
        gc.collect()
        assert all(reference() is None for reference in current_refs)
        record['eviction_releases_all_packed_references'] = True
        records.append(record)

    # Insufficient padding budget must leave both cached fields intact.
    b, bs = weight(256, 640)
    initial = placement.original_get_packed(b, bs)
    old_ptrs = initial.packed.data_ptr(), initial.scales.data_ptr()
    placement.padding_cap_bytes = 0
    assert candidate.get_packed(b, bs) is None
    assert old_ptrs == (initial.packed.data_ptr(), initial.scales.data_ptr())
    assert not placement.ledger and placement.padding_bytes == 0
    assert placement.last_failure['reason'] == 'padding_cap'
    placement.padding_cap_bytes = 16 << 20

    # Leave the full configured 100GB reserve, accounting for both temporary maps.
    saved_available = helper.mem_available_bytes
    try:
        helper.mem_available_bytes = lambda: placement.reserve_bytes
        assert candidate.get_packed(b, bs) is None
        assert placement.last_failure['reason'] == 'memory_reserve'
        assert old_ptrs == (initial.packed.data_ptr(), initial.scales.data_ptr())
    finally:
        helper.mem_available_bytes = saved_available

    # A second-copy failure must neither publish a half-replaced pack nor retain
    # the successful first clone. The injected error affects this fixture only.
    saved_clone = helper.clone_rows
    failed_clone_refs = []
    def fail_second(tensor, nodes, *, boundary_policy):
        if boundary_policy == 'nearest_page':
            raise OSError(12, 'fixture second clone failure')
        result = saved_clone(tensor, nodes, boundary_policy=boundary_policy)
        failed_clone_refs.append(weakref.ref(result))
        return result
    try:
        helper.clone_rows = fail_second
        assert candidate.get_packed(b, bs) is None
        assert placement.last_failure['reason'] == 'clone_failed'
        assert old_ptrs == (initial.packed.data_ptr(), initial.scales.data_ptr())
        assert not placement.ledger and placement.padding_bytes == 0
    finally:
        helper.clone_rows = saved_clone
    gc.collect()
    assert all(reference() is None for reference in failed_clone_refs)
    del initial
    candidate.clear()

    b, bs = weight(33, 96)
    assert candidate.get_packed(b, bs) is None
    assert placement.last_failure['reason'] == 'strict_row_boundary'
    candidate.clear()
    candidate.cap_bytes = 0
    assert candidate.get_packed(b, bs) is None
    assert placement.provider_fallbacks == 1
    assert not candidate.cache and not placement.ledger
    candidate.cap_bytes = 4 << 20

    # Clear and uninstall also release the separate padding charge immediately.
    b, bs = weight(256, 640)
    p = candidate.get_packed(b, bs)
    clear_refs = [weakref.ref(p), weakref.ref(p.packed), weakref.ref(p.scales)]
    del p
    candidate.clear()
    gc.collect()
    assert all(reference() is None for reference in clear_refs)
    assert not placement.ledger and placement.padding_bytes == 0
    p = candidate.get_packed(b, bs)
    uninstall_refs = [weakref.ref(p), weakref.ref(p.packed), weakref.ref(p.scales)]
    del p
    placement.uninstall()
    gc.collect()
    assert all(reference() is None for reference in uninstall_refs)
    assert not candidate.cache and not placement.ledger and placement.padding_bytes == 0
    assert candidate.get_packed is placement.original_get_packed
    assert candidate.backend.pack_provider is placement.original_backend_provider
    assert not hasattr(candidate, '_goal_exact16_packed_numa_0910')
    assert all(sha256(p) == h for p, h in hashes.items())
    result = dict(passed=True, cpu_affinity=sorted(os.sched_getaffinity(0)), source_sha256=hashes,
        cases=records, exact_values=exact_values, metrics=placement.metrics(),
        padding_cap_and_memory_reserve_checked=True, second_clone_failure_atomic=True,
        scale_eviction_clear_uninstall_release_verified=True, source_alias_cache_hit_verified=True,
        invalid_page_quarters_rejected=True, no_tensor_references_in_ledger=True,
        maximum_fixture_owned_pack_bytes=278528, full_model_loaded=False, performance_measured=False)
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / 'check.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_sha256', 'cases')}))


if __name__ == '__main__':
    main()
