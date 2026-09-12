#!/usr/bin/env python3
"""One-core, <=1 MiB synthetic NUMA byte/lifetime/physical-placement fixtures."""
from collections import Counter
import ctypes
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path

import torch

import goal_numa_0910 as numa

BASE = Path(__file__).resolve().parent


def containing_mapping(address):
    for line in Path('/proc/self/maps').read_text().splitlines():
        begin, end = [int(s, 16) for s in line.split()[0].split('-')]
        if begin <= address < end:
            return line
    return None


def process_policy():
    lib = ctypes.CDLL('libnuma.so.1', use_errno=True)
    fn = lib.get_mempolicy
    fn.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_ulong),
                   ctypes.c_ulong, ctypes.c_void_p, ctypes.c_ulong]
    fn.restype = ctypes.c_int
    mode, mask = ctypes.c_int(), ctypes.c_ulong()
    assert fn(ctypes.byref(mode), ctypes.byref(mask), 64, None, 0) == 0
    return mode.value, mask.value


def main():
    assert os.sched_getaffinity(0) == {0}, 'run with taskset -c 0'
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(410910)
    original_policy = process_policy()
    cases = []
    dtypes = [torch.float8_e4m3fn, torch.float8_e8m0fnu, torch.float4_e2m1fn_x2,
              torch.bfloat16, torch.float32, torch.uint8]
    for dtype in dtypes:
        raw = torch.randint(0, 256, (256, 1024), dtype=torch.uint8)
        source = raw.view(dtype)
        source.scale = torch.ones(8, 32, dtype=torch.float32)
        clone = numa.clone_rows(source)
        placement = clone._goal_numa_0910
        assert clone.shape == source.shape and clone.dtype == source.dtype and clone.is_contiguous()
        assert clone.scale is source.scale
        assert torch.equal(source.view(torch.uint8), clone.view(torch.uint8))
        locations = numa.query_page_nodes(clone)
        expected = [node for node, begin, end in zip(placement.nodes, placement.actual_byte_boundaries,
                                                    placement.actual_byte_boundaries[1:])
                    for _ in range((end - begin) // numa.PAGE_SIZE)]
        assert locations == expected, (dtype, Counter(locations))
        numa_lines = [line for line in Path('/proc/self/numa_maps').read_text().splitlines()
                      if placement.address <= int(line.split()[0], 16)
                      < placement.address + placement.allocated_bytes]
        assert len(numa_lines) == 4
        for node, line in enumerate(numa_lines):
            assert f'bind:{node} ' in line and f'N{node}=16' in line, line
        address = clone.data_ptr()
        parameter = torch.nn.Parameter(clone, requires_grad=False)
        parameter.scale = clone.scale
        del clone
        gc.collect()
        assert containing_mapping(address) is not None
        assert torch.equal(source.view(torch.uint8), parameter.view(torch.uint8))
        assert numa.query_page_nodes(parameter) == expected
        del parameter
        gc.collect()
        assert containing_mapping(address) is None, 'final tensor alias did not unmap storage'
        cases.append(dict(dtype=str(dtype), shape=list(source.shape), placement=asdict(placement),
                          actual_page_counts=dict(Counter(locations)), exact_native_bytes=True,
                          parameter_retains_storage=True, final_release_unmaps=True,
                          scale_alias_preserved=True, proc_numa_maps=numa_lines))
        del source, raw

    # Realistic small scale dimensions cannot have exact four-way row/page
    # boundaries. Strict mode rejects; nearest_page records the physical result.
    source = torch.randint(0, 256, (72, 160), dtype=torch.uint8).view(torch.float8_e8m0fnu)
    try:
        numa.clone_rows(source)
    except ValueError as error:
        assert 'page aligned' in str(error)
    else:
        raise AssertionError('unaligned rows silently accepted')
    clone = numa.clone_rows(source, boundary_policy='nearest_page')
    locations = numa.query_page_nodes(clone)
    placement = clone._goal_numa_0910
    assert torch.equal(source.view(torch.uint8), clone.view(torch.uint8))
    assert locations == [0, 2, 3]
    cases.append(dict(dtype=str(source.dtype), shape=list(source.shape), placement=asdict(placement),
                      actual_page_counts=dict(Counter(locations)), exact_native_bytes=True,
                      strict_rejected_unaligned=True))
    del clone, source
    gc.collect()
    assert os.sched_getaffinity(0) == {0}
    assert process_policy() == original_policy
    paths = [Path(__file__), BASE / 'goal_numa_0910.py']
    result = dict(passed=True, component_only=True, full_checkpoint_loaded=False,
                  performance_trial=False, cpu_affinity=[0], torch_threads=1,
                  process_memory_policy_unchanged=True, source_and_clone_bytes_max=524288,
                  actual_four_node_placement_verified=True, cases=cases,
                  sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    output = BASE / 'results/goal_numa_0910/fixture-check.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('cases', 'sha256')}))


if __name__ == '__main__':
    main()
