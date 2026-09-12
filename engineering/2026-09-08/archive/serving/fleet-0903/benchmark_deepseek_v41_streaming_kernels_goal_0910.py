#!/usr/bin/env python3
"""Prepared hot/streaming component comparisons, with explicit NUMA observations.

All weights, packs, outputs and native descriptors are allocated before timing.
Normal library-internal activation scratch and wrapper metadata remain part of
those implementations' costs. The wrapper arms use preallocated output tensors.
There are no model or tok/s measurements here.
"""
import argparse
import ctypes
import errno
import gc
import json
import os
from pathlib import Path
import time
import torch
import deepseek_v41_cpu_reference_0910 as cpu
from deepseek_v41_native_bridge_0910 import NativeGemm
from deepseek_v41_native_grouped_goal_0910 import GroupedNativeGemm, Descriptor as BaselineDescriptor
from deepseek_v41_native_grouped_vnni_goal_0910 import GroupedVnniGemm, Descriptor as VnniDescriptor
from deepseek_v41_native_vnni_goal_0910 import NativeVnniGemm
from deepseek_v41_native_fp8_vnni_goal_0910 import NativeFP8VnniGemm
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import sha256

BASE = Path(__file__).resolve().parent
LIB_B = BASE / 'results/deepseek-v41-native-gemm-0910b/libdeepseek-v41-native-gemm.so'
LIB_GB = BASE / 'results/deepseek-v41-native-grouped-goal-0910/libdeepseek-v41-native-grouped.so'
LIB_V4 = BASE / 'results/deepseek-v41-native-vnni-goal-0910/libdeepseek-v41-native-vnni.so'
LIB_GV = BASE / 'results/deepseek-v41-native-grouped-vnni-goal-0910/libdeepseek-v41-native-grouped-vnni.so'
LIB_V8 = BASE / 'results/deepseek-v41-native-fp8-vnni-goal-0910/libdeepseek-v41-native-fp8-vnni.so'
PERF_SECONDS = 0.0
PROOFS = [BASE / 'results' / name / 'kernel-check.json' for name in
    ['deepseek-v41-native-grouped-goal-0910', 'deepseek-v41-native-vnni-goal-0910',
     'deepseek-v41-native-grouped-vnni-goal-0910', 'deepseek-v41-native-fp8-vnni-goal-0910']]


def input_sources():
    files = [Path(__file__), LIB_B, LIB_GB, LIB_V4, LIB_GV, LIB_V8, *PROOFS]
    for proof_path in PROOFS:
        proof = json.loads(proof_path.read_text())
        assert proof['passed']
        for name, digest in proof['source_sha256'].items():
            assert sha256(name) == digest, ('Correctness input changed', name)
            files.append(Path(name))
    files += [Path(cpu.__file__)]
    return list(dict.fromkeys(files))


def rss_bytes():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            return int(line.split()[1]) * 1024
    return None


def numa_observations(labelled_tensors):
    maps = []
    for line in Path('/proc/self/maps').read_text().splitlines():
        start, end = [int(x, 16) for x in line.split()[0].split('-')]
        maps.append((start, end, line))
    numa = {int(line.split()[0], 16): line for line in Path('/proc/self/numa_maps').read_text().splitlines()}
    samples = []
    for label, tensor in labelled_tensors:
        start = tensor.data_ptr()
        end = start + tensor.numel() * tensor.element_size() - 1
        pages = [address & ~4095 for address in [start, (start + end) // 2, end]]
        samples.append(dict(label=label, bytes=tensor.numel() * tensor.element_size(), pages=pages))
    addresses = [address for sample in samples for address in sample['pages']]
    page_array = (ctypes.c_void_p * len(addresses))(*addresses)
    statuses = (ctypes.c_int * len(addresses))(*([-999] * len(addresses)))
    libc = ctypes.CDLL(None, use_errno=True)
    # x86-64 SYS_move_pages=279; nodes=NULL is query-only, never migrates pages.
    libc.syscall.restype = ctypes.c_long
    ctypes.set_errno(0)
    code = libc.syscall(ctypes.c_long(279), ctypes.c_int(0), ctypes.c_ulong(len(addresses)),
        page_array, ctypes.c_void_p(), statuses, ctypes.c_int(0))
    error = ctypes.get_errno() if code < 0 else 0
    for index, sample in enumerate(samples):
        if code >= 0:
            sample['page_node_status'] = list(statuses)[index * 3:index * 3 + 3]
        coverage = []
        for address in sample['pages']:
            mapping = next(((start, end, line) for start, end, line in maps if start <= address < end), None)
            coverage.append(None if mapping is None else dict(mapping_start=mapping[0], mapping_end=mapping[1],
                mapping=mapping[2], numa_mapping_aggregate=numa.get(mapping[0])))
        sample['mapping_coverage'] = coverage
    return dict(query_only_move_pages=dict(return_code=code, errno=error,
        error_name=errno.errorcode.get(error) if error else None), samples=samples,
        limitation='numa_maps records describe entire containing mappings; they do not identify sampled page nodes when move_pages fails.')


def representatives(native_weights, packed_weights):
    result = []
    for prefix, values in [('native', native_weights), ('packed', packed_weights)]:
        for label, index in [('first', 0), ('middle', len(values) // 2), ('last', len(values) - 1)]:
            result.append((prefix + '_' + label, values[index]))
    return result


def buffer_bytes(tensors):
    unique = {}
    for t in tensors:
        unique[(t.data_ptr(), t.numel() * t.element_size())] = t.numel() * t.element_size()
    return sum(unique.values())


def input_activation(k):
    return cpu.act_quant(torch.randn(1, k, dtype=torch.float32).bfloat16(), 32, 'ue8m0', torch.float8_e8m0fnu)


def input_weight(mode, n, k):
    raw = torch.randint(0, 256, (n, k // 2 if mode == 4 else k), dtype=torch.uint8)
    if mode == 8:
        raw[(raw & 127) == 127] = 126
    b = raw.view(torch.float4_e2m1fn_x2 if mode == 4 else torch.float8_e4m3fn)
    bs = torch.randint(118, 130, (n if mode == 4 else (n + 31) // 32, k // 32), dtype=torch.uint8).view(torch.float8_e8m0fnu)
    return b, bs


def measure(calls, epochs):
    global PERF_SECONDS
    began = time.perf_counter()
    for _ in range(epochs):
        for invoke in calls:
            invoke()
    elapsed = time.perf_counter() - began
    PERF_SECONDS += elapsed
    assert PERF_SECONDS <= 30, ('Total timed section budget', PERF_SECONDS)
    return elapsed


def paired(name, reference, candidate, epochs, projections_per_epoch, logical_weight_bytes):
    arms = []
    for variant, calls in [('baseline', reference), ('candidate', candidate), ('candidate', candidate), ('baseline', reference)]:
        elapsed = measure(calls, epochs)
        arms.append(dict(variant=variant, seconds=elapsed, seconds_per_epoch=elapsed / epochs))
    baseline = (arms[0]['seconds_per_epoch'] + arms[3]['seconds_per_epoch']) / 2
    optimized = (arms[1]['seconds_per_epoch'] + arms[2]['seconds_per_epoch']) / 2
    return dict(comparison=name, epochs=epochs, projections_per_epoch=projections_per_epoch,
        baseline_seconds_per_projection=baseline / projections_per_epoch,
        candidate_seconds_per_projection=optimized / projections_per_epoch,
        speedup=baseline / optimized, logical_weight_bytes_per_epoch=logical_weight_bytes,
        baseline_logical_weight_GB_s=logical_weight_bytes / baseline / 1e9,
        candidate_logical_weight_GB_s=logical_weight_bytes / optimized / 1e9, arms=arms)


def prepare_group(tasks, packs, baseline, gb, gv, workers):
    n = tasks[0][3].shape[0]
    out = torch.empty(len(tasks), n, dtype=torch.bfloat16)
    bdesc, vdesc = (BaselineDescriptor * len(tasks))(), (VnniDescriptor * len(tasks))()
    flat_arguments = []
    for index, (mode, a, asc, b, bs) in enumerate(tasks):
        k, output_pointer = a.shape[-1], out.data_ptr() + index * n * 2
        bdesc[index] = BaselineDescriptor(mode, n, k, 0, a.data_ptr(), asc.data_ptr(), b.data_ptr(), bs.data_ptr(), output_pointer)
        flat_arguments.append((mode, a.data_ptr(), asc.data_ptr(), b.data_ptr(), bs.data_ptr(), 1, n, k, workers, output_pointer))
        packed = packs[(b.data_ptr(), bs.data_ptr())] if mode == 4 else None
        vdesc[index] = VnniDescriptor(mode, n, k, 0, a.data_ptr(), asc.data_ptr(),
            packed.packed.data_ptr() if packed else b.data_ptr(), packed.scales.data_ptr() if packed else bs.data_ptr(),
            packed.sums.data_ptr() if packed else None, output_pointer)
    def serial():
        for arguments in flat_arguments:
            assert baseline.fn(*arguments) == 0
    def grouped_baseline():
        assert gb.fn(bdesc, len(tasks), workers) == 0
    def grouped_vnni():
        assert gv.fn(vdesc, len(tasks), workers) == 0
    # Explicit bound references ensure descriptors/pointers remain live.
    prepared = dict(tasks=tasks, output=out, bdesc=bdesc, vdesc=vdesc, arguments=flat_arguments,
        calls=dict(serial_native=serial, grouped_baseline_native=grouped_baseline,
            grouped_vnni_native=grouped_vnni, grouped_baseline_wrapper=lambda: gb.apply(tasks, out),
            grouped_vnni_wrapper=lambda: gv.apply(tasks, out)))
    grouped_baseline()
    expected = out.clone()
    mismatch = {}
    for name, invoke in prepared['calls'].items():
        invoke()
        mismatch[name] = int((out.view(torch.int16) != expected.view(torch.int16)).sum())
    prepared['baseline_bit_differences'] = mismatch
    return prepared


def fp4_phase(name, n, k, tasks_per_group, group_count, args):
    baseline = NativeGemm(LIB_B, args.workers)
    single = NativeVnniGemm(LIB_V4, args.workers)
    gb = GroupedNativeGemm(LIB_GB, args.workers)
    tasks_groups, packs, tensors, native_weights, packed_weights = [], {}, [], [], []
    packing_seconds = 0
    for _ in range(group_count):
        shared = input_activation(k) if tasks_per_group == 14 else None
        tasks = []
        for index in range(tasks_per_group):
            mode = 8 if index >= tasks_per_group - (2 if tasks_per_group == 14 else 1) else 4
            a, asc = shared if shared is not None else input_activation(k)
            b, bs = input_weight(mode, n, k)
            tasks.append((mode, a, asc, b, bs))
            tensors.extend([a, asc, b, bs]); native_weights.append(b)
            if mode == 4:
                began = time.perf_counter(); packed = single.pack(b, bs); packing_seconds += time.perf_counter() - began
                packs[(b.data_ptr(), bs.data_ptr())] = packed
                tensors.extend([packed.packed, packed.sums, packed.scales]); packed_weights.append(packed.packed)
        tasks_groups.append(tasks)
    gv = GroupedVnniGemm(LIB_GV, lambda b, bs: packs[(b.data_ptr(), bs.data_ptr())], args.workers, gb)
    groups = [prepare_group(tasks, packs, baseline, gb, gv, args.workers) for tasks in tasks_groups]
    hot = prepare_group([tasks_groups[0][0]], packs, baseline, gb, gv, args.workers)
    tensors += [group['output'] for group in groups] + [hot['output']]
    total = buffer_bytes(tensors)
    assert total <= 1 << 30, ('Prepared buffer limit', total)
    pages = representatives(native_weights, packed_weights)
    result = dict(name=name, matrix_shape=[n, k], groups=group_count, tasks_per_group=tasks_per_group,
        native_weight_bytes=sum(x.numel() for x in native_weights), packed_weight_bytes=sum(x.numel() for x in packed_weights),
        prepared_buffer_bytes=total, rss_bytes=rss_bytes(), packing_seconds=packing_seconds,
        pack_cache_entries=len(packs), timed_cache_misses=0,
        numa_before=numa_observations(pages), baseline_bit_differences=[g['baseline_bit_differences'] for g in groups], timings=[])
    for workload, chosen, epochs in [('single_matrix_hot', [hot], args.hot_epochs), ('streaming', groups, args.epochs)]:
        projections = sum(len(group['tasks']) for group in chosen)
        weight_bytes = sum(task[3].numel() for group in chosen for task in group['tasks'])
        callset = lambda key: [group['calls'][key] for group in chosen]
        for label, ref, candidate in [('grouping_native_only', 'serial_native', 'grouped_baseline_native'),
                ('vnni_native_only', 'grouped_baseline_native', 'grouped_vnni_native'),
                ('vnni_preallocated_wrappers', 'grouped_baseline_wrapper', 'grouped_vnni_wrapper')]:
            result['timings'].append(dict(workload=workload, **paired(label, callset(ref), callset(candidate), epochs, projections, weight_bytes)))
    assert len(packs) == result['pack_cache_entries'] and gv.fallback_calls == 0
    result['numa_after'] = numa_observations(pages)
    return result


def fp8_phase(name, n, k, count, args):
    baseline, native = NativeGemm(LIB_B, args.workers), NativeFP8VnniGemm(LIB_V8, args.workers)
    cases, tensors, native_weights, packed_weights = [], [], [], []
    packing_seconds = 0
    for _ in range(count):
        a, asc = input_activation(k)
        b, bs = input_weight(8, n, k)
        began = time.perf_counter(); packed = native.pack(b, bs); packing_seconds += time.perf_counter() - began
        output = torch.empty(1, n, dtype=torch.bfloat16)
        baseline_args = (8, a.data_ptr(), asc.data_ptr(), b.data_ptr(), bs.data_ptr(), 1, n, k, args.workers, output.data_ptr())
        candidate_args = (a.data_ptr(), asc.data_ptr(), packed.packed.data_ptr(), bs.data_ptr(), packed.nan_masks.data_ptr(), 1, n, k, args.workers, output.data_ptr())
        def ref(arguments=baseline_args):
            assert baseline.fn(*arguments) == 0
        def optimized(arguments=candidate_args):
            assert native.fn(*arguments) == 0
        ref(); expected = output.clone(); optimized()
        differences = int((output.view(torch.int16) != expected.view(torch.int16)).sum())
        cases.append(dict(ref=ref, candidate=optimized, keep=(a, asc, b, bs, packed, output), differences=differences))
        tensors.extend([a, asc, b, bs, packed.packed, packed.nan_masks, output])
        native_weights.append(b); packed_weights.append(packed.packed)
    total = buffer_bytes(tensors)
    assert total <= 1 << 30, ('Prepared buffer limit', total)
    pages = representatives(native_weights, packed_weights)
    result = dict(name=name, matrix_shape=[n, k], distinct_matrices=count,
        native_weight_bytes=sum(x.numel() for x in native_weights), packed_weight_bytes=sum(x.numel() for x in packed_weights),
        prepared_buffer_bytes=total, rss_bytes=rss_bytes(), packing_seconds=packing_seconds,
        timed_cache_misses=0, numa_before=numa_observations(pages), baseline_bit_differences=[x['differences'] for x in cases], timings=[])
    for workload, chosen, epochs in [('single_matrix_repeated', cases[:1], args.hot_epochs), ('streaming', cases, args.epochs)]:
        result['timings'].append(dict(workload=workload, **paired('fp8_native_only',
            [x['ref'] for x in chosen], [x['candidate'] for x in chosen], epochs, len(chosen), n * k * len(chosen))))
    result['numa_after'] = numa_observations(pages)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=Path)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--hot-epochs', type=int, default=128)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    assert os.sched_getaffinity(0) == set(range(48, 64)) and args.workers == 16
    assert 1 <= args.epochs <= 16 and 8 <= args.hot_epochs <= 256
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.set_flush_denormal(False); torch.manual_seed(419120)
    sources = input_sources()
    result = dict(passed=False, started=time.time(), workers=args.workers, cpus=sorted(os.sched_getaffinity(0)),
        omp_wait_policy=os.environ.get('OMP_WAIT_POLICY'), component_only=True, full_checkpoint_loaded=False,
        model_tok_s_measured=False, max_prepared_buffers_bytes=1 << 30,
        timing_scope='Weights, packing, output allocation and native descriptors are outside timers. Native internal scratch and normal wrapper metadata are included.',
        source_sha256={str(p): sha256(p) for p in sources}, phases=[])
    phases = [('mixed_gate_up', lambda: fp4_phase('mixed_gate_up', 2304, 5120, 14, 4, args)),
              ('mixed_down', lambda: fp4_phase('mixed_down', 5120, 2304, 7, 8, args)),
              ('fp8_small', lambda: fp8_phase('fp8_small', 1280, 5120, 48, args)),
              ('fp8_dense', lambda: fp8_phase('fp8_dense', 5120, 8192, 8, args)),
              ('fp8_head', lambda: fp8_phase('fp8_head', 32768, 1280, 8, args))]
    for name, invoke in phases:
        gc.collect()
        phase = invoke()
        result['phases'].append(phase)
        atomic_json(args.out / 'benchmark-check.json', result)
        print(json.dumps(dict(phase=name, prepared_buffer_bytes=phase['prepared_buffer_bytes'],
            packing_seconds=phase['packing_seconds'], timings=[dict(workload=t['workload'], comparison=t['comparison'],
                speedup=t['speedup'], baseline_ms=1000 * t['baseline_seconds_per_projection'], candidate_ms=1000 * t['candidate_seconds_per_projection']) for t in phase['timings']])), flush=True)
        del phase
        gc.collect()
    assert all(sha256(p) == h for p, h in result['source_sha256'].items())
    result.update(passed=True, finished=time.time(), total_timed_section_seconds=PERF_SECONDS)
    atomic_json(args.out / 'benchmark-check.json', result)


if __name__ == '__main__':
    main()
