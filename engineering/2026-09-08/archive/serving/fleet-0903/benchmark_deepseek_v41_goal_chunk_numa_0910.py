#!/usr/bin/env python3
"""Four-node perfect-draft verifier experiment; launch only through a controller.

Reserves physical CPUs 0..63 and their SMT siblings for a quiet trial. Common
matrices and bounded exact16 packed rowtiles are copied across four NUMA nodes;
routed native source weights remain unchanged. Torch uses 16 workers throughout.
The serial arm is an absolute rate in this four-node-copy configuration. It is
not a valid speedup denominator for the separately selected single-socket server.
Perfect proposals exclude draft/rejection costs and are not served throughput.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import threading
import time

# Apply before Torch or either OpenMP runtime is imported. OMP_PLACES would take
# precedence over GOMP_CPU_AFFINITY, so remove an inherited conflicting setting.
os.sched_setaffinity(0, set(range(64)))
os.environ.pop('OMP_PLACES', None)
os.environ['OMP_NUM_THREADS'] = '64'
os.environ['MKL_NUM_THREADS'] = '16'
os.environ['OPENBLAS_NUM_THREADS'] = '16'
os.environ['OMP_PROC_BIND'] = 'true'
os.environ['GOMP_CPU_AFFINITY'] = '0-63'
os.environ['OMP_WAIT_POLICY'] = 'PASSIVE'
os.environ['OMP_DYNAMIC'] = 'FALSE'
os.environ['MKL_DYNAMIC'] = 'FALSE'

import torch
import deepseek_v41_server_0910 as server
from deepseek_v41_resident_store_0910 import ResidentStore, bind
from goal_runtime_0910 import Optimizations
from goal_chunk_verify2_0910 import ChunkVerifier
from goal_exact16_runtime_0910 import install as install_exact16
from goal_chunk_grouped_moe_0910 import install as install_chunk_grouped
from deepseek_v41_ragged_int16_goal_0910 import RaggedExact16Gemm
from goal_numa_store_0910 import NumaResidentStore, first_partition_pages
from goal_exact16_packed_numa_0910 import Exact16NumaProvider
from goal_packed_runtime_placement_0910 import capture_placement, capture_threads
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import sha256

BASE = Path(__file__).resolve().parent
PLAN = [
    ('serial_warm', 1, False, 16, False),
    ('serial_before', 1, False, 16, True),
    ('chunk5_grouped64_warm', 5, True, 64, False),
    ('chunk5_grouped64_1', 5, True, 64, True),
    ('chunk5_grouped64_2', 5, True, 64, True),
    ('chunk6_grouped64_warm', 6, True, 64, False),
    ('chunk6_grouped64', 6, True, 64, True),
    ('chunk8_grouped64_warm', 8, True, 64, False),
    ('chunk8_grouped64', 8, True, 64, True),
    ('chunk5_grouped32_warm', 5, True, 32, False),
    ('chunk5_grouped32', 5, True, 32, True),
    ('chunk5_grouped64_repeat', 5, True, 64, True),
    ('serial_after', 1, False, 16, True),
]


class CommonOnlyNumaStore(NumaResidentStore):
    # Keep routed expert activations byte-for-byte in the original ResidentStore.
    # Its inherited NUMA release bookkeeping simply sees zero expert-copy bytes.
    activate = ResidentStore.activate


def capture_packed_partitions(candidate, max_entries=12):
    """Query each quarter of deterministic cache samples without data reads."""
    count = len(candidate.cache)
    take = min(count, max_entries)
    indices = ({0} if take == 1 else {i * (count - 1) // (take - 1) for i in range(take)}) if take else set()
    records = []
    for index, (_, (_, _, packed)) in enumerate(candidate.cache.items()):
        if index not in indices:
            continue
        row = dict(cache_index=index)
        for name in ['packed', 'scales']:
            tensor = getattr(packed, name)
            metadata = getattr(tensor, '_goal_numa_0910', None)
            row[name] = (first_partition_pages(tensor, metadata) if metadata is not None
                         else dict(verified=False, reason='No NUMA clone metadata'))
        records.append(row)
    return dict(sampled_entries=len(records), query_only=True,
                all_sampled_partitions_verified=all(row[name]['verified'] for row in records for name in ['packed', 'scales']),
                records=records)


def validate_proofs():
    names = [
        ('results/goal_chunk_verify2_0910/fixture-check.json', 'source_sha256'),
        ('results/goal_chunk_grouped_moe_0910/fixture-check.json', 'source_sha256'),
        ('results/goal_chunk_grouped_moe_0910/native-fixture-check.json', 'source_sha256'),
        ('results/deepseek-v41-grouped-int16-exact-goal-0910/runtime-check.json', 'sha256'),
        ('results/deepseek-v41-ragged-int16-exact-goal-0910/kernel-check.json', 'source_sha256'),
        ('results/goal_numa_store_0910/fixture-check.json', 'sha256'),
        ('results/goal_numa_0910/fixture-check.json', 'sha256'),
        ('results/goal-exact16-packed-numa-0910/check.json', 'source_sha256'),
    ]
    hashes = {}
    for name, key in names:
        path = BASE / name
        proof = json.loads(path.read_text())
        assert proof['passed'] and all(sha256(p) == h for p, h in proof[key].items()), name
        if name.startswith('results/goal_chunk_verify2_0910/'):
            assert proof['shared_runtime_exact'] and proof['hc_scope_restored']
        hashes.update(proof[key])
        hashes[str(path)] = sha256(path)
    names = ['benchmark_deepseek_v41_goal_chunk_grouped_0910.py', 'goal_runtime_0910.py',
        'goal_chunk_verify_0910.py', 'goal_chunk_verify2_0910.py', 'goal_exact16_runtime_0910.py',
        'goal_chunk_grouped_moe_0910.py', 'deepseek_v41_ragged_int16_goal_0910.py',
        'deepseek_v41_grouped_int16_exact_goal_0910.py', 'goal_numa_store_0910.py',
        'goal_numa_0910.py', 'goal_exact16_packed_numa_0910.py', 'goal_packed_runtime_placement_0910.py',
        'results/deepseek-v41-grouped-int16-exact-goal-0910/libdeepseek-v41-grouped-int16-exact.so',
        'results/deepseek-v41-ragged-int16-exact-goal-0910/libdeepseek-v41-ragged-int16-exact.so']
    hashes.update({str(BASE / name): sha256(BASE / name) for name in names})
    hashes[str(Path(__file__))] = sha256(__file__)
    return hashes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    hashes = validate_proofs()
    selected = json.loads((BASE / 'deepseek-v41-selected.json').read_text())
    protected = dict(selected['source_sha256'])
    assert all(sha256(p) == h for p, h in protected.items()), 'Selected source drift before trial'
    result = dict(passed=False, source_sha256=hashes, protected_selected_sources=protected, runs=[],
        requested_physical_cpus=list(range(64)), torch_workers=16, native_worker_arms=[16, 32, 64],
        runtime_environment={key: os.environ[key] for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'OPENBLAS_NUM_THREADS', 'OMP_PROC_BIND', 'GOMP_CPU_AFFINITY', 'OMP_WAIT_POLICY', 'OMP_DYNAMIC', 'MKL_DYNAMIC']},
        common_only_numa_store=True, routed_native_sources_copied=False, packed_numa_nodes=[0, 1, 2, 3],
        hc_per_token=True, candidate_chunk_router=True, perfect_draft_verifier_only=True,
        real_draft_model_loaded=False, served_generation_tok_s_measured=False,
        serial_baseline_scope='Absolute rate with four-node common copies; no speedup comparison to selected server',
        worker32_scope='GOMP ordered CPU list remains 0-63; inspect actual worker placement before interpreting this arm',
        six_position_arm='five perfect draft proposals plus one bonus position',
        excludes=['draft model time', 'proposal rejection and replay', 'network cold loads'],
        requires_quiet_all_physical_and_sibling_cpus=True)
    runtime = verifier = exact16 = placement = chunk_grouped = None
    try:
        server.ServingStore = CommonOnlyNumaStore
        server.bind = bind
        runtime = server.Runtime(args)
        common = runtime.store.clone_common(runtime.model)
        assert runtime.store.numa_counters['common_copy_passes'] == 1
        assert runtime.store.numa_counters['placement_sample_failures'] == 0
        assert runtime.store.numa_common_bytes > 0
        assert not runtime.store.numa_prefix_bytes and not runtime.store.numa_processed_prefixes
        result['common_copy_outcome'] = common
        atomic_json(args.output / 'common-numa-placement.json', runtime.store.stats())
        opt = Optimizations(runtime)
        opt.configure(dict(quant=True, reuse=True, grouped=True, hc=True, sparse=True, native_workers=16))
        ids, count = runtime.prepare(dict(messages=[dict(role='user', content='Hi.')], max_tokens=16))
        golden = json.loads((BASE / 'results/deepseek-v41-checkpoint-run-0910b/generation.json').read_text())['runs'][0]
        proposals = golden['token_ids'][:-1]
        references = []
        # This constructor captures function identities. Never call opt.configure
        # afterward; only change runtime.native.workers for individual arms.
        verifier = ChunkVerifier(runtime.model, runtime.module)
        exact16 = install_exact16(runtime, opt)
        exact16.enabled = False
        placement = Exact16NumaProvider(exact16)
        ragged = RaggedExact16Gemm(BASE / 'results/deepseek-v41-ragged-int16-exact-goal-0910/libdeepseek-v41-ragged-int16-exact.so',
                                   exact16.get_packed, workers=16)
        chunk_grouped = install_chunk_grouped(runtime, ragged, lambda: verifier.active)
        original_hc = runtime.module.hc_split_sinkhorn
        for label, width, use_grouped, workers, measured in PLAN:
            runtime.native.workers = workers
            chunk_grouped.enabled = use_grouped
            pack_before = exact16.pack_count
            clone_before = placement.relocated_entries
            common_before = runtime.store.numa_counters['copied_bytes_total']
            grouped_before = chunk_grouped.calls
            native_before = (ragged.native_calls, ragged.fallback_calls, exact16.cap_fallbacks)
            hc_before = (verifier.hc_batch_calls, verifier.hc_token_calls, opt.hc.native_calls, opt.hc.fallback_calls)
            print(json.dumps(dict(running=label, chunk_width=width, native_workers=workers, torch_workers=16)), flush=True)
            replies, failures = [], []
            def request():
                try:
                    torch.set_num_threads(16)
                    assert torch.get_num_threads() == 16 and runtime.native.workers == workers
                    before_threads = capture_threads()
                    with torch.inference_mode():
                        downloaded_before = runtime.store.downloaded_bytes
                        began = time.perf_counter()
                        initial = runtime.model(torch.tensor([ids], dtype=torch.int64), 0)
                        prefill_seconds = time.perf_counter() - began
                        logits = [initial[1].detach().clone()]
                        predicted = [int(initial[0].item())]
                        decode_seconds, calls = 0., []
                        for at in range(0, len(proposals), width):
                            batch = proposals[at:at + width]
                            x = torch.tensor([batch], dtype=torch.int64)
                            began = time.perf_counter()
                            if len(batch) > 1:
                                output = verifier.verify(x, len(ids) + at)
                                decoded = output[0][0].tolist()
                                values = [output[1][:, i].detach().clone() for i in range(len(batch))]
                                mode = verifier.last_mode
                            else:
                                output = runtime.model(x, len(ids) + at)
                                decoded = [int(output[0].item())]
                                values = [output[1].detach().clone()]
                                mode = 'single'
                            elapsed = time.perf_counter() - began
                            decode_seconds += elapsed
                            calls.append(dict(width=len(batch), mode=mode, seconds=elapsed))
                            predicted.extend(decoded)
                            logits.extend(values)
                        logits_hashes = [hashlib.sha256(x.float().numpy().tobytes()).hexdigest() for x in logits]
                        if not references:
                            references.extend(logits)
                        row = dict(label=label, measured=measured, chunk_width=width, grouped=use_grouped,
                            native_workers=workers, torch_workers=torch.get_num_threads(), prefill_seconds=prefill_seconds,
                            decode_seconds=decode_seconds, verified_positions=len(proposals),
                            ideal_verifier_positions_s=len(proposals) / decode_seconds,
                            downloaded_bytes=runtime.store.downloaded_bytes - downloaded_before,
                            logits_sha256=logits_hashes, predicted_ids=predicted, calls=calls,
                            exact_golden=logits_hashes == [step['logits_sha256'] for step in golden['steps']],
                            tokens_match=predicted == golden['token_ids'],
                            hc_calls_delta=dict(batch=verifier.hc_batch_calls - hc_before[0],
                                token=verifier.hc_token_calls - hc_before[1], native=opt.hc.native_calls - hc_before[2],
                                fallback=opt.hc.fallback_calls - hc_before[3]),
                            max_abs=max(float((a - b).abs().max()) for a, b in zip(logits, references)))
                    # Query outside all generation timers, while this request's
                    # existing native/Torch worker pools are still alive.
                    snapshot = capture_placement(exact16.cache, max_entries=12, previous=before_threads, label=label)
                    snapshot['packed_quarter_queries'] = capture_packed_partitions(exact16)
                    snapshot['packed_numa_provider'] = placement.metrics()
                    snapshot['common_copy_counters'] = dict(runtime.store.numa_counters)
                    snapshot['request_thread_native_id'] = threading.get_native_id()
                    atomic_json(args.output / (label + '-placement.json'), snapshot)
                    row['placement_snapshot'] = label + '-placement.json'
                    row['observed_single_cpu_affinities'] = sorted({int(record['cpus_allowed_list'])
                        for record in snapshot['threads']['records'] if record.get('cpus_allowed_list', '').isdigit()})
                    row['observed_last_processors'] = sorted({record['last_processor']
                        for record in snapshot['threads']['records'] if 'last_processor' in record})
                    if use_grouped and workers == 64:
                        assert set(range(64)).issubset(row['observed_single_cpu_affinities']), '64-core singleton binding coverage missing'
                    if use_grouped:
                        assert snapshot['packed_quarter_queries']['all_sampled_partitions_verified'], 'Packed NUMA quarter sample differs from binding'
                    replies.append(row)
                except BaseException as error:
                    failures.append(error)
            thread = threading.Thread(target=request)
            thread.start()
            thread.join()
            if failures:
                raise failures[0]
            row = replies[0]
            row.update(packed_bytes=exact16.packed_bytes, pack_delta=exact16.pack_count - pack_before,
                packed_clone_delta=placement.relocated_entries - clone_before,
                common_copy_bytes_delta=runtime.store.numa_counters['copied_bytes_total'] - common_before,
                chunk_grouped_calls=chunk_grouped.calls - grouped_before,
                ragged_native_calls=ragged.native_calls - native_before[0],
                ragged_fallback_calls=ragged.fallback_calls - native_before[1],
                cap_fallbacks=exact16.cap_fallbacks - native_before[2], packed_numa=placement.metrics())
            result['runs'].append(row)
            atomic_json(args.output / 'result.json', result)
            print(json.dumps({key: row[key] for key in ['label', 'ideal_verifier_positions_s', 'exact_golden',
                'tokens_match', 'max_abs', 'downloaded_bytes', 'pack_delta', 'packed_clone_delta', 'native_workers']}), flush=True)
            assert row['tokens_match'] and row['exact_golden'], label
            assert runtime.module.hc_split_sinkhorn is original_hc, 'Verifier HC wrapper leaked'
            assert runtime.store.numa_counters['common_copy_passes'] == 1
            assert not runtime.store.numa_prefix_bytes and not runtime.store.numa_processed_prefixes
            assert row['common_copy_bytes_delta'] == 0
            if measured:
                assert row['downloaded_bytes'] == row['pack_delta'] == row['packed_clone_delta'] == 0, label
            if use_grouped:
                assert row['chunk_grouped_calls'] > 0 and row['ragged_native_calls'] > 0, label
                assert row['ragged_fallback_calls'] == row['cap_fallbacks'] == 0, label
                assert not placement.fallbacks_by_reason and placement.provider_fallbacks == 0, label
                assert len(placement.ledger) == len(exact16.cache), 'A cached FP4 pack lacks NUMA placement'
            if width > 1:
                assert row['hc_calls_delta']['batch'] > 0 and row['hc_calls_delta']['native'] > 0, label
            assert all(sha256(p) == h for p, h in protected.items()), 'Selected source drift during trial'
        result['absolute_ideal_positions_s'] = {}
        for width, workers in [(1, 16), (5, 64), (6, 64), (8, 64), (5, 32)]:
            rows = [row for row in result['runs'] if row['measured'] and row['chunk_width'] == width and row['native_workers'] == workers]
            result['absolute_ideal_positions_s'][f'width{width}_native{workers}_torch16'] = (
                sum(row['verified_positions'] for row in rows) / sum(row['decode_seconds'] for row in rows))
        result['all_exact_golden'] = all(row['exact_golden'] for row in result['runs'])
        result['zero_measured_downloads_packs_copies'] = all(
            row['downloaded_bytes'] == row['pack_delta'] == row['packed_clone_delta'] == row['common_copy_bytes_delta'] == 0
            for row in result['runs'] if row['measured'])
        chunk_grouped.uninstall()
        chunk_grouped = None
        placement.uninstall()
        placement = None
        exact16.uninstall()
        exact16 = None
        verifier.uninstall()
        verifier = None
        assert all(sha256(p) == h for p, h in result['source_sha256'].items())
        result['passed'] = True
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        if chunk_grouped is not None:
            chunk_grouped.uninstall()
        if placement is not None:
            placement.uninstall()
        if exact16 is not None:
            exact16.uninstall()
        if verifier is not None:
            verifier.uninstall()
        if runtime is not None:
            runtime.store.release_all()
            runtime.store.close()
        result['selected_sources_preserved'] = all(sha256(p) == h for p, h in protected.items())
        atomic_json(args.output / 'result.json', result)


if __name__ == '__main__':
    main()
