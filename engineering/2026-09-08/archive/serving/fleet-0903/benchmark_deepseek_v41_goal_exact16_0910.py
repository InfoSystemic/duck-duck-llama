#!/usr/bin/env python3
"""Scheduled same-model exact16 trial; launch only under an exclusive controller.

Optimized16 baseline, exact16 warmup/two measurements, optional sparse-only and
combined warmup/two measurements, then optimized16 baseline again. No lifecycle or selected
manifest is modified here. The controller owns process handoff and restoration.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import threading
import time

import torch

import deepseek_v41_server_0910 as server
from deepseek_v41_resident_store_0910 import ResidentStore, bind
from goal_runtime_0910 import Optimizations
from goal_exact16_runtime_0910 import install as install_exact16
from goal_sparse_native_0910 import NativeSparse
from goal_packed_runtime_placement_0910 import capture_placement
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import sha256


BASE = Path(__file__).resolve().parent
FULL = dict(quant=True, reuse=True, grouped=True, hc=True, sparse=True, native_workers=16)


class Timers:
    def __init__(self):
        self.phase = 'prefill'
        self.values = {}
        self.originals = []

    def wrap(self, owner, attribute, label):
        original = getattr(owner, attribute)
        self.originals.append((owner, attribute, original))

        def timed(*args, **kwargs):
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                key = self.phase + ':' + label
                row = self.values.setdefault(key, dict(calls=0, seconds=0.0))
                row['calls'] += 1
                row['seconds'] += time.perf_counter() - started

        setattr(owner, attribute, timed)

    def restore(self):
        for owner, attribute, original in reversed(self.originals):
            setattr(owner, attribute, original)
        self.originals.clear()


def validate_proof(path, hash_key):
    proof = json.loads(path.read_text())
    assert proof['passed'], str(path)
    assert all(sha256(name) == digest for name, digest in proof[hash_key].items()), str(path)
    return proof


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--packed-cap-gib', type=int, default=64)
    parser.add_argument('--disable-native-sparse', action='store_true')
    args = parser.parse_args()
    assert 0 <= args.packed_cap_gib <= 128
    args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output / 'result.json').exists(), 'Use a fresh trial output directory'
    selected_path = BASE / 'deepseek-v41-selected.json'
    selected = json.loads(selected_path.read_text())
    protected = dict(selected['source_sha256'])
    inputs = [Path(__file__), BASE / 'goal_exact16_runtime_0910.py',
              BASE / 'deepseek_v41_grouped_int16_exact_goal_0910.py',
              BASE / 'deepseek-v41-grouped-int16-exact-goal-0910.cpp',
              BASE / 'results/deepseek-v41-grouped-int16-exact-goal-0910/libdeepseek-v41-grouped-int16-exact.so',
              BASE / 'results/deepseek-v41-grouped-int16-exact-goal-0910/kernel-check.json',
              BASE / 'results/deepseek-v41-grouped-int16-exact-goal-0910/runtime-check.json',
              BASE / 'goal_packed_runtime_placement_0910.py',
              BASE / 'goal_sparse_native_0910.py', BASE / 'goal_sparse_native_0910.cpp',
              BASE / 'results/goal_sparse_native_0910/libgoal_sparse_native_0910.so',
              BASE / 'results/goal_sparse_native_0910/fixture-check.json']
    hashes = {str(path): sha256(path) for path in inputs}
    result = dict(passed=False, runs=[], protected_selected_sources=protected, source_sha256=hashes,
                  selected_manifest_sha256=sha256(selected_path), affinity=sorted(os.sched_getaffinity(0)),
                  omp_wait_policy=os.environ.get('OMP_WAIT_POLICY'), torch_workers=16, native_workers=16,
                  packed_cap_bytes=args.packed_cap_gib << 30, timers_inclusive=True,
                  timers_must_not_be_summed=True, native_timer_scope='ctypes native-call wall time',
                  fp8_candidate_used=False, measurement='complete greedy requests; decode excludes prefill')
    runtime = candidate = native_sparse = None
    timers = Timers()
    hooks = []
    try:
        assert all(sha256(path) == digest for path, digest in protected.items()), 'Selected source drift before trial'
        validate_proof(BASE / 'results/deepseek-v41-grouped-int16-exact-goal-0910/kernel-check.json', 'input_sha256')
        validate_proof(BASE / 'results/deepseek-v41-grouped-int16-exact-goal-0910/runtime-check.json', 'sha256')
        if not args.disable_native_sparse:
            validate_proof(BASE / 'results/goal_sparse_native_0910/fixture-check.json', 'sha256')
        server.ServingStore, server.bind = ResidentStore, bind
        runtime = server.Runtime(args)
        opt = Optimizations(runtime)
        opt.configure(FULL)
        original_fp4, original_fp8 = runtime.module.fp4_gemm, runtime.module.fp8_gemm
        candidate = install_exact16(runtime, opt, args.packed_cap_gib << 30)
        candidate.configure(dict(exact16=False, timing=True))
        native_sparse = NativeSparse(BASE / 'results/goal_sparse_native_0910/libgoal_sparse_native_0910.so')
        native_sparse.install(runtime.module)
        native_sparse.enabled = False
        assert native_sparse.baseline is opt.sparse
        timers.wrap(runtime.native, 'fn', 'ungrouped_native')
        timers.wrap(candidate.original_backend, 'fn', 'grouped_baseline_native')
        timers.wrap(candidate.original_backend, 'apply', 'grouped_baseline_inclusive')
        timers.wrap(native_sparse, 'fn', 'sparse_native')
        ids, count = runtime.prepare(dict(messages=[dict(role='user', content='Hi.')], max_tokens=16))
        golden = json.loads((BASE / 'results/deepseek-v41-checkpoint-run-0910b/generation.json').read_text())['runs'][0]
        expected_hashes = [step['logits_sha256'] for step in golden['steps']]
        logits = []

        def pre_hook(module, args):
            timers.phase = 'prefill' if args[1] == 0 else 'decode'

        def output_hook(module, args, output):
            logits.append(output[1].detach().float().numpy().copy())

        hooks.append(runtime.model.register_forward_pre_hook(pre_hook))
        hooks.append(runtime.model.register_forward_hook(output_hook))
        plan = [('baseline_warmup', False, False, False), ('baseline_before', False, False, True),
                ('exact16_warmup', True, False, False), ('exact16_1', True, False, True),
                ('exact16_2', True, False, True)]
        if not args.disable_native_sparse:
            plan += [('sparse_warmup', False, True, False), ('sparse_1', False, True, True),
                     ('sparse_2', False, True, True),
                     ('exact16_sparse_warmup', True, True, False), ('exact16_sparse_1', True, True, True),
                     ('exact16_sparse_2', True, True, True)]
        plan.append(('baseline_after', False, False, True))
        reference_logits = None
        previous_placement = None
        for label, use_exact16, use_sparse, measured in plan:
            opt.configure(FULL)
            candidate.configure(dict(exact16=use_exact16, timing=True))
            native_sparse.enabled = use_sparse
            runtime.module.sparse_attn = native_sparse.sparse_attn
            assert runtime.module.sparse_attn == native_sparse.sparse_attn and native_sparse.baseline is opt.sparse
            assert opt.sparse.enabled and opt.reuse.enabled and opt.grouped.enabled
            assert opt.grouped.backend is (candidate.backend if use_exact16 else candidate.original_backend)
            assert runtime.module.act_quant == opt.quant.act_quant
            assert runtime.module.hc_split_sinkhorn == opt.hc.hc_split_sinkhorn
            assert runtime.module.fp4_gemm == original_fp4 and runtime.module.fp8_gemm == original_fp8
            before = candidate.metrics()
            sparse_before = (native_sparse.native_calls, native_sparse.fallback_calls)
            timers.values.clear(); logits.clear()
            replies, failures, placements = [], [], []
            print(json.dumps(dict(running=label, exact16=use_exact16, native_sparse=use_sparse, measured=measured)), flush=True)

            def request_thread():
                try:
                    torch.set_num_threads(16)
                    assert runtime.native.workers == 16
                    replies.append(runtime.generate(ids, count, lambda _: None))
                    if label.endswith('_warmup'):
                        # Keep the request thread alive for its affinity snapshot. The
                        # complete generation and every native timer have stopped here.
                        snapshot = capture_placement(candidate.cache, max_entries=12,
                                                     previous=previous_placement, label=label)
                        snapshot['request_thread_id'] = threading.get_native_id()
                        snapshot['outside_generation_timing'] = True
                        placements.append(snapshot)
                except BaseException as error:
                    failures.append(error)

            thread = threading.Thread(target=request_thread)
            thread.start(); thread.join()
            if failures:
                raise failures[0]
            if placements:
                previous_placement = placements[0]
                placement_path = args.output / (label + '-placement.json')
                atomic_json(placement_path, previous_placement)
            row = replies[0]
            actual_hashes = [hashlib.sha256(value.tobytes()).hexdigest() for value in logits]
            if reference_logits is None:
                reference_logits = list(logits)
            after = candidate.metrics()
            metrics_delta = {key: after[key] - before[key] for key in after if key not in ['packed_bytes', 'packed_entries', 'cap_bytes']}
            row.update(label=label, measured=measured, exact16=use_exact16, native_sparse=use_sparse,
                       decode_tok_s=(len(row['token_ids']) - 1) / row['timings']['decode_seconds'],
                       logits_sha256=actual_hashes, exact_golden=actual_hashes == expected_hashes,
                       tokens_match=row['token_ids'] == golden['token_ids'],
                       logits_max_abs=max(float(abs(a - b).max()) for a, b in zip(reference_logits, logits)),
                       packed_state={key: after[key] for key in ['packed_bytes', 'packed_entries', 'cap_bytes']},
                       exact16_metrics_delta=metrics_delta, other_timers=dict(timers.values),
                       sparse_calls_delta=dict(native=native_sparse.native_calls - sparse_before[0],
                                               fallback=native_sparse.fallback_calls - sparse_before[1]),
                       sparse_fallback_status=dict(native_sparse.fallback_status))
            row['flags_verified'] = True
            if placements:
                row['placement_path'] = str(placement_path)
                row['placement_summary'] = {key: previous_placement[key] for key in
                    ['cache_entries', 'sampled_entries', 'query_page_count', 'sampled_page_node_counts',
                     'sampled_page_error_counts', 'move_pages', 'capture_seconds']}
            result['runs'].append(row)
            atomic_json(args.output / 'result.json', result)
            print(json.dumps({key: row[key] for key in ['label', 'decode_tok_s', 'exact_golden', 'tokens_match', 'packed_state']}), flush=True)
            assert all(sha256(path) == digest for path, digest in protected.items()), 'Selected source drift during trial'
            assert (row['sparse_calls_delta']['native'] > 0) == use_sparse, 'Native sparse flag did not control native decode calls'
            assert (metrics_delta['grouped_native_calls'] > 0) == use_exact16 or (use_exact16 and metrics_delta['cap_fallbacks'] > 0), 'Exact16 flag did not control grouped native calls'
        result['all_exact_golden'] = all(row['exact_golden'] and row['tokens_match'] for row in result['runs'])
        result['zero_measured_downloads'] = all(row['timings']['downloaded_bytes'] == 0 for row in result['runs'] if row['measured'])
        result['zero_measured_packing'] = all(row['exact16_metrics_delta']['pack_count'] == 0
                                            for row in result['runs'] if row['measured'])
        for name, exact, sparse in [('baseline', False, False), ('exact16', True, False),
                                   ('sparse', False, True), ('exact16_sparse', True, True)]:
            rows = [row for row in result['runs'] if row['measured'] and row['exact16'] == exact and row['native_sparse'] == sparse]
            if rows:
                result[name + '_decode_tok_s'] = sum(len(row['token_ids']) - 1 for row in rows) / sum(row['timings']['decode_seconds'] for row in rows)
        assert all(sha256(path) == digest for path, digest in hashes.items()), 'Candidate source drift during trial'
        assert result['all_exact_golden'], 'Candidate changed the complete-model golden logits'
        assert result['zero_measured_downloads'] and result['zero_measured_packing'], 'Measured request was not warm'
        result['passed'] = True
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        for hook in hooks:
            hook.remove()
        timers.restore()
        if native_sparse is not None and runtime is not None:
            runtime.module.sparse_attn = native_sparse.baseline
        if candidate is not None:
            candidate.uninstall()
        if runtime is not None:
            runtime.store.release_all(); runtime.store.close()
        result['selected_sources_preserved'] = all(sha256(path) == digest for path, digest in protected.items())
        atomic_json(args.output / 'result.json', result)


if __name__ == '__main__':
    main()
