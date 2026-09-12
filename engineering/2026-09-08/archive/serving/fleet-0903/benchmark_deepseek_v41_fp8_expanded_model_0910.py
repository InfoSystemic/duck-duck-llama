#!/usr/bin/env python3
"""Real-model FP8 expansion A/B and a profile of residual CPU dispatch costs.

Short fixed-prompt timing is diagnostic only. The optional expanded candidate
requires completed component correctness proof and credible component gains.
"""
import argparse
from collections import defaultdict
import cProfile
import functools
import hashlib
import io
import json
import os
from pathlib import Path
import pstats
import threading
import time

import torch
import deepseek_v41_server_0910 as server
from deepseek_v41_resident_store_0910 import ResidentStore, bind
from deepseek_v41_fp8_bf16_0910 import ExpandedFP8
from deepseek_v41_native_bridge_0910 import NativeGemm
from deepseek_v41_native_grouped_goal_0910 import GroupedNativeGemm
from goal_runtime_0910 import Optimizations
from goal_hc_0910 import NativeHC
from goal_native_quant_0910 import NativeQuant
from goal_sparse_script_0910 import SparseScript
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import sha256

BASE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    library = BASE / 'results/deepseek-v41-fp8-bf16-0910/libdeepseek-v41-fp8-bf16.so'
    proof_path = library.with_name('kernel-check.json')
    proof = json.loads(proof_path.read_text())
    assert proof['passed'] and all(sha256(p) == h for p, h in proof['source_sha256'].items())
    useful = sum(r['speedup'] >= 1.10 and r['control_ratio'] <= 1.10 for r in proof['benchmarks']) >= 2
    paths = [Path(__file__), proof_path, library, BASE / 'deepseek_v41_fp8_bf16_0910.py',
             BASE / 'goal_runtime_0910.py', BASE / 'goal_grouped_moe_0910.py', BASE / 'goal_hc_0910.py',
             BASE / 'goal_native_quant_0910.py', BASE / 'goal_sparse_script_0910.py', BASE / 'goal_quant_reuse_0910.py',
             BASE / 'deepseek_v41_native_grouped_goal_0910.py',
             BASE / 'results/deepseek-v41-native-grouped-goal-0910/libdeepseek-v41-native-grouped.so',
             BASE / 'results/goal_native_quant_0910/libgoal_native_quant_0910.so',
             BASE / 'results/goal_hc_0910/libgoal_hc_0910.so']
    result = dict(passed=False, source_sha256={str(p): sha256(p) for p in paths}, runs=[],
                  affinity=sorted(os.sched_getaffinity(0)), expanded_component_gate_passed=useful,
                  timer_semantics='inclusive; nested durations overlap', measurement='single-request decode after prefill',
                  limitations='10-token greeting, including EOS; cProfile run is not a speed benchmark')
    timers = defaultdict(lambda: [0, 0.0])
    phase = ['loading']

    def instrument(owner, name, label):
        original = getattr(owner, name)

        @functools.wraps(original)
        def measured(*values, **kwargs):
            key = label(*values, **kwargs) if callable(label) else label
            began = time.perf_counter()
            try:
                return original(*values, **kwargs)
            finally:
                row = timers[phase[0] + ':' + key]
                row[0] += 1
                row[1] += time.perf_counter() - began

        setattr(owner, name, measured)

    instrument(NativeGemm, 'apply', lambda self, mode, a, asc, b, bsc: f'gemm_fp{mode}_{tuple(b.shape)}')
    instrument(GroupedNativeGemm, 'apply', 'grouped_gemm')
    instrument(ExpandedFP8, 'apply_packed', 'expanded_gemm')
    instrument(NativeQuant, 'act_quant', 'quant')
    instrument(NativeHC, 'hc_split_sinkhorn', 'hc_sinkhorn')
    instrument(SparseScript, '__call__', 'sparse')
    instrument(torch.nn.functional, 'linear', lambda x, w, *args, **kwargs: f'linear_{tuple(w.shape)}')
    instrument(torch, 'einsum', lambda equation, *args: 'einsum_' + equation)
    runtime = None
    try:
        server.ServingStore, server.bind = ResidentStore, bind
        runtime = server.Runtime(args)
        opt = Optimizations(runtime)
        opt.configure(dict(quant=True, reuse=True, grouped=True, hc=True, sparse=True))
        for name in ('hc_pre', 'hc_post', 'hc_mixes'):
            instrument(runtime.module.Block, name, name)
        instrument(runtime.module.RMSNorm, 'forward', 'rmsnorm')
        instrument(runtime.module.Engram, 'forward', 'engram')
        instrument(runtime.module.Indexer, 'forward', 'indexer')
        instrument(runtime.module.Attention, 'forward', 'attention')
        instrument(type(opt.grouped), 'forward', 'moe')
        candidate = ExpandedFP8(library, 16).install(runtime)
        candidate.enabled = False
        ids, count = runtime.prepare(dict(messages=[dict(role='user', content='Hi.')], max_tokens=16))
        golden = json.loads((BASE / 'results/deepseek-v41-checkpoint-run-0910b/generation.json').read_text())['runs'][0]
        expected = [s['logits_sha256'] for s in golden['steps']]
        hashes = []
        profiler = cProfile.Profile()
        profiling = [False]

        def before(_module, values):
            phase[0] = 'prefill' if values[1] == 0 else 'decode'
            if profiling[0] and phase[0] == 'decode':
                profiler.enable()

        def after(_module, _values, output):
            if profiling[0]:
                profiler.disable()
            hashes.append(hashlib.sha256(output[1].detach().float().numpy().tobytes()).hexdigest())

        runtime.model.register_forward_pre_hook(before)
        runtime.model.register_forward_hook(after)
        plan = [('warmup', False, 1), ('baseline_before', False, 1)]
        if useful:
            plan += [('expanded_warm', True, 1), ('expanded_t1', True, 1), ('expanded_t2', True, 2),
                     ('expanded_t4', True, 4), ('expanded_t1_repeat', True, 1)]
        plan += [('baseline_after', False, 1), ('decode_cprofile', False, 1)]
        for label, enabled, tile in plan:
            candidate.enabled, candidate.tile = enabled, tile
            profiling[0] = label == 'decode_cprofile'
            hashes.clear(); timers.clear()
            replies, failures = [], []
            print(json.dumps(dict(running=label, enabled=enabled, tile=tile)), flush=True)

            def request_thread():
                try:
                    torch.set_num_threads(16)
                    replies.append(runtime.generate(ids, count, lambda _: None))
                except BaseException as error:
                    failures.append(error)

            thread = threading.Thread(target=request_thread)
            thread.start(); thread.join()
            if failures:
                raise failures[0]
            row = replies[0]
            row.update(label=label, expanded=enabled, tile=tile, profiling=profiling[0],
                       decode_tok_s=(len(row['token_ids']) - 1) / row['timings']['decode_seconds'],
                       logits_sha256=list(hashes), exact_golden=hashes == expected,
                       operations={k: dict(calls=v[0], inclusive_seconds=v[1]) for k, v in sorted(timers.items())},
                       expanded_stats={k: getattr(candidate, k) for k in ('packed_bytes', 'pack_count', 'pack_seconds', 'hits', 'misses', 'fallback_calls')})
            result['runs'].append(row)
            atomic_json(args.output / 'result.json', result)
            print(json.dumps({k: row[k] for k in ('label', 'decode_tok_s', 'exact_golden', 'expanded_stats')}), flush=True)
            assert row['token_ids'] == golden['token_ids'] and row['exact_golden'], label
            assert row['timings']['downloaded_bytes'] == 0, label
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats('cumulative')
        stats.print_stats(100)
        stats.sort_stats('tottime').print_stats(100)
        (args.output / 'cprofile.txt').write_text(stream.getvalue())
        records = [dict(file=k[0], line=k[1], function=k[2], primitive_calls=v[0], total_calls=v[1],
                        self_seconds=v[2], cumulative_seconds=v[3]) for k, v in stats.stats.items()]
        atomic_json(args.output / 'cprofile.json', sorted(records, key=lambda r: -r['cumulative_seconds'])[:250])
        result['cprofile_sha256'] = sha256(args.output / 'cprofile.txt')
        assert all(sha256(p) == h for p, h in result['source_sha256'].items())
        result['passed'] = True
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        if runtime is not None:
            runtime.store.release_all(); runtime.store.close()
        atomic_json(args.output / 'result.json', result)


if __name__ == '__main__':
    main()
