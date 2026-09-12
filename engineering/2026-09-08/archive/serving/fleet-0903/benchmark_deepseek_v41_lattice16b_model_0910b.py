#!/usr/bin/env python3
"""Matched complete-model test of exact grouped FP4 row16 arithmetic."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import threading

import torch
import deepseek_v41_server_0910 as server
from deepseek_v41_resident_store_0910 import ResidentStore, bind
from deepseek_v41_lattice16b_runtime_0910 import Lattice16BRuntime
from goal_runtime_0910 import Optimizations
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import sha256

BASE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    proof_path = BASE / 'results/deepseek-v41-lattice16b-0910/kernel-check.json'
    proof = json.loads(proof_path.read_text())
    assert proof['passed'] and proof['fp32_tree_preserved']
    assert all(sha256(p) == h for p, h in proof['source_sha256'].items())
    # Component controls drifted. Permit a diagnostic model comparison only
    # because every candidate arm beats both controls by at least 20%; record
    # the drift, and leave serving promotion to matched real-model evidence.
    for row in proof['benchmarks']:
        values = [arm['seconds_per_call'] for arm in row['arms']]
        assert min(values[0], values[3]) > max(values[1], values[2]) * 1.20
    paths = [Path(__file__), proof_path, BASE / 'deepseek_v41_lattice16b_runtime_0910.py', BASE / 'results/deepseek-v41-lattice16b-0910/libdeepseek-v41-lattice16b.so', BASE / 'deepseek_v41_lattice16_runtime_0910.py',
             BASE / 'deepseek_v41_lattice16_0910.py', BASE / 'goal_vnni_runtime_0910.py',
             BASE / 'deepseek_v41_native_vnni_goal_0910.py', BASE / 'deepseek_v41_native_grouped_vnni_goal_0910.py',
             BASE / 'results/deepseek-v41-lattice16-0910/libdeepseek-v41-lattice16.so',
             BASE / 'results/deepseek-v41-native-vnni-goal-0910/libdeepseek-v41-native-vnni.so',
             BASE / 'goal_runtime_0910.py', BASE / 'goal_grouped_moe_0910.py', BASE / 'goal_hc_0910.py',
             BASE / 'goal_native_quant_0910.py', BASE / 'goal_sparse_script_0910.py', BASE / 'goal_quant_reuse_0910.py',
             BASE / 'deepseek_v41_native_grouped_goal_0910.py',
             BASE / 'results/deepseek-v41-native-grouped-goal-0910/libdeepseek-v41-native-grouped.so',
             BASE / 'results/goal_native_quant_0910/libgoal_native_quant_0910.so',
             BASE / 'results/goal_hc_0910/libgoal_hc_0910.so']
    result = dict(passed=False, source_sha256={str(p): sha256(p) for p in paths}, runs=[],
                  affinity=sorted(os.sched_getaffinity(0)), measurement='single-request decode after prefill',
                  limitations='short greeting; not broad quality or long-context evidence',
                  torch_workers=16, native_workers=16, prefill_unchanged=True,
                  component_control_ratios=[r['control_ratio'] for r in proof['benchmarks']],
                  component_gate='Each candidate arm faster than both controls by at least 20%; component ratios are not selected model speedups')
    runtime = None
    try:
        server.ServingStore, server.bind = ResidentStore, bind
        runtime = server.Runtime(args)
        opt = Optimizations(runtime)
        opt.configure(dict(quant=True, reuse=True, grouped=True, hc=True, sparse=True))
        candidate = Lattice16BRuntime(runtime)
        ids, count = runtime.prepare(dict(messages=[dict(role='user', content='Hi.')], max_tokens=16))
        golden = json.loads((BASE / 'results/deepseek-v41-checkpoint-run-0910b/generation.json').read_text())['runs'][0]
        expected_hashes = [s['logits_sha256'] for s in golden['steps']]
        hashes = []
        runtime.model.register_forward_hook(lambda _m, _a, out: hashes.append(
            hashlib.sha256(out[1].detach().float().numpy().tobytes()).hexdigest()))
        plan = [('baseline_warm', False), ('baseline_before', False), ('candidate_warm', True),
                ('candidate_first', True), ('candidate_second', True), ('baseline_after', False)]
        for label, enabled in plan:
            candidate.configure(enabled)
            candidate.backend.collect_stats = label == 'candidate_warm'
            hashes.clear()
            replies, failures = [], []
            print(json.dumps(dict(running=label, lattice16=enabled)), flush=True)

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
            row.update(label=label, candidate=enabled, logits_sha256=list(hashes),
                       exact_golden=hashes == expected_hashes,
                       decode_tok_s=(len(row['token_ids']) - 1) / row['timings']['decode_seconds'],
                       pack_stats=candidate.stats(), grouped_native_calls=candidate.backend.native_calls,
                       grouped_fallback_calls=candidate.backend.fallback_calls)
            result['runs'].append(row)
            atomic_json(args.output / 'result.json', result)
            print(json.dumps({k: row[k] for k in ('label', 'decode_tok_s', 'exact_golden', 'pack_stats')}), flush=True)
            assert row['token_ids'] == golden['token_ids'] and row['exact_golden'], label
            assert row['timings']['downloaded_bytes'] == 0, label
        controls = [r for r in result['runs'] if r['label'] in ('baseline_before', 'baseline_after')]
        measured = [r for r in result['runs'] if r['label'] in ('candidate_first', 'candidate_second')]
        result['baseline_decode_tok_s'] = 18 / sum(r['timings']['decode_seconds'] for r in controls)
        result['candidate_decode_tok_s'] = 18 / sum(r['timings']['decode_seconds'] for r in measured)
        result['speedup'] = result['candidate_decode_tok_s'] / result['baseline_decode_tok_s']
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
