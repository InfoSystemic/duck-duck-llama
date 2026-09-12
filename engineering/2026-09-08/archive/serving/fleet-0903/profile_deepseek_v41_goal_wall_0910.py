#!/usr/bin/env python3
"""Measure inclusive CPU wall time by operation on the real, cached text graph."""
import argparse
from collections import defaultdict
import cProfile
import functools
import hashlib
import io
import json
from pathlib import Path
import pstats
import time
import threading

import deepseek_v41_cpu_reference_0910 as cpu
import deepseek_v41_checkpoint_0910 as checkpoint
import deepseek_v41_server_0910 as server
from deepseek_v41_native_bridge_0910 import NativeGemm
from deepseek_v41_resident_store_0910 import ResidentStore as ServingStore, bind
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import sha256

BASE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    golden_path = BASE / 'results/deepseek-v41-checkpoint-run-0910b/generation.json'
    golden = json.loads(golden_path.read_text())['runs'][0]
    expected_hashes = [s['logits_sha256'] for s in golden['steps']]
    counts = defaultdict(lambda: [0, 0.0])
    phase = ['loading']

    def instrument(owner, name, label=None):
        original = getattr(owner, name)

        @functools.wraps(original)
        def measured(*values, **kwargs):
            key = label(*values, **kwargs) if callable(label) else (label or name)
            began = time.perf_counter()
            try:
                return original(*values, **kwargs)
            finally:
                row = counts[phase[0] + ':' + key]
                row[0] += 1
                row[1] += time.perf_counter() - began

        setattr(owner, name, measured)

    instrument(NativeGemm, 'apply', lambda self, mode, a, *_: 'gemm_fp' + str(mode))
    for name in ['act_quant', 'fp4_act_quant', 'sparse_attn', 'hc_split_sinkhorn']:
        instrument(cpu, name)
    instrument(checkpoint, 'load_parameters')
    for name in ['ensure', 'tensor', 'rows']:
        instrument(ServingStore, name, 'store_' + name)
    result = dict(passed=False, source_sha256=sha256(__file__), golden_sha256=sha256(golden_path),
                  inclusive_timers=True, timers_can_overlap=True, runs=[])
    runtime = None
    try:
        server.ServingStore = ServingStore; server.bind = bind
        runtime = server.Runtime(args)
        ids, count = runtime.prepare(dict(model=server.MODEL, messages=[dict(role='user', content='Hi.')],
                                          temperature=0, max_tokens=16))
        hashes = []
        step_times = []
        began = [0.0]
        profiler = cProfile.Profile()
        profiling = [False]

        def before(_module, values):
            phase[0] = 'prefill' if values[1] == 0 else 'decode'
            began[0] = time.perf_counter()
            if profiling[0] and phase[0] == 'decode':
                profiler.enable()

        def after(_module, _values, output):
            if profiling[0]:
                profiler.disable()
            step_times.append(time.perf_counter() - began[0])
            hashes.append(hashlib.sha256(output[1].detach().float().contiguous().numpy().tobytes()).hexdigest())

        runtime.model.register_forward_pre_hook(before)
        runtime.model.register_forward_hook(after)
        for index, name in enumerate(['warmup', 'wall_timers', 'decode_cprofile']):
            counts.clear(); hashes.clear(); step_times.clear()
            profiling[0] = index == 2
            print(json.dumps(dict(running=name)), flush=True)
            replies = []; failures = []
            def request_thread():
                try: replies.append(runtime.generate(ids, count, lambda _: None))
                except BaseException as error: failures.append(error)
            thread = threading.Thread(target=request_thread)
            thread.start(); thread.join()
            if failures: raise failures[0]
            row = replies[0]
            assert row['timings']['downloaded_bytes'] == 0
            assert row['token_ids'] == golden['token_ids'] and hashes == expected_hashes
            assert row['content'] == 'Hello! How can I help you today?' and row['finish_reason'] == 'stop'
            row.update(name=name, logits_sha256=list(hashes), step_seconds=list(step_times),
                       operations={k: dict(calls=v[0], inclusive_seconds=v[1]) for k, v in sorted(counts.items())})
            result['runs'].append(row)
            atomic_json(args.output / 'result.json', result)
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats('cumulative')
        stats.print_stats(100)
        stats.sort_stats('tottime').print_stats(100)
        (args.output / 'cprofile.txt').write_text(stream.getvalue())
        rows = [dict(file=k[0], line=k[1], function=k[2], primitive_calls=v[0], total_calls=v[1],
                     self_seconds=v[2], cumulative_seconds=v[3]) for k, v in stats.stats.items()]
        atomic_json(args.output / 'cprofile.json', sorted(rows, key=lambda r: -r['cumulative_seconds'])[:250])
        result.update(passed=True, exact_logits_all_runs=True, zero_downloads=True,
                      cprofile_sha256=sha256(args.output / 'cprofile.txt'))
    except BaseException as error:
        result['error'] = type(error).__name__
        raise
    finally:
        if runtime is not None:
            runtime.store.close()
        atomic_json(args.output / 'result.json', result)
        print(json.dumps(dict(passed=result['passed'], error=result.get('error'))), flush=True)


if __name__ == '__main__':
    main()
