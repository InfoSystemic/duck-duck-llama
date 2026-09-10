#!/usr/bin/env python3
"""Compare demand vs retained mappings in fresh threads matching HTTP requests."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import threading
import torch
from torch.overrides import _get_current_function_mode_stack
import deepseek_v41_server_0910 as server
from deepseek_v41_resident_store_0910 import ResidentStore, bind
from check_deepseek_v41_resident_store_0910 import check
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import sha256

BASE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    inputs = [Path(__file__), BASE / 'deepseek_v41_resident_store_0910.py', BASE / 'check_deepseek_v41_resident_store_0910.py']
    source_hashes = {str(p): sha256(p) for p in inputs}
    result = dict(passed=False, promote=False, source_sha256=source_hashes, runs=[])
    runtime = None
    try:
        result['cache_check'] = check(args.output / 'cache-check.json')
        server.ServingStore = ResidentStore; server.bind = bind
        runtime = server.Runtime(args)
        golden = json.loads((BASE / 'results/deepseek-v41-checkpoint-run-0910b/generation.json').read_text())['runs'][0]
        expected_hashes = [s['logits_sha256'] for s in golden['steps']]
        ids, count = runtime.prepare(dict(messages=[dict(role='user', content='Hi.')], max_tokens=16))
        hashes = []

        def logits_hook(_module, _inputs, output):
            hashes.append(hashlib.sha256(output[1].detach().float().contiguous().numpy().tobytes()).hexdigest())

        runtime.model.register_forward_hook(logits_hook)
        for label, enabled, measured in [('demand_warmup', False, False), ('demand_1', False, True),
            ('resident_warmup', True, False), ('resident_1', True, True), ('resident_2', True, True), ('demand_2', False, True)]:
            if not enabled:
                runtime.store.release_all()
            runtime.store.resident_enabled = enabled
            hashes.clear(); replies = []; failures = []
            loads = runtime.store.tensor_loads
            print(json.dumps(dict(running=label)), flush=True)

            def request_thread():
                try:
                    # set_default_device in Runtime.__init__ is thread local.
                    # HTTP inference likewise runs in a new thread without it.
                    assert not _get_current_function_mode_stack()
                    assert torch.get_default_device().type == 'cpu' and torch.get_num_threads() == 16
                    assert os.sched_getaffinity(0) == set(range(48, 64))
                    replies.append(runtime.generate(ids, count, lambda _: None))
                except BaseException as error:
                    failures.append(error)

            thread = threading.Thread(target=request_thread)
            thread.start(); thread.join()
            if failures:
                raise failures[0]
            row = replies[0]
            assert row['token_ids'] == golden['token_ids'] and hashes == expected_hashes
            assert row['content'] == 'Hello! How can I help you today?' and row['finish_reason'] == 'stop'
            assert row['timings']['downloaded_bytes'] == 0
            row.update(label=label, measured=measured, resident=enabled, logits_sha256=list(hashes),
                       tensor_loads=runtime.store.tensor_loads - loads, resident_experts=len(runtime.store.residents),
                       decode_tok_s=9 / row['timings']['decode_seconds'])
            if enabled and measured:
                assert row['tensor_loads'] == 0
            result['runs'].append(row)
            atomic_json(args.output / 'result.json', result)
        for enabled, name in [(False, 'demand'), (True, 'resident')]:
            rows = [r for r in result['runs'] if r['measured'] and r['resident'] == enabled]
            result[name + '_decode_tok_s'] = 18 / sum(r['timings']['decode_seconds'] for r in rows)
            result[name + '_decode_range_tok_s'] = [min(r['decode_tok_s'] for r in rows), max(r['decode_tok_s'] for r in rows)]
        result['speedup'] = result['resident_decode_tok_s'] / result['demand_decode_tok_s']
        assert all(sha256(p) == h for p, h in source_hashes.items())
        result.update(passed=True, exact_logits_all_runs=True, zero_downloads=True,
                      fresh_request_threads=True, promote=result['speedup'] > 1.05,
                      improvement_threshold=1.05, imc_bandwidth_measured=False)
    except BaseException as error:
        result['error'] = type(error).__name__
        raise
    finally:
        if runtime is not None:
            runtime.store.release_all(); runtime.store.close()
        atomic_json(args.output / 'result.json', result)
        print(json.dumps({k: result[k] for k in ['passed', 'promote', 'speedup', 'error'] if k in result}), flush=True)


if __name__ == '__main__':
    main()
