#!/usr/bin/env python3
"""Bounded exact attention fixtures and component timings; no model is loaded."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import types

import torch

from sparse_prefill_0912 import FLEET_0903, NativeSparsePrefill
import deepseek_v41_cpu_reference_0910 as cpu
from goal_sparse_script_0910 import scripted_sparse_attn


def raw_equal(left, right):
    return (left.shape == right.shape and left.dtype == right.dtype
            and torch.equal(left.detach().resolve_neg().contiguous().view(torch.uint8),
                            right.detach().resolve_neg().contiguous().view(torch.uint8)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--benchmark', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.set_flush_denormal(False)
    torch.manual_seed(410912)
    native = NativeSparsePrefill(baseline=scripted_sparse_attn)
    cases = []

    def check(name, q, kv, sink, ids, expected_rows, scale=None):
        scale = q.shape[-1] ** -.5 if scale is None else scale
        originals = [value.detach().clone() for value in (q, kv, sink, ids)]
        before = native.native_calls
        expected = cpu.sparse_attn(q, kv, sink, ids, scale)
        scripted = scripted_sparse_attn(q, kv, sink, ids, scale)
        actual = native.sparse_attn(q, kv, sink, ids, scale)
        assert raw_equal(actual, expected), (name, 'native output', (actual.float()-expected.float()).abs().max())
        assert raw_equal(scripted, expected), (name, 'script output')
        assert native.native_calls - before == expected_rows, (name, native.fallback_status)
        assert all(raw_equal(a, b) for a, b in zip((q, kv, sink, ids), originals)), name
        cases.append(dict(name=name, query_shape=list(q.shape), positions=ids.shape[-1],
                          native_rows=expected_rows, exact_output_values=actual.numel()))

    for seq, heads, dim in [(2, 1, 7), (7, 4, 16), (6, 64, 512), (17, 16, 128)]:
        for count in (0, 1, 63, 64, 65, 129, 257):
            q = torch.randn(1, seq, heads, dim).bfloat16()
            kv = torch.randn(1, 277, dim).bfloat16()
            sink = torch.randn(heads)
            ids = torch.randint(-1, 277, (1, seq, count), dtype=torch.int32)
            check(f'random_{seq}_{heads}_{dim}_{count}', q, kv, sink, ids, seq)

    seq, heads, dim = 6, 64, 512
    q = torch.randn(1, seq, heads, dim).bfloat16()
    kv = torch.randn(1, 320, dim).bfloat16()
    sink = torch.randn(heads)
    ids = torch.arange(129).reshape(1, 1, -1).expand(1, seq, 129).contiguous().int()
    causal = torch.where(ids > torch.arange(seq).reshape(1, seq, 1), -1, ids)
    check('causal_prefill', q, kv, sink, causal, seq)
    check('all_masked', q, kv, sink, torch.full_like(ids, -1), seq)
    mask = ids.clone()
    mask[:, :2, :64] = -1
    mask[:, 2:4, 64:128] = -1
    mask[:, 4:, 128:] = -1
    check('masked_chunks_at_different_queries', q, kv, sink, mask, seq)
    check('duplicate_positions', q, kv, sink, ids % 7, seq)
    check('strided_kv_sink_and_indices', q, kv[:, ::2],
          torch.stack([sink, sink], -1).flatten()[::2],
          torch.stack([ids, ids], -1).flatten(-2)[..., ::2], seq)
    check('int64_indices', q, kv, sink, ids.long(), seq)
    check('broadcast_indices', q, kv, sink, ids[:, :1].expand(1, seq, 129), seq)
    shifted = torch.cat([torch.zeros(1, dtype=torch.bfloat16), q.flatten()])[1:].reshape(q.shape)
    check('nonzero_query_storage_offset', shifted, kv, sink, ids, seq)
    check('decode_unchanged', q[:, :1], kv, sink, ids[:, :1], 1)
    check('autograd_fallback', q.clone().requires_grad_(True), kv, sink, ids, 0)
    check('batch_fallback', q.expand(2, *q.shape[1:]).contiguous(),
          kv.expand(2, *kv.shape[1:]).contiguous(), sink,
          ids.expand(2, *ids.shape[1:]).contiguous(), 0)
    check('lazy_negative_fallback', torch._neg_view(q), kv, sink, ids, 0)
    check('noncontiguous_query_fallback', q.transpose(-1, -2),
          torch.randn(1, 320, 64).bfloat16(), torch.randn(512), ids, 0)
    check('zero_scale_fallback', q, kv, sink, ids, 0, scale=0.)
    nonfinite = q.clone()
    nonfinite[:, 2, 0, 0] = float('nan')
    check('one_nonfinite_row_fallback', nonfinite, kv, sink, ids, seq - 1)
    native.enabled = False
    check('disabled_fallback', q, kv, sink, ids, 0)
    native.enabled = True
    torch.set_flush_denormal(True)
    try:
        check('denormal_mode_fallback', q, kv, sink, ids, 0)
    finally:
        torch.set_flush_denormal(False)
    strict = NativeSparsePrefill(baseline=cpu.sparse_attn)
    for invalid_index in (-2, kv.shape[1]):
        invalid = ids.clone()
        invalid[:, -1, -1] = invalid_index
        for function in (cpu.sparse_attn, strict.sparse_attn):
            try:
                function(q, kv, sink, invalid, 512 ** -.5)
            except AssertionError:
                pass
            else:
                raise AssertionError(('invalid index was accepted', invalid_index))
    module = types.SimpleNamespace(sparse_attn=cpu.sparse_attn)
    strict.install(module)
    assert module.sparse_attn == strict.sparse_attn and strict.baseline is cpu.sparse_attn
    strict.install(module)
    assert strict.baseline is cpu.sparse_attn

    timings = []
    if args.benchmark:
        for seq, count in [(6, 6), (6, 9), (32, 48), (128, 192)]:
            q = torch.randn(1, seq, 64, 512).bfloat16()
            kv = torch.randn(1, max(count, seq), 512).bfloat16()
            sink = torch.randn(64)
            ids = torch.arange(count).reshape(1, 1, -1).expand(1, seq, count).contiguous().int()
            ids = torch.where(ids >= torch.arange(1, seq + 1).reshape(1, seq, 1) * count // seq, -1, ids)
            values = (q, kv, sink, ids, 512 ** -.5)
            assert raw_equal(native.sparse_attn(*values), scripted_sparse_attn(*values))
            for _ in range(2):
                native.sparse_attn(*values)
                scripted_sparse_attn(*values)
            arms = {'native': [], 'script': []}
            for repeat in range(7):
                order = [('native', native.sparse_attn), ('script', scripted_sparse_attn)]
                if repeat % 2:
                    order.reverse()
                for name, function in order:
                    start = time.perf_counter()
                    function(*values)
                    arms[name].append(time.perf_counter() - start)
            timings.append(dict(query_shape=list(q.shape), positions=count, seconds=arms,
                                median_speedup=statistics.median(arms['script']) / statistics.median(arms['native'])))
    paths = [Path(__file__), Path(__file__).with_name('sparse_prefill_0912.py'),
             FLEET_0903 / 'goal_sparse_native_0910.py', FLEET_0903 / 'goal_sparse_script_0910.py',
             FLEET_0903 / 'results/goal_sparse_native_0910/libgoal_sparse_native_0910.so',
             FLEET_0903 / 'deepseek_v41_cpu_reference_0910.py']
    result = dict(passed=True, component_only=True, full_checkpoint_loaded=False,
                  service_modified=False, torch_version=torch.__version__,
                  torch_threads=torch.get_num_threads(), affinity=sorted(os.sched_getaffinity(0)),
                  exact_attention_cases=len(cases), exact_output_values=sum(x['exact_output_values'] for x in cases),
                  cases=cases, timings=timings, prefill_calls=native.prefill_calls,
                  prefill_rows=native.prefill_rows, native_calls=native.native_calls,
                  fallback_calls=native.fallback_calls, fallback_status=dict(native.fallback_status),
                  invalid_indices_rejected=True, idempotent_opt_in_install=True,
                  sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('cases', 'sha256', 'timings')}))
    for timing in timings:
        print(json.dumps(timing))


if __name__ == '__main__':
    with torch.inference_mode():
        main()
