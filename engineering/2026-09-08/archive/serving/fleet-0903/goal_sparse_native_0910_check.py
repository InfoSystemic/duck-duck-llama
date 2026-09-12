#!/usr/bin/env python3
"""One-core exact matmul/sparse-attention fixtures; no model or timing trial."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import types

import torch

import deepseek_v41_cpu_reference_0910 as cpu
from goal_sparse_native_0910 import NativeSparse
from goal_sparse_script_0910 import scripted_sparse_attn

BASE = Path(__file__).resolve().parent


def raw_equal(a, b):
    return (a.shape == b.shape and a.dtype == b.dtype
            and torch.equal(a.detach().resolve_neg().contiguous().view(torch.uint8),
                            b.detach().resolve_neg().contiguous().view(torch.uint8)))


def main():
    assert os.sched_getaffinity(0) == {0}, 'run with taskset -c 0'
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_flush_denormal(False)
    torch.manual_seed(410910)
    library = BASE / 'results/goal_sparse_native_0910/libgoal_sparse_native_0910.so'
    native = NativeSparse(library)
    fn = native.library.ds41_sparse_matmul_fixture
    fn.argtypes = [ctypes.c_int] * 4 + [ctypes.c_void_p] * 3
    fn.restype = ctypes.c_int
    matmul_cases = []
    for h, c, d in [(1, 1, 1), (1, 3, 7), (4, 7, 16), (16, 17, 128),
                    (64, 1, 512), (64, 7, 512), (64, 31, 512), (64, 63, 512),
                    (64, 64, 512), (128, 64, 1024)]:
        for mode in (0, 1):
            for source in ('float32_random', 'bf16_random', 'offset_float32'):
                a = torch.randn(h, d if mode == 0 else c)
                values = torch.randn(c, d)
                if source == 'bf16_random':
                    a, values = a.bfloat16().float(), values.bfloat16().float()
                elif source == 'offset_float32':
                    a = torch.cat([torch.zeros(1), a.flatten()])[1:].reshape(a.shape)
                    values = torch.cat([torch.zeros(1), values.flatten()])[1:].reshape(values.shape)
                output = torch.empty(h, c if mode == 0 else d)
                assert fn(mode, h, c, d, a.data_ptr(), values.data_ptr(), output.data_ptr()) == 0
                expected = a @ values.T if mode == 0 else a @ values
                assert raw_equal(output, expected), ('BLAS parity', mode, h, c, d, source,
                                                      float((output - expected).abs().max()))
                matmul_cases.append(dict(mode=mode, heads=h, positions=c, dim=d, source=source,
                                         exact_fp32_bytes=True, output_values=output.numel()))
    cases = []

    def check(name, q, kv, sink, ids, scale=None, expected_native=True):
        scale = q.shape[-1] ** -0.5 if scale is None else scale
        originals = [value.detach().clone() for value in (q, kv, sink, ids)]
        before = native.native_calls
        expected = cpu.sparse_attn(q, kv, sink, ids, scale)
        scripted = scripted_sparse_attn(q, kv, sink, ids, scale)
        actual = native.sparse_attn(q, kv, sink, ids, scale)
        assert raw_equal(actual, expected), (name, 'native bytes', float((actual - expected).abs().max()))
        assert raw_equal(scripted, expected), (name, 'existing selected script bytes')
        assert native.native_calls - before == int(expected_native), (name, native.fallback_status)
        assert all(raw_equal(a, b) for a, b in zip((q, kv, sink, ids), originals)), (name, 'input mutation')
        cases.append(dict(name=name, q_shape=list(q.shape), kv_shape=list(kv.shape),
                          index_count=ids.shape[-1], native=expected_native, exact_bf16_bytes=True,
                          script_also_exact=True, output_values=actual.numel()))

    for h, d in [(1, 7), (4, 16), (16, 128), (64, 512)]:
        for count in [0, 1, 7, 63, 64, 65, 127, 128, 135, 193, 257]:
            for magnitude in [0.25, 4.0]:
                q = (torch.randn(1, 1, h, d) * magnitude).bfloat16()
                kv = torch.randn(1, 277, d).bfloat16()
                sink = torch.randn(h)
                ids = torch.randint(-1, kv.shape[1], (1, 1, count), dtype=torch.int32)
                check(f'random_{h}_{d}_{count}_{magnitude}', q, kv, sink, ids)

    q = torch.randn(1, 1, 64, 512).bfloat16()
    kv = torch.randn(1, 320, 512).bfloat16()
    sink = torch.nn.Parameter(torch.randn(64), requires_grad=False)
    patterns = {
        'all_invalid': torch.full((1, 1, 193), -1, dtype=torch.int32),
        'first_chunk_invalid': torch.cat([torch.full((64,), -1), torch.arange(129)]).reshape(1, 1, -1).int(),
        'middle_chunk_invalid': torch.cat([torch.arange(64), torch.full((64,), -1), torch.arange(65)]).reshape(1, 1, -1).int(),
        'last_chunk_invalid': torch.cat([torch.arange(129), torch.full((64,), -1)]).reshape(1, 1, -1).int(),
        'duplicate_positions': torch.tensor([3, 3, -1, 3, 7, 7, 8] * 29).reshape(1, 1, -1).int(),
        'reverse_positions': torch.arange(256, -1, -1).reshape(1, 1, -1).long(),
    }
    for name, ids in patterns.items():
        check(name, q, kv, sink, ids)
    ids = torch.arange(0, 128).reshape(1, 1, -1).int()
    strided_ids = torch.stack([ids, ids], -1).flatten(-2)[..., ::2]
    check('strided_kv_sink_and_indices', q, kv[:, ::2],
          torch.stack([sink, sink], -1).flatten()[::2], strided_ids)
    bad_unused_kv = kv.clone()
    bad_unused_kv[:, 0] = float('nan')
    ids = torch.tensor([-1, 1, 2, 3] * 33).reshape(1, 1, -1).int()
    check('masked_invalid_gather_does_not_read_nan_row_zero', q, bad_unused_kv, sink, ids)
    qzeros = torch.zeros(1, 1, 4, 16, dtype=torch.bfloat16)
    kvzeros = torch.zeros(1, 100, 16, dtype=torch.bfloat16)
    qzeros[..., 1::2] = -0.0
    kvzeros[..., 1::2] = -0.0
    check('signed_zeros', qzeros, kvzeros, torch.zeros(4), torch.arange(65).reshape(1, 1, -1).int())
    qexp = torch.zeros(1, 1, 4, 16, dtype=torch.bfloat16)
    qexp[..., 0] = 1
    kvexp = torch.zeros(1, 128, 16, dtype=torch.bfloat16)
    kvexp[0, :, 0] = torch.linspace(-110, 0, 128).bfloat16()
    kvexp[0, :, 1:] = torch.randn(128, 15).bfloat16()
    check('exponential_underflow_and_sink_extremes', qexp, kvexp,
          torch.tensor([-100.0, -1.0, 1.0, 100.0]), torch.arange(128).reshape(1, 1, -1), scale=1.0)

    # Unsupported paths always preserve the baseline rather than changing shape,
    # stride/autograd behavior or asserting a weaker equivalence.
    ids = torch.arange(65).reshape(1, 1, -1).int()
    check('prefill_fallback', q.expand(1, 2, 64, 512).contiguous(), kv, sink,
          ids.expand(1, 2, 65).contiguous(), expected_native=False)
    check('batch_fallback', q.expand(2, 1, 64, 512).contiguous(), kv.expand(2, 320, 512).contiguous(),
          sink, ids.expand(2, 1, 65).contiguous(), expected_native=False)
    check('noncontiguous_q_fallback', q.transpose(-1, -2), torch.randn(1, 320, 64).bfloat16(),
          torch.randn(512), ids, expected_native=False)
    check('autograd_fallback', q.clone().requires_grad_(True), kv, sink, ids, expected_native=False)
    check('lazy_negative_fallback', torch._neg_view(q), kv, sink, ids, expected_native=False)
    check('scale_zero_fallback', q, kv, sink, ids, scale=0.0, expected_native=False)
    nonfinite_q = q.clone()
    nonfinite_q[0, 0, 0, 0] = float('nan')
    check('nonfinite_q_fallback', nonfinite_q, kv, sink, ids, expected_native=False)
    nonfinite_sink = sink.detach().clone()
    nonfinite_sink[0] = -float('inf')
    check('nonfinite_sink_fallback', q, kv, nonfinite_sink, ids, expected_native=False)
    for bad in [-2, kv.shape[1]]:
        invalid = ids.clone()
        invalid[0, 0, -1] = bad
        for function in [cpu.sparse_attn, native.sparse_attn]:
            try:
                function(q, kv, sink, invalid, 512 ** -0.5)
            except AssertionError:
                pass
            else:
                raise AssertionError(('invalid index not rejected', bad))
    torch.set_flush_denormal(True)
    try:
        check('flush_denormal_fallback', q, kv, sink, ids, expected_native=False)
    finally:
        torch.set_flush_denormal(False)
    native.enabled = False
    check('disabled_fallback', q, kv, sink, ids, expected_native=False)
    native.enabled = True
    module = types.SimpleNamespace(sparse_attn=cpu.sparse_attn)
    original = module.sparse_attn
    assert native.install(module) is native and native.baseline is original
    assert native.install(module) is native and native.baseline is original
    module.sparse_attn = native.baseline
    assert module.sparse_attn is original and cpu.sparse_attn is original

    paths = [Path(__file__), library, BASE / 'goal_sparse_native_0910.cpp',
             BASE / 'goal_sparse_native_0910.py', BASE / 'goal_sparse_script_0910.py',
             BASE / 'deepseek_v41_cpu_reference_0910.py']
    result = dict(passed=True, component_only=True, full_checkpoint_loaded=False,
                  performance_trial=False, cpu_affinity=[0], torch_threads=1,
                  torch_version=torch.__version__, matmul_cases=matmul_cases, cases=cases,
                  exact_fp32_matmul_cases=len(matmul_cases), exact_attention_cases=len(cases),
                  native_calls=native.native_calls, fallback_calls=native.fallback_calls,
                  fallback_status=dict(native.fallback_status), probability_sum_uses_aten=True,
                  full_64_position_online_softmax=True, max_attention_output_error=0.0,
                  module_only_install_restore=True,
                  sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    output = BASE / 'results/goal_sparse_native_0910/fixture-check.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('cases', 'matmul_cases', 'sha256')}))


if __name__ == '__main__':
    main()
