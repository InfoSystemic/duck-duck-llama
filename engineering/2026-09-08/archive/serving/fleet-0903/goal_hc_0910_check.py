#!/usr/bin/env python3
"""One-core HC fixtures, with no model/checkpoint loading or timed benchmark."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import types

import torch

import deepseek_v41_cpu_reference_0910 as cpu
from goal_hc_0910 import NativeHC

BASE = Path(__file__).resolve().parent


def raw_equal(a, b):
    return (a.shape == b.shape and a.dtype == b.dtype
            and torch.equal(a.resolve_neg().contiguous().view(torch.uint8),
                            b.resolve_neg().contiguous().view(torch.uint8)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--library', type=Path, default=BASE / 'results/goal_hc_0910/libgoal_hc_0910.so')
    parser.add_argument('--output', type=Path, default=BASE / 'results/goal_hc_0910/fixture-check.json')
    args = parser.parse_args()
    assert os.sched_getaffinity(0) == {0}, 'run with taskset -c 0'
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_flush_denormal(False)
    torch.manual_seed(410910)
    native = NativeHC(args.library)
    cases = []

    def check(name, mixes, scale, base, native_expected=True, **kwargs):
        originals = [x.detach().clone() for x in (mixes, scale, base)]
        before = native.native_calls
        expected = cpu.hc_split_sinkhorn(mixes, scale, base, **kwargs)
        actual = native.hc_split_sinkhorn(mixes, scale, base, **kwargs)
        assert native.native_calls - before == int(native_expected), name
        exact = [raw_equal(a, b) for a, b in zip(expected, actual)]
        max_abs = [float((a - b).abs().nan_to_num().max()) for a, b in zip(expected, actual)]
        assert all(exact), (name, exact, max_abs)
        assert all(raw_equal(a, b) for a, b in zip((mixes, scale, base), originals)), name
        cases.append(dict(name=name, shape=list(mixes.shape), native=native_expected,
                          exact_pre_post_comb=exact, max_abs_pre_post_comb=max_abs,
                          iterations=kwargs.get('sinkhorn_iters', 20), eps=kwargs.get('eps', 1e-6)))

    for multiplier in [0.0, 0.01, 0.1, 1.0, 4.0, 16.0, 128.0]:
        for seed_index in range(160):
            check(f'random_{multiplier}_{seed_index}', torch.randn(1, 1, 24) * multiplier,
                  torch.randn(3), torch.randn(24) * multiplier)

    for iterations in [1, 2, 3, 19, 20, 21, 64, 256]:
        for eps in [0.0, 1e-12, 1e-6, 0.01, 1.0]:
            check(f'iterations_{iterations}_eps_{eps}', torch.randn(1, 1, 24),
                  torch.randn(3), torch.randn(24), sinkhorn_iters=iterations, eps=eps)

    for shape in [(24,), (1, 24), (1, 1, 24), (1, 1, 1, 24)]:
        check(f'prefix_{shape}', torch.randn(shape), torch.randn(3), torch.randn(24))
    for exponent in [-149, -126, -64, -32, 0, 16, 64, 120]:
        check(f'extreme_exponent_{exponent}', torch.randn(1, 1, 24) * 2.0 ** exponent,
              torch.tensor([1.0, -1.0, 1.0]), torch.zeros(24))
    check('parameter_inputs', torch.randn(1, 1, 24),
          torch.nn.Parameter(torch.randn(3), requires_grad=False),
          torch.nn.Parameter(torch.randn(24), requires_grad=False))
    check('storage_offsets', torch.randn(48)[24:].reshape(1, 1, 24),
          torch.randn(6)[3:], torch.randn(48)[24:])
    check('signed_zero', torch.tensor([0.0, -0.0] * 12).reshape(1, 1, 24),
          torch.tensor([-0.0, 0.0, -0.0]), torch.tensor([0.0, -0.0] * 12), eps=0.0)

    check('multi_token_fallback', torch.randn(1, 2, 24), torch.randn(3), torch.randn(24),
          native_expected=False)
    check('noncontiguous_fallback', torch.randn(48)[::2].reshape(1, 1, 24),
          torch.randn(3), torch.randn(24), native_expected=False)
    check('lazy_negative_fallback', torch._neg_view(torch.randn(1, 1, 24)),
          torch.randn(3), torch.randn(24), native_expected=False)
    check('different_hc_mult_fallback', torch.randn(1, 1, 8), torch.randn(3), torch.randn(8),
          native_expected=False, hc_mult=2)
    check('iterations257_fallback', torch.randn(1, 1, 24), torch.randn(3), torch.randn(24),
          native_expected=False, sinkhorn_iters=257)
    check('large_eps_fallback', torch.randn(1, 1, 24), torch.randn(3), torch.randn(24),
          native_expected=False, eps=2.0)
    check('autograd_fallback', torch.randn(1, 1, 24, requires_grad=True),
          torch.randn(3, requires_grad=True), torch.randn(24, requires_grad=True),
          native_expected=False)
    for bad in [float('nan'), float('inf'), -float('inf')]:
        x = torch.randn(1, 1, 24)
        x[0, 0, 23] = bad
        check(f'nonfinite_{bad}_fallback', x, torch.randn(3), torch.randn(24), native_expected=False)
    check('overflowed_transform_fallback', torch.full((1, 1, 24), 3e38),
          torch.full((3,), 3e38), torch.zeros(24), native_expected=False)
    torch.set_flush_denormal(True)
    try:
        check('flush_denormals_fallback', torch.randn(1, 1, 24), torch.randn(3),
              torch.randn(24), native_expected=False)
    finally:
        torch.set_flush_denormal(False)

    module = types.SimpleNamespace(hc_split_sinkhorn=cpu.hc_split_sinkhorn)
    saved = module.hc_split_sinkhorn
    assert native.install(module) is native and native.baseline is saved
    assert module.hc_split_sinkhorn == native.hc_split_sinkhorn
    assert native.install(module) is native and native.baseline is saved
    module.hc_split_sinkhorn = native.baseline
    assert module.hc_split_sinkhorn is saved and cpu.hc_split_sinkhorn is saved

    paths = [Path(__file__), BASE / 'goal_hc_0910.cpp', BASE / 'goal_hc_0910.py',
             args.library, BASE / 'deepseek_v41_cpu_reference_0910.py']
    result = dict(passed=True, component_only=True, full_checkpoint_loaded=False,
                  performance_trial=False, cpu_affinity=[0], torch_threads=1,
                  torch_version=torch.__version__, mkl_available=torch.backends.mkl.is_available(),
                  native_calls=native.native_calls, fallback_calls=native.fallback_calls,
                  max_abs_pre_post_comb=[max(c['max_abs_pre_post_comb'][i] for c in cases) for i in range(3)],
                  cases=cases, cases_count=len(cases), exact_float32_outputs=True,
                  parameter_inputs_supported=True, module_only_install_restore=True,
                  sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('cases', 'sha256')}))


if __name__ == '__main__':
    main()
