#!/usr/bin/env python3
"""Byte-exact fixtures only: no checkpoint/model loading or performance trials."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import types

import torch

import deepseek_v41_cpu_reference_0910 as cpu
from goal_native_quant_0910 import NativeQuant

BASE = Path(__file__).resolve().parent
KWARGS = dict(block_size=32, scale_fmt='ue8m0', scale_dtype=torch.float8_e8m0fnu)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def raw_equal(left, right):
    return (left.shape == right.shape and left.dtype == right.dtype
            and torch.equal(left.contiguous().view(torch.uint8),
                            right.contiguous().view(torch.uint8)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--library', type=Path,
                        default=BASE / 'results/goal_native_quant_0910/libgoal_native_quant_0910.so')
    parser.add_argument('--output', type=Path,
                        default=BASE / 'results/goal_native_quant_0910/fixture-check.json')
    args = parser.parse_args()
    assert os.sched_getaffinity(0) <= {0, 1, 2, 3}, 'run with taskset -c 0-3'
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(410910)
    torch.set_flush_denormal(False)
    native = NativeQuant(args.library)
    cases = []

    def check(name, x, kwargs=None, expect_native=True):
        params = dict(KWARGS if kwargs is None else kwargs)
        original = x.clone()
        before = native.native_calls
        expected = cpu.act_quant(x, **params)
        actual = native.act_quant(x, **params)
        assert all(raw_equal(a, b) for a, b in zip(expected, actual)), name
        assert raw_equal(x, original), (name, 'out-of-place mutated input')
        left, right = x.clone(), x.clone()
        expected_inplace = cpu.act_quant(left, **params, inplace=True)
        actual_inplace = native.act_quant(right, **params, inplace=True)
        assert expected_inplace is left and actual_inplace is right, name
        assert raw_equal(left, right), (name, 'inplace bytes')
        assert native.native_calls - before == (2 if expect_native else 0), name
        assert actual[0].is_contiguous() and actual[1].is_contiguous(), name
        cases.append(dict(name=name, shape=list(x.shape), input_values=x.numel(),
                          native=expect_native, quant_bytes_exact=True,
                          scale_bytes_exact=True, inplace_bf16_bytes_exact=True))

    # All finite BF16 bit patterns as maxima, with varying signs and zeros.
    # This also exercises every input exponent, BF16 subnormals, and saturation.
    all_bits = torch.arange(65536, dtype=torch.int32)
    finite_bits = all_bits[(all_bits & 0x7fff) < 0x7f80].to(torch.uint16)
    exhaustive = finite_bits.view(torch.bfloat16)[:, None].repeat(1, 32)
    exhaustive[:, 1::4] *= -1
    exhaustive[:, 2::4] = 0
    exhaustive[:, 3::4] = -0.0
    check('all_finite_bf16_maxima', exhaustive)
    del exhaustive

    # Every FP8 midpoint plus both adjacent BF16 values. An independent 448*s
    # anchor fixes the scale, exposing ties-to-even and the FP8 subnormal edge.
    fp8_values = torch.arange(127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    midpoint = (fp8_values[:-1] + fp8_values[1:]) / 2
    boundaries = torch.cat([midpoint, fp8_values, torch.tensor([0.0, 448.0])])
    for exponent in [-22, -16, -8, 0, 8, 32, 100, 119]:
        scale = 2.0 ** exponent
        center = (boundaries * scale).bfloat16().view(torch.uint16).to(torch.int32)
        adjacent = torch.stack([(center - 1).clamp_min(0), center, center + 1], dim=-1)
        adjacent = adjacent.flatten().to(torch.uint16).view(torch.bfloat16)
        values = torch.cat([adjacent, -adjacent])
        x = torch.zeros(values.numel(), 32, dtype=torch.bfloat16)
        x[:, 0] = 448.0 * scale
        x[:, 1] = values
        x[:, 2] = -0.0
        check(f'fp8_midpoints_neighbors_scale_2^{exponent}', x)

    # Finite BF16 samples from the entire bit space, both coherent and highly
    # mixed exponents, and actual dimensions used by the official graph.
    for shape in [(32,), (1, 1, 5120), (1, 1, 2304), (1, 6, 5120),
                  (4, 2, 1280), (3, 7, 64), (1, 8192), (2, 32768)]:
        check(f'normal_random_{shape}', torch.randn(shape).bfloat16())
        indices = torch.randint(0, finite_bits.numel(), shape)
        check(f'finite_bits_random_{shape}', finite_bits.to(torch.int32)[indices]
              .to(torch.uint16).view(torch.bfloat16))

    # Amax ceiling transitions: exactly 448*2^e and both BF16 neighbours.
    maxima = torch.tensor([448.0 * 2.0 ** e for e in range(-126, 120)]).bfloat16()
    codes = maxima.view(torch.uint16).to(torch.int32)
    near = torch.stack([(codes - 1).clamp_min(0), codes, codes + 1], -1)
    near = near.flatten().to(torch.uint16).view(torch.bfloat16)
    x = torch.zeros(near.numel(), 32, dtype=torch.bfloat16)
    x[:, 0] = near
    x[:, 1:] = near[:, None] / 16
    check('scale_ceiling_neighbors_and_floor', x)
    check('storage_offset', torch.randn(5216).bfloat16()[32:5152])

    # Explicit fallback cases preserve the reference behavior and callable.
    check('float32_fallback', torch.randn(2, 64), expect_native=False)
    check('noncontiguous_fallback', torch.randn(64, 2).bfloat16().T,
          expect_native=False)
    check('block128_fallback', torch.randn(2, 128).bfloat16(),
          dict(KWARGS, block_size=128), expect_native=False)
    check('float32_scale_fallback', torch.randn(2, 64).bfloat16(),
          dict(KWARGS, scale_dtype=torch.float32), expect_native=False)
    check('unrounded_scale_fallback', torch.randn(2, 64).bfloat16(),
          dict(KWARGS, scale_fmt=None), expect_native=False)
    check('empty_batch_fallback', torch.empty(0, 64, dtype=torch.bfloat16),
          expect_native=False)
    negative_view = torch._neg_view(torch.randn(2, 64).bfloat16())
    # clone() resolves its lazy sign; only out-of-place is checked here.
    expected = cpu.act_quant(negative_view, **KWARGS)
    before = native.native_calls
    actual = native.act_quant(negative_view, **KWARGS)
    assert all(raw_equal(a, b) for a, b in zip(expected, actual))
    assert before == native.native_calls

    # Nonfinite rejection must precede all writes, including when a later block
    # is invalid. The original reference raises for nonfinite amax scales.
    for bad in [float('nan'), float('inf'), -float('inf')]:
        for inplace in [False, True]:
            x = torch.randn(3, 32).bfloat16()
            x[-1, -1] = bad
            saved = x.view(torch.uint8).clone()
            for func in [cpu.act_quant, native.act_quant]:
                try:
                    func(x, **KWARGS, inplace=inplace)
                except AssertionError:
                    pass
                else:
                    raise AssertionError(('nonfinite did not reject', bad, inplace))
                assert torch.equal(x.view(torch.uint8), saved)

    # Autograd semantics must not be bypassed by writing a leaf through ctypes.
    leaf = torch.randn(1, 32).bfloat16().requires_grad_(True)
    before = native.native_calls
    expected = cpu.act_quant(leaf, **KWARGS)
    actual = native.act_quant(leaf, **KWARGS)
    assert all(raw_equal(a, b) for a, b in zip(expected, actual))
    for func in [cpu.act_quant, native.act_quant]:
        try:
            func(leaf, **KWARGS, inplace=True)
        except RuntimeError as error:
            assert 'leaf Variable' in str(error)
        else:
            raise AssertionError('autograd mutation was accepted')
    assert native.native_calls == before

    torch.set_flush_denormal(True)
    try:
        check('denormal_flush_fallback', torch.randn(1, 64).bfloat16(),
              expect_native=False)
    finally:
        torch.set_flush_denormal(False)

    fake_module = types.SimpleNamespace(act_quant=cpu.act_quant)
    saved = fake_module.act_quant
    assert native.install(fake_module) is native and native.baseline is saved
    assert fake_module.act_quant == native.act_quant
    assert native.install(fake_module) is native and native.baseline is saved
    fake_module.act_quant = native.baseline
    assert fake_module.act_quant is saved and cpu.act_quant is saved

    result = dict(passed=True, component_only=True, full_checkpoint_loaded=False,
                  performance_trial=False, torch_version=torch.__version__,
                  cpu_affinity=sorted(os.sched_getaffinity(0)), torch_threads=4,
                  cases=cases, fixture_input_values=sum(c['input_values'] for c in cases),
                  native_calls=native.native_calls, fallback_calls=native.fallback_calls,
                  negative_view_fallback=True, nonfinite_no_mutation=True,
                  autograd_fallback=True, module_only_install_restore=True,
                  sha256={str(p): sha(p) for p in [Path(__file__), args.library,
                          BASE / 'goal_native_quant_0910.py', BASE / 'goal_native_quant_0910.cpp',
                          BASE / 'deepseek_v41_cpu_reference_0910.py']})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: val for key, val in result.items() if key not in ['cases', 'sha256']}))


if __name__ == '__main__':
    main()
