#!/usr/bin/env python3
"""Small synthetic MoE parity/lifetime fixture; no checkpoint or server work."""
import json
from pathlib import Path
import types

import torch

import deepseek_v41_cpu_reference_0910 as cpu
from deepseek_v41_native_bridge_0910 import NativeGemm
from goal_quant_reuse_0910 import install


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(4100925)
    base = Path(__file__).resolve().parent
    NativeGemm(base / 'results/deepseek-v41-native-gemm-0910b/libdeepseek-v41-native-gemm.so', 1).install()
    m = cpu.load_official_model('goal_quant_reuse_fixture')
    args = m.ModelArgs(**json.loads((cpu.OFFICIAL / 'config.json').read_text()))
    args.dim = args.moe_inter_dim = 64
    args.n_routed_experts, args.n_activated_experts = 6, 3
    args.vision_n_layers = 0
    moe = m.MoE(0, args).eval()
    with torch.no_grad():
        for name, p in moe.named_parameters():
            if p.dtype == torch.float8_e8m0fnu:
                p.view(torch.uint8).fill_(120)
            elif p.dtype == torch.float4_e2m1fn_x2:
                p.view(torch.uint8).random_(0, 256)
            elif p.dtype == torch.float8_e4m3fn:
                p.copy_((torch.randn(p.shape, dtype=torch.float32) * .7).to(p.dtype))
            else:
                p.copy_((torch.randn(p.shape, dtype=torch.float32) * .03).to(p.dtype))
    wrapped_calls = [0]
    for expert in moe.experts:
        original = expert.forward

        def bound(this, x, weights=None, _original=original):
            wrapped_calls[0] += 1
            return _original(x, weights)

        expert.forward = types.MethodType(bound, expert)
    inputs = [torch.randn(*shape, dtype=torch.float32).bfloat16()
              for shape in [(1, 1, 64), (1, 5, 64), (2, 1, 64)]]
    inputs.append(inputs[0].clone())
    references, baseline_calls = [], []
    with torch.inference_mode():
        for x in inputs:
            before = cpu.COUNTS['act_quant']
            references.append(moe(x).clone())
            baseline_calls.append(cpu.COUNTS['act_quant'] - before)
    wrappers_before = wrapped_calls[0]
    reuse = install(m)
    rows = []
    with torch.inference_mode():
        for x, expected, count in zip(inputs, references, baseline_calls):
            before = cpu.COUNTS['act_quant']
            actual = moe(x)
            assert torch.equal(actual, expected), (x.shape, (actual.float() - expected.float()).abs().max())
            candidate_calls = cpu.COUNTS['act_quant'] - before
            assert candidate_calls < count
            assert reuse.local.scope is None
            rows.append(dict(shape=list(x.shape), exact=True, baseline_quantizations=count,
                             candidate_quantizations=candidate_calls))
    assert wrapped_calls[0] == 2 * wrappers_before, 'Existing residency wrappers were bypassed'
    assert rows[0]['baseline_quantizations'] == 12 and rows[0]['candidate_quantizations'] == 5
    # Only explicitly admitted identities reuse quantization; writes to ordinary
    # tensors invalidate an entry even inside a scope, and an equal copy misses.
    with torch.no_grad():
        x = inputs[0].view(1, 64).clone()
        reuse.local.scope = {}
        reuse.admit(x)
        weight = moe.shared_experts.w1.weight
        first = m.linear(x, weight)
        assert torch.equal(first, reuse.original_linear(x, weight))
        hits = reuse.hits
        assert torch.equal(m.linear(x, weight), first) and reuse.hits == hits + 1
        misses = reuse.misses
        x.add_(1)
        assert torch.equal(m.linear(x, weight), reuse.original_linear(x, weight))
        assert reuse.misses == misses + 1
        hits = reuse.hits
        clone = x.clone()
        assert torch.equal(m.linear(clone, weight), reuse.original_linear(clone, weight))
        assert reuse.hits == hits
        reuse.local.scope.clear()
        reuse.local.scope = None
    # Exceptions must release activation buffers and leave the next call clean.
    old_shared = moe.shared_experts.forward

    def fail(*args, **kwargs):
        raise RuntimeError('fixture exception')

    moe.shared_experts.forward = fail
    with torch.inference_mode():
        try:
            moe(inputs[0])
        except RuntimeError as error:
            assert str(error) == 'fixture exception'
        else:
            raise AssertionError('Expected fixture exception')
    assert reuse.local.scope is None and reuse.calls == reuse.closed_scopes
    moe.shared_experts.forward = old_shared
    with torch.inference_mode():
        assert torch.equal(moe(inputs[0]), references[0])
        reuse.enabled = False
        before = cpu.COUNTS['act_quant']
        assert torch.equal(moe(inputs[0]), references[0])
        assert cpu.COUNTS['act_quant'] - before == baseline_calls[0]
        reuse.enabled = True
        assert torch.equal(moe(inputs[0]), references[0])
    reuse.uninstall()
    with torch.inference_mode():
        assert torch.equal(moe(inputs[0]), references[0])
    print(json.dumps(dict(passed=True, cases=rows, identity_and_mutation_checked=True,
                          residency_wrappers_preserved=True, exception_cleanup_checked=True,
                          uninstall_checked=True, full_checkpoint_loaded=False,
                          model_speed_measured=False), indent=2))


if __name__ == '__main__':
    main()
