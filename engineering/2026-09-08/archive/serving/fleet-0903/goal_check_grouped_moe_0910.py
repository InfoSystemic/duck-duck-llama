#!/usr/bin/env python3
"""One-core grouped-MoE parity/residency fixture with synthetic weights only."""
import argparse
import json
from pathlib import Path
import types

import torch

import deepseek_v41_cpu_reference_0910 as cpu
from deepseek_v41_native_bridge_0910 import NativeGemm
from goal_grouped_moe_0910 import install


class FixtureStore:
    def __init__(self):
        self.resident_enabled = True
        self.residents = {}
        self.activation_order = []
        self.ensure_calls = []

    def activate(self, expert, prefix):
        self.activation_order.append(prefix)
        self.residents[prefix] = expert

    def ensure(self, names):
        self.ensure_calls.append(list(names))
        assert len(names) == len(set(names))
        assert all('.ffn.experts.' in name for name in names)


class ReferenceGrouped:
    """Plumbing oracle when the grouped native library is not yet available."""
    def __init__(self, native):
        self.native = native
        self.workers = 1
        self.task_counts = []

    def apply(self, tasks):
        self.task_counts.append(len(tasks))
        return torch.cat([self.native.apply(*task) for task in tasks], 0)


def fixture(backend_kind, full_width=False):
    base = Path(__file__).resolve().parent
    native = NativeGemm(base / 'results/deepseek-v41-native-gemm-0910b/libdeepseek-v41-native-gemm.so', 1)
    native.install()
    m = cpu.load_official_model('goal_grouped_moe_fixture')
    args = m.ModelArgs(**json.loads((cpu.OFFICIAL / 'config.json').read_text()))
    args.dim, args.moe_inter_dim = (5120, 2304) if full_width else (128, 96)
    args.n_routed_experts, args.n_activated_experts = 8, 6
    args.vision_n_layers = 0
    moe = m.MoE(0, args).eval()
    with torch.no_grad():
        for name, p in moe.named_parameters():
            if p.dtype == torch.float8_e8m0fnu:
                p.view(torch.uint8).random_(118, 123)
            elif p.dtype == torch.float4_e2m1fn_x2:
                p.view(torch.uint8).random_(0, 256)
            elif p.dtype == torch.float8_e4m3fn:
                p.copy_((torch.randn(p.shape, dtype=torch.float32) * .7).to(p.dtype))
            else:
                p.copy_((torch.randn(p.shape, dtype=torch.float32) * .03).to(p.dtype))
    store = FixtureStore()
    wrapper_calls = [0]
    for index, expert in enumerate(moe.experts):
        original = expert.forward
        prefix = f'layers.0.ffn.experts.{index}.'

        def resident(this, x, weights=None, _original=original, _prefix=prefix):
            wrapper_calls[0] += 1
            store.activate(this, _prefix)
            return _original(x, weights)

        expert.forward = types.MethodType(resident, expert)
    gate_calls = []
    moe.gate.register_forward_hook(lambda module, inputs, output: gate_calls.append(output[1].clone()))
    runtime = types.SimpleNamespace(module=m, model=types.SimpleNamespace(layers=[types.SimpleNamespace(layer_id=0, ffn=moe)]),
                                    store=store, native=native)
    inputs = [torch.randn(1, 1, args.dim).bfloat16() * factor for factor in (.1, 1., 10., 100.)]
    references = []
    with torch.inference_mode():
        for x in inputs:
            store.activation_order.clear()
            references.append((moe(x).clone(), list(store.activation_order)))
    backend = ReferenceGrouped(native) if backend_kind == 'reference' else None
    candidate = install(runtime, backend)
    rows = []
    with torch.inference_mode():
        for number, (x, (expected, activations)) in enumerate(zip(inputs, references)):
            store.activation_order.clear()
            if number == 0:
                store.residents.clear()
            before = (len(gate_calls), wrapper_calls[0], cpu.COUNTS['act_quant'])
            actual = moe(x)
            assert torch.equal(actual, expected), (number, int((actual != expected).sum()),
                float((actual.float() - expected.float()).abs().max()))
            assert store.activation_order == activations
            assert len(gate_calls) == before[0] + 1
            assert wrapper_calls[0] == before[1], 'Grouped path unexpectedly called sequential expert wrappers'
            assert cpu.COUNTS['act_quant'] == before[2] + 2
            if number == 0:
                assert len(store.ensure_calls) == 1 and len(store.ensure_calls[0]) == 36
            rows.append(dict(input_scale=number, exact=True, activation_order=activations))
        grouped_before = candidate.calls
        candidate.enabled = False
        assert torch.equal(moe(inputs[0]), references[0][0])
        candidate.enabled = True
        assert candidate.calls == grouped_before
        for x, mask in [(torch.cat(inputs[:2], dim=1), None),
                        (inputs[0], torch.zeros((1, 1), dtype=torch.bool))]:
            candidate.enabled = False
            expected = moe(x, mask)
            candidate.enabled = True
            assert torch.equal(moe(x, mask), expected)
            assert candidate.calls == grouped_before
        store.resident_enabled = False
        assert torch.equal(moe(inputs[0]), references[0][0])
        assert candidate.calls == grouped_before
        store.resident_enabled = True
    # With autograd enabled the original path is selected; no backward is run.
    candidate.enabled = False
    expected = moe(inputs[0])
    candidate.enabled = True
    assert torch.equal(moe(inputs[0]), expected)
    assert candidate.calls == grouped_before
    candidate.uninstall()
    with torch.inference_mode():
        assert torch.equal(moe(inputs[0]), references[0][0])
    if backend is not None:
        assert backend.task_counts == [14, 7] * len(inputs)
    return dict(passed=True, backend=backend_kind, dim=args.dim, inter_dim=args.moe_inter_dim,
                cases=rows, gate_hooks=True,
                direct_residency_matches_wrappers=True, cold_selected_group_protected=True,
                prefill_image_autograd_disabled_fallbacks=True, quantizations_per_decode=2,
                two_grouped_calls=True, uninstall=True, full_checkpoint_loaded=False, model_speed_measured=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['reference', 'native'], default='native')
    parser.add_argument('--full-width', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(4100927)
    print(json.dumps(fixture(args.backend, args.full_width), indent=2))


if __name__ == '__main__':
    main()
