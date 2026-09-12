"""Decode-only grouped native MoE, with the original forward as a fallback.

The gate and resident-store activation path remain active. Sorted routed experts
plus the shared expert use one grouped gate/up call and one grouped down call.
No checkpoint weights are copied, requantized, or replaced by this module.
"""
from pathlib import Path
import types

import torch
import torch.nn.functional as F


class GroupedMoE:
    def __init__(self, runtime, backend):
        self.runtime = runtime
        self.module = runtime.module
        self.backend = backend
        self.enabled = True
        self.calls = 0
        self.fallback_calls = 0
        self.bindings = []
        for layer in runtime.model.layers:
            moe = layer.ffn
            original = moe.forward
            supported = self.compatible(moe)
            prefix = f'layers.{layer.layer_id}.ffn.experts.'

            def forward(this, x, image_mask=None, _original=original,
                        _supported=supported, _prefix=prefix):
                return self.forward(this, x, image_mask, _original, _supported, _prefix)

            self.bindings.append((moe, original))
            moe.forward = types.MethodType(forward, moe)

    def compatible(self, moe):
        m = self.module
        if (m.world_size != 1 or m.fp8_block_size != 32 or m.scale_fmt is None or
                m.scale_dtype != torch.float8_e8m0fnu or
                not 1 <= moe.n_activated_experts <= 6 or
                moe.experts_start_idx != 0 or moe.experts_end_idx != moe.n_routed_experts or
                not hasattr(self.runtime.store, 'activate') or
                not hasattr(self.runtime.store, 'residents')):
            return False
        shared = moe.shared_experts
        inter = shared.w1.out_features
        if moe.dim % 32 or inter % 32:
            return False
        for expert, mode in [(e, 4) for e in moe.experts] + [(shared, 8)]:
            if expert is None or expert.swiglu_limit != shared.swiglu_limit:
                return False
            dtype = torch.float4_e2m1fn_x2 if mode == 4 else torch.float8_e4m3fn
            for linear, rows, cols in [(expert.w1, inter, moe.dim),
                                       (expert.w3, inter, moe.dim),
                                       (expert.w2, moe.dim, inter)]:
                if (linear.weight.dtype != dtype or linear.bias is not None or
                        linear.weight.shape != (rows, cols // 2 if mode == 4 else cols) or
                        linear.scale.dtype != torch.float8_e8m0fnu or
                        linear.scale.shape != (rows if mode == 4 else (rows + 31) // 32, cols // 32)):
                    return False
        return True

    def forward(self, moe, x, image_mask, original, supported, prefix):
        store = self.runtime.store
        if (not self.enabled or not supported or image_mask is not None or
                torch.is_grad_enabled() or x.requires_grad or x.device.type != 'cpu' or
                x.dtype != torch.bfloat16 or x.numel() != moe.dim or x.ndim < 2 or
                not x.is_contiguous() or not getattr(store, 'resident_enabled', False)):
            self.fallback_calls += 1
            return original(x, image_mask)
        shape = x.shape
        x = x.view(1, moe.dim)
        # Invoke the module normally so routing hooks observe the real decision.
        weights, indices = moe.gate(x)
        selected = sorted((int(index), top) for top, index in enumerate(indices[0].tolist()))
        assert len(selected) == moe.n_activated_experts
        assert len({index for index, _ in selected}) == len(selected)
        routed = [(moe.experts[index], prefix + str(index) + '.', top) for index, top in selected]
        if any(expert_prefix not in store.residents for _, expert_prefix, _ in routed):
            # A grouped call needs all selected mappings concurrently. Protect
            # their complete parameter set during any cold-path cache eviction.
            names = [expert_prefix + name for expert, expert_prefix, _ in routed
                     for name, _ in expert.named_parameters()]
            store.ensure(names)
        for expert, expert_prefix, _ in routed:
            store.activate(expert, expert_prefix)
        experts = [(expert, 4) for expert, _, _ in routed] + [(moe.shared_experts, 8)]
        m = self.module
        quant, scales = m.act_quant(x, m.fp8_block_size, m.scale_fmt, m.scale_dtype)
        up_tasks = [(mode, quant, scales, projection.weight, projection.weight.scale)
                    for expert, mode in experts for projection in (expert.w1, expert.w3)]
        self.backend.workers = self.runtime.native.workers
        inter = moe.shared_experts.w1.out_features
        projected = self.backend.apply(up_tasks).view(len(experts), 2, inter)
        gate, up = projected[:, 0].float(), projected[:, 1].float()
        limit = moe.shared_experts.swiglu_limit
        if limit > 0:
            up = torch.clamp(up, min=-limit, max=limit)
            gate = torch.clamp(gate, max=limit)
        activated = F.silu(gate) * up
        # The published graph scales routed intermediate activations before BF16
        # rounding and the down projection. The shared expert is not scaled.
        for group, (_, _, top) in enumerate(routed):
            activated[group] = weights[0, top] * activated[group]
        intermediate = activated.to(x.dtype)
        down_quant, down_scales = m.act_quant(intermediate, m.fp8_block_size, m.scale_fmt, m.scale_dtype)
        down_tasks = [(mode, down_quant[group:group + 1], down_scales[group:group + 1],
                       expert.w2.weight, expert.w2.weight.scale)
                      for group, (expert, mode) in enumerate(experts)]
        outputs = self.backend.apply(down_tasks)
        y = torch.zeros_like(x, dtype=torch.float32)
        for group in range(len(experts)):
            y += outputs[group:group + 1]
        self.calls += 1
        return y.type_as(x).view(shape)

    def uninstall(self):
        for moe, original in self.bindings:
            moe.forward = original
        self.bindings.clear()
        del self.runtime._goal_grouped_moe_0910


def install(runtime, backend=None):
    if hasattr(runtime, '_goal_grouped_moe_0910'):
        raise RuntimeError('Grouped MoE is already installed')
    if backend is None:
        from deepseek_v41_native_grouped_goal_0910 import GroupedNativeGemm
        library = (Path(__file__).resolve().parent / 'results/deepseek-v41-native-grouped-goal-0910'
                   / 'libdeepseek-v41-native-grouped.so')
        backend = GroupedNativeGemm(library, runtime.native.workers)
    candidate = GroupedMoE(runtime, backend)
    runtime._goal_grouped_moe_0910 = candidate
    return candidate
