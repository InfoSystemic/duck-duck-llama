"""Reuse unchanged expert-input FP8 quantization within one official MoE call.

Install after Runtime construction with ``install(runtime.module)``. Existing
expert instance wrappers still perform residency/eviction work. The only graph
change is to share the one-row decode input instead of copying it per expert.
Projection arithmetic, selected-expert order, and output additions are unchanged.
"""
import threading

import torch


class QuantReuse:
    def __init__(self, module):
        self.module = module
        self.original_linear = module.linear
        self.original_moe = module.MoE.forward
        self.local = threading.local()
        self.hits = 0
        self.misses = 0
        self.calls = 0
        self.closed_scopes = 0
        self.enabled = True

    @staticmethod
    def version(x):
        # Inference tensors intentionally have no version counter. They are
        # admitted only as the exact input object to the unchanged Expert code,
        # which does not mutate x. Ordinary tensors additionally track writes.
        return None if x.is_inference() else x._version

    def admit(self, x):
        scope = self.local.scope
        key = id(x)
        if key not in scope:
            scope[key] = [x, self.version(x), None]

    def linear(self, x, weight, bias=None):
        scope = getattr(self.local, 'scope', None)
        entry = None if scope is None else scope.get(id(x))
        m = self.module
        if (entry is None or entry[0] is not x or bias is not None or
                weight.dtype not in (torch.float4_e2m1fn_x2, torch.float8_e4m3fn)):
            return self.original_linear(x, weight, bias)
        version = self.version(x)
        if entry[2] is None or entry[1] != version:
            entry[1] = version
            entry[2] = m.act_quant(x, m.fp8_block_size, m.scale_fmt, m.scale_dtype)
            self.misses += 1
        else:
            self.hits += 1
        quant, scales = entry[2]
        if weight.dtype == torch.float4_e2m1fn_x2:
            return m.fp4_gemm(quant, scales, weight, weight.scale, m.scale_dtype,
                              act_block_size=m.fp8_block_size)
        return m.fp8_gemm(quant, scales, weight, weight.scale, m.scale_dtype,
                          block_size=m.fp8_block_size)

    def forward(self, moe, x, image_mask=None):
        if not self.enabled or torch.is_grad_enabled():
            return self.original_moe(moe, x, image_mask)
        previous = getattr(self.local, 'scope', None)
        scope = {}
        self.local.scope = scope
        self.calls += 1
        try:
            shape = x.size()
            x = x.view(-1, moe.dim)
            weights, indices = moe.gate(x, None if image_mask is None else image_mask.flatten())
            y = torch.zeros_like(x, dtype=torch.float32)
            counts = torch.bincount(indices.flatten(), minlength=moe.n_routed_experts).tolist()
            for i in range(moe.experts_start_idx, moe.experts_end_idx):
                if counts[i] == 0:
                    continue
                idx, top = torch.where(indices == i)
                # A top-k result contains each selected expert once. For one
                # token, official x[idx] is exactly a copy of the sole input row.
                expert_input = x if x.shape[0] == 1 else x[idx]
                self.admit(expert_input)
                y[idx] += moe.experts[i](expert_input, weights[idx, top, None])
            if self.module.world_size > 1:
                self.module.dist.all_reduce(y)
            self.admit(x)
            y += moe.shared_experts(x)
            return y.type_as(x).view(shape)
        finally:
            scope.clear()
            self.local.scope = previous
            self.closed_scopes += 1

    def uninstall(self):
        self.module.linear = self.original_linear
        self.module.MoE.forward = self.original_moe
        del self.module._goal_quant_reuse_0910


def install(module):
    if hasattr(module, '_goal_quant_reuse_0910'):
        raise RuntimeError('Expert quantization reuse is already installed')
    reuse = QuantReuse(module)
    module.linear = reuse.linear

    def forward(moe, x, image_mask=None):
        return reuse.forward(moe, x, image_mask)

    module.MoE.forward = forward
    module._goal_quant_reuse_0910 = reuse
    return reuse
