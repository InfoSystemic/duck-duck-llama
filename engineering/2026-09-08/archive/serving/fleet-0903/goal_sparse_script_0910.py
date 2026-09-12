"""TorchScript execution of the existing CPU sparse-attention equations.

Keeps the 64-position online-softmax chunks and the BF16 probability rounding.
This changes dispatch only; no attention positions or arithmetic are removed.
"""
import torch

import deepseek_v41_cpu_reference_0910 as cpu


@torch.jit.script
def scripted_sparse_attn(q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor,
                        topk_idxs: torch.Tensor, softmax_scale: float) -> torch.Tensor:
    batch, seq, heads, dim = q.shape
    output = torch.empty_like(q)
    for b in range(batch):
        for token in range(seq):
            maximum = torch.full((heads,), -1e30, dtype=torch.float32, device='cpu')
            denominator = torch.zeros((heads,), dtype=torch.float32, device='cpu')
            numerator = torch.zeros((heads, dim), dtype=torch.float32, device='cpu')
            ids = topk_idxs[b, token]
            for begin in range(0, ids.numel(), 64):
                chunk = ids[begin:begin + 64].long()
                valid = chunk >= 0
                values = kv[b, chunk.clamp_min(0)].float().masked_fill(~valid[:, None], 0)
                scores = (q[b, token].float() @ values.T).masked_fill(~valid[None], -torch.inf) * softmax_scale
                next_max = torch.maximum(maximum, scores.amax(-1))
                correction = torch.exp(maximum - next_max)
                probs = torch.exp(scores - next_max[:, None])
                denominator = denominator * correction + probs.sum(-1)
                numerator = numerator * correction[:, None] + probs.to(torch.bfloat16).float() @ values
                maximum = next_max
            denominator += torch.exp(attn_sink - maximum)
            output[b, token] = (numerator / denominator[:, None]).to(q.dtype)
    return output


class SparseScript:
    def __init__(self, module):
        self.module = module
        self.original = module.sparse_attn
        self.enabled = True

    def __call__(self, q, kv, attn_sink, topk_idxs, softmax_scale):
        if not self.enabled:
            return self.original(q, kv, attn_sink, topk_idxs, softmax_scale)
        cpu._cpu(q, kv, attn_sink, topk_idxs)
        assert q.dtype == kv.dtype == torch.bfloat16
        batch, seq, heads, dim = q.shape
        assert kv.shape[0] == batch and kv.shape[-1] == dim and attn_sink.shape == (heads,)
        assert topk_idxs.shape[:2] == (batch, seq) and ((topk_idxs >= -1) & (topk_idxs < kv.shape[1])).all()
        cpu.COUNTS['sparse_attn'] += 1
        return scripted_sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale)

    def uninstall(self):
        self.module.sparse_attn = self.original
        del self.module._goal_sparse_script_0910


def install(module):
    if hasattr(module, '_goal_sparse_script_0910'):
        raise RuntimeError('Scripted sparse attention is already installed')
    candidate = SparseScript(module)
    module.sparse_attn = candidate
    module._goal_sparse_script_0910 = candidate
    return candidate
