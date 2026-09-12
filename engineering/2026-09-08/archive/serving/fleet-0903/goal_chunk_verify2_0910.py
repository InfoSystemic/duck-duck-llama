"""Causal chunk verifier v2: retain the selected single-token HC arithmetic.

The base verifier is unchanged. During verification only, batched HC mixes are
split into contiguous one-token calls, avoiding batch-dependent sigmoid rounding
and keeping the selected NativeHC path. Stacking adds no arithmetic.
"""
import torch

from goal_chunk_verify_0910 import ChunkVerifier as BaseChunkVerifier


class ChunkVerifier(BaseChunkVerifier):
    def __init__(self, model, module, max_chunk=8, strict_linear=True):
        self.original_hc_split_sinkhorn = module.hc_split_sinkhorn
        self.hc_batch_calls = self.hc_token_calls = 0
        super().__init__(model, module, max_chunk, strict_linear)

    def _hc_split_sinkhorn(self, mixes, *args, **kwargs):
        if mixes.numel() == 24:
            return self.original_hc_split_sinkhorn(mixes, *args, **kwargs)
        if mixes.ndim < 2 or mixes.shape[-1] != 24:
            raise ValueError('Chunk HC requires a final dimension of 24')
        flat = mixes.reshape(-1, 24)
        rows = [self.original_hc_split_sinkhorn(flat[i:i+1].contiguous().reshape(1, 1, 24),
                                               *args, **kwargs)
                for i in range(flat.shape[0])]
        self.hc_batch_calls += 1
        self.hc_token_calls += len(rows)
        leading = mixes.shape[:-1]
        pre = torch.cat([row[0] for row in rows], dim=1).reshape(*leading, 4)
        post = torch.cat([row[1] for row in rows], dim=1).reshape(*leading, 4)
        comb = torch.cat([row[2] for row in rows], dim=1).reshape(*leading, 4, 4)
        return pre, post, comb

    def verify(self, input_ids, start_pos, *, rollback=False):
        if self.module.hc_split_sinkhorn is not self.original_hc_split_sinkhorn:
            raise RuntimeError('hc_split_sinkhorn changed after verifier construction')
        self.module.hc_split_sinkhorn = self._hc_split_sinkhorn
        try:
            return super().verify(input_ids, start_pos, rollback=rollback)
        finally:
            self.module.hc_split_sinkhorn = self.original_hc_split_sinkhorn
