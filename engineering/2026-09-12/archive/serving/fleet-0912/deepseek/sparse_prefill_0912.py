"""Opt-in native sparse attention for V4.1 prefill, using the validated row kernel.

No checkpoint, service, or global module is modified on import. Call install on a
candidate runtime's model module; the existing NativeSparse fallback is retained.
"""
from pathlib import Path
import math
import sys

import torch

FLEET_0903 = Path(__file__).resolve().parents[2] / 'fleet-0903'
if str(FLEET_0903) not in sys.path:
    sys.path.insert(0, str(FLEET_0903))

from goal_sparse_native_0910 import NativeSparse
import deepseek_v41_cpu_reference_0910 as cpu


class NativeSparsePrefill(NativeSparse):
    def __init__(self, library=None, baseline=None):
        if library is None:
            library = FLEET_0903 / 'results/goal_sparse_native_0910/libgoal_sparse_native_0910.so'
        super().__init__(library, baseline)
        self.prefill_calls = 0
        self.prefill_rows = 0

    def sparse_attn(self, q, kv, attn_sink, topk_idxs, softmax_scale):
        tensors = (q, kv, attn_sink, topk_idxs)
        prefill = (
            self.enabled and self._supported_runtime
            and all(type(t) in (torch.Tensor, torch.nn.Parameter) and t.device.type == 'cpu'
                    and not t.requires_grad and not t.is_neg() and not t.is_conj()
                    for t in tensors)
            and q.dtype == kv.dtype == torch.bfloat16 and attn_sink.dtype == torch.float32
            and topk_idxs.dtype in (torch.int32, torch.int64)
            and q.ndim == 4 and q.shape[0] == 1 and q.shape[1] > 1
            and q.is_contiguous() and 1 <= q.shape[2] <= 128 and 1 <= q.shape[3] <= 1024
            and kv.ndim == 3 and kv.shape[0] == 1 and kv.shape[-1] == q.shape[-1]
            and 1 <= kv.shape[1] <= 2147483647
            and kv.stride(-1) == 1 and kv.stride(-2) >= kv.shape[-1]
            and attn_sink.shape == (q.shape[2],)
            and attn_sink.stride(0) > 0
            and topk_idxs.ndim == 3 and topk_idxs.shape[:2] == q.shape[:2]
            and topk_idxs.shape[-1] <= 2048 and topk_idxs.stride(-1) > 0
            and type(softmax_scale) in (float, int) and math.isfinite(softmax_scale)
            and 0 < softmax_scale <= 3.4028234e38
        )
        if not prefill:
            return super().sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale)

        # The baseline also loops over query rows, keeping each row's 64-key
        # softmax chunks and BF16 probability rounding independent of other rows.
        output = torch.empty_like(q)
        q_pointer, kv_pointer = q.data_ptr(), kv.data_ptr()
        sink_pointer, ids_pointer = attn_sink.data_ptr(), topk_idxs.data_ptr()
        output_pointer = output.data_ptr()
        q_step = q.stride(1) * q.element_size()
        ids_step = topk_idxs.stride(1) * topk_idxs.element_size()
        output_step = output.stride(1) * output.element_size()
        for token in range(q.shape[1]):
            status = self.fn(q_pointer + token * q_step, kv_pointer, sink_pointer,
                             ids_pointer + token * ids_step, topk_idxs.element_size(),
                             q.shape[2], q.shape[3], kv.shape[1], topk_idxs.shape[-1],
                             q.stride(-2), kv.stride(-2), attn_sink.stride(0),
                             topk_idxs.stride(-1), softmax_scale,
                             output_pointer + token * output_step)
            if status == 0:
                self.native_calls += 1
                cpu.COUNTS['sparse_attn'] += 1
            else:
                self.fallback_status[status] += 1
                self.fallback_calls += 1
                output[:, token:token + 1] = self.baseline(
                    q[:, token:token + 1], kv, attn_sink,
                    topk_idxs[:, token:token + 1], softmax_scale)
        self.prefill_calls += 1
        self.prefill_rows += q.shape[1]
        return output
