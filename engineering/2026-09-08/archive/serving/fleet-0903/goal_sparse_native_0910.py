"""Opt-in decode-only native sparse attention; unsupported cases use baseline."""
from collections import Counter
import ctypes
import math
from pathlib import Path

import torch

import deepseek_v41_cpu_reference_0910 as cpu


class NativeSparse:
    def __init__(self, library, baseline=None):
        self.library = ctypes.CDLL(str(Path(library).resolve()))
        self.fn = self.library.ds41_sparse_decode
        self.fn.argtypes = ([ctypes.c_void_p] * 4 + [ctypes.c_int] * 5
                            + [ctypes.c_int64] * 4 + [ctypes.c_float, ctypes.c_void_p])
        self.fn.restype = ctypes.c_int
        self.baseline = baseline if baseline is not None else cpu.sparse_attn
        self.enabled = True
        self.native_calls = 0
        self.fallback_calls = 0
        self.fallback_status = Counter()
        self._supported_runtime = (torch.__version__ == '2.10.0+cpu'
                                   and torch.backends.mkl.is_available())

    def sparse_attn(self, q, kv, attn_sink, topk_idxs, softmax_scale):
        supported = (
            self.enabled and self._supported_runtime
            and all(type(t) in (torch.Tensor, torch.nn.Parameter) and t.device.type == 'cpu'
                    and not t.requires_grad and not t.is_neg() and not t.is_conj()
                    for t in (q, kv, attn_sink, topk_idxs))
            and q.dtype == kv.dtype == torch.bfloat16 and attn_sink.dtype == torch.float32
            and topk_idxs.dtype in (torch.int32, torch.int64)
            and q.ndim == 4 and q.shape[:2] == (1, 1)
            and kv.ndim == 3 and kv.shape[0] == 1 and kv.shape[2] == q.shape[3]
            and 1 <= q.shape[2] <= 128 and 1 <= q.shape[3] <= 1024
            and 1 <= kv.shape[1] <= 2147483647 and attn_sink.shape == (q.shape[2],)
            and topk_idxs.ndim == 3 and topk_idxs.shape[:2] == (1, 1)
            and topk_idxs.shape[2] <= 2048 and q.is_contiguous()
            and kv.stride(-1) == 1 and kv.stride(-2) >= kv.shape[-1]
            and attn_sink.stride(0) > 0 and topk_idxs.stride(-1) > 0
            and type(softmax_scale) in (float, int) and math.isfinite(softmax_scale)
            and 0 < softmax_scale <= 3.4028234e38
        )
        if supported:
            output = torch.empty(q.shape, dtype=torch.bfloat16, device='cpu')
            status = self.fn(q.data_ptr(), kv.data_ptr(), attn_sink.data_ptr(), topk_idxs.data_ptr(),
                             topk_idxs.element_size(), q.shape[2], q.shape[3], kv.shape[1], topk_idxs.shape[2],
                             q.stride(-2), kv.stride(-2), attn_sink.stride(0), topk_idxs.stride(-1),
                             softmax_scale, output.data_ptr())
            if status == 0:
                self.native_calls += 1
                cpu.COUNTS['sparse_attn'] += 1
                return output
            self.fallback_status[status] += 1
        self.fallback_calls += 1
        return self.baseline(q, kv, attn_sink, topk_idxs, softmax_scale)

    def install(self, module):
        if module.sparse_attn == self.sparse_attn:
            return self
        self.baseline = module.sparse_attn
        module.sparse_attn = self.sparse_attn
        return self
