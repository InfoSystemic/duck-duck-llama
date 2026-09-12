"""Opt-in fused single-token hc_mult=4 CPU Sinkhorn candidate.

``candidate = NativeHC(library); candidate.install(runtime.module)`` preserves
``candidate.baseline`` for toggling and uses it for all unsupported inputs.
Only one token is supported, since PyTorch's small scalar exp/reduction path
differs from its vector path for larger inputs. No model loading occurs here.
"""
import ctypes
import math
from pathlib import Path

import torch

import deepseek_v41_cpu_reference_0910 as cpu


class NativeHC:
    def __init__(self, library, baseline=None):
        self.library = ctypes.CDLL(str(Path(library).resolve()))
        self.fn = self.library.ds41_hc4_single
        self.fn.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int, ctypes.c_float] + [ctypes.c_void_p] * 3
        self.fn.restype = ctypes.c_int
        self.baseline = baseline if baseline is not None else cpu.hc_split_sinkhorn
        self._supported_runtime = (torch.__version__ == '2.10.0+cpu'
                                   and torch.backends.mkl.is_available()
                                   and torch.backends.cpu.get_cpu_capability() == 'AVX512')
        self.native_calls = 0
        self.fallback_calls = 0

    def hc_split_sinkhorn(self, mixes, hc_scale, hc_base, hc_mult=4,
                          sinkhorn_iters=20, eps=1e-6):
        supported = (
            all(type(x) in (torch.Tensor, torch.nn.Parameter) and x.device.type == 'cpu'
                and x.dtype == torch.float32 and x.is_contiguous()
                and not x.is_neg() and not x.requires_grad
                for x in (mixes, hc_scale, hc_base))
            and mixes.ndim > 0 and mixes.shape[-1] == 24 and mixes.numel() == 24
            and hc_scale.shape == (3,) and hc_base.shape == (24,)
            and hc_mult == 4 and type(sinkhorn_iters) is int
            and 1 <= sinkhorn_iters <= 256
            and type(eps) in (float, int) and math.isfinite(eps) and 0 <= eps <= 1
            and self._supported_runtime
        )
        if supported:
            pre = torch.empty((*mixes.shape[:-1], 4), dtype=torch.float32, device='cpu')
            post = torch.empty_like(pre)
            comb = torch.empty((*mixes.shape[:-1], 4, 4), dtype=torch.float32, device='cpu')
            status = self.fn(mixes.data_ptr(), hc_scale.data_ptr(), hc_base.data_ptr(),
                             sinkhorn_iters, eps, pre.data_ptr(), post.data_ptr(), comb.data_ptr())
            if status == 0:
                self.native_calls += 1
                cpu.COUNTS['hc_split_sinkhorn'] += 1
                return pre, post, comb
            if status not in (2, 3):
                raise RuntimeError(f'native HC rejected validated arguments: {status}')
        self.fallback_calls += 1
        return self.baseline(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps)

    def install(self, module):
        if module.hc_split_sinkhorn == self.hc_split_sinkhorn:
            return self
        self.baseline = module.hc_split_sinkhorn
        module.hc_split_sinkhorn = self.hc_split_sinkhorn
        return self
