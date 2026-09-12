"""Opt-in exact CPU quantization candidate; does not load models or compile code.

Use ``candidate = NativeQuant(library); candidate.install(runtime.module)``.
``candidate.baseline`` saves that module's original act_quant function and can
be assigned back to module.act_quant for matched measurements. Unsupported
paths call the saved function, including nonfinite input and autograd tensors.
"""
import ctypes
from pathlib import Path

import torch

import deepseek_v41_cpu_reference_0910 as cpu


class NativeQuant:
    def __init__(self, library, baseline=None):
        self.library = ctypes.CDLL(str(Path(library).resolve()))
        self.fn = self.library.ds41_act_quant32_bf16
        self.fn.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p,
                           ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        self.fn.restype = ctypes.c_int
        self.baseline = baseline if baseline is not None else cpu.act_quant
        self.native_calls = 0
        self.fallback_calls = 0

    def act_quant(self, x, block_size=128, scale_fmt=None,
                  scale_dtype=torch.float32, inplace=False):
        supported = (
            type(x) is torch.Tensor and x.device.type == 'cpu'
            and x.dtype == torch.bfloat16 and x.ndim > 0
            and x.shape[-1] > 0 and x.shape[-1] % 32 == 0
            and block_size == 32 and scale_fmt is not None
            and scale_dtype == torch.float8_e8m0fnu
            and x.is_contiguous() and not x.is_neg() and not x.requires_grad
            and x.numel() > 0 and isinstance(inplace, bool)
        )
        if supported:
            if inplace:
                status = self.fn(x.data_ptr(), x.numel() // 32, None, None,
                                 x.data_ptr(), 1)
                result = x
            else:
                quant = torch.empty_like(x, dtype=torch.float8_e4m3fn)
                scales = torch.empty((*x.shape[:-1], x.shape[-1] // 32),
                                     dtype=torch.float8_e8m0fnu, device='cpu')
                status = self.fn(x.data_ptr(), x.numel() // 32,
                                 quant.data_ptr(), scales.data_ptr(), None, 0)
                result = quant, scales
            if status == 0:
                self.native_calls += 1
                cpu.COUNTS['act_quant'] += 1
                return result
            if status not in (2, 3):
                raise RuntimeError(f'native act_quant rejected validated arguments: {status}')
        self.fallback_calls += 1
        return self.baseline(x, block_size, scale_fmt, scale_dtype, inplace)

    def install(self, module):
        """Patch only the passed official module and retain its original callable."""
        if module.act_quant == self.act_quant:
            return self
        self.baseline = module.act_quant
        module.act_quant = self.act_quant
        return self
