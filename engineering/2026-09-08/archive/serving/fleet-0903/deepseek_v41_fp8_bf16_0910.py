"""Bounded experimental lossless expansion of unchanged FP8 common weights.

The cache owns its source tensors, and can be cleared explicitly. It expands
only FP8 storage; native FP4 experts are never retained or copied here.
"""
import ctypes
import math
from pathlib import Path
import time

import torch


class ExpandedFP8:
    def __init__(self, library, workers=16, tile=1, cap_bytes=16 << 30):
        if type(cap_bytes) is not int or cap_bytes < 0:
            raise ValueError('cap_bytes must be a nonnegative integer')
        self.library = ctypes.CDLL(str(Path(library).resolve()))
        self.pack_fn = self.library.ds41_pack_fp8_bf16
        self.pack_fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int]
        self.pack_fn.restype = ctypes.c_int
        self.gemm_fn = self.library.ds41_gemm_fp8_bf16
        self.gemm_fn.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 5 + [ctypes.c_void_p]
        self.gemm_fn.restype = ctypes.c_int
        self.workers, self.tile, self.cap_bytes = workers, tile, cap_bytes
        self.cache = {}
        self.packed_bytes = self.pack_count = self.hits = self.misses = self.fallback_calls = 0
        self.pack_seconds = 0.0

    def clear(self):
        self.cache.clear()
        self.packed_bytes = 0

    def pack(self, weight):
        assert weight.device.type == 'cpu' and weight.dtype == torch.float8_e4m3fn
        assert weight.is_contiguous() and weight.ndim == 2
        assert 1 <= self.workers <= 64
        output = torch.empty_like(weight, dtype=torch.bfloat16)
        status = self.pack_fn(weight.data_ptr(), output.data_ptr(), weight.numel(), self.workers)
        assert status == 0, ('FP8 expansion failed', status)
        return output

    def get_packed(self, weight):
        key = (weight.data_ptr(), tuple(weight.shape))
        if key in self.cache:
            assert self.cache[key][0] is weight, 'Cached source pointer was reused'
            self.hits += 1
            return self.cache[key][1]
        self.misses += 1
        size = weight.numel() * 2
        if self.packed_bytes + size > self.cap_bytes:
            return None
        began = time.perf_counter()
        packed = self.pack(weight)
        self.pack_seconds += time.perf_counter() - began
        self.cache[key] = (weight, packed)
        self.packed_bytes += size
        self.pack_count += 1
        return packed

    def apply_packed(self, a, asc, packed, scale):
        assert all(t.device.type == 'cpu' and t.is_contiguous() for t in (a, asc, packed, scale))
        assert a.dtype == torch.float8_e4m3fn and packed.dtype == torch.bfloat16
        assert asc.dtype == scale.dtype == torch.float8_e8m0fnu
        n, k = packed.shape
        m = math.prod(a.shape[:-1])
        assert a.shape[-1] == k and 1 <= m <= 512 and 1 <= n <= 131072 and 32 <= k <= 32768 and k % 32 == 0
        assert asc.shape == (*a.shape[:-1], k // 32)
        assert scale.shape == ((n + 31) // 32, k // 32)
        assert 1 <= self.workers <= 64 and self.tile in (1, 2, 4)
        output = torch.empty(*a.shape[:-1], n, dtype=torch.bfloat16, device='cpu')
        status = self.gemm_fn(a.data_ptr(), asc.data_ptr(), packed.data_ptr(), scale.data_ptr(),
                             m, n, k, self.workers, self.tile, output.data_ptr())
        assert status == 0, ('Expanded FP8 GEMM failed', status)
        return output

    def install(self, runtime):
        self.runtime, self.original = runtime, runtime.module.fp8_gemm
        self.enabled = True

        def fp8(a, asc, b, bsc, scale_dtype=torch.float32, block_size=128):
            if not self.enabled or block_size != 32:
                self.fallback_calls += 1
                return self.original(a, asc, b, bsc, scale_dtype, block_size=block_size)
            self.workers = runtime.native.workers
            packed = self.get_packed(b)
            if packed is None:
                self.fallback_calls += 1
                return self.original(a, asc, b, bsc, scale_dtype, block_size=block_size)
            return self.apply_packed(a, asc, packed, bsc)

        runtime.module.fp8_gemm = fp8
        return self

    def uninstall(self):
        self.runtime.module.fp8_gemm = self.original
        self.clear()
