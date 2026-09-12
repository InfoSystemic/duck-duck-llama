"""Experimental exact-input FP4 integer dot kernel with changed block reduction.

Packed weights remain nibble-packed, with int16 block sums and original scale
bytes in row-tile order. FP8 weight projections retain the baseline b kernel.
"""
import ctypes
import math
from pathlib import Path
import torch
import deepseek_v41_cpu_reference_0910 as cpu
from deepseek_v41_native_bridge_0910 import NativeGemm


class PackedFP4:
    def __init__(self, native, weight, scale):
        assert weight.device.type == scale.device.type == 'cpu'
        assert weight.is_contiguous() and scale.is_contiguous()
        assert weight.dtype == torch.float4_e2m1fn_x2 and scale.dtype == torch.float8_e8m0fnu
        assert weight.ndim == 2
        self.n, self.k = weight.shape[0], weight.shape[1] * 2
        assert self.k % 32 == 0 and scale.shape == (self.n, self.k // 32)
        tiles, blocks = (self.n + 15) // 16, self.k // 32
        self.packed = torch.empty((tiles, blocks, 8, 32), dtype=torch.uint8)
        self.sums = torch.empty((tiles, blocks, 16), dtype=torch.int16)
        self.scales = torch.empty((tiles, blocks, 16), dtype=torch.uint8)
        status = native.pack_fn(weight.data_ptr(), scale.data_ptr(), self.n, self.k, native.workers,
            self.packed.data_ptr(), self.sums.data_ptr(), self.scales.data_ptr())
        assert status == 0, ('FP4 pack rejected buffers', status)
        self.storage_bytes = self.packed.numel() + self.sums.numel() * 2 + self.scales.numel()


class NativeVnniGemm:
    def __init__(self, library, workers=16):
        self.library = ctypes.CDLL(str(Path(library).resolve()))
        self.baseline = NativeGemm(library, workers)
        self.workers = workers
        self.pack_fn = self.library.ds41_vnni_pack4
        self.pack_fn.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_int] * 3 + [ctypes.c_void_p] * 3
        self.pack_fn.restype = ctypes.c_int
        self.fn = self.library.ds41_vnni_gemm4
        self.fn.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
        self.fn.restype = ctypes.c_int
        self.cache = {}

    def pack(self, b, b_s):
        return PackedFP4(self, b, b_s)

    def clear(self):
        self.cache.clear()

    def apply_packed(self, a, a_s, packed):
        assert a.device.type == a_s.device.type == 'cpu' and a.is_contiguous() and a_s.is_contiguous()
        assert a.dtype == torch.float8_e4m3fn and a_s.dtype == torch.float8_e8m0fnu
        assert a.shape[-1] == packed.k and a_s.shape == (*a.shape[:-1], packed.k // 32)
        m = math.prod(a.shape[:-1])
        output = torch.empty((*a.shape[:-1], packed.n), dtype=torch.bfloat16)
        status = self.fn(a.data_ptr(), a_s.data_ptr(), packed.packed.data_ptr(), packed.sums.data_ptr(),
            packed.scales.data_ptr(), m, packed.n, packed.k, self.workers, output.data_ptr())
        assert status == 0, ('FP4 VNNI GEMM rejected shape', status, m, packed.n, packed.k)
        return output

    def apply(self, mode, a, a_s, b, b_s):
        assert mode in (4, 8)
        if mode == 8:
            self.baseline.workers = self.workers
            return self.baseline.apply(mode, a, a_s, b, b_s)
        key = (b.data_ptr(), b_s.data_ptr(), tuple(b.shape), tuple(b_s.shape))
        if key not in self.cache:
            # Retain source tensors to prevent storage-pointer reuse. This cache
            # assumes immutable checkpoint tensors and must be cleared on eviction.
            self.cache[key] = (b, b_s, self.pack(b, b_s))
        return self.apply_packed(a, a_s, self.cache[key][2])

    def fp4(self, a, a_s, b, b_s, scale_dtype=torch.float32, act_block_size=128):
        assert act_block_size == 32
        cpu.COUNTS['fp4_gemm'] += 1
        return self.apply(4, a, a_s, b, b_s)

    def install(self):
        cpu.fp4_gemm = self.fp4
