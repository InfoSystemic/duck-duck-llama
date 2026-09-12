"""Experimental exact-input FP8 integer dots; FP32 block reduction differs from baseline."""
import ctypes
import math
from pathlib import Path
import torch
import deepseek_v41_cpu_reference_0910 as cpu
from deepseek_v41_native_bridge_0910 import NativeGemm


class PackedFP8:
    def __init__(self, native, weight, scale):
        assert weight.device.type == scale.device.type == 'cpu'
        assert weight.is_contiguous() and scale.is_contiguous()
        assert weight.dtype == torch.float8_e4m3fn and scale.dtype == torch.float8_e8m0fnu
        assert weight.ndim == 2
        self.n, self.k = weight.shape
        assert self.k % 32 == 0 and scale.shape == ((self.n + 31) // 32, self.k // 32)
        tiles, blocks = (self.n + 15) // 16, self.k // 32
        self.packed = torch.empty((tiles, blocks, 16, 32), dtype=torch.uint8, device='cpu')
        self.nan_masks = torch.empty((tiles, blocks), dtype=torch.uint16, device='cpu')
        self.scale = scale
        status = native.pack_fn(weight.data_ptr(), self.n, self.k, native.workers,
            self.packed.data_ptr(), self.nan_masks.data_ptr())
        assert status == 0, ('FP8 pack rejected buffers', status)
        self.storage_bytes = self.packed.numel() + self.nan_masks.numel() * 2


class NativeFP8VnniGemm:
    def __init__(self, library, workers=16):
        self.library = ctypes.CDLL(str(Path(library).resolve()))
        self.baseline = NativeGemm(library, workers)
        self.workers = workers
        self.pack_fn = self.library.ds41_fp8_vnni_pack
        self.pack_fn.argtypes = [ctypes.c_void_p] + [ctypes.c_int] * 3 + [ctypes.c_void_p] * 2
        self.pack_fn.restype = ctypes.c_int
        self.fn = self.library.ds41_fp8_vnni_gemm
        self.fn.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
        self.fn.restype = ctypes.c_int
        self.cache = {}

    def pack(self, b, bs):
        return PackedFP8(self, b, bs)

    def clear(self):
        self.cache.clear()

    def apply_packed(self, a, asc, packed):
        assert a.device.type == asc.device.type == 'cpu' and a.is_contiguous() and asc.is_contiguous()
        assert a.dtype == torch.float8_e4m3fn and asc.dtype == torch.float8_e8m0fnu
        assert a.shape[-1] == packed.k and asc.shape == (*a.shape[:-1], packed.k // 32)
        m = math.prod(a.shape[:-1])
        output = torch.empty((*a.shape[:-1], packed.n), dtype=torch.bfloat16, device='cpu')
        status = self.fn(a.data_ptr(), asc.data_ptr(), packed.packed.data_ptr(), packed.scale.data_ptr(),
            packed.nan_masks.data_ptr(), m, packed.n, packed.k, self.workers, output.data_ptr())
        assert status == 0, ('FP8 VNNI GEMM rejected shape', status)
        return output

    def apply(self, mode, a, asc, b, bs):
        assert mode in (4, 8)
        if mode == 4:
            self.baseline.workers = self.workers
            return self.baseline.apply(mode, a, asc, b, bs)
        key = (b.data_ptr(), bs.data_ptr(), tuple(b.shape), tuple(bs.shape))
        if key not in self.cache:
            self.cache[key] = (b, bs, self.pack(b, bs))
        return self.apply_packed(a, asc, self.cache[key][2])

    def fp8(self, a, asc, b, bs, scale_dtype=torch.float32, block_size=128):
        assert block_size == 32
        cpu.COUNTS['fp8_gemm'] += 1
        return self.apply(8, a, asc, b, bs)

    def install(self):
        cpu.fp8_gemm = self.fp8
