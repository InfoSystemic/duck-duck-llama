"""Exact m<=8 FP4 batches sharing frozen exact16 packs; external provider owns cache."""
import ctypes
import math
from pathlib import Path
import torch
import deepseek_v41_cpu_reference_0910 as cpu
from deepseek_v41_native_bridge_0910 import NativeGemm
from deepseek_v41_grouped_int16_exact_goal_0910 import PackedFP4Exact16


class RetainedPackedFP4:
    """Borrow an existing lossless pack and retain its exact native fallback inputs."""
    def __init__(self, weight, scale, packed):
        self.source_weight, self.source_scale = weight, scale
        self.owner = packed
        self.n, self.k = packed.n, packed.k
        self.packed, self.scales = packed.packed, packed.scales
        self.storage_bytes = packed.storage_bytes
        assert weight.shape == (self.n, self.k // 2)
        assert scale.shape == (self.n, self.k // 32)


class BatchedExact16Gemm:
    def __init__(self, library, workers=16, pack_provider=None, baseline=None):
        self.library = ctypes.CDLL(str(Path(library).resolve()))
        self.workers = workers
        self.pack_provider = pack_provider
        self.baseline = baseline if baseline is not None else NativeGemm(library, workers)
        self.pack_fn = self.library.ds41_exact16_pack4
        self.pack_fn.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_int] * 3 + [ctypes.c_void_p] * 2
        self.pack_fn.restype = ctypes.c_int
        self.fn = self.library.ds41_exact16_batched
        self.fn.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 4 + [ctypes.c_void_p] * 2
        self.fn.restype = ctypes.c_int
        self.calls = self.fallback_calls = self.eligible_blocks = self.fallback_blocks = 0
        self.last_eligible_blocks = self.last_fallback_blocks = 0

    @staticmethod
    def bind_packed(weight, scale, packed):
        return RetainedPackedFP4(weight, scale, packed)

    def pack(self, weight, scale):
        return self.bind_packed(weight, scale, PackedFP4Exact16(self, weight, scale))

    def apply_packed(self, a, asc, packed, output=None):
        b, bs = packed.source_weight, packed.source_scale
        tensors = (a, asc, b, bs, packed.packed, packed.scales)
        assert all(t.device.type == 'cpu' and t.is_contiguous() for t in tensors)
        assert a.dtype == torch.float8_e4m3fn and asc.dtype == bs.dtype == torch.float8_e8m0fnu
        assert b.dtype == torch.float4_e2m1fn_x2
        assert packed.packed.dtype == packed.scales.dtype == torch.uint8
        k, n = packed.k, packed.n
        assert a.shape[-1] == k and asc.shape == (*a.shape[:-1], k // 32)
        assert b.shape == (n, k // 2) and bs.shape == (n, k // 32)
        assert packed.packed.shape == ((n + 15) // 16, k // 32, 8, 2, 16)
        assert packed.scales.shape == ((n + 15) // 16, k // 32, 16)
        m = math.prod(a.shape[:-1])
        assert 1 <= m <= 8 and 1 <= self.workers <= 64
        if output is None:
            output = torch.empty((*a.shape[:-1], n), dtype=torch.bfloat16, device='cpu')
        assert output.shape == (*a.shape[:-1], n) and output.dtype == torch.bfloat16
        assert output.device.type == 'cpu' and output.is_contiguous()
        start, end = output.data_ptr(), output.data_ptr() + output.numel() * 2
        for tensor in tensors:
            begin, finish = tensor.data_ptr(), tensor.data_ptr() + tensor.numel() * tensor.element_size()
            assert finish <= start or begin >= end, 'Output aliases an input'
        stats = (ctypes.c_uint64 * 2)()
        status = self.fn(a.data_ptr(), asc.data_ptr(), b.data_ptr(), bs.data_ptr(), packed.packed.data_ptr(),
            packed.scales.data_ptr(), m, n, k, self.workers, output.data_ptr(), stats)
        assert status == 0, ('Batched exact16 rejected shape', status, m, n, k)
        self.calls += 1
        self.last_eligible_blocks, self.last_fallback_blocks = int(stats[0]), int(stats[1])
        self.eligible_blocks += self.last_eligible_blocks
        self.fallback_blocks += self.last_fallback_blocks
        return output

    def apply(self, mode, a, asc, b, bs):
        assert mode in (4, 8)
        m = math.prod(a.shape[:-1])
        if mode == 8 or m > 8 or self.pack_provider is None:
            self.fallback_calls += 1
            self.baseline.workers = self.workers
            return self.baseline.apply(mode, a, asc, b, bs)
        packed = self.pack_provider(b, bs)
        if packed is None:
            self.fallback_calls += 1
            self.baseline.workers = self.workers
            return self.baseline.apply(mode, a, asc, b, bs)
        return self.apply_packed(a, asc, self.bind_packed(b, bs, packed))

    def fp4(self, a, asc, b, bs, scale_dtype=torch.float32, act_block_size=128):
        assert act_block_size == 32
        cpu.COUNTS['fp4_gemm'] += 1
        return self.apply(4, a, asc, b, bs)
