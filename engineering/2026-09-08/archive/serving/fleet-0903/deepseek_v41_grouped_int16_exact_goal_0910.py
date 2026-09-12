"""Exact grouped FP4 lattice path with per-block baseline fallback and FP8 baseline.

PackedFP4Exact16 uses a NEW pair-major nibble layout; old VNNI packed caches are
not interchangeable. A supplied provider owns its bounded cache and must drop
source references on resident eviction. This bridge retains no tensors between
calls. With no provider it packs temporary buffers for each call.
"""
import ctypes
from pathlib import Path

import torch

import deepseek_v41_cpu_reference_0910 as cpu
from deepseek_v41_native_grouped_goal_0910 import GroupedNativeGemm


class Descriptor(ctypes.Structure):
    _fields_ = [('mode', ctypes.c_int), ('n', ctypes.c_int), ('k', ctypes.c_int),
                ('reserved', ctypes.c_int), ('a', ctypes.c_void_p), ('asc', ctypes.c_void_p),
                ('b', ctypes.c_void_p), ('bsc', ctypes.c_void_p), ('packed', ctypes.c_void_p),
                ('packed_scale', ctypes.c_void_p), ('output', ctypes.c_void_p)]


class PackedFP4Exact16:
    def __init__(self, native, weight, scale):
        assert weight.device.type == scale.device.type == 'cpu'
        assert weight.is_contiguous() and scale.is_contiguous()
        assert weight.dtype == torch.float4_e2m1fn_x2 and scale.dtype == torch.float8_e8m0fnu
        assert weight.ndim == 2
        self.n, self.k = weight.shape[0], weight.shape[1] * 2
        assert self.k % 32 == 0 and scale.shape == (self.n, self.k // 32)
        tiles, blocks = (self.n + 15) // 16, self.k // 32
        self.packed = torch.empty((tiles, blocks, 8, 2, 16), dtype=torch.uint8, device='cpu')
        self.scales = torch.empty((tiles, blocks, 16), dtype=torch.uint8, device='cpu')
        status = native.pack_fn(weight.data_ptr(), scale.data_ptr(), self.n, self.k, native.workers,
                                self.packed.data_ptr(), self.scales.data_ptr())
        assert status == 0, ('Exact16 pack rejected buffers', status)
        self.storage_bytes = self.packed.numel() + self.scales.numel()


class GroupedExact16Gemm:
    def __init__(self, library, pack_provider=None, workers=16, fallback=None):
        self.library = ctypes.CDLL(str(Path(library).resolve()))
        self.workers = workers
        self.pack_provider = pack_provider if pack_provider is not None else self.pack
        self.fallback = fallback if fallback is not None else GroupedNativeGemm(
            Path(__file__).resolve().parent / 'results/deepseek-v41-native-grouped-goal-0910/libdeepseek-v41-native-grouped.so', workers)
        self.pack_fn = self.library.ds41_exact16_pack4
        self.pack_fn.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_int] * 3 + [ctypes.c_void_p] * 2
        self.pack_fn.restype = ctypes.c_int
        self.fn = self.library.ds41_exact16_grouped
        self.fn.argtypes = [ctypes.POINTER(Descriptor), ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        self.fn.restype = ctypes.c_int
        self.calls = self.native_calls = self.fallback_calls = 0
        self.last_eligible_blocks = self.last_fallback_blocks = 0
        self.eligible_blocks = self.fallback_blocks = 0
        assert ctypes.sizeof(Descriptor) == 72

    def pack(self, weight, scale):
        return PackedFP4Exact16(self, weight, scale)

    def apply(self, tasks, output=None):
        tasks = list(tasks)
        assert 1 <= len(tasks) <= 64 and 1 <= self.workers <= 64
        n = tasks[0][3].shape[0]
        assert 1 <= n <= 131072
        if output is None:
            output = torch.empty((len(tasks), n), dtype=torch.bfloat16, device='cpu')
        assert output.device.type == 'cpu' and output.is_contiguous()
        assert output.dtype == torch.bfloat16 and output.shape == (len(tasks), n)
        output_start, output_end = output.data_ptr(), output.data_ptr() + output.numel() * 2

        def distinct(tensor):
            start = tensor.data_ptr()
            end = start + tensor.numel() * tensor.element_size()
            assert end <= output_start or start >= output_end, 'Output aliases an input'

        for mode, a, asc, b, bs in tasks:
            assert mode in (4, 8)
            assert all(x.device.type == 'cpu' and x.is_contiguous() for x in (a, asc, b, bs))
            assert a.dtype == torch.float8_e4m3fn and asc.dtype == bs.dtype == torch.float8_e8m0fnu
            assert b.dtype == (torch.float4_e2m1fn_x2 if mode == 4 else torch.float8_e4m3fn)
            k = a.shape[-1]
            assert a.numel() == k and 32 <= k <= 32768 and k % 32 == 0
            assert asc.shape == (*a.shape[:-1], k // 32)
            assert b.shape == (n, k // 2 if mode == 4 else k)
            assert bs.shape == (n if mode == 4 else (n + 31) // 32, k // 32)
            for tensor in (a, asc, b, bs):
                distinct(tensor)
        self.calls += 1
        retained = []
        descriptors = (Descriptor * len(tasks))()
        for index, (mode, a, asc, b, bs) in enumerate(tasks):
            pp = ps = None
            k = a.shape[-1]
            if mode == 4:
                packed = self.pack_provider(b, bs)
                if packed is None:
                    self.fallback_calls += 1
                    self.last_eligible_blocks = self.last_fallback_blocks = 0
                    self.fallback.workers = self.workers
                    return self.fallback.apply(tasks, output)
                retained.append(packed)
                assert isinstance(packed, PackedFP4Exact16), 'Old VNNI pack layout is incompatible'
                assert (packed.n, packed.k) == (n, k)
                assert packed.packed.shape == ((n + 15) // 16, k // 32, 8, 2, 16)
                assert packed.scales.shape == ((n + 15) // 16, k // 32, 16)
                for tensor in (packed.packed, packed.scales):
                    assert tensor.device.type == 'cpu' and tensor.dtype == torch.uint8 and tensor.is_contiguous()
                    distinct(tensor)
                pp, ps = packed.packed.data_ptr(), packed.scales.data_ptr()
            descriptors[index] = Descriptor(mode, n, k, 0, a.data_ptr(), asc.data_ptr(), b.data_ptr(), bs.data_ptr(),
                                            pp, ps, output_start + index * n * 2)
        stats = (ctypes.c_uint64 * 2)()
        status = self.fn(descriptors, len(tasks), self.workers, stats)
        assert status == 0, ('Exact16 grouped rejected descriptors', status)
        self.last_eligible_blocks, self.last_fallback_blocks = map(int, stats)
        self.eligible_blocks += self.last_eligible_blocks
        self.fallback_blocks += self.last_fallback_blocks
        self.native_calls += 1
        cpu.COUNTS['fp4_gemm'] += sum(task[0] == 4 for task in tasks)
        cpu.COUNTS['fp8_gemm'] += sum(task[0] == 8 for task in tasks)
        return output
