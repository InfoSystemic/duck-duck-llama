"""One-region mixed VNNI-FP4/baseline-FP8 decode with an external bounded pack provider."""
import ctypes
from pathlib import Path
import torch
import deepseek_v41_cpu_reference_0910 as cpu
from deepseek_v41_native_grouped_goal_0910 import GroupedNativeGemm


class Descriptor(ctypes.Structure):
    _fields_ = [('mode', ctypes.c_int), ('n', ctypes.c_int), ('k', ctypes.c_int),
                ('reserved', ctypes.c_int), ('a', ctypes.c_void_p), ('asc', ctypes.c_void_p),
                ('b', ctypes.c_void_p), ('bsc', ctypes.c_void_p), ('sums', ctypes.c_void_p),
                ('output', ctypes.c_void_p)]


class GroupedVnniGemm:
    def __init__(self, library, pack_provider, workers=16, fallback=None):
        self.library = ctypes.CDLL(str(Path(library).resolve()))
        self.workers = workers
        self.pack_provider = pack_provider
        self.fallback = fallback if fallback is not None else GroupedNativeGemm(
            Path(__file__).resolve().parent / 'results/deepseek-v41-native-grouped-goal-0910/libdeepseek-v41-native-grouped.so', workers)
        self.fn = self.library.ds41_grouped_vnni
        self.fn.argtypes = [ctypes.POINTER(Descriptor), ctypes.c_int, ctypes.c_int]
        self.fn.restype = ctypes.c_int
        self.calls = self.native_calls = self.fallback_calls = 0
        assert ctypes.sizeof(Descriptor) == 64 and callable(pack_provider)

    def apply(self, tasks, output=None):
        """Same task tuples/result layout as GroupedNativeGemm.apply.

        pack_provider(weight,scale) owns cache/eviction and returns PackedFP4 or
        None. A None result sends the entire call through the baseline backend.
        This bridge retains no source/packed tensors after the call returns.
        """
        tasks = list(tasks)
        assert 1 <= len(tasks) <= 64 and 1 <= self.workers <= 64
        n = tasks[0][3].shape[0]
        assert 1 <= n <= 131072
        if output is None:
            output = torch.empty(len(tasks), n, dtype=torch.bfloat16, device='cpu')
        assert output.device.type == 'cpu' and output.is_contiguous()
        assert output.dtype == torch.bfloat16 and output.shape == (len(tasks), n)
        output_start = output.data_ptr()
        output_end = output_start + output.numel() * 2
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
            assert b.ndim == 2 and b.shape == (n, k // 2 if mode == 4 else k)
            assert bs.shape == (n if mode == 4 else (n + 31) // 32, k // 32)
            for tensor in (a, asc, b, bs):
                distinct(tensor)
        self.calls += 1
        retained = []
        descriptors = (Descriptor * len(tasks))()
        for index, (mode, a, asc, b, bs) in enumerate(tasks):
            k = a.shape[-1]
            if mode == 4:
                packed = self.pack_provider(b, bs)
                if packed is None:
                    self.fallback_calls += 1
                    self.fallback.workers = self.workers
                    return self.fallback.apply(tasks, output)
                retained.append(packed)
                assert (packed.n, packed.k) == (n, k)
                tiles, blocks = (n + 15) // 16, k // 32
                assert packed.packed.shape == (tiles, blocks, 8, 32) and packed.packed.dtype == torch.uint8
                assert packed.sums.shape == (tiles, blocks, 16) and packed.sums.dtype == torch.int16
                assert packed.scales.shape == (tiles, blocks, 16) and packed.scales.dtype == torch.uint8
                for tensor in (packed.packed, packed.sums, packed.scales):
                    assert tensor.device.type == 'cpu' and tensor.is_contiguous()
                    distinct(tensor)
                weight_pointer, scale_pointer, sum_pointer = packed.packed.data_ptr(), packed.scales.data_ptr(), packed.sums.data_ptr()
            else:
                weight_pointer, scale_pointer, sum_pointer = b.data_ptr(), bs.data_ptr(), None
            descriptors[index] = Descriptor(mode, n, k, 0, a.data_ptr(), asc.data_ptr(),
                weight_pointer, scale_pointer, sum_pointer, output_start + index * n * 2)
        status = self.fn(descriptors, len(tasks), self.workers)
        assert status == 0, ('Grouped VNNI rejected descriptors', status)
        self.native_calls += 1
        cpu.COUNTS['fp4_gemm'] += sum(task[0] == 4 for task in tasks)
        cpu.COUNTS['fp8_gemm'] += sum(task[0] == 8 for task in tasks)
        return output
