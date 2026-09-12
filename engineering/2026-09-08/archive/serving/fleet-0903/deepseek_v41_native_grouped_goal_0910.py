"""Batch decode projections into one native parallel region without weight copies."""
import ctypes
import math
from pathlib import Path
import torch
import deepseek_v41_cpu_reference_0910 as cpu


class Descriptor(ctypes.Structure):
    _fields_ = [('mode', ctypes.c_int), ('n', ctypes.c_int), ('k', ctypes.c_int),
                ('reserved', ctypes.c_int), ('a', ctypes.c_void_p), ('asc', ctypes.c_void_p),
                ('b', ctypes.c_void_p), ('bsc', ctypes.c_void_p), ('output', ctypes.c_void_p)]


class GroupedNativeGemm:
    def __init__(self, library, workers=16):
        self.library = ctypes.CDLL(str(Path(library).resolve()))
        self.workers = workers
        self.fn = self.library.ds41_gemm_grouped
        self.fn.argtypes = [ctypes.POINTER(Descriptor), ctypes.c_int, ctypes.c_int]
        self.fn.restype = ctypes.c_int
        assert ctypes.sizeof(Descriptor) == 56

    def apply(self, tasks, output=None):
        """tasks: (mode,a,a_scale,weight,weight_scale), one token per task.

        All output widths must agree. Result shape is [task_count,n], BF16;
        tasks may share activation pointers and may mix FP4 and FP8 weights.
        Input tensors and original weight storage are never modified or retained.
        """
        tasks = list(tasks)
        assert 1 <= len(tasks) <= 64 and 1 <= self.workers <= 64
        n = tasks[0][3].shape[0]
        if output is None:
            output = torch.empty(len(tasks), n, dtype=torch.bfloat16, device='cpu')
        assert output.device.type == 'cpu' and output.is_contiguous()
        assert output.dtype == torch.bfloat16 and output.shape == (len(tasks), n)
        output_start = output.data_ptr()
        output_end = output_start + output.numel() * output.element_size()
        descriptors = (Descriptor * len(tasks))()
        for index, (mode, a, a_s, b, b_s) in enumerate(tasks):
            assert mode in (4, 8)
            assert all(x.device.type == 'cpu' and x.is_contiguous() for x in (a, a_s, b, b_s))
            assert a.dtype == torch.float8_e4m3fn and a_s.dtype == b_s.dtype == torch.float8_e8m0fnu
            assert b.dtype == (torch.float4_e2m1fn_x2 if mode == 4 else torch.float8_e4m3fn)
            k = a.shape[-1]
            assert a.numel() == k and 32 <= k <= 32768 and k % 32 == 0
            assert a_s.shape == (*a.shape[:-1], k // 32)
            assert b.ndim == 2 and b.shape == (n, k // 2 if mode == 4 else k)
            assert b_s.shape == (n if mode == 4 else (n + 31) // 32, k // 32)
            for tensor in (a, a_s, b, b_s):
                start = tensor.data_ptr()
                end = start + tensor.numel() * tensor.element_size()
                assert end <= output_start or start >= output_end, 'Output aliases an input'
            descriptors[index] = Descriptor(mode, n, k, 0, a.data_ptr(), a_s.data_ptr(),
                b.data_ptr(), b_s.data_ptr(), output_start + index * n * 2)
        status = self.fn(descriptors, len(tasks), self.workers)
        assert status == 0, ('Grouped native GEMM rejected descriptors', status)
        cpu.COUNTS['fp4_gemm'] += sum(task[0] == 4 for task in tasks)
        cpu.COUNTS['fp8_gemm'] += sum(task[0] == 8 for task in tasks)
        return output
