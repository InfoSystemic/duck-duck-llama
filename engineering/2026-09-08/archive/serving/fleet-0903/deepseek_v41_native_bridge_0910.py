"""Install native 32-element FP8/FP4 GEMMs into the official CPU graph bridge."""
import ctypes
import math
from pathlib import Path
import torch
import deepseek_v41_cpu_reference_0910 as cpu


class NativeGemm:
    def __init__(self, library, workers=16):
        self.library = ctypes.CDLL(str(Path(library).resolve()))
        self.workers = workers
        self.fn = self.library.ds41_gemm
        self.fn.argtypes = [ctypes.c_int] + [ctypes.c_void_p]*4 + [ctypes.c_int]*4 + [ctypes.c_void_p]
        self.fn.restype = ctypes.c_int

    def apply(self, mode, a, a_s, b, b_s):
        assert all(x.device.type == 'cpu' and x.is_contiguous() for x in (a,a_s,b,b_s))
        assert a.dtype == torch.float8_e4m3fn
        assert a_s.dtype == b_s.dtype == torch.float8_e8m0fnu
        assert b.dtype == (torch.float4_e2m1fn_x2 if mode == 4 else torch.float8_e4m3fn)
        k,n = a.shape[-1],b.shape[0]
        m = math.prod(a.shape[:-1])
        assert b.ndim == 2 and b.shape[1] == (k//2 if mode == 4 else k)
        assert a_s.shape == (*a.shape[:-1], k//32)
        assert b_s.shape == (n if mode == 4 else (n+31)//32,k//32)
        output = torch.empty(*a.shape[:-1],n,dtype=torch.bfloat16,device='cpu')
        status=self.fn(mode,a.data_ptr(),a_s.data_ptr(),b.data_ptr(),b_s.data_ptr(),m,n,k,self.workers,output.data_ptr())
        assert status == 0, ('Native GEMM rejected shape',status,mode,m,n,k)
        return output

    def fp8(self,a,a_s,b,b_s,scale_dtype=torch.float32,block_size=128):
        assert block_size == 32
        cpu.COUNTS['fp8_gemm'] += 1
        return self.apply(8,a,a_s,b,b_s)

    def fp4(self,a,a_s,b,b_s,scale_dtype=torch.float32,act_block_size=128):
        assert act_block_size == 32
        cpu.COUNTS['fp4_gemm'] += 1
        return self.apply(4,a,a_s,b,b_s)

    def install(self):
        cpu.fp8_gemm=self.fp8
        cpu.fp4_gemm=self.fp4
