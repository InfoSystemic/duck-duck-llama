"""One native call for variable-token FP4/FP8 tasks with externally owned packs."""
import ctypes
from pathlib import Path

import torch
import deepseek_v41_cpu_reference_0910 as cpu
from deepseek_v41_native_bridge_0910 import NativeGemm
from deepseek_v41_grouped_int16_exact_goal_0910 import PackedFP4Exact16


class Descriptor(ctypes.Structure):
    _fields_=[('mode',ctypes.c_int),('m',ctypes.c_int),('n',ctypes.c_int),('k',ctypes.c_int),
              ('a',ctypes.c_void_p),('asc',ctypes.c_void_p),('b',ctypes.c_void_p),
              ('bsc',ctypes.c_void_p),('packed',ctypes.c_void_p),
              ('packed_scale',ctypes.c_void_p),('output',ctypes.c_void_p)]


class RaggedExact16Gemm:
    def __init__(self, library, pack_provider, workers=16, fallback=None):
        self.library=ctypes.CDLL(str(Path(library).resolve()))
        self.fn=self.library.ds41_exact16_ragged
        self.fn.argtypes=[ctypes.POINTER(Descriptor),ctypes.c_int,ctypes.c_int,ctypes.c_void_p]
        self.fn.restype=ctypes.c_int
        self.workers=workers
        self.pack_provider=pack_provider
        self.fallback=fallback or NativeGemm(Path(__file__).resolve().parent /
            'results/deepseek-v41-native-gemm-0910b/libdeepseek-v41-native-gemm.so',workers)
        self.calls=self.native_calls=self.fallback_calls=0
        self.eligible_blocks=self.fallback_blocks=0
        assert ctypes.sizeof(Descriptor)==72 and callable(pack_provider)

    def apply(self, tasks):
        """Return a list of BF16 [m,n] outputs, one per validated task tuple."""
        tasks=list(tasks)
        assert 1<=len(tasks)<=128 and 1<=self.workers<=64
        n=tasks[0][3].shape[0];k=tasks[0][1].shape[-1]
        assert 1<=n<=131072 and 32<=k<=32768 and k%32==0
        sizes=[]
        for mode,a,asc,b,bs in tasks:
            assert mode in (4,8) and a.ndim==2 and 1<=a.shape[0]<=8 and a.shape[-1]==k
            assert all(x.device.type=='cpu' and x.is_contiguous() for x in (a,asc,b,bs))
            assert a.dtype==torch.float8_e4m3fn and asc.dtype==bs.dtype==torch.float8_e8m0fnu
            assert b.dtype==(torch.float4_e2m1fn_x2 if mode==4 else torch.float8_e4m3fn)
            assert b.shape==(n,k//2 if mode==4 else k)
            assert asc.shape==(a.shape[0],k//32)
            assert bs.shape==(n if mode==4 else (n+31)//32,k//32)
            sizes.append(a.shape[0])
        self.calls+=1
        retained=[]
        for mode,_,_,b,bs in tasks:
            packed=self.pack_provider(b,bs) if mode==4 else None
            if mode==4 and packed is None:
                self.fallback_calls+=1
                self.fallback.workers=self.workers
                return [self.fallback.apply(*task) for task in tasks]
            if mode==4:
                assert isinstance(packed,PackedFP4Exact16) and (packed.n,packed.k)==(n,k)
                assert packed.packed.shape==((n+15)//16,k//32,8,2,16)
                assert packed.scales.shape==((n+15)//16,k//32,16)
                assert all(t.device.type=='cpu' and t.dtype==torch.uint8 and t.is_contiguous()
                           for t in (packed.packed,packed.scales))
            retained.append(packed)
        output=torch.empty(sum(sizes),n,dtype=torch.bfloat16,device='cpu')
        descriptors=(Descriptor*len(tasks))()
        at=0
        for i,((mode,a,asc,b,bs),packed) in enumerate(zip(tasks,retained)):
            pp=packed.packed.data_ptr() if packed is not None else None
            ps=packed.scales.data_ptr() if packed is not None else None
            descriptors[i]=Descriptor(mode,sizes[i],n,k,a.data_ptr(),asc.data_ptr(),
                b.data_ptr(),bs.data_ptr(),pp,ps,output.data_ptr()+at*n*2)
            at+=sizes[i]
        stats=(ctypes.c_uint64*2)()
        status=self.fn(descriptors,len(tasks),self.workers,stats)
        assert status==0,('Ragged exact16 kernel rejected tasks',status)
        self.eligible_blocks+=int(stats[0]);self.fallback_blocks+=int(stats[1])
        self.native_calls+=1
        cpu.COUNTS['fp4_gemm']+=sum(t[0]==4 for t in tasks)
        cpu.COUNTS['fp8_gemm']+=sum(t[0]==8 for t in tasks)
        return list(output.split(sizes,dim=0))
