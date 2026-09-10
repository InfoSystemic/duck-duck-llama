#!/usr/bin/env python3
"""Check the compiled mixed-precision kernels against independent CPU equations."""
import hashlib
import json
from pathlib import Path
import runpy
import sys
import time
import torch
import deepseek_v41_cpu_reference_0910 as cpu
from deepseek_v41_native_bridge_0910 import NativeGemm
from qwen_split_trial import sha256


def tree_oracle(mode,a,asc,b,bsc):
    k,n=a.shape[-1],b.shape[0]
    aa=a.float().reshape(-1,k)
    raw=b.view(torch.uint8)
    bb=cpu._fp4_values(torch.stack([raw&15,raw>>4],-1).flatten(-2)) if mode==4 else b.float()
    sa=asc.float().reshape(-1,k//32)
    sb=bsc.float() if mode==4 else bsc.float().repeat_interleave(32,0)[:n]
    out=torch.zeros(aa.shape[0],n,dtype=torch.float32)
    for block,start in enumerate(range(0,k,32)):
        products=aa[:,None,start:start+32]*bb[None,:,start:start+32]
        partial=products[...,:16]+products[...,16:]
        for half in [8,4,2,1]: partial=partial[...,:half]+partial[...,half:2*half]
        out+=(partial[...,0]*sa[:,block,None])*sb[:,block][None]
    return out.to(torch.bfloat16).reshape(*a.shape[:-1],n)


def main():
    out=Path(sys.argv[1]);library=out/'libdeepseek-v41-native-gemm.so'
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.bfloat16);torch.manual_seed(410911)
    native=NativeGemm(library,1)
    cases=[]
    for mode in [4,8]:
        for m,n,k in [(1,33,32),(3,64,96),(7,129,128),(1,2304,5120),(2,5120,2304)]:
            x=(torch.randn(m,k,dtype=torch.float32)*2).bfloat16()
            a,asc=cpu.act_quant(x,32,'ue8m0',torch.float8_e8m0fnu)
            if mode==4:
                b=torch.randint(0,256,(n,k//2),dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
            else:
                b=(torch.randn(n,k,dtype=torch.float32)*2).to(torch.float8_e4m3fn)
            bsc=torch.randint(118,130,(n if mode==4 else (n+31)//32,k//32),dtype=torch.uint8).view(torch.float8_e8m0fnu)
            expected=tree_oracle(mode,a,asc,b,bsc)
            old=(cpu.fp4_gemm(a,asc,b,bsc,act_block_size=32) if mode==4 else cpu.fp8_gemm(a,asc,b,bsc,block_size=32))
            outputs=[]
            for threads in [1,4,16]:
                native.workers=threads
                t=time.perf_counter();actual=native.apply(mode,a,asc,b,bsc);elapsed=time.perf_counter()-t
                assert torch.equal(actual,expected),(mode,m,n,k,threads,int((actual!=expected).sum()))
                assert torch.isfinite(actual).all()
                outputs.append(dict(workers=threads,seconds=elapsed))
            error=(old.float()-expected.float()).abs()
            assert torch.allclose(old.float(),expected.float(),rtol=.008,atol=.008),error.max().item()
            cases.append(dict(mode=mode,m=m,n=n,k=k,exact_tree=True,values=expected.numel(),
                prior_reference_mismatches=int((old!=expected).sum()),max_prior_reference_error=float(error.max()),timings=outputs))
    native.workers=16;native.install()
    checker=Path(__file__).with_name('check_deepseek_v41_cpu_graph_0910.py')
    # The graph fixture normally initializes this once in a fresh process.
    original_setter=torch.set_num_interop_threads
    def already_one(n): assert n==torch.get_num_interop_threads()==1
    torch.set_num_interop_threads=already_one
    try: runpy.run_path(str(checker),run_name='__main__')
    finally: torch.set_num_interop_threads=original_setter
    graph=json.loads((out/'graph-check.json').read_text());assert graph['passed']
    result=dict(passed=True,finished=time.time(),cases=cases,exact_values=sum(c['values']*3 for c in cases),
        full_official_synthetic_graph_passed=True,full_checkpoint_loaded=False,
        input_sha256={str(p):sha256(p) for p in [Path(__file__),Path(cpu.__file__),Path(__file__).with_name('deepseek_v41_native_bridge_0910.py'),library,checker]})
    (out/'kernel-check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,exact_values=result['exact_values'],synthetic_graph=True)),flush=True)


if __name__=='__main__': main()
