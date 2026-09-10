#!/usr/bin/env python3
"""Exhaustively validate direct FP8 expansion, then the matrix and official graph checks."""
import ctypes
import json
import math
from pathlib import Path
import struct
import sys
import check_deepseek_v41_native_gemm_0910 as original


def main():
    out=Path(sys.argv[1]);lib=ctypes.CDLL(str(out/'libdeepseek-v41-native-gemm.so'))
    fn=lib.ds41_decode8;fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int];fn.restype=ctypes.c_int
    codes=(ctypes.c_uint8*256)(*range(256));values=(ctypes.c_float*256)()
    assert fn(codes,values,256)==0
    exact=0
    for c,v in enumerate(values):
        mag=c&127;e=mag>>3;f=mag&7
        if mag==127:assert math.isnan(v);continue
        expected=math.ldexp(f,-9) if e==0 else math.ldexp(8+f,e-10)
        if c&128:expected=-expected
        assert struct.pack('<f',v)==struct.pack('<f',expected),(c,v,expected)
        exact+=1
    original.main()
    prior=json.loads((out.parent/'deepseek-v41-native-gemm-0910/graph-check.json').read_text())
    current=json.loads((out/'graph-check.json').read_text())
    assert prior['checks']==current['checks'],'Direct expansion changed the synthetic official-graph logits'
    (out/'decode-check.json').write_text(json.dumps(dict(passed=True,exact_finite_codes=exact,nan_codes=2,
        signed_zero_preserved=True,official_synthetic_graph_logits_unchanged=True),indent=2)+'\n')


if __name__=='__main__':main()
