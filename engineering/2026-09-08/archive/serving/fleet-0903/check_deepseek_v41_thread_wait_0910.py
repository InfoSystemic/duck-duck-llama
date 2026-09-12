#!/usr/bin/env python3
"""Time native GEMMs and a full native-precision expert under one wait policy."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import threading
import time
import torch
import torch.nn.functional as F
import deepseek_v41_cpu_reference_0910 as cpu
from deepseek_v41_native_bridge_0910 import NativeGemm
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import sha256

BASE = Path(__file__).resolve().parent


def main():
    output = Path(sys.argv[1])
    torch.set_num_threads(16); torch.set_num_interop_threads(1); torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(411001)
    library = BASE / 'results/deepseek-v41-native-gemm-0910b/libdeepseek-v41-native-gemm.so'
    native = NativeGemm(library, 16)
    x = torch.randn(1, 5120, dtype=torch.float32).bfloat16()
    weights = []
    for n, k in [(2304, 5120), (2304, 5120), (5120, 2304)]:
        w = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
        s = torch.randint(118, 123, (n, k // 32), dtype=torch.uint8).view(torch.float8_e8m0fnu)
        weights.append((w, s))
    a, asc = cpu.act_quant(x, 32, 'ue8m0', torch.float8_e8m0fnu)
    rows = []; errors = []

    def request_thread():
        try:
            def project(values, weight):
                q, scale = cpu.act_quant(values, 32, 'ue8m0', torch.float8_e8m0fnu)
                return native.apply(4, q, scale, *weight)

            def expert():
                gate = project(x, weights[0]).float().clamp(max=10)
                up = project(x, weights[1]).float().clamp(min=-10, max=10)
                return project((F.silu(gate) * up * .1).bfloat16(), weights[2])

            for name, fn in [('gemm', lambda: native.apply(4, a, asc, *weights[0])), ('expert', expert)]:
                for _ in range(50):
                    fn()
                reference = fn(); timings = []
                for _ in range(5):
                    began = time.perf_counter()
                    for _ in range(100):
                        value = fn()
                    timings.append((time.perf_counter() - began) / 100)
                    assert torch.equal(reference, value) and torch.isfinite(value).all()
                rows.append(dict(component=name, seconds_per_call=timings, median_seconds=statistics.median(timings),
                    output_sha256=hashlib.sha256(reference.view(torch.uint8).numpy().tobytes()).hexdigest()))
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=request_thread); thread.start(); thread.join()
    if errors:
        raise errors[0]
    result = dict(passed=True, component_only=True, fresh_request_thread=True, workers=16,
        wait_policy=os.environ['OMP_WAIT_POLICY'], spin_count=int(os.environ['GOMP_SPINCOUNT']), measurements=rows,
        source_sha256={str(p): sha256(p) for p in [Path(__file__), library, Path(cpu.__file__), BASE / 'deepseek_v41_native_bridge_0910.py']})
    atomic_json(output, result)
    print(json.dumps(dict(passed=True, spin_count=result['spin_count'],
        median_microseconds={r['component']: r['median_seconds'] * 1e6 for r in rows})), flush=True)


if __name__ == '__main__':
    main()
