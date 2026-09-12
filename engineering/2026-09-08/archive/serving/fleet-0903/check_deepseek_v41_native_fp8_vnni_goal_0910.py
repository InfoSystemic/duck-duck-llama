#!/usr/bin/env python3
"""FP8 word-VNNI exact-value integer oracle and experimental rounding fixture."""
import ctypes
import hashlib
import json
from pathlib import Path
import torch
import deepseek_v41_cpu_reference_0910 as cpu
from deepseek_v41_native_bridge_0910 import NativeGemm
from deepseek_v41_native_fp8_vnni_goal_0910 import NativeFP8VnniGemm
from check_deepseek_v41_native_vnni_goal_0910 import same, differences, hostile
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import sha256

BASE = Path(__file__).resolve().parent
OUT = BASE / 'results/deepseek-v41-native-fp8-vnni-goal-0910'


def oracle(a, asc, b, bs):
    m, k, n = a.numel() // a.shape[-1], a.shape[-1], b.shape[0]
    aa, bb = a.float().reshape(m, k), b.float()
    an, bn = aa.isnan(), bb.isnan()
    ai = torch.where(an, 0, aa).mul(512).to(torch.int64)
    bi = torch.where(bn, 0, bb).mul(512).to(torch.int64)
    sa, sb = asc.float().reshape(m, k // 32), bs.float().repeat_interleave(32, 0)[:n]
    output = torch.zeros(m, n, dtype=torch.float32)
    for block in range(k // 32):
        start = block * 32
        total = (ai[:, None, start:start + 32] * bi[None, :, start:start + 32]).sum(-1)
        assert int(total.abs().max()) <= 1683627180032
        partial = total.float() * (1 / 262144)
        nan = an[:, start:start + 32].any(-1)[:, None] | bn[:, start:start + 32].any(-1)[None]
        partial = torch.where(nan, float('nan'), partial)
        output += (partial * sa[:, block, None]) * sb[None, :, block]
    return output.to(torch.bfloat16).reshape(*a.shape[:-1], n)


def digest(t):
    return hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_flush_denormal(False)
    torch.manual_seed(419118)
    lib = OUT / 'libdeepseek-v41-native-fp8-vnni.so'
    native = NativeFP8VnniGemm(lib, 1)
    baseline = NativeGemm(BASE / 'results/deepseek-v41-native-gemm-0910b/libdeepseek-v41-native-gemm.so', 1)
    sources = [Path(__file__), lib, BASE / 'deepseek-v41-native-fp8-vnni-goal-0910.cpp',
        BASE / 'deepseek-v41-native-gemm-0910b.cpp', BASE / 'deepseek_v41_native_fp8_vnni_goal_0910.py']
    hashes = {str(p): sha256(p) for p in sources}
    fn = native.library.ds41_fp8_vnni_decode32
    fn.argtypes = [ctypes.c_void_p] * 3
    fn.restype = ctypes.c_int
    raw = torch.arange(256, dtype=torch.uint8)
    for start in range(0, 256, 32):
        high = torch.empty(32, dtype=torch.int16)
        low = torch.empty(32, dtype=torch.uint16)
        assert fn(raw[start:].data_ptr(), high.data_ptr(), low.data_ptr()) == 0
        for offset, (h, l) in enumerate(zip(high.tolist(), low.tolist())):
            code = start + offset
            if (code & 127) != 127:
                assert h * 256 + l == int(raw[code:code + 1].view(torch.float8_e4m3fn).float().item() * 512), (code, h, l)
    cases = []
    def check(label, a, asc, b, bs):
        before = [digest(x) for x in [a, asc, b, bs]]
        packed = native.pack(b, bs)
        n, k = b.shape
        tiles, blocks = (n + 15) // 16, k // 32
        reconstruction = packed.packed.reshape(tiles, blocks, 16, 16, 2).permute(0, 3, 1, 2, 4).reshape(tiles * 16, k)
        assert torch.equal(reconstruction[:n], b.view(torch.uint8)) and (reconstruction[n:] == 0).all()
        nan_rows = ((reconstruction & 127) == 127).reshape(tiles, 16, blocks, 32).any(-1).permute(0, 2, 1)
        expected_mask = (nan_rows.to(torch.int32) * (1 << torch.arange(16, dtype=torch.int32))).sum(-1)
        assert torch.equal(expected_mask, packed.nan_masks.to(torch.int32))
        assert packed.scale.data_ptr() == bs.data_ptr()
        actual = native.apply_packed(a, asc, packed)
        expected = oracle(a, asc, b, bs)
        assert same(actual, expected), (label, int((actual != expected).sum()))
        reference = baseline.apply(8, a, asc, b, bs)
        metrics = differences(reference, actual)
        assert before == [digest(x) for x in [a, asc, b, bs]]
        case = dict(label=label, shape=[a.numel() // k, n, k], integer_oracle_exact=True,
            native_fp8_bits_preserved=True, nan_metadata_exact=True, original_scale_storage_used=True,
            source_inputs_preserved=True, native_weight_bytes=b.numel(), packed_weight_bytes=packed.packed.numel(),
            nan_metadata_bytes=packed.nan_masks.numel() * 2, **metrics)
        if label == 'hostile_bf16_ties':
            case.update(baseline_values=reference.flatten().float().tolist(), candidate_values=actual.flatten().float().tolist())
            assert case['baseline_values'] == [10752.0, -10752.0]
            assert case['candidate_values'] == [10816.0, -10816.0]
        cases.append(case)
        print(json.dumps(case), flush=True)
    # Exhaustive 256x256 pair-products also checks both NaN codes and signed zero.
    values = torch.arange(256, dtype=torch.uint8)[:, None].expand(256, 32).clone().view(torch.float8_e4m3fn)
    check('all_code_pair_products', values, torch.full((256, 1), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu),
        values, torch.full((8, 1), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu))
    for m, n, k in [(3, 129, 128), (1, 1280, 5120), (1, 5120, 8192), (1, 32768, 1280)]:
        a, asc = cpu.act_quant(torch.randn(m, k).bfloat16(), 32, 'ue8m0', torch.float8_e8m0fnu)
        raw = torch.randint(0, 256, (n, k), dtype=torch.uint8)
        raw[(raw & 127) == 127] = 126
        b = raw.view(torch.float8_e4m3fn)
        bs = torch.randint(118, 130, ((n + 31) // 32, k // 32), dtype=torch.uint8).view(torch.float8_e8m0fnu)
        check('random_real_shape', a, asc, b, bs)
    a, asc, b4, bs = hostile()
    raw4 = b4.view(torch.uint8)
    b8 = cpu._fp4_values(torch.stack([raw4 & 15, raw4 >> 4], -1).flatten(-2)).to(torch.float8_e4m3fn)
    check('hostile_bf16_ties', a, asc, b8, bs)
    a = torch.arange(256, dtype=torch.uint8).reshape(8, 32)
    a[(a & 127) == 127] = 126
    b = torch.randint(0, 256, (33, 32), dtype=torch.uint8).view(torch.float8_e4m3fn)
    asc = torch.tensor([0, 1, 63, 126, 127, 190, 253, 254], dtype=torch.uint8).reshape(8, 1).view(torch.float8_e8m0fnu)
    bs = torch.tensor([0, 254], dtype=torch.uint8).reshape(2, 1).view(torch.float8_e8m0fnu)
    check('scale_edges', a.view(torch.float8_e4m3fn), asc, b, bs)
    assert all(sha256(p) == h for p, h in hashes.items())
    result = dict(passed=True, workers=1, fp8_finite_hi_lo_codes_exact=254, cases=cases, source_sha256=hashes,
        integer_oracle_values=sum(c['values'] for c in cases), baseline_bit_differences=sum(c['bf16_bit_differences'] for c in cases),
        fp32_reduction_changed=True, full_checkpoint_loaded=False, model_tok_s_measured=False, component_only=True,
        native_weight_expansion=1.0, nan_metadata_fraction_for_full_tiles=2 / 512,
        limitation='Exact native input values; integer block summation changes baseline FP32 reduction rounding. No model or performance conclusion.')
    atomic_json(OUT / 'kernel-check.json', result)
    print(json.dumps({key: value for key, value in result.items() if key not in ['cases', 'source_sha256']}), flush=True)


if __name__ == '__main__':
    main()
