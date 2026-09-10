#!/usr/bin/env python3
"""Native FP8 projection, independent CPU block arithmetic, and official Engram gate."""
import ast
import ctypes as C
import hashlib
import json
import mmap
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from check_deepseek_v41_engram_lookup_0910 import Config as LookupConfig, Shard, TokenizerAdapter
from qwen_split_trial import sha256

BASE = Path(__file__).resolve().parent
OFFICIAL = BASE / 'results/deepseek-v41-intake-0910/official/inference'
U8P, U16P, U32P, U64P, I64P, F32P = [C.POINTER(t) for t in
    [C.c_uint8, C.c_uint16, C.c_uint32, C.c_uint64, C.c_int64, C.c_float]]


class Config(C.Structure):
    _fields_ = [('input_dim', C.c_int64), ('dim', C.c_int64), ('hc_mult', C.c_int64),
        ('weight', U8P), ('weight_bytes', C.c_uint64), ('scale', U8P), ('scale_bytes', C.c_uint64),
        ('q_weight', U16P), ('k_weight', U16P), ('norm_elements', C.c_uint64), ('norm_eps', C.c_float)]


def ptr(value, kind):
    return None if value is None else value.ctypes.data_as(kind)


def bf(value):
    return torch.as_tensor(value, dtype=torch.float32).to(torch.bfloat16).view(torch.uint16).numpy().copy()


def f32(value):
    return torch.from_numpy(np.asarray(value, dtype=np.uint16)).view(torch.bfloat16).float()


def quant_reference(value):
    x = f32(value).reshape(-1, value.shape[-1] // 32, 32)
    # Independent log2/ceil on the FP32 ratio, rather than the native exponent-bit shortcut.
    ratio = (x.abs().amax(-1).clamp_min(1e-4) * np.float32(1 / 448)).numpy()
    exponent = np.ceil(np.log2(ratio.astype(np.float64))).astype(np.int32)
    scale = torch.from_numpy(np.exp2(exponent.astype(np.float64)).astype(np.float32))
    q = (x / scale[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.view(torch.uint8).numpy().reshape(value.shape).copy(), (exponent + 127).astype(np.uint8)


def projection_reference(lookup, weight, scales):
    q, codes = quant_reference(lookup)
    m, k = lookup.shape
    n = weight.shape[0]
    a = torch.from_numpy(q).view(torch.float8_e4m3fn).float().reshape(m, k // 32, 32)
    w = torch.from_numpy(weight).view(torch.float8_e4m3fn).float().reshape(n, k // 32, 32)
    sa = torch.from_numpy(codes.copy()).view(torch.float8_e8m0fnu).float()
    sw = torch.from_numpy(scales.copy()).view(torch.float8_e8m0fnu).float()
    result = torch.zeros(m, n, dtype=torch.float32)
    absolute_sum = torch.zeros_like(result)
    for block in range(k // 32):
        partial = a[:, block] @ w[:, block].T
        factor_a, factor_b = sa[:, block, None], sw[:, block].repeat_interleave(32)[None, :]
        result += (partial * factor_a) * factor_b
        absolute_sum += ((a[:, block].abs() @ w[:, block].abs().T) * factor_a) * factor_b
    return result.to(torch.bfloat16).view(torch.uint16).numpy().copy(), absolute_sum.numpy()


def load_official_gate():
    tree = ast.parse((OFFICIAL / 'model.py').read_text())
    node, = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Engram']
    namespace = dict(torch=torch, nn=nn, ModelArgs=object, EngramLayout=object)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(OFFICIAL / 'model.py'), 'exec'), namespace)
    return namespace['Engram'], hashlib.sha256(ast.dump(node).encode()).hexdigest()


def gate_reference(cls, kv, hidden, q, k, mask):
    # Execute the actual published forward method. Its projection output is
    # supplied independently so this comparison isolates the normalized gate.
    class FixedProjection(nn.Module):
        def forward(self, _): return f32(kv).to(torch.bfloat16)
    model = cls.__new__(cls)
    nn.Module.__init__(model)
    model.dim, model.hc_mult = hidden.shape[-1], hidden.shape[-2]
    model.clamp_value, model.eps = 1e-6, 1e-20
    model.q_weight = nn.Parameter(f32(q).to(torch.bfloat16), requires_grad=False)
    model.k_weight = nn.Parameter(f32(k).to(torch.bfloat16), requires_grad=False)
    model.embed, model.wkv = nn.Identity(), FixedProjection()
    mask_tensor = None if mask is None else torch.from_numpy(mask.astype(bool))
    ht = f32(hidden).to(torch.bfloat16)
    with torch.inference_mode():
        official = model(ht, torch.zeros(len(kv), 1, 1), mask_tensor).view(torch.uint16).numpy().copy()
    # Also retain the FP32 intermediate to distinguish output rounding from gate error.
    dim, copies = model.dim, model.hc_mult
    key = f32(kv[:, :copies * dim]).reshape(-1, copies, dim)
    value = f32(kv[:, copies * dim:])
    h, weight = f32(hidden), f32(q) * f32(k)
    rstd = torch.rsqrt(h.square().mean(-1) + model.eps) * torch.rsqrt(key.square().mean(-1) + model.eps)
    dot = (h * weight * key).sum(-1) * rstd * dim ** -0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(model.clamp_value).sqrt(), dot))
    if mask_tensor is not None: gate = gate.masked_fill(~mask_tensor[:, None], 0)
    raw = h + gate[..., None] * value[:, None, :]
    assert np.array_equal(raw.to(torch.bfloat16).view(torch.uint16).numpy(), official)
    return official, raw.numpy(), gate.numpy()


def actual_lookup():
    import importlib.util
    manifest_path = BASE / 'results/deepseek-v41-engram-lookup-0910/real-rows.json'
    manifest = json.loads(manifest_path.read_text())
    library = BASE / 'results/deepseek-v41-engram-lookup-0910/libdeepseek-v41-engram-lookup.so'
    proof = json.loads((library.parent / 'result.json').read_text())
    assert proof['passed'] and sha256(library) == proof['library_sha256']
    spec = importlib.util.spec_from_file_location('project_reference_engram', OFFICIAL / 'engram.py')
    official = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = official
    spec.loader.exec_module(official)
    args = SimpleNamespace(**json.loads((OFFICIAL / 'config.json').read_text()), max_batch_size=1, max_seq_len=8)
    layout = official.EngramLayout.from_args(args)
    reference = official.NgramHashState(args, layout, TokenizerAdapter())
    tokens = np.array([42], dtype=np.int64)
    expected_ids = reference(torch.from_numpy(tokens[None]), 0).numpy().astype(np.uint64)
    assert expected_ids.tolist() == manifest['selected_rows']
    token_map = reference.token_map.numpy().astype(np.uint32)
    moduli = reference.primes.numpy().astype(np.uint64).reshape(-1)
    multipliers = reference.multipliers.numpy().astype(np.uint64).reshape(-1)
    rows = np.array(layout.num_embeddings, dtype=np.uint64)
    config = LookupConfig(ptr(token_map, U32P), len(token_map), 99092, args.engram_pad_id,
        ptr(moduli, U64P), ptr(multipliers, U64P), ptr(rows, U64P))
    regions, shards = [], []
    file_hashes = {str(manifest_path): sha256(manifest_path), str(library): sha256(library)}
    try:
        for layer, n in enumerate(rows.tolist()):
            pointers = []
            for suffix, width in [('weight', 256), ('scale', 8)]:
                region = mmap.mmap(-1, n * width, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS | 0x4000,
                    prot=mmap.PROT_READ | mmap.PROT_WRITE)
                regions.append(region)
                pointers.append(C.addressof(C.c_uint8.from_buffer(region)))
                for record in manifest['records']:
                    if record['layer'] != layer or record['suffix'] != suffix: continue
                    assert sha256(record['file']) == record['sha256']
                    raw = Path(record['file']).read_bytes()
                    start = record['row'] * width
                    region[start:start+width] = raw
                    file_hashes[record['file']] = record['sha256']
            shards.append(Shard(layer, 0, n, C.cast(pointers[0], U8P), n * 256, C.cast(pointers[1], U8P), n * 8))
        lib = C.CDLL(str(library))
        lib.deepseek_v41_lookup_create.argtypes = [C.POINTER(LookupConfig), C.POINTER(Shard), C.c_int64, C.c_int64, C.c_int64, C.c_int, C.c_int]
        lib.deepseek_v41_lookup_create.restype = C.c_void_p
        lib.deepseek_v41_lookup_forward.argtypes = [C.c_void_p, I64P, U8P, C.c_int64, C.c_int64, U16P, C.c_uint64, U64P, C.c_uint64]
        lib.deepseek_v41_lookup_destroy.argtypes = [C.c_void_p]
        handle = lib.deepseek_v41_lookup_create(C.byref(config), (Shard * 2)(*shards), 2, 8, 8, 1, 1)
        assert handle
        try:
            output = np.zeros((1, 2, 24, 256), dtype=np.uint16)
            ids = np.zeros((1, 2, 24), dtype=np.uint64)
            assert lib.deepseek_v41_lookup_forward(handle, ptr(tokens, I64P), None, 1, 0,
                ptr(output, U16P), output.size, ptr(ids, U64P), ids.size) == 0
            assert np.array_equal(ids, expected_ids[0])
        finally: lib.deepseek_v41_lookup_destroy(handle)
    finally:
        for region in regions: region.close()
    assert torch.isfinite(f32(output)).all()
    return output.reshape(2, 6144), file_hashes


def main():
    out, library = map(Path, sys.argv[1:])
    assert not (out / 'correctness.json').exists()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    lib = C.CDLL(str(library))
    lib.deepseek_v41_act_quant.argtypes = [U16P, C.c_int64, C.c_int64, U8P, C.c_uint64, U8P, C.c_uint64]
    lib.deepseek_v41_project_create.argtypes = [C.POINTER(Config), C.c_int64, C.c_int, C.c_int]
    lib.deepseek_v41_project_create.restype = C.c_void_p
    lib.deepseek_v41_project_destroy.argtypes = [C.c_void_p]
    lib.deepseek_v41_project_apply.argtypes = [C.c_void_p, U16P, U16P, U8P, C.c_int64,
        U16P, C.c_uint64, U16P, C.c_uint64, F32P, C.c_uint64, F32P, C.c_uint64]
    cls, gate_ast_sha = load_official_gate()
    rng = np.random.default_rng(41009101)
    cases, rejected_calls, rejected_configs, timings = [], [], [], []
    inputs = {str(p): sha256(p) for p in [Path(__file__), library, OFFICIAL / 'model.py', OFFICIAL / 'kernel.py',
        OFFICIAL / 'config.json', BASE / 'check_deepseek_v41_engram_lookup_0910.py']}

    # Every finite BF16 code, including subnormals, both signed zeros, and exponent extremes.
    every = np.arange(65536, dtype=np.uint16)
    every = every[(every & 0x7f80) != 0x7f80].reshape(-1, 128)
    quantized = []
    for label, x in [('every-finite-bf16', every), ('random', bf(rng.normal(size=(17, 6144)) * 10)),
            ('zero', np.zeros((3, 32), dtype=np.uint16))]:
        q, s = np.full(x.size + 2, 0xa5, dtype=np.uint8), np.full(x.size // 32 + 2, 0xa5, dtype=np.uint8)
        expected_q, expected_s = quant_reference(x)
        assert lib.deepseek_v41_act_quant(ptr(x, U16P), len(x), x.shape[1], ptr(q[1:-1], U8P), x.size,
            ptr(s[1:-1], U8P), x.size // 32) == 0
        assert q[0] == q[-1] == s[0] == s[-1] == 0xa5
        assert np.array_equal(q[1:-1].reshape(x.shape), expected_q), label
        assert np.array_equal(s[1:-1].reshape(expected_s.shape), expected_s), label
        quantized.append(dict(label=label, values=x.size, scale_codes=x.size // 32, bit_exact=True))

    def native(original_handle, input_lookup, input_hidden, input_mask, omit=False, **changes):
        handle, lookup, hidden, mask = original_handle, input_lookup, input_hidden, input_mask
        count, copies, dim = hidden.shape
        result = np.full(hidden.size + 2, 0xa53c, dtype=np.uint16)
        kv = np.full(count * (copies + 1) * dim + 2, 0xa53c, dtype=np.uint16)
        raw = np.full(hidden.size + 2, 31415., dtype=np.float32)
        gates = np.full(count * copies + 2, 31415., dtype=np.float32)
        args = dict(handle=handle, lookup=ptr(lookup, U16P), hidden=ptr(hidden, U16P), mask=ptr(mask, U8P), tokens=count,
            output=ptr(result[1:-1], U16P), output_elements=hidden.size,
            projected=None if omit else ptr(kv[1:-1], U16P), projected_elements=0 if omit else kv.size - 2,
            raw_output=None if omit else ptr(raw[1:-1], F32P), raw_elements=0 if omit else raw.size - 2,
            gates=None if omit else ptr(gates[1:-1], F32P), gate_elements=0 if omit else gates.size - 2)
        args.update(changes)
        rc = lib.deepseek_v41_project_apply(*args.values())
        for array, canary in [(result, 0xa53c), (kv, 0xa53c), (raw, 31415.), (gates, 31415.)]:
            assert array[0] == array[-1] == canary
        return (rc, result[1:-1].reshape(hidden.shape), kv[1:-1].reshape(count, (copies + 1) * dim),
            raw[1:-1].reshape(hidden.shape), gates[1:-1].reshape(count, copies))

    def check(label, lookup, hidden, weight, scales, q, k, mask=None, edge=False, measure=False):
        count, copies, dim = hidden.shape
        config = Config(lookup.shape[1], dim, copies, ptr(weight, U8P), weight.size,
            ptr(scales, U8P), scales.size, ptr(q, U16P), ptr(k, U16P), q.size, 1e-20)
        expected_kv, absolute_sum = projection_reference(lookup, weight, scales)
        all_modes = []
        for vectorize, workers in [(0, 1), (1, 1), (1, 4)]:
            handle = lib.deepseek_v41_project_create(C.byref(config), max(5, count), vectorize, workers)
            assert handle, (label, vectorize, workers)
            try:
                rc, result, kv, raw, gates = native(handle, lookup, hidden, mask)
                assert rc == 0, (label, vectorize, workers, rc)
                official, raw_ref, gates_ref = gate_reference(cls, kv, hidden, q, k, mask)
                actual_kv, ref_kv = f32(kv).numpy(), f32(expected_kv).numpy()
                projection_error = np.abs(actual_kv - ref_kv)
                # One BF16 ULP plus a conservative FP32 reduction bound. Separately
                # report exact fraction; never describe this comparison as GPU parity.
                ulp = np.exp2(np.floor(np.log2(np.maximum(np.abs(ref_kv), np.float32(2**-126))))) / 128
                bound = ulp + (lookup.shape[1] // 32 + 32) * np.finfo(np.float32).eps * absolute_sum
                assert np.all(projection_error <= bound), (label, float(projection_error.max()))
                scaled = np.abs(raw - raw_ref) / (1 + np.abs(raw_ref))
                gate_error = np.abs(gates - gates_ref)
                assert np.max(scaled) <= 2e-5 and np.max(gate_error) <= 2e-6, (label, scaled.max(), gate_error.max())
                assert np.array_equal(result, bf(raw))
                if mask is not None: assert np.array_equal(result[mask == 0], hidden[mask == 0])
                if all_modes:
                    assert all(np.array_equal(a, b) for a, b in zip((result, kv, raw, gates), all_modes[0]))
                all_modes.append(tuple(a.copy() for a in (result, kv, raw, gates)))
                cases.append(dict(label=label, tokens=count, input_dim=lookup.shape[1], dim=dim, copies=copies,
                    vectorize=vectorize, workers=workers, values=result.size, projected_values=kv.size,
                    projection_exact_values=int(np.count_nonzero(kv == expected_kv)),
                    gate_bf16_exact_values=int(np.count_nonzero(result == official)),
                    max_projection_bf16_abs_error=float(projection_error.max()),
                    max_gate_abs_error=float(gate_error.max()), max_residual_scaled_error=float(scaled.max()),
                    output_sha256=hashlib.sha256(result.tobytes()).hexdigest(),
                    projected_sha256=hashlib.sha256(kv.tobytes()).hexdigest(), scalar_vector_workers_bit_exact=True))
                if measure and workers == 1:
                    observations = []
                    for _ in range(5):
                        started = time.perf_counter_ns()
                        checked = native(handle, lookup, hidden, mask, omit=True)
                        observations.append((time.perf_counter_ns() - started) / 1e6)
                        assert checked[0] == 0 and np.array_equal(checked[1], result)
                    timings.append(dict(label=label, vectorize=vectorize, workers=workers,
                        median_ms=statistics.median(observations), observations_ms=observations,
                        scope='Whole component including Python/ctypes wrapper and validation; fixed-input warm repetitions, not a controlled model benchmark.'))
                if edge:
                    for change in [dict(handle=None), dict(tokens=-1), dict(tokens=max(5, count)+1),
                            dict(lookup=None), dict(hidden=None), dict(output=None), dict(output_elements=0),
                            dict(projected_elements=0), dict(raw_elements=0), dict(gate_elements=0),
                            dict(projected=None), dict(raw_output=None), dict(gates=None)]:
                        bad = native(handle, lookup, hidden, mask, **change)
                        assert bad[0] == -1
                        assert all(np.all(a == v) for a, v in zip(bad[1:], [0xa53c, 0xa53c, 31415., 31415.]))
                        rejected_calls.append(dict(mode=[vectorize, workers], changed=list(change), unchanged=True))
                    bad_mask = np.full(count, 2, dtype=np.uint8)
                    bad_lookup, bad_hidden = lookup.copy(), hidden.copy()
                    bad_lookup.flat[-1] = 0x7fc0; bad_hidden.flat[-1] = 0x7f80
                    for lx, hx, mx in [(bad_lookup, hidden, mask), (lookup, bad_hidden, mask), (lookup, hidden, bad_mask)]:
                        bad = native(handle, lx, hx, mx)
                        assert bad[0] == -1 and all(np.all(a == v) for a, v in zip(bad[1:], [0xa53c, 0xa53c, 31415., 31415.]))
                        rejected_calls.append(dict(mode=[vectorize, workers], changed=['nonfinite input or bad mask'], unchanged=True))
                    assert native(handle, lookup, hidden, mask, omit=True)[0] == 0
                    inplace = hidden.copy()
                    ip = native(handle, lookup, inplace, mask, output=ptr(inplace, U16P))
                    assert ip[0] == 0 and np.array_equal(inplace, result)
                    empty = native(handle, lookup[:0], hidden[:0], None)
                    assert empty[0] == 0
            finally: lib.deepseek_v41_project_destroy(handle)
        if edge:
            for field, value in [('input_dim', 0), ('input_dim', 33), ('input_dim', 6176), ('dim', 0),
                    ('hc_mult', 5), ('weight_bytes', 1), ('scale_bytes', 1), ('norm_elements', 1),
                    ('norm_eps', 0), ('norm_eps', float('nan')), ('weight', None), ('scale', None), ('q_weight', None)]:
                changed = Config.from_buffer_copy(config)
                setattr(changed, field, value)
                assert not lib.deepseek_v41_project_create(C.byref(changed), 5, 1, 1)
                rejected_configs.append(field)
            for max_chunk, vectorize, workers in [(0, 1, 1), (513, 1, 1), (5, 2, 1), (5, 1, 0), (5, 1, 61)]:
                assert not lib.deepseek_v41_project_create(C.byref(config), max_chunk, vectorize, workers)
                rejected_configs.append('runtime limits')
        print(json.dumps(dict(completed=label, modes=len(all_modes), max_gate_error=max(c['max_gate_abs_error'] for c in cases if c['label'] == label))), flush=True)

    for index, (count, width, dim, copies) in enumerate([(1, 32, 32, 1), (3, 64, 64, 4),
            (5, 96, 128, 4), (17, 6144, 32, 4)]):
        lookup = bf(rng.normal(size=(count, width)))
        hidden = bf(rng.normal(size=(count, copies, dim)))
        weight = rng.integers(0, 256, ((copies+1)*dim, width), dtype=np.uint8)
        weight[(weight & 127) == 127] ^= 1
        scales = rng.integers(113, 120, ((copies+1)*dim//32, width//32), dtype=np.uint8)
        q = bf(rng.normal(1, .2, (copies, dim))); k = bf(rng.normal(1, .2, (copies, dim)))
        mask = None if count == 1 else np.arange(count, dtype=np.uint8) % 2
        check(f'synthetic-{index}', lookup, hidden, weight, scales, q, k, mask, edge=index == 1)
        if index == 1:
            check('zero-residual-and-lookup', np.zeros_like(lookup), np.zeros_like(hidden), weight, scales, q, k, None)

    manifest_path = out / 'weight-manifest.json'
    manifest = json.loads(manifest_path.read_text()); assert manifest['passed']
    inputs[str(manifest_path)] = sha256(manifest_path)
    lookup, lookup_inputs = actual_lookup()
    inputs.update(lookup_inputs)
    for layer_index, layer in enumerate([1, 14]):
        values = {}
        for r in manifest['records']:
            if r['layer'] != layer: continue
            assert sha256(r['file']) == r['sha256']
            values[r['suffix']] = np.fromfile(r['file'], dtype=np.uint16 if r['dtype'] == 'BF16' else np.uint8).reshape(r['shape'])
            inputs[r['file']] = r['sha256']
        hidden = bf(rng.normal(size=(1, 4, 5120)))
        check(f'actual-layer-{layer}-token42', lookup[layer_index:layer_index+1].copy(), hidden,
            values['wkv.weight'], values['wkv.scale'], values['q_weight'], values['k_weight'], measure=True)
    report = dict(passed=True, finished=time.time(), input_sha256=inputs, cases=cases, quantization=quantized,
        rejected_calls=rejected_calls, rejected_configurations=rejected_configs, timings=timings,
        official_engram_class_ast_sha256=gate_ast_sha, native_variants=3,
        activation_values_exact=sum(x['values'] for x in quantized),
        actual_projection_layers=[1, 14], actual_native_lookup_connected=True,
        full_model_loaded=False, gpu_reference_executed=False, imc_bandwidth_measured=False,
        limitations=['Pinned Engram.forward executes on CPU. FP8 GEMM is checked against an independent CPU transcription of publisher block arithmetic, not GPU execution.',
            'Complete real Engram projections and selected real lookup rows; residual inputs are synthetic.',
            'Scalar/AVX-512/one/four worker results match exactly. Publisher-style FP32 reductions are tolerance checked and exact BF16 fractions are reported.',
            'Lookup and projection/gate are composed in this checker, not wired to llama.cpp graph or request lifecycle.',
            'Warm component timings are descriptive only; no full-model speed, quality, or bandwidth claim.'])
    assert all(sha256(p) == digest for p, digest in inputs.items())
    (out / 'correctness.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(passed=True, cases=len(cases), exact_activation_values=report['activation_values_exact'], actual_layers=report['actual_projection_layers'])), flush=True)


if __name__ == '__main__':
    main()
