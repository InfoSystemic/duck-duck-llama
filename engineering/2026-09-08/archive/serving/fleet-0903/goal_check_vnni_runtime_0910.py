#!/usr/bin/env python3
"""One-core VNNI adapter parity, memory-cap and native-mapping lifetime fixture."""
import gc
import json
from pathlib import Path
import tempfile
import types
import weakref

import torch

import deepseek_v41_cpu_reference_0910 as cpu
from deepseek_v41_native_bridge_0910 import NativeGemm
from deepseek_v41_resident_store_0910 import ResidentStore
from goal_vnni_runtime_0910 import install


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(4100928)
    base = Path(__file__).resolve().parent
    baseline = NativeGemm(base / 'results/deepseek-v41-native-gemm-0910b/libdeepseek-v41-native-gemm.so', 1)
    store = ResidentStore.__new__(ResidentStore)
    store.residents, store.resident_names = {}, set()
    store.resident_enabled = True
    original_grouped_backend = object()
    grouped = types.SimpleNamespace(enabled=True, backend=original_grouped_backend)
    module = types.SimpleNamespace(fp4_gemm=baseline.fp4, fp8_gemm=baseline.fp8)
    runtime = types.SimpleNamespace(module=module, store=store, native=baseline,
                                    _goal_grouped_moe_0910=grouped)
    original_release = store.release
    release_observations = []
    candidate = None

    def observed_release(this, prefix):
        pointers = {p.data_ptr() for p in this.residents[prefix].parameters() if not p.is_meta}
        assert not any(key[0] in pointers or key[1] in pointers for key in candidate.native.cache)
        release_observations.append(prefix)
        return original_release(prefix)

    store.release = types.MethodType(observed_release, store)
    restored_release = store.release
    candidate = install(runtime, cap_bytes=1 << 20)
    assert not grouped.enabled and module.fp8_gemm == baseline.fp8
    rows = []
    with torch.inference_mode():
        for m, n, k in [(1, 33, 32), (3, 64, 96), (1, 129, 128)]:
            a, a_s = cpu.act_quant(torch.randn(m, k).bfloat16(), 32, 'ue8m0', torch.float8_e8m0fnu)
            b = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
            b_s = torch.randint(118, 130, (n, k // 32), dtype=torch.uint8).view(torch.float8_e8m0fnu)
            expected = baseline.apply(4, a, a_s, b, b_s)
            actual = module.fp4_gemm(a, a_s, b, b_s, act_block_size=32)
            assert torch.equal(actual, expected), (m, n, k, int((actual != expected).sum()))
            packs = candidate.pack_count
            assert torch.equal(module.fp4_gemm(a, a_s, b, b_s, act_block_size=32), expected)
            assert candidate.pack_count == packs
            candidate.enabled = False
            assert grouped.enabled
            assert torch.equal(module.fp4_gemm(a, a_s, b, b_s, act_block_size=32), expected)
            candidate.enabled = True
            assert not grouped.enabled
            rows.append(dict(shape=[m, n, k], random_baseline_exact=True))
    candidate.clear()
    assert not candidate.native.cache and candidate.packed_bytes == 0
    assert candidate.cache_hits == 3 and candidate.cache_misses == 3 and candidate.pack_count == 3
    assert candidate.pack_seconds >= 0
    alternate_grouped_backend = object()
    candidate.set_grouped_backend(alternate_grouped_backend)
    assert grouped.enabled and grouped.backend is alternate_grouped_backend
    candidate.enabled = False
    assert grouped.enabled and grouped.backend is original_grouped_backend
    candidate.enabled = True
    assert grouped.enabled and grouped.backend is alternate_grouped_backend
    candidate.cap_bytes = 608  # One 32x32 projection, including all metadata.
    with tempfile.TemporaryDirectory(prefix='goal-vnni-lifetime-') as temporary:
        root = Path(temporary)
        prefix = 'layers.0.ffn.experts.0.'

        def mapped_expert():
            expert = torch.nn.Module()
            for index, name in enumerate(['w1', 'w2', 'w3']):
                linear = torch.nn.Module()
                for kind, shape, dtype, code in [('weight', (32, 16), torch.float4_e2m1fn_x2, 0x12 + index),
                                                  ('scale', (32, 1), torch.float8_e8m0fnu, 123)]:
                    path = root / (name + '.' + kind)
                    if not path.exists():
                        path.write_bytes(bytes([code]) * (shape[0] * shape[1]))
                    raw = torch.from_file(str(path), shared=False, size=path.stat().st_size, dtype=torch.uint8)
                    parameter = torch.nn.Parameter(raw.view(dtype).reshape(shape), requires_grad=False)
                    linear.register_parameter(kind, parameter)
                linear.weight.scale = linear.scale
                expert.add_module(name, linear)
            store.residents[prefix] = expert
            store.resident_names.update(prefix + name for name, _ in expert.named_parameters())
            return expert

        expert = mapped_expert()
        reference = weakref.ref(expert.w1.weight)
        with torch.inference_mode():
            a, a_s = cpu.act_quant(torch.randn(1, 32).bfloat16(), 32, 'ue8m0', torch.float8_e8m0fnu)
            expected = baseline.apply(4, a, a_s, expert.w1.weight, expert.w1.scale)
            assert torch.equal(module.fp4_gemm(a, a_s, expert.w1.weight, expert.w1.scale, act_block_size=32), expected)
            assert candidate.packed_bytes == candidate.cap_bytes == 608
            assert len(candidate.native.cache) == 1
            second = baseline.apply(4, a, a_s, expert.w2.weight, expert.w2.scale)
            cap_fallbacks = candidate.cap_fallback_calls
            assert torch.equal(module.fp4_gemm(a, a_s, expert.w2.weight, expert.w2.scale, act_block_size=32), second)
            assert candidate.cap_fallback_calls == cap_fallbacks + 1
            assert candidate.packed_bytes == 608 and len(candidate.native.cache) == 1
        assert str(root) in Path('/proc/self/maps').read_text()
        store.release(prefix)
        gc.collect()
        assert reference() is None, 'Packed cache retained evicted source parameter'
        assert not candidate.native.cache and candidate.packed_bytes == 0
        assert all(p.is_meta for p in expert.parameters())
        assert str(root) not in Path('/proc/self/maps').read_text(), 'Evicted native files remain mapped'
        expert = mapped_expert()
        with torch.inference_mode():
            assert torch.equal(module.fp4_gemm(a, a_s, expert.w1.weight, expert.w1.scale, act_block_size=32), expected)
        assert candidate.packed_bytes == 608
        store.release_all()
        assert not candidate.native.cache and candidate.packed_bytes == 0
        assert str(root) not in Path('/proc/self/maps').read_text()
    candidate.uninstall()
    assert grouped.enabled and grouped.backend is original_grouped_backend and module.fp4_gemm == baseline.fp4
    assert module.fp8_gemm == baseline.fp8 and store.release == restored_release
    assert not candidate.native.cache and candidate.packed_bytes == 0
    print(json.dumps(dict(passed=True, cases=rows, packed_cap_enforced=True,
                          baseline_fallback_at_cap=True, dropped_before_resident_release=True,
                          released_native_file_mappings=True, source_weakref_released=True,
                          reload_and_release_all=True, grouped_toggle=True, fp8_unchanged=True,
                          clear_uninstall=True, release_calls=len(release_observations),
                          universal_bitwise_equivalence_claimed=False,
                          full_checkpoint_loaded=False, model_speed_measured=False), indent=2))


if __name__ == '__main__':
    main()
