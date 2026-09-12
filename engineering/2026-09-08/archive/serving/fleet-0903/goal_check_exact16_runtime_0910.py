#!/usr/bin/env python3
"""One-core exact16 adapter parity, cap, toggles, and native mapping lifetime."""
import gc
import hashlib
import json
from pathlib import Path
import tempfile
import types
import weakref

import torch

import deepseek_v41_cpu_reference_0910 as cpu
from deepseek_v41_native_bridge_0910 import NativeGemm
from deepseek_v41_native_grouped_goal_0910 import GroupedNativeGemm
from deepseek_v41_resident_store_0910 import ResidentStore
from goal_exact16_runtime_0910 import install


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(4100931)
    base = Path(__file__).resolve().parent
    baseline = NativeGemm(base / 'results/deepseek-v41-native-gemm-0910b/libdeepseek-v41-native-gemm.so', 1)
    original_backend = GroupedNativeGemm(
        base / 'results/deepseek-v41-native-grouped-goal-0910/libdeepseek-v41-native-grouped.so', 1)
    store = ResidentStore.__new__(ResidentStore)
    store.residents, store.resident_names = {}, set()
    store.resident_enabled = True
    grouped = types.SimpleNamespace(enabled=True, backend=original_backend)
    module = types.SimpleNamespace(fp4_gemm=baseline.fp4, fp8_gemm=baseline.fp8)
    runtime = types.SimpleNamespace(module=module, store=store, native=baseline,
                                    _goal_grouped_moe_0910=grouped)
    original_release = store.release
    release_observations = []
    candidate = None

    def observed_release(this, prefix):
        pointers = {p.data_ptr() for p in this.residents[prefix].parameters() if not p.is_meta}
        assert not any(key[0] in pointers or key[1] in pointers for key in candidate.cache)
        assert not pointers.intersection(candidate.source_keys)
        release_observations.append(prefix)
        return original_release(prefix)

    store.release = types.MethodType(observed_release, store)
    restored_release = store.release
    candidate = install(runtime, cap_bytes=1 << 20)
    assert grouped.enabled and grouped.backend is candidate.backend
    assert module.fp4_gemm == baseline.fp4 and module.fp8_gemm == baseline.fp8
    rows = []
    with torch.inference_mode():
        for n, k in [(17, 32), (33, 96), (129, 128)]:
            a, asc = cpu.act_quant(torch.randn(1, k).bfloat16(), 32, 'ue8m0', torch.float8_e8m0fnu)
            b = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
            bs = torch.randint(118, 130, (n, k // 32), dtype=torch.uint8).view(torch.float8_e8m0fnu)
            b8 = torch.randn(n, k).to(torch.float8_e4m3fn)
            bs8 = torch.randint(118, 130, ((n + 31) // 32, k // 32), dtype=torch.uint8).view(torch.float8_e8m0fnu)
            tasks = [(4, a, asc, b, bs), (8, a, asc, b8, bs8)]
            expected = original_backend.apply(tasks)
            assert torch.equal(grouped.backend.apply(tasks), expected)
            packs = candidate.pack_count
            assert torch.equal(grouped.backend.apply(tasks), expected)
            assert candidate.pack_count == packs
            candidate.configure(dict(exact16=False, timing=False))
            assert grouped.enabled and grouped.backend is original_backend
            assert torch.equal(grouped.backend.apply(tasks), expected)
            candidate.configure(dict(exact16=True, timing=False))
            duration = candidate.metrics()['grouped_seconds']
            assert torch.equal(grouped.backend.apply(tasks), expected)
            assert candidate.metrics()['grouped_seconds'] == duration
            assert module.fp4_gemm == baseline.fp4 and module.fp8_gemm == baseline.fp8
            rows.append(dict(shape=[n, k], mixed_grouped_baseline_exact=True))
        # Prefill remains the untouched native function and never populates pack cache.
        pre_a, pre_asc = cpu.act_quant(torch.randn(3, 128).bfloat16(), 32, 'ue8m0', torch.float8_e8m0fnu)
        packs = candidate.pack_count
        assert torch.equal(module.fp4_gemm(pre_a, pre_asc, b, bs, act_block_size=32),
                           baseline.apply(4, pre_a, pre_asc, b, bs))
        assert candidate.pack_count == packs
    candidate.clear()
    assert not candidate.cache and not candidate.source_keys and candidate.packed_bytes == 0
    assert candidate.cache_hits == 6 and candidate.cache_misses == 3 and candidate.pack_count == 3
    metrics = candidate.metrics()
    assert metrics['grouped_calls'] == metrics['grouped_native_calls'] == 9
    assert metrics['pack_native_calls'] == 3
    assert metrics['grouped_seconds'] >= metrics['grouped_native_seconds'] >= 0
    assert metrics['pack_seconds'] >= metrics['pack_native_seconds'] >= 0
    candidate.configure(dict(exact16=True, timing=True))
    candidate.cap_bytes = 544  # One padded 32x32 projection, including scales.
    with tempfile.TemporaryDirectory(prefix='goal-exact16-lifetime-') as temporary:
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
        packed_reference = None
        with torch.inference_mode():
            a, asc = cpu.act_quant(torch.randn(1, 32).bfloat16(), 32, 'ue8m0', torch.float8_e8m0fnu)
            tasks = [(4, a, asc, expert.w1.weight, expert.w1.scale)]
            expected = original_backend.apply(tasks)
            assert torch.equal(grouped.backend.apply(tasks), expected)
            assert candidate.packed_bytes == candidate.cap_bytes == 544
            assert len(candidate.cache) == 1
            packed_reference = weakref.ref(next(iter(candidate.cache.values()))[2])
            tasks += [(4, a, asc, expert.w2.weight, expert.w2.scale)]
            expected_both = original_backend.apply(tasks)
            before = candidate.metrics()
            assert torch.equal(grouped.backend.apply(tasks), expected_both)
            after = candidate.metrics()
            assert after['cap_fallbacks'] == before['cap_fallbacks'] + 1
            assert after['whole_call_fallbacks'] == before['whole_call_fallbacks'] + 1
            assert after['grouped_native_calls'] == before['grouped_native_calls']
            assert candidate.packed_bytes == 544 and len(candidate.cache) == 1
            del tasks
        assert str(root) in Path('/proc/self/maps').read_text()
        candidate.enabled = False  # Eviction must still work while candidate is disabled.
        store.release(prefix)
        gc.collect()
        assert reference() is None and packed_reference() is None
        assert not candidate.cache and not candidate.source_keys and candidate.packed_bytes == 0
        assert all(p.is_meta for p in expert.parameters())
        assert str(root) not in Path('/proc/self/maps').read_text(), 'Evicted native files remain mapped'
        candidate.enabled = True
        expert = mapped_expert()
        with torch.inference_mode():
            assert torch.equal(grouped.backend.apply([(4, a, asc, expert.w1.weight, expert.w1.scale)]), expected)
        assert candidate.packed_bytes == 544
        store.release_all()
        assert not candidate.cache and candidate.packed_bytes == 0
        assert str(root) not in Path('/proc/self/maps').read_text()
    candidate.uninstall()
    assert grouped.enabled and grouped.backend is original_backend and module.fp4_gemm == baseline.fp4
    assert module.fp8_gemm == baseline.fp8 and store.release == restored_release
    assert not candidate.cache and not candidate.source_keys and candidate.packed_bytes == 0
    assert not hasattr(runtime, '_goal_exact16_runtime_0910')
    # Preserve a disabled baseline grouping state too.
    grouped.enabled = False
    candidate = install(runtime, cap_bytes=0)
    candidate.enabled = False
    assert not grouped.enabled and grouped.backend is original_backend
    candidate.uninstall()
    paths = [Path(__file__), base / 'goal_exact16_runtime_0910.py',
             base / 'deepseek_v41_grouped_int16_exact_goal_0910.py',
             base / 'results/deepseek-v41-grouped-int16-exact-goal-0910/libdeepseek-v41-grouped-int16-exact.so']
    result = dict(passed=True, cases=rows, packed_cap_enforced=True,
                  whole_call_baseline_fallback_at_cap=True, dropped_before_resident_release=True,
                  released_native_file_mappings=True, source_and_pack_weakrefs_released=True,
                  disabled_eviction=True, reload_and_release_all=True, grouped_toggle=True,
                  prefill_fp4_unchanged=True, fp8_unchanged=True, inclusive_native_timers=True,
                  clear_uninstall=True, release_calls=len(release_observations),
                  full_checkpoint_loaded=False, model_speed_measured=False,
                  sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
    destination = base / 'results/deepseek-v41-grouped-int16-exact-goal-0910/runtime-check.json'
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
