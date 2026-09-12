#!/usr/bin/env python3
"""One-core exact grouped FP4 fixture; no model loading or performance timing."""
import gc
import hashlib
import json
from pathlib import Path
import weakref

import torch

import deepseek_v41_cpu_reference_0910 as cpu
from deepseek_v41_native_bridge_0910 import NativeGemm
from deepseek_v41_native_vnni_goal_0910 import NativeVnniGemm
from deepseek_v41_grouped_int16_exact_goal_0910 import GroupedExact16Gemm
from check_deepseek_v41_native_vnni_goal_0910 import hostile


BASE = Path(__file__).resolve().parent
OUT = BASE / 'results/deepseek-v41-grouped-int16-exact-goal-0910'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same(actual, expected):
    assert actual.shape == expected.shape
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16)), (
        int((actual.view(torch.int16) != expected.view(torch.int16)).sum()),
        float((actual.float() - expected.float()).abs().nan_to_num().max()))


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(4100930)
    library = OUT / 'libdeepseek-v41-grouped-int16-exact.so'
    source_paths = [Path(__file__), BASE / 'deepseek-v41-grouped-int16-exact-goal-0910.cpp',
                    BASE / 'deepseek_v41_grouped_int16_exact_goal_0910.py',
                    BASE / 'deepseek-v41-native-gemm-0910b.cpp', library]
    source_hashes = {str(p): sha(p) for p in source_paths}
    external = BASE / 'deepseek-v41-native-gemm-0910d.cpp'
    external_hash = sha(external)
    baseline = NativeGemm(BASE / 'results/deepseek-v41-native-gemm-0910b/libdeepseek-v41-native-gemm.so', 1)
    candidate = GroupedExact16Gemm(library, workers=1)
    cache = {}
    pack_checks = [0]

    def provider(b, bs):
        key = (b.data_ptr(), bs.data_ptr(), tuple(b.shape), tuple(bs.shape))
        if key not in cache:
            packed = candidate.pack(b, bs)
            tiles, blocks = (packed.n + 15) // 16, packed.k // 32
            reconstructed = packed.packed.permute(0, 4, 1, 2, 3).reshape(tiles * 16, packed.k // 2)
            assert torch.equal(reconstructed[:packed.n], b.view(torch.uint8))
            assert not reconstructed[packed.n:].any()
            reconstructed_scales = packed.scales.permute(0, 2, 1).reshape(tiles * 16, blocks)
            assert torch.equal(reconstructed_scales[:packed.n], bs.view(torch.uint8))
            assert packed.storage_bytes == packed.packed.numel() + packed.scales.numel()
            assert not hasattr(packed, 'sums')
            cache[key] = (b, bs, packed)
            pack_checks[0] += 1
        return cache[key][2]

    candidate.pack_provider = provider
    cases, exact_values = [], 0

    def check(label, tasks):
        nonlocal exact_values
        before = [(sha_tensor(x),) for task in tasks for x in task[1:]]
        expected = torch.cat([baseline.apply(*task) for task in tasks], 0)
        actual = candidate.apply(tasks)
        same(actual, expected)
        assert before == [(sha_tensor(x),) for task in tasks for x in task[1:]]
        row = dict(label=label, tasks=len(tasks), values=actual.numel(), baseline_bits_exact=True,
                   eligible_unique_blocks=candidate.last_eligible_blocks,
                   fallback_unique_blocks=candidate.last_fallback_blocks)
        cases.append(row)
        exact_values += actual.numel()
        return row

    def make_task(mode, n, k, activation=None):
        if activation is None:
            activation = cpu.act_quant(torch.randn(1, k).bfloat16(), 32, 'ue8m0', torch.float8_e8m0fnu)
        a, asc = activation
        if mode == 4:
            b = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
            bs = torch.randint(118, 130, (n, k // 32), dtype=torch.uint8).view(torch.float8_e8m0fnu)
        else:
            b = (torch.randn(n, k, dtype=torch.float32) * .7).to(torch.float8_e4m3fn)
            bs = torch.randint(118, 130, ((n + 31) // 32, k // 32), dtype=torch.uint8).view(torch.float8_e8m0fnu)
        return mode, a, asc, b, bs

    with torch.inference_mode():
        # The previous exact-integer/512 variant actually disagrees on this
        # cancellation/tie case. Tiny FP8 terms force our baseline block path.
        a, asc, b, bs = hostile()
        expected = baseline.apply(4, a, asc, b, bs)
        old = NativeVnniGemm(BASE / 'results/deepseek-v41-native-vnni-goal-0910/libdeepseek-v41-native-vnni.so', 1)
        changed = old.apply(4, a, asc, b, bs)
        assert not torch.equal(expected, changed), 'Hostile fixture no longer distinguishes the old arithmetic'
        row = check('hostile_subnormal_cancellation', [(4, a[i:i+1], asc[i:i+1], b, bs) for i in range(2)])
        assert row['eligible_unique_blocks'] == 0 and row['fallback_unique_blocks'] == 2
        row['baseline_values'] = expected.float().flatten().tolist()
        row['old_vnni_values'] = changed.float().flatten().tolist()
        eligible = a.clone()
        eligible.view(torch.uint8)[0, 16] = 8
        eligible.view(torch.uint8)[1, 16] = 136
        row = check('eligible_cancellation', [(4, eligible[i:i+1], asc[i:i+1], b, bs) for i in range(2)])
        assert row['eligible_unique_blocks'] == 2 and row['fallback_unique_blocks'] == 0
        old.clear()
        # All FP8 bit patterns, including signed zero and both NaNs, with each
        # code filling one K32 block. Exactly 206 codes meet the lattice rule.
        common = make_task(4, 33, 32)
        eligible_count = fallback_count = 0
        for begin in range(0, 256, 32):
            tasks = []
            for code in range(begin, begin + 32):
                av = torch.full((1, 32), code, dtype=torch.uint8).view(torch.float8_e4m3fn)
                tasks.append((4, av, torch.full((1, 1), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu), common[3], common[4]))
            row = check('all_fp8_codes_' + str(begin), tasks)
            eligible_count += row['eligible_unique_blocks']; fallback_count += row['fallback_unique_blocks']
        assert (eligible_count, fallback_count) == (206, 50)
        check('mixed_modes_and_k', [make_task(4, 65, 32), make_task(8, 65, 96),
                                  make_task(4, 65, 128), make_task(8, 65, 32)])
        special = make_task(4, 33, 224)
        special[2].view(torch.uint8)[0] = torch.tensor([0, 1, 126, 127, 253, 254, 255], dtype=torch.uint8)
        special[4].view(torch.uint8).copy_(torch.tensor([255, 254, 253, 127, 126, 1, 0], dtype=torch.uint8).expand(33, 7))
        check('special_scale_codes', [special])
        for n, k, modes in [(2304, 5120, [4]*12 + [8]*2), (5120, 2304, [4]*6 + [8])]:
            cache.clear(); gc.collect()
            shared_a = cpu.act_quant(torch.randn(1, k).bfloat16(), 32, 'ue8m0', torch.float8_e8m0fnu)
            tasks = [make_task(mode, n, k, shared_a) for mode in modes]
            row = check('full_width_up14' if len(modes) == 14 else 'full_width_down7', tasks)
            assert row['eligible_unique_blocks'] + row['fallback_unique_blocks'] == k // 32
            del tasks
        # A provider cap miss delegates the entire call to baseline, with no
        # output overlap or partial computation retained.
        cache.clear(); gc.collect()
        cap_tasks = [make_task(4, 33, 32), make_task(4, 33, 32), make_task(8, 33, 32)]
        call = [0]
        refs = []

        def cap_provider(b, bs):
            call[0] += 1
            if call[0] == 2:
                return None
            packed = candidate.pack(b, bs)
            refs.append(weakref.ref(packed))
            return packed

        candidate.pack_provider = cap_provider
        prior = candidate.fallback_calls
        check('whole_call_cap_fallback', cap_tasks)
        assert candidate.fallback_calls == prior + 1
        gc.collect(); assert all(ref() is None for ref in refs)
        # Successful calls also retain neither packed weights nor source refs.
        refs.clear()
        def transient_provider(b, bs):
            packed = candidate.pack(b, bs)
            refs.append(weakref.ref(packed))
            return packed
        candidate.pack_provider = transient_provider
        check('transient_packs_not_retained', cap_tasks)
        gc.collect(); assert all(ref() is None for ref in refs)
    assert all(sha(path) == digest for path, digest in source_hashes.items())
    assert sha(external) == external_hash
    result = dict(passed=True, workers=1, cases=cases, exact_baseline_values=exact_values,
                  exact_eligible_fp8_codes=206, ineligible_fp8_codes=50, lossless_packs_checked=pack_checks[0],
                  sums_metadata_bytes=0, no_retained_bridge_packs=True, source_buffers_preserved=True,
                  input_sha256=source_hashes, read_only_external_source=dict(path=str(external), sha256=external_hash),
                  full_checkpoint_loaded=False, model_speed_measured=False, kernel_speed_measured=False)
    (OUT / 'kernel-check.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['passed', 'workers', 'exact_baseline_values', 'exact_eligible_fp8_codes',
                                           'ineligible_fp8_codes', 'lossless_packs_checked', 'sums_metadata_bytes']}, indent=2))


def sha_tensor(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).reshape(-1).numpy().tobytes()).hexdigest()


if __name__ == '__main__':
    main()
