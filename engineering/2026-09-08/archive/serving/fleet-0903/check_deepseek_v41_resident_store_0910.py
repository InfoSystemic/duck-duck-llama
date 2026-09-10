#!/usr/bin/env python3
"""Exercise real mapped-file eviction, native format reload, and failure cleanup."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import types
import torch
from deepseek_v41_checkpoint_0910 import REVISION
from deepseek_v41_resident_store_0910 import ResidentStore
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import sha256


def check(destination):
    cases = []
    with tempfile.TemporaryDirectory(prefix='deepseek-v41-resident-check-') as temporary:
        root = Path(temporary)
        (root / 'owner.json').write_text(json.dumps(dict(task='deepseek-v41-native-0910', revision=REVISION)))
        store = ResidentStore.__new__(ResidentStore)
        store.root = root; store.manifest = root / 'tensors.jsonl'; store.manifest.touch()
        store.records = {}; store.verified = set(); store.existing_projection = {}
        store.residents = {}; store.resident_names = set(); store.resident_enabled = True
        store.touched = {}; store.tick = 0; store.evicted_bytes = 0
        store.downloaded_bytes = 0; store.network_seconds = 0; store.tensor_loads = 0; store.stored_bytes = 0
        store.catalog = types.SimpleNamespace(tensors={})
        store.pool = ThreadPoolExecutor(max_workers=1)
        prefixes = [f'layers.0.ffn.experts.{i}.' for i in range(3)]
        modules = []
        for index, prefix in enumerate(prefixes):
            module = torch.nn.Module()
            for w in ['w1', 'w2', 'w3']:
                linear = torch.nn.Module()
                for kind, shape, dtype, code in [('weight', (32, 16), torch.float4_e2m1fn_x2, 0x12 + index),
                                                 ('scale', (32, 1), torch.float8_e8m0fnu, 127 + index)]:
                    linear.register_parameter(kind, torch.nn.Parameter(torch.empty(shape, dtype=dtype, device='meta'), requires_grad=False))
                    name = prefix + w + '.' + kind
                    raw = bytes([code]) * (shape[0] * shape[1])
                    store.catalog.tensors[name] = dict(shape=list(shape), dtype='I8' if kind == 'weight' else 'F8_E8M0',
                        bytes=len(raw), start=0, total=len(raw), url='fixture://' + name, shard='fixture', raw=raw)
                linear.weight.scale = linear.scale
                module.add_module(w, linear)
            modules.append(module)
        store.cap_bytes = sum(v['bytes'] for n, v in store.catalog.tensors.items() if n.startswith(tuple(prefixes[:2])))
        fail = [False]

        def fetch(_self, meta, offset, size):
            if fail[0]:
                raise RuntimeError('Fixture transport failure after eviction')
            return bytearray(meta['raw'][offset:offset + size])

        store._fetch = types.MethodType(fetch, store)
        try:
            store.activate(modules[0], prefixes[0]); store.activate(modules[1], prefixes[1])
            pointer = modules[0].w1.weight.data_ptr(); loads = store.tensor_loads
            store.activate(modules[0], prefixes[0]); store.ensure([prefixes[0] + 'w1.weight'])
            assert modules[0].w1.weight.data_ptr() == pointer and store.tensor_loads == loads
            cases.append('warm_mapping_reused_without_tensor_reload')
            names = [prefix + name for prefix, module in [(prefixes[0], modules[0]), (prefixes[2], modules[2])]
                     for name, _ in module.named_parameters()]
            fail[0] = True
            try:
                store.ensure(names)
            except RuntimeError:
                pass
            else:
                raise AssertionError('Fixture download failure was swallowed')
            assert prefixes[0] in store.residents and prefixes[1] not in store.residents
            assert all(p.is_meta for p in modules[1].parameters())
            assert prefixes[1] not in Path('/proc/self/maps').read_text()
            assert modules[0].w1.weight.data_ptr() == pointer
            cases.append('evicted_mappings_released_on_download_failure')
            cases.append('requested_expert_protected_from_eviction')
            fail[0] = False
            store.activate(modules[2], prefixes[2])
            store.activate(modules[1], prefixes[1])
            assert all(p.is_meta for i in [0, 2] for p in modules[i].parameters())
            for name, parameter in modules[1].named_parameters():
                meta = store.catalog.tensors[prefixes[1] + name]
                assert bytes(parameter.view(torch.uint8).flatten().tolist()) == meta['raw']
            assert all(getattr(modules[1], w).weight.scale is getattr(modules[1], w).scale for w in ['w1', 'w2', 'w3'])
            cases.append('evicted_native_fp4_and_scales_reload_exactly')
            cases.append('scale_aliases_restored')
            del parameter
            store.release_all()
            assert not store.resident_names and all(p.is_meta for m in modules for p in m.parameters())
            assert not any(prefix in Path('/proc/self/maps').read_text() for prefix in prefixes)
            cases.append('release_all_unmaps_native_files')
        finally:
            store.pool.shutdown(wait=True, cancel_futures=True)
    result = dict(passed=True, cases=cases, source_sha256={str(p): sha256(p) for p in
                  [Path(__file__), Path(__file__).with_name('deepseek_v41_resident_store_0910.py')]})
    atomic_json(destination, result)
    return result


if __name__ == '__main__':
    import sys
    print(json.dumps(check(Path(sys.argv[1]))))
