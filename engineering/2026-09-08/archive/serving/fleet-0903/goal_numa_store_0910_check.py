#!/usr/bin/env python3
"""Synthetic one-core activation/eviction/accounting fixtures; no model load."""
import gc
import hashlib
import json
import os
from pathlib import Path
import tempfile
import types
from unittest.mock import patch

import torch

import deepseek_v41_resident_store_0910 as resident
import goal_numa_store_0910 as candidate
from goal_numa_0910_check import containing_mapping, process_policy

BASE = Path(__file__).resolve().parent


class Weight(torch.nn.Module):
    def __init__(self, dtype=torch.float8_e4m3fn):
        super().__init__()
        raw = torch.arange(16384, dtype=torch.int64).to(torch.uint8).reshape(32, 512)
        self.weight = torch.nn.Parameter(raw.view(dtype), requires_grad=False)
        self.scale = torch.nn.Parameter(torch.ones(4, 32), requires_grad=False)
        self.weight.scale = self.scale


class Expert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = Weight()
        self.w2 = Weight(torch.float4_e2m1fn_x2)
        self.w3 = Weight(torch.bfloat16)


def fake_init(self, root, *_args, **_kwargs):
    self.root = Path(root)
    self.catalog = types.SimpleNamespace(tensors={})
    self.records = {}
    self.existing_projection = {}
    self.cap_bytes = 1 << 20
    self.stored_bytes = 0
    self.tick = 0
    self.touched = {}


def fake_ensure(self, names):
    amount = sum(self.catalog.tensors[n]['bytes'] for n in names if n not in self.records)
    if self.stored_bytes + amount > self.cap_bytes:
        # Only the lower transport/cache layer is mocked. ResidentStore.ensure
        # must itself detect these removed records and dispatch our release().
        self.records.clear()
        self.stored_bytes = 0
    for name in names:
        if name not in self.records:
            self.records[name] = dict(name=name, **self.catalog.tensors[name])
            self.stored_bytes += self.catalog.tensors[name]['bytes']


def fake_load(module, prefix, store):
    parameters = list(module.named_parameters())
    for name, p in parameters:
        store.catalog.tensors[prefix + name] = dict(bytes=p.numel() * p.element_size())
    store.ensure([prefix + name for name, _ in parameters])
    for name, p in parameters:
        if p.is_meta:
            owner_name, _, attr = name.rpartition('.')
            owner = module.get_submodule(owner_name)
            data = torch.arange(p.numel() * p.element_size(), dtype=torch.int64).to(torch.uint8)
            owner._parameters[attr] = torch.nn.Parameter(data.view(p.dtype).reshape(p.shape), requires_grad=False)
    for owner in module.modules():
        if getattr(owner, 'weight', None) is not None and getattr(owner, 'scale', None) is not None:
            owner.weight.scale = owner.scale


def main():
    assert os.sched_getaffinity(0) == {0}
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    before_policy = process_policy()
    prefix = 'layers.0.ffn.experts.0.'
    verified = {}
    with tempfile.TemporaryDirectory(prefix='goal_numa_store_0910_') as directory, \
            patch.object(resident.ServingStore, '__init__', fake_init), \
            patch.object(resident.ServingStore, 'ensure', fake_ensure), \
            patch.object(resident, 'load_parameters', fake_load):
        store = candidate.NumaResidentStore(directory, copy_limit_bytes=1 << 20,
                                             min_weight_bytes=1, reserve_bytes=0)
        expert = Expert()
        originals = {name: p.detach().clone() for name, p in expert.named_parameters()}
        scale_addresses = [owner.scale.data_ptr() for owner in (expert.w1, expert.w2, expert.w3)]
        store.activate(expert, prefix)
        assert store.numa_owned_bytes == 3 * 16384
        assert store.stats()['current_expert_groups'][prefix]['status'] == 'full'
        assert all(record['actual_sample']['verified'] for record in store.stats()['placements'][prefix])
        for name, p in expert.named_parameters():
            assert torch.equal(p.view(torch.uint8), originals[name].view(torch.uint8))
        addresses = [owner.weight.data_ptr() for owner in (expert.w1, expert.w2, expert.w3)]
        assert scale_addresses == [owner.scale.data_ptr() for owner in (expert.w1, expert.w2, expert.w3)]
        assert all(owner.weight.scale is owner.scale for owner in (expert.w1, expert.w2, expert.w3))
        counters = dict(store.numa_counters)
        store.activate(expert, prefix)
        assert dict(store.numa_counters) == counters
        # Avoid retaining the final parameter from the loop while verifying unmap.
        del p
        store.release(prefix)
        gc.collect()
        assert store.numa_owned_bytes == 0 and not store.numa_prefix_bytes
        assert all(containing_mapping(address) is None for address in addresses)
        verified['activation_exact_bytes_scales_unchanged_and_idempotent'] = True
        verified['release_unmaps_and_clears_accounting'] = True

        store.activate(expert, prefix)
        addresses = [owner.weight.data_ptr() for owner in (expert.w1, expert.w2, expert.w3)]
        incoming = 'layers.0.ffn.experts.1.w1.weight'
        store.catalog.tensors[incoming] = dict(bytes=16384)
        store.cap_bytes = store.stored_bytes
        store.ensure([incoming])
        gc.collect()
        assert not store.residents and store.numa_owned_bytes == 0
        assert all(containing_mapping(address) is None for address in addresses)
        verified['resident_eviction_dispatch_releases_copy_storage'] = True

        limited = candidate.NumaResidentStore(directory, copy_limit_bytes=16384,
                                               min_weight_bytes=1, reserve_bytes=0)
        expert = Expert()
        limited.activate(expert, prefix)
        assert limited.stats()['current_expert_groups'][prefix]['status'] == 'partial'
        assert limited.numa_owned_bytes == 16384 and limited.stats()['copy_misses'] == 2
        assert limited.numa_counters['miss_copy_limit'] == 2
        limited.release_all()
        assert limited.numa_owned_bytes == 0
        verified['partial_group_copy_limit_fallback_and_release_all'] = True

        reserved = candidate.NumaResidentStore(directory, copy_limit_bytes=1 << 20,
                                                min_weight_bytes=1, reserve_bytes=100 << 30)
        expert = Expert()
        addresses = [owner.weight.data_ptr() for owner in (expert.w1, expert.w2, expert.w3)]
        with patch.object(candidate, 'available_memory_bytes', lambda: 100 << 30):
            reserved.activate(expert, prefix)
        assert reserved.numa_owned_bytes == 0 and reserved.numa_counters['miss_memory_reserve'] == 3
        assert reserved.stats()['current_expert_groups'][prefix]['status'] == 'uncopied'
        assert addresses == [owner.weight.data_ptr() for owner in (expert.w1, expert.w2, expert.w3)]
        reserved.release_all()
        verified['memory_reserve_fallback_preserves_native_storage'] = True

        common = candidate.NumaResidentStore(directory, copy_limit_bytes=1 << 20,
                                              min_weight_bytes=1, reserve_bytes=0)
        model = torch.nn.Module()
        model.embed = torch.nn.Embedding(32, 128)
        model.dense = torch.nn.Linear(128, 32, bias=False)
        model.alias = torch.nn.Linear(128, 32, bias=False)
        model.alias.weight = model.dense.weight
        model.head = Weight(torch.bfloat16)
        model.register_parameter('hc_attn_fn', torch.nn.Parameter(torch.ones(32, 128)))
        model.register_parameter('unused_scale', torch.nn.Parameter(torch.ones(32, 128)))
        model.experts = torch.nn.ModuleList([Expert()])
        model.meta = torch.nn.Linear(128, 32, bias=False, device='meta')
        model.requires_grad_(False)
        skipped = {name: p.data_ptr() for name, p in model.named_parameters()
                   if name.startswith(('embed.', 'experts.')) or name.endswith('scale')}
        outcome = common.clone_common(model)
        assert outcome['copied_weights'] == 3 and outcome['missed_weights'] == 0
        assert common.numa_common_bytes == common.numa_owned_bytes == 3 * 16384
        assert model.dense.weight is model.alias.weight
        assert model.head.weight.scale is model.head.scale
        assert model.meta.weight.is_meta
        assert all(dict(model.named_parameters())[name].data_ptr() == address for name, address in skipped.items())
        assert common.numa_counters['deduplicated_aliases'] == 1
        assert common.clone_common(model)['eligible_weights'] == 0
        assert common.numa_owned_bytes == 3 * 16384
        common.activate(Expert(), prefix)
        assert common.numa_owned_bytes == 6 * 16384
        common.release_all()
        assert common.numa_owned_bytes == common.numa_common_bytes == 3 * 16384
        report = common.stats()
        json.dumps(report)  # The report must contain no tensor references.
        verified['common_dense_head_hc_copy_embedding_scale_meta_expert_skip'] = True
        verified['common_alias_deduplication_and_idempotence'] = True
        verified['common_and_expert_accounting_share_cap_and_release_independently'] = True
        del model
        gc.collect()
    assert os.sched_getaffinity(0) == {0} and process_policy() == before_policy
    paths = [Path(__file__), BASE / 'goal_numa_store_0910.py', BASE / 'goal_numa_0910.py']
    result = dict(passed=True, component_only=True, full_checkpoint_loaded=False,
                  production_model_copied=False, performance_trial=False,
                  cpu_affinity=[0], torch_threads=1, verified=verified,
                  sample_stats=report,
                  sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    output = BASE / 'results/goal_numa_store_0910/fixture-check.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('sample_stats', 'sha256')}))


if __name__ == '__main__':
    main()
