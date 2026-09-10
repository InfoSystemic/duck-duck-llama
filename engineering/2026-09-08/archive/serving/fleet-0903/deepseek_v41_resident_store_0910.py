"""Retain native expert mappings while their files remain in the owned cache.

The caller serializes inference and holds the fleet lifecycle lock. Published
arithmetic, tensor bytes, first-load validation, and eviction policy are unchanged.
No second copy of the native expert weights is created.
"""
import types
import torch
from deepseek_v41_checkpoint_0910 import load_parameters
from deepseek_v41_serving_store_0910 import ServingStore
from run_deepseek_v41_checkpoint_0910b import bind as demand_bind


class ResidentStore(ServingStore):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.residents = {}
        self.resident_names = set()
        self.resident_enabled = True

    def release(self, prefix):
        module = self.residents.pop(prefix)
        for sub in module.modules():
            for name, parameter in list(sub._parameters.items()):
                if parameter is not None:
                    sub._parameters[name] = torch.nn.Parameter(
                        torch.empty(parameter.shape, dtype=parameter.dtype, device='meta'), requires_grad=False)
            if getattr(sub, 'scale', None) is not None:
                sub.weight.scale = sub.scale
        self.resident_names.difference_update(prefix + name for name, _ in module.named_parameters())

    def release_all(self):
        for prefix in list(self.residents):
            self.release(prefix)
        assert not self.resident_names

    def activate(self, module, prefix):
        if prefix in self.residents:
            assert self.residents[prefix] is module
            return
        load_parameters(module, prefix, self)
        self.residents[prefix] = module
        self.resident_names.update(prefix + name for name, _ in module.named_parameters())

    def ensure(self, names):
        names = list(dict.fromkeys(names))
        # Already mapped parameters were verified at activation. Touch their LRU
        # entries without repeatedly stat'ing the same six files on every token.
        if names and all(name in self.resident_names for name in names):
            self.tick += 1
            self.touched.update({name: self.tick for name in names})
            return
        amount = sum(self.catalog.tensors[name]['bytes'] for name in names
                     if name not in self.records and name not in self.existing_projection)
        previous = set(self.records) if self.stored_bytes + amount > self.cap_bytes else None
        try:
            super().ensure(names)
        finally:
            if previous is not None:
                # Parent eviction completes before fetching new tensors. Even a
                # failed fetch must release mappings of the evicted expert files.
                removed = previous - self.records.keys()
                prefixes = {name.rsplit('.', 2)[0] + '.' for name in removed if '.ffn.experts.' in name}
                for prefix in prefixes & self.residents.keys():
                    self.release(prefix)
                if (self.root / 'eviction.json').exists():
                    # A journal failure is fatal for this store; leave no live
                    # expert mappings behind while a fresh process recovers it.
                    self.release_all()


def bind(model, store, progress=print, dry_run=False):
    result = demand_bind(model, store, progress, dry_run)
    if not dry_run:
        for layer in model.layers:
            for index, expert in enumerate(layer.ffn.experts):
                prefix = f'layers.{layer.layer_id}.ffn.experts.{index}.'
                demand = expert.forward
                original = type(expert).forward

                def forward(this, x, weights=None, _prefix=prefix, _demand=demand, _original=original):
                    if not store.resident_enabled:
                        assert not store.residents
                        return _demand(x, weights)
                    store.activate(this, _prefix)
                    return _original(this, x, weights)

                expert.forward = types.MethodType(forward, expert)
        result['retain_cached_expert_mappings'] = True
    return result
