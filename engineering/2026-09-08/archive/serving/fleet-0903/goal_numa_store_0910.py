"""Bounded opt-in NUMA copies on top of the existing native ResidentStore.

The caller serializes activation/inference as for ResidentStore. Anonymous copy
accounting is separate from the existing file cache cap. No source bytes change.
"""
from collections import Counter
import ctypes
from dataclasses import asdict
from pathlib import Path

import torch

from deepseek_v41_resident_store_0910 import ResidentStore
from goal_numa_0910 import clone_rows, required_bytes, _move_pages


def available_memory_bytes():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('MemAvailable unavailable')


def first_partition_pages(tensor, placement):
    """Query one actual page in each nonempty partition; never migrate pages."""
    spans = [(node, begin) for node, begin, end in
             zip(placement.nodes, placement.actual_byte_boundaries,
                 placement.actual_byte_boundaries[1:]) if end > begin]
    pages = (ctypes.c_void_p * len(spans))(*(tensor.data_ptr() + begin for _, begin in spans))
    status = (ctypes.c_int * len(spans))()
    result = _move_pages(0, len(spans), pages, None, status, 0)
    if result < 0 or any(value < 0 for value in status):
        return dict(verified=False, errno=ctypes.get_errno(), page_status=list(status), sampled=True)
    values = [dict(expected_node=node, actual_node=actual, byte_offset=begin)
              for (node, begin), actual in zip(spans, status)]
    return dict(verified=all(v['expected_node'] == v['actual_node'] for v in values),
                sampled=True, pages=values)


def storage_key(parameter):
    return parameter.data_ptr(), tuple(parameter.shape), parameter.dtype


class NumaResidentStore(ResidentStore):
    def __init__(self, *args, nodes=(0, 1, 2, 3), copy_limit_bytes=64 << 30,
                 min_weight_bytes=1 << 20, reserve_bytes=100 << 30, **kwargs):
        if any(type(v) is not int or v < 0 for v in (copy_limit_bytes, min_weight_bytes, reserve_bytes)):
            raise ValueError('NUMA memory limits must be nonnegative integers')
        self.numa_nodes = tuple(nodes)
        if not self.numa_nodes or any(type(n) is not int or not 0 <= n < 64 for n in self.numa_nodes):
            raise ValueError('nodes must contain valid NUMA node IDs')
        self.numa_copy_limit_bytes = copy_limit_bytes
        self.numa_min_weight_bytes = min_weight_bytes
        self.numa_reserve_bytes = reserve_bytes
        self.numa_owned_bytes = 0
        self.numa_common_bytes = 0
        self.numa_prefix_bytes = {}
        self.numa_processed_prefixes = set()
        self.numa_common_names = set()
        self.numa_prefix_status = {}
        self.numa_placements = {}
        self.numa_counters = Counter()
        super().__init__(*args, **kwargs)

    def _miss(self, reason, aliases):
        self.numa_counters['copy_misses'] += 1
        self.numa_counters['miss_' + reason] += 1
        self.numa_counters['uncopied_parameter_aliases'] += len(aliases)

    def _copy_groups(self, parameters, bucket):
        """Parameters are temporary references; persisted state is numeric only."""
        groups = {}
        for owner, attr, label, parameter in parameters:
            if parameter.is_meta or parameter.device.type != 'cpu':
                self.numa_counters['skipped_meta_or_non_cpu'] += 1
                continue
            if parameter.ndim != 2 or parameter.numel() * parameter.element_size() < self.numa_min_weight_bytes:
                self.numa_counters['skipped_small_or_nonmatrix'] += 1
                continue
            groups.setdefault(storage_key(parameter), []).append((owner, attr, label, parameter))
        copied = 0
        missed = 0
        copied_bytes = 0
        for aliases in groups.values():
            parameter = aliases[0][3]
            charge = required_bytes(parameter)
            reason = None
            if self.numa_owned_bytes + charge > self.numa_copy_limit_bytes:
                reason = 'copy_limit'
            else:
                try:
                    if available_memory_bytes() < self.numa_reserve_bytes + charge:
                        reason = 'memory_reserve'
                except (OSError, RuntimeError, ValueError):
                    reason = 'memory_unavailable'
            if reason is not None:
                self._miss(reason, aliases)
                missed += 1
                continue
            try:
                cloned = clone_rows(parameter, self.numa_nodes)
            except ValueError:
                self._miss('unsupported_or_unaligned', aliases)
                missed += 1
                continue
            except (OSError, MemoryError):
                self._miss('allocation_or_mbind', aliases)
                missed += 1
                continue
            placement = cloned._goal_numa_0910
            sampled = first_partition_pages(cloned, placement)
            replacement = torch.nn.Parameter(cloned, requires_grad=False)
            if hasattr(parameter, 'scale'):
                replacement.scale = parameter.scale
            for owner, attr, _label, _old in aliases:
                owner._parameters[attr] = replacement
                if attr == 'weight' and getattr(owner, 'scale', None) is not None:
                    replacement.scale = owner.scale
            self.numa_owned_bytes += charge
            copied_bytes += charge
            copied += 1
            self.numa_counters['copied_weights'] += 1
            self.numa_counters['copied_bytes_total'] += charge
            self.numa_counters['deduplicated_aliases'] += len(aliases) - 1
            if not sampled['verified']:
                self.numa_counters['placement_sample_failures'] += 1
            self.numa_placements.setdefault(bucket, []).append(dict(
                names=[v[2] for v in aliases], **asdict(placement), actual_sample=sampled))
        return dict(eligible_weights=len(groups), copied_weights=copied, missed_weights=missed,
                    copied_bytes=copied_bytes)

    def activate(self, module, prefix):
        super().activate(module, prefix)
        if prefix in self.numa_processed_prefixes:
            return
        parameters = [(owner, 'weight', prefix + (local + '.' if local else '') + 'weight', owner._parameters['weight'])
                      for local, owner in module.named_modules()
                      if owner._parameters.get('weight') is not None]
        outcome = self._copy_groups(parameters, prefix)
        self.numa_prefix_bytes[prefix] = outcome['copied_bytes']
        self.numa_processed_prefixes.add(prefix)
        if outcome['eligible_weights'] == 0:
            status = 'no_eligible_weights'
        elif outcome['missed_weights'] == 0:
            status = 'full'
        elif outcome['copied_weights'] == 0:
            status = 'uncopied'
        else:
            status = 'partial'
        self.numa_prefix_status[prefix] = dict(status=status, **outcome)
        self.numa_counters[status + '_expert_groups'] += 1

    def release(self, prefix):
        super().release(prefix)
        released = self.numa_prefix_bytes.pop(prefix, 0)
        self.numa_owned_bytes -= released
        assert self.numa_owned_bytes >= self.numa_common_bytes
        self.numa_processed_prefixes.discard(prefix)
        self.numa_prefix_status.pop(prefix, None)
        self.numa_placements.pop(prefix, None)
        self.numa_counters['released_copy_bytes'] += released
        self.numa_counters['released_expert_groups'] += 1

    def clone_common(self, model):
        """Clone common 2-D matrices once, excluding embedding/expert storage.

        Includes FP32 head and HC projection matrices, and BF16/FP8/FP4 dense
        weights. Scale parameters are retained. Existing tied matrix aliases are
        replaced with one shared copy; embeddings and their tied aliases stay put.
        """
        modules = list(model.named_modules())
        embedding_paths = {name for name, owner in modules
                           if isinstance(owner, torch.nn.Embedding)
                           or 'embedding' in type(owner).__name__.lower()}
        embedding_keys = {storage_key(p) for name, owner in modules
                          if any(name == path or name.startswith(path + '.') for path in embedding_paths)
                          for p in owner._parameters.values() if p is not None and not p.is_meta}
        parameters = []
        for local, owner in modules:
            if 'experts' in local.split('.'):
                continue
            embedding = any(local == path or local.startswith(path + '.') for path in embedding_paths)
            for attr, parameter in owner._parameters.items():
                if parameter is None:
                    continue
                label = (local + '.' if local else '') + attr
                if label in self.numa_common_names:
                    continue
                self.numa_common_names.add(label)
                if embedding or (not parameter.is_meta and storage_key(parameter) in embedding_keys):
                    self.numa_counters['skipped_embedding_parameters'] += 1
                    continue
                if attr == 'scale' or attr.endswith('_scale'):
                    self.numa_counters['skipped_scale_parameters'] += 1
                    continue
                parameters.append((owner, attr, label, parameter))
        outcome = self._copy_groups(parameters, '@common')
        self.numa_common_bytes += outcome['copied_bytes']
        self.numa_counters['common_copy_passes'] += 1
        return outcome

    def stats(self):
        return dict(nodes=list(self.numa_nodes), copy_limit_bytes=self.numa_copy_limit_bytes,
                    min_weight_bytes=self.numa_min_weight_bytes, reserve_bytes=self.numa_reserve_bytes,
                    copied_bytes=self.numa_owned_bytes, common_copied_bytes=self.numa_common_bytes,
                    expert_copied_bytes=sum(self.numa_prefix_bytes.values()),
                    per_prefix_copied_bytes=dict(self.numa_prefix_bytes),
                    current_expert_groups={key: dict(value) for key, value in self.numa_prefix_status.items()},
                    copy_misses=self.numa_counters['copy_misses'], counters=dict(self.numa_counters),
                    placements={key: list(value) for key, value in self.numa_placements.items()},
                    placements_sampled=True)
