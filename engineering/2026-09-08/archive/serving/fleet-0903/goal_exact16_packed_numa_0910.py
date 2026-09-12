"""Opt-in four-node placement of an Exact16Runtime's bounded packed cache.

Install before constructing an external ragged bridge, then give that bridge
``placement.get_packed`` (or the replaced ``candidate.get_packed``). Calls and
eviction must remain serialized, as required by Exact16Runtime itself. Native
weights/scales are never moved. Only newly allocated packed buffers are copied.

Each successful call returns the same PackedFP4Exact16 object owned by the
candidate cache. This wrapper retains numeric placement metadata only, with no
additional tensor or packed-object references. The candidate's data-byte charge
is unchanged; anonymous-mapping page padding has a separate bounded charge.

If strict row boundaries, memory reserve, padding budget, or mbind cannot be
satisfied, return None so the caller's existing whole-call fallback applies.
The candidate's original cached pack is left intact. No affinity, global memory
policy, raw source storage, or other process state is changed.
"""
from dataclasses import asdict
from pathlib import Path

from deepseek_v41_grouped_int16_exact_goal_0910 import PackedFP4Exact16
from goal_numa_0910 import PAGE_SIZE, clone_rows, required_bytes


def mem_available_bytes():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('/proc/meminfo has no MemAvailable field')


class Exact16NumaProvider:
    """Install a serialized provider/eviction wrapper without another cache."""
    def __init__(self, candidate, *, nodes=(0, 1, 2, 3),
                 reserve_bytes=100_000_000_000, padding_cap_bytes=16 << 20):
        nodes = tuple(nodes)
        if len(nodes) != 4 or len(set(nodes)) != 4 or any(type(n) is not int or not 0 <= n < 64 for n in nodes):
            raise ValueError('exact16 NUMA placement requires four distinct node IDs')
        if any(type(x) is not int or x < 0 for x in (reserve_bytes, padding_cap_bytes)):
            raise ValueError('reserve_bytes and padding_cap_bytes must be nonnegative integers')
        if hasattr(candidate, '_goal_exact16_packed_numa_0910'):
            raise RuntimeError('Exact16 NUMA provider is already installed')
        self.candidate = candidate
        self.nodes = nodes
        self.reserve_bytes = reserve_bytes
        self.padding_cap_bytes = padding_cap_bytes
        self.ledger = {}
        self.padding_bytes = self.relocated_data_bytes = 0
        self.relocated_entries = self.relocated_total_bytes = self.reused_entries = 0
        self.evicted_entries = self.provider_fallbacks = 0
        self.fallbacks_by_reason = {}
        self.last_failure = None
        self.last_mem_available_bytes = None
        self.original_get_packed = candidate.get_packed
        self.original_drop_sources = candidate.drop_sources
        self.original_clear = candidate.clear
        self.original_backend_provider = candidate.backend.pack_provider
        self._installed_get = self.get_packed
        self._installed_drop = self.drop_sources
        self._installed_clear = self.clear
        candidate.get_packed = self._installed_get
        candidate.drop_sources = self._installed_drop
        candidate.clear = self._installed_clear
        candidate.backend.pack_provider = self._installed_get
        candidate._goal_exact16_packed_numa_0910 = self

    @staticmethod
    def _key(weight, scale):
        return (weight.data_ptr(), scale.data_ptr(), tuple(weight.shape), tuple(scale.shape))

    def _forget(self, key):
        metadata = self.ledger.pop(key, None)
        if metadata is not None:
            self.padding_bytes -= metadata['padding_bytes']
            self.relocated_data_bytes -= metadata['data_bytes']
            self.evicted_entries += 1
        assert self.padding_bytes >= 0 and self.relocated_data_bytes >= 0

    def _failure(self, reason, **metadata):
        self.fallbacks_by_reason[reason] = self.fallbacks_by_reason.get(reason, 0) + 1
        self.last_failure = dict(reason=reason, **metadata)
        return None

    def get_packed(self, weight, scale):
        key = self._key(weight, scale)
        packed = self.original_get_packed(weight, scale)
        if packed is None:
            self.provider_fallbacks += 1
            return None
        entry = self.candidate.cache.get(key)
        if entry is None or entry[2] is not packed or self._key(entry[0], entry[1]) != key:
            # Exact16Runtime owns all returned sources and packs. Do not silently
            # mutate a pack returned by an unrelated provider or a borrowed cache.
            raise RuntimeError('Exact16Runtime returned a pack outside its owned cache entry')
        if not isinstance(packed, PackedFP4Exact16):
            raise TypeError('Expected the pair-major PackedFP4Exact16 layout')
        metadata = self.ledger.get(key)
        if metadata is not None:
            if (metadata['packed_object_id'] == id(packed)
                    and metadata['packed']['address'] == packed.packed.data_ptr()
                    and metadata['scales']['address'] == packed.scales.data_ptr()):
                self.reused_entries += 1
                return packed
            self._forget(key)
        tiles, blocks = (packed.n + 15) // 16, packed.k // 32
        if (tuple(packed.packed.shape) != (tiles, blocks, 8, 2, 16)
                or tuple(packed.scales.shape) != (tiles, blocks, 16)):
            raise ValueError('Unexpected exact16 packed tensor shapes')
        data_bytes = packed.packed.numel() + packed.scales.numel()
        if packed.storage_bytes != data_bytes:
            raise ValueError('PackedFP4Exact16 storage_bytes differs from actual tensor bytes')
        row_bytes = blocks * 256
        requested = tuple((tiles * i // 4) * row_bytes for i in range(5))
        if any(boundary % PAGE_SIZE for boundary in requested[1:-1]):
            return self._failure('strict_row_boundary', requested_boundaries=requested)
        allocated_bytes = required_bytes(packed.packed) + required_bytes(packed.scales)
        padding_bytes = allocated_bytes - data_bytes
        if self.padding_bytes + padding_bytes > self.padding_cap_bytes:
            return self._failure('padding_cap', needed_padding_bytes=padding_bytes,
                                 current_padding_bytes=self.padding_bytes)
        try:
            available = mem_available_bytes()
        except (OSError, RuntimeError, ValueError) as error:
            return self._failure('mem_available_unreadable', error=str(error)[:256])
        self.last_mem_available_bytes = available
        # Both existing source and packed storage already contribute to current
        # memory usage. Budget the complete temporary replacement allocation.
        if available - allocated_bytes < self.reserve_bytes:
            return self._failure('memory_reserve', available_bytes=available,
                                 needed_bytes=allocated_bytes, reserve_bytes=self.reserve_bytes)
        try:
            new_packed = clone_rows(packed.packed, self.nodes, boundary_policy='strict')
            new_scales = clone_rows(packed.scales, self.nodes, boundary_policy='nearest_page')
        except (OSError, RuntimeError, ValueError, MemoryError) as error:
            return self._failure('clone_failed', error_type=type(error).__name__,
                                 error=str(error)[:256])
        # Prepare metadata before committing either field. No references to the
        # old tensors survive this call; the returned pack keeps its identity.
        metadata = dict(packed_object_id=id(packed), data_bytes=data_bytes,
                        allocated_bytes=allocated_bytes, padding_bytes=padding_bytes,
                        packed=asdict(new_packed._goal_numa_0910),
                        scales=asdict(new_scales._goal_numa_0910))
        packed.packed, packed.scales = new_packed, new_scales
        self.ledger[key] = metadata
        self.padding_bytes += padding_bytes
        self.relocated_data_bytes += data_bytes
        self.relocated_entries += 1
        self.relocated_total_bytes += allocated_bytes
        assert packed.storage_bytes == data_bytes
        assert self.padding_bytes <= self.padding_cap_bytes
        return packed

    def drop_sources(self, pointers):
        pointers = set(pointers)
        keys = set()
        for pointer in pointers:
            keys.update(self.candidate.source_keys.get(pointer, ()))
        try:
            return self.original_drop_sources(pointers)
        finally:
            for key in keys:
                if key not in self.candidate.cache:
                    self._forget(key)

    def clear(self):
        try:
            return self.original_clear()
        finally:
            # The normal provider clear removes all entries. If a future clear
            # partially fails, preserve charges for any still-owned allocations.
            for key in tuple(self.ledger):
                if key not in self.candidate.cache:
                    self._forget(key)

    def metrics(self):
        return dict(nodes=list(self.nodes), reserve_bytes=self.reserve_bytes,
                    padding_cap_bytes=self.padding_cap_bytes, padding_bytes=self.padding_bytes,
                    relocated_cache_entries=len(self.ledger), relocated_data_bytes=self.relocated_data_bytes,
                    relocated_allocated_bytes=self.relocated_data_bytes + self.padding_bytes,
                    candidate_packed_bytes=self.candidate.packed_bytes,
                    candidate_cap_bytes=self.candidate.cap_bytes,
                    relocated_entries=self.relocated_entries, relocated_total_bytes=self.relocated_total_bytes,
                    reused_entries=self.reused_entries, evicted_entries=self.evicted_entries,
                    provider_fallbacks=self.provider_fallbacks,
                    fallbacks_by_reason=dict(self.fallbacks_by_reason), last_failure=self.last_failure,
                    last_mem_available_bytes=self.last_mem_available_bytes)

    def uninstall(self):
        """Clear only the provider's packed cache, then restore its prior methods.

        Clearing prevents relocated mmap padding from becoming unaccounted after
        the wrapper is removed. Raw resident weights and scales remain unchanged.
        """
        candidate = self.candidate
        if (candidate.get_packed is not self._installed_get
                or candidate.drop_sources is not self._installed_drop
                or candidate.clear is not self._installed_clear
                or candidate.backend.pack_provider is not self._installed_get):
            raise RuntimeError('Provider methods changed after NUMA installation')
        self.clear()
        candidate.get_packed = self.original_get_packed
        candidate.drop_sources = self.original_drop_sources
        candidate.clear = self.original_clear
        candidate.backend.pack_provider = self.original_backend_provider
        del candidate._goal_exact16_packed_numa_0910
