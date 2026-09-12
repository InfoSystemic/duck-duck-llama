"""Bounded exact16 packed cache used only by the existing decode grouped MoE.

``candidate = install(runtime, optimizations)`` changes grouped.backend only.
The original FP4/FP8 module functions, including the prefill path, remain intact.
``candidate.configure({'exact16': True, 'timing': True})`` selects the candidate;
False restores the prior grouped backend. Eviction drops packed/source references
before ResidentStore.release unmaps the original checkpoint parameters.
"""
from pathlib import Path
import time
import types

from deepseek_v41_grouped_int16_exact_goal_0910 import GroupedExact16Gemm


class Exact16Runtime:
    def __init__(self, runtime, optimizations=None, cap_bytes=64 << 30, library=None):
        if type(cap_bytes) is not int or cap_bytes < 0:
            raise ValueError('cap_bytes must be a nonnegative integer')
        if getattr(getattr(runtime, '_goal_vnni_runtime_0910', None), 'enabled', False):
            raise RuntimeError('Disable the changed-reduction VNNI adapter before installing exact16')
        self.runtime = runtime
        self.grouped = optimizations.grouped if optimizations is not None else runtime._goal_grouped_moe_0910
        self.original_backend = self.grouped.backend
        self.original_grouped_enabled = self.grouped.enabled
        self.original_release = runtime.store.release
        self.cap_bytes = cap_bytes
        self.packed_bytes = 0
        self.cache = {}
        self.source_keys = {}
        self.cache_hits = self.cache_misses = self.pack_count = self.cap_fallbacks = self.dropped_entries = 0
        self.pack_seconds = self.pack_native_seconds = self.grouped_seconds = self.grouped_native_seconds = 0.0
        self.pack_native_calls = self.grouped_native_calls = self.grouped_calls = 0
        self.record_timing = True
        self._enabled = False
        if library is None:
            library = (Path(__file__).resolve().parent / 'results/deepseek-v41-grouped-int16-exact-goal-0910'
                       / 'libdeepseek-v41-grouped-int16-exact.so')
        self.backend = GroupedExact16Gemm(library, self.get_packed, runtime.native.workers, self.original_backend)
        original_pack_fn, original_fn, original_apply = self.backend.pack_fn, self.backend.fn, self.backend.apply

        def pack_fn(*args):
            start = time.perf_counter() if self.record_timing else 0
            try:
                return original_pack_fn(*args)
            finally:
                self.pack_native_calls += 1
                if self.record_timing:
                    self.pack_native_seconds += time.perf_counter() - start

        def native_fn(*args):
            start = time.perf_counter() if self.record_timing else 0
            try:
                return original_fn(*args)
            finally:
                self.grouped_native_calls += 1
                if self.record_timing:
                    self.grouped_native_seconds += time.perf_counter() - start

        def apply(*args, **kwargs):
            start = time.perf_counter() if self.record_timing else 0
            try:
                return original_apply(*args, **kwargs)
            finally:
                self.grouped_calls += 1
                if self.record_timing:
                    self.grouped_seconds += time.perf_counter() - start

        self.backend.pack_fn, self.backend.fn, self.backend.apply = pack_fn, native_fn, apply

        def release(store, prefix):
            expert = store.residents.get(prefix)
            if expert is not None:
                self.drop_sources({p.data_ptr() for p in expert.parameters() if not p.is_meta})
            return self.original_release(prefix)

        runtime.store.release = types.MethodType(release, runtime.store)
        self.enabled = True

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, enabled):
        self._enabled = bool(enabled)
        self.grouped.backend = self.backend if self._enabled else self.original_backend
        self.grouped.enabled = True if self._enabled else self.original_grouped_enabled

    def configure(self, config):
        self.record_timing = bool(config.get('timing', True))
        self.enabled = bool(config.get('exact16', False))

    @staticmethod
    def required_bytes(weight):
        n, half_k = weight.shape
        return ((n + 15) // 16) * (half_k // 16) * (256 + 16)

    def get_packed(self, weight, scale):
        key = (weight.data_ptr(), scale.data_ptr(), tuple(weight.shape), tuple(scale.shape))
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key][2]
        self.cache_misses += 1
        if self.packed_bytes + self.required_bytes(weight) > self.cap_bytes:
            self.cap_fallbacks += 1
            return None
        self.backend.workers = self.runtime.native.workers
        start = time.perf_counter() if self.record_timing else 0
        try:
            packed = self.backend.pack(weight, scale)
        finally:
            if self.record_timing:
                self.pack_seconds += time.perf_counter() - start
        assert packed.storage_bytes == self.required_bytes(weight)
        self.cache[key] = (weight, scale, packed)
        self.packed_bytes += packed.storage_bytes
        self.pack_count += 1
        for pointer in set(key[:2]):
            self.source_keys.setdefault(pointer, set()).add(key)
        assert self.packed_bytes <= self.cap_bytes
        return packed

    def drop_sources(self, pointers):
        keys = set()
        for pointer in pointers:
            keys.update(self.source_keys.get(pointer, ()))
        for key in keys:
            self.packed_bytes -= self.cache.pop(key)[2].storage_bytes
            for pointer in set(key[:2]):
                self.source_keys[pointer].discard(key)
                if not self.source_keys[pointer]:
                    del self.source_keys[pointer]
            self.dropped_entries += 1
        assert self.packed_bytes >= 0

    def clear(self):
        self.cache.clear()
        self.source_keys.clear()
        self.packed_bytes = 0

    def metrics(self):
        names = ['packed_bytes', 'cache_hits', 'cache_misses', 'pack_count', 'cap_fallbacks', 'dropped_entries',
                 'pack_seconds', 'pack_native_seconds', 'grouped_seconds', 'grouped_native_seconds',
                 'pack_native_calls', 'grouped_native_calls', 'grouped_calls']
        result = {name: getattr(self, name) for name in names}
        result.update(packed_entries=len(self.cache), eligible_blocks=self.backend.eligible_blocks,
                      fallback_blocks=self.backend.fallback_blocks, cap_bytes=self.cap_bytes,
                      whole_call_fallbacks=self.backend.fallback_calls)
        return result

    def uninstall(self):
        self.enabled = False
        self.clear()
        self.runtime.store.release = self.original_release
        del self.runtime._goal_exact16_runtime_0910


def install(runtime, optimizations=None, cap_bytes=64 << 30, library=None):
    if hasattr(runtime, '_goal_exact16_runtime_0910'):
        raise RuntimeError('Exact16 runtime integration is already installed')
    candidate = Exact16Runtime(runtime, optimizations, cap_bytes, library)
    runtime._goal_exact16_runtime_0910 = candidate
    return candidate
