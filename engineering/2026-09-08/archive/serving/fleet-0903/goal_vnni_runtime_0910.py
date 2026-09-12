"""Bounded opt-in VNNI FP4 integration tied to resident-weight lifetimes.

Call ``candidate = install(runtime)`` after configuring existing optimizations.
``candidate.enabled`` toggles the FP4 path and disables/restores grouped MoE.
FP8 and quantization/HC/attention callables are unchanged. ``clear()`` releases
packed buffers and their source-tensor references; ``uninstall()`` also restores
the FP4 callable and resident-store release method.

The VNNI kernel preserves FP4 bits but changes the dot-product reduction tree;
the kernel's hostile-tie evidence must accompany any numerical parity claim.
"""
from pathlib import Path
import time
import types

import torch

import deepseek_v41_cpu_reference_0910 as cpu
from deepseek_v41_native_vnni_goal_0910 import NativeVnniGemm


class VnniRuntime:
    def __init__(self, runtime, library, cap_bytes=64 << 30):
        if type(cap_bytes) is not int or cap_bytes < 0:
            raise ValueError('cap_bytes must be a nonnegative integer')
        self.runtime = runtime
        self.baseline = runtime.module.fp4_gemm
        self.native = NativeVnniGemm(library, runtime.native.workers)
        self.cap_bytes = cap_bytes
        self.packed_bytes = 0
        self.native_calls = 0
        self.fallback_calls = 0
        self.cap_fallback_calls = 0
        self.pack_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.pack_seconds = 0.0
        self.dropped_entries = 0
        self._sizes = {}
        self._source_keys = {}
        self._enabled = False
        self._grouped = getattr(runtime, '_goal_grouped_moe_0910', None)
        self._prior_grouped_enabled = None
        self._baseline_grouped_backend = getattr(self._grouped, 'backend', None)
        self._grouped_backend = None
        self.original_release = runtime.store.release

        def release(store, prefix):
            expert = store.residents.get(prefix)
            if expert is not None:
                pointers = {p.data_ptr() for p in expert.parameters() if not p.is_meta}
                self.drop_sources(pointers)
            # PackedFP4 cache entries own native source tensors, so clearing
            # them must precede the base method's unmapping/meta replacement.
            return self.original_release(prefix)

        self.release_wrapper = types.MethodType(release, runtime.store)
        runtime.store.release = self.release_wrapper
        runtime.module.fp4_gemm = self.fp4
        self.enabled = True

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, enabled):
        enabled = bool(enabled)
        if enabled:
            if not self._enabled and self._grouped is not None:
                self._prior_grouped_enabled = self._grouped.enabled
            if self._grouped is not None:
                self._grouped.enabled = self._grouped_backend is not None
                if self._grouped_backend is not None:
                    self._grouped.backend = self._grouped_backend
        elif self._enabled and self._grouped is not None:
            self._grouped.enabled = self._prior_grouped_enabled
            if self._grouped_backend is not None:
                self._grouped.backend = self._baseline_grouped_backend
            self._prior_grouped_enabled = None
        self._enabled = enabled

    def set_grouped_backend(self, backend):
        """Enable a grouped VNNI bridge using get_packed as its cache provider."""
        if self._grouped is None:
            raise RuntimeError('The runtime has no grouped MoE integration')
        self._grouped_backend = backend
        if self.enabled:
            self._grouped.backend = backend
            self._grouped.enabled = True

    @staticmethod
    def key(weight, scale):
        return (weight.data_ptr(), scale.data_ptr(), tuple(weight.shape), tuple(scale.shape))

    @staticmethod
    def required_bytes(weight):
        rows, half_k = weight.shape
        tiles, blocks = (rows + 15) // 16, half_k // 16
        return tiles * blocks * (8 * 32 + 16 * 2 + 16)

    def _register(self, key):
        if key in self._sizes or key not in self.native.cache:
            return
        size = self.native.cache[key][2].storage_bytes
        self._sizes[key] = size
        self.packed_bytes += size
        self.pack_count += 1
        for pointer in key[:2]:
            self._source_keys.setdefault(pointer, set()).add(key)
        assert self.packed_bytes <= self.cap_bytes

    def _drop(self, key):
        self.native.cache.pop(key, None)
        self.packed_bytes -= self._sizes.pop(key, 0)
        for pointer in key[:2]:
            keys = self._source_keys.get(pointer)
            if keys is not None:
                keys.discard(key)
                if not keys:
                    del self._source_keys[pointer]
        self.dropped_entries += 1

    def drop_sources(self, pointers):
        keys = set()
        for pointer in pointers:
            keys.update(self._source_keys.get(pointer, ()))
        for key in keys:
            self._drop(key)
        assert self.packed_bytes >= 0

    def clear(self):
        self.native.clear()
        self._sizes.clear()
        self._source_keys.clear()
        self.packed_bytes = 0

    def get_packed(self, weight, scale):
        """Return a cached PackedFP4, or None when the byte cap requires fallback.

        Grouped bridges may borrow this result for one synchronous call but must
        not retain it themselves; resident eviction owns its complete lifetime.
        """
        key = self.key(weight, scale)
        if key in self.native.cache:
            self.cache_hits += 1
            return self.native.cache[key][2]
        self.cache_misses += 1
        if self.packed_bytes + self.required_bytes(weight) > self.cap_bytes:
            self.cap_fallback_calls += 1
            return None
        self.native.workers = self.runtime.native.workers
        started = time.perf_counter()
        try:
            packed = self.native.pack(weight, scale)
        finally:
            self.pack_seconds += time.perf_counter() - started
        self.native.cache[key] = (weight, scale, packed)
        self._register(key)
        return packed

    def fp4(self, a, a_s, b, b_s, scale_dtype=torch.float32, act_block_size=128):
        if not self.enabled or act_block_size != 32:
            self.fallback_calls += 1
            return self.baseline(a, a_s, b, b_s, scale_dtype, act_block_size=act_block_size)
        assert b.ndim == 2 and b.dtype == torch.float4_e2m1fn_x2
        assert b.shape[1] % 16 == 0 and b_s.dtype == torch.float8_e8m0fnu
        assert all(t.device.type == 'cpu' and t.is_contiguous() for t in (a, a_s, b, b_s))
        if self.get_packed(b, b_s) is None:
            self.fallback_calls += 1
            return self.baseline(a, a_s, b, b_s, scale_dtype, act_block_size=act_block_size)
        self.native.workers = self.runtime.native.workers
        output = self.native.apply(4, a, a_s, b, b_s)
        cpu.COUNTS['fp4_gemm'] += 1
        self.native_calls += 1
        return output

    def uninstall(self):
        self.enabled = False
        self.clear()
        self.runtime.module.fp4_gemm = self.baseline
        self.runtime.store.release = self.original_release
        del self.runtime._goal_vnni_runtime_0910


def install(runtime, cap_bytes=64 << 30, library=None):
    if hasattr(runtime, '_goal_vnni_runtime_0910'):
        raise RuntimeError('VNNI runtime integration is already installed')
    if library is None:
        library = (Path(__file__).resolve().parent / 'results/deepseek-v41-native-vnni-goal-0910'
                   / 'libdeepseek-v41-native-vnni.so')
    candidate = VnniRuntime(runtime, library, cap_bytes)
    runtime._goal_vnni_runtime_0910 = candidate
    return candidate
