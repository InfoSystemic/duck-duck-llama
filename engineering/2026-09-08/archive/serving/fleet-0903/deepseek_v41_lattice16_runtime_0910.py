"""Decode-only exact grouped kernel; keep all prefill arithmetic unchanged.

Reuses the existing bounded pack cache and resident-eviction hook. Its earlier
standalone VNNI GEMM is always disabled. Only the grouped backend is replaced.
"""
from pathlib import Path

from deepseek_v41_lattice16_0910 import GroupedLattice16
from goal_vnni_runtime_0910 import install as install_pack_cache

BASE = Path(__file__).resolve().parent


class Lattice16Runtime:
    def __init__(self, runtime, cap_bytes=64 << 30):
        self.runtime = runtime
        self.grouped = runtime._goal_grouped_moe_0910
        self.original_backend = self.grouped.backend
        self.pack_cache = install_pack_cache(runtime, cap_bytes=cap_bytes)
        self.pack_cache.enabled = False
        library = BASE / 'results/deepseek-v41-lattice16-0910/libdeepseek-v41-lattice16.so'
        self.backend = GroupedLattice16(library, self.pack_cache.get_packed, runtime.native.workers)
        self.enabled = False

    def configure(self, enabled):
        assert not self.pack_cache.enabled
        self.grouped.backend = self.backend if enabled else self.original_backend
        self.enabled = bool(enabled)

    def stats(self):
        return {k: getattr(self.pack_cache, k) for k in ('packed_bytes', 'cache_hits', 'cache_misses',
            'pack_count', 'pack_seconds', 'cap_fallback_calls', 'dropped_entries')}

    def uninstall(self):
        self.configure(False)
        self.pack_cache.uninstall()
