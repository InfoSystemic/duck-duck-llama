"""Vectorized exact fallback revision with optional warmup-only diagnostics."""
from collections import defaultdict
from pathlib import Path

from deepseek_v41_lattice16_0910 import GroupedLattice16
from deepseek_v41_lattice16_runtime_0910 import Lattice16Runtime

BASE = Path(__file__).resolve().parent


class DiagnosedGroupedLattice16(GroupedLattice16):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.collect_stats = False
        self.block_counts = defaultdict(lambda: dict(tasks=0, eligible_blocks=0, total_blocks=0))

    def apply(self, tasks, output=None):
        tasks = list(tasks)
        if self.collect_stats:
            checked = {}
            for mode, activation, _asc, weight, _scale in tasks:
                if mode != 4:
                    continue
                key = activation.data_ptr(), activation.numel()
                if key not in checked:
                    checked[key] = self.eligible_blocks(activation)
                row = self.block_counts[f'{weight.shape[0]}x{activation.shape[-1]}']
                row['tasks'] += 1
                row['eligible_blocks'] += checked[key]
                row['total_blocks'] += activation.numel() // 32
        return super().apply(tasks, output)


class Lattice16BRuntime(Lattice16Runtime):
    def __init__(self, runtime, cap_bytes=64 << 30):
        super().__init__(runtime, cap_bytes)
        library = BASE / 'results/deepseek-v41-lattice16b-0910/libdeepseek-v41-lattice16b.so'
        self.backend = DiagnosedGroupedLattice16(library, self.pack_cache.get_packed, runtime.native.workers)

    def stats(self):
        return dict(super().stats(), activation_block_diagnostics=dict(self.backend.block_counts),
                    diagnostics_enabled=self.backend.collect_stats)
