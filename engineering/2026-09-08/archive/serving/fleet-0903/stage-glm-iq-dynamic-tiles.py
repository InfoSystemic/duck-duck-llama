#!/usr/bin/env python3
"""Prepare dynamic 32-row work sharing for the existing fused IQ expert path."""
import argparse
import difflib
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--apply', action='store_true')
a = p.parse_args()
base = Path(__file__).resolve().parent
source = base.parents[1] / 'engines/llama.cpp-glm5n-goal-0904/ggml/src/ggml-cpu/repack.cpp'
old = source.read_text()
if 'GGML_CPU_IQ_R16_DYNAMIC_TILES' in old:
    raise SystemExit('Already applied')
start = old.index('    void forward_mul_mat_id_swiglu(\n')
end = old.index('    bool compute_forward_mul_mat_id_weighted_sum(', start)
block = old[start:end]
active_start = block.index('        int active_nth = nth;')
active_end = block.index('        if (ith >= active_nth)', active_start)
active = block[active_start:active_end]
block = block[:active_start] + block[active_end:]
count_start = block.index('        if (ith == 0) {\n            memset(matrix_row_counts')
policy = '''        static const bool dynamic_enabled = [] {
            const char * value = getenv("GGML_CPU_IQ_R16_DYNAMIC_TILES");
            return value && atoi(value) == 1;
        }();
        const bool dynamic = expanded_iq && NB_COLS == 16 && dynamic_enabled && n_experts <= 1024;

'''
block = block[:count_start] + active + policy + block[count_start:]
anchor = '        if (ith == 0) {\n            memset(matrix_row_counts'
block = block.replace(anchor, '        if (ith == 0) {\n'
    '            if (dynamic) ggml_threadpool_chunk_set(params->threadpool, active_nth);\n'
    '            memset(matrix_row_counts', 1)
partition_start = block.index('        int64_t row_start = (ith * n_out) / active_nth;')
expert_start = block.index('        for (int64_t expert = 0; expert < n_experts; ++expert) {', partition_start)
partition = block[partition_start:expert_start]
body_start = expert_start + len('        for (int64_t expert = 0; expert < n_experts; ++expert) {\n')
assert block.endswith('        }\n    }\n\n')
body = block[body_start:-len('        }\n    }\n\n')]
assert body.count('continue;') == 2
body = body.replace('continue;', 'return;')
dispatch = '''        };

        if (dynamic) {
            int active[1024];
            int n_active = 0;
            for (int e = 0; e < n_experts; ++e) {
                if (matrix_row_counts[e] > 0) active[n_active++] = e;
            }
            constexpr int64_t tile = 32;
            const int64_t n_tiles = (n_out + tile - 1) / tile;
            const int64_t n_items = n_active * n_tiles;
            for (int64_t item = ith; item < n_items;
                    item = ggml_threadpool_chunk_add(params->threadpool, 1)) {
                const int64_t row_start = (item % n_tiles) * tile;
                compute_expert(active[item / n_tiles], row_start, std::min(row_start + tile, n_out));
            }
            return;
        }

'''
replacement = block[:partition_start]
replacement += '        const auto compute_expert = [&](int64_t expert, int64_t row_start, int64_t row_end) {\n'
replacement += body + dispatch + partition
replacement += '''        for (int64_t expert = 0; expert < n_experts; ++expert) {
            compute_expert(expert, row_start, row_end);
        }
    }

'''
new = old[:start] + replacement + old[end:]
relative = 'ggml/src/ggml-cpu/repack.cpp'
(base / 'glm-iq-dynamic-tiles.patch').write_text(''.join(difflib.unified_diff(
    old.splitlines(True), new.splitlines(True), 'a/' + relative, 'b/' + relative)))
if a.apply:
    source.with_name(source.name + '.before-goal-iq-dynamic-tiles').write_text(old)
    source.write_text(new)
print('Applied' if a.apply else 'Prepared only')
