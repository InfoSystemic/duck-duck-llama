#!/usr/bin/env python3
"""Prepare an opt-in reuse of ggml's existing atomic graph barrier under OpenMP."""
import argparse
import difflib
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--apply', action='store_true')
a = p.parse_args()
base = Path(__file__).resolve().parent
source = base.parents[1] / 'engines/llama.cpp-glm5n-goal-0904/ggml/src/ggml-cpu/ggml-cpu.c'
old = source.read_text()
if 'GGML_CPU_OMP_SIMPLE_BARRIER' in old:
    raise SystemExit('Already applied')
new = old.replace('void ggml_barrier(struct ggml_threadpool * tp) {',
    '#ifdef GGML_USE_OPENMP\nstatic bool ggml_cpu_omp_simple_barrier = false;\n#endif\n\n'
    'void ggml_barrier(struct ggml_threadpool * tp) {', 1)
new = new.replace('#ifdef GGML_USE_OPENMP\n    #pragma omp barrier\n#else\n    int n_passed',
    '#ifdef GGML_USE_OPENMP\n    if (!ggml_cpu_omp_simple_barrier) {\n'
    '        #pragma omp barrier\n        return;\n    }\n#endif\n    int n_passed', 1)
new = new.replace('    atomic_thread_fence(memory_order_seq_cst);\n    #endif\n#endif\n}',
    '    atomic_thread_fence(memory_order_seq_cst);\n    #endif\n}', 1)
anchor = '        {\n            const char * env = getenv("GGML_CPU_DISABLE_FUSION");'
new = new.replace(anchor, '#ifdef GGML_USE_OPENMP\n        {\n'
    '            const char * env = getenv("GGML_CPU_OMP_SIMPLE_BARRIER");\n'
    '            ggml_cpu_omp_simple_barrier = (env != NULL && atoi(env) == 1);\n'
    '        }\n#endif\n' + anchor, 1)
assert new != old and new.count('ggml_cpu_omp_simple_barrier') == 3
relative = 'ggml/src/ggml-cpu/ggml-cpu.c'
(base / 'glm-openmp-simple-barrier.patch').write_text(''.join(difflib.unified_diff(
    old.splitlines(True), new.splitlines(True), 'a/' + relative, 'b/' + relative)))
if a.apply:
    source.with_name(source.name + '.before-goal-openmp-simple-barrier').write_text(old)
    source.write_text(new)
print('Applied' if a.apply else 'Prepared only')
