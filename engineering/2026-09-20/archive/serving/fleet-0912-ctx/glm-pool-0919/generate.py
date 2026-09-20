#!/usr/bin/env python3
"""Reproducible private delta over the actual production unary-lib sources."""
from pathlib import Path
import hashlib
import json

D = Path(__file__).resolve().parent
P = D.parent.parent / 'fleet-0911' / 'parallel-unary-0911'
engine = Path('/home/user/InfoSystemic/AI-Server/engines/llama.cpp-glm5n-goal-0904')
header = D.parent.parent / 'fleet-0903/results/glm-flash-q8-pool-0908c/private-cpu/ops.h'
sources = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
           [P/'ggml-cpu.patched.c', P/'ops.patched.cpp', header]}
c = (P/'ggml-cpu.patched.c').read_text()
c = c.replace('static bool ggml_cpu_softmax_pool_fusion = false;',
              'static bool ggml_cpu_softmax_pool_fusion = false;\nstatic bool ggml_cpu_glm_pool_fusion = false;\nstatic bool ggml_cpu_glm_pool_probe = false;\nstatic atomic_int ggml_cpu_glm_pool_probe_left = 16;', 1)
c = c.replace('static int ggml_cpu_try_fuse_ops(', '#include "pool-fusion.inc"\n\nstatic int ggml_cpu_try_fuse_ops(', 1)
needle = '    struct ggml_tensor * node = cgraph->nodes[node_n];\n\n'
prefix, body = c.split('static int ggml_cpu_try_fuse_ops(', 1)
assert body.count(needle) >= 1
body = body.replace(needle, needle + '''    if (ggml_cpu_glm_pool_fusion && node->op == GGML_OP_GET_ROWS) {
        const int skipped = ggml_cpu_try_glm_pool_fusion(cgraph, node_n, params);
        if (skipped) return skipped;
    }

''', 1)
c = prefix + 'static int ggml_cpu_try_fuse_ops(' + body
needle = '            ggml_cpu_softmax_pool_fusion = env != NULL && atoi(env) == 1;'
assert c.count(needle) == 1
c = c.replace(needle, needle + '\n            env = getenv("GGML_CPU_GLM_POOL_FUSION");\n            ggml_cpu_glm_pool_fusion = env != NULL && atoi(env) == 1;\n            env = getenv("GGML_CPU_GLM_POOL_PROBE");\n            ggml_cpu_glm_pool_probe = env != NULL && atoi(env) == 1;', 1)
needle = '                n_tasks = 1;\n                if (node->src[0]->type == GGML_TYPE_F32 && ggml_cpu_parallel_copy_enabled()) {'
assert c.count(needle) == 1
c = c.replace(needle, '''                n_tasks = 1;
                if (ggml_cpu_glm_pool_fusion && node->src[0]->type == GGML_TYPE_F16 &&
                        strstr(node->name, "indexer_pool_members") != NULL) {
                    n_tasks = n_threads;
                }
                if (node->src[0]->type == GGML_TYPE_F32 && ggml_cpu_parallel_copy_enabled()) {''', 1)
(D/'ggml-cpu.pool.c').write_text(c)
ops = (P/'ops.patched.cpp').read_text()
ops = ops.replace('// ggml_compute_forward_soft_max\n', '#include "pool-kernel.inc"\n\n// ggml_compute_forward_soft_max\n', 1)
needle = """void ggml_compute_forward_cpy(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_dup(params, dst);
}"""
assert ops.count(needle) == 1
ops = ops.replace(needle, '#include "cpy-address-probe.inc"\n\n' + needle.replace(
    '    ggml_compute_forward_dup(params, dst);',
    '    glm_pool_cpy_address_probe(params, dst);\n    ggml_compute_forward_dup(params, dst);'), 1)
(D/'ops.pool.cpp').write_text(ops)
h = header.read_text()
needle = 'void ggml_compute_forward_softmax_pool_fused('
assert h.count(needle) == 1
h = h.replace(needle, 'void ggml_compute_forward_glm_pool_fused(const struct ggml_compute_params * params, const struct ggml_tensor * members, const struct ggml_tensor * ape, const struct ggml_tensor * probs, const struct ggml_tensor * mul, struct ggml_tensor * dst);\n' + needle, 1)
(D/'ops.h').write_text(h)
build = (P/'rebuild.sh').read_text().replace('ggml-cpu.patched.c', 'ggml-cpu.pool.c').replace('ops.patched.cpp', 'ops.pool.cpp')
(D/'rebuild.sh').write_text(build)
(D/'source-provenance.json').write_text(json.dumps(sources, indent=2) + '\n')
print('Generated private sources:', D)
