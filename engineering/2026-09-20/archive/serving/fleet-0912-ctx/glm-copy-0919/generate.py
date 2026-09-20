#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,difflib
D=Path(__file__).resolve().parent
P=D.parent.parent/'fleet-0911/parallel-unary-0911'
(D/'ggml-cpu.copy.c').write_bytes((P/'ggml-cpu.patched.c').read_bytes())
ops=(P/'ops.patched.cpp').read_text()
needle='''void ggml_compute_forward_cpy(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    ggml_compute_forward_dup(params, dst);
}'''
assert ops.count(needle)==1
replacement='#include "cpy-flat.inc"\n\n'+needle.replace('    ggml_compute_forward_dup(params, dst);','    if (glm_cpy_try_flat(params, dst)) return;\n    ggml_compute_forward_dup(params, dst);')
ops=ops.replace(needle,replacement,1)
(D/'ops.copy.cpp').write_text(ops)
build=(P/'rebuild.sh').read_text().replace('ggml-cpu.patched.c','ggml-cpu.copy.c').replace('ops.patched.cpp','ops.copy.cpp')
(D/'rebuild.sh').write_text(build)
(D/'production-parent-delta.patch').write_text(''.join(difflib.unified_diff((P/'ops.patched.cpp').read_text().splitlines(True),ops.splitlines(True),fromfile=str(P/'ops.patched.cpp'),tofile=str(D/'ops.copy.cpp'))))
(D/'source-provenance.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [P/'ggml-cpu.patched.c',P/'ops.patched.cpp']},indent=2)+'\n')
