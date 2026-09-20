#!/usr/bin/env python3
"""Combine the locked, independently gated pool and CPY candidates."""
from pathlib import Path
import hashlib,json,difflib
D=Path(__file__).resolve().parent
P=D.parent/'glm-pool-0919'
C=D.parent/'glm-copy-0919'
expected={P/'build/libggml-cpu.so.0.22.0':'f504f67f5a0184cce86378ba56fe04e937f62e971cb1f51bd0e4678446996ee0',C/'build/libggml-cpu.so.0.22.0':'5a809c34142cdd76c32bf2fa9da73c5611d72447edee1d7bc0b85aeb01304527'}
for path,want in expected.items():assert hashlib.sha256(path.read_bytes()).hexdigest()==want,path
sources=[P/'ggml-cpu.pool.c',P/'ops.pool.cpp',P/'ops.h',P/'pool-fusion.inc',P/'pool-kernel.inc',P/'cpy-address-probe.inc',C/'cpy-flat.inc']
for parent in [P,C]:
    manifest=json.loads((parent/'build-provenance.json').read_text())
    for path in sources+[P/'test_pool.cpp',C/'test_copy.cpp']:
        if path.parent==parent:assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest[str(path)],path
for name in ['ops.h','pool-fusion.inc','pool-kernel.inc','cpy-address-probe.inc']:(D/name).write_bytes((P/name).read_bytes())
(D/'cpy-flat.inc').write_bytes((C/'cpy-flat.inc').read_bytes())
(D/'ggml-cpu.combo.c').write_bytes((P/'ggml-cpu.pool.c').read_bytes())
ops=(P/'ops.pool.cpp').read_text()
needle='''void ggml_compute_forward_cpy(
        const ggml_compute_params * params,
        ggml_tensor * dst) {
    glm_pool_cpy_address_probe(params, dst);
    ggml_compute_forward_dup(params, dst);
}'''
assert ops.count(needle)==1
ops=ops.replace(needle,'#include "cpy-flat.inc"\n\n'+needle.replace('    ggml_compute_forward_dup(params, dst);','    if (glm_cpy_try_flat(params, dst)) return;\n    ggml_compute_forward_dup(params, dst);'),1)
(D/'ops.combo.cpp').write_text(ops)
(D/'rebuild.sh').write_text((P/'rebuild.sh').read_text().replace('ggml-cpu.pool.c','ggml-cpu.combo.c').replace('ops.pool.cpp','ops.combo.cpp'))
for parent,name in [(P,'test_pool.cpp'),(C,'test_copy.cpp')]:
    (D/name).write_bytes((parent/name).read_bytes());sources.append(parent/name)
(D/'source-provenance.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},indent=2)+'\n')
(D/'pool-parent-delta.patch').write_text(''.join(difflib.unified_diff((P/'ops.pool.cpp').read_text().splitlines(True),ops.splitlines(True),fromfile=str(P/'ops.pool.cpp'),tofile=str(D/'ops.combo.cpp'))))
