#!/usr/bin/env python3
"""build_glm5next.py <glm5next source> <outdir> -- private libllama: the deployed mtp-kv-only link line with models/glm5next.cpp.o rebuilt.
Gate: `build_glm5next.py --gate` recompiles the deployed candidate source and must reproduce its object and the deployed library hash."""
import json, subprocess, sys, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
K = Path('/home/user/InfoSystemic/AI-Server/serving/fleet-0912-ctx/glm-mtp-kv-only-0919')
rec = json.loads((K/'candidate-build.json').read_text()); r = json.loads((HERE/'recipe.json').read_text())
gate = sys.argv[1] == '--gate'
src = (K/'candidate/glm5next.cpp') if gate else Path(sys.argv[1]).resolve()
out = (HERE/'gate-glm5next') if gate else Path(sys.argv[2]).resolve(); out.mkdir(parents=True, exist_ok=True)
pin = ['nice', '-n', '10', 'taskset', '-c', '15,31,47,63,79,95,111,127']
comp = rec['compile'][:]
obj = str(K/'candidate/glm5next.cpp.o'); csrc = str(K/'candidate/glm5next.cpp')
comp[comp.index('-o') + 1] = str(out/'glm5next.cpp.o'); comp[-1] = str(src)
comp = [x.replace(f'-fmacro-prefix-map={csrc}=', f'-fmacro-prefix-map={src}=') for x in comp]
cwd = rec.get('compile_cwd') or r['compile_cwd']
subprocess.run(pin + comp, cwd=cwd, check=True)
if gate:
    same = (out/'glm5next.cpp.o').read_bytes() == Path(obj).read_bytes()
    print('LINEAGE GATE object byte-identical:', same)
link = r['link'][:]
link[link.index(obj)] = str(out/'glm5next.cpp.o')
link[link.index('-o') + 1] = str(out/'libllama.so.0.3.0')
subprocess.run(pin + link, cwd=r['link_cwd'], check=True)
for n in ('libllama.so', 'libllama.so.0'):
    p = out/n
    if not p.is_symlink(): p.symlink_to('libllama.so.0.3.0')
h = hashlib.sha256((out/'libllama.so.0.3.0').read_bytes()).hexdigest()
print(h, '<- built;', r['parent_sha256'], '<- deployed parent', '(IDENTICAL)' if h == r['parent_sha256'] else '')
