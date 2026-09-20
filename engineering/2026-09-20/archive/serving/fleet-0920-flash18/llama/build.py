#!/usr/bin/env python3
"""build.py <llama-context source> <outdir> -- private libllama: the deployed mtp-kv-only link line with llama-context.cpp.o rebuilt from <source>."""
import json, subprocess, sys, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
r = json.loads((HERE/'recipe.json').read_text())
src, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(); out.mkdir(parents=True, exist_ok=True)
pin = ['nice', '-n', '10', 'taskset', '-c', '15,31,47,63,79,95,111,127']
comp = r['compile'][:]
comp[comp.index('-o') + 1] = str(out/'llama-context.cpp.o'); comp[-1] = str(src)
if str(src) != r['source']:
    # a copy outside src/: keep the quote includes and the __FILE__ spelling of the engine tree
    comp[1:1] = ['-iquote' + str(Path(r['source']).parent), f'-fmacro-prefix-map={src}={r["source"]}']
subprocess.run(pin + comp, cwd=r['compile_cwd'], check=True)
link = r['link'][:]
link[link.index(r['context_object'])] = str(out/'llama-context.cpp.o')
link[link.index('-o') + 1] = str(out/'libllama.so.0.3.0')
subprocess.run(pin + link, cwd=r['link_cwd'], check=True)
for n in ('libllama.so', 'libllama.so.0'):
    p = out/n
    if not p.is_symlink(): p.symlink_to('libllama.so.0.3.0')
print(hashlib.sha256((out/'libllama.so.0.3.0').read_bytes()).hexdigest(), '<- built;', r['parent_sha256'], '<- deployed parent')
