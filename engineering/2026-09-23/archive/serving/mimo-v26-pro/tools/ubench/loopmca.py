#!/usr/bin/env python3
# loopmca.py <objdump.s> [min_dpbusd] : for each backward-jump loop whose body holds >= min vpdpbusd, print llvm-mca's
# Cascade Lake estimate (cycles per iteration, bottleneck). The loop body = jump target .. backward jump.
import re, subprocess, sys
lines = open(sys.argv[1]).read().splitlines()
mind = int(sys.argv[2]) if len(sys.argv) > 2 else 8
ins = []
for l in lines:
    m = re.match(r'\s*([0-9a-f]+):\s+(.*)$', l)
    if m: ins.append((int(m.group(1), 16), m.group(2).strip()))
addr_idx = {a: i for i, (a, _) in enumerate(ins)}
for i, (a, t) in enumerate(ins):
    m = re.match(r'(j[a-z]+)\s+([0-9a-f]+)\s*<', t)
    if not m: continue
    tgt = int(m.group(2), 16)
    if tgt >= a or tgt not in addr_idx: continue
    body = [x for _, x in ins[addr_idx[tgt]:i]]
    nd = sum(1 for x in body if x.startswith('vpdpbusd'))
    if nd < mind: continue
    asm = '\n'.join(re.sub(r'\s*<.*>$', '', x) for x in body if not x.startswith('j')) + '\n'
    r = subprocess.run(['llvm-mca-16', '-mtriple=x86_64', '-mcpu=cascadelake', '-iterations=200', '-bottleneck-analysis'],
                       input=asm, capture_output=True, text=True)
    out = r.stdout
    cyc = re.search(r'Total Cycles:\s+(\d+)', out); uops = re.search(r'Total uOps:\s+(\d+)', out)
    ipc = re.search(r'IPC:\s+([\d.]+)', out); rthr = re.search(r'Block RThroughput:\s+([\d.]+)', out)
    bott = re.findall(r'(Resource Pressure|Data Dependencies|Throughput Bottlenecks)[^\n]*\n?([^\n]*)', out)
    c = int(cyc.group(1)) / 200 if cyc else -1
    print(f'loop @{tgt:x}: {len(body)} insns, {nd} dpbusd: {c:.1f} cyc/iter ({nd / c:.2f} dpbusd/cyc), RThroughput {rthr.group(1) if rthr else "?"}, uops/iter {int(uops.group(1)) / 200 if uops else "?"}')
    if r.returncode != 0: print(r.stderr[:500])
    # port pressure summary
    m2 = re.search(r'Resource pressure per iteration:\n(.*)\n(.*)\n', out)
    if m2: print('   ports:', m2.group(1).split()); print('          ', m2.group(2).split())
