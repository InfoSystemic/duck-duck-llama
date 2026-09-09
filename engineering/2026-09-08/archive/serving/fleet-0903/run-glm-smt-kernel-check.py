#!/usr/bin/env python3
"""Compare one and two hardware threads per core in isolated matrix fixtures."""
import argparse
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('--reference', default='glm5n-goal-x16-chunk16-1024-t15')
p.add_argument('--binary-dir', default='validated-chunk16-bin')
a = p.parse_args()
base = Path(__file__).resolve().parent
engine = base.parents[1] / 'engines/llama.cpp-glm5n-goal-0904'
bindir = engine / a.binary_dir
result = json.loads((base / 'results' / a.reference / 'result.json').read_text())
assert result.get('server_exit') == 0 and not result.get('error')
pids = subprocess.check_output(['ps', '-C', 'llama-server', '-o', 'pid='], text=True).split()
assert pids == ['4005448'], 'Full Flash tests must exit before these focused tests'
for cpu in range(15):
    siblings = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list').read_text().strip()
    assert siblings == f'{cpu},{cpu+64}', (cpu, siblings)
env = {k: v for k, v in os.environ.items()
       if not k.startswith(('GGML_', 'LLAMA_GRAPH_PHASE', 'LLAMA_MTP_DRAFT_N_FILE', 'OMP_', 'GOMP_', 'REPACK_TEST_'))}
env.update({k: v for k, v in result['runtime_env'].items()
            if not k.endswith('_FILE') and not k.startswith('GGML_CPU_OP_PROFILE')})
env.update(LD_LIBRARY_PATH=str(bindir), GGML_CPU_NUMA_DEVICES='0',
           REPACK_TEST_SMALL_BATCHES='1', REPACK_TEST_PERSISTENT_POOL='1',
           REPACK_TEST_PIN_POOL='1', REPACK_TEST_REPEATS='20')

def run(command, label, env=None):
    log = base / 'results' / (label + '.log')
    with log.open('w') as output:
        rc = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, env=env, timeout=900).returncode
    print(label, rc, flush=True)
    if rc:
        raise SystemExit(rc)
    return log.read_text()

def production_ticks():
    fields = Path('/proc/4005448/stat').read_text().rsplit(')', 1)[1].split()
    return int(fields[11]) + int(fields[12])

flags = ['g++', '-O2', '-std=c++17']
flags += ['-I' + str(engine / d) for d in ['include', 'ggml/include', 'ggml/src', 'ggml/src/ggml-cpu']]
links = ['-L' + str(bindir), '-lggml', '-lggml-cpu', '-lggml-base', '-ldl', '-pthread']
test = base / 'glm-smt-kernel-check'
run(flags + [str(base / 'iq2-repack-check.cpp')] + links + ['-o', str(test)], 'glm-smt-kernel-check-build')
records = []
coverage = []
started = time.monotonic()
before_ticks = production_ticks()
for mode in ['q5-pair', 'iq-r16']:
    hashes = []
    for index, threads in enumerate([15, 30, 30, 15]):
        case_env = dict(env, REPACK_TEST_THREADS=str(threads))
        if mode == 'q5-pair':
            case_env.update(REPACK_TEST_Q5_REAL_SHAPES='1', REPACK_TEST_DENSE_WORK_SHARING='1')
        else:
            case_env.update(REPACK_TEST_WORK_SHARING='1', REPACK_TEST_CLAMP='1')
        cpus = '0-14' if threads == 15 else '0-14,64-78'
        log = run(['numactl', '--physcpubind=' + cpus, '--membind=0', str(test), mode],
                  f'glm-smt-{mode}-{index}-t{threads}', case_env)
        output_hashes = re.findall(r'hash=([0-9a-f]+)', log)
        assert len(output_hashes) == (4 if mode == 'q5-pair' else 90)
        hashes.append(output_hashes)
        for line in log.splitlines():
            if not line.startswith('PASS '):
                continue
            fields = dict(re.findall(r'(\w+)=([^ ]+)', line))
            records.append(dict(mode=mode, index=index, threads=threads, type=fields['type'],
                                k=int(fields['k']), rows=int(fields['rows']), tokens=int(fields['tokens']),
                                fused=int(fields['fused']), ms=float(fields['packed_ms'])))
    assert all(h == hashes[0] for h in hashes), 'Hardware-thread scheduling changed output bytes'
    coverage.append(dict(mode=mode, cases=len(hashes[0]), output_hashes_identical=True))
elapsed = time.monotonic() - started
ticks = production_ticks() - before_ticks
production_cpu = 100 * ticks / os.sysconf('SC_CLK_TCK') / elapsed
keys = {(r['mode'], r['type'], r['k'], r['rows'], r['tokens'], r['fused']) for r in records}
ratios = []
for key in sorted(keys):
    subset = [r for r in records if (r['mode'], r['type'], r['k'], r['rows'], r['tokens'], r['fused']) == key]
    physical = statistics.median(r['ms'] for r in subset if r['threads'] == 15)
    smt = statistics.median(r['ms'] for r in subset if r['threads'] == 30)
    ratios.append(dict(mode=key[0], type=key[1], k=key[2], rows=key[3], tokens=key[4], fused=key[5],
                       physical_ms=physical, smt_ms=smt, ratio=physical/smt))
out = dict(reference=a.reference, binary_dir=str(bindir), coverage=coverage,
           production_cpu_percent=production_cpu, ratios=ratios, records=records,
           note='Focused single-socket matrix timings only; this does not measure full-model throughput.')
(base / 'results/glm-smt-kernel-check-summary.json').write_text(json.dumps(out, indent=2) + '\n')
for mode in ['q5-pair', 'iq-r16']:
    selected = [r['ratio'] for r in ratios if r['mode'] == mode and r['k'] >= 1536 and r['rows'] >= 512]
    print(mode, 'physical/SMT median', statistics.median(selected), flush=True)
print('Production CPU percent:', production_cpu, flush=True)
