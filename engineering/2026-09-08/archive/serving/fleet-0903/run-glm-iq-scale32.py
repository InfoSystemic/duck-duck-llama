#!/usr/bin/env python3
"""Validate shared XXS scales and conditionally test the full GLM model."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess

root = Path(__file__).resolve().parents[2]
base = root / 'serving/fleet-0903'
assert subprocess.check_output(['ps', '-C', 'llama-server', '-o', 'pid='], text=True).split() == ['4005448']
assert (base / 'results/glm-smt-kernel-check-summary.json').exists()
selected = json.loads((base / 'results/glm5n-goal-x16-chunk16-1024-t15/result.json').read_text())
assert selected.get('server_exit') == 0 and not selected.get('error')
engine = root / 'engines/llama.cpp-glm5n-goal-0904'
bindir = engine / 'build-goal/bin'
snapshot = engine / 'q5-compact-p2-bin'
if not snapshot.exists():
    shutil.copytree(bindir, snapshot, symlinks=True)
hashes = {}
for path in bindir.iterdir():
    if path.is_file():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == hashlib.sha256((snapshot / path.name).read_bytes()).hexdigest()
        hashes[path.name] = digest
(base / 'results/glm-q5-p2-binary-snapshot.json').write_text(json.dumps({
    'directory': str(snapshot), 'sha256': hashes,
    'evidence': 'glm5n-goal-q5-compact-p2-1024-t15'}, indent=2) + '\n')

def run(command, label, env=None, timeout=1800):
    log = base / 'results' / (label + '.log')
    with log.open('w') as output:
        rc = subprocess.run(command, env=env, stdout=output, stderr=subprocess.STDOUT, timeout=timeout).returncode
    print(label, rc, flush=True)
    if rc:
        raise SystemExit(rc)
    return log.read_text()

run(['python3', str(base / 'stage-glm-iq-scale32.py'), '--apply'], 'glm-iq-scale32-apply')
run(['cmake', '--build', str(engine / 'build-goal'), '-j', '16', '--target', 'llama-server'], 'glm-iq-scale32-build')
env = {k: v for k, v in os.environ.items()
       if not k.startswith(('GGML_', 'LLAMA_GRAPH_PHASE', 'LLAMA_MTP_DRAFT_N_FILE', 'OMP_', 'GOMP_', 'REPACK_TEST_'))}
env.update({k: v for k, v in selected['runtime_env'].items()
            if not k.endswith('_FILE') and not k.startswith('GGML_CPU_OP_PROFILE')})
env['LD_LIBRARY_PATH'] = str(bindir)
flags = ['g++', '-O2', '-std=c++17']
flags += ['-I' + str(engine / p) for p in ['include', 'ggml/include', 'ggml/src', 'ggml/src/ggml-cpu']]
links = ['-L' + str(bindir), '-lggml', '-lggml-cpu', '-lggml-base', '-ldl', '-pthread']
test = base / 'glm-iq-scale32-check'
run(flags + [str(base / 'iq2-repack-check.cpp')] + links + ['-o', str(test)], 'glm-iq-scale32-check-build')
coverage = []
for mode in ['standard', 'padded', 'capped']:
    hashes = []
    for enabled in [False, True]:
        case_env = dict(env, GGML_CPU_NUMA_DEVICES='0', GGML_CPU_IQ_R16_SCALE32=str(int(enabled)),
                        REPACK_TEST_SMALL_BATCHES='1', REPACK_TEST_WORK_SHARING='1',
                        REPACK_TEST_CLAMP='1', REPACK_TEST_PERSISTENT_POOL='1')
        if mode == 'padded':
            case_env.update(REPACK_TEST_PADDED='1', REPACK_TEST_DOWN='1')
        if mode == 'capped':
            case_env['GGML_CPU_MOE_SINGLE_TOKEN_THREADS'] = '2'
        log = run(['numactl', '--physcpubind=0-15', '--membind=0', str(test), 'iq-r16'],
                  f'glm-iq-scale32-{mode}-{int(enabled)}', case_env, 900)
        hashes.append(re.findall(r'hash=([0-9a-f]+)', log))
    assert len(hashes[0]) == 270 and hashes[0] == hashes[1], 'Shared scales changed numerical output'
    coverage.append(dict(mode=mode, cases=270, output_hashes_identical=True))
records = []
for padded in [False, True]:
    hashes = []
    for index, enabled in enumerate([False, True, True, False]):
        case_env = dict(env, GGML_CPU_NUMA_DEVICES='0', GGML_CPU_IQ_R16_SCALE32=str(int(enabled)),
                        REPACK_TEST_WORK_SHARING='1', REPACK_TEST_IQ_REAL_SHAPES='1',
                        REPACK_TEST_CLAMP='1', REPACK_TEST_DOWN='1', REPACK_TEST_PERSISTENT_POOL='1',
                        REPACK_TEST_PIN_POOL='1', REPACK_TEST_THREADS='15', REPACK_TEST_REPEATS='20')
        if padded:
            case_env['REPACK_TEST_PADDED'] = '1'
        log = run(['numactl', '--physcpubind=0-14', '--membind=0', str(test), 'iq-r16'],
                  f'glm-iq-scale32-real-{int(padded)}-{index}-{int(enabled)}', case_env, 900)
        hashes.append(re.findall(r'hash=([0-9a-f]+)', log))
        for line in log.splitlines():
            if line.startswith('PASS '):
                v = dict(re.findall(r'(\w+)=([^ ]+)', line))
                records.append(dict(padded=padded, enabled=enabled, index=index, type=v['type'],
                                    k=int(v['k']), rows=int(v['rows']), tokens=int(v['tokens']),
                                    fused=int(v['fused']), ms=float(v['packed_ms'])))
    assert len(hashes[0]) == 36 and all(h == hashes[0] for h in hashes), 'Real-shape output mismatch'
    coverage.append(dict(mode='real-padded' if padded else 'real-standard', cases=36, output_hashes_identical=True))
keys = {(r['padded'], r['type'], r['k'], r['rows'], r['tokens'], r['fused']) for r in records}
ratios = []
for key in sorted(keys):
    subset = [r for r in records if (r['padded'], r['type'], r['k'], r['rows'], r['tokens'], r['fused']) == key]
    off = statistics.median(r['ms'] for r in subset if not r['enabled'])
    on = statistics.median(r['ms'] for r in subset if r['enabled'])
    ratios.append(dict(padded=key[0], type=key[1], k=key[2], rows=key[3], tokens=key[4], fused=key[5], off_ms=off, on_ms=on, ratio=off/on))
primary = [r for r in ratios if r['tokens'] == 3 and (
    (r['type'] == 'iq2_xxs' and r['k'] == 4096 and r['fused'] == 1) or
    (r['type'] == 'iq3_xxs' and r['k'] == 512 and r['fused'] == 0))]
assert len(primary) == 4
gain = statistics.median(r['ratio'] for r in primary)
(base / 'results/glm-iq-scale32-check-summary.json').write_text(json.dumps({
    'coverage': coverage, 'records': records, 'ratios': ratios,
    'primary_three_token_ratios': primary, 'primary_median_ratio': gain,
    'selected_reference': selected['config']['label']}, indent=2) + '\n')
print('Primary three-token old/new median ratio:', gain, flush=True)
if gain < 1.03:
    print('No sufficient focused gain to justify a full-model run.', flush=True)
    raise SystemExit(0)
env['GGML_CPU_IQ_R16_SCALE32'] = '1'
cfg = dict(selected['config'])
draft = json.loads((base / 'results/glm-mtp-q8-persistent.json').read_text())
cfg.update(label='glm5n-goal-iq-scale32-1024-t15', port=18135, mtp=draft['destination'])
args = ['python3', str(base / 'qwen-goal-case.py'), cfg.pop('label')]
for key, value in cfg.items():
    if isinstance(value, bool):
        if value:
            args.append('--' + key.replace('_', '-'))
    elif value is not None:
        args.extend(['--' + key.replace('_', '-'), str(value)])
raise SystemExit(subprocess.call(args, env=env))
