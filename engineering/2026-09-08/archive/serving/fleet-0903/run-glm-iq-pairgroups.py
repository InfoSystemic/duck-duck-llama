#!/usr/bin/env python3
"""Validate the two-group XXS kernel, then conditionally test a combined runtime."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess

from inference_contention_guard import InferenceContentionGuard, wait_for_idle

base = Path(__file__).resolve().parent
engine = base.parents[1] / 'engines/llama.cpp-glm5n-goal-0904'
bindir = engine / 'build-goal/bin'
assert not (base / 'results/glm-iq-pairgroups-check-summary.json').exists()
previous = json.loads((base / 'results/glm5n-goal-hugepages-1024-t15/result.json').read_text())
assert previous.get('server_exit') == 0 and not previous.get('error')
selected = json.loads((base / 'results/glm5n-goal-x16-chunk16-1024-t15/result.json').read_text())
assert selected.get('server_exit') == 0 and not selected.get('error')

def snapshot():
    result = {}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            name = (proc / 'comm').read_text().strip()
            if name != 'llama-server' and not name.startswith(('glm-mtp-head', 'qwen-mtp-head')):
                continue
            fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
            args = (proc / 'cmdline').read_bytes().split(b'\0')
            port = args[args.index(b'--port') + 1].decode() if b'--port' in args else None
            result[proc.name] = (int(fields[11]) + int(fields[12]), fields[19], port)
        except (OSError, ValueError, IndexError):
            continue
    return result

wait_for_idle(snapshot, base / 'results/glm-iq-pairgroups-waiting-for-idle.json')
assert list(snapshot()) == ['4005448'], 'Only the idle production inference process may remain'
snapshot_dir = engine / 'iq-scale32-bin'
assert not snapshot_dir.exists()
shutil.copytree(bindir, snapshot_dir, symlinks=True)
hashes = {}
for path in bindir.iterdir():
    if path.is_file():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == hashlib.sha256((snapshot_dir / path.name).read_bytes()).hexdigest()
        hashes[path.name] = digest
(base / 'results/glm-iq-scale32-binary-snapshot.json').write_text(json.dumps({
    'directory': str(snapshot_dir), 'sha256': hashes,
    'evidence': 'glm5n-goal-iq-scale32-clean-1024-t15'}, indent=2) + '\n')

def run(command, label, env=None, timeout=1800):
    log = base / 'results' / (label + '.log')
    assert not log.exists(), label
    with log.open('w') as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, env=env)
        guard = InferenceContentionGuard(process, snapshot, base / 'results' / (label + '-contention.json'))
        guard.start()
        try:
            rc = process.wait(timeout=timeout)
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        finally:
            contention = guard.stop()
    print(label, rc, flush=True)
    assert not contention, 'Other inference interrupted the isolated check'
    if rc:
        raise SystemExit(rc)
    return log.read_text()

run(['python3', str(base / 'stage-glm-iq-pairgroups.py'), '--apply'], 'glm-iq-pairgroups-apply')
run(['cmake', '--build', str(engine / 'build-goal'), '-j', '12', '--target', 'llama-server'], 'glm-iq-pairgroups-build')
env = {k: v for k, v in os.environ.items()
       if not k.startswith(('GGML_', 'LLAMA_GRAPH_PHASE', 'LLAMA_MTP_DRAFT_N_FILE', 'OMP_', 'GOMP_', 'REPACK_TEST_'))}
env.update({k: v for k, v in selected['runtime_env'].items()
            if not k.endswith('_FILE') and not k.startswith('GGML_CPU_OP_PROFILE')})
env.update(LD_LIBRARY_PATH=str(bindir), GGML_CPU_IQ_R16_SCALE32='1', GGML_CPU_NUMA_DEVICES='0')
flags = ['g++', '-O2', '-std=c++17']
flags += ['-I' + str(engine / p) for p in ['include', 'ggml/include', 'ggml/src', 'ggml/src/ggml-cpu']]
links = ['-L' + str(bindir), '-lggml', '-lggml-cpu', '-lggml-base', '-ldl', '-pthread']
test = base / 'glm-iq-pairgroups-check'
run(flags + [str(base / 'iq2-repack-check.cpp')] + links + ['-o', str(test)], 'glm-iq-pairgroups-check-build')
coverage = []
for mode in ['standard', 'padded', 'capped', 'tails', 'tails-padded']:
    hashes = []
    for enabled in [False, True]:
        case_env = dict(env, GGML_CPU_IQ_R16_PAIRGROUPS=str(int(enabled)),
                        REPACK_TEST_SMALL_BATCHES='1', REPACK_TEST_CLAMP='1', REPACK_TEST_PERSISTENT_POOL='1')
        if mode.startswith('tails'):
            case_env.update(REPACK_TEST_IQ_PAIR_TAILS='1', REPACK_TEST_THREADS='1')
        else:
            case_env['REPACK_TEST_WORK_SHARING'] = '1'
        if 'padded' in mode:
            case_env.update(REPACK_TEST_PADDED='1', REPACK_TEST_DOWN='1')
        if mode == 'capped':
            case_env['GGML_CPU_MOE_SINGLE_TOKEN_THREADS'] = '2'
        log = run(['numactl', '--physcpubind=0-15', '--membind=0', str(test), 'iq-r16'],
                  f'glm-iq-pairgroups-{mode}-{int(enabled)}', case_env, 900)
        hashes.append(re.findall(r'hash=([0-9a-f]+)', log))
    count = 108 if mode.startswith('tails') else 270
    assert len(hashes[0]) == count and hashes[0] == hashes[1], 'Two-group output mismatch'
    coverage.append(dict(mode=mode, cases=count, output_hashes_identical=True))
records = []
for padded in [False, True]:
    hashes = []
    for index, enabled in enumerate([False, True, True, False]):
        case_env = dict(env, GGML_CPU_IQ_R16_PAIRGROUPS=str(int(enabled)), REPACK_TEST_WORK_SHARING='1',
                        REPACK_TEST_IQ_REAL_SHAPES='1', REPACK_TEST_CLAMP='1', REPACK_TEST_DOWN='1',
                        REPACK_TEST_PERSISTENT_POOL='1', REPACK_TEST_PIN_POOL='1', REPACK_TEST_THREADS='15',
                        REPACK_TEST_REPEATS='50', REPACK_TEST_TIMING_MEDIAN='1')
        if padded:
            case_env['REPACK_TEST_PADDED'] = '1'
        log = run(['numactl', '--physcpubind=0-14', '--membind=0', str(test), 'iq-r16'],
                  f'glm-iq-pairgroups-real-{int(padded)}-{index}-{int(enabled)}', case_env, 900)
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
(base / 'results/glm-iq-pairgroups-check-summary.json').write_text(json.dumps({
    'coverage': coverage, 'records': records, 'ratios': ratios,
    'primary_three_token_ratios': primary, 'primary_median_ratio': gain,
    'timing_method': 'Median of 50 per-graph durations per case; off/on/on/off order.',
    'note': 'Focused single-socket timings. No full-model speed is implied.'}, indent=2) + '\n')
print('Primary three-token old/new median ratio:', gain, flush=True)
if gain < 1.03:
    print('No sufficient focused gain to justify a full-model run.', flush=True)
    raise SystemExit(0)
env.update(GGML_CPU_IQ_R16_PAIRGROUPS='1', GGML_CPU_X16_Q5_COMPACT_P2='1', GGML_CPU_NUMA_DEVICES='1')
cfg = dict(selected['config'])
draft = json.loads((base / 'results/glm-mtp-q8-persistent.json').read_text())
cfg.update(label='glm5n-goal-iq-pairgroups-combined-1024', port=18137,
           mtp=draft['destination'], binary=str(bindir / 'llama-server'), thread_sweep='15,12,10')
args = ['python3', str(base / 'qwen-goal-case.py'), cfg.pop('label')]
for key, value in cfg.items():
    if isinstance(value, bool):
        if value:
            args.append('--' + key.replace('_', '-'))
    elif value is not None:
        args.extend(['--' + key.replace('_', '-'), str(value)])
raise SystemExit(subprocess.call(args, env=env))
