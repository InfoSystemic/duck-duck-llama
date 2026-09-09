#!/usr/bin/env python3
"""Validate a smaller Q5 register footprint after the active full model exits."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import time

root = Path(__file__).resolve().parents[2]
base = root / 'serving/fleet-0903'
previous = base / 'results/glm5n-goal-single16k-1024-t15'
while True:
    try:
        finished = json.loads((previous / 'result.json').read_text())
        if 'server_exit' in finished:
            assert finished['server_exit'] == 0 and not finished.get('error')
            break
    except (FileNotFoundError, ValueError):
        pass
    time.sleep(3)

def run(command, label, env=None, timeout=1800):
    log = base / 'results' / (label + '.log')
    with log.open('w') as output:
        result = subprocess.run(command, env=env, stdout=output, stderr=subprocess.STDOUT, timeout=timeout)
    print(label, result.returncode, flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
    return log.read_text()

run(['python3', str(base / 'validate-glm-full-reference.py'), finished['config']['label']], 'glm-single16k-reference-check')
def score(result):
    checks = [c for c in result['checks'] if c['expected'] is None and c['reasoning_budget_tokens'] is None]
    if any(c['throughput_contended'] for c in checks):
        return 0
    return min(c['response']['timings']['predicted_per_second'] for c in checks)
references = [finished, json.loads((base / 'results/glm5n-goal-x16-chunk16-1024-t15/result.json').read_text())]
selected = max(references, key=score)
if score(selected) >= 20:
    print('The runtime already meets the target; skip this candidate.', flush=True)
    raise SystemExit(0)

engine = root / 'engines/llama.cpp-glm5n-goal-0904'
bindir = engine / 'build-goal/bin'
snapshot = engine / 'validated-chunk16-bin'
if not snapshot.exists():
    shutil.copytree(bindir, snapshot, symlinks=True)
hashes = {}
for path in bindir.iterdir():
    if path.is_file():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == hashlib.sha256((snapshot / path.name).read_bytes()).hexdigest()
        hashes[path.name] = digest
(base / 'results/glm-chunk16-binary-snapshot.json').write_text(json.dumps({
    'directory': str(snapshot), 'sha256': hashes,
    'evidence': [r['config']['label'] for r in references]}, indent=2) + '\n')
run(['python3', str(base / 'pin-glm-flash-chunk16.py')], 'glm-chunk16-pin')
run(['python3', str(base / 'stage-glm-q5-compact-p2.py'), '--apply'], 'glm-q5-p2-apply')
run(['cmake', '--build', str(engine / 'build-goal'), '-j', '16', '--target', 'llama-server'], 'glm-q5-p2-build')
env = {k: v for k, v in os.environ.items()
       if not k.startswith(('GGML_', 'LLAMA_GRAPH_PHASE', 'LLAMA_MTP_DRAFT_N_FILE', 'OMP_', 'GOMP_', 'REPACK_TEST_'))}
env.update({k: v for k, v in selected['runtime_env'].items()
            if not k.endswith('_FILE') and not k.startswith('GGML_CPU_OP_PROFILE')})
env['LD_LIBRARY_PATH'] = str(bindir)
flags = ['g++', '-O2', '-std=c++17']
flags += ['-I' + str(engine / p) for p in ['include', 'ggml/include', 'ggml/src', 'ggml/src/ggml-cpu']]
links = ['-L' + str(bindir), '-lggml', '-lggml-cpu', '-lggml-base', '-ldl', '-pthread']
test = base / 'glm-q5-p2-check'
run(flags + [str(base / 'iq2-repack-check.cpp')] + links + ['-o', str(test)], 'glm-q5-p2-check-build')
summary = []
timings = []
for padded in [False, True]:
    for real in [False, True]:
        runs = []
        # Reverse the second pass to reduce launch-order bias in focused timing.
        for index, enabled in enumerate([False, True, True, False] if real else [False, True]):
            case_env = dict(env, GGML_CPU_NUMA_DEVICES='0', GGML_CPU_X16_Q5_COMPACT_P2=str(int(enabled)),
                            REPACK_TEST_SMALL_BATCHES='1', REPACK_TEST_DENSE_WORK_SHARING='1',
                            REPACK_TEST_PERSISTENT_POOL='1')
            if real:
                case_env['REPACK_TEST_Q5_REAL_SHAPES'] = '1'
            if padded:
                case_env.update(REPACK_TEST_PADDED='1', REPACK_TEST_DOWN='1')
            label = 'glm-q5-p2-' + ('real' if real else 'coverage') + ('-padded-' if padded else '-standard-') + str(index) + '-' + str(int(enabled))
            log = run(['numactl', '--physcpubind=0-15', '--membind=0', str(test), 'q5-pair'], label, case_env, 900)
            hashes = re.findall(r'hash=([0-9a-f]+)', log)
            assert len(hashes) == (12 if real else 90)
            runs.append(hashes)
            if real:
                for line in log.splitlines():
                    if not line.startswith('PASS '):
                        continue
                    values = dict(re.findall(r'(\w+)=([^ ]+)', line))
                    timings.append(dict(padded=padded, enabled=enabled, index=index,
                                        k=int(values['k']), rows=int(values['rows']),
                                        threads=int(values['threads']), ms=float(values['packed_ms'])))
        assert all(h == runs[0] for h in runs), 'Q5 accumulator grouping changed output bytes'
        summary.append(dict(padded=padded, real_shapes=real, cases=len(runs[0]), output_hashes_identical=True))
ratios = []
for padded in [False, True]:
    for k, rows in [(4096, 4096), (1536, 4096), (3072, 4096), (4096, 512)]:
        subset = [t for t in timings if t['padded'] == padded and t['k'] == k and t['rows'] == rows and t['threads'] == 15]
        off = statistics.median(t['ms'] for t in subset if not t['enabled'])
        on = statistics.median(t['ms'] for t in subset if t['enabled'])
        ratios.append(dict(padded=padded, k=k, rows=rows, off_ms=off, on_ms=on, ratio=off/on))
gain = statistics.median(r['ratio'] for r in ratios)
(base / 'results/glm-q5-p2-check-summary.json').write_text(json.dumps({
    'coverage': summary, 'timings': timings, 'ratios_15_workers': ratios,
    'median_ratio_15_workers': gain, 'selected_reference': selected['config']['label']}, indent=2) + '\n')
print('Focused old/new median ratio at 15 workers:', gain, flush=True)
if gain < 1.03:
    print('No sufficient focused gain to justify a full-model run.', flush=True)
    raise SystemExit(0)
env['GGML_CPU_X16_Q5_COMPACT_P2'] = '1'
cfg = dict(selected['config'])
preserved = json.loads((base / 'results/glm-mtp-q8-persistent.json').read_text())
cfg.update(label='glm5n-goal-q5-compact-p2-1024-t15', port=18134, mtp=preserved['destination'])
args = ['python3', str(base / 'qwen-goal-case.py'), cfg.pop('label')]
for key, value in cfg.items():
    if isinstance(value, bool):
        if value:
            args.append('--' + key.replace('_', '-'))
    elif value is not None:
        args.extend(['--' + key.replace('_', '-'), str(value)])
raise SystemExit(subprocess.call(args, env=env))
