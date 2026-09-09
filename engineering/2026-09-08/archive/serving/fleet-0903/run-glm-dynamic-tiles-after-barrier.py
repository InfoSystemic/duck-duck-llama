#!/usr/bin/env python3
"""Test expert work sharing after the current full-model barrier trial finishes."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

root = Path(__file__).resolve().parents[2]
base = root / 'serving/fleet-0903'
previous = base / 'results/glm5n-goal-simple-barrier-spin1000-1024-t15'
while True:
    try:
        finished = json.loads((previous / 'result.json').read_text())
        if 'server_exit' in finished:
            if finished.get('error') or finished['server_exit'] != 0:
                raise SystemExit('Previous run did not finish successfully')
            break
    except (FileNotFoundError, ValueError):
        pass
    time.sleep(3)

def score(result):
    checks = [c for c in result['checks']
              if c['expected'] is None and c['reasoning_budget_tokens'] is None]
    if result.get('error') or any(c['throughput_contended'] for c in checks):
        return 0
    return min(c['response']['timings']['predicted_per_second'] for c in checks)

if score(finished) >= 20:
    print('Barrier candidate reached the throughput target; leave this candidate unapplied.', flush=True)
    raise SystemExit(0)
references = [finished, json.loads((base / 'results/glm5n-goal-compact-p4-draft-sweep-1024-t15/result.json').read_text())]
selected = max(references, key=score)
(base / 'results/glm-dynamic-tiles-reference.json').write_text(json.dumps({
    'selected': selected['config']['label'],
    'scores': {r['config']['label']: score(r) for r in references}}, indent=2) + '\n')
engine = root / 'engines/llama.cpp-glm5n-goal-0904'
bindir = engine / 'build-goal/bin'
snapshot = engine / 'simple-barrier-bin'
if not snapshot.exists():
    shutil.copytree(bindir, snapshot, symlinks=True)

def run(command, label, env=None, timeout=1800):
    log = base / 'results' / (label + '.log')
    with log.open('w') as output:
        result = subprocess.run(command, env=env, stdout=output, stderr=subprocess.STDOUT, timeout=timeout)
    print(label, result.returncode, flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
    return log.read_text()

run(['python3', str(base / 'stage-glm-iq-dynamic-tiles.py'), '--apply'], 'glm-dynamic-tiles-apply')
run(['cmake', '--build', str(engine / 'build-goal'), '-j', '16', '--target', 'llama-server'], 'glm-dynamic-tiles-build')
flags = ['g++', '-O2', '-std=c++17']
flags += ['-I' + str(engine / p) for p in ['include', 'ggml/include', 'ggml/src', 'ggml/src/ggml-cpu']]
links = ['-L' + str(bindir), '-lggml', '-lggml-cpu', '-lggml-base', '-ldl', '-pthread']
env = {k: v for k, v in os.environ.items()
       if not k.startswith(('GGML_', 'LLAMA_GRAPH_PHASE', 'LLAMA_MTP_DRAFT_N_FILE', 'OMP_', 'GOMP_'))}
env.update({k: v for k, v in selected['runtime_env'].items()
            if not k.endswith('_FILE') and not k.startswith('GGML_CPU_OP_PROFILE')})
env['LD_LIBRARY_PATH'] = str(bindir)
test = base / 'glm-dynamic-tiles-check'
run(flags + [str(base / 'iq2-repack-check.cpp')] + links + ['-o', str(test)], 'glm-dynamic-tiles-check-build')
summary = []
for mode in ['standard', 'padded', 'capped']:
    hashes = []
    for enabled in [False, True]:
        case_env = dict(env, GGML_CPU_NUMA_DEVICES='0',
                        GGML_CPU_IQ_R16_DYNAMIC_TILES=str(int(enabled)),
                        REPACK_TEST_SMALL_BATCHES='1', REPACK_TEST_WORK_SHARING='1', REPACK_TEST_CLAMP='1')
        if mode == 'padded':
            case_env.update(REPACK_TEST_PADDED='1', REPACK_TEST_DOWN='1')
        if mode == 'capped':
            case_env['GGML_CPU_MOE_SINGLE_TOKEN_THREADS'] = '2'
        label = 'glm-dynamic-tiles-' + mode + ('-on' if enabled else '-off')
        log = run(['numactl', '--physcpubind=0-15', '--membind=0', str(test), 'iq-r16'], label, case_env, 900)
        hashes.append(re.findall(r'hash=([0-9a-f]+)', log))
    if len(hashes[0]) != 270 or hashes[0] != hashes[1]:
        raise SystemExit('Dynamic work sharing changed numerical outputs')
    summary.append(dict(mode=mode, cases=270, output_hashes_identical=True))
(base / 'results/glm-dynamic-tiles-check-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
env['GGML_CPU_IQ_R16_DYNAMIC_TILES'] = '1'
cfg = dict(selected['config'])
cfg.update(label='glm5n-goal-dynamic-iq-tiles-1024-t15', port=18130,
           draft_n=2, draft_sweep=None, thread_sweep=None)
args = ['python3', str(base / 'qwen-goal-case.py'), cfg.pop('label')]
for key, value in cfg.items():
    if isinstance(value, bool):
        if value:
            args.append('--' + key.replace('_', '-'))
    elif value is not None:
        args.extend(['--' + key.replace('_', '-'), str(value)])
raise SystemExit(subprocess.call(args, env=env))
