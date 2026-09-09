#!/usr/bin/env python3
"""Validate the atomic graph barrier after the isolated OpenMP spin trial exits."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

root = Path(__file__).resolve().parents[2]
base = root / 'serving/fleet-0903'
previous = base / 'results/glm5n-goal-compact-p4-spin1000-1024-t15'
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

engine = root / 'engines/llama.cpp-glm5n-goal-0904'
bindir = engine / 'build-goal/bin'
snapshot = engine / 'compact-p4-bin'
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

run(['python3', str(base / 'stage-glm-openmp-simple-barrier.py'), '--apply'], 'glm-simple-barrier-apply')
run(['cmake', '--build', str(engine / 'build-goal'), '-j', '16', '--target', 'llama-server'], 'glm-simple-barrier-build')
flags = ['g++', '-O2', '-std=c++17']
flags += ['-I' + str(engine / p) for p in ['include', 'ggml/include', 'ggml/src', 'ggml/src/ggml-cpu']]
links = ['-L' + str(bindir), '-lggml', '-lggml-cpu', '-lggml-base', '-ldl', '-pthread']
old = json.loads((previous / 'config.json').read_text())
env = {k: v for k, v in os.environ.items()
       if not k.startswith(('GGML_', 'LLAMA_GRAPH_PHASE', 'LLAMA_MTP_DRAFT_N_FILE', 'OMP_', 'GOMP_'))}
env.update({k: v for k, v in old['runtime_env'].items() if not k.endswith('_FILE')})
env['LD_LIBRARY_PATH'] = str(bindir)
test = base / 'glm-simple-barrier-repack-check'
run(flags + [str(base / 'iq2-repack-check.cpp')] + links + ['-o', str(test)], 'glm-simple-barrier-repack-check-build')
summary = []
for padded in [False, True]:
    for mode, count in [('iq-r16', 360), ('q5-pair', 120)]:
        hashes = []
        for enabled in [False, True]:
            case_env = dict(env, GGML_CPU_NUMA_DEVICES='0',
                            GGML_CPU_OMP_SIMPLE_BARRIER=str(int(enabled)),
                            REPACK_TEST_SMALL_BATCHES='1', REPACK_TEST_CLAMP='1')
            if padded:
                case_env.update(REPACK_TEST_PADDED='1', REPACK_TEST_DOWN='1')
            label = f'glm-simple-barrier-{mode}-' + ('padded' if padded else 'standard') + ('-on' if enabled else '-off')
            log = run(['numactl', '--physcpubind=0-3', '--membind=0', str(test), mode], label, case_env, 600)
            hashes.append(re.findall(r'hash=([0-9a-f]+)', log))
        if len(hashes[0]) != count or hashes[0] != hashes[1]:
            raise SystemExit('Barrier changed numerical outputs')
        summary.append(dict(mode=mode, padded=padded, cases=count, output_hashes_identical=True))
env['GGML_CPU_OMP_SIMPLE_BARRIER'] = '1'
for fixture in ['numa-reduce-check', 'numa-thread-limit-check']:
    target = base / ('glm-simple-barrier-' + fixture)
    run(flags + [str(base / (fixture + '.cpp'))] + links + ['-o', str(target)], 'glm-simple-barrier-' + fixture + '-build')
    command = [str(target)]
    if fixture == 'numa-thread-limit-check':
        command += [str(base / 'results/glm-simple-barrier-thread-limit')]
    case_env = dict(env, GGML_CPU_NUMA_DEVICES='1', GGML_CPU_NUMA_THREADS='2')
    log = run(command, 'glm-simple-barrier-' + fixture, case_env, 180)
    count = len(re.findall(r' PASS$', log, re.MULTILINE)) if fixture == 'numa-reduce-check' else len(re.findall(r'^PASS ', log, re.MULTILINE))
    expected = 20 if fixture == 'numa-reduce-check' else 40
    if count != expected:
        raise SystemExit(f'{fixture}: expected {expected} passing cases, got {count}')
    summary.append(dict(fixture=fixture, cases=count, pass_=True))
(base / 'results/glm-simple-barrier-check-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
cfg = dict(old['config'])
cfg.update(label='glm5n-goal-simple-barrier-spin1000-1024-t15', port=18129)
args = ['python3', str(base / 'qwen-goal-case.py'), cfg.pop('label')]
for key, value in cfg.items():
    if isinstance(value, bool):
        if value:
            args.append('--' + key.replace('_', '-'))
    elif value is not None:
        args.extend(['--' + key.replace('_', '-'), str(value)])
raise SystemExit(subprocess.call(args, env=env))
