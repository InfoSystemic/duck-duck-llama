#!/usr/bin/env python3
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

root = Path(__file__).resolve().parents[2]
base = root / 'serving/fleet-0903'
previous = base / 'results/glm5n-goal-iq-batch3-clamp-1024-t15'
while True:
    try:
        if 'server_exit' in json.loads((previous / 'result.json').read_text()):
            break
    except (FileNotFoundError, ValueError):
        pass
    time.sleep(3)

engine = root / 'engines/llama.cpp-glm5n-goal-0904'
bindir = engine / 'build-goal/bin'
snapshot = engine / 'clamp-bin'
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

run(['cmake', '--build', str(engine / 'build-goal'), '-j', '16', '--target', 'llama-server'], 'glm-compact-draft-build')
flags = ['g++', '-O2', '-std=c++17']
flags += ['-I' + str(engine / p) for p in ['include', 'ggml/include', 'ggml/src', 'ggml/src/ggml-cpu']]
links = ['-L' + str(bindir), '-lggml', '-lggml-cpu', '-lggml-base', '-ldl', '-pthread']
test = base / 'glm-compact-p4-check'
run(flags + [str(base / 'iq2-repack-check.cpp')] + links + ['-o', str(test)], 'glm-compact-p4-check-build')
summary = []
for padded in [False, True]:
    hashes = []
    for mode in ['old', 'compact', 'bytes']:
        env = dict(os.environ, LD_LIBRARY_PATH=str(bindir),
                   GGML_CPU_X16_Q5_BYTES=str(int(mode == 'bytes')),
                   GGML_CPU_X16_Q5_COMPACT_P4=str(int(mode == 'compact')),
                   GGML_CPU_X16_Q5_BATCH2='1', GGML_CPU_X16_Q5_BYTES_BATCH3='1',
                   REPACK_TEST_SMALL_BATCHES='1')
        if padded:
            env.update(REPACK_TEST_PADDED='1', REPACK_TEST_DOWN='1')
        label = 'glm-compact-p4-check-' + ('padded' if padded else 'standard') + '-' + mode
        log = run(['numactl', '--physcpubind=0-3', '--membind=0', str(test), 'q5-pair'], label, env, timeout=600)
        hashes.append(re.findall(r'hash=([0-9a-f]+)', log))
    if any(len(value) != 120 or value != hashes[0] for value in hashes):
        raise SystemExit('Compact Q5 p4 changed outputs versus the original and byte kernels')
    summary.append(dict(padded=padded, cases=120, modes=['old', 'compact', 'bytes'], output_hashes_identical=True))
(base / 'results/glm-compact-p4-check-summary.json').write_text(json.dumps(summary, indent=2) + '\n')

old = json.loads((previous / 'config.json').read_text())
cfg = dict(old['config'])
cfg.update(label='glm5n-goal-compact-p4-draft-sweep-1024-t15', port=18127,
           binary=str(bindir / 'llama-server'), draft_n=4, draft_sweep='2,3,4,1', profile_perf=True)
env = dict(os.environ, **old['runtime_env'])
env.update(GGML_CPU_X16_Q5_BYTES='0', GGML_CPU_X16_Q5_COMPACT_P4='1')
args = ['python3', str(base / 'qwen-goal-case.py'), cfg.pop('label')]
for key, value in cfg.items():
    if isinstance(value, bool):
        if value:
            args.append('--' + key.replace('_', '-'))
    elif value is not None:
        args.extend(['--' + key.replace('_', '-'), str(value)])
raise SystemExit(subprocess.call(args, env=env))
