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
waiting = base / 'results/q4e-goal-nibble-sort-thread-sweep-1024/result.json'
while True:
    try:
        if 'server_exit' in json.loads(waiting.read_text()):
            break
    except (FileNotFoundError, ValueError):
        pass
    time.sleep(3)

engine = root / 'engines/llama.cpp-glm5n-goal-0904'
bindir = engine / 'build-goal/bin'
snapshot = engine / 'triple-nibble-sort-bin'
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

run(['cmake', '--build', str(engine / 'build-goal'), '-j', '16', '--target', 'llama-server'], 'glm-iq-batch-build')
flags = ['g++', '-O2', '-std=c++17']
flags += ['-I' + str(engine / p) for p in ['include', 'ggml/include', 'ggml/src', 'ggml/src/ggml-cpu']]
links = ['-L' + str(bindir), '-lggml', '-lggml-cpu', '-lggml-base', '-ldl', '-pthread']
test = base / 'glm-iq-batch-check'
run(flags + [str(base / 'iq2-repack-check.cpp')] + links + ['-o', str(test)], 'glm-iq-batch-check-build')
summary = []
for padded in [False, True]:
    hashes = []
    for enabled in [False, True]:
        env = dict(os.environ, LD_LIBRARY_PATH=str(bindir),
                   GGML_CPU_IQ_R16_REPACK='1',
                   GGML_CPU_IQ_R16_NIBBLE2='1', GGML_CPU_IQ_R16_BATCH3=str(int(enabled)),
                   REPACK_TEST_SMALL_BATCHES='1')
        if padded:
            env.update(REPACK_TEST_PADDED='1', REPACK_TEST_DOWN='1')
        label = 'glm-iq-batch-check-' + ('padded' if padded else 'standard') + '-' + ('on' if enabled else 'off')
        log = run(['numactl', '--physcpubind=0-3', '--membind=0', str(test), 'iq-r16'], label, env, timeout=600)
        hashes.append(re.findall(r'hash=([0-9a-f]+)', log))
    if hashes[0] != hashes[1] or len(hashes[0]) != 360:
        raise SystemExit('IQ batch kernel changed outputs')
    summary.append(dict(padded=padded, cases=360, output_hashes_identical=True))
(base / 'results/glm-iq-batch-check-summary.json').write_text(json.dumps(summary, indent=2) + '\n')

previous = base / 'results/glm5n-goal-triple-nibble-sort-thread-sweep'
old = json.loads((previous / 'config.json').read_text())
measured = json.loads((previous / 'result.json').read_text())
cfg = dict(old['config'])
cfg.update(label='glm5n-goal-iq-batch3-best-threads-1024', port=18124,
           binary=str(bindir / 'llama-server'), threads=measured['selected_numa_threads'],
           thread_sweep=None, cache_check=True, bench_tokens=1024)
env = dict(os.environ, **old['runtime_env'])
env.pop('GGML_CPU_NUMA_THREADS_FILE', None)
env.update(GGML_CPU_IQ_R16_BATCH3='1')
args = ['python3', str(base / 'qwen-goal-case.py'), cfg.pop('label')]
for key, value in cfg.items():
    if isinstance(value, bool):
        if value:
            args.append('--' + key.replace('_', '-'))
    elif value is not None:
        args.extend(['--' + key.replace('_', '-'), str(value)])
raise SystemExit(subprocess.call(args, env=env))
