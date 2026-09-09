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
waiting = base / 'results/q4e-goal-nibble-sort-mtp2-single4k-1024-t15/result.json'
while True:
    try:
        if 'server_exit' in json.loads(waiting.read_text()):
            break
    except (FileNotFoundError, ValueError):
        pass
    time.sleep(3)
engine = root / 'engines/llama.cpp-glm5n-goal-0904'
bindir = engine / 'build-goal/bin'
snapshot = engine / 'iq-nibble2-bin'
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

run(['cmake', '--build', str(engine / 'build-goal'), '-j', '16', '--target', 'llama-server'], 'glm-combined-sweep-build')
flags = ['g++', '-O2', '-std=c++17']
flags += ['-I' + str(engine / p) for p in ['include', 'ggml/include', 'ggml/src', 'ggml/src/ggml-cpu']]
links = ['-L' + str(bindir), '-lggml', '-lggml-cpu', '-lggml-base', '-ldl', '-pthread']
for source, name in [('iq2-repack-check.cpp', 'glm-q5-triple-check'),
                     ('argsort-topk-check.cpp', 'glm-argsort-topk-check'),
                     ('numa-thread-limit-check.cpp', 'glm-numa-thread-limit-check')]:
    run(flags + [str(base / source)] + links + ['-o', str(base / name)], name + '-build')
summary = []
for padded in [False, True]:
    hashes = []
    for enabled in [False, True]:
        env = dict(os.environ, LD_LIBRARY_PATH=str(bindir), GGML_CPU_X16_Q5_BYTES='1',
                   GGML_CPU_X16_Q5_BATCH2='1', GGML_CPU_X16_Q5_BYTES_BATCH3=str(int(enabled)))
        if padded:
            env.update(REPACK_TEST_PADDED='1', REPACK_TEST_DOWN='1')
        label = 'glm-q5-triple-check-' + ('padded' if padded else 'standard') + '-' + ('on' if enabled else 'off')
        log = run(['numactl', '--physcpubind=0-3', '--membind=0', str(base / 'glm-q5-triple-check'), 'q5-pair'],
                  label, env, timeout=300)
        hashes.append(re.findall(r'hash=([0-9a-f]+)', log))
    if hashes[0] != hashes[1] or len(hashes[0]) != 72:
        raise SystemExit('Q5 triple kernel changed outputs')
    summary.append(dict(kind='q5-triple', padded=padded, cases=72, output_hashes_identical=True))
hashes = []
for enabled in [False, True]:
    env = dict(os.environ, LD_LIBRARY_PATH=str(bindir), GGML_CPU_ARGSORT_TOP_K=str(int(enabled)))
    label = 'glm-argsort-topk-check-' + ('on' if enabled else 'off')
    log = run(['numactl', '--physcpubind=0-3', '--membind=0', str(base / 'glm-argsort-topk-check')],
              label, env, timeout=300)
    hashes.append(re.findall(r'hash=([0-9a-f]+)', log))
if hashes[0] != hashes[1] or len(hashes[0]) != 720:
    raise SystemExit('GLM partial sort changed selected indices')
summary.append(dict(kind='argsort', cases=720, output_hashes_identical=True))
env = dict(os.environ, LD_LIBRARY_PATH=str(bindir))
log = run([str(base / 'glm-numa-thread-limit-check'), str(base / 'results/glm-thread-limit-control')],
          'glm-numa-thread-limit-check', env, timeout=180)
if len(re.findall(r'^PASS ', log, re.MULTILINE)) != 40:
    raise SystemExit('Dynamic NUMA worker counts did not pass all 40 cases')
summary.append(dict(kind='thread-limit', cases=40, passed=True))
(base / 'results/glm-combined-sweep-check-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
old = json.loads((base / 'results/glm5n-goal-q5-bytes-q8mtp2-single4k-t15/config.json').read_text())
cfg = dict(old['config'])
cfg.update(label='glm5n-goal-triple-nibble-sort-thread-sweep', port=18122,
           binary=str(bindir / 'llama-server'), threads=16, numa_poll=100,
           thread_sweep='16,15,12,8', cache_check=True)
env = dict(os.environ, **old['runtime_env'])
env.update(GGML_CPU_IQ_R16_NIBBLE2='1', GGML_CPU_ARGSORT_TOP_K='1', GGML_CPU_X16_Q5_BYTES_BATCH3='1')
args = ['python3', str(base / 'qwen-goal-case.py'), cfg.pop('label')]
for key, value in cfg.items():
    if isinstance(value, bool):
        if value:
            args.append('--' + key.replace('_', '-'))
    elif value is not None:
        args.extend(['--' + key.replace('_', '-'), str(value)])
raise SystemExit(subprocess.call(args, env=env))
