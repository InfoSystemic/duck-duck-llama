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
waiting = base / 'results/q4e-goal-iq-batch3-threads-1024/result.json'
while True:
    try:
        if 'server_exit' in json.loads(waiting.read_text()):
            break
    except (FileNotFoundError, ValueError):
        pass
    time.sleep(3)

engine = root / 'engines/llama.cpp-glm5n-goal-0904'
bindir = engine / 'build-goal/bin'
snapshot = engine / 'iq-batch3-bin'
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

run(['cmake', '--build', str(engine / 'build-goal'), '-j', '16', '--target', 'llama-server'], 'glm-clamp-build')
flags = ['g++', '-O2', '-std=c++17']
flags += ['-I' + str(engine / p) for p in ['include', 'ggml/include', 'ggml/src', 'ggml/src/ggml-cpu']]
links = ['-L' + str(bindir), '-lggml', '-lggml-cpu', '-lggml-base', '-ldl', '-pthread']
test = base / 'glm-clamp-check'
run(flags + [str(base / 'iq2-repack-check.cpp')] + links + ['-o', str(test)], 'glm-clamp-check-build')
summary = []
for mode in ['standard', 'padded', 'consumer']:
    hashes = []
    for enabled in [False, True]:
        env = dict(os.environ, LD_LIBRARY_PATH=str(bindir), GGML_CPU_IQ_R16_REPACK='1',
                   GGML_CPU_IQ_R16_NIBBLE2='1', GGML_CPU_IQ_R16_BATCH3='1',
                   GGML_CPU_MOE_CLAMP_FUSION=str(int(enabled)),
                   REPACK_TEST_SMALL_BATCHES='1', REPACK_TEST_CLAMP='1')
        if mode == 'padded':
            env.update(REPACK_TEST_PADDED='1', REPACK_TEST_DOWN='1')
        if mode == 'consumer':
            env['REPACK_TEST_CLAMP_CONSUMER'] = '1'
        label = 'glm-clamp-check-' + mode + '-' + ('on' if enabled else 'off')
        log = run(['numactl', '--physcpubind=0-3', '--membind=0', str(test), 'iq-r16'], label, env, timeout=600)
        hashes.append(re.findall(r'hash=([0-9a-f]+)', log))
        if enabled:
            active = 'MOE_CLAMP_FUSION_ACTIVE' in log
            if active != (mode != 'consumer'):
                raise SystemExit('Clamped fusion activation or external-consumer guard is incorrect')
    if hashes[0] != hashes[1] or len(hashes[0]) != 360:
        raise SystemExit('Clamped expert fusion changed outputs')
    summary.append(dict(mode=mode, cases=360, output_hashes_identical=True,
                        fused_path_active=mode != 'consumer'))
(base / 'results/glm-clamp-check-summary.json').write_text(json.dumps(summary, indent=2) + '\n')

previous = base / 'results/glm5n-goal-iq-batch3-best-threads-1024'
old = json.loads((previous / 'config.json').read_text())
cfg = dict(old['config'])
cfg.update(label='glm5n-goal-iq-batch3-clamp-1024-t15', port=18126,
           binary=str(bindir / 'llama-server'), profile_perf=True)
env = dict(os.environ, **old['runtime_env'])
env.update(GGML_CPU_MOE_CLAMP_FUSION='1')
args = ['python3', str(base / 'qwen-goal-case.py'), cfg.pop('label')]
for key, value in cfg.items():
    if isinstance(value, bool):
        if value:
            args.append('--' + key.replace('_', '-'))
    elif value is not None:
        args.extend(['--' + key.replace('_', '-'), str(value)])
raise SystemExit(subprocess.call(args, env=env))
