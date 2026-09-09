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
waiting = base / 'results/glm5n-goal-pthread-best-mtp-single4k-poll0-t15/result.json'
while True:
    try:
        if 'server_exit' in json.loads(waiting.read_text()):
            break
    except (FileNotFoundError, ValueError):
        pass
    time.sleep(3)
engine = root / 'engines/llama.cpp-q4e-goal-0904'
snapshot = engine / 'rs-valid-bin'
if not snapshot.exists():
    shutil.copytree(engine / 'build-goal/bin', snapshot, symlinks=True)

def run(command, label, env=None, timeout=1800):
    log = base / 'results' / (label + '.log')
    with log.open('w') as output:
        result = subprocess.run(command, env=env, stdout=output, stderr=subprocess.STDOUT, timeout=timeout)
    print(label, result.returncode, flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
    return log.read_text()

def score(result):
    rows = [c for c in result['checks'] if c['expected'] is None and c['reasoning_budget_tokens'] is None]
    if result.get('error') or any(c['throughput_contended'] for c in rows):
        return 0
    return min(c['response']['timings']['predicted_per_second'] for c in rows)
pthread_result = json.loads(waiting.read_text())
reference_label = json.loads((base / 'results/glm-pthread-reference.json').read_text())['source']
reference = json.loads((base / 'results' / reference_label / 'result.json').read_text())
use_pthread = score(pthread_result) > 1.05 * score(reference)
build = Path('/dev/shm/flash-goal-0904-qwen-pthread-build') if use_pthread else engine / 'build-goal'
if use_pthread:
    seed = []
    for line in (engine / 'build-goal/CMakeCache.txt').read_text().splitlines():
        match = re.fullmatch(r'((?:GGML_|LLAMA_)[A-Z0-9_]+|BUILD_SHARED_LIBS|CMAKE_BUILD_TYPE):(BOOL|STRING)=(.*)', line)
        if not match:
            continue
        name, kind, value = match.groups()
        if name in ['GGML_OPENMP', 'GGML_CCACHE']:
            value = 'OFF'
        seed.append(f'set({name} "{value}" CACHE {kind} "Existing goal build setting" FORCE)')
    seed_file = base / 'qwen-pthread-cache.cmake'
    seed_file.write_text('\n'.join(seed) + '\n')
    run(['cmake', '-S', str(engine), '-B', str(build), '-C', str(seed_file)], 'qwen-nibble-sort-configure')
(base / 'results/qwen-nibble-sort-backend.json').write_text(json.dumps({
    'pthread': use_pthread, 'glm_pthread_score': score(pthread_result),
    'glm_reference_score': score(reference), 'glm_reference': reference_label,
    'criterion': 'Use pthread if the isolated GLM regular-sample minimum improves by over 5%.'}, indent=2) + '\n')
run(['cmake', '--build', str(build), '-j', '16', '--target', 'llama-server'], 'qwen-nibble-sort-build')
bindir = build / 'bin'
flags = ['g++', '-O2', '-std=c++17']
flags += ['-I' + str(engine / p) for p in ['include', 'ggml/include', 'ggml/src', 'ggml/src/ggml-cpu']]
links = ['-L' + str(bindir), '-lggml', '-lggml-cpu', '-lggml-base', '-ldl', '-pthread']
run(flags + [str(base / 'iq2-repack-check.cpp')] + links + ['-o', str(base / 'qwen-nibble-sort-repack-check')],
    'qwen-nibble-sort-repack-check-build')
run(flags + [str(base / 'argsort-topk-check.cpp')] + links + ['-o', str(base / 'qwen-argsort-topk-check')],
    'qwen-argsort-topk-check-build')
summary = []
for padded in [False, True]:
    hashes = []
    for enabled in [False, True]:
        env = dict(os.environ, LD_LIBRARY_PATH=str(bindir), GGML_CPU_IQ_R16_REPACK='1',
                   GGML_CPU_IQ_R16_NIBBLE2=str(int(enabled)))
        if padded:
            env.update(REPACK_TEST_PADDED='1', REPACK_TEST_DOWN='1')
        label = 'qwen-iq-nibble2-check-' + ('padded' if padded else 'standard') + '-' + ('on' if enabled else 'off')
        log = run(['numactl', '--physcpubind=0-3', '--membind=0',
                   str(base / 'qwen-nibble-sort-repack-check'), 'iq-r16-qwen'], label, env, timeout=300)
        hashes.append(re.findall(r'hash=([0-9a-f]+)', log))
    if hashes[0] != hashes[1] or len(hashes[0]) != 144:
        raise SystemExit('Qwen IQ nibble layout changed outputs')
    summary.append(dict(kind='iq', padded=padded, cases=len(hashes[0]), output_hashes_identical=True))
hashes = []
for enabled in [False, True]:
    env = dict(os.environ, LD_LIBRARY_PATH=str(bindir), GGML_CPU_ARGSORT_TOP_K=str(int(enabled)))
    label = 'qwen-argsort-topk-check-' + ('on' if enabled else 'off')
    log = run(['numactl', '--physcpubind=0-3', '--membind=0', str(base / 'qwen-argsort-topk-check')],
              label, env, timeout=300)
    hashes.append(re.findall(r'hash=([0-9a-f]+)', log))
if hashes[0] != hashes[1] or len(hashes[0]) != 720:
    raise SystemExit('Partial sort changed selected indices')
summary.append(dict(kind='argsort', cases=720, output_hashes_identical=True))
(base / 'results/qwen-nibble-sort-check-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
env = dict(os.environ, QWEN_ROLLBACK_BUILD_DIR=str(build), QWEN_ROLLBACK_LABEL_PREFIX='qwen-nibble-sort-rs')
run(['python3', str(base / 'run-qwen-rollback-valid-range-fixtures.py')], 'qwen-nibble-sort-rs-fixtures', env)
old = json.loads((base / 'results/q4e-goal-rs-valid-mtp2-single4k-1024-t15/config.json').read_text())
cfg = dict(old['config'])
cfg.update(label='q4e-goal-nibble-sort-mtp2-single4k-1024-t15', port=18121,
           binary=str(bindir / 'llama-server'), numa_poll=0 if use_pthread else 100)
env = dict(os.environ, **old['runtime_env'])
env.update(GGML_CPU_IQ_R16_NIBBLE2='1', GGML_CPU_ARGSORT_TOP_K='1')
args = ['python3', str(base / 'qwen-goal-case.py'), cfg.pop('label')]
for key, value in cfg.items():
    if isinstance(value, bool):
        if value:
            args.append('--' + key.replace('_', '-'))
    elif value is not None:
        args.extend(['--' + key.replace('_', '-'), str(value)])
raise SystemExit(subprocess.call(args, env=env))
