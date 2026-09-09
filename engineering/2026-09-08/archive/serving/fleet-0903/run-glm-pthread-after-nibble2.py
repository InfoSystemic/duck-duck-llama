#!/usr/bin/env python3
import json
import os
from pathlib import Path
import re
import subprocess
import time

root = Path(__file__).resolve().parents[2]
base = root / 'serving/fleet-0903'
waiting = base / 'results/glm5n-goal-iq-nibble2-best-mtp-single4k-t15/result.json'
while True:
    try:
        if 'server_exit' in json.loads(waiting.read_text()):
            break
    except (FileNotFoundError, ValueError):
        pass
    time.sleep(3)
engine = root / 'engines/llama.cpp-glm5n-goal-0904'
build = Path('/dev/shm/flash-goal-0904-glm-pthread-build')
bindir = build / 'bin'

def run(command, label, env=None, timeout=1800):
    log = base / 'results' / (label + '.log')
    with log.open('w') as output:
        result = subprocess.run(command, env=env, stdout=output, stderr=subprocess.STDOUT, timeout=timeout)
    print(label, result.returncode, flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
    return log.read_text()

seed = []
for line in (engine / 'build-goal/CMakeCache.txt').read_text().splitlines():
    match = re.fullmatch(r'((?:GGML_|LLAMA_)[A-Z0-9_]+|BUILD_SHARED_LIBS|CMAKE_BUILD_TYPE):(BOOL|STRING)=(.*)', line)
    if not match:
        continue
    name, kind, value = match.groups()
    if name == 'GGML_OPENMP':
        value = 'OFF'
    seed.append(f'set({name} "{value}" CACHE {kind} "Existing goal build setting" FORCE)')
seed_file = base / 'glm-pthread-cache.cmake'
seed_file.write_text('\n'.join(seed) + '\n')
run(['cmake', '-S', str(engine), '-B', str(build), '-C', str(seed_file)], 'glm-pthread-configure')
run(['cmake', '--build', str(build), '-j', '16', '--target', 'llama-server'], 'glm-pthread-build')
flags = ['g++', '-O2', '-std=c++17']
flags += ['-I' + str(engine / p) for p in ['include', 'ggml/include', 'ggml/src', 'ggml/src/ggml-cpu']]
links = ['-L' + str(bindir), '-lggml', '-lggml-cpu', '-lggml-base', '-ldl', '-pthread']
run(flags + [str(base / 'iq2-repack-check.cpp')] + links + ['-o', str(base / 'glm-pthread-repack-check')],
    'glm-pthread-repack-check-build')
labels = ['glm5n-goal-q5-bytes-q8mtp2-single4k-t15', 'glm5n-goal-iq-nibble2-best-mtp-single4k-t15']
results = [json.loads((base / 'results' / label / 'result.json').read_text()) for label in labels]
def score(result):
    regular = [c for c in result['checks'] if c['expected'] is None and c['reasoning_budget_tokens'] is None]
    if result.get('error') or any(c['throughput_contended'] for c in regular):
        return 0
    return min(c['response']['timings']['predicted_per_second'] for c in regular)
old = max(results, key=score)
env = dict(os.environ, **old['runtime_env'], LD_LIBRARY_PATH=str(bindir))
env['GGML_CPU_NUMA_POLL'] = '0'
summary = []
for padded in [False, True]:
    case_env = dict(env)
    if padded:
        case_env.update(REPACK_TEST_PADDED='1', REPACK_TEST_DOWN='1')
    for mode, count in [('iq-r16', 216), ('q5-pair', 72)]:
        label = 'glm-pthread-check-' + mode + '-' + ('padded' if padded else 'standard')
        # A CPU-only check must expose the ordinary CPU device on the pinned socket.
        case_env['GGML_CPU_NUMA_DEVICES'] = '0'
        log = run(['numactl', '--physcpubind=0-3', '--membind=0',
                   str(base / 'glm-pthread-repack-check'), mode], label, case_env, timeout=300)
        passed = len(re.findall(r'^PASS ', log, re.MULTILINE))
        if passed != count:
            raise SystemExit(f'{label}: {passed} cases, expected {count}')
        summary.append(dict(label=label, passed=passed))
run(flags + [str(base / 'numa-reduce-check.cpp')] + links + ['-o', str(base / 'glm-pthread-numa-reduce-check')],
    'glm-pthread-numa-reduce-check-build')
reduce_env = dict(env, GGML_CPU_NUMA_DEVICES='1', GGML_CPU_NUMA_THREADS='2')
log = run([str(base / 'glm-pthread-numa-reduce-check')], 'glm-pthread-numa-reduce-check', reduce_env, timeout=120)
if len(re.findall(r' PASS$', log, re.MULTILINE)) != 20:
    raise SystemExit('Not all 20 NUMA reduction cases passed')
(base / 'results/glm-pthread-check-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
cfg = dict(old['config'])
cfg.update(label='glm5n-goal-pthread-best-mtp-single4k-poll0-t15', port=18120,
           binary=str(bindir / 'llama-server'), numa_poll=0)
(base / 'results/glm-pthread-reference.json').write_text(json.dumps({
    'source': old['config']['label'], 'scores': {r['config']['label']: score(r) for r in results},
    'change': 'GGML_OPENMP=OFF and numa_poll=0; current source retains Q5 byte p4.'}, indent=2) + '\n')
args = ['python3', str(base / 'qwen-goal-case.py'), cfg.pop('label')]
for key, value in cfg.items():
    if isinstance(value, bool):
        if value:
            args.append('--' + key.replace('_', '-'))
    elif value is not None:
        args.extend(['--' + key.replace('_', '-'), str(value)])
raise SystemExit(subprocess.call(args, env=env))
