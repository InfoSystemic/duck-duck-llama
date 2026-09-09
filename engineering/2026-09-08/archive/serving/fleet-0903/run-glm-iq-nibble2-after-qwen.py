#!/usr/bin/env python3
import json
import os
from pathlib import Path
import re
import subprocess
import time

root = Path(__file__).resolve().parents[2]
base = root / 'serving/fleet-0903'
waiting = base / 'results/q4e-goal-rs-valid-mtp2-single4k-1024-t15/result.json'
while True:
    try:
        if 'server_exit' in json.loads(waiting.read_text()):
            break
    except (FileNotFoundError, ValueError):
        pass
    time.sleep(3)
engine = root / 'engines/llama.cpp-glm5n-goal-0904'
bindir = engine / 'build-goal/bin'

def run(command, label, env=None):
    log = base / 'results' / (label + '.log')
    with log.open('w') as output:
        result = subprocess.run(command, env=env, stdout=output, stderr=subprocess.STDOUT)
    print(label, result.returncode, flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
    return log.read_text()

run(['cmake', '--build', str(engine / 'build-goal'), '-j', '16', '--target', 'llama-server'],
    'glm-iq-nibble2-build')
command = ['g++', '-O2', '-std=c++17']
command += ['-I' + str(engine / p) for p in ['include', 'ggml/include', 'ggml/src', 'ggml/src/ggml-cpu']]
command += [str(base / 'iq2-repack-check.cpp'), '-L' + str(bindir), '-lggml', '-lggml-cpu',
            '-lggml-base', '-ldl', '-pthread', '-o', str(base / 'glm-iq-nibble2-check')]
run(command, 'glm-iq-nibble2-check-build')
summary = []
for padded in [False, True]:
    hashes = []
    for enabled in [False, True]:
        env = dict(os.environ, LD_LIBRARY_PATH=str(bindir), GGML_CPU_IQ_R16_REPACK='1',
                   GGML_CPU_IQ_R16_NIBBLE2=str(int(enabled)))
        if padded:
            env.update(REPACK_TEST_PADDED='1', REPACK_TEST_DOWN='1')
        label = 'glm-iq-nibble2-check-' + ('padded' if padded else 'standard') + '-' + ('on' if enabled else 'off')
        log = run(['numactl', '--physcpubind=0-3', '--membind=0',
                   str(base / 'glm-iq-nibble2-check'), 'iq-r16'], label, env)
        hashes.append(re.findall(r'hash=([0-9a-f]+)', log))
    same = hashes[0] == hashes[1] and len(hashes[0]) == 216
    summary.append(dict(padded=padded, cases=len(hashes[0]), output_hashes_identical=same))
    print(summary[-1], flush=True)
    if not same:
        raise SystemExit(1)
(base / 'results/glm-iq-nibble2-check-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
labels = ['glm5n-goal-q5-bytes-q8mtp2-single4k-t15', 'glm5n-goal-q5-bytes-p4-q8mtp1-single4k-t15']
results = [json.loads((base / 'results' / label / 'result.json').read_text()) for label in labels]
def score(result):
    regular = [check for check in result['checks'] if check['expected'] is None and check['reasoning_budget_tokens'] is None]
    if result.get('error') or any(check['throughput_contended'] for check in regular):
        return 0
    return min(check['response']['timings']['predicted_per_second'] for check in regular)
old = max(results, key=score)
cfg = dict(old['config'])
cfg.update(label='glm5n-goal-iq-nibble2-best-mtp-single4k-t15', port=18119)
env = dict(os.environ, **old['runtime_env'])
env['GGML_CPU_IQ_R16_NIBBLE2'] = '1'
(base / 'results/glm-iq-nibble2-reference.json').write_text(json.dumps({
    'source': old['config']['label'], 'scores': {result['config']['label']: score(result) for result in results},
    'note': 'Current binary includes Q5 byte p4; reference MTP2 used the initial byte kernel.'}, indent=2) + '\n')
args = ['python3', str(base / 'qwen-goal-case.py'), cfg.pop('label')]
for key, value in cfg.items():
    if isinstance(value, bool):
        if value:
            args.append('--' + key.replace('_', '-'))
    elif value is not None:
        args.extend(['--' + key.replace('_', '-'), str(value)])
raise SystemExit(subprocess.call(args, env=env))
