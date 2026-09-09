#!/usr/bin/env python3
import json
import os
from pathlib import Path
import re
import subprocess
import time

root = Path(__file__).resolve().parents[2]
base = root / 'serving/fleet-0903'
waiting = base / 'results/glm5n-goal-triple-nibble-sort-thread-sweep/result.json'
while True:
    try:
        if 'server_exit' in json.loads(waiting.read_text()):
            break
    except (FileNotFoundError, ValueError):
        pass
    time.sleep(3)

engine = root / 'engines/llama.cpp-q4e-goal-0904'
bindir = engine / 'build-goal/bin'
env = dict(os.environ, LD_LIBRARY_PATH=str(bindir))
test = base / 'qwen-numa-thread-limit-check'
command = ['g++', '-O2', '-std=c++17', '-I' + str(engine / 'ggml/include'),
           str(base / 'numa-thread-limit-check.cpp'), '-L' + str(bindir),
           '-lggml', '-lggml-cpu', '-lggml-base', '-ldl', '-pthread', '-o', str(test)]
for label, command in [
    ('qwen-thread-sweep-check-build', command),
    ('qwen-thread-sweep-check', [str(test), str(base / 'results/qwen-thread-limit-control')]),
]:
    path = base / 'results' / (label + '.log')
    with path.open('w') as output:
        result = subprocess.run(command, env=env, stdout=output, stderr=subprocess.STDOUT, timeout=180)
    print(label, result.returncode, flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
    if label.endswith('-check') and len(re.findall(r'^PASS ', path.read_text(), re.MULTILINE)) != 40:
        raise SystemExit('Qwen dynamic worker counts did not pass all 40 cases')

old = json.loads((base / 'results/q4e-goal-nibble-sort-mtp2-single4k-1024-t15/config.json').read_text())
cfg = dict(old['config'])
cfg.update(label='q4e-goal-nibble-sort-thread-sweep-1024', port=18123,
           binary=str(bindir / 'llama-server'), threads=16, numa_poll=100,
           thread_sweep='16,15,12,8', cache_check=True)
env = dict(os.environ, **old['runtime_env'])
# This runtime predates the Q5 triple assertion fix; its byte mode stays off.
env.update(GGML_CPU_X16_Q5_BYTES='0', GGML_CPU_X16_Q5_BYTES_BATCH3='0')
args = ['python3', str(base / 'qwen-goal-case.py'), cfg.pop('label')]
for key, value in cfg.items():
    if isinstance(value, bool):
        if value:
            args.append('--' + key.replace('_', '-'))
    elif value is not None:
        args.extend(['--' + key.replace('_', '-'), str(value)])
raise SystemExit(subprocess.call(args, env=env))
