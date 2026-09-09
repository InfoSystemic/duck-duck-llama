#!/usr/bin/env python3
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

root = Path(__file__).resolve().parents[2]
base = root / 'serving/fleet-0903'
waiting = base / 'results/glm5n-goal-q5-bytes-p4-q8mtp1-single4k-t15/result.json'
while True:
    try:
        if 'server_exit' in json.loads(waiting.read_text()):
            break
    except (FileNotFoundError, ValueError):
        pass
    time.sleep(3)
engine = root / 'engines/llama.cpp-q4e-goal-0904'
bindir = engine / 'build-goal/bin'
snapshot = engine / 'q5-bytes-p4-bin'
if not snapshot.exists():
    shutil.copytree(bindir, snapshot, symlinks=True)

def run(command, label, env=None, expected_failure=None):
    log = base / 'results' / (label + '.log')
    with log.open('w') as output:
        result = subprocess.run(command, env=env, stdout=output, stderr=subprocess.STDOUT)
    print(label, result.returncode, flush=True)
    if expected_failure:
        if not result.returncode or expected_failure not in log.read_text():
            raise SystemExit('Baseline did not reproduce the expected stale-snapshot failure')
    elif result.returncode:
        raise SystemExit(result.returncode)

command = ['g++', '-O2', '-std=c++17']
command += ['-I' + str(engine / p) for p in ['include', 'common', 'ggml/include', 'src', 'vendor']]
command += [str(engine / 'tests/test-recurrent-state-rollback.cpp'), '-L' + str(bindir),
            '-lllama-common', str(engine / 'build-goal/common/libllama-common-base.a'),
            '-lllama', '-lggml', '-lggml-cpu', '-lggml-base', '-pthread',
            '-o', str(base / 'qwen-rollback-range-baseline-check')]
run(command, 'qwen-rs-range-baseline-check-build')
env = dict(os.environ, LD_LIBRARY_PATH=str(snapshot), GGML_Q4E_RS_ROLLBACK='1',
           LLAMA_TEST_REQUIRE_RS='1', LLAMA_TEST_SPECULATIVE_RS='1', LLAMA_TEST_RS_VALID_RANGE='1')
run([str(base / 'qwen-rollback-range-baseline-check'), '-m',
     '/dev/shm/flash-goal-0904-qwen-rollback-fixtures/qwen4exp-ple-moe-split-experts.gguf',
     '-c', '256', '-t', '2', '--flash-attn', 'on', '--fit', 'off', '-ngl', '0'],
    'qwen-rs-range-baseline-failure', env, 'stale snapshot rejection and continuation failed')
run(['cmake', '--build', str(engine / 'build-goal'), '-j', '16', '--target', 'llama-server'],
    'qwen-rs-valid-range-build')
run(['python3', str(base / 'run-qwen-rollback-valid-range-fixtures.py')], 'qwen-rs-valid-range-fixtures')
old = json.loads((base / 'results/q4e-goal-rs-iq-r16-q5-pair-mtp2-single4k-t15/config.json').read_text())
cfg = old['config']
cfg.update(label='q4e-goal-rs-valid-mtp2-single4k-1024-t15', port=18118, bench_tokens=1024, cache_check=True, numa_poll=0)
env = dict(os.environ, **old['runtime_env'])
env['GGML_CPU_X16_Q5_BYTES'] = '0'
args = ['python3', str(base / 'qwen-goal-case.py'), cfg.pop('label')]
for key, value in cfg.items():
    if isinstance(value, bool):
        if value:
            args.append('--' + key.replace('_', '-'))
    elif value is not None:
        args.extend(['--' + key.replace('_', '-'), str(value)])
raise SystemExit(subprocess.call(args, env=env))
