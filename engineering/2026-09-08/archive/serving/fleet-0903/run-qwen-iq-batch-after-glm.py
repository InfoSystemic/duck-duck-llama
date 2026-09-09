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
waiting = base / 'results/glm5n-goal-iq-batch3-best-threads-1024/result.json'
while True:
    try:
        measured_glm = json.loads(waiting.read_text())
        if 'server_exit' in measured_glm:
            break
    except (FileNotFoundError, ValueError):
        pass
    time.sleep(3)
if measured_glm.get('server_exit') != 0 or not all(check['pass'] for check in measured_glm['checks'][:3]):
    raise SystemExit('GLM shared-expert candidate did not finish its short checks successfully')
proof = json.loads((base / 'results/glm-iq-batch-check-summary.json').read_text())
assert len(proof) == 2 and all(case['cases'] == 360 and case['output_hashes_identical'] for case in proof)

engine = root / 'engines/llama.cpp-q4e-goal-0904'
bindir = engine / 'build-goal/bin'
snapshot = engine / 'nibble-sort-bin'
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

patch = base / 'glm-iq-r16-batch3.patch'
source = engine / 'ggml/src/ggml-cpu/repack.cpp'
run(['git', '-C', str(engine), 'apply', '--check', str(patch)], 'qwen-iq-batch-patch-check')
backup = source.with_name(source.name + '.before-goal-iq-r16-batch3')
assert not backup.exists()
shutil.copy2(source, backup)
run(['git', '-C', str(engine), 'apply', str(patch)], 'qwen-iq-batch-patch')
shutil.copy2(patch, base / 'q4e-iq-r16-batch3.patch')
run(['cmake', '--build', str(engine / 'build-goal'), '-j', '16', '--target', 'llama-server'], 'qwen-iq-batch-build')
flags = ['g++', '-O2', '-std=c++17']
flags += ['-I' + str(engine / p) for p in ['include', 'ggml/include', 'ggml/src', 'ggml/src/ggml-cpu']]
links = ['-L' + str(bindir), '-lggml', '-lggml-cpu', '-lggml-base', '-ldl', '-pthread']
test = base / 'qwen-iq-batch-check'
run(flags + [str(base / 'iq2-repack-check.cpp')] + links + ['-o', str(test)], 'qwen-iq-batch-check-build')
summary = []
for padded in [False, True]:
    hashes = []
    for enabled in [False, True]:
        env = dict(os.environ, LD_LIBRARY_PATH=str(bindir), GGML_CPU_IQ_R16_REPACK='1',
                   GGML_CPU_IQ_R16_NIBBLE2='1', GGML_CPU_IQ_R16_BATCH3=str(int(enabled)),
                   REPACK_TEST_SMALL_BATCHES='1')
        if padded:
            env.update(REPACK_TEST_PADDED='1', REPACK_TEST_DOWN='1')
        label = 'qwen-iq-batch-check-' + ('padded' if padded else 'standard') + '-' + ('on' if enabled else 'off')
        log = run(['numactl', '--physcpubind=0-3', '--membind=0', str(test), 'iq-r16-qwen'], label, env, timeout=600)
        hashes.append(re.findall(r'hash=([0-9a-f]+)', log))
    if hashes[0] != hashes[1] or len(hashes[0]) != 240:
        raise SystemExit('Qwen IQ batch kernel changed outputs')
    summary.append(dict(padded=padded, cases=240, output_hashes_identical=True))
(base / 'results/qwen-iq-batch-check-summary.json').write_text(json.dumps(summary, indent=2) + '\n')

previous = base / 'results/q4e-goal-nibble-sort-thread-sweep-1024'
old = json.loads((previous / 'config.json').read_text())
measured = json.loads((previous / 'result.json').read_text())
cfg = dict(old['config'])
best_threads = measured['selected_numa_threads']
cfg.update(label='q4e-goal-iq-batch3-threads-1024', port=18125,
           binary=str(bindir / 'llama-server'), threads=max(best_threads, 10),
           thread_sweep=','.join(map(str, dict.fromkeys([max(best_threads, 10), best_threads, 10]))),
           cache_check=True, bench_tokens=1024)
env = dict(os.environ, **old['runtime_env'])
env.pop('GGML_CPU_NUMA_THREADS_FILE', None)
env.update(GGML_CPU_IQ_R16_BATCH3='1', GGML_CPU_X16_Q5_BYTES='0', GGML_CPU_X16_Q5_BYTES_BATCH3='0')
args = ['python3', str(base / 'qwen-goal-case.py'), cfg.pop('label')]
for key, value in cfg.items():
    if isinstance(value, bool):
        if value:
            args.append('--' + key.replace('_', '-'))
    elif value is not None:
        args.extend(['--' + key.replace('_', '-'), str(value)])
raise SystemExit(subprocess.call(args, env=env))
