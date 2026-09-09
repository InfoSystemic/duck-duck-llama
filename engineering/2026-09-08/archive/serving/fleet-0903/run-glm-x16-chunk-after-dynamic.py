#!/usr/bin/env python3
"""Try the existing dense x16 chunk limit after the current dynamic-IQ full run."""
import json
import os
from pathlib import Path
import re
import subprocess
import time

root = Path(__file__).resolve().parents[2]
base = root / 'serving/fleet-0903'
previous = base / 'results/glm5n-goal-dynamic-iq-tiles-1024-t15'
while True:
    try:
        finished = json.loads((previous / 'result.json').read_text())
        if 'server_exit' in finished:
            if finished.get('error') or finished['server_exit'] != 0:
                raise SystemExit('Previous run did not finish successfully')
            break
    except (FileNotFoundError, ValueError):
        pass
    time.sleep(3)

def score(result):
    checks = [c for c in result['checks']
              if c['expected'] is None and c['reasoning_budget_tokens'] is None]
    if result.get('error') or any(c['throughput_contended'] for c in checks):
        return 0
    return min(c['response']['timings']['predicted_per_second'] for c in checks)

if score(finished) >= 20:
    print('Dynamic IQ reached the throughput target; skip this experiment.', flush=True)
    raise SystemExit(0)
references = [finished, json.loads((base / 'results/glm5n-goal-compact-p4-draft-sweep-1024-t15/result.json').read_text())]
selected = max(references, key=score)
(base / 'results/glm-x16-chunk-reference.json').write_text(json.dumps({
    'selected': selected['config']['label'],
    'scores': {r['config']['label']: score(r) for r in references}}, indent=2) + '\n')
engine = root / 'engines/llama.cpp-glm5n-goal-0904'
bindir = engine / 'build-goal/bin'

def run(command, label, env=None, timeout=900):
    log = base / 'results' / (label + '.log')
    with log.open('w') as output:
        result = subprocess.run(command, env=env, stdout=output, stderr=subprocess.STDOUT, timeout=timeout)
    print(label, result.returncode, flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
    return log.read_text()

env = {k: v for k, v in os.environ.items()
       if not k.startswith(('GGML_', 'LLAMA_GRAPH_PHASE', 'LLAMA_MTP_DRAFT_N_FILE', 'OMP_', 'GOMP_'))}
env.update({k: v for k, v in selected['runtime_env'].items()
            if not k.endswith('_FILE') and not k.startswith('GGML_CPU_OP_PROFILE')})
env['LD_LIBRARY_PATH'] = str(bindir)
flags = ['g++', '-O2', '-std=c++17']
flags += ['-I' + str(engine / p) for p in ['include', 'ggml/include', 'ggml/src', 'ggml/src/ggml-cpu']]
links = ['-L' + str(bindir), '-lggml', '-lggml-cpu', '-lggml-base', '-ldl', '-pthread']
test = base / 'glm-x16-chunk-check'
run(flags + [str(base / 'iq2-repack-check.cpp')] + links + ['-o', str(test)], 'glm-x16-chunk-check-build')
summary = []
for padded in [False, True]:
    for mode, count in [('q5-pair', 90), ('q8', 60)]:
        hashes = []
        for maximum in [64, 16]:
            case_env = dict(env, GGML_CPU_NUMA_DEVICES='0', GGML_CPU_X16_CHUNK_MAX=str(maximum),
                            REPACK_TEST_SMALL_BATCHES='1', REPACK_TEST_DENSE_WORK_SHARING='1')
            if padded:
                case_env.update(REPACK_TEST_PADDED='1', REPACK_TEST_DOWN='1')
            label = 'glm-x16-chunk-' + mode + ('-padded' if padded else '-standard') + '-' + str(maximum)
            log = run(['numactl', '--physcpubind=0-15', '--membind=0', str(test), mode], label, case_env)
            hashes.append(re.findall(r'hash=([0-9a-f]+)', log))
        if len(hashes[0]) != count or hashes[0] != hashes[1]:
            raise SystemExit('Dense chunk scheduling changed numerical outputs')
        summary.append(dict(mode=mode, padded=padded, cases=count, output_hashes_identical=True))
(base / 'results/glm-x16-chunk-check-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
env['GGML_CPU_X16_CHUNK_MAX'] = '16'
cfg = dict(selected['config'])
cfg.update(label='glm5n-goal-x16-chunk16-1024-t15', port=18131,
           draft_n=2, draft_sweep=None, thread_sweep=None)
args = ['python3', str(base / 'qwen-goal-case.py'), cfg.pop('label')]
for key, value in cfg.items():
    if isinstance(value, bool):
        if value:
            args.append('--' + key.replace('_', '-'))
    elif value is not None:
        args.extend(['--' + key.replace('_', '-'), str(value)])
raise SystemExit(subprocess.call(args, env=env))
