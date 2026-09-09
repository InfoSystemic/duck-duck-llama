#!/usr/bin/env python3
"""Run the prepared hardware-thread matrix comparison after full P2 validation."""
import json
from pathlib import Path
import subprocess
import time

base = Path(__file__).resolve().parent
label = 'glm5n-goal-q5-compact-p2-1024-t15'
while True:
    try:
        result = json.loads((base / 'results' / label / 'result.json').read_text())
        if 'server_exit' in result:
            assert result['server_exit'] == 0 and not result.get('error')
            break
    except (FileNotFoundError, ValueError):
        pass
    time.sleep(3)
subprocess.run(['python3', str(base / 'validate-glm-full-reference.py'), label], check=True)
baseline = json.loads((base / 'results/glm5n-goal-x16-chunk16-1024-t15/result.json').read_text())

def score(r):
    samples = [c for c in r['checks'] if c['expected'] is None and c['reasoning_budget_tokens'] is None]
    return 0 if any(c['throughput_contended'] for c in samples) else min(c['response']['timings']['predicted_per_second'] for c in samples)

selected = max([baseline, result], key=score)
if score(selected) >= 20:
    print('GLM reached the measured target; skip further tuning.', flush=True)
    raise SystemExit(0)
bindir = 'build-goal/bin' if selected is result else 'validated-chunk16-bin'
print('SMT matrix reference:', selected['config']['label'], bindir, flush=True)
raise SystemExit(subprocess.call(['python3', str(base / 'run-glm-smt-kernel-check.py'),
                                 '--reference', selected['config']['label'], '--binary-dir', bindir]))
