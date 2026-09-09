#!/usr/bin/env python3
"""Check completed GLM experiments against their complete reference messages."""
import argparse
import json
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument('label')
p.add_argument('--reference', default='glm5n-goal-iq-batch3-best-threads-1024')
a = p.parse_args()
base = Path(__file__).resolve().parent
out = base / 'results' / a.label
refdir = base / 'results' / a.reference
result = json.loads((out / 'result.json').read_text())
reference = json.loads((refdir / 'result.json').read_text())
assert result.get('server_exit') == 0 and not result.get('error'), 'Run is incomplete or failed'
checks = list(result['checks'])
seen = {c['response']['id'] for c in checks}
for key in ['thread_sweep', 'draft_sweep']:
    for group in result.get(key, []):
        for check in group['checks']:
            if check['response']['id'] not in seen:
                checks.append(check)
                seen.add(check['response']['id'])
comparisons = []
for i, check in enumerate(checks):
    assert check['expected'] is None or check['pass'], f'Short check {i} failed'
    matching = [old for old in reference['checks']
                if (old['prompt'], old.get('reasoning_budget_tokens')) ==
                   (check['prompt'], check.get('reasoning_budget_tokens'))]
    assert len(matching) == 1
    same = check['response']['choices'][0]['message'] == matching[0]['response']['choices'][0]['message']
    assert same, f'Full message {i} differs from reference'
    comparisons.append(dict(index=i, effective_draft_n=check.get('effective_draft_n'),
                            effective_numa_threads=check.get('effective_numa_threads'), same_message=same))
assert len(result['cache_checks']) == 4 and all(c['pass'] for c in result['cache_checks'])
(out / 'response-identity-check.json').write_text(json.dumps({
    'reference': str(refdir), 'checks': comparisons, 'all_identical': True}, indent=2) + '\n')
if result['config'].get('bench_direct'):
    assert any(c['reasoning_budget_tokens'] == 0 and 'Python function' in c['prompt'] for c in checks)
    proof = json.loads((refdir / 'generated-direct-code-check.json').read_text())
    proof['identical_code_reference'] = str(refdir)
    (out / 'generated-direct-code-check.json').write_text(json.dumps(proof, indent=2) + '\n')
for script, active in [('summarize-cpu-profile.py', result['config']['profile_cpu']),
                       ('summarize-phase-profile.py', result['config']['profile_phase'])]:
    if active:
        subprocess.run(['python3', str(base / script), str(out)], check=True, stdout=subprocess.DEVNULL)
summary = {'label': a.label, 'identical_full_messages': len(checks), 'cache_checks_pass': 4,
           'long_samples': [dict(draft_n=c.get('effective_draft_n'),
                                 direct=c['reasoning_budget_tokens'] == 0,
                                 tokens=c['response']['timings']['predicted_n'],
                                 tok_s=c['response']['timings']['predicted_per_second'],
                                 finish_reason=c['response']['choices'][0]['finish_reason'],
                                 inference_contended=c['throughput_contended'])
                            for c in checks if c['expected'] is None]}
print(json.dumps(summary, indent=2))
