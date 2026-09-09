#!/usr/bin/env python3
"""Compare the guarded raw Flash clamp trial with its same-precision baseline."""
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
REFERENCE = BASE / 'results/glm-flash-q8-experts-raw-0908/result.json'
CANDIDATE = BASE / 'results/glm-flash-q8-clamp-raw-0908/result.json'
PARITY = BASE / 'results/glm-flash-q8-clamp-raw-parity-0908/result.json'
OUT = BASE / 'results/glm-flash-q8-clamp-model-comparison-0908.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stream_text(path):
    chunks = json.loads(path.read_text())
    return {key: ''.join(choice.get('delta', {}).get(key) or ''
                        for chunk in chunks for choice in chunk.get('choices', []))
            for key in ('content', 'reasoning_content')}


def main():
    a, b = (json.loads(path.read_text()) for path in (REFERENCE, CANDIDATE))
    parity = json.loads(PARITY.read_text())
    assert parity['passed'] and all(x['tokens_equal'] for x in parity['responses'])
    assert len(parity['cache_checks']) == 4 and all(x['passed'] for x in parity['cache_checks'])
    for result in (a, b):
        assert result.get('finished') and not result.get('error') and result['input_integrity_verified']
        assert all(x['pass_check'] for x in result['checks'])
    assert a['server_command'][1:] == b['server_command'][1:]
    assert sha(a['server_command'][0]) == sha(b['server_command'][0])
    assert b['target_pid'] == parity['current']['pid']
    assert b['runtime_env'] == parity['current']['runtime_env']
    differences = {key: [a['runtime_env'].get(key), b['runtime_env'].get(key)]
                   for key in a['runtime_env'].keys() | b['runtime_env'].keys()
                   if a['runtime_env'].get(key) != b['runtime_env'].get(key)}
    assert set(differences) == {'LD_LIBRARY_PATH', 'GGML_CPU_X16_Q8_CLAMP_FUSION'}
    assert differences['GGML_CPU_X16_Q8_CLAMP_FUSION'] == [None, '1']
    for key in ('port', 'alias', 'drafts', 'tokens', 'request_timeout_seconds', 'bandwidth_capacity_gb_s',
                'bandwidth_target_gb_s', 'chat_template_kwargs', 'reasoning_budget_tokens', 'check_reasoning_budget_tokens'):
        assert a['config'][key] == b['config'][key], key
    rows = []
    for newer in b['measurements']:
        kind = newer['kind']
        older = next(x for x in a['measurements'] if x['kind'] == kind)
        assert older['prompt'] == newer['prompt'] and older['draft_n'] == newer['draft_n'] == 0
        before_path = REFERENCE.parent / f'{kind}-draft0/chunks.json'
        after_path = CANDIDATE.parent / f'{kind}-draft0/chunks.json'
        before, after = stream_text(before_path), stream_text(after_path)
        rows.append(dict(kind=kind,
            baseline_tok_s=older['timings']['predicted_per_second'],
            candidate_tok_s=newer['timings']['predicted_per_second'],
            speed_change_percent=100 * (newer['timings']['predicted_per_second'] / older['timings']['predicted_per_second'] - 1),
            baseline_adjusted_gb_s=older['background_subtracted_gb_s'],
            candidate_adjusted_gb_s=newer['background_subtracted_gb_s'],
            candidate_utilization=newer['background_subtracted_utilization'],
            completed_answer=newer['completed_answer'],
            stream_fields_equal={key: before[key] == after[key] for key in before},
            stream_sha256={str(path): sha(path) for path in (before_path, after_path)}))
    result = dict(input_comparison_passed=True, rows=rows, runtime_env_changes=differences,
        server_binary_sha256=sha(b['server_command'][0]),
        source_sha256={str(path): sha(path) for path in (Path(__file__), REFERENCE, CANDIDATE, PARITY)},
        all_streams_equal=all(all(row['stream_fields_equal'].values()) for row in rows),
        target_reached=all(row['candidate_adjusted_gb_s'] >= 285 for row in rows),
        note='Matched raw Q8 decode probes. Small single-run changes do not establish repeatability or source-checkpoint quality.')
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
