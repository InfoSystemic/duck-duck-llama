#!/usr/bin/env python3
"""Compare a guarded Q8 tile trial against both saved pooling measurements."""
import argparse
import json
from pathlib import Path

from compare_flash_pool_0908 import sha, stream_text

BASE = Path(__file__).resolve().parent
REFERENCES = [BASE/f'results/{label}/result.json' for label in
              ('glm-flash-q8-pool-raw-0908', 'glm-flash-q8-pool-raw-repeat-0908')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('label')
    parser.add_argument('--parity-label', required=True)
    parser.add_argument('--tile-rows', type=int, choices=(32, 48, 64), required=True)
    args = parser.parse_args()
    assert all(label and set(label) <= set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-')
               for label in (args.label, args.parity_label))
    candidate_path = BASE/f'results/{args.label}/result.json'
    parity_path = BASE/f'results/{args.parity_label}/result.json'
    candidate = json.loads(candidate_path.read_text())
    parity = json.loads(parity_path.read_text())
    assert parity['passed'] and all(x['tokens_equal'] for x in parity['responses'])
    assert len(parity['cache_checks']) == 4 and all(x['passed'] for x in parity['cache_checks'])
    assert candidate['target_pid'] == parity['current']['pid']
    assert candidate['runtime_env'] == parity['current']['runtime_env']
    assert parity['current']['expert_tile_rows'] == args.tile_rows
    rows = []
    changes = []
    for reference_path in REFERENCES:
        reference = json.loads(reference_path.read_text())
        for result in (reference, candidate):
            assert result.get('finished') and not result.get('error') and result['input_integrity_verified']
            assert all(x['pass_check'] for x in result['checks'])
        assert reference['server_command'][1:] == candidate['server_command'][1:]
        assert sha(reference['server_command'][0]) == sha(candidate['server_command'][0])
        differences = {key: [reference['runtime_env'].get(key), candidate['runtime_env'].get(key)]
                       for key in reference['runtime_env'].keys() | candidate['runtime_env'].keys()
                       if reference['runtime_env'].get(key) != candidate['runtime_env'].get(key)}
        assert set(differences) == {'LD_LIBRARY_PATH', 'GGML_CPU_Q8_MOE_TILE_ROWS'}
        assert differences['GGML_CPU_Q8_MOE_TILE_ROWS'] == [None, str(args.tile_rows)]
        changes.append(dict(reference=str(reference_path), runtime_env_changes=differences))
        for key in ('port', 'alias', 'drafts', 'tokens', 'request_timeout_seconds', 'bandwidth_capacity_gb_s',
                    'bandwidth_target_gb_s', 'chat_template_kwargs', 'reasoning_budget_tokens', 'check_reasoning_budget_tokens'):
            assert reference['config'][key] == candidate['config'][key], key
        for newer in candidate['measurements']:
            kind = newer['kind']
            older = next(x for x in reference['measurements'] if x['kind'] == kind)
            assert older['prompt'] == newer['prompt'] and older['draft_n'] == newer['draft_n'] == 0
            before_path = reference_path.parent/f'{kind}-draft0/chunks.json'
            after_path = candidate_path.parent/f'{kind}-draft0/chunks.json'
            before, after = stream_text(before_path), stream_text(after_path)
            rows.append(dict(kind=kind, reference=str(reference_path),
                baseline_tok_s=older['timings']['predicted_per_second'],
                candidate_tok_s=newer['timings']['predicted_per_second'],
                speed_change_percent=100*(newer['timings']['predicted_per_second']/older['timings']['predicted_per_second']-1),
                baseline_adjusted_gb_s=older['background_subtracted_gb_s'],
                candidate_adjusted_gb_s=newer['background_subtracted_gb_s'],
                candidate_utilization=newer['background_subtracted_utilization'],
                completed_answer=newer['completed_answer'],
                stream_fields_equal={key: before[key] == after[key] for key in before},
                stream_sha256={str(path): sha(path) for path in (before_path, after_path)}))
    sources = [Path(__file__), BASE/'compare_flash_pool_0908.py', *REFERENCES, candidate_path, parity_path]
    result = dict(input_comparison_passed=True, rows=rows, changes=changes,
                  server_binary_sha256=sha(candidate['server_command'][0]),
                  source_sha256={str(path): sha(path) for path in sources},
                  all_streams_equal=all(all(row['stream_fields_equal'].values()) for row in rows),
                  target_reached=all(row['candidate_adjusted_gb_s'] >= 285 for row in rows),
                  note='Matched raw Q8 decode probes. Earlier controls are not interleaved. Text equality is not a complete token-ID or source-model quality comparison.')
    output = BASE/f'results/{args.label}-comparison.json'
    assert not output.exists()
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
