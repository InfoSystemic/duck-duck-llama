#!/usr/bin/env python3
"""Compare measured Flash Q8 worker counts with unchanged weights and libraries."""
import argparse
import json
from pathlib import Path
import re

from compare_flash_q8_clamp_0908 import BASE, sha, stream_text


def without_workers(command, workers):
    values = list(command)
    for option in ('--threads', '--threads-batch'):
        assert values.count(option) == 1
        index = values.index(option)
        assert values[index + 1] == str(workers)
        del values[index:index + 2]
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate')
    parser.add_argument('--reference', default='glm-flash-q8-clamp-raw-0908')
    parser.add_argument('--parity', required=True)
    args = parser.parse_args()
    assert all(re.fullmatch(r'[A-Za-z0-9_-]+', value) for value in vars(args).values())
    reference_path = BASE / 'results' / args.reference / 'result.json'
    candidate_path = BASE / 'results' / args.candidate / 'result.json'
    parity_path = BASE / 'results' / args.parity / 'result.json'
    a, b, parity = (json.loads(path.read_text()) for path in (reference_path, candidate_path, parity_path))
    for result in (a, b):
        assert result.get('finished') and not result.get('error') and result['input_integrity_verified']
        assert all(check['pass_check'] for check in result['checks'])
        assert all(m['counter_metadata']['valid'] and not m['abort'] for m in result['measurements'])
    assert parity['passed'] and all(r['tokens_equal'] for r in parity['responses'])
    assert len(parity['cache_checks']) == 4 and all(c['passed'] for c in parity['cache_checks'])
    assert parity['current']['pid'] == b['target_pid']
    assert parity['current']['runtime_env'] == b['runtime_env']
    assert parity['current']['command'] == b['server_command']
    workers = [int(result['runtime_env']['GGML_CPU_NUMA_THREADS']) for result in (a, b)]
    assert all(1 <= value <= 16 for value in workers)
    assert without_workers(a['server_command'], workers[0]) == without_workers(b['server_command'], workers[1])
    differences = {k: [a['runtime_env'].get(k), b['runtime_env'].get(k)]
                   for k in a['runtime_env'].keys() | b['runtime_env'].keys()
                   if a['runtime_env'].get(k) != b['runtime_env'].get(k)}
    assert set(differences) <= {'GGML_CPU_NUMA_THREADS'}
    for key in ('port', 'alias', 'drafts', 'tokens', 'request_timeout_seconds', 'bandwidth_capacity_gb_s',
                'bandwidth_target_gb_s', 'chat_template_kwargs', 'reasoning_budget_tokens', 'check_reasoning_budget_tokens'):
        assert a['config'][key] == b['config'][key], key
    rows = []
    for newer in b['measurements']:
        kind = newer['kind']
        older = next(m for m in a['measurements'] if m['kind'] == kind)
        assert older['prompt'] == newer['prompt'] and older['draft_n'] == newer['draft_n'] == 0
        paths = [path.parent / f'{kind}-draft0/chunks.json' for path in (reference_path, candidate_path)]
        before, after = map(stream_text, paths)
        rows.append(dict(kind=kind,
            reference_tok_s=older['timings']['predicted_per_second'],
            candidate_tok_s=newer['timings']['predicted_per_second'],
            speed_change_percent=100 * (newer['timings']['predicted_per_second'] / older['timings']['predicted_per_second'] - 1),
            reference_adjusted_gb_s=older['background_subtracted_gb_s'],
            candidate_adjusted_gb_s=newer['background_subtracted_gb_s'],
            candidate_utilization=newer['background_subtracted_utilization'],
            completed_answer=newer['completed_answer'],
            stream_fields_equal={key: before[key] == after[key] for key in before},
            stream_sha256={str(path): sha(path) for path in paths}))
    result = dict(input_comparison_passed=True, workers_per_socket=workers, runtime_env_changes=differences,
        rows=rows, all_streams_equal=all(all(row['stream_fields_equal'].values()) for row in rows),
        target_reached=all(row['candidate_adjusted_gb_s'] >= 285 for row in rows),
        cpu_sha256=parity['current']['cpu_sha256'], server_sha256=sha(b['server_command'][0]),
        source_sha256={str(path): sha(path) for path in (Path(__file__), reference_path, candidate_path, parity_path)},
        note='Worker-count comparison on fixed raw Q8 prompts; repetition and broad quality are separate requirements.')
    destination = BASE / 'results' / (args.candidate + '-worker-comparison.json')
    assert not destination.exists()
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
