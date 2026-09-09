#!/usr/bin/env python3
"""Compare qualified raw and speculative Qwen measurements without assuming output equality."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import time

from compare_qwen_get_rows_bandwidth_0909 import digest, read_run

BASE = Path(__file__).resolve().parent


def without_speculation(command):
    output = []
    index = 0
    while index < len(command):
        if command[index].startswith('--spec-'):
            assert index + 1 < len(command) and not command[index + 1].startswith('--')
            index += 2
        else:
            output.append(command[index])
            index += 1
    return output


def main(labels, output_label):
    assert len(labels) == len(set(labels))
    runs = [read_run(label) for label in labels]
    reference = next(run for run in runs if run['plan']['drafts'] == 4)
    assert {run['plan']['drafts'] for run in runs} == {0, 3, 4}
    evidence = {str(Path(__file__).resolve()): digest(__file__)}
    by_depth = defaultdict(list)
    for run in runs:
        plan = run['plan']
        assert plan['gather'] == 'columns' and plan['shared_dispatch'] == 'on'
        assert plan['cpu_sha256'] == '12c61b337736ca9210433f57c64ce7fffbf7e4b66920aba9a03f97eaef3fd9b7'
        assert not plan['profile']
        for key in ('cpu', 'base', 'base_sha256', 'llama', 'server_sha256', 'model_records', 'runtime_env', 'peer', 'peer_runtime_env'):
            assert plan[key] == reference['plan'][key], key
        assert without_speculation(plan['command']) == without_speculation(reference['plan']['command'])
        if plan['drafts']:
            normalized = list(plan['command'])
            normalized[normalized.index('--spec-draft-n-max') + 1] = '4'
            assert normalized == reference['plan']['command']
        else:
            assert without_speculation(plan['command']) == plan['command']
        result_path = BASE / 'results' / run['label'] / 'result.json'
        result = json.loads(result_path.read_text())
        if plan['drafts'] != 4:
            assert result['peer_environment_preserved']
        measured = json.loads(Path(result['measurement']).read_text())
        assert all(row['counter_metadata']['required_counters_per_interval'] == 48 for row in measured['measurements'])
        for kind, row in run['rows'].items():
            row['output_matches_mtp4_reference'] = row['output_sha256'] == reference['rows'][kind]['output_sha256']
            row['bandwidth_target_met'] = row['attribution_valid'] and row['adjusted_gb_s'] >= 250
        evidence.update(run['evidence'])
        by_depth[plan['drafts']].append(run)
    summaries = []
    for depth, group in sorted(by_depth.items()):
        rows = {}
        for kind in ('prose', 'code'):
            samples = [run['rows'][kind] for run in group]
            rows[kind] = dict(mean_tok_s=sum(row['tok_s'] for row in samples) / len(samples),
                             mean_adjusted_gb_s=sum(row['adjusted_gb_s'] for row in samples) / len(samples),
                             all_attribution_valid=all(row['attribution_valid'] for row in samples),
                             outputs_match_within_mode=len({row['output_sha256'] for row in samples}) == 1,
                             repeated_bandwidth_target_met=len(samples) >= 2 and all(row['bandwidth_target_met'] for row in samples))
        summaries.append(dict(drafts=depth, completed_runs=len(group), repeated=len(group) >= 2, rows=rows))
    analysis = dict(time=time.time(), passed=True, evidence=evidence,
                    runs=[dict(label=run['label'], drafts=run['plan']['drafts'], rows=run['rows']) for run in runs],
                    summaries=summaries,
                    repeated_configuration_reaches_250_gb_s=any(
                        all(row['repeated_bandwidth_target_met'] for row in group['rows'].values()) for group in summaries),
                    scope='Same corrected Q6 target runtime, worker settings and fresh prompts. Speculative runs retain the Q8 draft and p_min=0.3; only launch-time draft depth differs.',
                    limitations=[
                        'Raw and MTP3 have one run each; MTP4 has two preceding corrected-gather runs.',
                        'The mode comparison is not an interleaved repeated trial and does not establish a mode speedup with identical outputs.',
                        'Full output hashes are compared explicitly. Matching two short answers is not broad quality certification.',
                        'These qualified whole-server counters use adjacent idle subtraction; neither components nor instrumented requests contribute.'])
    path = BASE / 'results' / (output_label + '.json')
    with path.open('x') as handle:
        json.dump(analysis, handle, indent=2)
        handle.write('\n')
    print(json.dumps({key: value for key, value in analysis.items() if key != 'evidence'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('labels', nargs=4)
    parser.add_argument('--output-label', required=True)
    args = parser.parse_args()
    assert re.fullmatch(r'qwen-decode-modes-[A-Za-z0-9_-]+', args.output_label)
    main(args.labels, args.output_label)
