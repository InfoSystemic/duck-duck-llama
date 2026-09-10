#!/usr/bin/env python3
"""Recover the completed HC-only pair without treating the interrupted ABBA as complete."""
import json
from pathlib import Path
import re
import time

from compare_qwen_get_rows_bandwidth_0909 import read_run
from qwen_split_trial import sha256

BASE = Path(__file__).resolve().parent


def main():
    out = BASE / 'results/qwen-hc-partial-pair-0909.json'
    assert not out.exists()
    outer_path = BASE / 'results/qwen-decode-scheduling-model-0909/result.json'
    outer = json.loads(outer_path.read_text())
    assert not outer['passed'] and outer['finished'] and outer['error'] == 'AssertionError()'
    labels = [f'qwen-private-decode-scheduling-250-0909-{i}-{arm}' for i, arm in enumerate(('parent', 'candidate'))]
    runs = [read_run(label) for label in labels]
    reference = read_run('qwen-private-get-rows-250-0909-1-columns')
    evidence = {str(Path(__file__).resolve()): sha256(__file__), str(outer_path): sha256(outer_path)}
    for run in runs:
        plan = run['plan']
        for key in ('model_records', 'drafts', 'base', 'base_sha256', 'llama', 'server_sha256', 'peer', 'peer_runtime_env'):
            assert plan[key] == runs[0]['plan'][key]
        assert plan['command'][1:] == runs[0]['plan']['command'][1:]
        clean = dict(plan['runtime_env'])
        control = dict(runs[0]['plan']['runtime_env'])
        for env in (clean, control):
            for key in ('LD_LIBRARY_PATH', 'GGML_CPU_QWEN_HC_ORDERED_K', 'GGML_CPU_QWEN_HC_ROW_SPLIT', 'GGML_CPU_QWEN_Q6_MOE_TILE_ROWS'):
                env.pop(key, None)
        assert clean == control
        log_path = BASE / 'results' / run['label'] / 'model.log'
        log = log_path.read_text()
        used = [int(value) for value in re.findall(r'qwen4exp\.expert_used_count\s+u32\s+=\s+(\d+)', log)]
        experts = [int(value) for value in re.findall(r'qwen4exp\.expert_count\s+u32\s+=\s+(\d+)', log)]
        assert used == [10, 10] and experts == [512, 512], (used, experts)
        markers = dict(hc_ordered='QWEN_HC_ORDERED_K ' in log, hc_rows='QWEN_HC_ROW_SPLIT ' in log,
                       q6_tile='QWEN_Q6_MOE_TILE ' in log)
        enabled = plan['scheduling'] == 'candidate'
        assert markers == dict(hc_ordered=enabled, hc_rows=enabled, q6_tile=False)
        for kind, row in run['rows'].items():
            assert row['attribution_valid']
            assert all(row[key] == reference['rows'][kind][key] for key in
                       ('output_sha256', 'generated_tokens', 'draft_tokens', 'accepted_draft_tokens'))
        run['execution_markers'] = markers
        evidence.update(run['evidence'])
        evidence[str(log_path)] = sha256(log_path)
    operations_path = BASE / 'results/qwen-private-cpy-strides-0909/prose-operations.json'
    operations = json.loads(operations_path.read_text())
    shapes = set()

    def walk(value):
        if isinstance(value, dict):
            if value.get('op') == 'MUL_MAT_ID' and value.get('src0_type') == 'q6_K':
                shapes.add(tuple(value['src0_ne']))
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(operations)
    assert (2560, 160, 512) in shapes
    evidence[str(operations_path)] = sha256(operations_path)
    summaries = []
    for kind in ('prose', 'code'):
        before, after = [run['rows'][kind] for run in runs]
        summaries.append(dict(workload=kind, parent_tok_s=before['tok_s'], hc_only_tok_s=after['tok_s'],
                              speed_change_percent=100 * (after['tok_s'] / before['tok_s'] - 1),
                              parent_gb_s=before['adjusted_gb_s'], hc_only_gb_s=after['adjusted_gb_s'],
                              bandwidth_target_met=after['adjusted_gb_s'] >= 250))
    result = dict(time=time.time(), passed=True, comparison_complete=False, repeated_gain_established=False,
                  expert_tile_executed=False, actual_expert_used_count=10, actual_expert_count=512,
                  target_q6_weight_shapes=sorted(shapes), all_outputs_match=True, all_attribution_valid=True,
                  target_gb_s=250, target_reached=False, summaries=summaries,
                  runs=[dict(label=run['label'], rows=run['rows'], execution_markers=run['execution_markers']) for run in runs],
                  input_sha256=evidence,
                  scope='One qualified parent/HC-only pair. Both model children completed and exited cleanly; the outer ABBA stopped on a missing expert execution marker. The prototype required eight routes; the actual target and draft use ten. No expert-tile or repeated model gain is established.')
    with out.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps({key: value for key, value in result.items() if key not in ('input_sha256', 'runs')}, indent=2))


if __name__ == '__main__':
    main()
