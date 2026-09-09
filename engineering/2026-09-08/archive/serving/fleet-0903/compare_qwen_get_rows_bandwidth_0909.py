#!/usr/bin/env python3
"""Compare repeated parent/corrected gather model runs at the 250 GB/s target."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time

BASE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_run(label):
    assert re.fullmatch(r'qwen-private-[A-Za-z0-9_-]+', label)
    out = BASE / 'results' / label
    result_path, plan_path = out / 'result.json', out / 'plan.json'
    result, plan = [json.loads(path.read_text()) for path in (result_path, plan_path)]
    assert result['passed'] and result['finished'] and result['peer_preserved']
    assert result['model_started'] and result['owned_model_exit'] == 0 and result['dispatch_cleanup_valid']
    assert result['plan_sha256'] == digest(plan_path)
    assert all(digest(path) == value for path, value in plan['source_sha256'].items())
    measurement_path = Path(result['measurement'])
    assert result['measurement_sha256'] == digest(measurement_path)
    measured = json.loads(measurement_path.read_text())
    assert measured['input_integrity_verified'] and measured['finished'] and not measured.get('error')
    assert all(digest(path) == value for path, value in measured['input_sha256'].items())
    assert measured['target_pid'] == result['model_pid']
    assert measured['server_command'] == plan['command'] and measured['runtime_env'] == plan['runtime_env']
    assert measured['bandwidth_capacity_gb_s'] == 380 and measured['bandwidth_target_gb_s'] == plan['target_gb_s'] == 250
    assert len(measured['checks']) == 2 and all(row['pass_check'] and not row['abort'] for row in measured['checks'])
    assert len(measured['measurements']) == 2
    rows = {}
    evidence = {str(path):digest(path) for path in (result_path, plan_path, measurement_path)}
    for row in measured['measurements']:
        assert row['kind'] not in rows and row['kind'] in ('prose', 'code')
        assert row['counter_metadata']['valid'] and row['counter_metadata']['exit_code'] == 0
        assert not row['abort'] and not row['inference_churn']
        assert all(item['cpu_percent'] <= 20 for item in row['other_inference'])
        assert row['timings']['cache_n'] == 0 and row['draft_n'] == plan['drafts']
        assert bool(row['timings'].get('draft_n', 0)) == bool(plan['drafts'])
        chunks_path = measurement_path.parent / f"{row['kind']}-draft{plan['drafts']}" / 'chunks.json'
        chunks = json.loads(chunks_path.read_text())
        content, reasoning = [], []
        for chunk in chunks:
            for choice in chunk.get('choices', []):
                delta = choice.get('delta', {})
                content.append(delta.get('content') or '')
                reasoning.append(delta.get('reasoning_content') or delta.get('reasoning') or '')
        output = json.dumps([ ''.join(content), ''.join(reasoning), row['finish_reasons'] ], ensure_ascii=False)
        assert row['timings']['predicted_n'] >= 128
        before = row['baseline_before']['total_gb_s']
        after = row['baseline_after']['total_gb_s']
        attribution_valid = max(before, after) <= .05 * 380 and abs(before - after) <= .025 * 380
        rows[row['kind']] = dict(tok_s=row['timings']['predicted_per_second'],
            attribution_valid=attribution_valid, idle_before_gb_s=before, idle_after_gb_s=after,
            adjusted_gb_s=row['background_subtracted_gb_s'],
            utilization=row['background_subtracted_gb_s'] / 380,
            output_sha256=hashlib.sha256(output.encode()).hexdigest(),
            generated_tokens=row['timings']['predicted_n'],
            draft_tokens=row['timings'].get('draft_n', 0),
            accepted_draft_tokens=row['timings'].get('draft_n_accepted', 0),
            observed_other_host_cores=sum(item['cpu_percent'] for item in row['other_host_cpu']) / 100)
        evidence[str(chunks_path)] = digest(chunks_path)
    assert set(rows) == {'prose', 'code'}
    return dict(label=label, plan=plan, rows=rows, evidence=evidence)



def compare(labels):
    assert len(labels) == 4 and len(set(labels)) == 4
    runs = [read_run(label) for label in labels]
    assert [run['plan']['gather'] for run in runs] == ['parent', 'columns', 'columns', 'parent']
    baseline = runs[0]
    for run in runs:
        plan, control = run['plan'], baseline['plan']
        assert plan['shared_dispatch'] == 'on' and plan['quantize_blocks'] == 'off'
        for key in ('drafts', 'model_records', 'base_sha256', 'base', 'server_sha256', 'llama', 'peer', 'peer_runtime_env'):
            assert plan[key] == control[key], key
        assert plan['command'][1:] == control['command'][1:]
        pe, ce = dict(plan['runtime_env']), dict(control['runtime_env'])
        pe.pop('LD_LIBRARY_PATH')
        ce.pop('LD_LIBRARY_PATH')
        assert pe == ce
        expected_cpu = {'parent': 'e68c0c543aa3334a1f1ed732c11f8f4fe8ecf097ea816eb89f82fe9ae1ed6a0a',
                        'columns': '12c61b337736ca9210433f57c64ce7fffbf7e4b66920aba9a03f97eaef3fd9b7'}
        assert plan['cpu_sha256'] == expected_cpu[plan['gather']]
        for kind in ('prose', 'code'):
            row, reference = run['rows'][kind], baseline['rows'][kind]
            row['output_matches_control'] = all(row[key] == reference[key] for key in
                ('output_sha256', 'generated_tokens', 'draft_tokens', 'accepted_draft_tokens'))
            row['bandwidth_target_met'] = row['attribution_valid'] and row['adjusted_gb_s'] >= 250
            row['speed_target_met'] = row['tok_s'] > 30
    summaries = []
    for kind in ('prose', 'code'):
        control = [runs[i]['rows'][kind] for i in (0, 3)]
        candidate = [runs[i]['rows'][kind] for i in (1, 2)]
        speed_parent = sum(row['tok_s'] for row in control) / 2
        speed_fixed = sum(row['tok_s'] for row in candidate) / 2
        bandwidth_parent = sum(row['adjusted_gb_s'] for row in control) / 2
        bandwidth_fixed = sum(row['adjusted_gb_s'] for row in candidate) / 2
        valid = all(row['attribution_valid'] for row in [*control, *candidate])
        summaries.append(dict(workload=kind, parent_mean_tok_s=speed_parent, corrected_mean_tok_s=speed_fixed,
            speed_change_percent=100 * (speed_fixed / speed_parent - 1), all_attribution_valid=valid,
            parent_mean_gb_s=bandwidth_parent, corrected_mean_gb_s=bandwidth_fixed,
            bandwidth_change_percent=(100 * (bandwidth_fixed / bandwidth_parent - 1)) if valid else None,
            repeated_bandwidth_target_met=all(row['bandwidth_target_met'] for row in candidate)))
    return dict(time=time.time(), runs=[dict(label=run['label'], gather=run['plan']['gather'], rows=run['rows']) for run in runs],
        summaries=summaries, all_outputs_match=all(row['output_matches_control'] for run in runs for row in run['rows'].values()),
        all_candidate_bandwidth_targets_met=all(row['repeated_bandwidth_target_met'] for row in summaries),
        evidence={path: value for run in runs for path, value in run['evidence'].items()}, analyzer_sha256=digest(__file__),
        scope='Fresh single-conversation prose/code decode, parent/corrected/corrected/parent. Same Q6 target and Q8 MTP4 draft. Adjacent idle-subtracted IMC bandwidth; generated tok/s reported separately. No broad quality certification or service promotion.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('labels', nargs=4)
    parser.add_argument('--output-label', required=True)
    args = parser.parse_args()
    assert re.fullmatch(r'qwen-get-rows-comparison-[A-Za-z0-9_-]+', args.output_label)
    path = BASE / 'results' / (args.output_label + '.json')
    assert not path.exists()
    result = compare(args.labels)
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'evidence'}, indent=2))
