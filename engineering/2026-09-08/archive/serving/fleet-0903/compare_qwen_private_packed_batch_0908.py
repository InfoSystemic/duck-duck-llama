#!/usr/bin/env python3
"""Compare completed private Q6 runs using their actual outputs and IMC records."""
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
    assert result['model_started'] and result['owned_model_exit'] is not None
    assert result['plan_sha256'] == digest(plan_path)
    assert all(digest(path) == value for path, value in plan['source_sha256'].items())
    measurement_path = Path(result['measurement'])
    assert result['measurement_sha256'] == digest(measurement_path)
    measured = json.loads(measurement_path.read_text())
    assert measured['input_integrity_verified'] and measured['finished'] and not measured.get('error')
    assert all(digest(path) == value for path, value in measured['input_sha256'].items())
    assert measured['target_pid'] == result['model_pid']
    assert measured['server_command'] == plan['command'] and measured['runtime_env'] == plan['runtime_env']
    assert measured['bandwidth_capacity_gb_s'] == 380 and measured['bandwidth_target_gb_s'] == 190
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
        rows[row['kind']] = dict(tok_s=row['timings']['predicted_per_second'],
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


def compare(control, candidates):
    assert control['plan']['batch'] == 'off'
    comparisons = []
    for candidate in candidates:
        cp, pp = control['plan'], candidate['plan']
        assert pp['batch'] == 'on'
        for key in ('command', 'drafts', 'model_records', 'cpu_sha256', 'server_sha256', 'llama'):
            assert cp[key] == pp[key], key
        ce, pe = dict(cp['runtime_env']), dict(pp['runtime_env'])
        assert ce.pop('GGML_CPU_X16_Q6_EXPERT_BATCH') == '0'
        assert pe.pop('GGML_CPU_X16_Q6_EXPERT_BATCH') == '1'
        assert ce == pe
        workloads = []
        for kind in ('prose', 'code'):
            baseline, row = control['rows'][kind], candidate['rows'][kind]
            equal = all(row[key] == baseline[key] for key in ('output_sha256', 'generated_tokens', 'draft_tokens', 'accepted_draft_tokens'))
            workloads.append(dict(workload=kind, **row, output_matches_control=equal,
                speed_change_percent=100 * (row['tok_s'] / baseline['tok_s'] - 1),
                bandwidth_change_percent=100 * (row['adjusted_gb_s'] / baseline['adjusted_gb_s'] - 1),
                both_thresholds_met=row['tok_s'] > 30 and row['adjusted_gb_s'] > 190))
        comparisons.append(dict(label=candidate['label'], workloads=workloads))
    return comparisons


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('control')
    parser.add_argument('candidates', nargs='+')
    parser.add_argument('--output-label', required=True)
    args = parser.parse_args()
    assert re.fullmatch(r'qwen-private-comparison-[A-Za-z0-9_-]+', args.output_label)
    assert len(set(args.candidates)) == len(args.candidates)
    output = BASE / 'results' / (args.output_label + '.json')
    assert not output.exists(), 'Preserve the previous comparison'
    control = read_run(args.control)
    candidates = [read_run(label) for label in args.candidates]
    rows = compare(control, candidates)
    result = dict(time=time.time(), control=control['label'], control_rows=control['rows'], comparisons=rows,
        all_outputs_match=all(row['output_matches_control'] for run in rows for row in run['workloads']),
        all_candidate_thresholds_met=all(row['both_thresholds_met'] for run in rows for row in run['workloads']),
        candidate_repeat_available=len(candidates) >= 2,
        evidence={path:value for run in [control, *candidates] for path,value in run['evidence'].items()},
        analyzer_sha256=digest(__file__),
        scope='Fresh single-conversation decode. IMC attribution uses adjacent idle subtraction. This comparison does not itself certify broad quality or mark a goal complete.')
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key:value for key,value in result.items() if key != 'evidence'}, indent=2))
