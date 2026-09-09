#!/usr/bin/env python3
"""Audit saved Full replay/Q5 captures; send no inference requests or signals."""
import ast
import csv
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import time

from dram_bandwidth import parse_records, summarize_samples
from model_measurement_guard import read_service
from qwen_split_trial import (identity_matches, inference_snapshot, process_environment,
                              process_info, runtime_environment, sha256)


def capture(directory, expected_metadata):
    saved = json.loads((directory / 'samples.json').read_text())
    meta = saved['metadata']
    assert meta == expected_metadata
    assert meta['valid'] and meta['exit_code'] == 0 and not meta['errors']
    assert meta['required_counters_per_interval'] == 48
    records, running = [], []
    for line in (directory / 'perf.csv').read_text().splitlines(True):
        row = next(csv.reader([line]))
        if len(row) >= 7 and row[1].startswith('CPU'):
            records.append((meta['anchor_monotonic'] + float(row[0]), line))
            running.append(float(row[6]))
    parsed, check = parse_records(records, meta['cpus'], meta['pmus'],
                                  {int(k): v for k, v in meta['socket_ids'].items()})
    assert check['valid'] and min(running) == max(running) == 100
    assert len(parsed) == len(saved['samples'])
    for actual, recorded in zip(parsed, saved['samples']):
        assert actual['valid'] and actual['sockets'] == recorded['sockets']
        assert actual['duration'] == recorded['duration']
        for key in ('start', 'end'):
            assert math.isclose(actual[key], recorded[key], rel_tol=0, abs_tol=1e-6)
    return saved['samples']


def text(chunks, key):
    return ''.join(c.get('delta', {}).get(key) or '' for chunk in chunks
                   for c in chunk.get('choices', []))


def main():
    base = Path(__file__).resolve().parent
    directory = base / 'results/glm53-full-replay-bandwidth-0906'
    destination = directory / 'validation-audit.json'
    assert not destination.exists()
    result = json.loads((directory / 'result.json').read_text())
    assert result['completed'] and result['input_integrity_verified'] and not result.get('error')
    assert result['idle_gate']['quiet_seconds'] >= 60 and len(result['measurements']) == 6
    for path, digest in result['source_sha256'].items():
        assert sha256(path) == sha256(directory / Path(path).name) == digest, path
    dataset = json.loads((directory / 'full-replay-workloads-0906.json').read_text())
    runtime = json.loads((base / 'results/glm53-draft-sweep-bandwidth-0905/result.json').read_text())
    assert result['runtime_env'] == runtime['runtime_env']
    assert result['server_command'] == runtime['server_command']
    audit = dict(started=time.time(), source_sha256=sha256(__file__),
                 result_sha256=sha256(directory / 'result.json'), measurements=[], components=[],
                 real_model_requests=0, real_model_signals=0)
    owned_pids = {result['pid']}
    for m in result['measurements']:
        assert not m['abort'] and not m['inference_churn']
        assert all(x['pid'] == 2308651 and x['cpu_percent'] < 1 for x in m['other_inference'])
        case = directory / f"task{m['workload']}-n{m['draft_n']}p{int(m['p_min'] * 100):02d}"
        samples = capture(case, m['counter_metadata'])
        decode = summarize_samples(samples, m['first_content_monotonic'] + .5,
                                   m['last_content_monotonic'] - .5)
        assert decode == m['decode'] and decode['sampled_seconds'] >= 4
        assert all(m[k]['valid'] and m[k]['sampled_seconds'] >= 4 for k in ('before', 'after'))
        adjusted = max(0, decode['total_gb_s'] - max(m[k]['total_gb_s'] for k in ('before', 'after')))
        assert adjusted == m['adjusted_gb_s']
        chunks = json.loads((case / 'chunks.json').read_text())
        timings = [c['timings'] for c in chunks if c.get('timings')][-1]
        assert timings == m['timings'] and 0 < timings['draft_n_accepted'] <= timings['draft_n']
        rate = 1000 * (timings['predicted_n'] - 1) / timings['predicted_ms']
        assert math.isclose(rate, timings['predicted_per_second'])
        content = text(chunks, 'content')
        assert content == m['content'] and len(text(chunks, 'reasoning_content')) == m['reasoning_characters']
        match = re.fullmatch(r'\s*```(?:python|py)?\r?\n(.*?)\r?\n```\s*', content, re.S)
        assert match, 'All six saved outputs should contain exactly one fenced code block'
        code = match.group(1)
        ast.parse(code)  # Validate syntax; never execute generated code.
        expected = dataset['workloads'][m['workload'] - 1]['expected_code']
        finish = [c['finish_reason'] for chunk in chunks for c in chunk.get('choices', []) if c.get('finish_reason')]
        passed = code.rstrip('\n') == expected.rstrip('\n') and finish == ['stop']
        assert passed == m['quality']['passed'] and finish == m['finish_reasons']
        assert passed == (m['workload'] in (1, 2)), 'Preserve the two task-3 quality failures'
        if not passed:
            assert code == 'class ConfigError(ValueError):\n    pass\n\n' + expected
        traffic = adjusted / rate
        assert math.isclose(traffic, m['approx_gb_per_generated_token'])
        audit['measurements'].append(dict(workload=m['workload'], draft_n=m['draft_n'], p_min=m['p_min'],
            quality_passed=passed, tok_s=rate, adjusted_gb_s=adjusted, utilization_percent=adjusted / 3.8,
            approx_gb_per_generated_token=traffic, projection93_tok_s=353.4 / traffic,
            projection100_tok_s=380 / traffic, draft_accepted=timings['draft_n_accepted'],
            draft_offered=timings['draft_n'], prompt_ms=timings['prompt_ms'], decode_ms=timings['predicted_ms'],
            decode_seconds=decode['sampled_seconds'], decode_intervals=decode['samples'],
            content_characters=len(content), reasoning_characters=m['reasoning_characters'],
            before_gb_s=m['before']['total_gb_s'], after_gb_s=m['after']['total_gb_s'],
            artifact_sha256={p.name: sha256(p) for p in case.iterdir() if p.is_file()}))
    assert result['all_outputs_exact'] is False
    for label in ('cold-q5-two-chain-0906', 'cold-q5-asymmetric-chain-0906'):
        root = base / 'results' / label
        d = json.loads((root / 'result.json').read_text())
        assert d['finished'] and not d.get('error') and not d['build_contention']
        assert d['idle_gate']['quiet_seconds'] >= 60 and len(d['runs']) == 6
        assert sha256(root / 'read-bandwidth-check.cpp') == d['source_sha256']
        assert sha256(root / 'read-bandwidth-check') == d['binary_sha256']
        # Private runner/workload sources intentionally evolved between these experiments.
        # Check each saved version, without requiring it to match the latest working copy.
        for collection in ('guard_sources', 'fixture_dependencies'):
            for path, digest in d[collection].items():
                assert sha256(root / Path(path).name) == digest, path
        for path, digest in d['libraries'].items():
            assert sha256(path) == digest, path
        candidate = d['candidate_source']
        assert sha256(candidate['source']) == candidate['source_sha256']
        assert sha256(root / 'cold-x16-candidates.h') == candidate['generated_sha256']
        original_batch = str(base / 'cold-q5-batch.h')
        assert sha256(original_batch) == d['fixture_dependencies'][original_batch]
        owned_pids.add(d['pid'])
        by_mode = {}
        for r in d['runs']:
            assert r['complete'] and not r['contention'] and not r['inference_churn']
            assert not r['before_state']['busy'] and not r['after_state']['busy']
            assert all(x['cpu_percent'] < 1 for x in r['other_inference'])
            assert r['measurement']['checksums_exact'] and r['ready']['verified_page_samples'] == 3840
            samples = capture(root / f"arm{r['index']}-imc", r['counter_metadata'])
            assert summarize_samples(samples, *r['stable_read_window']) == r['dram']
            for bounds, background in zip((r['background_before'], r['background_after']), r['background']):
                assert summarize_samples(samples, bounds[0] + .5, bounds[1] - .5) == background
            assert r['adjusted_read_gb_s'] == r['dram']['read_gb_s'] - max(b['read_gb_s'] for b in r['background'])
            by_mode.setdefault(r['mode'], []).append(r['measurement']['logical_gb_s'])
            owned_pids.add(r['ready']['pid'])
        analysis = json.loads((root / 'analysis.json').read_text())
        assert analysis['result_sha256'] == sha256(root / 'result.json') and not analysis['selected']
        medians = {mode: statistics.median(values) for mode, values in by_mode.items()}
        assert medians == analysis['median_logical_gb_s']
        audit['components'].append(dict(label=label, complete_arms=6, selected=False,
                                        median_logical_gb_s=medians, result_sha256=sha256(root / 'result.json')))
    prior = json.loads((base / 'results/qwen-expert-moe-split-0906-final-state.json').read_text())
    audit['verified_original_engine_sha256'] = {}
    for path, digest in prior['verified_original_engine_sha256'].items():
        assert sha256(path) == digest, path
        audit['verified_original_engine_sha256'][path] = digest
    mapped = prior['qwen_mapped_library']
    # Linux restricts opening map_files backing inodes even for the process owner.
    mapped_digest = subprocess.run(['sudo', '-n', 'sha256sum', mapped['map_file']],
                                   check=True, capture_output=True, text=True, timeout=15).stdout.split()[0]
    assert mapped_digest == mapped['sha256']
    audit['qwen_mapped_library'] = mapped
    plan = json.loads((directory / 'plan.json').read_text())
    audit['protected'] = {}
    for pid, expected in plan['protected'].items():
        info = process_info(int(pid))
        assert identity_matches(info, expected)
        if pid == '4005448':
            assert info['command'] == result['server_command']
            assert runtime_environment(process_environment(int(pid))) == result['runtime_env']
            assert info['affinity'] == [15, 31, 47, 63]
        else:
            assert info['command'] == plan['original_command'] and info['affinity'] == list(range(128))
            assert runtime_environment(process_environment(int(pid))) == plan['original_environment']
        audit['protected'][pid] = dict(start=info['start'], exe=info['exe'], affinity=info['affinity'],
                                        port=expected['port'], **read_service(expected['port']))
    assert set(inference_snapshot()) == set(plan['protected'])
    audit['recorded_owned_pids'] = sorted(owned_pids)
    audit['owned_pids_absent'] = all(not Path(f'/proc/{pid}').exists() for pid in owned_pids)
    assert audit['owned_pids_absent'], 'Inspect any reused PID before classifying its ownership'
    audit['note'] = ('Four of six replay outputs pass exact edits; task 3 adds an unrequested class under both profiles. '
                     'Replay ceilings use short stable systemwide counter windows with estimated background subtraction '
                     'and whole-request decode rates; they are approximate conditional projections, not attained speeds '
                     'or compute-inclusive bounds. Both private Q5 candidates remain unselected.')
    audit['finished'] = time.time()
    destination.write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(dict(audit=str(destination), replay_cases=len(audit['measurements']),
                         exact_replay_outputs=sum(m['quality_passed'] for m in audit['measurements']),
                         component_arms=12, engine_hashes=len(audit['verified_original_engine_sha256']),
                         owned_pids_absent=audit['owned_pids_absent'], protected=audit['protected'])))


if __name__ == '__main__':
    main()
