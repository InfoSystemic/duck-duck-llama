#!/usr/bin/env python3
"""Audit the finished Flash comparison and the restored Qwen service."""
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import statistics
import time

from dram_bandwidth import parse_records, summarize_samples
from flash_bandwidth250_trial_0909 import measurement_rows
from glm_flash_q8_trial import memory_status, node_memory_status
from model_measurement_guard import ModelMeasurementGuard
from qwen_high_quant_trial import port_available, unit_state
from qwen_split_trial import inference_snapshot, process_environment, process_info, runtime_environment, sha256

BASE = Path(__file__).resolve().parent
ROOT = BASE / 'results/flash-bandwidth250-comparison-0909'
OUTPUT = BASE / 'results/flash-bandwidth250-assessment-0910.json'


def validate_counters(directory, row):
    saved = json.loads((directory / 'samples.json').read_text())
    metadata = saved['metadata']
    assert metadata == row['counter_metadata'] and metadata['valid']
    assert metadata['exit_code'] == 0 and metadata['required_counters_per_interval'] == 48
    assert metadata['cpus'] == [0, 16, 32, 48]
    assert metadata['pmus'] == ['uncore_imc_' + str(i) for i in range(6)]
    assert metadata['socket_ids'] == {'0': 0, '16': 1, '32': 2, '48': 3}
    records = []
    for line in (directory / 'perf.csv').read_text().splitlines():
        fields = next(csv.reader([line]))
        if len(fields) >= 7 and fields[1].startswith('CPU'):
            records.append((metadata['anchor_monotonic'] + float(fields[0]), line))
    parsed, capture = parse_records(records, metadata['cpus'], metadata['pmus'],
                                    {int(k): v for k, v in metadata['socket_ids'].items()})
    assert capture['valid'] and capture['required_counters_per_interval'] == 48
    assert len(parsed) == len(saved['samples'])
    for actual, expected in zip(parsed, saved['samples']):
        assert actual['valid'] and expected['valid'] and actual['sockets'] == expected['sockets']
        for key in ('start', 'end', 'duration', 'total_gb_s'):
            assert math.isclose(actual[key], expected[key], rel_tol=1e-12, abs_tol=1e-7), key
    decode = summarize_samples(saved['samples'], row['first_content_monotonic'] + .5,
                               row['last_content_monotonic'] - .5)
    assert decode == row['decode'] and decode['sampled_seconds'] >= 4
    before, after = row['baseline_before']['total_gb_s'], row['baseline_after']['total_gb_s']
    assert max(before, after) <= 19 and abs(before - after) <= 9.5
    adjusted = max(0, decode['total_gb_s'] - max(before, after))
    assert adjusted == row['background_subtracted_gb_s']
    return dict(intervals=len(parsed), decode_intervals=decode['samples'],
                decode_seconds=decode['sampled_seconds'], counters_per_interval=48,
                baseline_before_gb_s=before, baseline_after_gb_s=after,
                gross_decode_gb_s=decode['total_gb_s'], adjusted_decode_gb_s=adjusted,
                adjusted_sockets_gb_s=[
                    socket['total_gb_s'] - max(row['baseline_before']['sockets'][i]['total_gb_s'],
                                               row['baseline_after']['sockets'][i]['total_gb_s'])
                    for i, socket in enumerate(decode['sockets'])])


def analyze():
    assert not OUTPUT.exists()
    path = ROOT / 'result.json'
    result, plan = [json.loads(p.read_text()) for p in (path, ROOT / 'plan.json')]
    assert result['passed'] and result['finished'] and not result.get('error')
    assert result['qwen_restored'] and not result.get('restoration_error')
    assert result['plan_sha256'] == sha256(ROOT / 'plan.json')
    assert plan['target_gb_s'] == 250 and plan['capacity_gb_s'] == 380
    assert [run['configuration'] for run in result['runs']] == plan['sequence']
    assert len(result['runs']) == 8
    sources = {str(p): sha256(p) for p in (Path(__file__).resolve(), path, ROOT / 'plan.json',
               BASE / 'dram_bandwidth.py', BASE / 'flash_bandwidth250_trial_0909.py',
               BASE / 'qwen_split_trial.py', BASE / 'model_measurement_guard.py')}
    assert all(sha256(p) == digest for p, digest in plan['source_sha256'].items())
    checked = []
    for run in result['runs']:
        measurement = Path(run['measurement'])
        assert sha256(measurement) == run['measurement_sha256']
        current = plan['configurations'][run['configuration']]
        rows = measurement_rows(measurement, current)
        assert rows == run['rows']
        data = json.loads(measurement.read_text())
        assert data['target_pid'] == run['model_pid']
        assert data['config']['allowed_idle_pids'] == ''
        assert data['config']['chat_template_kwargs'] == {'reasoning_effort': 'max'}
        assert data['config']['reasoning_budget_tokens'] is None
        counters = {}
        for row in data['measurements']:
            directory = measurement.parent / f"{row['kind']}-draft{row['draft_n']}"
            counters[row['kind']] = validate_counters(directory, row)
            for name in ('samples.json', 'perf.csv', 'chunks.json', 'command.json'):
                source = directory / name
                sources[str(source)] = sha256(source)
        sources[str(measurement)] = sha256(measurement)
        checked.append(dict(index=run['index'], configuration=run['configuration'], rows=rows,
                            counters=counters, measurement_sha256=run['measurement_sha256']))
    summaries = {}
    for name in plan['configurations']:
        runs = [run for run in checked if run['configuration'] == name]
        assert len(runs) == 2
        reference = next(run for run in checked if run['configuration'] == name.split('_')[0] + '_control')
        summaries[name] = {}
        for kind in ('prose', 'code'):
            rows = [run['rows'][kind] for run in runs]
            for row in rows:
                assert all(row[k] == reference['rows'][kind][k] for k in (
                    'output_sha256', 'generated_tokens', 'draft_tokens', 'accepted_draft_tokens'))
            summaries[name][kind] = dict(
                mean_tok_s=statistics.mean(row['tok_s'] for row in rows),
                mean_adjusted_gb_s=statistics.mean(row['adjusted_gb_s'] for row in rows),
                tok_s_range=[min(row['tok_s'] for row in rows), max(row['tok_s'] for row in rows)],
                adjusted_gb_s_range=[min(row['adjusted_gb_s'] for row in rows), max(row['adjusted_gb_s'] for row in rows)],
                both_over_250=all(row['over_250_gb_s'] for row in rows),
                complete_answers=all(row['completed_answer'] for row in rows))
    comparisons = []
    for mode in ('raw', 'mtp'):
        for kind in ('prose', 'code'):
            control, candidate = [summaries[mode + '_' + arm][kind] for arm in ('control', 'candidate')]
            comparisons.append(dict(mode=mode, kind=kind,
                mean_speed_change_percent=100 * (candidate['mean_tok_s'] / control['mean_tok_s'] - 1),
                mean_bandwidth_change_percent=100 * (candidate['mean_adjusted_gb_s'] / control['mean_adjusted_gb_s'] - 1)))
    targets = {name: all(row['both_over_250'] for row in kinds.values()) for name, kinds in summaries.items()}
    assert targets == result['configuration_targets'] and any(targets.values()) == result['target_reached']
    peer_pid = result['restored_qwen_pid']
    peer = process_info(peer_pid)
    assert all(peer[key] == result['restored_qwen'][key] for key in ('start', 'exe', 'command', 'cwd', 'affinity'))
    assert all(peer[key] == plan['qwen'][key] for key in ('exe', 'command', 'cwd', 'affinity'))
    assert runtime_environment(process_environment(peer_pid)) == plan['qwen_runtime_env']
    assert set(inference_snapshot()) == {str(peer_pid)}
    ModelMeasurementGuard(peer_pid, {peer_pid: 18095}, inference_snapshot).assert_idle()
    mapped = {line.split()[-1] for line in Path(f'/proc/{peer_pid}/maps').read_text().splitlines()
              if any('/' + stem in line for stem in ('libggml-', 'libggml.so.', 'libllama.so.'))}
    assert mapped == set(plan['qwen_libraries'])
    assert all(sha256(p) == digest for p, digest in plan['qwen_libraries'].items())
    assert unit_state()['ActiveState'] == 'inactive'
    assert all(port_available(port) for port in (18131, 18155, 18161))
    for state_path in ('results/glm-flash-q8-trial-0908/state.json', 'results/qwen-q6-trial-0907/state.json'):
        assert json.loads((BASE / state_path).read_text())['current'] is None
    audit = dict(time=time.time(), passed=True, source_sha256=sources, runs=checked,
        configurations=summaries, candidate_comparisons=comparisons, configuration_targets=targets,
        flash_target_reached=any(targets.values()), all_model_goal_complete=False,
        restored_qwen_pid=peer_pid, restored_qwen_start=peer['start'], exact_runtime_restore_verified=True,
        controller_verified_full_environment_restore=True, private_restore_context_read=False,
        only_restored_qwen_loaded=True, full_inactive=True, private_ports_free=True,
        memory=memory_status(), node_memory=node_memory_status(),
        scope='Eight isolated fresh single-conversation decode runs, two prose/code windows per configuration. '
              'All 48 IMC counters are reparsed and stable decode summaries reproduced. '
              'Idle subtraction estimates model-attributable traffic. Output and draft counts match within mode. '
              'Generated reasoning-token rates are not completed-answer rates. This does not establish a model ceiling '
              'or near-lossless checkpoint quality, and cannot complete the three-model goal.')
    with OUTPUT.open('x') as handle:
        json.dump(audit, handle, indent=2)
        handle.write('\n')
    print(json.dumps({key: audit[key] for key in ('passed', 'configurations', 'candidate_comparisons',
          'flash_target_reached', 'restored_qwen_pid', 'restored_qwen_start')}, indent=2))


if __name__ == '__main__':
    assert os.sched_getaffinity(0) == {127}
    os.umask(0o077)
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        analyze()
