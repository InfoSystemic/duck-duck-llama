#!/usr/bin/env python3
"""Reconcile the frozen R8 fixtures and all four model arms after Flash restoration."""
import fcntl
import json
import os
from pathlib import Path
import re
import statistics
import time

from benchmark_flash_q4_selected_0910 import validate_counters
from model_measurement_guard import ModelMeasurementGuard
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import inference_snapshot, sha256
from select_flash_q4_0910c import Manager
from trace_qwen_shared_dispatch_ops_0909 import output_record

BASE = Path(__file__).resolve().parent
OUT = BASE / 'results/qwen-r8-projection-audit-0910.json'
FLAGS = {'GGML_CPU_Q8_R8_K160_PREP', 'GGML_CPU_Q8_R8_SSM_TILE8'}
COUNTS = ('predicted_n', 'cache_n', 'draft_n', 'draft_n_accepted')


def read(path):
    path = Path(path)
    assert '.private.' not in str(path)
    return json.loads(path.read_text())


def check_hashes(mapping):
    for path, digest in mapping.items():
        assert '.private.' not in path
        assert sha256(path) == digest, path


def fixtures():
    directory = BASE / 'results/qwen-r8-projection-validation-0910'
    path = directory / 'result.json'
    result = read(path)
    assert result['passed'] and result['finished'] and result['peer_preserved']
    assert not result['model_loaded'] and not result['runtime_promoted']
    check_hashes(result['input_sha256'])
    assert sha256(directory / 'qwen-r8-projection.cpp') == result['fixture_sha256']
    assert sha256(directory / 'qwen-r8-projection') == result['binary_sha256']
    steps = {step['label']: step for step in result['steps']}
    for label, step in steps.items():
        assert step['exit_code'] == 0
        assert sha256(directory / (label + '.log')) == step['log_sha256']
    expected = {}
    totals = {False: 0, True: 0}
    coverage = []
    for run in result['runs']:
        label = f"run-{run['index']:02d}-{run['mode']}-t{run['threads']}-pad{int(run['padded'])}-timing{int(run['timing'])}"
        log = (directory / (label + '.log')).read_text()
        rows = [dict(field.split('=', 1) for field in line.split()[1:])
                for line in log.splitlines() if line.startswith('PASS ')]
        assert rows == run['rows'] and 'FAIL ' not in log
        assert len(rows) == (6 if run['timing'] else 30)
        digest = sha256(directory / (label + '.bin'))
        assert digest == run['output_sha256'] and run['exact_parent_outputs']
        key = (run['threads'], run['padded'], run['timing'])
        if key not in expected:
            assert run['mode'] in ['parent', 'off']
            expected[key] = digest
        assert digest == expected[key]
        trace, = re.findall(r'q8_0_r8: AVX-512/VNNI GEMV n=(\d+) nr=(\d+) nc=(\d+) x_tile=(\d+)', log)
        assert list(trace) == run['kernel_trace']
        assert int(trace[3]) == (8 if run['mode'] in ['ssm', 'both'] else 1)
        for row in rows:
            assert float(row['max_scaled']) <= 2e-4
            calls = int(row['calls'])
            specialized = not run['timing'] and run['mode'] in ['prep', 'both'] and int(row['k']) == 160 and int(row['moe']) == 1
            assert (calls > 0) if specialized else (calls == 0)
        totals[run['timing']] += len(rows)
        coverage.append(dict(index=run['index'], mode=run['mode'], threads=run['threads'], padded=run['padded'],
            timing=run['timing'], cases=len(rows), output_sha256=digest,
            specialized_calls=sum(int(row['calls']) for row in rows), ssm_tile=int(trace[3])))
    assert totals == {False: 330, True: 24}
    return dict(result_sha256=sha256(path), correctness_cases=totals[False], timing_cases=totals[True],
                all_exact_outputs=True, native_tolerance=2e-4, coverage=coverage)


def main():
    assert os.sched_getaffinity(0) == {127} and not OUT.exists()
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manager = Manager()
        current = manager.validate_current()
        guard = ModelMeasurementGuard(current['pid'], {current['pid']: 18131}, inference_snapshot)
        guard.assert_idle()
        fixture_result = fixtures()
        checks_path = BASE / 'results/qwen-r8-projection-controller-checks-0910.json'
        checks = read(checks_path)
        check_hashes(checks['sources'])
        assert checks['passed'] and len(checks['cases']) == 12 and all(c['passed'] for c in checks['cases'])
        directory = BASE / 'results/qwen-r8-projection-model-0910'
        plan_path, result_path = directory / 'plan.json', directory / 'result.json'
        plan, result = read(plan_path), read(result_path)
        assert result['plan_sha256'] == sha256(plan_path)
        check_hashes(plan['source_sha256'])
        assert result['passed'] and result['finished'] and result['stage'] == 'finished'
        assert all(result[k] for k in ['restored', 'full_environment_restored', 'restored_command_and_affinity', 'restored_libraries_match', 'all_outputs_match'])
        assert result['restored_flash_pid'] == current['pid'] and result['restored_flash_start'] == current['info']['start']
        assert current['info']['command'] == plan['peer']['command']
        assert not result['runtime_promoted'] and not result['target_reached'] and not result['tok_s_target_reached']
        assert [arm['mode'] for arm in result['trials']] == plan['schedule'] == ['off', 'both', 'both', 'off']
        rows, references = [], {}
        for arm in result['trials']:
            guard.assert_idle()
            assert arm['owned_qwen_exit'] == 0 and arm['dispatch_cleanup_valid'] and arm['outputs_and_counts_match']
            pid_path = Path('/proc') / str(arm['model_pid'])
            if pid_path.exists():
                assert pid_path.joinpath('stat').read_text().rsplit(')', 1)[1].split()[19] != arm['current']['start']
            runtime = arm['runtime_env']
            assert {k: v for k, v in runtime.items() if k not in FLAGS} == {k: v for k, v in plan['runtime_env'].items() if k not in FLAGS}
            assert all(runtime[k] == ('1' if arm['mode'] == 'both' else '0') for k in FLAGS)
            assert runtime['GGML_CPU_Q8_R8_K160_AUDIT'] == '0'
            assert arm['current']['command'] == plan['command']
            path = Path(arm['measurement'])
            assert sha256(path) == arm['measurement_sha256']
            measured = read(path)
            assert measured['finished'] and measured['input_integrity_verified'] and not measured.get('error')
            assert measured['server_command'] == plan['command'] and measured['runtime_env'] == runtime
            assert all(c['pass_check'] and not c['abort'] for c in measured['checks'])
            assert len(measured['measurements']) == 2
            for measurement in measured['measurements']:
                kind = measurement['kind']
                assert not measurement['abort'] and not measurement['other_inference'] and not measurement['inference_churn']
                sample_dir = path.parent / (kind + '-draft4')
                counters = validate_counters(sample_dir, measurement)
                _, timing, digest = output_record(read(sample_dir / 'chunks.json'))
                assert timing == measurement['timings'] and counters == arm['rows'][kind]['counters']
                counts = {k: timing.get(k, 0) for k in COUNTS}
                assert counts['cache_n'] == 0
                if kind not in references:
                    references[kind] = dict(output_sha256=digest, counts=counts)
                assert references[kind] == dict(output_sha256=digest, counts=counts)
                assert digest == arm['rows'][kind]['output_sha256']
                rows.append(dict(arm=arm['index'], mode=arm['mode'], kind=kind, tok_s=timing['predicted_per_second'],
                    total_decode_gb_s=measurement['decode']['total_gb_s'], counters=counters,
                    counts=counts, output_sha256=digest, completed_answer=measurement['completed_answer'],
                    retained_files={name: sha256(sample_dir / name) for name in ['perf.csv', 'samples.json', 'chunks.json']}))
        comparisons = []
        for kind in ['prose', 'code']:
            selected = [r for r in rows if r['kind'] == kind]
            assert [r['counters']['adjacent_idle_qualifies'] for r in selected] == [False, True, True, True]
            control = selected[-1]
            candidate = selected[1:3]
            comparisons.append(dict(kind=kind, excluded_unqualified_control_arms=[0], qualified_control_arm=3,
                qualified_control_tok_s=control['tok_s'], candidate_mean_tok_s=statistics.mean(r['tok_s'] for r in candidate),
                candidate_mean_change_percent=100 * (statistics.mean(r['tok_s'] for r in candidate) / control['tok_s'] - 1),
                adjacent_candidate_change_percent=100 * (candidate[-1]['tok_s'] / control['tok_s'] - 1),
                causal_gain_established=False, promotion_supported=False))
        manager.validate_current()
        guard.assert_idle()
        audit = dict(passed=True, finished=time.time(), source_sha256=sha256(__file__),
            fixture=fixture_result, controller_checks_sha256=sha256(checks_path), lifecycle_cases=12,
            model_result_sha256=sha256(result_path), plan_sha256=sha256(plan_path), rows=rows, comparisons=comparisons,
            selected_flash_pid=current['pid'], selected_flash_preserved=True, historical_restoration_reported_exact=True,
            all_outputs_and_counts_match=True, runtime_promoted=False, target_reached=False,
            limitations=['The first control fails the adjacent-idle attribution gate; the aggregate ABBA increase is confounded.',
                'The qualified final control is a useful comparison, not a repeated uncontended baseline.',
                'Full-model branch counters were disabled. Component branch coverage and controller library-map checks are separate evidence.',
                'Private restoration context was not read. Historical full environment restoration is asserted by the frozen controller.',
                'Component timing improvements do not establish a model speed gain. Prose ended at its 512-token cap.'])
        atomic_json(OUT, audit)
        print(json.dumps(dict(passed=True, correctness_cases=330, lifecycle_cases=12, measured_requests=len(rows),
            comparisons=comparisons, selected_flash_pid=current['pid'])))


if __name__ == '__main__':
    os.umask(0o077)
    main()
