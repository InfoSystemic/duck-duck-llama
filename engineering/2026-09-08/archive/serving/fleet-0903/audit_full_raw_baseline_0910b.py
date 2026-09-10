#!/usr/bin/env python3
"""Audit retained traffic and restoration after the Full repeatability gate rejected the run."""
import fcntl
import json
import os
from pathlib import Path
import time

from benchmark_flash_q4_selected_0910 import validate_counters
from model_measurement_guard import ModelMeasurementGuard
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import inference_snapshot, sha256
from select_flash_q4_0910c import Manager, mapped_libraries
from trace_qwen_shared_dispatch_ops_0909 import output_record

BASE = Path(__file__).resolve().parent
OUT = BASE / 'results/full-raw-baseline-audit-0910b.json'


def read(path):
    assert '.private.' not in str(path)
    return json.loads(Path(path).read_text())


def main():
    assert os.sched_getaffinity(0) == {127} and not OUT.exists()
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manager = Manager()
        current = manager.validate_current()
        guard = ModelMeasurementGuard(current['pid'], {current['pid']: 18131}, inference_snapshot)
        guard.assert_idle()
        directory = BASE / 'results/full-raw-baseline-0910'
        result, plan = read(directory / 'result.json'), read(directory / 'plan.json')
        assert result['stage'] == 'finished' and result['finished'] and not result['passed']
        assert result['error'] == "AssertionError('Raw Full repeat changed output or counts')"
        assert not result['runtime_promoted']
        assert result['plan_sha256'] == sha256(directory / 'plan.json')
        for path, digest in plan['source_sha256'].items():
            assert '.private.' not in path and sha256(path) == digest
        assert all(result[k] for k in ['restored', 'owned_full_stopped', 'mapped_libraries_verified',
            'full_environment_restored', 'restored_command_and_affinity', 'restored_libraries_match'])
        assert result['restored_flash_pid'] == current['pid'] and result['restored_flash_start'] == current['info']['start']
        assert current['info']['command'] == plan['peer']['command']
        assert mapped_libraries(current['pid']) == set(plan['peer_libraries'])
        pid_path = Path('/proc') / str(result['full_pid'])
        if pid_path.exists():
            assert pid_path.joinpath('stat').read_text().rsplit(')', 1)[1].split()[19] != result['full_info']['start']
        assert plan['command'][plan['command'].index('--spec-type') + 1] == 'none'
        assert '--spec-draft-model' not in plan['command'] and '--no-cache-prompt' in plan['command']
        assert plan['context'] == 32768 and plan['workers_per_socket'] == 15
        checks = read(BASE / 'results/full-raw-controller-checks-0910.json')
        assert checks['passed'] and len(checks['checks']) == 12 and all(c['passed'] for c in checks['checks'])
        rows, references, output_differences = [], {}, []
        assert len(result['runs']) == 2
        for run in result['runs']:
            observed_match = True
            path = Path(run['measurement'])
            assert sha256(path) == run['measurement_sha256']
            data = read(path)
            assert data['finished'] and data['input_integrity_verified'] and not data.get('error')
            assert data['server_command'] == plan['command'] and data['runtime_env'] == plan['runtime_env']
            assert all(c['pass_check'] and not c['abort'] for c in data['checks'])
            assert len(data['measurements']) == 2
            for item in data['measurements']:
                assert not item['abort'] and not item['other_inference'] and not item['inference_churn']
                sample = path.parent / (item['kind'] + '-draft0')
                counters = validate_counters(sample, item)
                output, timing, digest = output_record(read(sample / 'chunks.json'))
                assert timing == item['timings'] and counters == run['rows'][item['kind']]['counters']
                assert digest == run['rows'][item['kind']]['output_sha256']
                assert timing['cache_n'] == 0 and timing.get('draft_n', 0) == 0 and timing.get('draft_n_accepted', 0) == 0
                signature = dict(output_sha256=digest, generated_tokens=timing['predicted_n'])
                assert timing['predicted_n'] == 512 and not item['completed_answer']
                assert item['answer_characters'] == 0 and item['reasoning_characters'] > 0
                if item['kind'] not in references:
                    references[item['kind']] = (signature, output)
                else:
                    reference_signature, reference_output = references[item['kind']]
                    assert signature['generated_tokens'] == reference_signature['generated_tokens']
                    matches = signature == reference_signature
                    observed_match = observed_match and matches
                    common_prefix = 0
                    for a, b in zip(reference_output[1], output[1]):
                        if a != b: break
                        common_prefix += 1
                    output_differences.append(dict(kind=item['kind'], repetition=run['index'],
                        matches=matches, generated_counts_match=True, first_output_sha256=reference_signature['output_sha256'],
                        repeated_output_sha256=digest, common_reasoning_prefix_characters=common_prefix,
                        first_reasoning_characters=len(reference_output[1]), repeated_reasoning_characters=len(output[1])))
                rows.append(dict(repetition=run['index'], kind=item['kind'], tok_s=timing['predicted_per_second'],
                    counters=counters, output_sha256=digest, generated_tokens=timing['predicted_n'],
                    cache_tokens=0, draft_tokens=0, total_decode_gb_s=item['decode']['total_gb_s'],
                    completed_answer=item['completed_answer'],
                    retained_sha256={name: sha256(sample / name) for name in ['perf.csv', 'samples.json', 'chunks.json']}))
            assert run['outputs_and_counts_match'] == observed_match
            guard.assert_idle()
        assert len(output_differences) == 2 and all(not row['matches'] for row in output_differences)
        qualified = all(row['counters']['adjacent_idle_qualifies'] for row in rows)
        reached240 = qualified and all(row['counters']['adjusted_gb_s'] >= 240 for row in rows)
        reached250 = qualified and all(row['counters']['adjusted_gb_s'] >= 250 for row in rows)
        assert qualified and not reached240 and not reached250 and not result['target_reached']
        manager.validate_current(); guard.assert_idle()
        audit = dict(passed=True, finished=time.time(), source_sha256=sha256(__file__),
            model_result_sha256=sha256(directory / 'result.json'), plan_sha256=sha256(directory / 'plan.json'),
            lifecycle_cases=12, rows=rows, all_outputs_and_counts_match=False, generated_counts_match=True,
            model_run_passed=False, repeatability_gate_correctly_rejected=True, output_differences=output_differences,
            all_attribution_valid=qualified, full_raw_240_gb_s_reached=reached240, full_raw_250_gb_s_reached=reached250,
            selected_flash_pid=current['pid'], selected_flash_preserved=True, runtime_promoted=False,
            limitations=['Two prose/code pairs share one fresh Full load. They are repeated measurements, not independent reloads.',
                'Outputs differ under the same greedy seed-42 requests. Root cause is not established; this is not a qualified tuning comparison.',
                'All four benchmark responses hit 512 generated tokens with reasoning content only and no completed answer.',
                'Raw decode removes the draft model. It does not establish MTP throughput or bandwidth.',
                'System-wide physical DRAM read plus write is adjusted by the larger adjacent idle baseline.',
                'Historical exact full environment restoration is reported by the frozen controller; private context is not read.',
                'Existing mixed Q4 runtime precision and Q8 KV remain in use at context 32768. No new quantization or general quality claim.'])
        atomic_json(OUT, audit)
        print(json.dumps(dict(passed=True, model_run_passed=False, output_differences=output_differences, rows=rows, full_raw_240_gb_s_reached=reached240,
                             full_raw_250_gb_s_reached=reached250, selected_flash_pid=current['pid'])))


if __name__ == '__main__':
    os.umask(0o077)
    main()
