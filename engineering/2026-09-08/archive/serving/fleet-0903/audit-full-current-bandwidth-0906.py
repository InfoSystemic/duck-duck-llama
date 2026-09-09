#!/usr/bin/env python3
"""Validate the saved Full counter capture and read current service state."""
import csv
import json
import math
from pathlib import Path
import time

from dram_bandwidth import parse_records, summarize_samples
from model_measurement_guard import read_service
from qwen_split_trial import identity_matches, process_info, sha256


def stream_text(chunks, key):
    return ''.join(choice.get('delta', {}).get(key) or ''
                   for chunk in chunks for choice in chunk.get('choices', []))


def main():
    base = Path(__file__).resolve().parent
    directory = base / 'results/glm53-full-current-bandwidth-0906'
    destination = directory / 'validation-audit.json'
    assert not destination.exists()
    result = json.loads((directory / 'result.json').read_text())
    assert result.get('finished') and not result.get('error') and result['input_integrity_verified']
    assert len(result['checks']) == 2 and all(c['pass_check'] and not c['abort'] for c in result['checks'])
    assert result['idle_gate']['quiet_seconds'] >= 60
    assert len(result['measurements']) == 2
    for path, digest in result['input_sha256'].items():
        assert sha256(path) == sha256(directory / Path(path).name) == digest, path
    old_directory = base / 'results/glm53-full-bandwidth-baseline-0905c'
    old = json.loads((old_directory / 'result.json').read_text())
    runtime_reference = base / 'results/glm53-draft-sweep-bandwidth-0905/result.json'
    runtime = json.loads(runtime_reference.read_text())
    assert result['runtime_env'] == runtime['runtime_env']
    assert result['server_command'] == runtime['server_command']
    audit = dict(time=time.time(), result_sha256=sha256(directory / 'result.json'),
                 source_sha256=result['input_sha256'], measurements=[], protected={},
                 runtime_and_command_match_prior=True, runtime_reference=str(runtime_reference),
                 real_model_requests=0)
    for measurement in result['measurements']:
        assert measurement['draft_n'] == 2 and not measurement['abort'] and not measurement['inference_churn']
        assert all(x['pid'] == 2308651 and x['cpu_percent'] < 1 for x in measurement['other_inference'])
        case = directory / (measurement['kind'] + '-draft2')
        capture = json.loads((case / 'samples.json').read_text())
        metadata = capture['metadata']
        assert metadata == measurement['counter_metadata']
        assert metadata['valid'] and metadata['exit_code'] == 0 and not metadata['errors']
        assert metadata['required_counters_per_interval'] == 48
        records, running = [], []
        for line in (case / 'perf.csv').read_text().splitlines(True):
            row = next(csv.reader([line]))
            if len(row) >= 7 and row[1].startswith('CPU'):
                records.append((metadata['anchor_monotonic'] + float(row[0]), line))
                running.append(float(row[6]))
        samples, check = parse_records(records, metadata['cpus'], metadata['pmus'],
                                       {int(k): v for k, v in metadata['socket_ids'].items()})
        assert check['valid'] and min(running) == max(running) == 100
        assert len(samples) == len(capture['samples'])
        for actual, saved in zip(samples, capture['samples']):
            assert actual['sockets'] == saved['sockets'] and actual['duration'] == saved['duration']
            assert math.isclose(actual['start'], saved['start'], rel_tol=0, abs_tol=1e-6)
            assert math.isclose(actual['end'], saved['end'], rel_tol=0, abs_tol=1e-6)
        decode = summarize_samples(capture['samples'], measurement['first_content_monotonic'] + .5,
                                   measurement['last_content_monotonic'] - .5)
        assert decode == measurement['decode'] and decode['sampled_seconds'] > 4
        for key in ('baseline_before', 'baseline_after'):
            assert measurement[key]['valid'] and measurement[key]['sampled_seconds'] >= 4
        background = max(measurement[key]['total_gb_s'] for key in ('baseline_before', 'baseline_after'))
        adjusted = decode['total_gb_s'] - background
        assert adjusted == measurement['background_subtracted_gb_s']
        chunks = json.loads((case / 'chunks.json').read_text())
        timings = [chunk['timings'] for chunk in chunks if chunk.get('timings')][-1]
        assert timings == measurement['timings'] and timings['predicted_n'] == 512
        assert 0 < timings['draft_n_accepted'] <= timings['draft_n']
        rate = timings['predicted_per_second']
        assert math.isclose(rate, 511000 / timings['predicted_ms'])
        traffic = adjusted / rate
        previous = next(x for x in old['measurements'] if x['kind'] == measurement['kind'] and x['draft_n'] == 2)
        prior_chunks = json.loads((old_directory / (measurement['kind'] + '-draft2') / 'chunks.json').read_text())
        content = stream_text(chunks, 'content')
        reasoning = stream_text(chunks, 'reasoning_content')
        audit['measurements'].append(dict(
            kind=measurement['kind'], tok_s=rate, adjusted_gb_s=adjusted, utilization_percent=adjusted / 3.8,
            approx_gb_per_generated_token=traffic, projection93_tok_s=353.4 / traffic,
            projection100_tok_s=380 / traffic, draft_accepted=timings['draft_n_accepted'],
            draft_offered=timings['draft_n'], prior_draft_accepted=previous['timings']['draft_n_accepted'],
            prior_draft_offered=previous['timings']['draft_n'],
            decode_seconds=decode['sampled_seconds'], decode_intervals=decode['samples'],
            all_capture_intervals=len(samples), minimum_counter_running_percent=min(running),
            content_characters=len(content), reasoning_characters=len(reasoning),
            reasoning_same_as_prior=reasoning == stream_text(prior_chunks, 'reasoning_content'),
            finish_reasons=measurement['finish_reasons'], other_host_cpu=measurement['other_host_cpu'],
            other_inference=measurement['other_inference'],
            artifact_sha256={p.name: sha256(p) for p in case.iterdir() if p.is_file()}))
    plan = json.loads((base / 'results/qwen-even-split-model-trial-0906-staging/plan.json').read_text())
    for pid, expected in plan['protected'].items():
        info = process_info(int(pid))
        assert identity_matches(info, expected)
        if pid == '4005448':
            assert info['command'] == result['server_command']
        audit['protected'][pid] = dict(start=info['start'], exe=info['exe'], port=expected['port'],
                                       affinity=info['affinity'], **read_service(expected['port']))
    prior_audit = json.loads((base / 'results/qwen-expert-moe-split-0906-final-state.json').read_text())
    verified = {}
    for path, expected in prior_audit['verified_original_engine_sha256'].items():
        if '/llama.cpp-sr950-glm/build-dev2/bin/' in path:
            assert sha256(path) == expected, path
            verified[path] = expected
    assert verified
    audit['verified_full_binary_sha256'] = verified
    audit['note'] = ('These 512-token Full windows contain reasoning tokens and reach the length limit '
                     'before final-answer content. Generated-token rates are not completed-answer rates. '
                     'Outputs and acceptance differ from the earlier samples; no engine speedup is inferred. '
                     'IMC counters are systemwide and background subtraction is an attribution estimate.')
    audit['finished'] = time.time()
    destination.write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(dict(audit=str(destination), measurements=[{k: v for k, v in x.items()
                         if k not in ('other_host_cpu', 'artifact_sha256')} for x in audit['measurements']],
                         verified_full_binary_entries=len(verified), protected=audit['protected'], note=audit['note'])))


if __name__ == '__main__':
    main()
