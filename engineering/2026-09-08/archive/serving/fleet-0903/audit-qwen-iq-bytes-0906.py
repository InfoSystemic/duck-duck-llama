#!/usr/bin/env python3
"""Reconcile private Qwen IQ byte captures and read current service state."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import statistics
import subprocess
import time

from dram_bandwidth import summarize_samples
from model_measurement_guard import read_service
from qwen_split_trial import (identity_matches, inference_snapshot, process_environment,
                              process_info, runtime_environment, sha256)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', nargs='+', required=True)
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    destination = base / 'results/qwen-iq-bytes-0906-audit.json'
    assert not destination.exists()
    helper = base / 'audit-full-replay-bandwidth-0906.py'
    spec = importlib.util.spec_from_file_location('capture_audit', helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    audit = dict(started=time.time(), source_sha256={str(p): sha256(p) for p in (Path(__file__), helper)},
                 experiments=[], verified_original_engine_sha256={}, real_model_requests=0, real_model_signals=0)
    owned = set()
    for label in args.labels:
        assert '/' not in label and label.startswith('qwen-iq-bytes-')
        root = base / 'results' / label
        d = json.loads((root / 'result.json').read_text())
        assert d['completed'] and d['finished'] and not d.get('error') and not d.get('guard_failure')
        assert d['idle_gate']['quiet_seconds'] >= 60 and len(d['phases']) == 8
        assert not d['before_state']['busy'] and not d['after_state']['busy']
        assert all(code == 0 for code in d['owned_exit_codes'].values())
        owned.add(d['pid'])
        owned.update(int(pid) for pid in d['owned_exit_codes'])
        for path, digest in d['input_sha256'].items():
            copy = root / Path(path).name
            if copy.exists():
                assert sha256(copy) == digest, str(copy)
            if Path(path).name != 'run-qwen-iq-byte-rows.py':
                assert sha256(path) == digest, path
            else:
                assert copy.exists(), 'The runner version for this case must be preserved'
        cpu = d['private_cpu']
        assert sha256(cpu['library']) == cpu['library_sha256']
        assert sha256(Path(cpu['library']).parent / 'repack.cpp') == cpu['private_source_sha256']
        assert sha256(Path(cpu['library']).parent / 'repack.cpp.o') == cpu['object_sha256']
        for collection in (cpu['input_sha256'], d['private_policy']['input_sha256']):
            for path, digest in collection.items():
                assert sha256(path) == digest, path
                if '/engines/' in path:
                    prior = audit['verified_original_engine_sha256'].setdefault(path, digest)
                    assert prior == digest
        assert sha256(d['private_policy']['library']) == d['private_policy']['library_sha256']
        samples = module.capture(root / 'imc', d['counter_metadata'])
        for bounds, background in zip(d['background_windows'], d['backgrounds']):
            assert summarize_samples(samples, *bounds) == background and background['valid']
        expected_values = 3 * d['config']['matrices'] * 2560 * 10 * d['config']['tokens']
        quality = d['cross_layout_outputs']
        assert quality['values'] == expected_values and quality['tolerance'] == 0
        assert quality['max_absolute_error'] == quality['max_scaled_error'] == 0
        assert len(set(quality['sha256'].values())) == 1
        for path, digest in quality['sha256'].items():
            assert Path(path).stat().st_size == expected_values * 4 and sha256(path) == digest
        for mode, fixture in d['fixtures'].items():
            ready = fixture['ready']
            assert fixture['exit_code'] == 0 and ready['down_checked'] and ready['max_down_scaled_error'] <= 2e-5
            assert ready['workers'] == 60 and ready['controller_cores'] == 4
            assert ready['chunk'] == d['config']['split_granularity']
            assert ready['tokens'] == d['config']['tokens'] and ready['weight_type'] == d['config']['qwen_weight_type']
            assert all(n['pages'] > 0 and n['nonlocal_pages'] == 0 for n in fixture['placement']['nodes'].values())
            mapped_cpu = [p for p in fixture['mapped_libraries'] if '/libggml-cpu.so.' in p]
            assert len(mapped_cpu) == 1
            if mode == 'candidate':
                assert mapped_cpu == [cpu['library']]
            else:
                assert '/validated-iq-batch3-bin/' in mapped_cpu[0]
            log = (root / f'fixture-{mode}.log').read_text()
            enabled = mode == 'candidate' and ready['weight_type'] != 'q8_0'
            assert ('IQ_R16_BYTES block=4256 ' in log) == enabled
            block_bytes = 4256 if enabled else 2208
            copies = d['config']['matrices']
            per_expert = copies * (2 * 40 * 10 * block_bytes + 2560 * 20 * 18)
            if ready['weight_type'] == 'q8_0':
                per_expert = copies * (2 * 640 * 80 * 34 + 2560 * 20 * 34)
            pool = per_expert * 32
            per_pass = per_expert * (10 + 2 * (ready['tokens'] - 1))
            assert ready['packed_weight_bytes'] == d['resident_weight_bytes'][mode] == pool
            assert ready['bytes_per_pass'] == d['logical_bytes_per_pass'][mode] == per_pass
            assert all(n['pages'] * fixture['placement']['page_size'] >= pool // 4
                       for n in fixture['placement']['nodes'].values())
        assert len({f['ready']['weight_hash'] for f in d['fixtures'].values()}) == 1
        assert len({f['ready']['output_hash'] for f in d['fixtures'].values()}) == 1
        for phase in d['warmups'] + d['phases']:
            measured = phase['measurement']
            assert measured['passes'] > 0 and measured['checksums_exact']
            assert measured['bytes'] == measured['passes'] * d['logical_bytes_per_pass'][phase['mode']]
            assert all(load <= 5 for load in phase['inactive_fixture_cpu_percent'].values())
            assert math.isclose(measured['graph_ms'], 1000 * (measured['end'] - measured['start']) / measured['passes'], rel_tol=1e-7)
            if phase.get('warmup'):
                continue
            assert phase['complete']
            assert summarize_samples(samples, *phase['stable_window']) == phase['dram']
            backgrounds = [d['backgrounds'][phase[k]] for k in ('background_before_index', 'background_after_index')]
            assert phase['adjusted_read_gb_s'] == phase['dram']['read_gb_s'] - max(b['read_gb_s'] for b in backgrounds)
            assert phase['adjusted_total_gb_s'] == phase['dram']['total_gb_s'] - max(b['total_gb_s'] for b in backgrounds)
        checks = d.get('layout_checks', [])
        if d['config']['check_layouts']:
            import re
            assert len(checks) == 8 and d['layout_checks_exact']
            assert sha256(root / 'iq2-repack-check') == d['layout_check_binary_sha256']
            for check in checks:
                log_path = root / check['log']
                assert sha256(log_path) == check['log_sha256']
                text = log_path.read_text()
                assert 'Repacking: 240 cases, 0 failures' in text
                actual = dict(re.findall(r'^PASS (type=.*? fused=\d).*? hash=([0-9a-f]{16})$', text, re.M))
                assert len(actual) == check['cases'] == 240 and actual == check['output_hashes']
            for padded in (False, True):
                for batch3 in (0, 1):
                    pair = [c['output_hashes'] for c in checks if c['padded'] == padded and c['batch3'] == batch3]
                    assert len(pair) == 2 and pair[0] == pair[1]
        ratios = []
        for index in range(0, len(d['phases']), 2):
            pair = {p['mode']: p['measurement']['graph_ms'] for p in d['phases'][index:index + 2]}
            ratios.append(pair['baseline'] / pair['candidate'])
        assert ratios == d['analysis']['pair_speedups'] and not d['analysis']['selected_for_production']
        audit['experiments'].append(dict(label=label, result_sha256=sha256(root / 'result.json'),
            weight_type=d['config']['qwen_weight_type'], tokens=d['config']['tokens'],
            split_granularity=d['config']['split_granularity'], exact_output_values=expected_values,
            analysis=d['analysis'], counter_intervals=len(samples), phases=8,
            median_adjusted_read_gb_s={mode: statistics.median(p['adjusted_read_gb_s'] for p in d['phases'] if p['mode'] == mode)
                                       for mode in ('baseline', 'candidate')},
            background_total_gb_s=[b['total_gb_s'] for b in d['backgrounds']],
            private_cpu_sha256=cpu['library_sha256'], native_layout_cases=sum(c['cases'] for c in checks),
            resident_weight_bytes=d['resident_weight_bytes'],
            median_read_counter_over_logical={mode: statistics.median(p['read_counter_over_logical'] for p in d['phases'] if p['mode'] == mode) for mode in ('baseline', 'candidate')}))
    previous = json.loads((base / 'results/glm53-full-replay-bandwidth-0906/validation-audit.json').read_text())
    for path, digest in previous['verified_original_engine_sha256'].items():
        assert sha256(path) == digest, path
        prior = audit['verified_original_engine_sha256'].setdefault(path, digest)
        assert prior == digest
    mapped = previous['qwen_mapped_library']
    actual = subprocess.run(['sudo', '-n', 'sha256sum', mapped['map_file']], check=True,
                            capture_output=True, text=True, timeout=15).stdout.split()[0]
    assert actual == mapped['sha256']
    audit['qwen_mapped_library'] = mapped
    plan = json.loads((base / 'results/qwen-even-split-model-trial-0906-staging/plan.json').read_text())
    full_runtime = json.loads((base / 'results/glm53-full-replay-bandwidth-0906/result.json').read_text())
    audit['protected'] = {}
    for pid, expected in plan['protected'].items():
        info = process_info(int(pid))
        assert identity_matches(info, expected)
        command = full_runtime['server_command'] if pid == '4005448' else plan['original_command']
        environment = full_runtime['runtime_env'] if pid == '4005448' else plan['original_environment']
        assert info['command'] == command and runtime_environment(process_environment(int(pid))) == environment
        assert info['affinity'] == ([15, 31, 47, 63] if pid == '4005448' else list(range(128)))
        audit['protected'][pid] = dict(start=info['start'], exe=info['exe'], affinity=info['affinity'],
                                        port=expected['port'], **read_service(expected['port']))
    audit['current_inference_pids'] = sorted(inference_snapshot())
    audit['recorded_owned_pids'] = sorted(owned)
    audit['owned_pids_absent'] = all(not Path(f'/proc/{pid}').exists() for pid in owned)
    assert audit['owned_pids_absent'], 'Inspect any reused PID before classifying its ownership'
    audit['total_exact_output_values'] = sum(e['exact_output_values'] for e in audit['experiments'])
    audit['total_phases'] = sum(e['phases'] for e in audit['experiments'])
    audit['total_native_layout_cases'] = sum(e['native_layout_cases'] for e in audit['experiments'])
    audit['note'] = ('Private gate/up, SwiGLU, down, and NUMA-reduction graphs. Native checks cover three changing '
                     'inputs/routes; timed validation compares exact output at phase boundaries, not every inner pass. '
                     'Byte expansion increases actual storage and traffic; success requires faster computation. '
                     'Short stable IMC windows are systemwide with estimated background subtraction. No complete '
                     'model was loaded or reconfigured, and these results do not establish model bandwidth or speed.')
    audit['finished'] = time.time()
    destination.write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(dict(audit=str(destination), experiments=len(audit['experiments']), phases=audit['total_phases'],
        exact_output_values=audit['total_exact_output_values'], engine_hashes=len(audit['verified_original_engine_sha256']),
        owned_pids_absent=audit['owned_pids_absent'], protected=audit['protected'])))


if __name__ == '__main__':
    main()
