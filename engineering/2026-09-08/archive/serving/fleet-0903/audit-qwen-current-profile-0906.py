#!/usr/bin/env python3
"""Reconcile saved Qwen cycle samples and verify current protected services."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import subprocess
import time

from model_measurement_guard import read_service
from qwen_split_trial import (identity_matches, inference_snapshot, process_environment,
                              process_info, runtime_environment, sha256)


def category(symbol, dso):
    if dso.endswith('libgomp.so.1.0.0'):
        return 'OpenMP library'
    if 'ggml_backend_meta_fused_reduce_op' in symbol:
        return 'NUMA reduction'
    if 'iq_r16' in symbol:
        return 'IQ r16 matrix'
    if 'gemv_q8' in symbol or 'gemm_q8' in symbol:
        return 'Q8 matrix'
    if 'gemv_q5' in symbol:
        return 'Q5 matrix'
    if 'gemv_q4_b32' in symbol:
        return '4-bit LUT matrix'
    if 'tinyBLAS' in symbol:
        return 'tinyBLAS'
    return 'Other'


def main():
    base = Path(__file__).resolve().parent
    root = base / 'results/qwen-current-mtp2-profile-0906'
    destination = root / 'validation-audit.json'
    assert not destination.exists()
    result_path = root / 'result.json'
    d = json.loads(result_path.read_text())
    assert d['completed'] and d['finished'] and not d.get('error')
    assert d['idle_gate']['quiet_seconds'] >= 60
    assert len(d['checks']) == 2 and all(c['passed'] and not c['abort'] for c in d['checks'])
    assert [p['kind'] for p in d['profiles']] == ['prose', 'code']
    assert all(sha256(path) == value and sha256(root / Path(path).name) == value
               for path, value in d['source_sha256'].items())
    assert sha256(base / d['config']['after']) == d['baseline_sha256']
    audit = dict(started=time.time(), source_sha256=sha256(Path(__file__)),
                 result_sha256=sha256(result_path), profiles=[], real_model_signals=0)
    pattern = re.compile(r'^(\d+)/(\d+)\s+\[(\d+)\]\s+([0-9.]+):\s+(\d+)\s+([0-9a-f]+)\s+(.*)\s+\((/.*|\[.*\])\)$')
    owned_perf = []
    cpu_socket = {cpu: int(Path(f'/sys/devices/system/cpu/cpu{cpu}/topology/physical_package_id').read_text())
                  for cpu in range(128)}
    for profile in d['profiles']:
        directory = root / (profile['kind'] + '-draft2')
        assert not profile['abort'] and profile['perf_exit'] == profile['report_exit'] == 0
        assert profile['timings']['draft_n'] > 0 and not profile['inference_churn']
        assert not any(p['cpu_percent'] > 20 for p in profile['other_inference'])
        assert all(not key.startswith('speculative.') for key in profile['payload'])
        chunks = json.loads((directory / 'chunks.json').read_text())
        assert [c['timings'] for c in chunks if c.get('timings')][-1] == profile['timings']
        assert sha256(directory / 'report.txt') == profile['report_sha256']
        assert sha256(directory / 'samples-by-thread.txt') == profile['samples_sha256']
        report = (directory / 'report.txt').read_text()
        assert '# Total Lost Samples: 0' in report
        expected_total = int(re.search(r'# Event count \(approx\.\): (\d+)', report)[1])
        total = count = 0
        by_category, by_cpu, by_symbol = Counter(), Counter(), Counter()
        by_thread = defaultdict(Counter)
        times = []
        for line in (directory / 'samples-by-thread.txt').read_text().splitlines():
            if not line.strip():
                continue
            match = pattern.fullmatch(line)
            assert match, line[:200]
            pid, tid, cpu, timestamp, period, ip, symbol, dso = match.groups()
            cpu, period, timestamp = int(cpu), int(period), float(timestamp)
            assert int(pid) == d['config']['pid'] and period > 0 and cpu in cpu_socket
            before, after = profile['threads_before'][tid], profile['threads_after'][tid]
            assert before['start'] == after['start']
            assert cpu in set(before['affinity']) | set(after['affinity'])
            total += period
            count += 1
            times.append(timestamp)
            group = category(symbol, dso)
            by_category[group] += period
            by_cpu[cpu] += period
            by_thread[tid][group] += period
            by_symbol[(Path(dso).name, symbol)] += period
        assert total == expected_total
        assert [min(times), max(times)] == profile['sample_window_monotonic']
        first, last = profile['first_content_monotonic'], profile['last_content_monotonic']
        start, end = profile['collection_window_monotonic']
        assert first <= start <= min(times) <= max(times) <= end <= last
        pinned_workers = sorted(int(tid) for tid in by_thread
                                if len(profile['threads_before'][tid]['affinity']) == 1
                                and profile['threads_before'][tid]['affinity'][0] % 16 in range(1, 15))
        assert len(pinned_workers) == 112
        early, late = pinned_workers[:56], pinned_workers[56:]
        assert max(early) - min(early) == max(late) - min(late) == 55
        assert min(late) - max(early) > 1000
        groups = {}
        for name, tids in [('earlier pinned workers', early), ('later pinned workers', late)]:
            values = Counter()
            for tid in tids:
                values.update(by_thread[str(tid)])
            group_total = sum(values.values())
            groups[name] = dict(tids=tids, percent_of_all_cycles=100 * group_total / total,
                category_percent_of_group={k: 100 * v / group_total for k, v in values.items()},
                openmp_percent_of_all_cycles=100 * values['OpenMP library'] / total)
        info = dict(kind=profile['kind'], samples=count, total_period=total, sampled_threads=len(by_thread),
            category_percent={k: 100 * v / total for k, v in by_category.items()},
            socket_percent={str(socket): 100 * sum(value for cpu, value in by_cpu.items() if cpu_socket[cpu] == socket) / total
                            for socket in range(4)},
            thread_groups=groups, profiled_tok_s=profile['timings']['predicted_per_second'],
            draft_acceptance=[profile['timings']['draft_n_accepted'], profile['timings']['draft_n']],
            sample_window=profile['sample_window_monotonic'], other_host_cpu=profile['other_host_cpu'],
            top_symbols=[dict(dso=key[0], symbol=key[1], percent=100 * value / total)
                         for key, value in by_symbol.most_common(35)],
            perf_data_sha256=subprocess.run(['sudo', '-n', 'sha256sum', str(directory / 'perf.data')],
                check=True, capture_output=True, text=True, timeout=15).stdout.split()[0],
            chunks_sha256=sha256(directory / 'chunks.json'))
        audit['profiles'].append(info)
        owned_perf.append(profile['owned_perf_pid'])
    previous = json.loads((base / 'results/qwen-iq-bytes-0906-audit.json').read_text())
    mapped = previous['qwen_mapped_library']
    actual = subprocess.run(['sudo', '-n', 'sha256sum', mapped['map_file']], check=True,
                            capture_output=True, text=True, timeout=15).stdout.split()[0]
    assert actual == mapped['sha256']
    audit['qwen_mapped_library'] = mapped
    audit['verified_runtime_libraries'] = {}
    for path, value in previous['verified_original_engine_sha256'].items():
        if '/validated-iq-batch3-bin/' in path or '/build-dev2/bin/' in path:
            assert sha256(path) == value
            audit['verified_runtime_libraries'][path] = value
    plan = json.loads((base / 'results/qwen-even-split-model-trial-0906-staging/plan.json').read_text())
    audit['protected'] = {}
    for pid, expected in plan['protected'].items():
        info = process_info(int(pid))
        before = d['protected_before'][pid]
        after = d['protected_after'][pid]
        assert identity_matches(info, expected)
        assert info['command'] == before['info']['command'] == after['info']['command']
        assert info['affinity'] == before['info']['affinity'] == after['info']['affinity']
        assert runtime_environment(process_environment(int(pid))) == before['environment'] == after['environment']
        state = read_service(expected['port'])
        assert not state['processing'] and not state['queued']
        audit['protected'][pid] = dict(start=info['start'], port=expected['port'], **state)
    audit['current_inference_pids'] = sorted(inference_snapshot())
    audit['owned_perf_pids'] = owned_perf
    assert all(not Path(f'/proc/{pid}').exists() for pid in owned_perf)
    audit['owned_perf_pids_absent'] = True
    audit['note'] = ('All-cycle-period categories, rather than only displayed top symbols. OpenMP library cycles '
        'include active waits and runtime work; they do not measure removable wall time. The thread groups are '
        'identified by observed TID ranges and pinned affinities, without assuming every pool role. No IMC '
        'bandwidth was collected with these profiles. Sampling timings are diagnostic, not an engine speedup.')
    audit['finished'] = time.time()
    destination.write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(dict(audit=str(destination), profiles=[{k: v for k, v in p.items()
        if k in ('kind', 'samples', 'sampled_threads', 'category_percent', 'socket_percent', 'thread_groups')}
        for p in audit['profiles']], protected=audit['protected'])))


if __name__ == '__main__':
    main()
