#!/usr/bin/env python3
"""Summarize validated off/on cycle profiles after the matched model repeats."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import time

from compare_qwen_private_shared_dispatch_0909 import compare, read_run

BASE = Path(__file__).resolve().parent
OUT = BASE / 'results/qwen-shared-dispatch-profile-analysis-0909'
SAMPLE = re.compile(r'\s*(\d+)/(\d+)\s+\[(\d+)\]\s+([\d.]+):\s+(\d+)\s+([0-9a-f]+)\s+(.*?)\s+\(([^()]*)\)$')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def percentages(values):
    total = sum(values.values())
    assert total > 0
    return {key: 100 * value / total for key, value in values.most_common()}


def category(symbol, dso):
    if dso.endswith('/libgomp.so.1.0.0'):
        return 'OpenMP-runtime'
    if symbol == 'ggml_gemv_q8_0_x16_q8_0':
        return 'Q8-x16'
    if symbol == 'ggml_gemv_q6_K_x16_q8_K':
        return 'Q6-x16'
    if symbol in ('ggml_gemv_q8_0_8x8_q8_0', 'ggml_gemm_q8_0_8x8_q8_0'):
        return 'Q8-8x8'
    if 'tinyBLAS' in symbol:
        return 'tinyBLAS'
    if 'ggml_backend_meta_fused_reduce_op' in symbol:
        return 'cross-socket-reduction'
    if 'qwen_hc_' in symbol:
        return 'HC-fused-helpers'
    return 'other'


def analyze(run):
    trial_path = BASE / 'results' / run['label'] / 'result.json'
    trial = json.loads(trial_path.read_text())
    source = Path(trial['profile'])
    assert sha(source) == trial['profile_sha256']
    result = json.loads(source.read_text())
    assert result['completed'] and result['finished'] and not result.get('error')
    assert len(result['checks']) == 2 and all(r['passed'] and not r['abort'] for r in result['checks'])
    assert result['baseline_sha256'] == trial['measurement_sha256']
    assert result['current']['pid'] == trial['model_pid']
    assert result['current']['command'] == run['plan']['command']
    assert result['current']['runtime_env'] == run['plan']['runtime_env']
    assert result['config']['call_graph'] is None and result['config']['sample_frequency'] == 199
    assert result['config']['record_event'] == 'cycles:u'
    assert all(sha(path) == digest for path, digest in result['source_sha256'].items())
    assert len(result['profiles']) == 2 and {r['kind'] for r in result['profiles']} == {'prose', 'code'}
    evidence = {str(source): sha(source), **run['evidence']}
    summaries = []
    for profile in result['profiles']:
        assert not profile['abort'] and profile['perf_exit'] == profile['report_exit'] == 0
        assert not profile['inference_churn'] and all(r['cpu_percent'] <= 20 for r in profile['other_inference'])
        assert profile['timings']['cache_n'] == 0 and profile['timings']['draft_n'] > 0
        first, last = profile['first_content_monotonic'], profile['last_content_monotonic']
        start, end = profile['collection_window_monotonic']
        lo, hi = profile['sample_window_monotonic']
        assert first <= start <= lo <= hi <= end <= last
        directory = source.parent / (profile['kind'] + '-draft4')
        for name, key in [('report.txt', 'report_sha256'), ('samples-by-thread.txt', 'samples_sha256')]:
            path = directory / name
            assert sha(path) == profile[key]
            evidence[str(path)] = sha(path)
        assert 'Total Lost Samples: 0' in (directory / 'report.txt').read_text()
        assert len(profile['collection_window_monotonic']) == 2
        active_per_cpu = Counter()
        stable_single_cpu = set()
        before, after = profile['threads_before'], profile['threads_after']
        for tid, row in before.items():
            later = after.get(tid)
            if later and row['start'] == later['start'] and row['affinity'] == later['affinity'] and len(row['affinity']) == 1:
                stable_single_cpu.add(tid)
                if later['ticks'] > row['ticks']:
                    active_per_cpu[str(row['affinity'][0])] += 1
        counts, symbols, by_thread, by_cpu = Counter(), Counter(), defaultdict(Counter), Counter()
        samples = 0
        for line in (directory / 'samples-by-thread.txt').read_text().splitlines():
            if not line.strip():
                continue
            match = SAMPLE.fullmatch(line)
            assert match, line
            pid, tid, worker_cpu, stamp, period, address, symbol, dso = match.groups()
            assert int(pid) == trial['model_pid'] and lo <= float(stamp) <= hi
            period = int(period)
            assert period > 0
            label = category(symbol, dso)
            counts[label] += period
            symbols[(Path(dso).name, symbol)] += period
            by_thread[tid][label] += period
            by_cpu[worker_cpu] += period
            samples += 1
        total = sum(counts.values())
        pinned = Counter()
        other = Counter()
        for tid, values in by_thread.items():
            (pinned if tid in stable_single_cpu else other).update(values)
        summaries.append(dict(workload=profile['kind'], samples=samples, sampled_periods=total,
            category_percent=percentages(counts), sampled_threads=len(by_thread),
            active_single_cpu_threads=sum(active_per_cpu.values()), active_single_cpu_threads_per_cpu=dict(active_per_cpu),
            single_cpu_thread_period_fraction=sum(pinned.values()) / total,
            single_cpu_thread_category_percent=percentages(pinned) if pinned else {},
            other_thread_category_percent=percentages(other) if other else {},
            sampled_cpu_percent=percentages(by_cpu),
            top_symbols=[dict(dso=dso, symbol=symbol, percent=100 * value / total) for (dso, symbol), value in symbols.most_common(40)],
            profile_window_seconds=end-start, decode_window_seconds=last-first,
            observed_other_host_cores=sum(r['cpu_percent'] for r in profile['other_host_cpu']) / 100))
    return dict(label=run['label'], shared_dispatch=run['plan']['shared_dispatch'], workloads=summaries), evidence


def main():
    assert not OUT.exists()
    off = read_run('qwen-private-shared-dispatch-off-0909c')
    on = read_run('qwen-private-shared-dispatch-on-0909c')
    comparisons = compare(off, [on])
    assert all(r['output_matches_control'] for c in comparisons for r in c['workloads'])
    rows = []
    evidence = {str(Path(__file__).resolve()): sha(__file__),
                str(BASE / 'compare_qwen_private_shared_dispatch_0909.py'): sha(BASE / 'compare_qwen_private_shared_dispatch_0909.py')}
    for run in [off, on]:
        row, inputs = analyze(run)
        rows.append(row)
        evidence.update(inputs)
    result = dict(time=time.time(), passed=True, configurations=rows, input_sha256=evidence,
        scope='199 Hz user-cycle samples enclosed by fresh single-conversation decode. Profiled rates are excluded from performance claims. OpenMP samples are not classified as wait loops without disassembly, and concurrent CPU-cycle fractions are not removable wall-time fractions. Thread counts use observed affinity and activity, without assuming target/draft roles.')
    OUT.mkdir()
    (OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'input_sha256'}, indent=2))


if __name__ == '__main__':
    main()
