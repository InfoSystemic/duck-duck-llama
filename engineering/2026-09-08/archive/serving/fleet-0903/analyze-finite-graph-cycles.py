#!/usr/bin/env python3
"""Validate and summarize saved private graph cycle samples, without timing claims."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re


def analyze(directory):
    result_file = directory / 'result.json'
    result = json.loads(result_file.read_text())
    assert result.get('finished') and not result.get('error')
    summaries = []
    pattern = re.compile(r'\s*(\d+)/(\d+)\s+\[(\d+)\]\s+(\d+\.\d+):\s+(\d+)\s+([0-9a-fA-F]+)\s+(.+)\s+\(([^()]*)\)\s*$')
    for run in result['runs']:
        assert run.get('complete') and not run.get('contention')
        assert run['measurement']['checksums_exact'] and run['measurement']['profiled']
        info = run['profile']
        trace = Path(info['trace_file'])
        report = Path(info['report_file']).read_text()
        assert re.search(r'Total Lost Samples:\s+0\b', report), 'Missing zero-loss report'
        start, end = info['selected_window_monotonic']
        categories, symbols = Counter(), Counter()
        by_cpu, by_tid = defaultdict(Counter), defaultdict(Counter)
        tid_cpus = defaultdict(set)
        samples = 0
        timestamps = []
        for line in trace.read_text().splitlines():
            if not line.strip():
                continue
            match = pattern.fullmatch(line)
            assert match, line
            pid, tid, cpu, stamp, period, ip, symbol, dso = match.groups()
            pid, tid, cpu, period, stamp = int(pid), int(tid), int(cpu), int(period), float(stamp)
            assert pid == info['target_pid'] and start <= stamp <= end and period > 0
            assert 0 <= cpu < 64, 'Sample outside the declared physical CPU sets'
            category = ('q5_gemv' if symbol == 'ggml_gemv_q5_K_x16_q8_K' else
                        'openmp_library' if 'libgomp' in dso else
                        'fused_reduce' if 'fused_reduce' in symbol else 'other')
            categories[category] += period
            symbols[(dso, symbol)] += period
            by_cpu[cpu][category] += period
            by_tid[tid][category] += period
            tid_cpus[tid].add(cpu)
            timestamps.append(stamp)
            samples += 1
        assert samples == info['selected_samples']
        assert min(timestamps) <= start + 1 and max(timestamps) >= end - 1
        total = sum(categories.values())

        def percentages(counts):
            subtotal = sum(counts.values())
            return dict(sampled_cycles=subtotal, percent_of_total=100 * subtotal / total,
                        within_group_percent={k: 100 * v / subtotal for k, v in counts.items()})

        roles = defaultdict(Counter)
        for cpu, counts in by_cpu.items():
            role = 'reserved_controller_cpus' if cpu in (15, 31, 47, 63) else (
                'socket_leader_cpus' if cpu in (0, 16, 32, 48) else 'other_worker_cpus')
            roles[role].update(counts)
        cohorts = defaultdict(Counter)
        for tid, counts in by_tid.items():
            cohorts['threads_with_q5_samples' if counts['q5_gemv'] else 'threads_without_q5_samples'].update(counts)
        summaries.append(dict(mode=run['mode'], tokens=result['config']['tokens'], samples=samples,
                              target_pid=info['target_pid'], sampled_cycles=total,
                              category_percent={k: 100 * v / total for k, v in categories.items()},
                              cpu_roles={k: percentages(v) for k, v in roles.items()},
                              thread_cohorts={k: percentages(v) for k, v in cohorts.items()},
                              by_cpu={str(k): percentages(v) for k, v in sorted(by_cpu.items())},
                              by_thread={str(k): dict(cpus=sorted(tid_cpus[k]), **percentages(v)) for k, v in sorted(by_tid.items())},
                              top_symbols=[dict(dso=dso, symbol=symbol, percent_of_total=100 * cycles / total)
                                           for (dso, symbol), cycles in symbols.most_common(20)],
                              trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest()))
    output = dict(scope='Period-weighted user-cycle samples from private finite graphs. These are neither removable wall time nor model throughput.',
                  source_result_sha256=hashlib.sha256(result_file.read_bytes()).hexdigest(), runs=summaries)
    (directory / 'cycle-analysis.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(dict(directory=str(directory), runs=[{k: r[k] for k in ('mode', 'tokens', 'samples', 'category_percent', 'cpu_roles', 'thread_cohorts')} for r in summaries]), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    analyze(parser.parse_args().directory.resolve())
