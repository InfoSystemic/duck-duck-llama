#!/usr/bin/env python3
"""Validate and summarize the completed prose arm of a failed two-arm profile."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import time

BASE = Path(__file__).resolve().parent
ROOT = BASE / 'results/qwen-private-hc-norm-flat-on-profile-0908-profile'
OUT = BASE / 'results/qwen-hc-norm-flat-prose-analysis-0909'
SAMPLE = re.compile(r'\s*(\d+)/(\d+)\s+\[(\d+)\]\s+([\d.]+):\s+(\d+)\s+([0-9a-f]+)\s+(.*?)\s+\(([^()]*)\)$')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    source = ROOT / 'result.json'
    result = json.loads(source.read_text())
    assert result['finished'] and not result['completed'] and result['error'] == 'AssertionError()'
    assert all(sha(path) == digest for path, digest in result['source_sha256'].items())
    profile, = [row for row in result['profiles'] if row['kind'] == 'prose']
    failed, = [row for row in result['profiles'] if row['kind'] == 'code']
    assert 'report_sha256' not in failed
    assert not profile['abort'] and profile['perf_exit'] == profile['report_exit'] == 0
    first, last = profile['first_content_monotonic'], profile['last_content_monotonic']
    start, end = profile['collection_window_monotonic']
    lo, hi = profile['sample_window_monotonic']
    assert first <= start <= lo <= hi <= end <= last
    assert profile['timings']['cache_n'] == 0 and profile['timings']['draft_n'] > 0
    cpu = (Path(result['current']['runtime_env']['LD_LIBRARY_PATH']) / 'libggml-cpu.so.0').resolve()
    assert sha(cpu) == 'c79e19e38acfd6ada3f0637c136cf9cb10253127778cb214528b2f966fb6dfcf'
    directory = ROOT / 'prose-draft4'
    evidence = [source, cpu, Path(__file__).resolve()]
    for name, key in [('report.txt', 'report_sha256'), ('samples-by-thread.txt', 'samples_sha256'), ('callchains.txt', 'callgraph_sha256')]:
        path = directory / name
        assert sha(path) == profile[key]
        evidence.append(path)
    assert 'Total Lost Samples: 0' in (directory / 'report.txt').read_text()
    before, after = profile['threads_before'], profile['threads_after']
    pairs = defaultdict(list)
    for tid, row in before.items():
        if tid in after and row['start'] == after[tid]['start'] and len(row['affinity']) == 1:
            ticks = after[tid]['ticks'] - row['ticks']
            if ticks > 0:
                pairs[row['affinity'][0]].append((ticks, tid))
    assert len(pairs) == 60 and all(len(rows) == 2 for rows in pairs.values())
    cohorts = {}
    for rows in pairs.values():
        for name, (_, tid) in zip(('higher-activity', 'lower-activity'), sorted(rows, reverse=True)):
            cohorts[tid] = name
    counts, symbols = Counter(), Counter()
    by_cohort = defaultdict(Counter)
    samples = 0
    for line in (directory / 'samples-by-thread.txt').read_text().splitlines():
        if not line.strip():
            continue
        match = SAMPLE.fullmatch(line)
        assert match, line
        pid, tid, worker_cpu, stamp, period, address, symbol, dso = match.groups()
        assert int(pid) == result['current']['pid'] and lo <= float(stamp) <= hi
        period = int(period)
        category = 'other'
        if dso.endswith('/libgomp.so.1.0.0'):
            category = 'OpenMP-runtime'
        elif symbol == 'ggml_gemv_q8_0_x16_q8_0':
            category = 'Q8-x16'
        elif symbol == 'ggml_gemv_q6_K_x16_q8_K':
            category = 'Q6-x16'
        elif symbol in ('ggml_gemv_q8_0_8x8_q8_0', 'ggml_gemm_q8_0_8x8_q8_0'):
            category = 'Q8-8x8'
        elif 'tinyBLAS' in symbol:
            category = 'tinyBLAS'
        counts[category] += period
        by_cohort[cohorts.get(tid, 'other-threads')][category] += period
        symbols[(Path(dso).name, symbol)] += period
        samples += 1
    total = sum(counts.values())
    def percentages(counter):
        denominator = sum(counter.values())
        return {key: 100 * value / denominator for key, value in counter.most_common()}
    output = dict(time=time.time(), validated_prose_arm=True, full_profile_completed=False,
                  samples=samples, sampled_periods=total, category_percent=percentages(counts),
                  cohorts=[dict(cohort=key, fraction_of_all_periods=sum(value.values()) / total,
                                within_cohort_percent=percentages(value)) for key, value in by_cohort.items()],
                  top_symbols=[dict(dso=dso, symbol=symbol, percent=100 * value / total)
                               for (dso, symbol), value in symbols.most_common(20)],
                  input_sha256={str(path): sha(path) for path in evidence},
                  scope='Only the completed prose capture is qualified. Code profiling failed. Period-weighted CPU samples are not removable wall-time fractions. Thread cohorts use observed activity, not assumed target/draft roles. OpenMP entries are not classified as wait loops without disassembly.')
    OUT.mkdir(exist_ok=False)
    (OUT / 'result.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({key: value for key, value in output.items() if key not in ('input_sha256', 'top_symbols')}, indent=2))


if __name__ == '__main__':
    main()
