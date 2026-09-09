#!/usr/bin/env python3
"""Classify the saved Flash decode samples using the verified host mapping."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

BASE = Path(__file__).resolve().parent
ROOT = BASE / 'results/glm-flash-q8-experts-raw-profile-0908'
SHA = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    profile = json.loads((ROOT / 'result.json').read_text())
    assert profile['completed']
    entry = profile['profiles'][0]
    host = json.loads((ROOT / 'analysis/host-library.json').read_text())
    assert host['pid'] == profile['current']['pid']
    assert host['start'] == profile['current']['info']['start']
    samples = ROOT / 'prose-draft0/samples-by-thread.txt'
    assert SHA(samples) == entry['samples_sha256']
    mappings = []
    for row in host['libgomp_maps']:
        fields = row.split()
        lo, hi = (int(s, 16) for s in fields[0].split('-'))
        mappings.append((lo, hi, int(fields[2], 16)))
    # Verified PAUSE/poll loops, including the loads and loop-control instructions.
    spin_ranges = ((0x256b7, 0x256d7), (0x2587b, 0x258b3))
    pattern = re.compile(r'^(\d+)/(\d+)\s+\[(\d+)\]\s+([\d.]+):\s+(\d+)\s+([0-9a-f]+)\s+(.+)\s+\((.+)\)$')
    by_thread, by_cpu = defaultdict(Counter), defaultdict(Counter)
    gomp_offsets = Counter()
    sample_count = 0
    for line in samples.read_text().splitlines():
        if not line.strip():
            continue
        match = pattern.match(line.strip())
        assert match, line
        pid, tid, cpu, timestamp, period, ip, symbol, dso = match.groups()
        assert int(pid) == host['pid']
        tid, cpu, period, ip = int(tid), int(cpu), int(period), int(ip, 16)
        category = 'other'
        if dso == host['libgomp_path']:
            candidates = [ip - lo + offset for lo, hi, offset in mappings if lo <= ip < hi]
            assert len(candidates) == 1
            offset = candidates[0]
            gomp_offsets[hex(offset)] += period
            category = 'gomp_spin' if any(lo <= offset < hi for lo, hi in spin_ranges) else 'gomp_other'
        elif symbol == 'ggml_gemv_q8_0_x16_q8_0':
            category = 'q8_x16'
        elif symbol == 'ggml_gemv_q8_0_8x8_q8_0':
            category = 'q8_x8'
        elif symbol == 'ggml_vec_dot_f32':
            category = 'f32_dot'
        by_thread[tid][category] += period
        by_cpu[cpu][category] += period
        sample_count += 1

    def compact(counter):
        return dict(total_sampled_cycles=counter.total(),
                    **{k + '_pct': v / counter.total() * 100 for k, v in counter.items()})

    before, after = entry['threads_before'], entry['threads_after']
    duration = entry['collection_window_monotonic'][1] - entry['collection_window_monotonic'][0]
    ticks_per_second = int(host['clock_ticks'])
    rows = []
    assert before.keys() == after.keys()
    for tid, initial in before.items():
        final = after[tid]
        assert initial['start'] == final['start'] and initial['affinity'] == final['affinity']
        delta_s = (final['ticks'] - initial['ticks']) / ticks_per_second
        rows.append(dict(tid=int(tid), cpu_percent=100 * delta_s / duration,
                         affinity=initial['affinity'], name=initial['name'],
                         **(compact(by_thread[int(tid)]) if by_thread[int(tid)] else {})))
    rows.sort(key=lambda r: -r['cpu_percent'])
    total = sum(by_thread.values(), Counter())
    result = dict(
        sources={str(p): SHA(p) for p in [Path(__file__), ROOT / 'result.json',
                 ROOT / 'analysis/host-library.json', ROOT / 'analysis/libgomp-barrier-disassembly.txt', samples]},
        pid=host['pid'], samples=sample_count, categories=compact(total), threads=rows,
        cpus={str(cpu): compact(c) for cpu, c in sorted(by_cpu.items())},
        gomp_offsets=dict(gomp_offsets.most_common()),
        clock_tick_duration_s=duration,
        total_core_equivalents=sum(r['cpu_percent'] for r in rows) / 100,
        sample_window_monotonic=entry['sample_window_monotonic'],
        notes=[
            'Sample categories are weighted by perf event periods, not by sample counts.',
            'Tick deltas cover the wider perf launch/exit interval; samples cover only user cycles.',
            'Spin-cycle fractions are not removable elapsed-time fractions.',
            'Incomplete x16 stack unwinding prevents dense versus routed attribution.',
            'CPU numbers are reported directly; no socket topology is inferred here.',
        ],
    )
    (ROOT / 'analysis/thread-analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    compute = [r for r in rows if r['cpu_percent'] > 50]
    print(json.dumps(dict(samples=sample_count, categories=result['categories'],
        total_core_equivalents=result['total_core_equivalents'], compute_threads=len(compute),
        compute_cpu_percent_range=[min(r['cpu_percent'] for r in compute), max(r['cpu_percent'] for r in compute)],
        compute_spin_percent_range=[min(r.get('gomp_spin_pct', 0) for r in compute), max(r.get('gomp_spin_pct', 0) for r in compute)])))


if __name__ == '__main__':
    main()
