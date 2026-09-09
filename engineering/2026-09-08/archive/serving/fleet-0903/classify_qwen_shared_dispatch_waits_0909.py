#!/usr/bin/env python3
"""Classify the two inspected OpenMP spin loops in the completed profiles."""
import json
from pathlib import Path
import time

from inspect_qwen_shared_dispatch_waits_0909b import perf_sha, sha

BASE = Path(__file__).resolve().parent
SOURCE = BASE / 'results/qwen-shared-dispatch-wait-addresses-0909b/result.json'
OUT = BASE / 'results/qwen-shared-dispatch-wait-classification-0909.json'


def main():
    assert not OUT.exists()
    result = json.loads(SOURCE.read_text())
    assert result['passed'] and len(result['rows']) == 4
    for path, digest in result['input_sha256'].items():
        assert (perf_sha(path) if Path(path).name == 'perf.data' else sha(path)) == digest, path
    assert len(result['libraries']) == 1
    library, record = next(iter(result['libraries'].items()))
    assert record['sha256'] == '135f3c8f006d2fe5e68e51281c7974cb991a03de3bfb3593d68d174dfcf854d1'
    assert record['build_id'] == 'fa0c1b446610c4b7aca51e1183050acf5aa79503'
    raw = Path(library).read_bytes()
    loops = [
        ('spin-loop-256c0', 0x256c0, 0x256d1, 'f3904883c0014839f074158b0f39ca74ef'),
        ('spin-loop-258a0', 0x258a0, 0x258b3, 'f3904883c0014839d07435418b0c2439cb74ed'),
    ]
    for label, start, end, expected in loops:
        segment = [row for row in record['executable_segments'] if row[2] <= start < end <= row[2] + row[1] - row[0]]
        assert len(segment) == 1
        offset, _, vaddr = segment[0]
        assert raw[start - vaddr + offset:end - vaddr + offset] == bytes.fromhex(expected), label
    rows = []
    for row in result['rows']:
        total = row['all_sampled_periods']
        addresses = {int(address, 16): periods for address, periods in row['all_openmp_addresses'].items()}
        parts = {label: sum(periods for address, periods in addresses.items() if start <= address < end)
                 for label, start, end, expected in loops}
        spin = sum(parts.values())
        assert spin <= row['openmp_periods'] <= total
        rows.append(dict(arm=row['arm'], workload=row['workload'],
            verified_spin_percent_of_all_cycles=100 * spin / total,
            verified_spin_fraction_of_openmp_cycles=spin / row['openmp_periods'],
            per_loop_percent_of_all_cycles={label: 100 * periods / total for label, periods in parts.items()},
            other_openmp_percent_of_all_cycles=100 * (row['openmp_periods'] - spin) / total))
    output = dict(time=time.time(), passed=True, rows=rows,
        inspected_loops=[dict(name=label, start=hex(start), end_exclusive=hex(end), bytes=expected) for label, start, end, expected in loops],
        input_sha256={str(SOURCE): sha(SOURCE), str(Path(__file__).resolve()): sha(__file__),
                      str(BASE / 'inspect_qwen_shared_dispatch_waits_0909b.py'): sha(BASE / 'inspect_qwen_shared_dispatch_waits_0909b.py')},
        scope='Both inspected loops execute PAUSE, increment a spin counter, and reload/compare a shared synchronization word before looping or leaving the spin path. This classifies sampled instructions as spinning; it does not distinguish inherent synchronization cost from waiting for useful work elsewhere, or establish removable wall time. More graph-level attribution is required before choosing the next synchronization change.')
    OUT.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({key: value for key, value in output.items() if key not in ['input_sha256', 'inspected_loops']}, indent=2))


if __name__ == '__main__':
    main()
