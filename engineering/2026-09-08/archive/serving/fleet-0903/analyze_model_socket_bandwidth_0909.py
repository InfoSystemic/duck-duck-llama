#!/usr/bin/env python3
"""Decompose completed decode counters by socket without starting another measurement."""
import json
import math
from pathlib import Path
import time

from qwen_split_trial import sha256

BASE = Path(__file__).resolve().parent


def main():
    out = BASE / 'results/model-socket-bandwidth-0909.json'
    assert not out.exists()
    labels = [
        'glm-flash-q8-r8-ordered-k-raw-0908',
        'glm-flash-q8-r8-ordered-k-raw-repeat-0908',
        'glm53-full-current-bandwidth-0906',
        'qwen-private-raw-columns-250-0909-decode',
        'qwen-private-decode-scheduling-250-0909-1-candidate-decode',
    ]
    inputs = {str(Path(__file__).resolve()): sha256(__file__)}
    rows = []
    for label in labels:
        path = BASE / 'results' / label / 'result.json'
        measured = json.loads(path.read_text())
        assert measured['finished'] and not measured.get('error')
        assert len(measured['measurements']) == 2
        inputs[str(path)] = sha256(path)
        for measurement in measured['measurements']:
            meta = measurement['counter_metadata']
            assert meta['valid'] and meta['exit_code'] == 0 and meta['required_counters_per_interval'] == 48
            assert not measurement['abort'] and not measurement['inference_churn']
            before, after, decode = [measurement[name] for name in ('baseline_before', 'baseline_after', 'decode')]
            assert all(item['valid'] for item in (before, after, decode))
            assert max(before['total_gb_s'], after['total_gb_s']) <= .05 * 380
            assert abs(before['total_gb_s'] - after['total_gb_s']) <= .025 * 380
            baseline = before if before['total_gb_s'] >= after['total_gb_s'] else after
            selected = 'before' if baseline is before else 'after'
            by_socket = {}
            for phase, values in (('decode', decode), ('baseline', baseline)):
                by_socket[phase] = {row['socket']: row for row in values['sockets']}
                assert set(by_socket[phase]) == {0, 1, 2, 3}
                assert math.isclose(sum(row['total_gb_s'] for row in values['sockets']), values['total_gb_s'], abs_tol=1e-8)
            sockets = [dict(socket=socket, adjusted_gb_s=by_socket['decode'][socket]['total_gb_s'] -
                             by_socket['baseline'][socket]['total_gb_s']) for socket in range(4)]
            total = sum(row['adjusted_gb_s'] for row in sockets)
            assert math.isclose(total, measurement['background_subtracted_gb_s'], abs_tol=1e-8)
            for row in sockets:
                row['share'] = row['adjusted_gb_s'] / total
                assert row['adjusted_gb_s'] > 0
            rows.append(dict(label=label, workload=measurement['kind'], generated_tok_s=measurement['timings']['predicted_per_second'],
                             adjusted_gb_s=total, baseline_selected=selected, sockets=sockets,
                             busiest_to_quietest_ratio=max(row['adjusted_gb_s'] for row in sockets) / min(row['adjusted_gb_s'] for row in sockets),
                             target_gb_s=250, target_met=total >= 250))
    result = dict(time=time.time(), passed=True, input_sha256=inputs, rows=rows,
                  no_new_measurements=True,
                  scope='Historical stable decode windows, all 48 IMC counters. Subtract the per-socket values from the same higher-total adjacent idle window used by the original measurement, so the four adjusted socket rates sum to the retained whole-server rate.',
                  limits=['These counters measure physical DRAM traffic, not remote NUMA accesses or compute utilization.',
                          'Balanced socket contributions do not establish a memory-bandwidth ceiling or guarantee that 250 GB/s is attainable.',
                          'The Qwen HC-only sample is one completed pair from an interrupted comparison; no repeated gain is implied.'])
    with out.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps(dict(passed=True, rows=[dict(label=row['label'], workload=row['workload'],
                      adjusted_gb_s=row['adjusted_gb_s'], socket_gb_s=[round(socket['adjusted_gb_s'], 3) for socket in row['sockets']],
                      busiest_to_quietest_ratio=row['busiest_to_quietest_ratio']) for row in rows]), indent=2))


if __name__ == '__main__':
    main()
