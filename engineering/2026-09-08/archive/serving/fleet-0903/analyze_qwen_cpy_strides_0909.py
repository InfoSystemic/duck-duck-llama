#!/usr/bin/env python3
"""Audit the completed Qwen copy-layout diagnostic without treating it as a benchmark."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import time

from qwen_split_trial import sha256

BASE = Path(__file__).resolve().parent


def main(label):
    out = BASE / 'results' / label
    plan_path, result_path = out / 'plan.json', out / 'result.json'
    plan, result = [json.loads(path.read_text()) for path in (plan_path, result_path)]
    assert result['passed'] and result['finished'] and not result.get('error')
    assert result['diagnostic_only'] and not result['target_reached']
    assert result['plan_sha256'] == sha256(plan_path)
    assert all(sha256(path) == digest for path, digest in plan['source_sha256'].items())
    assert result['owned_model_exit'] == 0 and result['dispatch_cleanup_valid']
    assert result['peer_preserved'] and result['peer_environment_preserved']
    assert len(result['checks']) == 2 and all(row['passed'] and not row['abort'] for row in result['checks'])
    assert len(result['traces']) == 2 and {row['kind'] for row in result['traces']} == {'prose', 'code'}
    inputs = {str(path): sha256(path) for path in (Path(__file__).resolve(), plan_path, result_path)}
    build_path = BASE / 'results/qwen-get-rows-columns-build-0909/result.json'
    build = json.loads(build_path.read_text())
    assert build['passed'] and build['library_sha256'] == plan['cpu_sha256']
    assert all(sha256(path) == digest for path, digest in build['private_source_sha256'].items())
    inputs.update({str(build_path): sha256(build_path), **build['private_source_sha256']})
    rows = []
    for trace in result['traces']:
        assert trace['output_and_draft_counts_match'] and not trace['abort']
        layouts_path, operations_path = Path(trace['copy_layouts']), Path(trace['operations'])
        assert sha256(layouts_path) == trace['copy_layouts_sha256']
        assert sha256(operations_path) == trace['operations_sha256']
        inputs.update({str(path): sha256(path) for path in (layouts_path, operations_path)})
        layouts = json.loads(layouts_path.read_text())
        assert len(layouts) == trace['copy_cases'] == 128
        grouped = Counter()
        workers = Counter()
        for row in layouts:
            assert Path(row['forward_library']).resolve() == Path(plan['cpu']).resolve()
            assert row['ith'] == 0 and row['nth'] == 15
            if row['src_contiguous'] and row['dst_contiguous']:
                branch, active = 'contiguous block split', 15
            elif row['same_shape'] and row['src_nb'][0] == row['dst_nb'][0] == 4:
                nr = row['src_ne'][1]
                chunk = (nr + 14) // 15
                branch, active = 'same-shape strided row split', (nr + chunk - 1) // chunk
            else:
                branch, active = 'other copy path', None
            workers[(branch, active)] += 1
            grouped[(tuple(row['src_ne']), tuple(row['dst_ne']), tuple(row['src_nb']), tuple(row['dst_nb']), branch, active)] += 1
        groups = [dict(src_ne=key[0], dst_ne=key[1], src_nb=key[2], dst_nb=key[3],
                       source_path=key[4], inferred_active_workers=key[5], count=count)
                  for key, count in sorted(grouped.items(), key=lambda item: -item[1])]
        operations = json.loads(operations_path.read_text())
        graph_groups = []
        for group in operations['groups']:
            graph_groups.append(dict(first=group['first'], last=group['last'], graphs=group['graphs'],
                                     summed_socket_graph_ms=group['total_ms'],
                                     summed_copy_stage_ms=group['operations'].get('CPY', 0)))
        rows.append(dict(kind=trace['kind'], output_and_draft_counts_match=True,
                         copy_contiguity_counts=trace['copy_contiguity_counts'],
                         cpu_counts=dict(Counter(row['cpu'] for row in layouts)),
                         copy_paths=[dict(source_path=key[0], inferred_active_workers=key[1], count=count)
                                     for key, count in workers.items()],
                         layouts=groups, operation_groups=graph_groups))
    analysis = dict(time=time.time(), passed=True, diagnostic_only=True, target_reached=False,
                    input_sha256=inputs, runtime_cpu_sha256=plan['cpu_sha256'], rows=rows,
                    underfilled_strided_copy_observed=any(
                        group['source_path'] == 'same-shape strided row split' and group['inferred_active_workers'] < 15
                        for row in rows for group in row['copy_paths']),
                    limitations=[
                        'Worker counts are inferred from the verified current source branch and tensor layout, not sampled worker execution.',
                        'Only the first 128 large FP32 copies after arming each request were logged.',
                        'Operation timings include following barriers and diagnostic overhead. Concurrent socket times cannot be summed into request wall time.',
                        'No model throughput or memory-bandwidth claim follows from these diagnostic requests.'])
    target = out / 'copy-layout-analysis.json'
    with target.open('x') as handle:
        json.dump(analysis, handle, indent=2)
        handle.write('\n')
    print(json.dumps(analysis, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('label')
    args = parser.parse_args()
    assert re.fullmatch(r'qwen-private-cpy-strides-[A-Za-z0-9_-]+', args.label)
    main(args.label)
