#!/usr/bin/env python3
"""Record only qualified graph pairs and the limits of the Q6 kernel experiment."""
import json
import math
from pathlib import Path
import statistics
import time

from qwen_split_trial import sha256

BASE = Path(__file__).resolve().parent


def main():
    paths = [BASE/'results'/name for name in (
        'qwen-q6-packed-single-component-qualified-0910/result.json',
        'qwen-q6-single-build-0910b/result.json',
        'qwen-q6-single-validation-0910/result.json',
        'qwen-q6-single-host-audit-0910.json',
        'qwen-q8-precision-assessment-0910.json')]
    component, build, validation, audit, precision = [json.loads(path.read_text()) for path in paths]
    assert all(record['passed'] for record in (component, build, validation, audit, precision))
    assert component['finished'] and build['finished'] and validation['finished']
    assert validation['cpu_sha256'] == build['library_sha256'] == sha256(build['library'])
    assert validation['bit_exact'] and not validation['model_test_eligible']
    assert audit['owned_components_released'] and audit['peer_preserved']
    assert len(validation['checks']) == 7 and all(len(row['rows']) == 30 for row in validation['checks'])
    assert len(validation['numa_checks']) == 10
    paths += [Path(__file__).resolve()]
    for record in (component, build, validation, audit, precision):
        for key in ('input_sha256', 'private_source_sha256'):
            assert all(sha256(path) == digest for path, digest in record.get(key, {}).items())
    primitives = []
    for case in component['cases']:
        row = case['attempts'][case['accepted_attempt']]
        assert row['background_within_gate'] and row['other_host_cores'] <= 4
        primitives.append(dict(rows=case['rows'], matrices=case['matrices'], speed_ratio=row['packed_batch_speed_ratio']))
    pairs = []
    for socket in (0, 2):
        arms = [row for row in validation['timings'] if row['socket'] == socket]
        assert len(arms) == 4
        for control_index, candidate_index in ((0, 1), (3, 2)):
            control, candidate = arms[control_index], arms[candidate_index]
            assert control['parent'] and candidate['enabled']
            if not control['background_within_gate'] or not candidate['background_within_gate']:
                continue
            rows = []
            for before, after in zip(control['rows'], candidate['rows']):
                assert before['hash'] == after['hash']
                if before['rotating']:
                    rows.append(dict(tokens=before['nr'], disjoint=before['disjoint'],
                                     parent_us=before['median_us'], candidate_us=after['median_us'],
                                     speed_ratio=before['median_us']/after['median_us']))
            assert len(rows) == 3
            pairs.append(dict(socket=socket, control=control['label'], candidate=candidate['label'], rows=rows,
                              cold_geomean_speed_ratio=math.exp(statistics.mean(math.log(row['speed_ratio']) for row in rows))))
    assert len(pairs) == 2 and all(row['cold_geomean_speed_ratio'] < 1.02 for row in pairs)
    result = dict(time=time.time(), passed=True, input_sha256={str(path): sha256(path) for path in paths},
                  single_core_qualified_cases=primitives, expert_graph_cases=210, four_numa_graph_arms=10,
                  exact_outputs=True, packed_weights_unchanged=True, cpu_sha256=build['library_sha256'],
                  parent_cpu_sha256=build['parent_sha256'], graph_timing_arms=8,
                  qualified_graph_timing_arms=sum(row['background_within_gate'] for row in validation['timings']),
                  qualified_cold_pairs=pairs, repeated_graph_speed_gain_established=False,
                  model_trial_eligible=False, model_controller_executed=False, new_model_bandwidth_measurements=0,
                  candidate_promoted=False, bandwidth_target_reached=False,
                  q8_precision_assessment=str(paths[4]),
                  scope='The single-core arithmetic gain does not establish a graph or model gain. Four graph timing arms fail the existing background limit; the two fully qualified cold pairs do not meet the predeclared 2% graph gate. Exact expert and four-NUMA outputs pass, but the prepared full-model controller remains unexecuted. All three models remain below the requested 250 GB/s acceptance target.')
    out = BASE/'results/qwen-q6-single-assessment-0910.json'
    with out.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'input_sha256'}, indent=2))


if __name__ == '__main__':
    main()
