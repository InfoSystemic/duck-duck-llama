#!/usr/bin/env python3
"""Re-audit completed Q4 measurements without generating or changing a model."""
import json
import os
from pathlib import Path
import time

from benchmark_flash_q4_selected_0910 import read_measurement
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import inference_snapshot, sha256
from select_flash_q4_0910c import BASE, SELECTED, Manager


def main():
    assert os.sched_getaffinity(0) == {127}
    destination = BASE / 'results/glm-flash-q4-measurement-audit-0910.json'
    assert not destination.exists()
    path = BASE / 'results/glm-flash-q4-measured-0910/result.json'
    measured = json.loads(path.read_text())
    assert measured['finished'] and measured['passed'] and len(measured['runs']) == 2
    assert measured['source_sha256'] == sha256(BASE / 'benchmark_flash_q4_selected_0910.py')
    assert measured['selected_config_sha256'] == sha256(SELECTED)
    activation = BASE / 'results/glm-flash-q4-activation-audit-0910c.json'
    assert measured['activation_audit_sha256'] == sha256(activation)
    current = Manager().validate_current()
    assert current['pid'] == measured['current']['pid']
    assert current['command'] == measured['current']['command']
    assert current['runtime_env'] == measured['current']['runtime_env']
    assert current['quant'] == 'UD-Q4_K_XL' and current['drafts'] == 2
    assert set(inference_snapshot()) == {str(current['pid'])}
    runs = []
    for run in measured['runs']:
        source = Path(run['measurement'])
        assert sha256(source) == run['measurement_sha256']
        rows = read_measurement(source, current)
        assert rows == run['rows']
        runs.append(dict(index=run['index'], rows=rows, source=str(source), source_sha256=sha256(source)))
    for kind in ('prose', 'code'):
        for field in ('output_sha256', 'generated_tokens', 'draft_tokens', 'accepted_draft_tokens'):
            assert runs[0]['rows'][kind][field] == runs[1]['rows'][kind][field]
    result = dict(passed=True, time=time.time(), selected_pid=current['pid'],
        selected_quant=current['quant'], runs=runs, summaries=measured['summaries'],
        all_48_counters_reparsed=True, repeated_outputs_and_counts_match=True,
        current_host_load_only=True, uncontended_maximum_established=False,
        target_reached=all(row['qualified_over_250'] for run in runs for row in run['rows'].values()),
        all_model_goal_complete=False,
        source_sha256={str(p): sha256(p) for p in (Path(__file__), path, SELECTED, activation,
            BASE / 'benchmark_flash_q4_selected_0910.py')})
    assert result['target_reached'] == measured['target_reached']
    atomic_json(destination, result)
    print(json.dumps({k: result[k] for k in ('passed', 'selected_pid', 'summaries',
        'all_48_counters_reparsed', 'repeated_outputs_and_counts_match', 'target_reached')}, indent=2))


if __name__ == '__main__':
    main()
