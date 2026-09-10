#!/usr/bin/env python3
"""Independently recheck the selected Q4 process and saved decode counters."""
import fcntl
import json
import os
from pathlib import Path
import statistics
import time

from analyze_flash_bandwidth250_0910 import validate_counters
from model_measurement_guard import ModelMeasurementGuard
from qwen_high_quant_trial import port_available, unit_state
from qwen_split_trial import inference_snapshot, process_environment, process_info, runtime_environment, sha256
from select_flash_q4_0910 import BASE, OUT, PORT, SELECTED, Manager, mapped_libraries, measurement_rows, verify_plan


def main():
    output = BASE / 'results/glm-flash-q4-selection-audit-0910.json'
    assert not output.exists()
    plan, result, selected = [json.loads(p.read_text()) for p in (OUT/'plan.json', OUT/'result.json', SELECTED)]
    assert result['finished'] and result['passed'] and result['selected'] and not result.get('error')
    assert result['plan_sha256'] == sha256(OUT/'plan.json') and result['selected_config_sha256'] == sha256(SELECTED)
    assert selected['quant'] == selected['user_authorized_quant'] == 'UD-Q4_K_XL'
    assert selected['drafts'] == 2 and selected['draft_sha256'] == plan['configurations']['mtp2']['draft_sha256']
    verify_plan(plan)
    assert [run['configuration'] for run in result['runs']] == plan['sequence']
    sources = {str(p): sha256(p) for p in (Path(__file__).resolve(), OUT/'plan.json', OUT/'result.json', SELECTED,
               BASE/'select_flash_q4_0910.py', BASE/'analyze_flash_bandwidth250_0910.py', BASE/'dram_bandwidth.py')}
    checked = []
    for run in result['runs']:
        path = Path(run['measurement'])
        assert sha256(path) == run['measurement_sha256']
        current = plan['configurations'][run['configuration']]
        rows = measurement_rows(path, current)
        assert rows == run['rows']
        data = json.loads(path.read_text())
        assert data['target_pid'] == run['model_pid'] and data['config']['allowed_idle_pids'] == ''
        assert data['config']['chat_template_kwargs'] == {'reasoning_effort': 'max'}
        assert data['config']['reasoning_budget_tokens'] is None
        counters = {}
        for row in data['measurements']:
            directory = path.parent / f"{row['kind']}-draft{row['draft_n']}"
            counters[row['kind']] = validate_counters(directory, row)
            for name in ('samples.json', 'perf.csv', 'chunks.json', 'command.json'):
                p = directory / name
                sources[str(p)] = sha256(p)
        assert counters == run['counters']
        sources[str(path)] = sha256(path)
        checked.append(dict(index=run['index'], configuration=run['configuration'], rows=rows, counters=counters,
                            measurement=str(path), measurement_sha256=sha256(path)))
    summaries = {}
    for name in ('raw', 'mtp2'):
        runs = [run for run in checked if run['configuration'] == name]
        assert len(runs) == (1 if name == 'raw' else 2)
        summaries[name] = {}
        for kind in ('prose', 'code'):
            rows = [run['rows'][kind] for run in runs]
            for row in rows:
                assert all(row[k] == rows[0][k] for k in ('output_sha256', 'generated_tokens', 'draft_tokens', 'accepted_draft_tokens'))
            summaries[name][kind] = dict(repeats=len(rows), mean_tok_s=statistics.mean(row['tok_s'] for row in rows),
                mean_adjusted_gb_s=statistics.mean(row['adjusted_gb_s'] for row in rows),
                tok_s_range=[min(row['tok_s'] for row in rows), max(row['tok_s'] for row in rows)],
                adjusted_gb_s_range=[min(row['adjusted_gb_s'] for row in rows), max(row['adjusted_gb_s'] for row in rows)],
                all_over_250=all(row['over_250_gb_s'] for row in rows),
                complete_answers=all(row['completed_answer'] for row in rows))
    assert len(result['request_checks']) == 4 and all(check['passed'] for check in result['request_checks'])
    for check in result['request_checks']:
        assert check['fresh']['tokens'] == check['default']['tokens']
        assert check['fresh']['timings']['cache_n'] == check['default']['timings']['cache_n'] == 0
    current = Manager().validate_current()
    assert current['pid'] == result['selected_pid'] and current['info']['start'] == result['selected_start']
    peer = process_info(current['pid'])
    assert peer['command'] == selected['command'] and peer['affinity'] == selected['affinity']
    assert runtime_environment(process_environment(current['pid'])) == selected['runtime_env']
    mapped = mapped_libraries(current['pid'])
    assert str(Path(selected['cpu_library']).resolve()) in mapped
    assert all(sha256(p) == digest for p, digest in selected['runtime_sha256'].items())
    assert set(inference_snapshot()) == {str(current['pid'])}
    ModelMeasurementGuard(current['pid'], {current['pid']:PORT}, inference_snapshot).assert_idle()
    assert port_available(18095) and unit_state()['ActiveState'] == 'inactive'
    reached = all(row['all_over_250'] for row in summaries['mtp2'].values())
    assert reached == result['target_reached']
    audit = dict(time=time.time(), passed=True, selected_quant='UD-Q4_K_XL', selected_pid=current['pid'],
                 selected_start=current['info']['start'], port=PORT, summaries=summaries, runs=checked,
                 source_sha256=sources, request_checks_passed=4, runtime_identity_verified=True,
                 qwen_q6_preserved_but_unloaded=True, full_inactive=True, flash_target_reached=reached,
                 all_model_goal_complete=False,
                 scope='Three fresh single-conversation Q4 decode runs; two MTP2 repetitions with matching full '
                       'output and counts. Reparse all 48 IMC counters and reproduce stable decode windows and '
                       'adjacent idle subtraction. Rates measure generated tokens, including reasoning. '
                       'Quality equivalence to Q8 or the released checkpoint is not claimed.')
    output.write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps({key:audit[key] for key in ('passed', 'selected_quant', 'selected_pid', 'summaries', 'flash_target_reached')}, indent=2))


if __name__ == '__main__':
    os.umask(0o077)
    assert os.sched_getaffinity(0) == {127}
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
