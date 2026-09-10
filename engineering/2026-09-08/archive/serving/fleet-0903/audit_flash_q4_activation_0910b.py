#!/usr/bin/env python3
"""Verify Q4 activation independently of performance-measurement conditions."""
import fcntl
import json
import os
from pathlib import Path
import time

from model_measurement_guard import ModelMeasurementGuard
from qwen_high_quant_trial import port_available, unit_state
from qwen_split_trial import inference_snapshot, process_environment, process_info, runtime_environment, sha256
from select_flash_q4_0910b import BASE, OUT, PORT, SELECTED, Manager, mapped_libraries, verify_plan


def main():
    output = BASE / 'results/glm-flash-q4-activation-audit-0910b.json'
    assert not output.exists()
    plan, result, selected = [json.loads(p.read_text()) for p in (OUT/'plan.json', OUT/'result.json', SELECTED)]
    assert result['finished'] and result['passed'] and result['selected'] and not result.get('error')
    assert result['plan_sha256'] == sha256(OUT/'plan.json') and result['selected_config_sha256'] == sha256(SELECTED)
    assert selected['quant'] == selected['user_authorized_quant'] == 'UD-Q4_K_XL'
    assert selected['drafts'] == 2 and selected['draft_sha256'] == plan['configurations']['mtp2']['draft_sha256']
    verify_plan(plan)
    assert result['runs'] == [] and result['bandwidth_measurements_pending'] and not result['target_reached']
    assert len(result['smoke_checks']) == 2 and all(row['passed'] for row in result['smoke_checks'])
    expected = {'arithmetic':'391','geography':'Paris'}
    for row in result['smoke_checks']:
        content = row['response']['choices'][0]['message'].get('content') or ''
        assert content.strip().strip('.').strip() == expected[row['name']]
    assert len(result['request_checks']) == 4 and all(check['passed'] for check in result['request_checks'])
    for check in result['request_checks']:
        assert check['fresh']['tokens'] and check['fresh']['tokens'] == check['default']['tokens']
        assert check['fresh']['timings']['cache_n'] == check['default']['timings']['cache_n'] == 0
    current = Manager().validate_current()
    assert current['pid'] == result['selected_pid'] and current['info']['start'] == result['selected_start']
    peer = process_info(current['pid'])
    assert peer['command'] == selected['command'] and peer['affinity'] == selected['affinity']
    assert runtime_environment(process_environment(current['pid'])) == selected['runtime_env']
    mapped = mapped_libraries(current['pid'])
    assert str(Path(selected['cpu_library']).resolve()) in mapped
    assert all(sha256(p) == digest for p,digest in selected['runtime_sha256'].items())
    assert set(inference_snapshot()) == {str(current['pid'])}
    ModelMeasurementGuard(current['pid'], {current['pid']:PORT}, inference_snapshot).assert_idle()
    assert port_available(18095) and unit_state()['ActiveState'] == 'inactive'
    paths = [Path(__file__).resolve(),OUT/'plan.json',OUT/'result.json',SELECTED,BASE/'select_flash_q4_0910b.py',
             BASE/'results/flash-q4-selection-checks-0910b.json',BASE/'results/glm-flash-q4-download-0910/status.json',
             BASE/'results/glm-flash-q4-final-shard-0910/result.json']
    audit = dict(time=time.time(),passed=True,selected_quant='UD-Q4_K_XL',selected_pid=current['pid'],
        selected_start=current['info']['start'],port=PORT,smoke_checks_passed=2,request_checks_passed=4,
        runtime_identity_verified=True,qwen_q6_preserved_but_unloaded=True,full_inactive=True,
        performance_measurements_separate=True,all_model_goal_complete=False,
        source_sha256={str(p):sha256(p) for p in paths},
        scope='Independent model/runtime identity, arithmetic/geography and default-cache-off continuation checks. '
              'No performance or checkpoint-quality equivalence claim.')
    output.write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k!='source_sha256'},indent=2))


if __name__ == '__main__':
    os.umask(0o077)
    assert os.sched_getaffinity(0) == {127}
    with (BASE/'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
