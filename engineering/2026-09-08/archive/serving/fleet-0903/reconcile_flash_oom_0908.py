#!/usr/bin/env python3
"""Reconcile the verified Flash OOM loss; never stop or start a model."""
import copy
import fcntl
import json
import os
from pathlib import Path
import time
import traceback
from unittest.mock import patch

from exclusive_model_launch import inference_processes
import glm_flash_q8_trial as flash
from model_measurement_guard import read_service
from qwen_high_quant_trial import STATE as QWEN_STATE, unit_state
from qwen_split_trial import sha256


def snapshot():
    rows = inference_processes()
    for row in rows:
        row['start_ticks'] = (Path('/proc') / str(row['pid']) / 'stat').read_text().rsplit(')', 1)[1].split()[19]
    return rows


def main():
    os.umask(0o077)
    out = flash.BASE / 'results/glm-flash-oom-reconcile-0908'
    result = dict(started=time.time(), passed=False, model_started=False, model_stopped=False)
    with (flash.BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manager = flash.Manager()
        retained_path = flash.BASE / 'results/glm-flash-q8-r8-ordered-k-post-model-0908.json'
        incident_path = flash.BASE / 'results/glm-flash-external-qwen-oom-0908.json'
        retained = json.loads(retained_path.read_text())
        incident = json.loads(incident_path.read_text())
        current = copy.deepcopy(manager.state['current'])
        assert retained['passed'] and current == retained['current']
        assert current['pid'] == 462943 and current['info']['start'] == '103663696'
        assert incident['flash_process_absent'] and incident['flash_pid'] == current['pid']
        assert not (Path('/proc') / str(current['pid'])).exists()
        assert flash.port_available(flash.PORT)
        assert manager.qwen.state['current'] is None and manager.qwen.state['full_stopped']
        assert unit_state()['ActiveState'] == 'inactive'
        before = snapshot()
        assert len(before) == 1 and before[0]['pid'] == 1219506
        assert before[0]['start_ticks'] == '103969952' and before[0]['port'] == '18095'
        state_hash = sha256(flash.STATE)
        qwen_hash = sha256(QWEN_STATE)
        controller_hash = sha256(flash.BASE / 'glm_flash_q8_trial.py')
        out.mkdir(exist_ok=False)
        try:
            backup = out / 'state.before.json'
            backup.write_bytes(flash.STATE.read_bytes())
            assert sha256(backup) == state_hash == sha256(flash.STATE)
            (out / 'last-measured-configuration.json').write_text(json.dumps(current, indent=2) + '\n')
            assert snapshot() == before
            assert not (Path('/proc') / str(current['pid'])).exists()
            assert flash.port_available(flash.PORT)
            result.update(inference_before=before, state_before_sha256=state_hash,
                          preserved_configuration=str(out / 'last-measured-configuration.json'),
                          sources={str(p): sha256(p) for p in [Path(__file__).resolve(), retained_path,
                                   incident_path, flash.BASE / 'glm_flash_q8_trial.py']})
            manager.state['current'] = None
            manager.state['last_lost_current'] = dict(configuration=current, cause='verified external-launch OOM',
                                                     evidence=str(incident_path), reconciled_at=time.time())
            manager.record('reconciled_oom_loss', pid=current['pid'], start_ticks=current['info']['start'],
                           evidence=str(incident_path), preserved_configuration=result['preserved_configuration'],
                           external_qwen_preserved=before[0]['pid'])
            reloaded = flash.Manager()
            assert reloaded.state['current'] is None
            assert reloaded.state['last_lost_current']['configuration'] == current
            assert reloaded.state['qwen_rollback'] == json.loads(backup.read_text())['qwen_rollback']
            assert sha256(QWEN_STATE) == qwen_hash
            checks = []
            try:
                reloaded.quiet()
            except AssertionError as error:
                assert str(error) == 'Another inference process is loaded'
                checks.append('manager quiet gate refuses external Qwen')
            else:
                raise AssertionError('Manager incorrectly accepted a resident external model')
            with patch.object(flash.subprocess, 'Popen', side_effect=RuntimeError('Model startup must not be reached')):
                try:
                    reloaded.start_flash({}, {})
                except AssertionError:
                    checks.append('direct Flash start refuses before process creation')
                else:
                    raise AssertionError('Flash start incorrectly accepted a resident external model')
            assert snapshot() == before
            assert sha256(flash.BASE / 'glm_flash_q8_trial.py') == controller_hash
            assert not (Path('/proc') / str(current['pid'])).exists()
            result.update(passed=True, checks=checks, inference_after=snapshot(),
                          qwen_service=read_service(18095), state_after_sha256=sha256(flash.STATE),
                          qwen_state_unchanged=True, controller_unchanged=True,
                          flash_current=None, full_stopped=True, memory=flash.memory_status(),
                          remaining='External Qwen release or user permission to pause it is required before a whole-server model trial')
        except BaseException:
            result['error'] = traceback.format_exc()
            raise
        finally:
            result['finished'] = time.time()
            (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps({key: result[key] for key in ['passed', 'checks', 'model_started', 'model_stopped',
                              'inference_after', 'qwen_service', 'flash_current', 'remaining', 'error'] if key in result}, indent=2))


if __name__ == '__main__':
    main()
