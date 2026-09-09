#!/usr/bin/env python3
"""Check Flash candidate provenance and rollback without launching any model."""
import difflib
import fcntl
import json
from pathlib import Path
import time
import traceback

from exclusive_model_launch import inference_processes
from glm_flash_q8_trial import BASE, Manager, STATE, memory_status
from model_measurement_guard import read_service
from qwen_split_trial import sha256


def snapshot():
    rows = inference_processes()
    for row in rows:
        stat = Path('/proc') / str(row['pid']) / 'stat'
        row['start_ticks'] = stat.read_text().rsplit(')', 1)[1].split()[19]
    return rows


def main():
    out = BASE / 'results/glm-flash-dissemination-control-0908'
    assert out.is_dir()
    destination = out / 'preflight.json'
    assert not destination.exists(), 'Preserve the existing preflight record'
    controller = BASE / 'glm_flash_q8_trial.py'
    backup = out / 'glm_flash_q8_trial.before-dissemination.py'
    assert sha256(backup) == '257e11ea269caf6879fcf019ca684b5f5d62ec7c205c4d81a1ffbf1d5ac250dc'
    compile(controller.read_text(), str(controller), 'exec')
    (out / 'controller.patch').write_text(''.join(difflib.unified_diff(
        backup.read_text().splitlines(keepends=True), controller.read_text().splitlines(keepends=True),
        fromfile=str(backup), tofile=str(controller))))
    result = dict(time=time.time(), passed=False, scope='offline candidate and rollback preflight; no model load',
                  model_loaded=False, model_gain_established=False, target_reached=False)
    try:
        with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result['inference_before'] = snapshot()
            state_before = sha256(STATE)
            manager = Manager()
            kwargs = dict(batching=True, drafts=0, workers=15, q8_experts=True,
                          q8_clamp=True, pooling=True, sum16=True, rms_guard=True, ordered_k=True)
            baseline, _ = manager.candidate(**kwargs)
            candidate, _ = manager.candidate(**kwargs, dissemination=True)
            audit_path = BASE / 'results/glm-flash-q8-r8-ordered-k-post-model-0908.json'
            audit = json.loads(audit_path.read_text())
            assert audit['passed']
            retained = audit['current']
            assert all(value == retained.get(key, False) for key, value in baseline.items())
            assert baseline['dissemination'] is False and candidate['dissemination'] is True
            assert candidate['command'][1:] == baseline['command'][1:]
            changes = {key for key in candidate if candidate[key] != baseline[key]}
            assert changes == {'command', 'runtime_env', 'cpu_library', 'cpu_sha256',
                               'runtime_directory', 'dissemination'}, changes
            before, after = baseline['runtime_env'], candidate['runtime_env']
            env_changes = {key: dict(before=before.get(key), after=after.get(key))
                           for key in before.keys() | after.keys() if before.get(key) != after.get(key)}
            assert set(env_changes) == {'LD_LIBRARY_PATH', 'GGML_CPU_DISSEMINATION_BARRIER'}
            assert after['GGML_CPU_DISSEMINATION_BARRIER'] == '1'
            assert not any('AUDIT' in key or 'PROFILE' in key for key in after)
            assert sha256(baseline['command'][0]) == sha256(candidate['command'][0]) == \
                '20fa1560caa0bee73fc6b23672d7ef8e84640c72ec555a640fc2373fd82587ed'
            assert candidate['cpu_sha256'] == '1aba1739b7bc010e284cb63b03f12094af3757fab9354dbf6e800ae4fb05d8bf'
            result.update(baseline=baseline, candidate=candidate, changed_fields=sorted(changes),
                          runtime_env_changes=env_changes, controller_sha256=sha256(controller))
            paths = [Path(__file__).resolve(), controller, backup, audit_path,
                     BASE / 'results/glm-flash-local-barrier-probe-0908b/result.json',
                     BASE / 'results/glm-flash-dissemination-0908b/result.json',
                     BASE / 'results/glm-flash-dissemination-0908b/private-cpu/manifest.json',
                     BASE / 'results/glm-flash-dissemination-component-0908/result.json',
                     BASE / 'results/glm-flash-dissemination-runtime-0908/result.json']
            result['source_sha256'] = {str(path): sha256(path) for path in paths}
            result['inference_after'] = snapshot()
            assert result['inference_before'] == result['inference_after']
            assert sha256(STATE) == state_before
            result['manager_state_unchanged'] = True
            result['flash_recorded_pid'] = manager.state['current']['pid']
            result['flash_recorded_pid_absent'] = not (Path('/proc') / str(result['flash_recorded_pid'])).exists()
            result['services'] = {str(row['pid']): read_service(int(row['port']))
                                  for row in result['inference_after'] if row['port']}
            result['memory'] = memory_status()
            result['pending_coordination'] = 'Preserve external Qwen; wait for permission to pause it or independent release before model trial'
            result['decision'] = 'ready for exclusive model trial; candidate not loaded or retained'
            result['passed'] = True
    except BaseException:
        result['error'] = traceback.format_exc()
        raise
    finally:
        result['finished'] = time.time()
        path = destination if result['passed'] else out / ('preflight-failed-%d.json' % time.time_ns())
        path.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({key: result[key] for key in ('passed', 'scope', 'model_loaded', 'target_reached',
                          'decision', 'controller_sha256', 'inference_after', 'services', 'error') if key in result}, indent=2))


if __name__ == '__main__':
    main()
