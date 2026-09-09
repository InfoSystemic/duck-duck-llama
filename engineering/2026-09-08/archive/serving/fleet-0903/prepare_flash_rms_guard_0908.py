#!/usr/bin/env python3
"""Verify the guarded RMS launch and the unchanged sum16 rollback."""
import fcntl
import json
import os
import time

from glm_flash_q8_trial import BASE, Manager, PORT, memory_status
from model_measurement_guard import ModelMeasurementGuard
from qwen_split_trial import inference_snapshot, sha256


def main():
    os.umask(0o077)
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manager = Manager()
        current = manager.validate_current()
        guard = ModelMeasurementGuard(current['pid'], {current['pid']: PORT}, inference_snapshot)
        guard.assert_idle()
        assert current['sum16'] and not current.get('rms_guard') and current['drafts'] == 0
        assert not manager.qwen.state.get('current') and manager.qwen.state['full_stopped']
        options = dict(batching=True, drafts=0, workers=15, q8_experts=True,
                       q8_clamp=True, pooling=True, sum16=True)
        baseline, _ = manager.candidate(**options)
        assert all(current.get(key, False if key == 'rms_guard' else None) == value
                   for key, value in baseline.items())
        candidate, _ = manager.candidate(**options, rms_guard=True)
        assert baseline['command'][1:] == candidate['command'][1:]
        assert sha256(baseline['command'][0]) == sha256(candidate['command'][0])
        expected_changes = {'command', 'runtime_env', 'rms_guard', 'cpu_library',
                            'cpu_sha256', 'runtime_directory'}
        assert {key for key in baseline if baseline[key] != candidate[key]} == expected_changes
        a, b = baseline['runtime_env'], candidate['runtime_env']
        changes = {key: [a.get(key), b.get(key)] for key in a.keys() | b.keys()
                   if a.get(key) != b.get(key)}
        assert set(changes) == {'LD_LIBRARY_PATH', 'GGML_CPU_RMS_F64_SIMD'}
        assert changes['GGML_CPU_RMS_F64_SIMD'] == [None, '1']
        assert not any('AUDIT' in key or 'PROFILE' in key for key in b)
        output = BASE / 'results/glm-flash-rms-guard-control-0908/preflight.json'
        assert not output.exists()
        result = dict(time=time.time(), passed=True, parent=current, baseline=baseline,
                      candidate=candidate, runtime_env_changes=changes,
                      memory=memory_status(), controller_sha256=sha256(BASE / 'glm_flash_q8_trial.py'),
                      source_sha256=sha256(__file__))
        output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(dict(passed=True, parent_pid=current['pid'],
                              candidate_cpu_sha256=candidate['cpu_sha256'],
                              runtime_env_changes=changes)), flush=True)


if __name__ == '__main__':
    main()
