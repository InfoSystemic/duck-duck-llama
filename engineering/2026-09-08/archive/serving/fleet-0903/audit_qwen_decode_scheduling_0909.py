#!/usr/bin/env python3
"""Check cleanup and the preserved peer after the incomplete scheduling comparison."""
import fcntl
import json
import os
from pathlib import Path
import time

from benchmark_qwen_resident_0908 import normalized_command
from glm_flash_q8_trial import memory_status, node_memory_status
from model_measurement_guard import ModelMeasurementGuard
from qwen_high_quant_trial import port_available, unit_state
from qwen_split_trial import inference_snapshot, process_environment, process_info, runtime_environment, sha256

BASE = Path(__file__).resolve().parent


def main():
    assert os.sched_getaffinity(0) == {127}
    out = BASE / 'results/qwen-decode-scheduling-host-audit-0909.json'
    assert not out.exists()
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        preset_path = BASE / 'qwen-flash-20tps.json'
        preset = json.loads(preset_path.read_text())
        peer = process_info(1219506)
        assert peer['start'] == '103969952'
        assert normalized_command(peer['command']) == normalized_command(preset['command'])
        environment = process_environment(1219506)
        assert runtime_environment(environment) == preset['runtime_env']
        assert not environment.get('LD_PRELOAD')
        assert set(inference_snapshot()) == {'1219506'}
        assert unit_state()['ActiveState'] == 'inactive'
        assert port_available(18131) and port_available(18155)
        ModelMeasurementGuard(1219506, {1219506: 18095}, inference_snapshot).assert_idle()
        memory, nodes = memory_status(), node_memory_status()
        assert memory['MemAvailable'] > 300_000_000_000
        assert len(nodes) == 4 and all(row['estimated_available'] > 70_000_000_000 for row in nodes.values())
        inputs = {str(path): sha256(path) for path in (Path(__file__).resolve(), preset_path)}
        trials = []
        for label in ('qwen-private-decode-scheduling-250-0909-0-parent', 'qwen-private-decode-scheduling-250-0909-1-candidate'):
            path = BASE / 'results' / label / 'result.json'
            row = json.loads(path.read_text())
            assert row['passed'] and row['finished'] and not row.get('error')
            assert row['owned_model_exit'] == 0 and row['dispatch_cleanup_valid']
            assert row['peer_preserved'] and row['peer_environment_preserved']
            assert not Path('/proc') .joinpath(str(row['model_pid'])).exists()
            inputs[str(path)] = sha256(path)
            trials.append(dict(label=label, model_pid=row['model_pid'], owned_model_exit=0,
                               shared_groups_released=True))
        result = dict(time=time.time(), passed=True, input_sha256=inputs, trials=trials,
                      peer_pid=1219506, peer_start='103969952', peer_preserved=True,
                      peer_runtime_matches_selected_preset=True, peer_probe_absent=True,
                      peer_idle=True, full_inactive=True, private_ports_free=True,
                      lifecycle_lock_acquired=True, available_bytes=memory['MemAvailable'],
                      node_available_bytes={key: row['estimated_available'] for key, row in nodes.items()},
                      no_models_started_or_stopped=True)
        with out.open('x') as handle:
            json.dump(result, handle, indent=2)
            handle.write('\n')
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
