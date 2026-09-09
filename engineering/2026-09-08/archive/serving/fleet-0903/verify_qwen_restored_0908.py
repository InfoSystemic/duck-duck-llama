#!/usr/bin/env python3
"""Verify the selected Q6 preset after a private trial."""
import argparse
import fcntl
import json
from pathlib import Path
import re
import time

from model_measurement_guard import ModelMeasurementGuard, read_service
from qwen_high_quant_trial import BASE, OUT, STATE, unit_state
from qwen_split_trial import inference_snapshot, process_environment, process_info, runtime_environment, sha256

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--label', default='qwen-q6-post-profile-0908')
parser.add_argument('--after-flash', action='store_true')
args = parser.parse_args()
assert re.fullmatch(r'[A-Za-z0-9_-]+', args.label)
destination = BASE / 'results' / (args.label + '.json')
assert not destination.exists()
with (OUT / 'lifecycle.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = json.loads(STATE.read_text())
    current = state['current']
    preset_path = BASE / 'qwen-flash-20tps.json'
    preset = json.loads(preset_path.read_text())
    pid = current['pid']
    actual = process_info(pid)
    assert actual['start'] == current['info']['start']
    assert actual['command'] == current['command'] == preset['command']
    environment = runtime_environment(process_environment(pid))
    assert environment == current['runtime_env']
    assert not any(k.startswith('GGML_CPU_OP_PROFILE') for k in environment)
    assert 'GGML_CPU_SINGLE_SMALL_F32_MM' not in environment
    control_file = environment.pop('GGML_CPU_NUMA_THREADS_FILE')
    assert int(Path(control_file).read_text()) == preset['threads_per_socket'] == 15
    assert environment == preset['runtime_env']
    assert sha256(current['cpu_library']) == current['cpu_sha256']
    guard = ModelMeasurementGuard(pid, {pid: 18095}, inference_snapshot)
    guard.assert_idle()
    unit = unit_state()
    assert state['full_stopped'] and unit['ActiveState'] == 'inactive' and unit['MainPID'] == '0'
    if args.after_flash:
        flash = json.loads((BASE / 'results/glm-flash-q8-trial-0908/state.json').read_text())
        assert flash['current'] is None
        assert flash['qwen_rollback']['command'] == current['command']
    result = dict(time=time.time(), passed=True, pid=pid, start=actual['start'], health=read_service(18095),
                  profile_disabled=True, experimental_scheduling_loaded=False,
                  preset=str(preset_path), preset_sha256=sha256(preset_path),
                  command=actual['command'], runtime_env=environment,
                  cpu_library=current['cpu_library'], cpu_sha256=current['cpu_sha256'],
                  full_unit=unit, model_files_deleted=False,
                  source_sha256=sha256(__file__), after_flash=args.after_flash)
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('passed', 'pid', 'health', 'profile_disabled', 'experimental_scheduling_loaded')}))
