#!/usr/bin/env python3
"""Save the authorized, measured Q6 runtime with a reversible Q2 config backup."""
import fcntl
import json
import os
from pathlib import Path
import time

from model_measurement_guard import ModelMeasurementGuard
from qwen_high_quant_trial import unit_state
from qwen_split_trial import inference_snapshot, process_info, sha256

BASE = Path(__file__).resolve().parent
TRIAL = BASE / 'results/qwen-q6-trial-0907'
OUT = BASE / 'results/qwen-q6-promotion-0907'


def main():
    state = json.loads((TRIAL / 'state.json').read_text())
    current = state['current']
    assert state['full_stopped'] and not current['original']
    assert current['q8_wide_batch'] and current['drafts'] == 4 and current['draft_p_min'] == 0.3
    assert current['no_cache_prompt'] and '--no-cache-prompt' in current['command']
    assert process_info(current['pid'])['start'] == current['info']['start']
    assert process_info(current['pid'])['command'] == current['command']
    guard = ModelMeasurementGuard(current['pid'], {current['pid']: 18095}, inference_snapshot)
    guard.assert_idle()
    assert int(Path(current['runtime_env']['GGML_CPU_NUMA_THREADS_FILE']).read_text()) == 15
    assert current['runtime_env']['GGML_CPU_NUMA_THREADS'] == '15'
    unit = unit_state()
    assert unit['ActiveState'] == 'inactive' and int(unit['MainPID']) == 0
    quality_path = BASE / 'results/qwen-q6-default-cache-off-0907-quality/result.json'
    quality = json.loads(quality_path.read_text())
    assert quality['passed'] and quality['default_cache_off'] and quality['current']['pid'] == current['pid']
    assert quality['code_check']['exit_code'] == 0
    assert all(x['passed'] and not x['explicit_cache_request'] for x in quality['cache_checks'])
    labels = ('qwen-q6-mtp4-pmin03-q8wide-0907', 'qwen-q6-final-q8wide-0907')
    measurements = []
    for label in labels:
        path = BASE / 'results' / label / 'result.json'
        record = json.loads(path.read_text())
        controller = json.loads((BASE / 'results' / (label + '-controller.json')).read_text())
        assert controller['passed'] and controller['config']['threads'] == 15
        assert controller['current']['cpu_sha256'] == current['cpu_sha256']
        assert record['input_integrity_verified'] and all(x['pass_check'] for x in record['checks'])
        measurements.append(dict(path=str(path), sha256=sha256(path), rates={
            x['kind']: x['timings']['predicted_per_second'] for x in record['measurements']}))
    assert any(x['rates']['code'] > 28 for x in measurements)
    hashes = dict(state['binary_sha256'])
    hashes[current['cpu_library']] = current['cpu_sha256']
    split = BASE / 'results/qwen-expert-even-split-policy-0906b/private-split/libllama.so.0.3.0'
    hashes[str(split)] = 'd213ba477ef5d3f2a68568b30a31595ef9578b6dce4b3b81e95da6f83a3dac8b'
    assert all(sha256(p) == h for p, h in hashes.items())
    old_path = BASE / 'qwen-flash-20tps.json'
    old_bytes = old_path.read_bytes()
    old = json.loads(old_bytes)
    assert any('/UD-Q2_K_XL/' in x for x in old['command']), 'Default configuration changed; inspect before replacing'
    OUT.mkdir(exist_ok=False)
    (OUT / 'previous-q2-config.json').write_bytes(old_bytes)
    environment = dict(current['runtime_env'])
    # The durable launcher uses the fixed 15-worker setting, without depending
    # on the temporary experiment's mutable worker-control file.
    environment.pop('GGML_CPU_NUMA_THREADS_FILE')
    configuration = dict(command=current['command'], runtime_env=environment,
        binary_sha256=hashes, evidence=[x['path'] for x in measurements] + [str(quality_path)],
        threads_per_socket=15, measured_rates=measurements,
        model_quantization='UD-Q6_K_XL', model_revision='38bb39ee97821de2c9009abb7e93950eec396e66',
        note='Whole-server Q6, balanced NUMA split, MTP4 with confidence 0.3 and wider Q8 row batching. '
             'Code measured above 28 tok/s; rates depend on workload and background activity. '
             'Prompt reuse is off by default because explicit reuse has an unresolved extension-parity mismatch. '
             'The Q2 configuration is backed up; model files and the stopped Full service are retained.')
    serialized = json.dumps(configuration, indent=2) + '\n'
    selected = OUT / 'selected-q6-config.json'
    selected.write_text(serialized)
    guard.assert_idle()
    assert old_path.read_bytes() == old_bytes
    temporary = old_path.with_suffix('.json.tmp')
    temporary.write_text(serialized)
    temporary.replace(old_path)
    assert json.loads(old_path.read_text()) == configuration
    result = dict(passed=True, time=time.time(), live_pid=current['pid'], port=18095,
        previous_config_sha256=sha256(OUT / 'previous-q2-config.json'),
        selected_config=str(old_path), selected_config_sha256=sha256(old_path),
        backup=str(OUT / 'previous-q2-config.json'), measurements=measurements,
        default_cache_quality=str(quality_path), full_unit=unit)
    (OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(passed=True, live_pid=current['pid'], port=18095,
                         config=str(old_path), rates=measurements)))


if __name__ == '__main__':
    os.umask(0o077)
    with (TRIAL / 'lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
