#!/usr/bin/env python3
"""Read only the Q8 tile candidate's one-byte activation flag."""
import fcntl
import json
from pathlib import Path
import subprocess
import time

from glm_flash_q8_trial import BASE, Manager, PORT
from model_measurement_guard import ModelMeasurementGuard
from qwen_split_trial import inference_snapshot, sha256
from verify_flash_q8_clamp_dispatch_0908 import READER


def main():
    with (BASE/'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manager = Manager()
        current = manager.validate_current()
        assert current['expert_tile_rows'] == 48 and current['drafts'] == 0
        assert current['runtime_env']['GGML_CPU_Q8_MOE_TILE_ROWS'] == '48'
        assert not current.get('op_profile_count')
        library = Path(current['cpu_library']).resolve()
        assert sha256(library) == current['cpu_sha256'] == 'd0449ea0773ee19cfef28b1efdc618fccea0304e700218e0ef682f1f2369ba70'
        guard = ModelMeasurementGuard(current['pid'], {current['pid']: PORT}, inference_snapshot)
        guard.assert_idle()
        parity_path = BASE/'results/glm-flash-q8-tiles48-parity-0908/result.json'
        parity = json.loads(parity_path.read_text())
        assert parity['passed'] and parity['current']['pid'] == current['pid']
        table = subprocess.check_output(['readelf', '--dyn-syms', '-W', str(library)], text=True)
        entries = [line.split() for line in table.splitlines() if line.split()
                   and 'forward_x16_moe_swiglu' in line.split()[-1]
                   and line.split()[-1].endswith('11tile_logged')]
        assert len(entries) == 1
        entry, = entries
        assert entry[2:5] == ['1', 'OBJECT', 'UNIQUE']
        symbol, offset = entry[-1], int(entry[1], 16)
        completed = subprocess.run(['sudo', '-n', 'python3', '-c', READER,
                                    str(current['pid']), current['info']['start'], str(library), str(offset)],
                                   capture_output=True, text=True, timeout=10)
        assert completed.returncode == 0, completed.stderr
        observation = json.loads(completed.stdout)
        manager.validate_current()
        guard.assert_idle()
        result = dict(time=time.time(), pid=current['pid'], cpu_library=str(library),
                      cpu_sha256=current['cpu_sha256'], symbol=symbol, symbol_offset=offset,
                      observation=observation, passed=observation['value'] == 1,
                      source_sha256={str(p): sha256(p) for p in
                                     (Path(__file__), BASE/'verify_flash_q8_clamp_dispatch_0908.py')},
                      parity_sha256=sha256(parity_path),
                      scope='The flag is set only when a nondefault Q8 tile executes on worker zero. The verified environment selects 48 rows. This proves execution, not its frequency or a speed gain.')
        destination = BASE/'results/glm-flash-q8-tiles48-dispatch-0908.json'
        assert not destination.exists()
        destination.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, indent=2))
        assert result['passed'], 'The 48-row Q8 tile has not executed'


if __name__ == '__main__':
    main()
