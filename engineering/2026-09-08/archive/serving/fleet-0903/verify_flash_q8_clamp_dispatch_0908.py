#!/usr/bin/env python3
"""Read only the candidate's one-byte activation flag after model requests."""
import fcntl
import json
from pathlib import Path
import subprocess
import time

from glm_flash_q8_trial import BASE, Manager, PORT
from model_measurement_guard import ModelMeasurementGuard
from qwen_split_trial import inference_snapshot, sha256

SYMBOL = '_ZZN4ggml3cpu6repack17tensor_traits_x1641compute_forward_mul_mat_id_swiglu_clampedEP19ggml_compute_paramsP11ggml_tensorS6_S6_E6logged'

READER = r'''
import json,os,pathlib,sys
pid,start,library,offset=sys.argv[1:]
proc=pathlib.Path('/proc')/pid
assert (proc/'stat').read_text().rsplit(')',1)[1].split()[19]==start
maps=[line.split() for line in (proc/'maps').read_text().splitlines() if '/libggml-cpu.so.' in line]
assert {line[-1] for line in maps}=={library}
bases=[int(line[0].split('-')[0],16) for line in maps if int(line[2],16)==0]
assert len(bases)==1
address=bases[0]+int(offset)
fd=os.open(proc/'mem',os.O_RDONLY)
try:
    value=os.pread(fd,1,address)
finally:
    os.close(fd)
assert len(value)==1 and value[0] in (0,1)
assert (proc/'stat').read_text().rsplit(')',1)[1].split()[19]==start
print(json.dumps(dict(base_address=hex(bases[0]),symbol_address=hex(address),bytes_read=1,value=value[0])))
'''


def main():
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manager = Manager()
        current = manager.validate_current()
        assert current['q8_clamp'] and current['drafts'] == 0 and not current.get('op_profile_count')
        library = Path(current['cpu_library']).resolve()
        assert sha256(library) == current['cpu_sha256'] == '0f2feaaa6a173a23bf5adde440ec5541348f1cb7ac2edee7d77bebc15af9ec23'
        guard = ModelMeasurementGuard(current['pid'], {current['pid']: PORT}, inference_snapshot)
        guard.assert_idle()
        parity_path = BASE / 'results/glm-flash-q8-clamp-raw-parity-0908/result.json'
        parity = json.loads(parity_path.read_text())
        assert parity['passed'] and parity['current']['pid'] == current['pid']
        table = subprocess.check_output(['readelf', '--dyn-syms', '-W', str(library)], text=True)
        entries = [line.split() for line in table.splitlines() if line.split() and line.split()[-1] == SYMBOL]
        assert len(entries) == 1
        entry = entries[0]
        assert entry[2:5] == ['1', 'OBJECT', 'UNIQUE']
        offset = int(entry[1], 16)
        completed = subprocess.run(['sudo', '-n', 'python3', '-c', READER,
            str(current['pid']), current['info']['start'], str(library), str(offset)],
            capture_output=True, text=True, timeout=10)
        assert completed.returncode == 0, completed.stderr
        observation = json.loads(completed.stdout)
        manager.validate_current()
        guard.assert_idle()
        result = dict(time=time.time(), pid=current['pid'], cpu_library=str(library),
                      cpu_sha256=current['cpu_sha256'], symbol=SYMBOL, symbol_offset=offset,
                      observation=observation, passed=observation['value'] == 1,
                      source_sha256=sha256(__file__), parity_sha256=sha256(parity_path),
                      scope='The flag is set only after the clamped Q8 method returns success on worker zero. This proves dispatch occurred, not how many layers fused or a speed gain.')
        destination = BASE / 'results/glm-flash-q8-clamp-dispatch-0908.json'
        destination.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2))
        assert result['passed'], 'Clamped Q8 dispatch has not executed'


if __name__ == '__main__':
    main()
