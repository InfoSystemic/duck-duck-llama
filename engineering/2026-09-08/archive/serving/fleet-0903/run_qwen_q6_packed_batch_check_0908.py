#!/usr/bin/env python3
"""Check a standalone packed-Q6 batch primitive; no performance claim."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from model_measurement_guard import ModelMeasurementGuard
from qwen_split_trial import inference_snapshot, process_info, sha256

BASE = Path(__file__).resolve().parent
ENGINE = BASE.parent.parent / 'engines/llama.cpp-q4e-goal-0904'
OUT = BASE / 'results/qwen-q6-packed-batch-proof-0908'


def main():
    OUT.mkdir(exist_ok=False)
    proof_path = BASE / 'results/qwen-q6-lossless-bytes-prototype-0908/result.json'
    proof = json.loads(proof_path.read_text())
    assert proof['passed'] and all(sha256(p) == digest for p,digest in proof['source_sha256'].items())
    cpu = BASE / 'results/qwen-q6-q8-wide-batch-0907/private-cpu/libggml-cpu.so.0.22.0'
    command = list(proof['compile_command'])
    source = BASE / 'check-qwen-q6-packed-batch-0908.cpp'
    command[command.index(str(BASE / 'check-qwen-q6-lossless-bytes-0908.cpp'))] = str(source)
    command[command.index('-o')+1] = str(OUT / 'check')
    inputs = [Path(__file__).resolve(),source,BASE / 'qwen-q6-packed-batch-0908.h',cpu,proof_path,
              BASE / 'model_measurement_guard.py',BASE / 'qwen_split_trial.py']
    result = dict(started=time.time(),controller_pid=os.getpid(),passed=False,compile_command=command,
                  source_sha256={str(path):sha256(path) for path in inputs},scope=__doc__)
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            assert process_info(1219506)['start'] == '103969952'
            guard = ModelMeasurementGuard(1219506,{1219506:18095},inference_snapshot)
            guard.assert_idle()
            for path in inputs[:3]:
                (OUT / path.name).write_bytes(path.read_bytes())
            with (OUT / 'compile.log').open('w') as log:
                compiled = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=120)
            result['compile_exit'] = compiled.returncode
            assert compiled.returncode == 0
            environment = dict(os.environ,LD_LIBRARY_PATH=str(cpu.parent)+':'+str(ENGINE / 'validated-iq-batch3-bin'))
            checked = subprocess.run(['taskset','-c','127',str(OUT / 'check')],env=environment,
                                     capture_output=True,text=True,timeout=120)
            result['check_exit'] = checked.returncode
            (OUT / 'check.log').write_text(checked.stdout+checked.stderr)
            assert checked.returncode == 0
            result['check'] = json.loads(checked.stdout)
            assert result['check']['passed'] and Path(result['check']['cpu_library']).resolve() == cpu.resolve()
            assert all(sha256(path) == digest for path,digest in result['source_sha256'].items())
            result['binary_sha256'] = sha256(OUT / 'check')
            result['passed'] = True
        except BaseException as error:
            result['error'] = repr(error)
            raise
        finally:
            result['finished'] = time.time()
            result['peer_preserved'] = process_info(1219506)['start'] == '103969952'
            (OUT / 'result.json').write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps({key:result.get(key) for key in ('passed','check','error','peer_preserved')}),flush=True)


if __name__ == '__main__':
    main()
