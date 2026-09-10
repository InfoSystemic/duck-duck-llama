#!/usr/bin/env python3
"""Guard a bounded CPU reference-graph integration check beside idle Flash."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from model_measurement_guard import ModelMeasurementGuard
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import inference_snapshot, sha256
from select_flash_q4_0910c import Manager

BASE = Path(__file__).resolve().parent
OUT = BASE / 'results/deepseek-v41-cpu-graph-0910'


def main():
    assert os.sched_getaffinity(0) == {127} and not OUT.exists()
    manager = Manager(); peer = manager.validate_current()
    guard = ModelMeasurementGuard(peer['pid'], {peer['pid']:18131}, inference_snapshot)
    setup = json.loads((BASE/'results/deepseek-v41-hash-reference-setup-0910/result.json').read_text())
    assert setup['passed']
    checker = BASE/'check_deepseek_v41_cpu_graph_0910.py'
    source = BASE/'deepseek_v41_cpu_reference_0910.py'
    result = dict(passed=False,started=time.time(),controller_pid=os.getpid(),
        selected_flash_pid=peer['pid'],selected_flash_start=peer['info']['start'],
        input_sha256={str(p):sha256(p) for p in [Path(__file__),checker,source,BASE/'model_measurement_guard.py',BASE/'select_flash_q4_0910c.py']},
        full_checkpoint_loaded=False,released_model_generated_text=False)
    owned = None
    def cancel(*_): raise InterruptedError('Stop only the owned reduced-weight CPU graph fixture')
    for sig in [signal.SIGINT,signal.SIGTERM,signal.SIGHUP]: signal.signal(sig,cancel)
    with (BASE/'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        guard.assert_idle();OUT.mkdir()
        def save():atomic_json(OUT/'result.json',result)
        save()
        try:
            command=['taskset','-c','48',str(Path(setup['venv'])/'bin/python'),str(checker),str(OUT)]
            with (OUT/'check.log').open('w') as log:
                owned=subprocess.Popen(command,cwd=BASE,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                    start_new_session=True,env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false'))
                result['owned_pid']=owned.pid;result['command']=command;save()
                deadline,report=time.monotonic()+600,0
                while owned.poll() is None:
                    guard.assert_idle();assert time.monotonic()<deadline
                    if time.monotonic()-report>=30:
                        print(json.dumps(dict(running='cpu-graph-fixture',pid=owned.pid)),flush=True);report=time.monotonic()
                    time.sleep(.25)
            result['exit_code']=owned.returncode
            assert owned.returncode==0,owned.returncode
            proof=json.loads((OUT/'graph-check.json').read_text());assert proof['passed']
            assert all(sha256(p)==h for p,h in result['input_sha256'].items())
            current=manager.validate_current();assert current['pid']==peer['pid'] and current['info']['start']==peer['info']['start']
            result.update(passed=True,selected_flash_preserved=True,graph_check_sha256=sha256(OUT/'graph-check.json'))
        except BaseException as error:
            result['error']=repr(error);raise
        finally:
            if owned is not None and owned.poll() is None:
                os.killpg(owned.pid,signal.SIGTERM)
                try:owned.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(owned.pid,signal.SIGKILL);owned.wait(timeout=10)
            result['finished']=time.time();save()
    print(json.dumps(dict(passed=result['passed'],released_model_generated_text=False)),flush=True)


if __name__=='__main__':
    os.umask(0o077)
    main()
