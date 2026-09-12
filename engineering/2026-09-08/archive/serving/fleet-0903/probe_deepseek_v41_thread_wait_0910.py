#!/usr/bin/env python3
"""Build and validate a bounded DeepSeek CPU kernel/graph fixture."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import inference_snapshot,sha256,process_info
from select_flash_q4_0910c import Manager
from model_measurement_guard import ModelMeasurementGuard

BASE=Path(__file__).resolve().parent
OUT=BASE/'results/deepseek-v41-thread-wait-0910'


def main():
    assert os.sched_getaffinity(0)=={127} and not OUT.exists()
    peer=Manager().validate_current()
    guard=ModelMeasurementGuard(peer['pid'],{peer['pid']:18131},inference_snapshot)
    inputs=[Path(__file__),BASE/'check_deepseek_v41_thread_wait_0910.py',BASE/'deepseek_v41_native_bridge_0910.py',BASE/'deepseek_v41_cpu_reference_0910.py',BASE/'results/deepseek-v41-native-gemm-0910b/libdeepseek-v41-native-gemm.so']
    selected=json.loads((BASE/'deepseek-v41-selected.json').read_text())
    assert process_info(selected['pid'])['start']==selected['start']
    assert all(sha256(p)==h for p,h in selected['source_sha256'].items())
    request_record=Path(selected['request_record'])
    request_count=json.loads(request_record.read_text())['completed']
    def check_idle():
        guard.assert_idle()
        assert process_info(selected['pid'])['start']==selected['start']
        with urllib.request.urlopen('http://127.0.0.1:18170/health',timeout=2) as response:
            assert not json.load(response)['busy']
        assert json.loads(request_record.read_text())['completed']==request_count
    result=dict(passed=False,started=time.time(),selected_flash_pid=peer['pid'],
        deepseek_pid=selected['pid'],deepseek_start=selected['start'],input_sha256={str(p):sha256(p) for p in inputs})
    owned=None
    def cancel(*_):raise InterruptedError('Cancel owned DeepSeek CPU kernel fixture')
    for s in [signal.SIGINT,signal.SIGTERM,signal.SIGHUP]:signal.signal(s,cancel)
    with (BASE/'results/deepseek-v41-controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);check_idle();OUT.mkdir()
        try:
            spins=[0,1000,10000,100000,10000,1000,0]
            python=str(BASE.parents[1]/'tools/deepseek-v41-cpu-reference-0910/venv/bin/python')
            commands=[['taskset','-c','48-63',python,str(inputs[1]),str(OUT/f'arm-{i}.json')] for i in range(len(spins))]
            for i,cmd in enumerate(commands):
                with (OUT/f'step-{i}.log').open('w') as log:
                    owned=subprocess.Popen(cmd,cwd=BASE,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,
                        env=dict(os.environ,OMP_NUM_THREADS='16',MKL_NUM_THREADS='16',OMP_WAIT_POLICY='PASSIVE',GOMP_SPINCOUNT=str(spins[i]),TOKENIZERS_PARALLELISM='false'))
                    deadline,report=time.monotonic()+900,0
                    while owned.poll() is None:
                        check_idle();assert time.monotonic()<deadline
                        if time.monotonic()-report>30:
                            print(json.dumps(dict(step=i,pid=owned.pid)),flush=True);report=time.monotonic()
                        time.sleep(.25)
                assert owned.returncode==0,(i,owned.returncode)
            arms=[json.loads((OUT/f'arm-{i}.json').read_text()) for i in range(len(spins))]
            assert all(a['passed'] for a in arms)
            assert all([r['output_sha256'] for r in a['measurements']]==[r['output_sha256'] for r in arms[0]['measurements']] for a in arms)
            assert all(sha256(p)==h for p,h in result['input_sha256'].items())
            current=Manager().validate_current();assert current['pid']==peer['pid'] and current['info']['start']==peer['info']['start']
            check_idle()
            result.update(passed=True,peer_preserved=True,deepseek_preserved=True,arms=arms,component_only=True)
        except BaseException as e:result['error']=repr(e);raise
        finally:
            if owned and owned.poll() is None:
                os.killpg(owned.pid,signal.SIGTERM)
                try:owned.wait(30)
                except subprocess.TimeoutExpired:os.killpg(owned.pid,signal.SIGKILL);owned.wait(30)
            result['finished']=time.time();atomic_json(OUT/'result.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
