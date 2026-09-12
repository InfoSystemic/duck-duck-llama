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
OUT=BASE/'results/deepseek-v41-native-gemm-0910d'


def measure():
    assert os.sched_getaffinity(0)=={127} and not OUT.exists()
    assert 'avx512_vnni' in Path('/proc/cpuinfo').read_text().split('\n\n',1)[0]
    peer=Manager().validate_current()
    guard=ModelMeasurementGuard(peer['pid'],{peer['pid']:18131},inference_snapshot)
    inputs=[Path(__file__),BASE/'deepseek-v41-native-gemm-0910d.cpp',BASE/'deepseek_v41_native_bridge_0910.py',BASE/'check_deepseek_v41_native_gemm_0910d.py',BASE/'deepseek_v41_cpu_reference_0910.py',BASE/'check_deepseek_v41_native_gemm_0910.py']
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
    if True:
        check_idle();OUT.mkdir()
        try:
            commands=[['taskset','-c','48','g++','-std=c++17','-O3','-shared','-fPIC','-fopenmp','-mavx512f','-mavx512bw','-mavx512vl','-mavx512dq','-mavx512vnni','-ffp-contract=off',str(inputs[1]),'-o',str(OUT/'libdeepseek-v41-native-gemm.so')],
                ['taskset','-c','48-63',str(BASE.parents[1]/'tools/deepseek-v41-cpu-reference-0910/venv/bin/python'),str(inputs[3]),str(OUT)]]
            for i,cmd in enumerate(commands):
                with (OUT/f'step-{i}.log').open('w') as log:
                    owned=subprocess.Popen(cmd,cwd=BASE,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,
                        env=dict(os.environ,OMP_NUM_THREADS='16',MKL_NUM_THREADS='1',OMP_WAIT_POLICY='PASSIVE',TOKENIZERS_PARALLELISM='false'))
                    deadline,report=time.monotonic()+900,0
                    while owned.poll() is None:
                        check_idle();assert time.monotonic()<deadline
                        if time.monotonic()-report>30:
                            print(json.dumps(dict(step=i,pid=owned.pid)),flush=True);report=time.monotonic()
                        time.sleep(.25)
                assert owned.returncode==0,(i,owned.returncode)
            proof=json.loads((OUT/'kernel-check.json').read_text());assert proof['passed']
            assert all(sha256(p)==h for p,h in result['input_sha256'].items())
            current=Manager().validate_current();assert current['pid']==peer['pid'] and current['info']['start']==peer['info']['start']
            tile=json.loads((OUT/'integer-check.json').read_text());assert tile['passed'];check_idle()
            result.update(passed=True,peer_preserved=True,deepseek_preserved=True,exact_values=proof['exact_values'],integer_exact_values=tile['exact_values'])
        except BaseException as e:result['error']=repr(e);raise
        finally:
            if owned and owned.poll() is None:
                os.killpg(owned.pid,signal.SIGTERM)
                try:owned.wait(30)
                except subprocess.TimeoutExpired:os.killpg(owned.pid,signal.SIGKILL);owned.wait(30)
            result['finished']=time.time();atomic_json(OUT/'result.json',result);print(json.dumps(result),flush=True)


def main():
    def cancel(*_):raise InterruptedError('Cancel the waiting component controller')
    for sig in [signal.SIGINT,signal.SIGTERM,signal.SIGHUP]:signal.signal(sig,cancel)
    with (BASE/'results/deepseek-v41-controller.lock').open('a') as lock:
        deadline=time.monotonic()+900;report=0
        while True:
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                assert time.monotonic()<deadline,'Existing DeepSeek controller is still active'
                if time.monotonic()-report>=30:
                    print(json.dumps(dict(waiting_for_deepseek_controller=True)),flush=True);report=time.monotonic()
                time.sleep(.25)
        measure()

if __name__=='__main__':main()
