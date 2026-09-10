#!/usr/bin/env python3
"""Guard one real native-checkpoint text run and a repeated-cache check."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import inference_snapshot,sha256
from select_flash_q4_0910c import Manager
from model_measurement_guard import ModelMeasurementGuard
from glm_flash_q8_trial import memory_status

BASE=Path(__file__).resolve().parent
OUT=BASE/'results/deepseek-v41-checkpoint-run-0910b'
CACHE=Path('/dev/shm/deepseek-v41-native-fb2764-0910')


def main():
    assert os.sched_getaffinity(0)=={127} and not OUT.exists()
    manager=Manager();peer=manager.validate_current()
    guard=ModelMeasurementGuard(peer['pid'],{peer['pid']:18131},inference_snapshot)
    sources=[Path(__file__),BASE/'run_deepseek_v41_checkpoint_0910b.py',BASE/'deepseek_v41_checkpoint_0910.py',BASE/'run_deepseek_v41_checkpoint_0910.py',BASE/'deepseek_v41_checkpoint_transport_0910b.py',
        BASE/'deepseek_v41_native_bridge_0910.py',BASE/'deepseek_v41_cpu_reference_0910.py',
        BASE/'results/deepseek-v41-native-gemm-0910/libdeepseek-v41-native-gemm.so',
        BASE/'results/deepseek-v41-intake-0910/official/inference/model.py',BASE/'results/deepseek-v41-intake-0910/official/inference/config.json',
        BASE/'results/deepseek-v41-intake-0910/official/inference/engram.py',BASE/'results/deepseek-v41-cpu-source-0910/encoding/encoding.py']
    result=dict(passed=False,started=time.time(),controller_pid=os.getpid(),selected_flash_pid=peer['pid'],selected_flash_start=peer['info']['start'],
        input_sha256={str(p):sha256(p) for p in sources},memory_before=memory_status(),owned_cache=str(CACHE),cache_limit_bytes=90<<30)
    owned=None
    def cancel(*_):raise InterruptedError('Stop only the owned DeepSeek checkpoint run')
    for s in [signal.SIGINT,signal.SIGTERM,signal.SIGHUP]:signal.signal(s,cancel)
    with (BASE/'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guard.assert_idle();OUT.mkdir()
        atomic_json(OUT/'result.json',result)
        try:
            assert memory_status()['MemAvailable']>200<<30
            if not CACHE.exists():
                CACHE.mkdir(mode=0o700)
                atomic_json(CACHE/'owner.json',dict(task='deepseek-v41-native-0910',revision='fb2764a5cf321eaa5070ca8f9e892818f477c16d'))
            assert not CACHE.is_symlink() and CACHE.stat().st_uid==os.getuid()
            for phase in ['inspect','run']:
                command=['taskset','-c','48-63',str(BASE.parents[1]/'tools/deepseek-v41-cpu-reference-0910/venv/bin/python'),
                    str(sources[1]),'--output',str(OUT),'--cache',str(CACHE)]
                if phase=='inspect':command.append('--inspect')
                with (OUT/(phase+'.log')).open('w') as log:
                    owned=subprocess.Popen(command,cwd=BASE,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,
                        env=dict(os.environ,OMP_NUM_THREADS='16',MKL_NUM_THREADS='16',OMP_WAIT_POLICY='PASSIVE',TOKENIZERS_PARALLELISM='false'))
                    result['owned_pid']=owned.pid;result['phase']=phase;atomic_json(OUT/'result.json',result)
                    deadline,report=time.monotonic()+(300 if phase=='inspect' else 10800),0
                    while owned.poll() is None:
                        guard.assert_idle();memory=memory_status();assert memory['MemAvailable']>80<<30,('RAM reserve reached',memory)
                        assert time.monotonic()<deadline
                        if time.monotonic()-report>=30:
                            print(json.dumps(dict(phase=phase,pid=owned.pid,memory_available=memory['MemAvailable'],log_bytes=(OUT/(phase+'.log')).stat().st_size)),flush=True)
                            report=time.monotonic()
                        time.sleep(.5)
                result[phase+'_exit_code']=owned.returncode
                assert owned.returncode==0,(phase,owned.returncode)
                if phase=='inspect':assert json.loads((OUT/'shape-check.json').read_text())['passed']
            proof=json.loads((OUT/'generation.json').read_text());assert proof['passed'] and proof['released_model_generated_text']
            assert all(sha256(p)==h for p,h in result['input_sha256'].items())
            current=manager.validate_current();assert current['pid']==peer['pid'] and current['info']['start']==peer['info']['start']
            result.update(passed=True,peer_preserved=True,released_model_generated_text=True,exact_repeated_logits=proof['exact_repeated_logits'])
        except BaseException as e:result['error']=repr(e);raise
        finally:
            if owned and owned.poll() is None:
                os.killpg(owned.pid,signal.SIGTERM)
                try:owned.wait(30)
                except subprocess.TimeoutExpired:os.killpg(owned.pid,signal.SIGKILL);owned.wait(30)
            result.update(finished=time.time(),memory_after=memory_status());atomic_json(OUT/'result.json',result)
            print(json.dumps({k:v for k,v in result.items() if k!='input_sha256'}),flush=True)


if __name__=='__main__':main()
