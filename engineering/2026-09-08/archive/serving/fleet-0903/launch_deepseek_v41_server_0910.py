#!/usr/bin/env python3
"""Validate and retain one localhost native DeepSeek text server."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request
from qwen_high_quant_trial import atomic_json,port_available
from qwen_split_trial import inference_snapshot,process_info,sha256
from select_flash_q4_0910c import Manager
from model_measurement_guard import ModelMeasurementGuard
from glm_flash_q8_trial import memory_status

BASE=Path(__file__).resolve().parent
OUT=BASE/'results/deepseek-v41-server-0910'
CACHE=Path('/dev/shm/deepseek-v41-native-fb2764-0910')
PORT=18170


def main():
    assert os.sched_getaffinity(0)=={127} and not OUT.exists()
    baseline_path=BASE/'results/deepseek-v41-checkpoint-run-0910b/generation.json'
    baseline=json.loads(baseline_path.read_text())
    assert baseline['passed'] and baseline['exact_repeated_logits'] and baseline['released_model_generated_text']
    assert baseline['runs'][0]['token_ids'][:2]==[19923,3]
    manager=Manager();peer=manager.validate_current()
    guard=ModelMeasurementGuard(peer['pid'],{peer['pid']:18131},inference_snapshot)
    names=['launch_deepseek_v41_server_0910.py','deepseek_v41_server_0910.py','deepseek_v41_serving_store_0910.py',
        'check_deepseek_v41_serving_store_0910.py','deepseek_v41_checkpoint_transport_0910b.py',
        'run_deepseek_v41_checkpoint_0910b.py','run_deepseek_v41_checkpoint_0910.py','deepseek_v41_checkpoint_0910.py',
        'deepseek_v41_cpu_reference_0910.py','deepseek_v41_native_bridge_0910.py',
        'results/deepseek-v41-native-gemm-0910b/libdeepseek-v41-native-gemm.so',
        'results/deepseek-v41-intake-0910/official/inference/model.py','results/deepseek-v41-intake-0910/official/inference/config.json',
        'results/deepseek-v41-intake-0910/official/inference/engram.py','results/deepseek-v41-cpu-source-0910/encoding/encoding.py']
    result=dict(passed=False,started=time.time(),controller_pid=os.getpid(),selected_flash_pid=peer['pid'],
        selected_flash_start=peer['info']['start'],source_sha256={str(BASE/n):sha256(BASE/n) for n in names},
        baseline_sha256=sha256(baseline_path),memory_before=memory_status())
    owned=None;keep=False
    def cancel(*_):raise InterruptedError('Stop only the owned DeepSeek server startup')
    for s in [signal.SIGINT,signal.SIGTERM,signal.SIGHUP]:signal.signal(s,cancel)
    with (BASE/'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guard.assert_idle();assert port_available(PORT)
        OUT.mkdir();atomic_json(OUT/'result.json',result)
        python=str(BASE.parents[1]/'tools/deepseek-v41-cpu-reference-0910/venv/bin/python')
        environment=dict(os.environ,OMP_NUM_THREADS='16',MKL_NUM_THREADS='16',OMP_WAIT_POLICY='PASSIVE',TOKENIZERS_PARALLELISM='false')
        try:
            command=['taskset','-c','48',python,str(BASE/names[3]),str(OUT/'cache-check.json')]
            with (OUT/'cache-check.log').open('w') as log:
                checked=subprocess.run(command,cwd=BASE,env=environment,stdout=log,stderr=subprocess.STDOUT,timeout=120)
            assert checked.returncode==0 and json.loads((OUT/'cache-check.json').read_text())['passed']
            command=['taskset','-c','48-63',python,str(BASE/names[1]),'--cache',str(CACHE),'--output',str(OUT),
                '--port',str(PORT),'--lifecycle-lock-fd',str(lock.fileno())]
            with (OUT/'server.log').open('w') as log:
                owned=subprocess.Popen(command,cwd=BASE,env=environment,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                    start_new_session=True,pass_fds=(lock.fileno(),))
            result['owned_pid']=owned.pid;atomic_json(OUT/'result.json',result)
            deadline,report=time.monotonic()+600,0
            while not (OUT/'ready.json').exists():
                assert owned.poll() is None and time.monotonic()<deadline
                guard.assert_idle();assert memory_status()['MemAvailable']>100<<30
                if time.monotonic()-report>=30:
                    print(json.dumps(dict(loading_deepseek_server=owned.pid)),flush=True);report=time.monotonic()
                time.sleep(.5)
            ready=json.loads((OUT/'ready.json').read_text());assert ready['pid']==owned.pid and ready['port']==PORT
            result['server']=process_info(owned.pid);result['ready']=ready
            for endpoint in ['health','v1/models']:
                with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/{endpoint}',timeout=5) as response:result[endpoint]=json.load(response)
            assert result['health']['status']=='ok' and result['v1/models']['data'][0]['id']=='DeepSeek-V4.1-Flash'
            # Validate both API formats with the same real checkpoint prefix already generated.
            for streaming in [False,True]:
                guard.assert_idle()
                payload=dict(model='DeepSeek-V4.1-Flash',messages=[dict(role='user',content='Hi.')],temperature=0,max_tokens=2,stream=streaming)
                request=urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
                print(json.dumps(dict(validating_deepseek_api=True,stream=streaming,pid=owned.pid)),flush=True)
                start=time.monotonic()
                with urllib.request.urlopen(request,timeout=600) as response:
                    assert response.status==200
                    if streaming:
                        chunks=[];done=False
                        for raw in response:
                            if not raw.startswith(b'data: '):continue
                            data=raw[6:].strip()
                            if data==b'[DONE]':done=True;break
                            item=json.loads(data);assert 'error' not in item;chunks.append(item)
                        text=''.join(c['choices'][0]['delta'].get('content','') for c in chunks)
                        assert done and chunks[-1]['choices'][0]['finish_reason']=='length'
                        result['stream_check']=dict(passed=text=='Hello!',text=text,seconds=time.monotonic()-start,chunks=len(chunks))
                        last=json.loads((OUT/'last-request.json').read_text());assert last['timings']['downloaded_bytes']==0
                        result['cached_api_timings']=last
                    else:
                        data=json.load(response);text=data['choices'][0]['message']['content']
                        result['json_check']=dict(passed=text=='Hello!',text=text,seconds=time.monotonic()-start,usage=data['usage'],timings=data['timings'])
                    assert text=='Hello!',text
            guard.assert_idle();current=manager.validate_current()
            assert current['pid']==peer['pid'] and current['info']['start']==peer['info']['start']
            assert all(sha256(p)==h for p,h in result['source_sha256'].items()) and owned.poll() is None
            result.update(passed=True,peer_preserved=True,server_retained=True,memory_after=memory_status())
            atomic_json(BASE/'deepseek-v41-selected.json',dict(model='DeepSeek-V4.1-Flash',pid=owned.pid,start=result['server']['start'],
                endpoint=f'http://127.0.0.1:{PORT}/v1',evidence=str(OUT/'result.json'),revision=ready['revision'],
                native_precision=True,context=256,vision=False,dspark=False,source_sha256=result['source_sha256'],
                temporary_ram_cache=str(CACHE),lifecycle_lock_owned_by_server=True))
            keep=True
        except BaseException as e:result['error']=repr(e);raise
        finally:
            if owned and owned.poll() is None and not keep:
                os.killpg(owned.pid,signal.SIGTERM)
                try:owned.wait(30)
                except subprocess.TimeoutExpired:os.killpg(owned.pid,signal.SIGKILL);owned.wait(30)
            result['finished']=time.time();atomic_json(OUT/'result.json',result)
            print(json.dumps({k:result[k] for k in ['passed','owned_pid','server_retained','error'] if k in result}),flush=True)


if __name__=='__main__':main()
