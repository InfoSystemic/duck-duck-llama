#!/usr/bin/env python3
"""Repeat HC timing with matched workspace capacity before loading a model."""
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time

from benchmark_qwen_q6 import wait_background
from model_measurement_guard import ModelMeasurementGuard
from qwen_split_trial import inference_snapshot, process_info, runtime_environment, sha256

BASE = Path(__file__).resolve().parent
OUT = BASE/'results/qwen-decode-scheduling-reserved-recheck-0909'


def main():
    assert os.sched_getaffinity(0)=={127}
    assert process_info(1219506)['start']=='103969952'
    def interrupted(signum,frame):
        raise InterruptedError('Release only the owned HC retiming fixture')
    for signum in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):
        signal.signal(signum,interrupted)
    build_path = BASE/'results/qwen-decode-scheduling-build-0909/result.json'
    runtime_path = BASE/'results/qwen-get-rows-runtime-0909/result.json'
    build,runtime = [json.loads(path.read_text()) for path in (build_path,runtime_path)]
    assert build['passed'] and build['finished'] and build['bit_exact'] and build['expert_bit_exact']
    assert sha256(build['library'])==build['library_sha256']
    for key in ('input_sha256','private_source_sha256'):
        assert all(sha256(path)==digest for path,digest in build[key].items())
    fixture_path = BASE/'results/qwen-hc-reserved-fixture-0909/result.json'
    fixture = json.loads(fixture_path.read_text())
    assert fixture['passed'] and fixture['finished']
    assert all(sha256(path)==digest for path,digest in fixture['input_sha256'].items())
    assert sha256(fixture['source'])==fixture['source_sha256']
    binary = Path(fixture['binary'])
    assert sha256(binary)==fixture['binary_sha256']
    guard = ModelMeasurementGuard(1219506,{1219506:18095},inference_snapshot)
    inputs = [Path(__file__).resolve(),build_path,runtime_path,fixture_path,Path(fixture['source']),binary,Path(build['library']),Path(runtime['cpu']),
              BASE/'benchmark_qwen_q6.py',BASE/'model_measurement_guard.py',BASE/'qwen_split_trial.py']
    with (BASE/'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        guard.assert_idle()
        OUT.mkdir(exist_ok=False)
        result = dict(started=time.time(),passed=False,model_loaded=False,arms=[],controller_pid=os.getpid(),
                      input_sha256={str(path):sha256(path) for path in inputs},
                      scope='The model libraries are unchanged. A matched 64-token graph first reserves equal backend workspace. On each of two sockets run parent/private-off/private-on/private-on/private-off/parent. Retain all original timing evidence. No model speed or memory-bandwidth claim.')

        def save():
            (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')

        save()
        try:
            result['idle_gate'] = guard.wait_idle(OUT/'waiting-for-idle.json')
            clean = {key:value for key,value in os.environ.items() if key not in runtime_environment(os.environ) and key!='LD_PRELOAD'}
            clean.update(runtime['runtime_env'])
            for socket,mask in [(0,'0-14'),(1,'16-30')]:
                for index,mode in enumerate(('parent','off','on','on','off','parent')):
                    name = f'socket{socket}-{index}-{mode}'
                    background = wait_background(guard,1219506,4,OUT/'background-wait.json')
                    cpu = Path(runtime['cpu'] if mode=='parent' else build['library'])
                    env = dict(clean,LD_LIBRARY_PATH=str(cpu.parent)+':'+runtime['runtime_directory'],
                               GGML_CPU_QWEN_HC_ORDERED_K='1' if mode=='on' else '0',
                               GGML_CPU_QWEN_HC_ROW_SPLIT='1' if mode=='on' else '0',
                               GGML_CPU_QWEN_HC_ORDERED_K_AUDIT='0',GGML_CPU_QWEN_Q6_MOE_TILE_ROWS='64')
                    command = ['taskset','-c',mask,str(binary)]
                    with (OUT/(name+'.log')).open('w') as log:
                        owned = subprocess.Popen(command,cwd=BASE,env=env,stdout=log,stderr=subprocess.STDOUT)
                        try:
                            deadline = time.monotonic()+120
                            while owned.poll() is None:
                                guard.assert_idle()
                                assert time.monotonic()<deadline
                                time.sleep(.25)
                            assert owned.returncode==0
                        finally:
                            if owned.poll() is None:
                                owned.terminate()
                                owned.wait(timeout=15)
                    log = (OUT/(name+'.log')).read_text()
                    paths = [line.removeprefix('CPU_LIBRARY ') for line in log.splitlines() if line.startswith('CPU_LIBRARY ')]
                    assert len(paths)==1 and Path(paths[0]).resolve()==cpu.resolve()
                    rows = [json.loads(line.removeprefix('HC_TIME ')) for line in log.splitlines() if line.startswith('HC_TIME ')]
                    assert len(rows)==13 and all(row['samples']==40 for row in rows)
                    reserve = rows.pop(0)
                    assert (reserve['nc'],reserve['nr'],reserve['matrices'])==(8,64,1)
                    assert reserve['planned_work_bytes']==reserve['workspace_max_bytes']>650000
                    assert all(row['workspace_max_bytes']==reserve['planned_work_bytes'] and row['planned_work_bytes']<=reserve['planned_work_bytes'] for row in rows)
                    result['arms'].append(dict(name=name,socket=socket,mode=mode,cpu_sha256=sha256(cpu),command=command,
                                               background_gate=background,reserve=reserve,rows=rows))
                    save()
                    print(json.dumps(dict(completed=name)),flush=True)
            assert len({arm["reserve"]["planned_work_bytes"] for arm in result["arms"]})==1
            assert len({arm["reserve"]["hash"] for arm in result["arms"]})==1
            comparisons = []
            for socket in (0,1):
                arms = [arm for arm in result['arms'] if arm['socket']==socket]
                for row in arms[0]['rows']:
                    key = {name:row[name] for name in ('nc','nr','matrices','weight_bytes')}
                    matching = [(arm['mode'],other) for arm in arms for other in arm['rows'] if all(other[k]==v for k,v in key.items())]
                    assert len(matching)==6 and len({other['hash'] for _,other in matching})==1
                    means = {mode:statistics.mean(other['median_us'] for name,other in matching if name==mode) for mode in ('parent','off','on')}
                    comparisons.append(dict(socket=socket,**key,means_us=means,speedup=means['parent']/means['on'],
                                            private_enabled_speedup=means['off']/means['on']))
            aggregates = []
            for nc in (64,96):
                for nr in (1,4,5):
                    rows = [row for row in comparisons if row['nc']==nc and row['nr']==nr and row['matrices']==96]
                    assert len(rows)==2
                    ratio = math.exp(statistics.mean(math.log(row['speedup']) for row in rows))
                    aggregates.append(dict(nc=nc,nr=nr,geomean_speedup=ratio))
            eligible = all(row['geomean_speedup']>=1/1.05 for row in aggregates) and any(row['nc']==64 and row['geomean_speedup']>=1.1 for row in aggregates)
            assert all(sha256(path)==digest for path,digest in result['input_sha256'].items())
            guard.assert_idle()
            result.update(passed=True,comparisons=comparisons,aggregates=aggregates,model_test_eligible=eligible,
                          peer_preserved=process_info(1219506)['start']=='103969952')
            print(json.dumps(dict(passed=True,aggregates=aggregates,model_test_eligible=eligible)),flush=True)
        except BaseException as error:
            result['error'] = repr(error)
            raise
        finally:
            result['finished'] = time.time()
            save()


if __name__ == '__main__':
    os.umask(0o077)
    main()
