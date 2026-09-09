#!/usr/bin/env python3
"""Check a guarded SIMD RMS mean before changing the model library."""
import fcntl,json,os
from pathlib import Path
import subprocess,time
from glm_flash_q8_trial import BASE,Manager,PORT
from model_measurement_guard import ModelMeasurementGuard
from qwen_split_trial import inference_snapshot,sha256

OUT=BASE/'results/glm-flash-rms-guard-probe-0908'

def main():
    os.umask(0o077)
    m=Manager();current=m.validate_current()
    guard=ModelMeasurementGuard(current['pid'],{current['pid']:PORT},inference_snapshot);guard.assert_idle()
    assert current['sum16'] and current['drafts']==0
    assert sha256(current['cpu_library'])==current['cpu_sha256']
    sources=[BASE/'flash-rms-guarded-0908.h',BASE/'flash-rms-guard-check-0908.cpp',Path(__file__)]
    OUT.mkdir(exist_ok=False)
    for p in sources:(OUT/p.name).write_bytes(p.read_bytes())
    inputs={str(p):sha256(p) for p in sources}
    result=dict(started=time.time(),passed=False,current_pid=current['pid'],cpu_sha256=current['cpu_sha256'],input_sha256=inputs,steps=[])
    def save():(OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    def run(command,label):
        m.validate_current();guard.assert_idle()
        log=OUT/(label+'.log')
        with log.open('w') as stream:
            process=subprocess.Popen(command,cwd=BASE,stdout=stream,stderr=subprocess.STDOUT)
            try:
                deadline=time.monotonic()+300
                while process.poll() is None:
                    guard.assert_idle();assert time.monotonic()<deadline;time.sleep(.5)
                assert process.returncode==0,(label,process.returncode)
            finally:
                if process.poll() is None:process.terminate();process.wait(timeout=10)
        result['steps'].append(dict(label=label,command=command));save();print(json.dumps(dict(completed=label)),flush=True)
        return log.read_text()
    save()
    try:
        binary=OUT/'rms-guard-check'
        command=['c++','-O3','-std=c++17','-march=native','-I'+str(BASE),str(BASE/'flash-rms-guard-check-0908.cpp'),'-o',str(binary)]
        run(command,'compile')
        log=run(['taskset','-c','48',str(binary)],'probe')
        events=[json.loads(line) for line in log.splitlines() if line.startswith('{')]
        result['events']=events
        assert events[0]['event']=='correctness' and events[0]['passed']
        assert events[-1]['event']=='done' and events[-1]['passed']
        timings=[x for x in events if x['event']=='timing'];assert len(timings)==8
        result['binary_sha256']=sha256(binary)
        result['timing_change_percent']={str(n):100*(next(x['median_us'] for x in timings if x['n']==n and x['mode']==1)/next(x['median_us'] for x in timings if x['n']==n and x['mode']==0)-1) for n in (1024,4096,16384,65536)}
        assert all(sha256(p)==digest for p,digest in inputs.items())
        m.validate_current();guard.assert_idle()
        result['passed']=True
        print(json.dumps(dict(passed=True,correctness=events[0],timing_change_percent=result['timing_change_percent'])),flush=True)
    except BaseException as error:
        result['error']=repr(error);raise
    finally:
        result['finished']=time.time();save()

if __name__=='__main__':
    with (BASE/'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main()
