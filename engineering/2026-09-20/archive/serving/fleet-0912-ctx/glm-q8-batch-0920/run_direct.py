#!/usr/bin/env python3
"""Guarded one-core exact-output and microtiming probe; never restart production."""
import hashlib,json,os,resource,signal,subprocess,sys,time
from pathlib import Path
B=Path('/home/user/sr950-strategy/codex-bench-0919');sys.path.insert(0,str(B))
from capture_runtime import snapshot
from pool_window import pid_for
from thread_window import idle
HERE=Path(__file__).resolve().parent
ENGINE=Path('/home/user/InfoSystemic/AI-Server/engines/llama.cpp-glm5n-goal-0904')
FLEET=Path('/home/user/InfoSystemic/AI-Server/serving/fleet-0912-ctx')
CPU=FLEET/'glm-kpool-wide-0919/deploy/libggml-cpu.so.0.22.0'
BASE=FLEET/'glm-fix/libggml-base.so.0.22.0'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')

def main():
    report=HERE/'direct-report.json';assert not report.exists()
    pid=pid_for('glm53-flash-production.service');before=snapshot(pid,18131);idle(18131)
    expected=json.loads((B/'mtp-kv-only-final-audit.json').read_text())
    for k in ['proc_start_ticks','command','environment','mapped_libraries_sha256']:assert before[k]==expected[k],k
    assert before['mapped_libraries_sha256'][str(CPU)]==sha(CPU)
    assert before['mapped_libraries_sha256'][str(BASE)]==sha(BASE)
    save(HERE/'runtime-before.json',before)
    inputs=[HERE/'run_direct.py',HERE/'test_q8_batch.cpp',HERE/'q8-batch-candidates.h',HERE/'production-q8-batch.h',CPU,BASE]
    r=dict(started=time.time(),production_pid=pid,passed=False,steps=[],input_sha256={str(p):sha(p) for p in inputs},scope='One-core direct kernel parity and timing; not graph or model throughput')
    def checkpoint():save(report,r)
    def guard():assert pid_for('glm53-flash-production.service')==pid;idle(18131)
    def limit():
        os.nice(15)
        resource.setrlimit(resource.RLIMIT_AS,(4<<30,4<<30))
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    def run(cmd,label,env):
        guard();start=time.monotonic();log=HERE/(label+'.log')
        with log.open('w') as stream:
            proc=subprocess.Popen(cmd,cwd=HERE,stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True,preexec_fn=limit)
            try:
                while proc.poll() is None:
                    guard();assert time.monotonic()-start<600,label
                    time.sleep(.5)
                assert proc.returncode==0,(label,proc.returncode)
            finally:
                if proc.poll() is None:os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=10)
        r['steps'].append(dict(label=label,command=cmd,seconds=time.monotonic()-start));checkpoint()
        print('completed',label,flush=True);return log
    checkpoint()
    try:
        env={k:v for k,v in os.environ.items() if not k.startswith(('GGML_','LLAMA_','OMP_','GOMP_'))}
        env.update(LD_LIBRARY_PATH=str(CPU.parent)+':'+str(BASE.parent),GGML_CPU_Q8_FAST_SUM='1',OMP_NUM_THREADS='1')
        binary=HERE/'test_q8_batch'
        command=['taskset','-c','48','/usr/bin/c++','-O3','-std=c++17','-march=native','-fno-fast-math','-I'+str(HERE)]
        command += ['-I'+str(ENGINE/p) for p in ['include','ggml/include','ggml/src','ggml/src/ggml-cpu']]
        command += [str(HERE/'test_q8_batch.cpp'),str(CPU),str(BASE),'-ldl','-pthread','-o',str(binary)]
        run(command,'compile',env);r['binary_sha256']=sha(binary);checkpoint()
        log=run(['taskset','-c','48',str(binary),'gate',str(HERE/'outputs.bin')],'gate',env)
        events=[json.loads(x) for x in log.read_text().splitlines() if x.startswith('{')]
        assert Path(events[0]['path']).resolve()==CPU.resolve();assert events[-1]==dict(event='done',passed=True)
        check=next(x for x in events if x['event']=='correctness');assert check['sum_cases']==18448 and check['matrix_cases']==1296 and check['passed']
        r['correctness']=check;r['output_bytes']=(HERE/'outputs.bin').stat().st_size;r['output_sha256']=sha(HERE/'outputs.bin')
        assert r['output_bytes']==check['compared_values']*4;checkpoint()
        log=run(['taskset','-c','48',str(binary),'bench','unused'],'bench',env)
        events=[json.loads(x) for x in log.read_text().splitlines() if x.startswith('{')]
        assert Path(events[0]['path']).resolve()==CPU.resolve();assert events[-1]==dict(event='done',passed=True)
        rows=[x for x in events if x['event']=='timing'];assert len(rows)==336 and all(x['samples']==10 for x in rows)
        r['timings']=rows
        for p,d in r['input_sha256'].items():assert sha(p)==d,p
        after=snapshot(pid,18131);guard()
        for k in ['proc_start_ticks','command','environment','mapped_libraries_sha256']:assert after[k]==before[k],k
        save(HERE/'runtime-after.json',after);r['production_unchanged']=True;r['passed']=True
        print('Direct parity and timing completed; production unchanged',flush=True)
    except BaseException as e:r['error']=repr(e);raise
    finally:r['finished']=time.time();checkpoint()

if __name__=='__main__':main()
