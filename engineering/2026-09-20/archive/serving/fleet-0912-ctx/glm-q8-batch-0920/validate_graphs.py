#!/usr/bin/env python3
"""Guarded compiled-library direct and graph gates; no model outage."""
import hashlib,json,os,resource,signal,struct,subprocess,sys,time
from pathlib import Path
B=Path('/home/user/sr950-strategy/codex-bench-0919');sys.path.insert(0,str(B))
from capture_runtime import snapshot
from pool_window import pid_for
from thread_window import idle
HERE=Path(__file__).resolve().parent
ENGINE=Path('/home/user/InfoSystemic/AI-Server/engines/llama.cpp-glm5n-goal-0904')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')

def main():
    report=HERE/'graph-validation.json';assert not report.exists()
    build=json.loads((HERE/'build-report.json').read_text());assert build['passed'] and build['parent_library_exact']
    pid=pid_for('glm53-flash-production.service');before=snapshot(pid,18131);idle(18131)
    original=json.loads((HERE/'runtime-before.json').read_text())
    for k in ['proc_start_ticks','command','environment','mapped_libraries_sha256']:assert before[k]==original[k],k
    libs=before['mapped_libraries_sha256'];by_name={Path(k).name:Path(k) for k in libs}
    candidate=HERE/'candidate/libggml-cpu.so.0.22.0';assert sha(candidate)==build['candidate_sha256']
    r=dict(passed=False,started=time.time(),production_pid=pid,candidate_sha256=sha(candidate),steps=[],direct=[],graphs=[])
    inputs=[Path(__file__),HERE/'test_graph.cpp',HERE/'test_numa.cpp',HERE/'test_q8_batch',candidate,*map(Path,libs)]
    r['input_sha256']={str(p):sha(p) for p in inputs}
    def checkpoint():save(report,r)
    def guard():assert pid_for('glm53-flash-production.service')==pid;idle(18131)
    def limit():os.nice(15);resource.setrlimit(resource.RLIMIT_AS,(24<<30,24<<30));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    def run(cmd,label,env):
        guard();start=time.monotonic();log=HERE/(label+'.log');peak=0
        with log.open('w') as out:
            p=subprocess.Popen(cmd,cwd=HERE,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=limit)
            try:
                while p.poll() is None:
                    guard();assert time.monotonic()-start<600,label
                    path=Path(f'/proc/{p.pid}/status')
                    if path.exists():
                        rss=next((int(x.split()[1]) for x in path.read_text().splitlines() if x.startswith('VmRSS:')),0)
                        peak=max(peak,rss);assert rss<6*1024**2,'RSS limit'
                    time.sleep(.5)
                assert p.returncode==0,(label,p.returncode)
            finally:
                if p.poll() is None:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=10)
        r['steps'].append(dict(label=label,command=cmd,seconds=time.monotonic()-start,peak_rss_kib=peak));checkpoint();print('completed',label,flush=True)
        return [json.loads(x) for x in log.read_text().splitlines() if x.startswith('{')]
    base_env={k:v for k,v in os.environ.items() if not k.startswith(('GGML_','LLAMA_','OMP_','GOMP_','REPACK_','Q8_TEST_'))}
    base_env.update(before['environment'])
    for k in list(base_env):
        if k.startswith('GGML_CPU_OP_PROFILE') or 'ARM_FILE' in k:base_env.pop(k)
    base_env.update(REPACK_TEST_REPEATS='3',REPACK_TEST_FULL_GLM_Q5='1',OMP_NUM_THREADS='1')
    checkpoint()
    try:
        for stem in ['test_graph','test_numa']:
            cmd=['taskset','-c','48','/usr/bin/c++','-O3','-std=c++17','-march=native','-fopenmp']
            cmd += ['-I'+str(ENGINE/p) for p in ['include','ggml/include','ggml/src','ggml/src/ggml-cpu']]
            cmd += [str(HERE/(stem+'.cpp')),str(by_name['libggml.so.0.22.0']),str(by_name['libggml-cpu.so.0.22.0']),str(by_name['libggml-base.so.0.22.0']),'-ldl','-pthread','-o',str(HERE/stem)]
            run(cmd,'compile-'+stem,base_env)
        for label,value in [('production',0),('off',0),('on',1)]:
            env=base_env.copy();env.update(GGML_CPU_Q8_BATCH_FAST=str(value),GGML_CPU_Q8_BATCH_FAST_PROBE='1')
            if label!='production':env['LD_LIBRARY_PATH']=str(candidate.parent)+':'+env['LD_LIBRARY_PATH']
            output=HERE/f'direct-library-{label}.bin'
            events=run(['taskset','-c','48',str(HERE/'test_q8_batch'),'gate',str(output)],'direct-library-'+label,env)
            loaded=by_name['libggml-cpu.so.0.22.0'] if label=='production' else candidate
            assert Path(events[0]['path']).resolve()==loaded.resolve();assert events[-1]==dict(event='done',passed=True)
            reference=HERE/'direct-library-production.bin';assert output.read_bytes()==reference.read_bytes()
            r['direct'].append(dict(label=label,exact=True,bytes=output.stat().st_size,sha256=sha(output)));checkpoint()
        for workers in [1,15]:
            for label,value in [('production',0),('off',0),('on',1),('switch',1)]:
                env=base_env.copy();env.update(GGML_CPU_Q8_BATCH_FAST=str(value),GGML_CPU_Q8_BATCH_FAST_PROBE='1',REPACK_TEST_PERSISTENT_POOL='1',REPACK_TEST_PIN_POOL='1')
                if label!='production':env['LD_LIBRARY_PATH']=str(candidate.parent)+':'+env['LD_LIBRARY_PATH']
                if label=='switch':
                    control=HERE/f'graph-control-w{workers}.u32';control.write_bytes(struct.pack('<I',0))
                    env.update(GGML_CPU_Q8_BATCH_FAST_CONTROL_FILE=str(control),Q8_TEST_SWITCH_FILE=str(control))
                tag=f'graph-w{workers}-{label}';output=HERE/(tag+'.bin')
                events=run(['taskset','-c','48' if workers==1 else '48-62',str(HERE/'test_graph'),str(workers),str(output)],tag,env)
                loaded=by_name['libggml-cpu.so.0.22.0'] if label=='production' else candidate
                assert Path(events[0]['path']).resolve()==loaded.resolve()
                final=events[-1];assert final['passed'] and final['cases']==242
                assert bool(sum(final['calls']))==bool(value)
                if label=='switch':assert final['switches']==242*4
                reference=HERE/f'graph-w{workers}-production.bin';assert output.read_bytes()==reference.read_bytes(),tag
                r['graphs'].append(dict(workers=workers,label=label,exact=True,bytes=output.stat().st_size,sha256=sha(output),execution=final));checkpoint()
        for p,d in r['input_sha256'].items():assert sha(p)==d,p
        after=snapshot(pid,18131);guard()
        for k in ['proc_start_ticks','command','environment','mapped_libraries_sha256']:assert after[k]==before[k]
        save(HERE/'graph-runtime-after.json',after);r['production_unchanged']=True;r['passed']=True
        print('Compiled-library and graph parity gates passed; production unchanged',flush=True)
    except BaseException as e:r['error']=repr(e);raise
    finally:r['finished']=time.time();checkpoint()

if __name__=='__main__':main()
