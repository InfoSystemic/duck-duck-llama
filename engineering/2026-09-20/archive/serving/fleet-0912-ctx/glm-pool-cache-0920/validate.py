#!/usr/bin/env python3
"""Guarded exact-output validation of the pooled-result cache; no model outage."""
import hashlib,json,os,resource,signal,struct,subprocess,sys,time
from pathlib import Path
B=Path('/home/user/sr950-strategy/codex-bench-0919');sys.path.insert(0,str(B))
from capture_runtime import snapshot
from pool_window import pid_for
from thread_window import idle
HERE=Path(__file__).resolve().parent
ENGINE=Path('/home/user/InfoSystemic/AI-Server/engines/llama.cpp-glm5n-goal-0904')
WIDE=HERE.parent/'glm-kpool-wide-0919'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()
def exact(a,b):
    assert a.stat().st_size==b.stat().st_size,(a,b,'size')
    with a.open('rb') as x,b.open('rb') as y:
        offset=0
        while True:
            u=x.read(1<<20);v=y.read(1<<20)
            if u!=v:
                first=next(i for i,(j,k) in enumerate(zip(u,v)) if j!=k);raise AssertionError((str(a),str(b),'first unequal byte',offset+first))
            if not u:return
            offset+=len(u)
def save(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def main():
    report=HERE/'validation-report.json';assert not report.exists()
    build=json.loads((HERE/'build-report.json').read_text());assert build['passed'] and build['parent_library_exact']
    candidate=HERE/'candidate/libggml-cpu.so.0.22.0';assert sha(candidate)==build['candidate_sha256']
    pid=pid_for('glm53-flash-production.service');before=snapshot(pid,18131);idle(18131)
    old=json.loads((HERE/'runtime-before.json').read_text())
    for k in ['proc_start_ticks','command','environment','mapped_libraries_sha256']:assert before[k]==old[k],k
    r=dict(passed=False,started=time.time(),production_pid=pid,candidate_sha256=sha(candidate),steps=[],legacy=[],stateful=[])
    r['input_sha256']={str(p):sha(p) for p in [Path(__file__),HERE/'test_cache.cpp',candidate,WIDE/'test_pool',WIDE/'test_pool.cpp',HERE/'pool-cache.inc',HERE/'pool-kernel.inc']}
    def checkpoint():save(report,r)
    def guard():assert pid_for('glm53-flash-production.service')==pid;idle(18131)
    def limits():os.nice(15);resource.setrlimit(resource.RLIMIT_AS,(16<<30,16<<30));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    def run(cmd,label,env):
        guard();start=time.monotonic();peak=0
        with (HERE/(label+'.log')).open('w') as out:
            p=subprocess.Popen(cmd,cwd=HERE,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=limits)
            try:
                while p.poll() is None:
                    guard();assert time.monotonic()-start<600,label
                    proc=Path(f'/proc/{p.pid}/status')
                    if proc.exists():
                        rss=next((int(x.split()[1]) for x in proc.read_text().splitlines() if x.startswith('VmRSS:')),0);peak=max(peak,rss);assert rss<6*1024**2
                    time.sleep(.5)
                assert p.returncode==0,(label,p.returncode)
            finally:
                if p.poll() is None:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=10)
        r['steps'].append(dict(label=label,command=cmd,seconds=time.monotonic()-start,peak_rss_kib=peak));checkpoint();print('completed',label,flush=True)
        return (HERE/(label+'.log')).read_text()
    base={k:v for k,v in os.environ.items() if not k.startswith(('GGML_','LLAMA_','OMP_','GOMP_','REPACK_','POOL_'))};base.update(before['environment'])
    for k in list(base):
        if k.startswith('GGML_CPU_OP_PROFILE') or 'ARM_FILE' in k:base.pop(k)
    libs={Path(p).name:Path(p) for p in before['mapped_libraries_sha256']}
    checkpoint()
    try:
        cmd=['taskset','-c','48','/usr/bin/c++','-std=c++17','-O2','-march=native','-I'+str(ENGINE/'ggml/include'),str(HERE/'test_cache.cpp'),str(libs['libggml-cpu.so.0.22.0']),str(libs['libggml-base.so.0.22.0']),'-ldl','-pthread','-o',str(HERE/'test_cache')]
        run(cmd,'compile-test_cache',base);r['input_sha256'][str(HERE/'test_cache')]=sha(HERE/'test_cache')
        for workers in [1,2,15]:
            for label,value in [('production',0),('on',1)]:
                env=base.copy();env.update(GGML_CPU_GLM_POOL_CACHE=str(value),EXPECT_WIDE='1')
                if label!='production':env['LD_LIBRARY_PATH']=str(candidate.parent)+':'+env['LD_LIBRARY_PATH']
                tag=f'legacy-w{workers}-{label}';output=HERE/(tag+'.bin')
                log=run(['taskset','-c','48' if workers==1 else f'48-{47+workers}',str(WIDE/'test_pool'),str(output),'1',str(workers)],tag,env)
                assert 'PASS 77 cases, two executions each' in log
                expected=libs['libggml-cpu.so.0.22.0'] if label=='production' else candidate
                loaded=next(x.split(' ',1)[1] for x in log.splitlines() if x.startswith('CPU_LIBRARY '));assert Path(loaded).resolve()==expected.resolve()
                reference=HERE/f'legacy-w{workers}-production.bin';exact(reference,output)
                r['legacy'].append(dict(workers=workers,label=label,exact=True,bytes=output.stat().st_size,sha256=sha(output)));checkpoint()
        for workers in [2,15]:
            for label,value in [('production',0),('off',0),('on',1),('switch',1),('limited',1)]:
                env=base.copy();env.update(GGML_CPU_GLM_POOL_CACHE=str(value),GGML_CPU_GLM_POOL_CACHE_PROBE='1')
                if label!='production':env['LD_LIBRARY_PATH']=str(candidate.parent)+':'+env['LD_LIBRARY_PATH']
                if label=='switch':
                    control=HERE/f'gate-w{workers}.u32';control.write_bytes(struct.pack('<I',0));env.update(GGML_CPU_GLM_POOL_CACHE_CONTROL_FILE=str(control),POOL_CACHE_TEST_SWITCH_FILE=str(control))
                if label=='limited':env['GGML_CPU_GLM_POOL_CACHE_MAX_MIB']='1'
                tag=f'stateful-w{workers}-{label}';output=HERE/(tag+'.bin')
                log=run(['taskset','-c',f'48-{47+workers}',str(HERE/'test_cache'),str(output),str(workers)],tag,env)
                events=[json.loads(x) for x in log.splitlines() if x.startswith('{')];assert events[-1]['passed']
                expected=libs['libggml-cpu.so.0.22.0'] if label=='production' else candidate
                assert Path(events[0]['path']).resolve()==expected.resolve()
                final=events[-1];assert final['computes']==2*final['records'] and final['records']>=100
                if value:
                    assert final['stats'][1]>0 and final['stats'][2]>0,'cache engagement'
                    assert final['stats'][5]<=(1 if label=='limited' else 4096)*2**20,'cache bound'
                else:assert final['stats']==[0]*7
                reference=HERE/f'stateful-w{workers}-production.bin';exact(reference,output)
                r['stateful'].append(dict(workers=workers,label=label,exact=True,bytes=output.stat().st_size,sha256=sha(output),execution=final));checkpoint()
        for p,d in r['input_sha256'].items():assert sha(p)==d,p
        after=snapshot(pid,18131);guard()
        for k in ['proc_start_ticks','command','environment','mapped_libraries_sha256']:assert after[k]==before[k],k
        save(HERE/'validation-runtime-after.json',after);r['production_unchanged']=True;r['passed']=True
        print('All pooled-result cache exact-output gates passed; production unchanged',flush=True)
    except BaseException as e:r['error']=repr(e);raise
    finally:r['finished']=time.time();checkpoint()
if __name__=='__main__':main()
