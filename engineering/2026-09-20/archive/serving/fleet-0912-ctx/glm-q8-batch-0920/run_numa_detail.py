#!/usr/bin/env python3
"""Confirm NUMA engagement and isolate target/draft timing with distinct weights."""
import hashlib,json,os,resource,signal,struct,subprocess,sys,time
from pathlib import Path
from statistics import median
B=Path('/home/user/sr950-strategy/codex-bench-0919');sys.path.insert(0,str(B))
from capture_runtime import snapshot
from pool_window import pid_for
from thread_window import idle
HERE=Path(__file__).resolve().parent
ENGINE=Path('/home/user/InfoSystemic/AI-Server/engines/llama.cpp-glm5n-goal-0904')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def main():
    report=HERE/'numa-detail-report.json';assert not report.exists()
    gate=json.loads((HERE/'graph-validation.json').read_text());assert gate['passed']
    candidate=HERE/'candidate/libggml-cpu.so.0.22.0';assert sha(candidate)==gate['candidate_sha256']
    pid=pid_for('glm53-flash-production.service');before=snapshot(pid,18131);idle(18131)
    old=json.loads((HERE/'runtime-before.json').read_text())
    for k in ['proc_start_ticks','command','environment','mapped_libraries_sha256']:assert before[k]==old[k],k
    r=dict(passed=False,started=time.time(),production_pid=pid,candidate_sha256=sha(candidate),runs=[],scope='Synthetic Q8 NUMA graph phases, same-process balanced flag toggles; not model decode throughput')
    r['input_sha256']={str(p):sha(p) for p in [Path(__file__),HERE/'test_numa_detail.cpp',candidate,HERE/'graph-validation.json']}
    def checkpoint():save(report,r)
    def guard():assert pid_for('glm53-flash-production.service')==pid;idle(18131)
    def limits():os.nice(10);resource.setrlimit(resource.RLIMIT_AS,(24<<30,24<<30));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    def run(cmd,label,env):
        guard();start=time.monotonic();peak=0
        with (HERE/(label+'.log')).open('w') as out:
            p=subprocess.Popen(cmd,cwd=HERE,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=limits)
            try:
                while p.poll() is None:
                    guard();assert time.monotonic()-start<180,label
                    proc=Path(f'/proc/{p.pid}/status')
                    if proc.exists():
                        rss=next((int(x.split()[1]) for x in proc.read_text().splitlines() if x.startswith('VmRSS:')),0);peak=max(peak,rss);assert rss<6*1024**2
                    time.sleep(.5)
                assert p.returncode==0,(label,p.returncode)
            finally:
                if p.poll() is None:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=10)
        events=[json.loads(x) for x in (HERE/(label+'.log')).read_text().splitlines() if x.startswith('{')]
        return dict(label=label,command=cmd,seconds=time.monotonic()-start,peak_rss_kib=peak,events=events)
    envbase={k:v for k,v in os.environ.items() if not k.startswith(('GGML_','LLAMA_','OMP_','GOMP_','REPACK_','Q8_TEST_','PROBE_'))}
    envbase.update(before['environment'])
    for k in list(envbase):
        if k.startswith('GGML_CPU_OP_PROFILE') or 'ARM_FILE' in k:envbase.pop(k)
    envbase['LD_LIBRARY_PATH']=str(candidate.parent)+':'+envbase['LD_LIBRARY_PATH']
    checkpoint()
    try:
        by_name={Path(k).name:Path(k) for k in before['mapped_libraries_sha256']}
        cmd=['taskset','-c','48','/usr/bin/c++','-O3','-std=c++17','-march=native','-fopenmp']
        cmd+=['-I'+str(ENGINE/p) for p in ['include','ggml/include','ggml/src','ggml/src/ggml-cpu']]
        cmd += [str(HERE/'test_numa_detail.cpp'),str(by_name['libggml.so.0.22.0']),str(by_name['libggml-cpu.so.0.22.0']),str(by_name['libggml-base.so.0.22.0']),'-ldl','-pthread','-o',str(HERE/'test_numa_detail')]
        r['compile']=run(cmd,'compile-test_numa_detail',envbase);r['input_sha256'][str(HERE/'test_numa_detail')]=sha(HERE/'test_numa_detail');checkpoint()
        for label,sets,rounds,sequence,probe in [('engagement',4,1,'01',True),('distinct4',4,61,'01101001',False),('distinct32',32,61,'01101001',False)]:
            control=HERE/f'numa-detail-{label}.u32';control.write_bytes(struct.pack('<I',0))
            env=envbase.copy();env.update(GGML_CPU_Q8_BATCH_FAST='1',GGML_CPU_Q8_BATCH_FAST_CONTROL_FILE=str(control),PROBE_WEIGHT_SETS=str(sets),PROBE_SEQUENCE=sequence,PROBE_OUTPUT_PREFIX=str(HERE/f'numa-detail-{label}'))
            if probe:env['GGML_CPU_Q8_BATCH_FAST_PROBE']='1'
            result=run(['taskset','-c','0-127',str(HERE/'test_numa_detail'),'4096','8192','32','2',str(rounds)],'numa-detail-'+label,env)
            events=result['events'];assert Path(events[0]['cpu_library']).resolve()==candidate.resolve();assert events[0]['workers']=='15';assert events[-1]=={'event':'done','passed':True}
            arms=[x for x in events if x.get('event')=='arm'];assert len(arms)==len(sequence)
            for i,x in enumerate(arms):
                assert x['rounds']==rounds and x['order']==i and x['enabled']==int(sequence[i]) and x['exact']
                assert bool(sum(x['calls']))==bool(probe and x['enabled']),x['calls']
            if not probe:
                result['summary']={}
                for phase in ['cycle','target','draft']:
                    a=[x for x in arms if not x['enabled']];b=[x for x in arms if x['enabled']]
                    key=f'median_{phase}_ms';total=f'total_{phase}_ms'
                    result['summary'][phase]=dict(off_median_ms=median(x[key] for x in a),on_median_ms=median(x[key] for x in b),median_speedup=median(x[key] for x in a)/median(x[key] for x in b),aggregate_speedup=sum(x[total] for x in a)/sum(x[total] for x in b))
                result['blocks']=[]
                for start in [0,4]:
                    block=arms[start:start+4];result['blocks'].append({p:sum(x[f'total_{p}_ms'] for x in block if not x['enabled'])/sum(x[f'total_{p}_ms'] for x in block if x['enabled']) for p in ['cycle','target','draft']})
            result['outputs']={kind:{'bytes':(HERE/f'numa-detail-{label}.{kind}.bin').stat().st_size,'sha256':sha(HERE/f'numa-detail-{label}.{kind}.bin')} for kind in ['target','draft']}
            if sets==4:
                for kind in ['target','draft']:assert (HERE/f'numa-detail-{label}.{kind}.bin').read_bytes()==(HERE/f'numa-0-0.{kind}.bin').read_bytes()
            assert control.read_bytes()==struct.pack('<I',0)
            r['runs'].append(result);checkpoint();print(label,result.get('summary',{'engagement_passed':True}),flush=True)
        for p,d in r['input_sha256'].items():assert sha(p)==d,p
        after=snapshot(pid,18131);guard()
        for k in ['proc_start_ticks','command','environment','mapped_libraries_sha256']:assert after[k]==before[k],k
        save(HERE/'numa-detail-runtime-after.json',after);r['production_unchanged']=True;r['passed']=True
    except BaseException as e:r['error']=repr(e);raise
    finally:r['finished']=time.time();checkpoint()
if __name__=='__main__':main()
