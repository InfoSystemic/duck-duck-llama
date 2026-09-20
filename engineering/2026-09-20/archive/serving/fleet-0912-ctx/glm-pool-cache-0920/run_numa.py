#!/usr/bin/env python3
"""Four-NUMA pooling-chain gate and balanced timing, no model outage."""
import json,os,resource,signal,struct,subprocess,sys,time
from pathlib import Path
from statistics import median
from validate import sha,save
B=Path('/home/user/sr950-strategy/codex-bench-0919');sys.path.insert(0,str(B))
from capture_runtime import snapshot
from pool_window import pid_for
from thread_window import idle
HERE=Path(__file__).resolve().parent
ENGINE=Path('/home/user/InfoSystemic/AI-Server/engines/llama.cpp-glm5n-goal-0904')
def main():
    report=HERE/'numa-report.json';assert not report.exists()
    gate=json.loads((HERE/'validation-report.json').read_text());assert gate['passed'] and len(gate['stateful'])==10
    candidate=HERE/'candidate/libggml-cpu.so.0.22.0';assert sha(candidate)==gate['candidate_sha256']
    pid=pid_for('glm53-flash-production.service');before=snapshot(pid,18131);idle(18131)
    old=json.loads((HERE/'runtime-before.json').read_text())
    for k in ['proc_start_ticks','command','environment','mapped_libraries_sha256']:assert before[k]==old[k],k
    mem=dict((x.split(':')[0],int(x.split()[1])) for x in Path('/proc/meminfo').read_text().splitlines());assert mem['MemAvailable']>30*1024**2
    r=dict(passed=False,started=time.time(),production_pid=pid,candidate_sha256=sha(candidate),runs=[],scope='Eleven fused pooling layers across four NUMA devices, three tail cells overwritten per step. Synthetic pooling-only graph timing; not model tok/s.')
    r['input_sha256']={str(p):sha(p) for p in [Path(__file__),HERE/'bench_numa.cpp',candidate,HERE/'validation-report.json']}
    def checkpoint():save(report,r)
    def guard():assert pid_for('glm53-flash-production.service')==pid;idle(18131)
    def limits():os.nice(10);resource.setrlimit(resource.RLIMIT_AS,(24<<30,24<<30));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    def run(cmd,label,env):
        guard();start=time.monotonic();peak=0
        with (HERE/(label+'.log')).open('w') as out:
            p=subprocess.Popen(cmd,cwd=HERE,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=limits)
            try:
                while p.poll() is None:
                    guard();assert time.monotonic()-start<300,label
                    proc=Path(f'/proc/{p.pid}/status')
                    if proc.exists():
                        rss=next((int(x.split()[1]) for x in proc.read_text().splitlines() if x.startswith('VmRSS:')),0);peak=max(peak,rss);assert rss<10*1024**2
                    time.sleep(.5)
                assert p.returncode==0,(label,p.returncode)
            finally:
                if p.poll() is None:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=10)
        events=[json.loads(x) for x in (HERE/(label+'.log')).read_text().splitlines() if x.startswith('{')]
        return dict(label=label,command=cmd,seconds=time.monotonic()-start,peak_rss_kib=peak,events=events)
    base={k:v for k,v in os.environ.items() if not k.startswith(('GGML_','LLAMA_','OMP_','GOMP_','REPACK_','POOL_'))};base.update(before['environment'])
    for k in list(base):
        if k.startswith('GGML_CPU_OP_PROFILE') or 'ARM_FILE' in k:base.pop(k)
    base['LD_LIBRARY_PATH']=str(candidate.parent)+':'+base['LD_LIBRARY_PATH'];libs={Path(p).name:Path(p) for p in before['mapped_libraries_sha256']}
    checkpoint()
    try:
        cmd=['taskset','-c','48','/usr/bin/c++','-std=c++17','-O3','-march=native','-I'+str(ENGINE/'ggml/include'),str(HERE/'bench_numa.cpp'),str(libs['libggml-cpu.so.0.22.0']),str(libs['libggml-base.so.0.22.0']),'-ldl','-pthread','-o',str(HERE/'bench_numa')]
        r['compile']=run(cmd,'compile-bench_numa',base);r['input_sha256'][str(HERE/'bench_numa')]=sha(HERE/'bench_numa');checkpoint()
        for label,pools,rounds,probe in [('observer',1026,3,True),('p1026',1026,31,False),('p8194',8194,31,False),('p25002',25002,31,False)]:
            control=HERE/f'numa-{label}.u32';control.write_bytes(struct.pack('<I',0));output=HERE/f'numa-{label}.bin'
            env=base.copy();sequence='01' if probe else '01101001';env.update(GGML_CPU_GLM_POOL_CACHE='1',GGML_CPU_GLM_POOL_CACHE_CONTROL_FILE=str(control),POOL_BENCH_OUTPUT=str(output),POOL_BENCH_SEQUENCE=sequence)
            if probe:env.update(GGML_CPU_GLM_POOL_CACHE_PROBE='1',POOL_BENCH_OBSERVER='1')
            result=run(['taskset','-c','0-127',str(HERE/'bench_numa'),str(pools),'11',str(rounds),'4'],'numa-'+label,env)
            events=result['events'];assert Path(events[0]['path']).resolve()==candidate.resolve();assert events[0]['devices']==4 and events[0]['workers']=='15' and events[0]['observer']==probe
            final=events[-1];assert final['passed'] and final['fusions']==final['graphs']*44;assert bool(final['stats'][1])==probe
            assert final['stats'][5]<=4096*2**20
            arms=[x for x in events if x.get('event')=='arm'];assert len(arms)==len(sequence)
            for i,x in enumerate(arms):assert x['rounds']==rounds and x['order']==i and x['enabled']==int(sequence[i]) and x['exact']
            if not probe:
                off=[x for x in arms if not x['enabled']];on=[x for x in arms if x['enabled']]
                result['summary']=dict(off_median_ms=median(x['median_ms'] for x in off),on_median_ms=median(x['median_ms'] for x in on),median_speedup=median(x['median_ms'] for x in off)/median(x['median_ms'] for x in on),aggregate_speedup=sum(x['total_ms'] for x in off)/sum(x['total_ms'] for x in on),first_cache_fill_ms=final['first_on_ms'],peak_reserved_bytes=final['stats'][5])
                result['blocks']=[sum(x['total_ms'] for x in arms[start:start+4] if not x['enabled'])/sum(x['total_ms'] for x in arms[start:start+4] if x['enabled']) for start in [0,4]]
            result['output']={'bytes':output.stat().st_size,'sha256':sha(output)};assert output.stat().st_size==final['output_values_per_step']*4*rounds;assert control.read_bytes()==struct.pack('<I',0)
            r['runs'].append(result);checkpoint();print(label,result.get('summary',{'all_mirrors_parity':True,'stats':final['stats']}),'blocks',result.get('blocks'),flush=True)
        for p,d in r['input_sha256'].items():assert sha(p)==d,p
        after=snapshot(pid,18131);guard()
        for k in ['proc_start_ticks','command','environment','mapped_libraries_sha256']:assert after[k]==before[k],k
        save(HERE/'numa-runtime-after.json',after);r['production_unchanged']=True;r['passed']=True
    except BaseException as e:r['error']=repr(e);raise
    finally:r['finished']=time.time();checkpoint()
if __name__=='__main__':main()
