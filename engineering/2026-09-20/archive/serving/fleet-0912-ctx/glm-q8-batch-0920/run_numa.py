#!/usr/bin/env python3
"""Balanced four-NUMA Q8 graph timing after exact graph validation; no model outage."""
import hashlib,json,os,resource,signal,subprocess,sys,time
from pathlib import Path
from statistics import median
B=Path('/home/user/sr950-strategy/codex-bench-0919');sys.path.insert(0,str(B))
from capture_runtime import snapshot
from pool_window import pid_for
from thread_window import idle
HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def main():
    report=HERE/'numa-report.json';assert not report.exists()
    gate=json.loads((HERE/'graph-validation.json').read_text());assert gate['passed'] and len(gate['graphs'])==8
    pid=pid_for('glm53-flash-production.service');before=snapshot(pid,18131);idle(18131)
    old=json.loads((HERE/'runtime-before.json').read_text())
    for k in ['proc_start_ticks','command','environment','mapped_libraries_sha256']:assert before[k]==old[k],k
    candidate=HERE/'candidate/libggml-cpu.so.0.22.0';assert sha(candidate)==gate['candidate_sha256']
    r=dict(passed=False,started=time.time(),production_pid=pid,candidate_sha256=sha(candidate),runs=[],scope='Synthetic four-NUMA Q8 target/draft graph timing; excludes real model attention, experts and caches; not decode throughput')
    r['input_sha256']={str(p):sha(p) for p in [Path(__file__),HERE/'test_numa.cpp',HERE/'test_numa',candidate,HERE/'graph-validation.json']}
    def checkpoint():save(report,r)
    def guard():assert pid_for('glm53-flash-production.service')==pid;idle(18131)
    def limits():os.nice(10);resource.setrlimit(resource.RLIMIT_AS,(24<<30,24<<30));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    envbase={k:v for k,v in os.environ.items() if not k.startswith(('GGML_','LLAMA_','OMP_','GOMP_','REPACK_','Q8_TEST_','PROBE_'))}
    envbase.update(before['environment'])
    for k in list(envbase):
        if k.startswith('GGML_CPU_OP_PROFILE') or 'ARM_FILE' in k:envbase.pop(k)
    envbase.update(LD_LIBRARY_PATH=str(candidate.parent)+':'+envbase['LD_LIBRARY_PATH'])
    checkpoint()
    try:
        for order,enabled in enumerate([0,1,1,0,1,0,0,1]):
            guard();tag=f'numa-{order}-{enabled}';env=envbase.copy();env.update(GGML_CPU_Q8_BATCH_FAST=str(enabled),PROBE_OUTPUT_PREFIX=str(HERE/tag))
            cmd=['taskset','-c','0-127',str(HERE/'test_numa'),'4096','8192','32','2','61'];peak=0;start=time.monotonic()
            with (HERE/(tag+'.log')).open('w') as out:
                p=subprocess.Popen(cmd,cwd=HERE,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=limits)
                try:
                    while p.poll() is None:
                        guard();assert time.monotonic()-start<180
                        proc=Path(f'/proc/{p.pid}/status')
                        if proc.exists():
                            rss=next((int(x.split()[1]) for x in proc.read_text().splitlines() if x.startswith('VmRSS:')),0);peak=max(peak,rss);assert rss<6*1024**2
                        time.sleep(.5)
                    assert p.returncode==0,(tag,p.returncode)
                finally:
                    if p.poll() is None:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=10)
            events=[json.loads(x) for x in (HERE/(tag+'.log')).read_text().splitlines() if x.startswith('{')]
            assert Path(events[0]['cpu_library']).resolve()==candidate.resolve()
            assert events[0]['workers']=='15'
            end=events[-1];assert end['completed'] and end['rounds']==61
            outputs={}
            for kind in ['target','draft']:
                output=HERE/f'{tag}.{kind}.bin';ref=HERE/f'numa-0-0.{kind}.bin';assert output.read_bytes()==ref.read_bytes(),(tag,kind)
                outputs[kind]=dict(bytes=output.stat().st_size,sha256=sha(output))
            r['runs'].append(dict(order=order,enabled=bool(enabled),command=cmd,seconds=time.monotonic()-start,peak_rss_kib=peak,measurement=end,output_bytes_exact=True,outputs=outputs));checkpoint()
            print(tag,'median_cycle_ms',end['median_cycle_ms'],flush=True)
        off=[x['measurement']['median_cycle_ms'] for x in r['runs'] if not x['enabled']]
        on=[x['measurement']['median_cycle_ms'] for x in r['runs'] if x['enabled']]
        r['median_of_run_medians_ms']={'off':median(off),'on':median(on)};r['speedup']=median(off)/median(on)
        r['blocks']=[]
        for start in [0,4]:
            rows=r['runs'][start:start+4];a=sum(x['measurement']['wall_seconds'] for x in rows if not x['enabled']);b=sum(x['measurement']['wall_seconds'] for x in rows if x['enabled'])
            r['blocks'].append(dict(first_order=start,aggregate_speedup=a/b))
        for p,d in r['input_sha256'].items():assert sha(p)==d,p
        after=snapshot(pid,18131);guard()
        for k in ['proc_start_ticks','command','environment','mapped_libraries_sha256']:assert after[k]==before[k]
        save(HERE/'numa-runtime-after.json',after);r['production_unchanged']=True;r['passed']=True
        print('Four-NUMA graph speed ratio',r['speedup'],'blocks',r['blocks'],flush=True)
    except BaseException as e:r['error']=repr(e);raise
    finally:r['finished']=time.time();checkpoint()

if __name__=='__main__':main()
