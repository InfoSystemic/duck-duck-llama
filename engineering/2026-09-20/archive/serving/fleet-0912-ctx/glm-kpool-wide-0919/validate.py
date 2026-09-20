#!/usr/bin/env python3
import argparse, hashlib, json, os, resource, subprocess, time, urllib.request
from pathlib import Path
D=Path(__file__).resolve().parent
F=D.parent
BASE=':'.join(map(str,[F/'glm-fix',F/'unary-lib',F.parent/'fleet-0903/results/glm-flash-q8-r8-ordered-k-0908/private-cpu',F.parent.parent/'engines/llama.cpp-glm5n-goal-0904/validated-chunk16-bin']))
ARMS=[('unfused',F/'unary-lib',0,0),('deployed',F/'glm-cpu-fast-0919',1,0),('candidate-off',D/'build',0,0),('candidate-scalar',D/'build',1,0),('candidate-wide',D/'build',1,1)]
def idle():
    with urllib.request.urlopen('http://127.0.0.1:18131/slots',timeout=10) as r:
        assert not any(s['is_processing'] for s in json.load(r)), 'Live endpoint busy; refusing standalone load'
def limits():
    resource.setrlimit(resource.RLIMIT_AS,(12*1024**3,12*1024**3))
def run(arm,threads,kind='pool',pools=0,suffix=''):
    name,lib,pool,wide=arm
    idle()
    dest=D/'validation';dest.mkdir(exist_ok=True)
    stem=f'{kind}-t{threads}-{name}'+suffix
    env=dict(os.environ,LD_LIBRARY_PATH=str(lib)+':'+BASE,GGML_CPU_GLM_POOL_FUSION=str(pool),
        GGML_CPU_GLM_POOL_WIDE=str(wide),EXPECT_WIDE=str(wide),GGML_CPU_CPY_FLAT='1',
        GGML_CPU_SOFTMAX_POOL_FUSION='1',GGML_CPU_PARALLEL_UNARY='4096',
        GGML_CPU_PARALLEL_COPY='1',GGML_CPU_SINGLE_TASK_MAX_ELEMENTS='4096',OMP_NUM_THREADS=str(threads))
    if pools:env['POOL_BENCH_POOLS']=str(pools)
    expected=pool if kind=='pool' else int(name!='unfused')
    affinity='0-14' if threads>4 else '124-127'
    start=time.monotonic()
    with (dest/(stem+'.log')).open('w') as log:
        child=subprocess.Popen(['taskset','-c',affinity,str(D/('test_'+kind)),str(dest/(stem+'.bin')),str(expected),str(threads)],
            env=env,stdout=log,stderr=subprocess.STDOUT,preexec_fn=limits)
        try:
            while child.poll() is None:
                if time.monotonic()-start>180:raise TimeoutError(stem)
                idle();time.sleep(1)
            if child.returncode:raise RuntimeError((dest/(stem+'.log')).read_text()[-4000:])
        finally:
            if child.poll() is None:child.terminate();child.wait(timeout=10)
    log=(dest/(stem+'.log')).read_text()
    loaded_cpu=Path(next(x.split(' ',1)[1] for x in log.splitlines() if x.startswith('CPU_LIBRARY '))).resolve()
    loaded_base=Path(next(x.split(' ',1)[1] for x in log.splitlines() if x.startswith('BASE_LIBRARY '))).resolve()
    assert loaded_cpu==(lib/'libggml-cpu.so.0').resolve()
    assert loaded_base==(F/'glm-fix/libggml-base.so.0').resolve()
    data=(dest/(stem+'.bin')).read_bytes()
    row=dict(arm=name,threads=threads,kind=kind,pools=pools,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),
        cpu_sha256=hashlib.sha256(loaded_cpu.read_bytes()).hexdigest(),wall_seconds=time.monotonic()-start,
        output=str(dest/(stem+'.bin')),activation_verified=True)
    if pools:row['milliseconds']=[float(line.rsplit('=',1)[1]) for line in log.splitlines() if line.startswith('BENCH ')]
    return row,data

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--threads',nargs='+',type=int,default=[3]);a=ap.parse_args()
    report={'candidate_sha256':hashlib.sha256((D/'build/libggml-cpu.so.0.22.0').read_bytes()).hexdigest(),'rows':[],'passed':False}
    dest=D/('validation-'+','.join(map(str,a.threads))+'.json')
    reference=None
    for threads in a.threads:
        for arm in ARMS:
            row,data=run(arm,threads)
            if reference is None:reference=data
            if data!=reference:
                first=next((i for i,(x,y) in enumerate(zip(reference,data)) if x!=y),min(len(data),len(reference)))
                raise AssertionError(f'{row["arm"]} t{threads}: first differing byte {first}')
            row['bit_exact']=True;report['rows'].append(row)
            dest.write_text(json.dumps(report,indent=2)+'\n')
            print('PASS',row['arm'],'threads',threads,'bytes',len(data),flush=True)
    report['passed']=True;dest.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
