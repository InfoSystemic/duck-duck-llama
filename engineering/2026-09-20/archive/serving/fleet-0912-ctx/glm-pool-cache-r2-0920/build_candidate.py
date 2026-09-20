#!/usr/bin/env python3
"""Build an exact deployed CPU parent and the private pooled-result cache."""
import hashlib,json,os,resource,shlex,signal,subprocess,sys,time
from pathlib import Path
B=Path('/home/user/sr950-strategy/codex-bench-0919');sys.path.insert(0,str(B))
from capture_runtime import snapshot
from pool_window import pid_for
from thread_window import idle
HERE=Path(__file__).resolve().parent
WIDE=HERE.parent/'glm-kpool-wide-0919'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def main():
    report=HERE/'build-report.json';assert not report.exists()
    pid=pid_for('glm53-flash-production.service');before=snapshot(pid,18131);idle(18131)
    expected=json.loads((HERE/'runtime-before.json').read_text())
    for k in ['proc_start_ticks','command','environment','mapped_libraries_sha256']:assert before[k]==expected[k],k
    parent=WIDE/'deploy/libggml-cpu.so.0.22.0';assert sha(parent)=='3f957a341321b940d93be53c250cdd068825093faa2d9efda142ebe56427b1b3'
    provenance=json.loads((WIDE/'build-provenance.json').read_text())
    for p,d in provenance.items():assert sha(p)==d,p
    r=dict(passed=False,started=time.time(),production_pid=pid,steps=[],parent_expected_sha256=sha(parent))
    def checkpoint():save(report,r)
    def guard():assert pid_for('glm53-flash-production.service')==pid;idle(18131)
    def limit():os.nice(15);resource.setrlimit(resource.RLIMIT_AS,(4<<30,4<<30));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    def run(cmd,label):
        guard();start=time.monotonic()
        with (HERE/(label+'.log')).open('w') as out:
            p=subprocess.Popen(['taskset','-c','48']+cmd,cwd=HERE,stdout=out,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=limit)
            try:
                while p.poll() is None:
                    guard();assert time.monotonic()-start<600;time.sleep(.5)
                assert p.returncode==0,(label,p.returncode)
            finally:
                if p.poll() is None:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=10)
        r['steps'].append(dict(label=label,command=cmd,seconds=time.monotonic()-start));checkpoint();print('completed',label,flush=True)
    checkpoint()
    try:
        for folder in ['reference','candidate']:(HERE/folder).mkdir(exist_ok=False)
        lines=[x for x in (WIDE/'rebuild.sh').read_text().splitlines() if x.startswith('/usr/bin/c++')];assert len(lines)==2
        compile_cmd=shlex.split(lines[0].replace('$D',str(WIDE)).replace('$OUT',str(HERE/'reference')))
        run(compile_cmd,'parent-compile')
        r['parent_object_sha256']=sha(HERE/'reference/ops.cpp.o');r['parent_object_exact']=r['parent_object_sha256']==sha(WIDE/'build/ops.cpp.o');assert r['parent_object_exact']
        link=shlex.split(lines[1].replace('$OUT',str(WIDE/'build')))
        for value in link:
            if value.endswith('.o'):assert sha(value)==provenance[value],value
        link[link.index('-o')+1]=str(HERE/'reference/libggml-cpu.so.0.22.0')
        link[link.index(str(WIDE/'build/ops.cpp.o'))]=str(HERE/'reference/ops.cpp.o')
        run(link,'parent-link')
        r['parent_library_sha256']=sha(HERE/'reference/libggml-cpu.so.0.22.0');r['parent_library_exact']=r['parent_library_sha256']==sha(parent);assert r['parent_library_exact']
        (HERE/'ops.combo.cpp').write_bytes((WIDE/'ops.combo.cpp').read_bytes())
        candidate_compile=compile_cmd[:];candidate_compile[1:1]=['-I'+str(HERE)]
        candidate_compile[candidate_compile.index('-o')+1]=str(HERE/'candidate/ops.cpp.o');candidate_compile[-1]=str(HERE/'ops.combo.cpp')
        run(candidate_compile,'candidate-compile')
        candidate_link=link[:];candidate_link[candidate_link.index('-o')+1]=str(HERE/'candidate/libggml-cpu.so.0.22.0')
        candidate_link[candidate_link.index(str(HERE/'reference/ops.cpp.o'))]=str(HERE/'candidate/ops.cpp.o')
        run(candidate_link,'candidate-link')
        for folder in ['reference','candidate']:
            p=HERE/folder;(p/'libggml-cpu.so.0').symlink_to('libggml-cpu.so.0.22.0');(p/'libggml-cpu.so').symlink_to('libggml-cpu.so.0')
        r['candidate_sha256']=sha(HERE/'candidate/libggml-cpu.so.0.22.0')
        r['input_sha256']={str(p):sha(p) for p in [Path(__file__),HERE/'ops.combo.cpp',HERE/'pool-kernel.inc',HERE/'pool-cache.inc',WIDE/'pool-kernel.inc',parent]}
        after=snapshot(pid,18131);guard()
        for k in ['proc_start_ticks','command','environment','mapped_libraries_sha256']:assert after[k]==before[k],k
        save(HERE/'build-runtime-after.json',after);r['production_unchanged']=True;r['passed']=True
        print('Exact parent and pooled-result cache candidate built; production unchanged',flush=True)
    except BaseException as e:r['error']=repr(e);raise
    finally:r['finished']=time.time();checkpoint()
if __name__=='__main__':main()
