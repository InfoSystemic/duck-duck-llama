#!/usr/bin/env python3
"""Rebuild the exact deployed parent, then change only the Q8 batch x86 object."""
import difflib,hashlib,json,os,resource,shlex,signal,subprocess,sys,time
from pathlib import Path
B=Path('/home/user/sr950-strategy/codex-bench-0919');sys.path.insert(0,str(B))
from capture_runtime import snapshot
from pool_window import pid_for
from thread_window import idle
HERE=Path(__file__).resolve().parent
F=HERE.parent
OLD=Path('/home/user/InfoSystemic/AI-Server/serving/fleet-0903/results/glm-flash-q8-sum16-0908')
WIDE=F/'glm-kpool-wide-0919'
CPU=WIDE/'deploy/libggml-cpu.so.0.22.0'
SOURCE=OLD/'private-cpu/repack-x86.cpp'
OBJ=OLD/'private-cpu/repack-x86.cpp.o'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')

def main():
    report=HERE/'build-report.json';assert not report.exists()
    assert json.loads((HERE/'direct-report.json').read_text())['passed']
    pid=pid_for('glm53-flash-production.service');before=snapshot(pid,18131);idle(18131)
    expected=json.loads((B/'mtp-kv-only-final-audit.json').read_text())
    for k in ['proc_start_ticks','command','environment','mapped_libraries_sha256']:assert before[k]==expected[k]
    r=dict(passed=False,started=time.time(),production_pid=pid,steps=[],parent_expected_sha256=sha(CPU))
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
        history=json.loads((OLD/'result.json').read_text())
        compile_cmd=next(x['command'] for x in history['steps'] if x['label']=='private-compile')[:]
        compile_cmd[compile_cmd.index('-o')+1]=str(HERE/'reference/repack-x86.cpp.o')
        assert Path(compile_cmd[-1])==SOURCE
        run(compile_cmd,'parent-compile')
        r['parent_object_sha256']=sha(HERE/'reference/repack-x86.cpp.o');r['parent_object_exact']=r['parent_object_sha256']==sha(OBJ)
        assert r['parent_object_exact'],'Parent x86 object is not byte exact'
        line=next(x for x in (WIDE/'rebuild.sh').read_text().splitlines() if x.startswith('/usr/bin/c++ -fPIC'))
        link=shlex.split(line.replace('$OUT',str(WIDE/'build')))
        provenance=json.loads((WIDE/'build-provenance.json').read_text())
        for value in link:
            if value.endswith('.o'):assert sha(value)==provenance[value],value
        link[link.index('-o')+1]=str(HERE/'reference/libggml-cpu.so.0.22.0')
        link[link.index(str(OBJ))]=str(HERE/'reference/repack-x86.cpp.o')
        run(link,'parent-link')
        r['parent_library_sha256']=sha(HERE/'reference/libggml-cpu.so.0.22.0')
        r['parent_library_exact']=r['parent_library_sha256']==sha(CPU);assert r['parent_library_exact']
        source=SOURCE.read_text()
        needle='''void ggml_gemv_q8_0_x16_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
'''
        assert source.count(needle)==1
        source=source.replace(needle,'''#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
#include "q8-batch-fast.h"
#endif

'''+needle+'''    if (nr >= 2 && nr <= 4 && nc == 16 && glm_q8_batch_fast_enabled()) {
        glm_q8_batch_fast_count(nr);
        switch (nr) {
            case 2: return glm_q8_batch_candidate<2,2,2>(n,s,bs,vx,vy,nr,nc);
            case 3: return glm_q8_batch_candidate<3,2,2>(n,s,bs,vx,vy,nr,nc);
            case 4: return glm_q8_batch_candidate<4,2,2>(n,s,bs,vx,vy,nr,nc);
        }
    }
''',1)
        (HERE/'candidate/repack-x86.cpp').write_text(source)
        (HERE/'q8-batch-fast.patch').write_text(''.join(difflib.unified_diff(SOURCE.read_text().splitlines(True),source.splitlines(True),fromfile=str(SOURCE),tofile='candidate/repack-x86.cpp')))
        candidate_compile=compile_cmd[:]
        candidate_compile[1:1]=['-I'+str(HERE),'-I'+str(OLD/'private-cpu')]
        candidate_compile[candidate_compile.index('-o')+1]=str(HERE/'candidate/repack-x86.cpp.o')
        candidate_compile[-1]=str(HERE/'candidate/repack-x86.cpp')
        run(candidate_compile,'candidate-compile')
        candidate_link=link[:];candidate_link[candidate_link.index('-o')+1]=str(HERE/'candidate/libggml-cpu.so.0.22.0')
        candidate_link[candidate_link.index(str(HERE/'reference/repack-x86.cpp.o'))]=str(HERE/'candidate/repack-x86.cpp.o')
        run(candidate_link,'candidate-link')
        for folder in ['reference','candidate']:
            p=HERE/folder;(p/'libggml-cpu.so.0').symlink_to('libggml-cpu.so.0.22.0');(p/'libggml-cpu.so').symlink_to('libggml-cpu.so.0')
        r['candidate_sha256']=sha(HERE/'candidate/libggml-cpu.so.0.22.0')
        r['input_sha256']={str(p):sha(p) for p in [Path(__file__),SOURCE,OBJ,HERE/'candidate/repack-x86.cpp',HERE/'q8-batch-fast.h',HERE/'q8-batch-candidates.h',CPU]}
        after=snapshot(pid,18131);guard()
        for k in ['proc_start_ticks','command','environment','mapped_libraries_sha256']:assert after[k]==before[k]
        r['production_unchanged']=True;r['passed']=True
        print('Exact parent and candidate built; production unchanged',flush=True)
    except BaseException as e:r['error']=repr(e);raise
    finally:r['finished']=time.time();checkpoint()

if __name__=='__main__':main()
