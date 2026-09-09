#!/usr/bin/env python3
"""Build guarded SIMD RMS means on top of the retained sum16 runtime."""
import difflib,fcntl,json,os
from pathlib import Path
import re,subprocess,time
from glm_flash_q8_trial import BASE,Manager,PORT
from model_measurement_guard import ModelMeasurementGuard
from qwen_split_trial import inference_snapshot,sha256

OUT=BASE/'results/glm-flash-rms-guard-0908'
PRIVATE=OUT/'private-cpu'
ENGINE=BASE.parents[1]/'engines/llama.cpp-glm5n-goal-0904'

def main():
    os.umask(0o077)
    manager=Manager();current=manager.validate_current()
    guard=ModelMeasurementGuard(current['pid'],{current['pid']:PORT},inference_snapshot);guard.assert_idle()
    assert current['sum16'] and current['drafts']==0
    parent_path=BASE/'results/glm-flash-q8-sum16-0908/private-cpu/manifest.json'
    ops_path=BASE/'results/glm-flash-q8-pool-0908c/private-cpu/manifest.json'
    probe_path=BASE/'results/glm-flash-rms-guard-probe-0908/result.json'
    parent=json.loads(parent_path.read_text());ops=json.loads(ops_path.read_text());probe=json.loads(probe_path.read_text())
    assert probe['passed'] and all(sha256(p)==h for p,h in probe['input_sha256'].items())
    assert sha256(parent['library'])==parent['library_sha256']==current['cpu_sha256']==probe['cpu_sha256']
    original_command=list(ops['compile_commands'][1]);source_path=Path(original_command[-1])
    old_object=original_command[original_command.index('-o')+1];assert old_object in parent['link_command']
    original=source_path.read_text()
    helper='''#include <atomic>
#include "flash-rms-guarded-0908.h"

static std::atomic<uint64_t> flash_rms_guard_counts[FLASH_RMS_GUARD_STATUS_COUNT];
extern "C" uint64_t ggml_cpu_rms_guard_count(int status);
extern "C" uint64_t ggml_cpu_rms_guard_count(int status) {
    return status >= 0 && status < FLASH_RMS_GUARD_STATUS_COUNT ? flash_rms_guard_counts[status].load(std::memory_order_relaxed) : 0;
}
static bool flash_rms_try_guarded(const ggml_compute_params * params, const float * x, int64_t n, float * mean) {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_CPU_RMS_F64_SIMD");
        return value && std::strcmp(value,"1") == 0;
    }();
    static const bool audit = [] {
        const char * value = std::getenv("GGML_CPU_RMS_F64_AUDIT");
        return value && std::strcmp(value,"1") == 0;
    }();
    if (!enabled || params->use_ref) return false;
    const auto status = flash_rms_guarded_mean(x,n,mean);
    if (audit) flash_rms_guard_counts[status].fetch_add(1,std::memory_order_relaxed);
    return status == FLASH_RMS_GUARD_OK;
}

'''
    marker='bool ggml_cpu_parallel_copy_enabled(void) {'
    assert original.count(marker)==1
    changed=original.replace(marker,helper+marker)
    old='''                ggml_float sum = 0.0;
                // worth switching to explicit SIMD?
                for (int64_t i00 = 0; i00 < ne00; i00++) {
                    sum += (ggml_float)(x[i00] * x[i00]);
                }

                const float mean  = sum/ne00;'''
    new='''                float mean;
                if (!flash_rms_try_guarded(params,x,ne00,&mean)) {
                    ggml_float sum = 0.0;
                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        sum += (ggml_float)(x[i00] * x[i00]);
                    }
                    mean = sum/ne00;
                }'''
    assert changed.count(old)==1;changed=changed.replace(old,new)
    fixture_path=BASE/'flash-rms-matmul-check-0908.cpp';fixture=fixture_path.read_text()
    fixture=fixture.replace('bool weighted = false, swapped = false, per_row_scale = false;', 'bool weighted = false, swapped = false, per_row_scale = false, boundary = false;')
    marker='    bool good = true;'
    assert fixture.count(marker)==1
    fixture=fixture.replace(marker,'''    const auto rms_count = reinterpret_cast<count_fn>(dlsym(RTLD_DEFAULT,"ggml_cpu_rms_guard_count"));
    const bool rms_audit = rms_count && enabled("GGML_CPU_RMS_F64_AUDIT");
    uint64_t accepted_rows = 0, boundary_rows = 0;
'''+marker)
    marker='                    row[k] = values[ix] = sample == 2 ? 0.0f : float(int((ix*53+sample*71+17)%1021)-510)/128.0f;'
    assert fixture.count(marker)==1
    fixture=fixture.replace(marker,'''                    row[k] = values[ix] = c.boundary ? (k == c.k/2 ? 1.0f + float(c.k)*0x1p-25f : 1.0f)
                        : sample == 2 ? 0.0f : float(int((ix*53+sample*71+17)%1021)-510)/128.0f;''')
    marker='        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) good = false;'
    assert fixture.count(marker)==1
    fixture=fixture.replace(marker,'''        const uint64_t accepted_before = rms_audit ? rms_count(0) : 0;
        const uint64_t boundary_before = rms_audit ? rms_count(4) : 0;
'''+marker+'''
        const uint64_t used = rms_audit ? rms_count(0)-accepted_before : 0;
        const uint64_t fallback = rms_audit ? rms_count(4)-boundary_before : 0;
        if (rms_audit) {
            if (!enabled("GGML_CPU_RMS_F64_SIMD") || c.reference) {
                if (used || fallback) good = false;
            } else if (c.boundary) {
                if (used || !fallback) good = false;
            } else if (!used) good = false;
        }
        accepted_rows += used; boundary_rows += fallback;''')
    marker='    std::fflush(stdout);'
    assert fixture.count(marker)==1
    fixture=fixture.replace(marker,'''    std::printf("RMS_GUARD id=%d accepted=%llu boundary=%llu reference=%d boundary_input=%d\\\\n",
        id,(unsigned long long)accepted_rows,(unsigned long long)boundary_rows,c.reference,c.boundary);
'''.replace('\\\\n','\\n')+marker)
    marker='        if (std::fclose(dump)) ++failures;'
    assert fixture.count(marker)==1
    fixture=fixture.replace(marker,'''        for (bool weighted : {false,true}) for (int tokens : {1,3}) for (bool reference : {false,true}) {
            test_case c; c.k=4096; c.weighted=weighted; c.tokens=tokens; c.reference=reference; c.boundary=true;
            failures += !run_case(c,dump,false,cases++);
        }
'''+marker)
    marker='int main(int argc, char ** argv) {'
    assert fixture.count(marker)==1
    fixture=fixture.replace(marker,marker+'''
    Dl_info runtime{};
    if (!dladdr(reinterpret_cast<void *>(ggml_backend_cpu_init),&runtime)) std::abort();
    std::printf("CPU_LIBRARY %s\\\\n",runtime.dli_fname);
'''.replace('\\\\n','\\n'))
    inputs={str(p):sha256(p) for p in (Path(__file__),parent_path,ops_path,probe_path,source_path,fixture_path,BASE/'flash-rms-guarded-0908.h',Path(parent['library']))}
    for value in parent['link_command']:
        if value.endswith(('.o','.a','.so.0.22.0')) and Path(value).exists():inputs[value]=sha256(value)
    for root in ('ggml/src','ggml/include'):
        for p in (ENGINE/root).rglob('*.h'):inputs[str(p)]=sha256(p)
    OUT.mkdir(exist_ok=False);PRIVATE.mkdir()
    (PRIVATE/'ops.cpp').write_text(changed)
    (PRIVATE/'flash-rms-guarded-0908.h').write_bytes((BASE/'flash-rms-guarded-0908.h').read_bytes())
    (PRIVATE/'ops.cpp.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),changed.splitlines(True))))
    fixture_out=OUT/'rms-graph-check.cpp';fixture_out.write_text(fixture)
    inputs[str(fixture_out)]=sha256(fixture_out)
    for p in (Path(__file__),BASE/'glm_flash_q8_trial.py'):(OUT/p.name).write_bytes(p.read_bytes())
    result=dict(started=time.time(),passed=False,parent_sha256=parent['library_sha256'],input_sha256=inputs,steps=[],checks=[],timings=[])
    cwd=ENGINE/'build-goal/ggml/src'
    def save():(OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    def run(command,label,env=None):
        manager.validate_current();guard.assert_idle()
        log=OUT/(label+'.log')
        with log.open('w') as stream:
            process=subprocess.Popen(command,cwd=cwd,env=env,stdout=stream,stderr=subprocess.STDOUT)
            try:
                deadline=time.monotonic()+600
                while process.poll() is None:
                    guard.assert_idle();assert time.monotonic()<deadline,label;time.sleep(.5)
                assert process.returncode==0,(label,process.returncode)
            finally:
                if process.poll() is None:process.terminate();process.wait(timeout=10)
        result['steps'].append(dict(label=label,command=command));save();print(json.dumps(dict(completed=label)),flush=True)
        return log.read_text()
    save()
    try:
        link=list(parent['link_command']);link[link.index('-o')+1]=str(PRIVATE/'parent-link.so')
        run(link,'parent-link');assert sha256(PRIVATE/'parent-link.so')==parent['library_sha256']
        command=list(original_command);command[command.index('-o')+1]=str(PRIVATE/'baseline.o');run(command,'baseline-compile')
        for label,path in [('original',old_object),('rebuilt',str(PRIVATE/'baseline.o'))]:
            run(['objcopy','--dump-section','.text='+str(PRIVATE/(label+'.text')),path,str(PRIVATE/(label+'.copy.o'))],label+'-text')
        assert (PRIVATE/'original.text').read_bytes()==(PRIVATE/'rebuilt.text').read_bytes()
        command[-1]=str(PRIVATE/'ops.cpp');command[command.index('-o')+1]=str(PRIVATE/'ops.cpp.o');command[1:1]=['-I'+str(PRIVATE)]
        run(command,'private-compile')
        library=PRIVATE/'libggml-cpu.so.0.22.0'
        link[link.index(old_object)]=command[command.index('-o')+1];link[link.index('-o')+1]=str(library);run(link,'private-link')
        (PRIVATE/'libggml-cpu.so.0').symlink_to(library.name);(PRIVATE/'libggml-cpu.so').symlink_to('libggml-cpu.so.0')
        manifest=dict(library=str(library),library_sha256=sha256(library),parent_manifest=str(parent_path),parent_sha256=parent['library_sha256'],
            compile_command=command,link_command=link,input_sha256=inputs,private_source_sha256={str(p):sha256(p) for p in (PRIVATE/'ops.cpp',PRIVATE/'flash-rms-guarded-0908.h')},
            baseline_link_identical=True,unpatched_text_identical=True)
        (PRIVATE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        result.update(library=str(library),library_sha256=sha256(library),baseline_link_identical=True,unpatched_text_identical=True)
        pinned=Path(current['pinned_directory']);binary=OUT/'rms-graph-check'
        flags=['c++','-O3','-std=c++17','-march=native','-fopenmp']
        flags+=['-I'+str(ENGINE/p) for p in ('include','ggml/include','ggml/src','ggml/src/ggml-cpu')]
        flags+=[str(fixture_out),'-L'+str(PRIVATE),'-L'+str(pinned),'-Wl,-rpath,'+str(PRIVATE)+':'+str(pinned),'-lggml-cpu','-lggml-base','-ldl','-pthread','-o',str(binary)]
        run(flags,'fixture-compile');result['binary_sha256']=sha256(binary)
        env={k:v for k,v in os.environ.items() if not k.startswith(('GGML_','LLAMA_','OMP_','GOMP_','REPACK_TEST_'))}
        env.update(LD_LIBRARY_PATH=str(PRIVATE)+':'+str(pinned),GGML_CPU_SINGLE_TASK_MAX_ELEMENTS='4096',
            GGML_CPU_SOFTMAX_POOL_FUSION='1',GGML_CPU_X16_Q8_BATCH='1',GGML_CPU_Q8_FAST_SUM='1')
        outputs=[]
        for label,enabled,disabled,is_parent in [('parent','0','0',True),('off','0','0',False),('on','1','0',False),('no-fusion','1','1',False)]:
            trial=dict(env,GGML_CPU_RMS_F64_SIMD=enabled,GGML_CPU_RMS_F64_AUDIT='1',GGML_CPU_DISABLE_FUSION=disabled)
            expected_cpu=Path(parent['library']) if is_parent else library
            if is_parent:trial['LD_LIBRARY_PATH']=str(expected_cpu.parent)+':'+str(pinned)
            output=OUT/(label+'.bin')
            log=run(['taskset','-c','48-62',str(binary),str(output)],'check-'+label,trial)
            loaded,=re.findall(r'^CPU_LIBRARY (.+)$',log,re.M);assert Path(loaded).resolve()==expected_cpu.resolve()
            assert 'SUMMARY cases=94 failures=0' in log
            rows=re.findall(r'^RMS_GUARD id=(\d+) accepted=(\d+) boundary=(\d+) reference=(\d+) boundary_input=(\d+)$',log,re.M)
            assert len(rows)==94
            accepted=sum(int(x[1]) for x in rows);boundary=sum(int(x[2]) for x in rows)
            if enabled=='1':assert accepted>0 and boundary>0
            else:assert accepted==boundary==0
            outputs.append(output)
            result['checks'].append(dict(label=label,cases=94,samples_per_case=3,accepted_rows=accepted,boundary_rows=boundary,
                output_bytes=output.stat().st_size,output_sha256=sha256(output),cpu_library=loaded))
            save()
        assert all(path.read_bytes()==outputs[0].read_bytes() for path in outputs[1:]);result['bit_exact']=True
        for index,enabled in enumerate(('0','1','1','0')):
            log=run(['taskset','-c','48-62',str(binary),'--timing'],f'timing-{index}-{enabled}',dict(env,GGML_CPU_RMS_F64_SIMD=enabled))
            rows=re.findall(r'^PASS .* tokens=(\d+) .* weighted=(\d+) .* ms=([\d.]+)$',log,re.M)
            assert len(rows)==4
            result['timings'].append(dict(enabled=enabled=='1',graph_ms={f'{w}-{t}':float(ms) for t,w,ms in rows}))
        assert all(sha256(p)==digest for p,digest in inputs.items())
        manager.validate_current();guard.assert_idle();result['passed']=True
        print(json.dumps(dict(passed=True,library_sha256=sha256(library),checks=result['checks'],timings=result['timings'])),flush=True)
    except BaseException as error:
        result['error']=repr(error);raise
    finally:
        result['finished']=time.time();save()

if __name__=='__main__':
    with (BASE/'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main()
