#!/usr/bin/env python3
"""Build and check a private Q8 sum path for 16-row, single-token calls."""
import difflib
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time
from glm_flash_q8_trial import BASE,Manager,PORT
from model_measurement_guard import ModelMeasurementGuard
from qwen_split_trial import inference_snapshot,sha256

OUT=BASE/'results/glm-flash-q8-sum16-0908'
PRIVATE=OUT/'private-cpu'
ENGINE=BASE.parents[1]/'engines/llama.cpp-glm5n-goal-0904'

def main():
    os.umask(0o077)
    manager=Manager();current=manager.validate_current()
    guard=ModelMeasurementGuard(current['pid'],{current['pid']:PORT},inference_snapshot)
    guard.assert_idle()
    parent_path=BASE/'results/glm-flash-q8-pool-0908c/private-cpu/manifest.json'
    original_path=BASE/'results/glm-flash-q8-batch-0908/private-cpu/manifest.json'
    probe_path=BASE/'results/glm-flash-q8-sums-0908/result.json'
    parent=json.loads(parent_path.read_text());original=json.loads(original_path.read_text())
    probe=json.loads(probe_path.read_text())
    assert probe['passed'] and all(sha256(p)==h for p,h in probe['input_sha256'].items())
    assert sha256(parent['library'])==parent['library_sha256']==current['cpu_sha256']==probe['cpu_sha256']
    assert all(sha256(p)==h for p,h in original['private_source_sha256'].items())
    command=list(original['compile_commands'][1]);source_path=Path(command[-1])
    old_object=command[command.index('-o')+1]
    assert old_object in parent['link_command']
    source=source_path.read_text()
    marker='void ggml_gemv_q8_0_x16_q8_0('
    assert source.count(marker)==1
    source=source.replace(marker,'#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)\n#include "flash-q8-sum16.h"\n#endif\n\n'+marker)
    marker='    GGML_ASSERT(nr == 1 && n % QK8_0 == 0 && nc % 16 == 0);\n'
    assert source.count(marker)==1
    replacement=marker+'''    static const bool fast_sum = [] {
        const char * value = std::getenv("GGML_CPU_Q8_FAST_SUM");
        return value && std::strcmp(value, "1") == 0;
    }();
    if (fast_sum && nc == 16) return flash_q8_sum_candidate<2>(n, s, bs, vx, vy, nr, 16);
'''
    source=source.replace(marker,replacement)
    header_path=probe_path.parent/'q8-sums-candidates.h'
    header=header_path.read_text()
    header=header.replace('#define GGML_X16_BC32(p) _mm512_set1_epi32(*(const int32_t *)(p))\n','')
    header=header.replace('#undef GGML_X16_BC32\n','')
    header=header.replace('candidate_q8','flash_q8_sum_candidate').replace('q8_activation_sum','flash_q8_activation_sum')
    assert 'GGML_X16_BC32' in header and '#define GGML_X16_BC32' not in header
    batch_header=source_path.parent/'qwen-q8-batch.h'
    inputs={str(p):sha256(p) for p in (Path(__file__),parent_path,original_path,probe_path,header_path,source_path,batch_header,Path(parent['library']),BASE/'glm_flash_q8_trial.py')}
    for value in parent['link_command']:
        if value.endswith(('.o','.a','.so.0.22.0')) and Path(value).exists():inputs[value]=sha256(value)
    for root in ('ggml/src','ggml/include'):
        for p in (ENGINE/root).rglob('*.h'):inputs[str(p)]=sha256(p)
    OUT.mkdir(exist_ok=False);PRIVATE.mkdir()
    (PRIVATE/'repack-x86.cpp').write_text(source)
    (PRIVATE/'flash-q8-sum16.h').write_text(header)
    (PRIVATE/batch_header.name).write_bytes(batch_header.read_bytes())
    (PRIVATE/'repack-x86.cpp.patch').write_text(''.join(difflib.unified_diff(source_path.read_text().splitlines(True),source.splitlines(True))))
    (OUT/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    result=dict(started=time.time(),passed=False,parent_sha256=parent['library_sha256'],input_sha256=inputs,steps=[],checks=[])
    cwd=ENGINE/'build-goal/ggml/src'
    def save():(OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    def run(args,label,env=None):
        manager.validate_current();guard.assert_idle()
        log=OUT/(label+'.log')
        with log.open('w') as stream:
            process=subprocess.Popen(args,cwd=cwd,env=env,stdout=stream,stderr=subprocess.STDOUT)
            try:
                deadline=time.monotonic()+600
                while process.poll() is None:
                    guard.assert_idle();assert time.monotonic()<deadline,label;time.sleep(.5)
                assert process.returncode==0,(label,process.returncode)
            finally:
                if process.poll() is None:process.terminate();process.wait(timeout=10)
        result['steps'].append(dict(label=label,command=args));save()
        print(json.dumps(dict(completed=label)),flush=True)
        return log.read_text()
    save()
    try:
        link=list(parent['link_command'])
        link[link.index('-o')+1]=str(PRIVATE/'parent-link.so')
        run(link,'parent-link')
        assert sha256(PRIVATE/'parent-link.so')==parent['library_sha256']
        baseline=PRIVATE/'baseline.o';command[command.index('-o')+1]=str(baseline)
        run(command,'baseline-compile')
        for name,path in [('original',old_object),('rebuilt',str(baseline))]:
            run(['objcopy','--dump-section','.text='+str(PRIVATE/(name+'.text')),path,str(PRIVATE/(name+'.copy.o'))],name+'-text')
        assert (PRIVATE/'original.text').read_bytes()==(PRIVATE/'rebuilt.text').read_bytes()
        command[-1]=str(PRIVATE/'repack-x86.cpp')
        command[command.index('-o')+1]=str(PRIVATE/'repack-x86.cpp.o')
        run(command,'private-compile')
        library=PRIVATE/'libggml-cpu.so.0.22.0'
        link[link.index(old_object)]=command[command.index('-o')+1]
        link[link.index('-o')+1]=str(library);run(link,'private-link')
        (PRIVATE/'libggml-cpu.so.0').symlink_to(library.name)
        (PRIVATE/'libggml-cpu.so').symlink_to('libggml-cpu.so.0')
        manifest=dict(library=str(library),library_sha256=sha256(library),parent_manifest=str(parent_path),
            parent_sha256=parent['library_sha256'],input_sha256=inputs,compile_command=command,link_command=link,
            private_source_sha256={str(PRIVATE/name):sha256(PRIVATE/name) for name in ('repack-x86.cpp','flash-q8-sum16.h','qwen-q8-batch.h')},
            baseline_link_identical=True,unpatched_text_identical=True,gate='GGML_CPU_Q8_FAST_SUM=1; NR=1 and NC=16 only')
        (PRIVATE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        result.update(library=str(library),library_sha256=sha256(library),baseline_link_identical=True,unpatched_text_identical=True)
        pinned=Path(current['pinned_directory'])
        env={k:v for k,v in os.environ.items() if not k.startswith(('GGML_','LLAMA_','OMP_','GOMP_','REPACK_TEST_'))}
        env.update(LD_LIBRARY_PATH=str(PRIVATE)+':'+str(pinned),GGML_CPU_X16_Q8_0='1',GGML_CPU_X16_Q8_BATCH='1',
            GGML_CPU_SINGLE_TASK_MAX_ELEMENTS='4096',GGML_CPU_X16_CHUNK_MAX='16',REPACK_TEST_SMALL_BATCHES='1',REPACK_TEST_REPEATS='1')
        direct=probe_path.parent/'q8-sums-check'
        assert sha256(direct)==probe['binary_sha256']
        inputs[str(direct)]=sha256(direct)
        for enabled in ('0','1'):
            output=OUT/f'direct-{enabled}.bin'
            log=run(['taskset','-c','48',str(direct),str(output)],'direct-'+enabled,dict(env,GGML_CPU_Q8_FAST_SUM=enabled))
            events=[json.loads(line) for line in log.splitlines() if line.startswith('{')]
            assert Path(events[0]['path']).resolve()==library.resolve()
            assert events[-1]==dict(event='done',passed=True)
            assert output.read_bytes()==(probe_path.parent/'outputs.bin').read_bytes()
            result['checks'].append(dict(kind='direct',enabled=enabled,output_bytes=output.stat().st_size,output_sha256=sha256(output),events=events))
            save()
        paths_root=BASE/'results/glm-flash-q8-dense-chunk-paths-0908'
        paths=json.loads((paths_root/'result.json').read_text());assert paths['passed']
        inputs[str(paths_root/'result.json')]=sha256(paths_root/'result.json')
        for kind in ('attention','fused'):
            binary=paths_root/kind;inputs[str(binary)]=sha256(binary)
            for reference in (r for r in paths['runs'] if r['kind']==kind and r['chunk']==16):
                expected_path=paths_root/(reference['label']+'.bin')
                assert sha256(expected_path)==reference['output_sha256']
                inputs[str(expected_path)]=sha256(expected_path)
                for enabled in ('0','1'):
                    label=reference['label']+'-sum'+enabled;output=OUT/(label+'.bin')
                    trial=dict(env,GGML_CPU_Q8_FAST_SUM=enabled,REPACK_TEST_DUMP=str(output))
                    if reference['threads']:trial['REPACK_TEST_THREADS']=str(reference['threads'])
                    if reference['padded']:trial['REPACK_TEST_PADDED']='1'
                    log=run(['taskset','-c','48-62',str(binary)]+(['q8'] if kind=='fused' else []),label,trial)
                    loaded,=re.findall(r'^CPU_LIBRARY (.+)$',log,re.M)
                    assert Path(loaded).resolve()==library.resolve()
                    marker='Attention: 168 cases, 0 failures' if kind=='attention' else 'Repacking: 6 cases, 0 failures'
                    assert marker in log and output.read_bytes()==expected_path.read_bytes()
                    result['checks'].append(dict(kind=kind,enabled=enabled,cases=reference['cases'],threads=reference['threads'],padded=reference['padded'],
                        output_bytes=output.stat().st_size,output_sha256=sha256(output),reference=str(expected_path)))
                    save()
        assert all(sha256(p)==digest for p,digest in inputs.items())
        manager.validate_current();guard.assert_idle()
        result['passed']=True
        print(json.dumps(dict(passed=True,library_sha256=sha256(library),checks=len(result['checks']))),flush=True)
    except BaseException as error:
        result['error']=repr(error);raise
    finally:
        result['finished']=time.time();save()

if __name__=='__main__':
    with (BASE/'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
