#!/usr/bin/env python3
"""Compare private Q8 gate/up tile sizes on the validated Flash expert graphs."""
import difflib
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time

from glm_flash_q8_trial import Manager, PORT
from model_measurement_guard import ModelMeasurementGuard
from qwen_split_trial import inference_snapshot, sha256

BASE = Path(__file__).resolve().parent
ENGINE = BASE.parents[1] / 'engines/llama.cpp-glm5n-goal-0904'
PINNED = ENGINE / 'validated-chunk16-bin'
OUT = BASE / 'results/glm-flash-q8-expert-tiles-0908'
PRIVATE = OUT / 'private-cpu'


def main():
    manager = Manager()
    current = manager.validate_current()
    assert current['pooling'] and current['q8_clamp'] and current['workers'] == 15 and current['drafts'] == 0
    guard = ModelMeasurementGuard(current['pid'], {current['pid']:PORT}, inference_snapshot)
    guard.assert_idle()
    OUT.mkdir(exist_ok=False); PRIVATE.mkdir()
    parent_path = BASE/'results/glm-flash-q8-pool-0908c/private-cpu/manifest.json'
    parent = json.loads(parent_path.read_text())
    assert parent['library_sha256'] == current['cpu_sha256'] == sha256(parent['library'])
    repack_manifest_path = BASE/'results/glm-flash-q8-clamp-0908/private-cpu/manifest.json'
    repack_manifest = json.loads(repack_manifest_path.read_text())
    command = list(repack_manifest['compile_command'])
    source_path = Path(command[-1])
    source = source_path.read_text()
    start = source.index('    bool forward_x16_moe_swiglu(')
    end = source.index('\n    bool compute_forward_mul_mat_swiglu(', start)
    body = source[start:end]
    marker = '''        // Dynamic work items (expert, 64-row tile): threads steal tiles, so a core that is
        // shared with another process does not stall the whole socket at the next barrier.
        constexpr int64_t tile = 64;'''
    assert body.count(marker) == 1
    body = body.replace(marker, '''        static const int64_t requested_tile = [] {
            const char * value = getenv("GGML_CPU_Q8_MOE_TILE_ROWS");
            if (value && strcmp(value, "32") == 0) return int64_t(32);
            if (value && strcmp(value, "48") == 0) return int64_t(48);
            return int64_t(64);
        }();
        constexpr int64_t max_tile = 64;
        const int64_t tile = sp.dst_type == GGML_TYPE_Q8_0 && k == 4096 &&
            n_out == 512 && n_experts == 288 ? requested_tile : max_tile;
        static std::atomic<bool> tile_logged{false};
        if (ith == 0 && tile != max_tile && !tile_logged.load(std::memory_order_relaxed) &&
                !tile_logged.exchange(true, std::memory_order_relaxed)) {
            GGML_LOG_INFO("Q8_MOE_TILE_ACTIVE rows=%lld\\n", (long long) tile);
        }''')
    marker = '        float gate_tmp[tile]; float up_tmp[tile];'
    assert body.count(marker) == 1
    body = body.replace(marker, '        float gate_tmp[max_tile]; float up_tmp[max_tile];')
    changed = source[:start] + body + source[end:]
    private_source = PRIVATE/'repack.cpp'; private_source.write_text(changed)
    (PRIVATE/'expert-tiles.patch').write_text(''.join(difflib.unified_diff(
        source.splitlines(True), changed.splitlines(True), fromfile=str(source_path), tofile=str(private_source))))
    old_object = command[command.index('-o')+1]
    assert old_object in parent['link_command']
    inputs = dict(parent['input_sha256'])
    inputs.update(parent['private_source_sha256'])
    fixtures = {
        'full': BASE/'results/glm-flash-q8-clamp-0908/expert-check',
        'prefill': BASE/'results/glm-flash-q8-clamp-prefill-0908/prefill',
    }
    fixture_source = BASE/'results/glm-flash-q8-clamp-0908/expert-check.cpp'
    for path in (parent_path, repack_manifest_path, Path(parent['library']), source_path,
                 Path(__file__), fixture_source, *fixtures.values()):
        if str(path) in inputs: assert inputs[str(path)] == sha256(path)
        inputs[str(path)] = sha256(path)
    for path in parent['link_command']:
        if path.endswith(('.o','.so.0.22.0','.a')) and Path(path).exists():
            if path in inputs: assert inputs[path] == sha256(path)
            inputs[path] = sha256(path)
    assert all(sha256(path)==digest for path,digest in inputs.items())
    (OUT/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    aux = fixture_source.read_text()
    marker = '        if (!moe || (shape.first == 512 && fused)) continue;'
    assert aux.count(marker) == 1
    aux = aux.replace(marker, '        if (!moe || !fused || shape.first != 4096) continue;')
    aux_path = OUT/'fused-check.cpp'; aux_path.write_text(aux)
    fixtures['aux'] = OUT/'fused-check'
    result = dict(started=time.time(),passed=False,parent_sha256=parent['library_sha256'],experts=288,
        tile_options=[32,48,64],steps=[],runs=[],comparisons=[],input_sha256=inputs,
        scope='Same-precision Q8 expert component comparisons and timings; no full-model speed or bandwidth claim.')
    cwd = ENGINE/'build-goal/ggml/src'

    def save(): (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')

    def run(cmd, label, environment=None):
        manager.validate_current(); guard.assert_idle()
        log_path = OUT/(label+'.log')
        with log_path.open('w') as log:
            proc = subprocess.Popen(cmd,cwd=cwd,env=environment,stdout=log,stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic()+600
                while proc.poll() is None:
                    guard.assert_idle()
                    assert time.monotonic() < deadline, label
                    time.sleep(0.5)
                assert proc.returncode == 0, (label,proc.returncode)
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try: proc.wait(timeout=10)
                    except subprocess.TimeoutExpired: proc.kill();proc.wait(timeout=10)
        result['steps'].append(dict(label=label,command=cmd,log=str(log_path)));save()
        print(json.dumps(dict(completed=label)),flush=True)
        return log_path.read_text()

    save()
    try:
        baseline_link = list(parent['link_command'])
        baseline_link[baseline_link.index('-o')+1] = str(PRIVATE/'baseline.so')
        run(baseline_link,'baseline-link')
        assert sha256(PRIVATE/'baseline.so') == parent['library_sha256']
        baseline_command = list(command)
        baseline_object = PRIVATE/'baseline-repack.o'
        baseline_command[baseline_command.index('-o')+1] = str(baseline_object)
        run(baseline_command,'baseline-compile')
        sections = []
        for obj,label in ((old_object,'original'),(str(baseline_object),'rebuilt')):
            section = PRIVATE/(label+'.text'); sections.append(section)
            run(['objcopy','--dump-section','.text='+str(section),obj,str(PRIVATE/(label+'.copy.o'))],label+'-text')
        assert sections[0].read_bytes() == sections[1].read_bytes()
        new_object = str(PRIVATE/'repack.cpp.o')
        command[command.index('-o')+1] = new_object
        command[-1] = str(private_source)
        run(command,'private-compile')
        library = PRIVATE/'libggml-cpu.so.0.22.0'
        link = [new_object if arg==old_object else arg for arg in parent['link_command']]
        link[link.index('-o')+1] = str(library)
        run(link,'private-link')
        (PRIVATE/'libggml-cpu.so.0').symlink_to(library.name)
        (PRIVATE/'libggml-cpu.so').symlink_to('libggml-cpu.so.0')
        manifest = dict(library=str(library),library_sha256=sha256(library),parent_manifest=str(parent_path),
            parent_sha256=parent['library_sha256'],input_sha256=inputs,private_source_sha256={str(private_source):sha256(private_source)},
            compile_command=command,link_command=link,baseline_link_identical=True,unpatched_text_identical=True,tile_options=[32,48,64])
        (PRIVATE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        result.update(library=str(library),library_sha256=sha256(library),baseline_link_identical=True,unpatched_text_identical=True)
        flags = ['c++','-O3','-std=c++17','-march=native','-fopenmp']
        flags += ['-I'+str(ENGINE/p) for p in ('ggml/include','ggml/src','ggml/src/ggml-cpu')]
        flags += [str(aux_path),'-L'+str(PINNED),'-Wl,-rpath,'+str(PINNED),'-lggml','-lggml-cpu','-lggml-base','-ldl','-o',str(fixtures['aux'])]
        run(flags,'aux-compile')
        result['fixture_sha256']={str(path):sha256(path) for path in [aux_path,*fixtures.values()]}
        environment = {k:v for k,v in os.environ.items() if not k.startswith(('GGML_','LLAMA_','OMP_','GOMP_','REPACK_TEST_'))}
        environment.update(GGML_CPU_X16_Q8_0='1',GGML_CPU_X16_Q8_BATCH='1',GGML_CPU_X16_CHUNK_MAX='16',
            GGML_CPU_X16_Q8_EXPERTS='1',GGML_CPU_MOE_CLAMP_FUSION='1',GGML_CPU_X16_Q8_CLAMP_FUSION='1',
            GGML_CPU_SOFTMAX_POOL_FUSION='1',REPACK_TEST_DOWN='1',REPACK_TEST_THREADS='15',
            REPACK_TEST_TIMING_MEDIAN='1',REPACK_TEST_PERSISTENT_POOL='1',REPACK_TEST_PIN_POOL='1')
        modes = []
        def add(label,tile=None,fixture='full',padded=False,clamped=True,consumer=False,threads=15,repeats=2,reference='parent-normal'):
            modes.append(dict(label=label,tile=tile,fixture=fixture,padded=padded,clamped=clamped,
                consumer=consumer,threads=threads,repeats=repeats,reference=reference))
        add('parent-normal',repeats=20)
        for index,tile in enumerate((64,32,48,48,32,64)):
            add(f'normal-{index}-tile{tile}',tile,repeats=20)
        for variant,extra in [('padded',dict(padded=True)),('unclamped',dict(clamped=False)),('consumer',dict(consumer=True))]:
            parent_label='parent-'+variant
            add(parent_label,fixture='aux',**extra)
            for tile in (32,48): add(variant+'-tile'+str(tile),tile,fixture='aux',reference=parent_label,**extra)
        for threads in (1,4):
            for tile in (32,48): add(f'threads{threads}-tile{tile}',tile,fixture='aux',threads=threads)
        add('parent-prefill',fixture='prefill',padded=True,repeats=1)
        for tile in (32,48): add('prefill-tile'+str(tile),tile,fixture='prefill',padded=True,repeats=1,reference='parent-prefill')
        references = {}
        for mode in modes:
            label=mode['label']; values=OUT/label; values.mkdir()
            env=dict(environment,REPACK_TEST_OUTPUT_DIR=str(values),REPACK_TEST_THREADS=str(mode['threads']),REPACK_TEST_REPEATS=str(mode['repeats']))
            cpu = Path(parent['library']) if mode['tile'] is None else library
            env['LD_LIBRARY_PATH']=str(cpu.parent)+':'+str(PINNED)
            if mode['tile'] is not None: env['GGML_CPU_Q8_MOE_TILE_ROWS']=str(mode['tile'])
            for key,flag in [('REPACK_TEST_PADDED',mode['padded']),('REPACK_TEST_CLAMP',mode['clamped']),('REPACK_TEST_CLAMP_CONSUMER',mode['consumer'])]:
                if flag: env[key]='1'
            mask=','.join(str(cpu) for cpu in range(48,48+mode['threads']))
            log=run(['taskset','-c',mask,str(fixtures[mode['fixture']]),'q8'],label,env)
            mapped,=re.findall(r'^CPU_LIBRARY (.+)$',log,re.M)
            assert Path(mapped).resolve()==cpu.resolve()
            rows=[dict(item.split('=',1) for item in line.split()[1:]) for line in log.splitlines() if line.startswith('PASS ')]
            assert len(rows)==(9 if mode['fixture']=='full' else 3), label
            active=re.findall(r'Q8_MOE_TILE_ACTIVE rows=(\d+)',log)
            expected=[str(mode['tile'])] if mode['tile'] in (32,48) and not mode['consumer'] and mode['threads']>1 else []
            assert active==expected,(label,active,expected)
            if mode['fixture']=='prefill': assert re.findall(r'^ACTIVE_EXPERTS (\d+)$',log,re.M)==['288']*6
            result['runs'].append(dict(mode,rows=rows,values=str(values),tile_marker=active))
            if mode['tile'] is None:
                references[label]=values
            else:
                reference=references[mode['reference']]
                for path in sorted(values.glob('*.f32')):
                    original=reference/path.name
                    a,b=original.read_bytes(),path.read_bytes()
                    exact=a==b
                    result['comparisons'].append(dict(run=label,case=path.stem,values=len(b)//4,bit_exact=exact,
                        reference_sha256=sha256(original),candidate_sha256=sha256(path)))
                    assert exact,(label,path.name,'output bits changed')
            save()
        assert len(result['comparisons'])==90
        assert all(sha256(path)==digest for path,digest in inputs.items())
        manager.validate_current();guard.assert_idle()
        result['passed']=True
    except BaseException as error:
        result['error']=repr(error)
        raise
    finally:
        result['finished']=time.time();save()


if __name__ == '__main__':
    with (BASE/'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
