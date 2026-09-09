#!/usr/bin/env python3
"""Compare validated tile sizes while rotating through all 288 stored experts."""
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time

from benchmark_qwen_q6 import wait_background
from glm_flash_q8_trial import Manager, PORT
from model_measurement_guard import ModelMeasurementGuard
from qwen_split_trial import inference_snapshot, sha256

BASE=Path(__file__).resolve().parent
BUILD=BASE/'results/glm-flash-q8-expert-tiles-0908'
OUT=BASE/'results/glm-flash-q8-expert-tiles-rotating-0908'
ENGINE=BASE.parents[1]/'engines/llama.cpp-glm5n-goal-0904'
PINNED=ENGINE/'validated-chunk16-bin'


def main():
    manager=Manager(); current=manager.validate_current()
    guard=ModelMeasurementGuard(current['pid'],{current['pid']:PORT},inference_snapshot)
    guard.assert_idle()
    build=json.loads((BUILD/'result.json').read_text())
    assert build['passed'] and all(x['bit_exact'] for x in build['comparisons'])
    candidate=Path(build['library']);parent=Path(current['cpu_library'])
    assert sha256(candidate)==build['library_sha256'] and sha256(parent)==build['parent_sha256']
    OUT.mkdir(exist_ok=False)
    source_path=BUILD/'fused-check.cpp'
    source=source_path.read_text()
    def replace(old,new):
        nonlocal source
        assert source.count(old)==1,old
        source=source.replace(old,new)
    replace(': std::vector<int>{1, 3, 4};',': std::vector<int>{1};')
    replace('    auto compute = [&]() {','''    int rotate_step = 0;
    std::vector<int32_t> rotating_routes(used * tokens);
    auto compute = [&]() {''')
    replace('        if (!iqk) return ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS;', '''        if (!iqk) {
            if (moe) {
                for (int t = 0; t < tokens; ++t) for (int j = 0; j < used; ++j) {
                    rotating_routes[t * used + j] = (j + 31 * t + 250 + 8 * rotate_step) % experts;
                }
                ggml_backend_tensor_set(ids, rotating_routes.data(), 0, ggml_nbytes(ids));
                rotate_step = (rotate_step + 1) % 36;
            }
            return ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS;
        }''')
    marker='        if (median_timing) samples.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - iteration_start).count());'
    replace(marker,marker+'''
        if (packed) {
            ggml_backend_tensor_get(y, result.values.data(), 0, ggml_nbytes(y));
            const char * dir = std::getenv("REPACK_TEST_ITER_OUTPUT_DIR");
            if (!dir) std::abort();
            std::ofstream output(std::string(dir) + "/iteration-" + std::to_string(i) + ".f32", std::ios::binary);
            output.write(reinterpret_cast<const char *>(result.values.data()), result.values.size() * sizeof(float));
            if (!output) std::abort();
        }''')
    fixture=OUT/'rotating-check.cpp';fixture.write_text(source)
    binary=OUT/'rotating-check'
    inputs={str(p):sha256(p) for p in (Path(__file__),source_path,BUILD/'result.json',parent,candidate,fixture)}
    result=dict(started=time.time(),passed=False,input_sha256=inputs,runs=[],comparisons=[],
        experts=288,active_experts=8,route_cycle=36,repeats=72,
        note='One-socket component timings with changing expert routes and output checks outside timing; not model throughput or DRAM-bandwidth measurements.')
    (OUT/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    def save(): (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    def run(cmd,label,env=None):
        guard.assert_idle()
        log_path=OUT/(label+'.log')
        with log_path.open('w') as log:
            proc=subprocess.Popen(cmd,env=env,cwd=BASE,stdout=log,stderr=subprocess.STDOUT)
            try:
                deadline=time.monotonic()+240
                while proc.poll() is None:
                    guard.assert_idle()
                    assert time.monotonic()<deadline,label
                    time.sleep(0.5)
                assert proc.returncode==0,(label,proc.returncode)
            finally:
                if proc.poll() is None:
                    proc.terminate();proc.wait(timeout=10)
        print(json.dumps(dict(completed=label)),flush=True)
        return log_path.read_text()
    save()
    try:
        flags=['c++','-O3','-std=c++17','-march=native','-fopenmp']
        flags+=['-I'+str(ENGINE/p) for p in ('ggml/include','ggml/src','ggml/src/ggml-cpu')]
        flags+=[str(fixture),'-L'+str(PINNED),'-Wl,-rpath,'+str(PINNED),'-lggml','-lggml-cpu','-lggml-base','-ldl','-o',str(binary)]
        result['compile_command']=flags;save();run(flags,'compile')
        result['binary_sha256']=sha256(binary)
        result['background_gate']=wait_background(guard,current['pid'],5.0,OUT/'background-wait.json')
        environment={k:v for k,v in os.environ.items() if not k.startswith(('GGML_','LLAMA_','OMP_','GOMP_','REPACK_TEST_'))}
        environment.update(GGML_CPU_X16_Q8_0='1',GGML_CPU_X16_Q8_BATCH='1',GGML_CPU_X16_CHUNK_MAX='16',
            GGML_CPU_X16_Q8_EXPERTS='1',GGML_CPU_MOE_CLAMP_FUSION='1',GGML_CPU_X16_Q8_CLAMP_FUSION='1',
            GGML_CPU_SOFTMAX_POOL_FUSION='1',REPACK_TEST_CLAMP='1',REPACK_TEST_DOWN='1',REPACK_TEST_THREADS='15',
            REPACK_TEST_REPEATS='72',REPACK_TEST_TIMING_MEDIAN='1',REPACK_TEST_PERSISTENT_POOL='1',REPACK_TEST_PIN_POOL='1')
        reference=None
        for index,tile in enumerate((None,64,32,48,48,32,64)):
            label=f'run-{index}-'+('parent' if tile is None else 'tile'+str(tile))
            directory=OUT/label;directory.mkdir()
            cpu=parent if tile is None else candidate
            env=dict(environment,LD_LIBRARY_PATH=str(cpu.parent)+':'+str(PINNED),
                REPACK_TEST_OUTPUT_DIR=str(directory),REPACK_TEST_ITER_OUTPUT_DIR=str(directory))
            if tile is not None: env['GGML_CPU_Q8_MOE_TILE_ROWS']=str(tile)
            log=run(['taskset','-c','48-62',str(binary),'q8'],label,env)
            mapped,=re.findall(r'^CPU_LIBRARY (.+)$',log,re.M);assert Path(mapped).resolve()==cpu.resolve()
            rows=[dict(item.split('=',1) for item in line.split()[1:]) for line in log.splitlines() if line.startswith('PASS ')]
            assert len(rows)==1 and rows[0]['tokens']=='1' and rows[0]['fused']=='1'
            active=re.findall(r'Q8_MOE_TILE_ACTIVE rows=(\d+)',log)
            assert active==([str(tile)] if tile in (32,48) else [])
            outputs=sorted(directory.glob('iteration-*.f32'));assert len(outputs)==72
            if reference is None: reference=directory
            else:
                for path in outputs:
                    original=reference/path.name
                    assert original.read_bytes()==path.read_bytes(),(label,path.name)
                    result['comparisons'].append(dict(run=label,iteration=path.stem,values=path.stat().st_size//4,
                        bit_exact=True,sha256=sha256(path)))
            result['runs'].append(dict(label=label,tile=tile,rows=rows,all_timed_outputs_saved=True));save()
        assert len(result['comparisons'])==432
        assert all(sha256(path)==digest for path,digest in inputs.items())
        manager.validate_current();guard.assert_idle();result['passed']=True
    except BaseException as error:
        result['error']=repr(error);raise
    finally:
        result['finished']=time.time();save()


if __name__ == '__main__':
    with (BASE/'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
