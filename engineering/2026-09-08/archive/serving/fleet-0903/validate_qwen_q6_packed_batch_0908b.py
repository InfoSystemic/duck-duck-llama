#!/usr/bin/env python3
"""Validate Q6 expert batching through CPU and four-socket graphs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time

from glm_flash_q8_trial import memory_status
from model_measurement_guard import ModelMeasurementGuard, read_service
from qwen_split_trial import inference_snapshot, process_info, sha256

BASE = Path(__file__).resolve().parent
ENGINE = BASE.parent.parent / 'engines/llama.cpp-q4e-goal-0904'
PINNED = ENGINE / 'validated-iq-batch3-bin'

AUDIT = r'''
    Dl_info runtime{};
    if (!dladdr(reinterpret_cast<void *>(ggml_backend_cpu_init), &runtime)) std::abort();
    std::printf("Q6_CPU_LIBRARY %s\n", runtime.dli_fname);
    std::atexit([] {
        using counter_fn = uint64_t (*)(int);
        const auto counter = reinterpret_cast<counter_fn>(dlsym(RTLD_DEFAULT, "ggml_cpu_q6_expert_batch_count"));
        std::printf("Q6_BATCH_CALLS %llu %llu\n", (unsigned long long) (counter ? counter(2) : 0),
                    (unsigned long long) (counter ? counter(3) : 0));
    });
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('label')
    args = parser.parse_args()
    assert re.fullmatch(r'qwen-q6-packed-batch-validation-[A-Za-z0-9_-]+',args.label)
    out = BASE / 'results' / args.label
    build_path = BASE / 'results/qwen-q6-packed-batch-build-0908b/result.json'
    manifest_path = build_path.parent / 'private-cpu/manifest.json'
    build,manifest = [json.loads(path.read_text()) for path in (build_path,manifest_path)]
    assert build['build_completed'] and build['baseline_link_identical'] and build['unpatched_text_identical']
    assert all(sha256(path) == digest for path,digest in build['input_sha256'].items())
    assert all(sha256(path) == digest for path,digest in manifest['private_source_sha256'].items())
    library = Path(manifest['library'])
    assert sha256(library) == build['library_sha256'] == manifest['library_sha256']
    parent = BASE / 'results/qwen-q6-q8-wide-batch-0907/private-cpu/libggml-cpu.so.0.22.0'
    assert sha256(parent) == build['parent_sha256']
    llama = BASE / 'results/qwen-expert-even-split-policy-0906b/private-split/libllama.so.0.3.0'
    assert sha256(llama) == 'd213ba477ef5d3f2a68568b30a31595ef9578b6dce4b3b81e95da6f83a3dac8b'
    preset_path = BASE / 'qwen-flash-20tps.json'
    preset = json.loads(preset_path.read_text())
    source_small = BASE / 'results/qwen-q6-simple-barrier-0907/q6-check.cpp'
    source_numa = BASE / 'results/qwen-q6-512-expert-validation-0907/numeric.cpp'
    sources = [Path(__file__).resolve(),build_path,manifest_path,library,parent,llama,preset_path,
               source_small,source_numa,BASE / 'model_measurement_guard.py',BASE / 'qwen_split_trial.py',BASE / 'glm_flash_q8_trial.py']
    result = dict(started=time.time(),controller_pid=os.getpid(),passed=False,checks=[],numa_checks=[],steps=[],model_loaded=False,
                  cpu_library=str(library),cpu_sha256=sha256(library),source_sha256={str(path):sha256(path) for path in sources},
                  scope='Graph correctness and actual batching-path checks; all component timings excluded from model claims.')
    assert process_info(1219506)['start'] == '103969952'
    guard = ModelMeasurementGuard(1219506,{1219506:18095},inference_snapshot)
    env = {k:v for k,v in os.environ.items() if not k.startswith(('GGML_','LLAMA_','OMP_','GOMP_','REPACK_TEST_','COLD_GRAPH_'))}
    env.update(preset['runtime_env'])
    env['GGML_CPU_X16_Q6_EXPERT_BATCH_AUDIT'] = '1'

    def save():
        (out / 'result.json').write_text(json.dumps(result,indent=2)+'\n')

    def run(command,label,environment,input_text=None):
        guard.assert_idle()
        assert process_info(1219506)['start'] == '103969952'
        assert memory_status()['MemAvailable'] > 32 << 30
        with (out / (label+'.log')).open('w') as log:
            child = subprocess.Popen(command,cwd=BASE,env=environment,stdout=log,stderr=subprocess.STDOUT,
                                     stdin=subprocess.PIPE if input_text else subprocess.DEVNULL,start_new_session=True)
            result['owned_component'] = dict(pid=child.pid,label=label)
            save()
            try:
                if input_text:
                    child.stdin.write(input_text.encode()); child.stdin.close()
                deadline = time.monotonic()+600
                while child.poll() is None:
                    guard.assert_idle()
                    assert time.monotonic() < deadline,label
                    time.sleep(.25)
                assert child.returncode == 0,(label,child.returncode)
            finally:
                if child.poll() is None:
                    child.terminate(); child.wait(timeout=10)
        result['steps'].append(dict(label=label,command=command,exit_code=child.returncode))
        save()
        print(json.dumps(dict(completed=label)),flush=True)
        return (out / (label+'.log')).read_text()

    def check_runtime(log,expected):
        loaded, = re.findall(r'^Q6_CPU_LIBRARY (.+)$',log,re.M)
        assert Path(loaded).resolve() == expected.resolve()
        counts, = re.findall(r'^Q6_BATCH_CALLS (\d+) (\d+)$',log,re.M)
        return list(map(int,counts))

    def compile_source(source,name):
        command = ['g++','-O3','-std=c++17','-march=native','-fopenmp']
        command += ['-I'+str(ENGINE / path) for path in ('include','src','ggml/include','ggml/src','ggml/src/ggml-cpu')]
        command += [str(source),'-L'+str(PINNED),'-lllama','-lggml','-lggml-cpu','-lggml-base','-ldl','-pthread','-o',str(out / name)]
        if name == 'numa-check':
            command[1:1] = ['-DQWEN_MOE_CHECK','-DQWEN_Q6_CHECK']
        run(command,'compile-'+name,env)

    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        out.mkdir(exist_ok=False)
        save()
        try:
            result['idle_gate'] = guard.wait_idle(out / 'idle.json',quiet_seconds=30)
            small = source_small.read_text()
            marker = 'int main(int argc, char ** argv) {'
            assert small.count(marker) == 1
            small = small.replace(marker,marker+AUDIT)
            old_shapes = '{{256, 64}, {768, 160}, {1536, 256}, {2560, 320}, {6144, 64}, {10240, 64}}'
            assert small.count(old_shapes) == 1
            small = small.replace(old_shapes,'{{2560, 160}}')
            small_path = out / 'q6-check.cpp'; small_path.write_text(small)
            compile_source(small_path,'q6-check')
            graph_env = {k:v for k,v in env.items() if not k.startswith('GGML_CPU_NUMA_')}
            graph_env.update(REPACK_TEST_WORK_SHARING='1',REPACK_TEST_SMALL_BATCHES='1',REPACK_TEST_REPEATS='1')
            for arm,enabled,unfused,is_parent in [('parent',False,False,True),('off',False,False,False),('on',True,False,False),
                                                 ('parent-unfused',False,True,True),('on-unfused',True,True,False)]:
                expected = parent if is_parent else library
                for padded in (False,True):
                    trial = dict(graph_env,LD_LIBRARY_PATH=str(expected.parent)+':'+str(PINNED),
                                 GGML_CPU_X16_Q6_EXPERT_BATCH=str(int(enabled)),GGML_CPU_DISABLE_FUSION=str(int(unfused)))
                    if padded: trial['REPACK_TEST_PADDED']='1'
                    label = arm+'-padded'+str(int(padded))
                    log = run(['taskset','-c','0-127',str(out / 'q6-check'),'q8'],label,trial)
                    counts = check_runtime(log,expected)
                    hashes = re.findall(r'^PASS .* hash=([^\n]+)',log,re.M)
                    assert len(hashes)==36 and ((min(counts)>0) if enabled else counts==[0,0]),(label,len(hashes),counts)
                    result['checks'].append(dict(label=arm,padded=padded,unfused=unfused,cases=len(hashes),hashes=hashes,batch_calls=counts,log_sha256=sha256(out / (label+'.log'))))
                    save()
            for padded in (False,True):
                for unfused,length in ((False,3),(True,2)):
                    group=[row for row in result['checks'] if row['padded']==padded and row['unfused']==unfused]
                    assert len(group)==length and all(row['hashes']==group[0]['hashes'] for row in group)
            result['small_graph_bit_exact']=True
            save()
            numa = source_numa.read_text()
            assert numa.count(marker)==1
            numa = numa.replace(marker,marker+AUDIT)
            old='(u + 10 * t + (m % 2 ? 490 : 250) + probe) % experts'
            assert numa.count(old)==1
            numa = numa.replace(old,'(u + route_step * t + (m % 2 ? 508 : 250) + probe) % experts')
            anchor='    const int copies = std::stoi(std::getenv("COLD_GRAPH_MATRICES"));'
            assert numa.count(anchor)==1
            numa = numa.replace(anchor,anchor+'\n    const int route_step = std::stoi(std::getenv("COLD_GRAPH_ROUTE_STEP"));\n    if (route_step != 2 && route_step != 10) return 2;')
            source = out / 'numa-check.cpp'; source.write_text(numa)
            compile_source(source,'numa-check')
            for tokens,step,unfused in [(1,2,False),(3,2,False),(5,2,False),(64,10,False),(5,2,True)]:
                pair=[]
                for enabled in (False,True):
                    expected=library if enabled else parent
                    label=f'numa-t{tokens}-step{step}-unfused{int(unfused)}-on{int(enabled)}'
                    output=out / (label+'.f32')
                    trial=dict(env,LD_LIBRARY_PATH=str(expected.parent)+':'+str(llama.parent)+':'+str(PINNED),
                               GGML_CPU_X16_Q6_EXPERT_BATCH=str(int(enabled)),GGML_CPU_DISABLE_FUSION=str(int(unfused)),
                               GGML_CPU_MOE_GATE_UP_FUSION=str(int(not unfused)),GGML_Q4E_EXPERT_EVEN_SPLIT='1',
                               COLD_GRAPH_K_PER_SOCKET='2560',COLD_GRAPH_ROWS='640',COLD_GRAPH_MATRICES='4',
                               COLD_GRAPH_TOKENS=str(tokens),COLD_GRAPH_ROUTE_STEP=str(step),COLD_GRAPH_OUTPUT_PATH=str(output))
                    log=run(['taskset','-c','0-127',str(out / 'numa-check'),'32','15','fixture','3'],label,trial,'exit\n')
                    counts=check_runtime(log,expected)
                    ready,=[json.loads(line) for line in log.splitlines() if line.startswith('{') and json.loads(line).get('event')=='ready']
                    assert ready['experts']==512 and ready['down_checked'] and ready['weight_type']=='q6_K'
                    assert Path(ready['policy_library']).resolve()==llama.resolve()
                    assert output.stat().st_size==3*4*10*tokens*2560*4
                    assert ((sum(counts)>0) if enabled and tokens>1 else counts==[0,0]),(label,counts)
                    if enabled and tokens in (3,5): assert min(counts)>0
                    pair.append(sha256(output))
                    result['numa_checks'].append(dict(label=label,tokens=tokens,route_step=step,unfused=unfused,enabled=enabled,
                                                     output_sha256=pair[-1],reference=ready,batch_calls=counts))
                    save()
                assert pair[0]==pair[1],(tokens,step,unfused)
            assert all(sha256(path)==digest for path,digest in result['source_sha256'].items())
            result.update(passed=True,bit_exact=True,fixture_sha256={str(path):sha256(path) for path in (small_path,source)},
                          binary_sha256={name:sha256(out / name) for name in ('q6-check','numa-check')},peer_after=read_service(18095))
        except BaseException as error:
            result['error']=repr(error)
            raise
        finally:
            result['finished']=time.time()
            result['peer_preserved']=process_info(1219506)['start']=='103969952'
            save()
            print(json.dumps(dict(passed=result['passed'],error=result.get('error'))),flush=True)


if __name__=='__main__':
    main()
