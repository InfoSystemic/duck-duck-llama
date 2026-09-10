#!/usr/bin/env python3
"""Build a timing fixture that gives both HC arms the same initial workspace capacity."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from model_measurement_guard import ModelMeasurementGuard
from qwen_split_trial import inference_snapshot,process_info,sha256

BASE = Path(__file__).resolve().parent
ENGINE = BASE.parents[1]/'engines/llama.cpp-q4e-goal-0904'
OUT = BASE/'results/qwen-hc-reserved-fixture-0909'


def main():
    assert os.sched_getaffinity(0)=={127}
    assert process_info(1219506)['start']=='103969952'
    source_path = BASE/'time-qwen-hc-ordered-k-0909.cpp'
    runtime_path = BASE/'results/qwen-get-rows-runtime-0909/result.json'
    source = source_path.read_text()
    old = '    for (int i = 0; i < 5; ++i) if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) std::abort();'
    assert source.count(old)==1
    source = source.replace(old,'''    const auto plan = ggml_graph_plan(graph,15,nullptr);
    static size_t workspace_max = 0;
    workspace_max = std::max(workspace_max,plan.work_size);
''' + old)
    old = r'\"weight_bytes\":%zu,\"samples\":40'
    assert source.count(old)==1
    source = source.replace(old,r'\"weight_bytes\":%zu,\"planned_work_bytes\":%zu,\"workspace_max_bytes\":%zu,\"samples\":40')
    old = 'nc,nr,count,ggml_nbytes(weights[0])*count,(times[19]+times[20])/2'
    assert source.count(old)==1
    source = source.replace(old,'nc,nr,count,ggml_nbytes(weights[0])*count,plan.work_size,workspace_max,(times[19]+times[20])/2')
    old = '    for (int count : {1,96}) for (int nc : {64,96}) for (int nr : {1,4,5}) run(nc,nr,count,backend);'
    assert source.count(old)==1
    source = source.replace(old,'    run(8,64,1,backend);\n'+old)
    runtime = json.loads(runtime_path.read_text())
    assert runtime['passed'] and runtime['finished']
    for key in ('cpu','base','server','llama'):
        assert sha256(runtime[key])==runtime[key+'_sha256']
    guard = ModelMeasurementGuard(1219506,{1219506:18095},inference_snapshot)
    with (BASE/'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        guard.assert_idle()
        OUT.mkdir(exist_ok=False)
        target = OUT/'hc-reserved-timing.cpp'
        target.write_text(source)
        binary = OUT/'hc-reserved-timing'
        flags = ['/usr/bin/c++','-O3','-std=c++17','-march=native','-fopenmp','-DGGML_USE_OPENMP']
        flags += ['-I'+str(ENGINE/p) for p in ('ggml/include','ggml/src','ggml/src/ggml-cpu')]
        command = [*flags,str(target),'-L'+runtime['runtime_directory'],'-Wl,-rpath,'+runtime['runtime_directory'],
                   '-lggml','-lggml-cpu','-lggml-base','-ldl','-o',str(binary)]
        inputs = [Path(__file__).resolve(),source_path,runtime_path,Path(runtime['cpu']),Path(runtime['base']),
                  BASE/'model_measurement_guard.py',BASE/'qwen_split_trial.py']
        inputs += list((ENGINE/'ggml/include').glob('*.h'))+list((ENGINE/'ggml/src/ggml-cpu').glob('*.h'))
        result = dict(started=time.time(),passed=False,model_loaded=False,command=command,
                      input_sha256={str(path):sha256(path) for path in inputs},source=str(target),source_sha256=sha256(target),
                      scope='Same timing workload after an identical eight-output, 64-token graph reserves CPU workspace in each arm. Report planned and retained maximum work bytes. This tests a workspace-history hypothesis; it does not establish its cause or model performance.')
        try:
            checked = subprocess.run(command,cwd=OUT,capture_output=True,text=True,timeout=60)
            (OUT/'compile.log').write_text(checked.stdout+checked.stderr)
            result['exit_code'] = checked.returncode
            assert checked.returncode==0
            guard.assert_idle()
            assert all(sha256(path)==digest for path,digest in result['input_sha256'].items())
            result.update(passed=True,binary=str(binary),binary_sha256=sha256(binary))
            print(json.dumps(dict(passed=True,binary=str(binary),binary_sha256=sha256(binary))),flush=True)
        except BaseException as error:
            result['error'] = repr(error)
            raise
        finally:
            result.update(finished=time.time(),peer_preserved=process_info(1219506)['start']=='103969952')
            (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    os.umask(0o077)
    main()
