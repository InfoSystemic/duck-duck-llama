#!/usr/bin/env python3
"""Check token-row HC work and the component-qualified 64-row raw path."""
import difflib
import fcntl
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time

from benchmark_qwen_q6 import wait_background
from model_measurement_guard import ModelMeasurementGuard, read_service
from qwen_hc_row_split_transform_0909 import transform, graph_fixture
from qwen_split_trial import inference_snapshot, process_info, runtime_environment, sha256

BASE = Path(__file__).resolve().parent
ENGINE = BASE.parents[1] / 'engines/llama.cpp-q4e-goal-0904'
OUT = BASE / 'results/qwen-hc-row-split-build-0909'


def main():
    assert os.sched_getaffinity(0) == {127}
    assert process_info(1219506)['start'] == '103969952'
    parent_path = BASE / 'results/qwen-get-rows-columns-build-0909/private-cpu/manifest.json'
    repack_path = BASE / 'results/qwen-q6-q8-wide-batch-0907/private-cpu/manifest.json'
    runtime_path = BASE / 'results/qwen-get-rows-runtime-0909/result.json'
    proof_path = BASE / 'results/qwen-hc-ordered-k-proof-0909/result.json'
    parent, repack, runtime, proof = [json.loads(p.read_text()) for p in (parent_path,repack_path,runtime_path,proof_path)]
    assert proof['passed'] and proof['finished'] and proof['summary'] == [dict(cases=96, failures=0)]
    assert runtime['passed'] and runtime['finished']
    assert sha256(parent['library']) == parent['library_sha256'] == runtime['cpu_sha256'] == '12c61b337736ca9210433f57c64ce7fffbf7e4b66920aba9a03f97eaef3fd9b7'
    inputs = {}
    for record in (parent,repack,runtime,proof):
        for key in ('input_sha256','private_source_sha256','sources'):
            for path, digest in record.get(key, {}).items():
                assert sha256(path) == digest, path
                inputs[path] = digest
    command, = [row for row in repack['compile_commands'] if row[-1].endswith('/repack.cpp')]
    original_path = Path(command[command.index('-c')+1])
    original_object = Path(command[command.index('-o')+1])
    assert sha256(original_object) == 'accfc5d7daceeadf97396d897f3db5b4a821c00daf592c03c7174660e4584283'
    assert parent['link_command'].count(str(original_object)) == 1
    original = original_path.read_text()
    changed = transform(original)
    source_fixture = BASE / 'check-qwen-hc-ordered-k-0909.cpp'
    source_timing = BASE / 'time-qwen-hc-ordered-k-0909.cpp'
    header = proof_path.parent / 'qwen-q8-hc-ordered-k-0909.h'
    paths = [Path(__file__).resolve(), BASE / 'qwen_hc_ordered_k_transform_0909.py', BASE / 'qwen_hc_row_split_transform_0909.py',
             BASE / 'build_qwen_hc_ordered_k_0909.py', parent_path, repack_path,
             runtime_path, proof_path, original_path, original_object, source_fixture, source_timing, header,
             BASE / 'model_measurement_guard.py', BASE / 'benchmark_qwen_q6.py', BASE / 'qwen_split_trial.py']
    paths += [Path(value) for value in parent['link_command'] if value.endswith(('.o','.a','.so.0.22.0')) and Path(value).is_file()]
    paths += list((ENGINE / 'ggml/src').rglob('*.h')) + list((ENGINE / 'ggml/include').rglob('*.h'))
    inputs.update({str(path):sha256(path) for path in paths})
    guard = ModelMeasurementGuard(1219506, {1219506:18095}, inference_snapshot)
    owned = None

    def interrupted(signum, frame):
        raise InterruptedError('Release only the owned HC build or component test')

    for signum in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):
        signal.signal(signum,interrupted)
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        guard.assert_idle()
        OUT.mkdir(exist_ok=False)
        private = OUT / 'private-cpu'
        private.mkdir()
        source = private / 'repack.cpp'
        source.write_text(changed)
        (private / header.name).write_bytes(header.read_bytes())
        (private / 'repack.cpp.patch').write_text(''.join(difflib.unified_diff(
            original.splitlines(True),changed.splitlines(True),fromfile=str(original_path),tofile='private/repack.cpp')))
        fixture = OUT / 'hc-graph-check.cpp'
        fixture.write_text(graph_fixture(source_fixture.read_text()))
        timing = OUT / source_timing.name
        timing.write_bytes(source_timing.read_bytes())
        result = dict(started=time.time(),passed=False,build_completed=False,model_loaded=False,
                      controller_pid=os.getpid(),input_sha256=inputs,
                      private_source_sha256={str(p):sha256(p) for p in (source,private/header.name,fixture,timing)},
                      parent_sha256=parent['library_sha256'],steps=[],checks=[],timings=[],
                      scope='Only repack.cpp.o changes. Opt-in token-row work for the observed 10240-input Q8 HC projections with 2-8 activation rows; ordered-K is limited to 64-output raw projections. Reuse existing GEMV arithmetic and preserve exact quantization and weights. Component results are not model throughput or bandwidth.')

        def save():
            (OUT / 'result.json').write_text(json.dumps(result,indent=2)+'\n')

        def run(args,name,env=None):
            nonlocal owned
            guard.assert_idle()
            log_path = OUT / (name+'.log')
            with log_path.open('w') as log:
                owned = subprocess.Popen(args,cwd=ENGINE/'build-goal',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                result['owned'] = dict(pid=owned.pid,step=name)
                save()
                deadline = time.monotonic()+600
                while owned.poll() is None:
                    guard.assert_idle()
                    assert time.monotonic()<deadline,name
                    time.sleep(.25)
            result['steps'].append(dict(name=name,command=args,exit_code=owned.returncode,log_sha256=sha256(log_path)))
            save()
            assert owned.returncode == 0,name
            print(json.dumps(dict(completed=name)),flush=True)
            return log_path.read_text()

        def checked_library(log,expected):
            paths = [line.removeprefix('CPU_LIBRARY ') for line in log.splitlines() if line.startswith('CPU_LIBRARY ')]
            assert len(paths)==1 and Path(paths[0]).resolve()==Path(expected).resolve()

        save()
        try:
            baseline_object = OUT / 'baseline-repack.cpp.o'
            baseline_compile = list(command)
            baseline_compile[baseline_compile.index('-o')+1] = str(baseline_object)
            run(baseline_compile,'baseline-compile')
            result['baseline_object_identical'] = sha256(baseline_object)==sha256(original_object)
            assert result['baseline_object_identical']
            baseline_library = OUT / 'baseline-libggml-cpu.so'
            link = [str(baseline_object) if value==str(original_object) else value for value in parent['link_command']]
            link[link.index('-o')+1] = str(baseline_library)
            run(link,'baseline-link')
            result['baseline_library_identical'] = sha256(baseline_library)==parent['library_sha256']
            assert result['baseline_library_identical']
            obj = private / 'repack.cpp.o'
            compile_command = list(command)
            compile_command[compile_command.index('-c')+1] = str(source)
            compile_command[compile_command.index('-o')+1] = str(obj)
            compile_command.insert(1,'-I'+str(original_path.parent))
            run(compile_command,'private-compile')
            library = private / 'libggml-cpu.so.0.22.0'
            link = [str(obj) if value==str(original_object) else value for value in parent['link_command']]
            link[link.index('-o')+1] = str(library)
            run(link,'private-link')
            (private/'libggml-cpu.so.0').symlink_to(library.name)
            (private/'libggml-cpu.so').symlink_to('libggml-cpu.so.0')
            result.update(build_completed=True,library=str(library),library_sha256=sha256(library))
            manifest = dict(input_sha256=inputs,private_source_sha256=result['private_source_sha256'],
                            parent_manifest=str(parent_path),parent_sha256=parent['library_sha256'],base=parent['base'],
                            library=str(library),library_sha256=sha256(library),compile_command=compile_command,link_command=link,
                            baseline_object_identical=True,baseline_library_identical=True,scope=result['scope'])
            (private/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            runtime_dir = Path(runtime['runtime_directory'])
            flags = ['/usr/bin/c++','-O3','-std=c++17','-march=native','-fopenmp','-DGGML_USE_OPENMP']
            flags += ['-I'+str(ENGINE/p) for p in ('ggml/include','ggml/src','ggml/src/ggml-cpu')]
            flags += ['-I'+str(private)]
            links = ['-L'+str(runtime_dir),'-Wl,-rpath,'+str(runtime_dir),'-lggml','-lggml-cpu','-lggml-base','-ldl']
            fixture_bin, timing_bin = OUT/'hc-graph-check', OUT/'hc-timing'
            run([*flags,str(fixture),*links,'-o',str(fixture_bin)],'compile-graph-check')
            run([*flags,str(timing),*links,'-o',str(timing_bin)],'compile-timing')
            result['idle_gate'] = guard.wait_idle(OUT/'waiting-for-idle.json')
            result['background_gate'] = wait_background(guard,1219506,4,OUT/'background-wait.json')
            env = {key:value for key,value in os.environ.items() if key not in runtime_environment(os.environ) and key!='LD_PRELOAD'}
            env.update(runtime['runtime_env'])
            outputs = []
            for name,enabled,is_parent,threads in [('parent',False,True,15),('off',False,False,15),('on',True,False,15),('one-worker',True,False,1),('four-workers',True,False,4)]:
                trial = dict(env,GGML_CPU_QWEN_HC_ORDERED_K=str(int(enabled)),GGML_CPU_QWEN_HC_ROW_SPLIT=str(int(enabled)),GGML_CPU_QWEN_HC_ORDERED_K_AUDIT='1')
                expected = Path(parent['library']) if is_parent else library
                trial['LD_LIBRARY_PATH'] = str(expected.parent)+':'+str(runtime_dir)
                output = OUT/(name+'.bin')
                log = run(['taskset','-c','0-14',str(fixture_bin),str(output),str(threads)],'check-'+name,trial)
                checked_library(log,expected)
                assert 'HC_SUMMARY {"cases":96,"failures":0}' in log
                assert 'HC_COUNTER_PRESENT '+str(int(not is_parent)) in log
                expected_calls = 90 if enabled and threads==15 else 0
                assert 'HC_TOTAL_CALLS '+str(expected_calls)+'\n' in log
                outputs.append(output)
                result['checks'].append(dict(name=name,threads=threads,cases=96,selected_calls=expected_calls,
                                             output_bytes=output.stat().st_size,output_sha256=sha256(output),cpu_sha256=sha256(expected)))
                save()
            assert all(path.read_bytes()==outputs[0].read_bytes() for path in outputs[1:])
            result['bit_exact'] = True
            for index,enabled in enumerate((False,True,True,False)):
                expected = library if enabled else Path(parent['library'])
                trial = dict(env,LD_LIBRARY_PATH=str(expected.parent)+':'+str(runtime_dir),
                             GGML_CPU_QWEN_HC_ORDERED_K=str(int(enabled)),GGML_CPU_QWEN_HC_ROW_SPLIT=str(int(enabled)),GGML_CPU_QWEN_HC_ORDERED_K_AUDIT='0')
                log = run(['taskset','-c','0-14',str(timing_bin)],f'timing-{index}',trial)
                checked_library(log,expected)
                rows = [json.loads(line.removeprefix('HC_TIME ')) for line in log.splitlines() if line.startswith('HC_TIME ')]
                assert len(rows)==12 and all(row['samples']==40 for row in rows)
                result['timings'].append(dict(enabled=enabled,cpu_sha256=sha256(expected),rows=rows))
                save()
            comparisons = []
            for row in result['timings'][0]['rows']:
                key = {name:row[name] for name in ('nc','nr','matrices','weight_bytes')}
                matching = [(trial['enabled'],other) for trial in result['timings'] for other in trial['rows'] if all(other[k]==v for k,v in key.items())]
                assert len(matching)==4 and len({other['hash'] for _,other in matching})==1
                before = statistics.mean(other['median_us'] for enabled,other in matching if not enabled)
                after = statistics.mean(other['median_us'] for enabled,other in matching if enabled)
                comparisons.append(dict(**key,parent_us=before,candidate_us=after,change_percent=100*(after/before-1),speedup=before/after))
            assert all(sha256(path)==digest for path,digest in {**inputs,**result['private_source_sha256']}.items())
            guard.assert_idle()
            result.update(passed=True,comparisons=comparisons,fixture_sha256=sha256(fixture_bin),timing_sha256=sha256(timing_bin))
            print(json.dumps(dict(passed=True,cpu_sha256=result['library_sha256'],bit_exact=True,comparisons=comparisons)),flush=True)
        except BaseException as error:
            result['error'] = repr(error)
            raise
        finally:
            if owned is not None and owned.poll() is None:
                os.killpg(owned.pid,signal.SIGTERM)
                owned.wait(timeout=15)
            result.update(finished=time.time(),peer_preserved=process_info(1219506)['start']=='103969952',peer_service=read_service(18095))
            save()


if __name__ == '__main__':
    os.umask(0o077)
    main()
