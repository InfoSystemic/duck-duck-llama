#!/usr/bin/env python3
"""Compare the qualified Q6 single-activation kernel with the current Qwen runtime."""
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time

from compare_qwen_get_rows_bandwidth_0909 import read_run
from model_measurement_guard import ModelMeasurementGuard
from qwen_private_decode_modes_0909 import execute, memory_gate, verify_plan
from qwen_split_trial import inference_snapshot, process_info, runtime_environment, sha256

BASE = Path(__file__).resolve().parent
OUT = BASE / 'results/qwen-q6-single-model-0910'
REFERENCE = 'qwen-private-get-rows-250-0909-1-columns'


def package(build, parent, guard):
    out = BASE / 'results/qwen-q6-single-runtime-0910'
    out.mkdir(exist_ok=False)
    binary_dir = out / 'bin'
    binary_dir.mkdir()
    source_dir = Path(parent['runtime_directory'])
    for name in ('llama-server','loader-check'):
        shutil.copy2(source_dir/name,binary_dir/name)
        assert sha256(binary_dir/name)==sha256(source_dir/name)
    cpu = Path(build['library'])
    for path in source_dir.glob('lib*.so*'):
        target = cpu if path.name.startswith('libggml-cpu.so') else path.resolve()
        (binary_dir/path.name).symlink_to(target.resolve())
    runtime = dict(parent['runtime_env'], LD_LIBRARY_PATH=str(binary_dir), GGML_CPU_QWEN_Q6_PACKED_SINGLE='1')
    environment = {key:value for key,value in os.environ.items() if key not in runtime_environment(os.environ) and key!='LD_PRELOAD'}
    environment.update(runtime)
    sources = {str(path.resolve()):sha256(path) for path in source_dir.iterdir() if path.is_file()}
    sources[str(cpu)] = sha256(cpu)
    result = dict(started=time.time(),passed=False,model_loaded=False,runtime_directory=str(binary_dir),runtime_env=runtime,
                  cpu=str(cpu),cpu_sha256=sha256(cpu),base=parent['base'],base_sha256=parent['base_sha256'],
                  llama=parent['llama'],llama_sha256=parent['llama_sha256'],server=str(binary_dir/'llama-server'),
                  server_sha256=sha256(binary_dir/'llama-server'),loader_sha256=sha256(binary_dir/'loader-check'),
                  sources=sources,feature="Q6 single-activation correction",
                  symlinks={str(path):str(path.resolve()) for path in binary_dir.iterdir() if path.is_symlink()})
    try:
        guard.assert_idle()
        checked = subprocess.run(['taskset','-c','0-127',str(binary_dir/'loader-check'),'private',str(source_dir)],
                                 cwd=BASE,env=environment,capture_output=True,text=True,timeout=30)
        (out/'loader.log').write_text(checked.stdout+checked.stderr)
        assert checked.returncode==0
        maps = {}
        for name,path in [('CPU',cpu),('BASE',Path(parent['base'])),('LLAMA',Path(parent['llama']))]:
            actual = {line.split()[-1] for line in checked.stdout.splitlines() if line.startswith(name+'_MAP ')}
            symbols = [line.removeprefix(name+'_SYMBOL ') for line in checked.stdout.splitlines() if line.startswith(name+'_SYMBOL ')]
            assert actual=={str(path.resolve())} and len(symbols)==1 and Path(symbols[0]).resolve()==path.resolve()
            maps[name] = sorted(actual)
        devices = [line.removeprefix('DEVICE ') for line in checked.stdout.splitlines() if line.startswith('DEVICE ')]
        assert all('CPU-NUMA'+str(i) in devices for i in range(4))
        guard.assert_idle()
        checked = subprocess.run(['taskset','-c','0-127',result['server'],'--list-devices'],
                                 cwd=BASE,env=environment,capture_output=True,text=True,timeout=30)
        (out/'server-devices.log').write_text(checked.stdout+checked.stderr)
        assert checked.returncode==0 and all('CPU-NUMA'+str(i) in checked.stdout+checked.stderr for i in range(4))
        assert all(sha256(path)==digest for path,digest in sources.items())
        result.update(passed=True,maps=maps,devices=devices)
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        result['finished'] = time.time()
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result,out/'result.json'


def main():
    assert os.sched_getaffinity(0)=={127} and not os.environ.get('LD_PRELOAD')
    assert process_info(1219506)['start']=='103969952'
    build_path = BASE/'results/qwen-q6-single-build-0910b/result.json'
    parent_path = BASE/'results/qwen-get-rows-runtime-0909/result.json'
    validation_path = BASE/'results/qwen-q6-single-validation-0910/result.json'
    build,parent = [json.loads(path.read_text()) for path in (build_path,parent_path)]
    assert build['passed'] and build['finished'] and not build.get('error')
    assert build['build_completed'] and build['baseline_object_identical'] and build['baseline_library_identical']
    assert sha256(build['library'])==build['library_sha256']
    assert parent['passed'] and parent['finished'] and build['parent_sha256']==parent['cpu_sha256']
    assert parent['cpu_sha256']==sha256(parent['cpu'])=='12c61b337736ca9210433f57c64ce7fffbf7e4b66920aba9a03f97eaef3fd9b7'
    for key in ('cpu','base','llama','server'):
        assert sha256(parent[key])==parent[key+'_sha256']
    for record in (build,parent):
        for key in ('input_sha256','private_source_sha256','sources'):
            assert all(sha256(path)==digest for path,digest in record.get(key,{}).items())
    validation = json.loads(validation_path.read_text())
    assert validation['passed'] and validation['finished'] and validation['model_test_eligible'] and validation['bit_exact']
    assert validation['cpu_sha256'] == build['library_sha256'] and validation['parent_sha256'] == parent['cpu_sha256']
    assert len(validation['checks']) == 7 and len(validation['numa_checks']) == 10 and len(validation['timings']) == 8
    assert validation['input_sha256'][str(build_path)] == sha256(build_path)
    for key in ('input_sha256', 'private_source_sha256'):
        assert all(sha256(path) == digest for path, digest in validation[key].items())
    reference = read_run(REFERENCE)
    verify_plan(reference['plan'])
    sources = [Path(__file__).resolve(),build_path,parent_path,validation_path,BASE/'validate_qwen_q6_single_0910.py',BASE/'qwen_private_decode_modes_0909.py',
               BASE/'compare_qwen_get_rows_bandwidth_0909.py',BASE/'qwen_split_trial.py',BASE/'model_measurement_guard.py',
               BASE/'results'/REFERENCE/'plan.json',BASE/'results'/REFERENCE/'result.json']
    guard = ModelMeasurementGuard(1219506,{1219506:18095},inference_snapshot)
    with (BASE/'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        guard.assert_idle()
        OUT.mkdir(exist_ok=False)
        result = dict(started=time.time(),passed=False,controller_pid=os.getpid(),target_gb_s=250,trials=[],
                      input_sha256={str(path):sha256(path) for path in sources},
                      scope='Parent/candidate/candidate/parent fresh single-conversation prose and code. Same Q6 target, Q8 MTP4 draft, server bytes, workers and corrected gather. Candidate enables only the component-qualified Q6 single-activation correction, with the same packed weights and expert scheduling. No service promotion.')

        def save(stage):
            result['stage'] = stage
            (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')

        save('checking private runtime')
        try:
            candidate,bundle_path = package(build,parent,guard)
            result['candidate_bundle'] = str(bundle_path)
            result['input_sha256'][str(bundle_path)] = sha256(bundle_path)
            baseline = None
            for index,enabled in enumerate((False,True,True,False)):
                mode = 'candidate' if enabled else 'parent'
                label = f'qwen-private-q6-single-250-0910-{index}-{mode}'
                out = BASE/'results'/label
                assert not out.exists()
                bundle = candidate if enabled else parent
                plan = copy.deepcopy(reference['plan'])
                plan.update(prepared=time.time(),label=label,runtime_env=bundle['runtime_env'],cpu=bundle['cpu'],
                            cpu_sha256=bundle['cpu_sha256'],profile=False,no_model_loaded=True,memory_preflight=memory_gate(),
                            q6_single=mode,
                            reference_outputs=reference['rows'],scope=result['scope'])
                plan['command'][0] = bundle['server']
                plan['source_sha256'].update(result['input_sha256'])
                plan['source_sha256'].update({str(Path(bundle[key])):sha256(bundle[key]) for key in ('cpu','base','llama','server')})
                verify_plan(plan)
                out.mkdir()
                (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
                save('running '+label)
                execute(out,lock.fileno())
                run = read_run(label)
                trial_result = json.loads((out/'result.json').read_text())
                assert trial_result['peer_environment_preserved']
                assert not Path('/proc',str(trial_result['model_pid'])).exists()
                for kind in ('prose','code'):
                    row = run['rows'][kind]
                    assert row['attribution_valid']
                    assert all(row[key]==reference['rows'][kind][key] for key in
                               ('output_sha256','generated_tokens','draft_tokens','accepted_draft_tokens'))
                if baseline is None:
                    baseline = run
                for key in ('model_records','drafts','base','base_sha256','llama','server_sha256','peer','peer_runtime_env'):
                    assert run['plan'][key]==baseline['plan'][key]
                assert run['plan']['command'][1:]==baseline['plan']['command'][1:]
                env_before,env_after = dict(baseline['plan']['runtime_env']),dict(run['plan']['runtime_env'])
                for key in ('LD_LIBRARY_PATH','GGML_CPU_QWEN_Q6_PACKED_SINGLE'):
                    env_before.pop(key,None)
                    env_after.pop(key,None)
                assert env_before==env_after
                text = (out/'model.log').read_text()
                markers = dict(q6_single='QWEN_Q6_PACKED_SINGLE nr=1 storage=unchanged' in text)
                result['trials'].append(dict(label=label,candidate=enabled,rows=run['rows'],evidence=run['evidence'],execution_markers=markers))
                save('captured '+label)
                assert markers['q6_single'] == enabled
                save('completed '+label)
                print(json.dumps(dict(completed_model_trial=label,rows=run['rows'],execution_markers=markers)),flush=True)
            summaries = []
            for kind in ('prose','code'):
                before = [row['rows'][kind] for row in result['trials'] if not row['candidate']]
                after = [row['rows'][kind] for row in result['trials'] if row['candidate']]
                bp,cp = [statistics.mean(row['tok_s'] for row in arm) for arm in (before,after)]
                bb,cb = [statistics.mean(row['adjusted_gb_s'] for row in arm) for arm in (before,after)]
                summaries.append(dict(workload=kind,parent_mean_tok_s=bp,candidate_mean_tok_s=cp,speed_change_percent=100*(cp/bp-1),
                                      parent_mean_gb_s=bb,candidate_mean_gb_s=cb,all_attribution_valid=True,
                                      repeated_bandwidth_target_met=all(row['adjusted_gb_s']>=250 for row in after)))
            assert all(sha256(path)==digest for path,digest in result['input_sha256'].items())
            guard.assert_idle()
            result.update(passed=True,summaries=summaries,all_outputs_match=True,
                          target_reached=all(row['repeated_bandwidth_target_met'] for row in summaries))
            print(json.dumps(dict(completed=True,summaries=summaries,target_reached=result['target_reached'])),flush=True)
        except BaseException as error:
            result['error'] = repr(error)
            raise
        finally:
            result['finished'] = time.time()
            save('finished')


if __name__ == '__main__':
    os.umask(0o077)
    main()
