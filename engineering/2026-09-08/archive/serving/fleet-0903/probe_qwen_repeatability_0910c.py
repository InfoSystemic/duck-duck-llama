#!/usr/bin/env python3
"""Probe repeated requests on the original Qwen CPU, then restore Flash Q4."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import urllib.request

from benchmark_flash_q4_selected_0910 import background, validate_counters
from glm_flash_q8_trial import memory_status, node_memory_status
from guarded_inference_request import stream_completion
from model_measurement_guard import ModelMeasurementGuard, read_service
from qwen_high_quant_trial import atomic_json, port_available, set_option, unit_state
from qwen_private_decode_modes_0909 import model_records, dispatch_log
from qwen_split_trial import inference_snapshot, process_info, process_environment, runtime_environment, sha256
from select_flash_q4_0910c import Manager, SELECTED, mapped_libraries
from trace_qwen_shared_dispatch_ops_0909 import output_record

BASE = Path(__file__).resolve().parent
OUT = BASE / 'results/qwen-repeatability-0910c'
PORT = 18155
COUNT = 160


def memory_gate(global_bytes, node_bytes):
    memory, nodes = memory_status(), node_memory_status()
    assert memory['MemAvailable'] > global_bytes, ('Global RAM reserve', memory)
    assert len(nodes) == 4 and all(row['estimated_available'] > node_bytes for row in nodes.values()), ('NUMA RAM reserve', nodes)
    return dict(memory=memory, nodes=nodes)


def reference_payload(data, row):
    # The recorded short-check request comes from the same measurement factory.
    body = copy.deepcopy(data['checks'][0]['request'])
    body.update(messages=[dict(role='user',content=row['prompt'])],max_tokens=512,stream=True)
    body['speculative.n_max'] = row['draft_n']
    assert body['speculative.p_min'] == 0 and body['speculative.n_max'] == 4
    assert body['chat_template_kwargs'] == {'enable_thinking': False}
    assert body['temperature'] == 0 and body['seed'] == 42 and body['cache_prompt'] is False
    return body


def verify_sources(plan):
    assert all(sha256(path) == value for path,value in plan['source_sha256'].items())
    for row in plan['model_records']:
        st = Path(row['path']).stat()
        assert (st.st_size,st.st_ino,st.st_mtime_ns) == (row['size'],row['inode'],row['mtime_ns'])
    assert unit_state()['ActiveState'] == 'inactive'


def prepare():
    assert not OUT.exists() and port_available(PORT)
    current = Manager().validate_current()
    assert current['quant'] == 'UD-Q4_K_XL' and current['drafts'] == 2
    assert set(inference_snapshot()) == {str(current['pid'])}
    build_path = BASE / 'results/qwen-timeline-build-0910/result.json'
    runtime_path = BASE / 'results/qwen-get-rows-runtime-0909/result.json'
    preset_path = BASE / 'qwen-flash-20tps.json'
    build, runtime, preset = [json.loads(p.read_text()) for p in (build_path,runtime_path,preset_path)]
    assert build['passed'] and build['baseline_library_identical'] and build['baseline_object_identical']
    assert build['parent_cpu_sha256'] == runtime['cpu_sha256']
    assert sha256(build['library']) == build['library_sha256']
    # Run the original corrected CPU; no timeline instrumentation or profile flags.
    build = dict(build,library=runtime['cpu'],library_sha256=runtime['cpu_sha256'],server=runtime['server'],runtime_env=runtime['runtime_env'])
    records, extra = model_records(preset)
    command = list(preset['command'])
    command[0] = build['server']
    for flag,value in [('--port',PORT),('--alias','qwen-q6-private'),('--verbosity',4)]:
        command = set_option(command,flag,value)
    command += ['--slots','--slot-save-path',str(OUT/'slot-action')]
    arm = OUT / 'op-profile.arm'
    environment = dict(build['runtime_env'])
    assert not any('PROFILE' in key for key in environment)
    paths = [Path(__file__), BASE/'analyze_qwen_timeline_0910b.py', BASE/'profile_qwen_ops_0908.py',
        BASE/'benchmark_flash_q4_selected_0910.py', BASE/'guarded_inference_request.py',
        BASE/'model_measurement_guard.py', BASE/'measure-model-bandwidth.py', BASE/'dram_bandwidth.py',
        BASE/'trace_qwen_shared_dispatch_ops_0909.py', BASE/'select_flash_q4_0910c.py',
        SELECTED, build_path,runtime_path,preset_path,*extra,Path(build['library']),
        Path(build['base']),Path(build['llama']),Path(build['server'])]
    sources = {str(p):sha256(p) for p in paths}
    for p,h in build['private_source_sha256'].items():
        assert sha256(p) == h
        sources[p] = h
    plan = dict(prepared=time.time(), peer=current, peer_libraries=sorted(mapped_libraries(current['pid'])),
        command=command,runtime_env=environment,port=PORT,count=COUNT,arm_file=str(arm),
        library=build['library'],base=build['base'],llama=build['llama'],
        source_sha256=sources,model_records=records,quant='UD-Q6_K_XL',drafts=4,
        scope='Repeated identical fresh requests on the original corrected Q6 CPU, with and without explicit slot erasure. Exact Flash Q4 restoration. Record host load; no new optimization or profile.',
        target_tok_s=40,target_gb_s=250,capacity_gb_s=380)
    verify_sources(plan)
    OUT.mkdir()
    (OUT/'slot-action').mkdir()
    atomic_json(OUT/'plan.json',plan)
    print(json.dumps(dict(prepared=True,peer_pid=current['pid'],quant=plan['quant'],probe='repeatability',original_cpu=True)),flush=True)


def execute(lock_fd):
    plan = json.loads((OUT/'plan.json').read_text())
    assert not (OUT/'result.json').exists()
    verify_sources(plan)
    manager = Manager()
    original = manager.validate_current()
    assert original == plan['peer'] and set(inference_snapshot()) == {str(original['pid'])}
    assert port_available(PORT)
    environment = process_environment(original['pid'])
    saved_cwd = process_info(original['pid'])['cwd']
    atomic_json(OUT/'flash-restore-context.private.json',dict(environment=environment,cwd=saved_cwd,current=original))
    configuration = {k:v for k,v in original.items() if k not in {'pid','info','log'}}
    result = dict(started=time.time(),passed=False,controller_pid=os.getpid(),plan_sha256=sha256(OUT/'plan.json'),
        stage='prepared',model_started=False,checks=[],traces=[],restored=False,target_reached=False,diagnostic_only=True)
    cancelled, recovering, handoff_started = False, False, False
    model, child = None, None

    def save(stage=None):
        if stage: result['stage'] = stage
        atomic_json(OUT/'result.json',result)

    def cancel(*_):
        nonlocal cancelled
        if not recovering: cancelled = True

    def check_cancel():
        if cancelled and not recovering: raise InterruptedError('Release Qwen diagnostic and restore Flash Q4')

    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP): signal.signal(sig,cancel)

    def wait_idle(guard,label,seconds=15):
        quiet = None
        while True:
            check_cancel()
            guard.assert_idle()
            now = time.monotonic()
            quiet = now if quiet is None else quiet
            atomic_json(OUT/(label+'.json'),dict(time=time.time(),quiet_seconds=now-quiet))
            if now-quiet >= seconds: return
            time.sleep(1)

    def load(command,env,cwd,port,log_path,threshold,node_threshold):
        assert not inference_snapshot() and port_available(port)
        memory_gate(threshold,node_threshold)
        with log_path.open('w') as log:
            proc = subprocess.Popen(['taskset','-c','0-127',*command],env=env,cwd=cwd,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        deadline, report = time.monotonic()+1200, 0
        try:
            while True:
                check_cancel()
                assert proc.poll() is None, ('Model exited while loading',proc.returncode)
                assert set(inference_snapshot()) <= {str(proc.pid)}, 'Competing model appeared'
                memory_gate(32<<30,8<<30)
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2) as f:
                        if json.load(f).get('status') == 'ok': break
                except OSError: pass
                now = time.monotonic()
                assert now < deadline, 'Model load timeout'
                if now-report >= 30:
                    print(json.dumps(dict(loading_pid=proc.pid,port=port,recovering=recovering)),flush=True);report=now
                time.sleep(1)
            info = process_info(proc.pid)
            assert info['command'] == command and info['cwd'] == str(cwd) and info['affinity'] == list(range(128))
            assert process_environment(proc.pid) == env
            return proc,info
        except BaseException:
            if proc.poll() is None:
                proc.terminate()
                try: proc.wait(timeout=30)
                except subprocess.TimeoutExpired: proc.kill();proc.wait(timeout=15)
            raise

    save()
    try:
        guard = ModelMeasurementGuard(original['pid'],{original['pid']:18131},inference_snapshot)
        wait_idle(guard,'flash-idle')
        result['background_before_handoff'] = background(original['pid'])
        manager.validate_current();verify_sources(plan);check_cancel()
        handoff_started = True
        save('unloading selected Flash Q4')
        manager.stop_flash()
        check_cancel()
        result['memory_after_flash_stop'] = memory_gate(220_000_000_000,50_000_000_000)
        qwen_env = {k:v for k,v in os.environ.items() if k not in runtime_environment(os.environ)}
        qwen_env.update(plan['runtime_env'])
        log = OUT/'model.log'
        save('loading diagnostic Qwen Q6')
        model,info = load(plan['command'],qwen_env,BASE,PORT,log,220_000_000_000,50_000_000_000)
        result.update(model_started=True,model_pid=model.pid,current=info)
        for stem,key in [('libggml-cpu.so.','library'),('libggml-base.so.','base'),('libllama.so.','llama')]:
            assert {p for p in mapped_libraries(model.pid) if '/'+stem in p} == {str(Path(plan[key]).resolve())}
        dispatch = dispatch_log(OUT)
        assert any(ranks=='4' and dispatch['attached'].count((group,ranks))>=2 for group,ranks in dispatch['created'])
        result['dispatch_after_load'] = dispatch
        guard = ModelMeasurementGuard(model.pid,{model.pid:PORT},inference_snapshot)
        wait_idle(guard,'qwen-idle')
        result['background_before_reference'] = background(model.pid)
        assert not Path(plan['arm_file']).exists()
        label = 'qwen-repeatability-reference-0910c'
        args = [sys.executable,'-u',str(BASE/'measure-model-bandwidth.py'),label,'--port',str(PORT),
            '--pid',str(model.pid),'--alias','qwen-q6-private','--drafts','4','--tokens','512',
            '--request-timeout-seconds','300','--allowed-idle-pids','','--skip-idle-gate',
            '--bandwidth-target-gb-s','250','--bandwidth-capacity-gb-s','380']
        save('measuring unarmed diagnostic reference under recorded load')
        child = subprocess.Popen(args,pass_fds=(lock_fd,))
        while child.poll() is None:
            check_cancel();memory_gate(32<<30,8<<30);assert model.poll() is None;time.sleep(.5)
        assert child.returncode == 0
        path = BASE/'results'/label/'result.json'
        data = json.loads(path.read_text())
        assert data['input_integrity_verified'] and not data.get('error') and all(c['pass_check'] for c in data['checks'])
        assert data['server_command'] == plan['command'] and data['runtime_env'] == plan['runtime_env']
        assert 'CPU_OP_PROFILE index=' not in log.read_text()
        reference = {}
        for row in data['measurements']:
            assert not row['abort'] and row['timings']['cache_n']==0
            directory = path.parent/(row['kind']+'-draft4')
            counters = validate_counters(directory,row)
            _,timings,digest = output_record(json.loads((directory/'chunks.json').read_text()))
            reference[row['kind']] = dict(output_sha256=digest,timings=timings,counters=counters)
        result.update(reference=reference,measurement=str(path),measurement_sha256=sha256(path))
        result['repeat_requests'] = []
        save('probing repeated fresh requests on original CPU')
        by_kind = {row['kind']:row for row in data['measurements']}
        sequence = [('prose',False),('prose',False),('code',False),('prose',False),('prose',True),('code',True)]
        for index,(kind,erase) in enumerate(sequence):
            check_cancel();guard.assert_idle();guard.reset_activity()
            row=by_kind[kind]
            body=reference_payload(data,row)
            entry=dict(index=index,kind=kind,erase_before=erase,request=body,abort=[],started=time.time())
            result['repeat_requests'].append(entry);save()
            if erase:
                request=urllib.request.Request(f'http://127.0.0.1:{PORT}/slots/0?action=erase',data=b'{}',headers={'Content-Type':'application/json'},method='POST')
                with urllib.request.urlopen(request,timeout=10) as response:
                    entry['erase_response']=json.load(response)
                assert entry['erase_response']['id_slot']==0
                guard.assert_idle();guard.reset_activity();save()
            def repeat_abort():
                if cancelled:return dict(reason='Controller cancelled')
                memory_gate(32<<30,8<<30)
                return guard.abort_reason()
            chunks=[]
            try:
                stream_completion(PORT,body,repeat_abort,entry['abort'],chunks.append,interval=.5,max_elapsed=300)
            finally:
                chunks_path=OUT/f'repeat-{index:02d}-{kind}-chunks.json'
                atomic_json(chunks_path,chunks)
                entry.update(finished=time.time(),chunks_sha256=sha256(chunks_path))
                save()
            value,timings,digest=output_record(chunks)
            expected=reference[kind]
            same_counts=all(timings.get(k,0)==expected['timings'].get(k,0) for k in ('predicted_n','cache_n','draft_n','draft_n_accepted'))
            original_chunks=path.parent/(kind+'-draft4')/'chunks.json'
            original_value,_,_=output_record(json.loads(original_chunks.read_text()))
            content=value[0];original_content=original_value[0]
            first_difference=next((i for i,(a,b) in enumerate(zip(content,original_content)) if a!=b),min(len(content),len(original_content)))
            entry.update(output_sha256=digest,timings=timings,output_matches_reference=digest==expected['output_sha256'],
                counts_match_reference=same_counts,reference_content_length=len(original_content),content_length=len(content),
                first_content_difference=first_difference if content!=original_content else None)
            save()
            assert not entry['abort'] and timings['cache_n']==0
            print(json.dumps(dict(index=index,kind=kind,erase_before=erase,output_matches_reference=entry['output_matches_reference'],
                counts_match_reference=same_counts,generated=timings['predicted_n'],drafted=timings.get('draft_n'),accepted=timings.get('draft_n_accepted'))),flush=True)
        assert 'CPU_OP_PROFILE index=' not in log.read_text()
        result['repeatability_passed']=all(row['output_matches_reference'] and row['counts_match_reference'] for row in result['repeat_requests'])
        result['collection_complete']=True
        verify_sources(plan)
        result['passed'] = True  # Successful collection; repeatability is reported separately.
    except BaseException as error:
        result['error'] = repr(error)
        result['error_traceback'] = traceback.format_exc()
        raise
    finally:
        recovering = True
        Path(plan['arm_file']).unlink(missing_ok=True)
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=10)
        if model is not None:
            if model.poll() is None:
                model.terminate()
                try:model.wait(timeout=45)
                except subprocess.TimeoutExpired:model.kill();model.wait(timeout=15)
            result['owned_qwen_exit'] = model.returncode
            dispatch = dispatch_log(OUT)
            result['dispatch_cleanup_valid'] = sorted(g for g,r in dispatch['created']) == sorted(dispatch['destroyed'])
        if handoff_started and not inference_snapshot():
            save('restoring selected Flash Q4')
            try:
                restored,info = load(original['command'],environment,saved_cwd,18131,OUT/'restored-flash.log',260_000_000_000,60_000_000_000)
                assert mapped_libraries(restored.pid) == set(plan['peer_libraries'])
                assert sha256(SELECTED) == plan['source_sha256'][str(SELECTED)]
                current = dict(configuration,pid=restored.pid,info=info,log=str(OUT/'restored-flash.log'))
                manager.state['current'] = current
                manager.record('flash_restored_after_qwen_timeline',pid=restored.pid,port=18131)
                manager.validate_current()
                result.update(restored=True,restored_flash_pid=restored.pid,restored_flash_start=info['start'],
                    full_environment_restored=True,restored_command_and_affinity=True,restored_libraries_match=True,
                    restored_service=read_service(18131))
            except BaseException as error:
                result.update(passed=False,restore_error=repr(error),finished=time.time())
                save('Flash restoration failed')
                raise
        elif handoff_started:
            try:
                manager.validate_current()
                result['original_flash_preserved'] = True
            except BaseException as error:
                result.update(passed=False,restore_error='Competing or unidentified inference prevents restoration: '+repr(error))
        result['finished'] = time.time()
        save('finished')


if __name__ == '__main__':
    os.umask(0o077)
    assert os.sched_getaffinity(0)=={127}
    parser = argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','execute']);args=parser.parse_args()
    with (BASE/'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.action=='prepare': prepare()
        else: execute(lock.fileno())
