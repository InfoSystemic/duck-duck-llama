#!/usr/bin/env python3
"""Compare scoped R8 changes off/on/on/off, with one Flash handoff and exact restoration."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import signal
import statistics
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
OUT = BASE / 'results/qwen-r8-projection-model-0910'
PORT = 18155
COUNT = 160


def common_library_paths(maps_text):
    paths=set()
    for line in maps_text.splitlines():
        fields=line.split(maxsplit=5)
        if len(fields)==6 and Path(fields[5]).name.startswith('libllama-common.so.'):
            paths.add(fields[5])
    return paths


def memory_gate(global_bytes, node_bytes):
    memory, nodes = memory_status(), node_memory_status()
    assert memory['MemAvailable'] > global_bytes, ('Global RAM reserve', memory)
    assert len(nodes) == 4 and all(row['estimated_available'] > node_bytes for row in nodes.values()), ('NUMA RAM reserve', nodes)
    return dict(memory=memory, nodes=nodes)


def loaded_common_libraries(pid):
    return common_library_paths(Path(f'/proc/{pid}/maps').read_text())


def read_arm_measurement(path, plan, environment):
    data = json.loads(path.read_text())
    assert data['finished'] and data['input_integrity_verified'] and not data.get('error')
    assert len(data['checks']) == 2 and all(c['pass_check'] and not c['abort'] for c in data['checks'])
    assert data['server_command'] == plan['command'] and data['runtime_env'] == environment
    assert all(sha256(p) == digest for p, digest in data['input_sha256'].items())
    rows = {}
    for row in data['measurements']:
        assert not row['abort'] and not row['inference_churn'] and not row['other_inference']
        assert row['draft_n'] == 4 and row['timings']['cache_n'] == 0
        directory = path.parent / (row['kind'] + '-draft4')
        counters = validate_counters(directory, row)
        _, timings, digest = output_record(json.loads((directory / 'chunks.json').read_text()))
        assert timings == row['timings'] and 128 <= timings['predicted_n'] <= 512
        assert timings['draft_n'] > 0
        rows[row['kind']] = dict(tok_s=timings['predicted_per_second'],
            adjusted_gb_s=row['background_subtracted_gb_s'], counters=counters,
            generated_tokens=timings['predicted_n'], draft_tokens=timings['draft_n'],
            accepted_draft_tokens=timings['draft_n_accepted'], output_sha256=digest,
            cache_tokens=timings['cache_n'], completed_answer=row['completed_answer'])
    assert set(rows) == {'prose', 'code'}
    return rows


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
    build_path = BASE / 'results/qwen-r8-projection-build-0910/result.json'
    fixture_path = BASE / 'results/qwen-r8-projection-validation-0910/result.json'
    fresh_path = BASE / 'results/qwen-mtp-fresh-seed-0910f/result.json'
    checks_path = BASE / 'results/qwen-r8-projection-controller-checks-0910.json'
    preset_path = BASE / 'qwen-flash-20tps.json'
    build, fixture, fresh, checks, preset = [json.loads(p.read_text()) for p in
        [build_path, fixture_path, fresh_path, checks_path, preset_path]]
    assert all(row['passed'] for row in [build, fixture, fresh, checks])
    assert fixture['finished'] and fixture['correctness_cases'] == 330 and fixture['timing_cases'] == 24
    assert len(fixture['runs']) == 15 and all(row['exact_parent_outputs'] for row in fixture['runs'])
    assert fresh['fix_repeatability_passed'] and len(fresh['repeat_requests']) == 8
    assert checks['sources'][str(Path(__file__))] == sha256(__file__)
    assert sha256(build['library']) == build['library_sha256']
    assert sha256(build['common']) == build['common_sha256']
    assert build['common_sha256'] == 'bd121009dcbb121005a587262bfda25008cd63dc8d46016c3f4fbd50d5458f8d'
    records, extra = model_records(preset)
    command = list(preset['command'])
    command[0] = build['server']
    for flag, value in [('--port', PORT), ('--alias', 'qwen-q6-private'), ('--verbosity', 4)]:
        command = set_option(command, flag, value)
    command += ['--slots', '--slot-save-path', str(OUT / 'slot-action')]
    environment = dict(build['runtime_env'])
    environment.update(GGML_CPU_Q8_0_REPACK_X_TILE='1', GGML_CPU_Q8_R8_K160_AUDIT='0')
    assert not any('PROFILE' in key for key in environment)
    paths = [Path(__file__), BASE / 'probe_qwen_mtp_fresh_seed_0910f.py',
        BASE / 'check_qwen_r8_projection_controller_0910.py',
        BASE / 'benchmark_flash_q4_selected_0910.py', BASE / 'guarded_inference_request.py',
        BASE / 'model_measurement_guard.py', BASE / 'measure-model-bandwidth.py', BASE / 'dram_bandwidth.py',
        BASE / 'trace_qwen_shared_dispatch_ops_0909.py', BASE / 'select_flash_q4_0910c.py',
        SELECTED, build_path, fixture_path, fresh_path, checks_path, preset_path, *extra,
        Path(build['library']), Path(build['common']), Path(build['base']), Path(build['llama']), Path(build['server'])]
    sources = {str(p): sha256(p) for p in paths}
    for record, key in [(build, 'private_source_sha256'), (fixture, 'input_sha256'), (checks, 'sources')]:
        for p, digest in record[key].items():
            assert sha256(p) == digest
            sources[p] = digest
    plan = dict(prepared=time.time(), peer=current, peer_libraries=sorted(mapped_libraries(current['pid'])),
        command=command, runtime_env=environment, port=PORT, arm_file=str(OUT / 'unused-profile.arm'),
        library=build['library'], base=build['base'], llama=build['llama'], common=build['common'],
        source_sha256=sources, model_records=records, quant='UD-Q6_K_XL', drafts=4,
        schedule=['off', 'both', 'both', 'off'],
        scope='Four fresh Q6/Q8-MTP4 model launches. Identical candidate CPU and fresh-MTP common library; only K160 preparation and K1536 SSM tile flags vary. One Flash handoff and exact restoration. No automatic promotion.',
        target_tok_s=40, target_gb_s=250, capacity_gb_s=380)
    verify_sources(plan)
    OUT.mkdir()
    (OUT / 'slot-action').mkdir()
    atomic_json(OUT / 'plan.json', plan)
    print(json.dumps(dict(prepared=True, peer_pid=current['pid'], quant=plan['quant'], schedule=plan['schedule'])), flush=True)


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
        stage='prepared',model_started=False,trials=[],restored=False,target_reached=False,runtime_promoted=False)
    cancelled, recovering, handoff_started = False, False, False
    model, child = None, None
    arm_dir = OUT

    def save(stage=None):
        if stage: result['stage'] = stage
        atomic_json(OUT/'result.json',result)

    def cancel(*_):
        nonlocal cancelled
        if not recovering: cancelled = True

    def check_cancel():
        if cancelled and not recovering: raise InterruptedError('Release the owned Qwen arm and restore Flash Q4')

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
        baseline = None
        for index, mode in enumerate(plan['schedule']):
            check_cancel()
            verify_sources(plan)
            arm_dir = OUT / f'arm-{index:02d}-{mode}'
            arm_dir.mkdir()
            enabled = mode == 'both'
            arm_env = dict(plan['runtime_env'], GGML_CPU_Q8_R8_K160_PREP='1' if enabled else '0',
                GGML_CPU_Q8_R8_SSM_TILE8='1' if enabled else '0')
            qwen_env = {k: v for k, v in os.environ.items() if k not in runtime_environment(os.environ)}
            qwen_env.update(arm_env)
            arm = dict(index=index, mode=mode, started=time.time(), runtime_env=arm_env)
            result['trials'].append(arm)
            save(f'loading Qwen arm {index} {mode}')
            model, info = load(plan['command'], qwen_env, BASE, PORT, arm_dir / 'model.log',
                               220_000_000_000, 50_000_000_000)
            result['model_started'] = True
            arm.update(model_pid=model.pid, current=info)
            save()
            for stem, key in [('libggml-cpu.so.', 'library'), ('libggml-base.so.', 'base'), ('libllama.so.', 'llama')]:
                assert {p for p in mapped_libraries(model.pid) if '/' + stem in p} == {str(Path(plan[key]).resolve())}
            assert loaded_common_libraries(model.pid) == {str(Path(plan['common']).resolve())}
            arm['dispatch_after_load'] = dispatch_log(arm_dir)
            dispatch = arm['dispatch_after_load']
            assert any(ranks == '4' and dispatch['attached'].count((group, ranks)) >= 2 for group, ranks in dispatch['created'])
            guard = ModelMeasurementGuard(model.pid, {model.pid: PORT}, inference_snapshot)
            wait_idle(guard, f'arm-{index}-idle')
            arm['background_before_measurement'] = background(model.pid)
            label = f'qwen-r8-projection-measure-0910-{index}-{mode}'
            args = [sys.executable, '-u', str(BASE / 'measure-model-bandwidth.py'), label, '--port', str(PORT),
                '--pid', str(model.pid), '--alias', 'qwen-q6-private', '--drafts', '4', '--tokens', '512',
                '--request-timeout-seconds', '300', '--allowed-idle-pids', '', '--skip-idle-gate',
                '--bandwidth-target-gb-s', '250', '--bandwidth-capacity-gb-s', '380']
            save(f'measuring Qwen arm {index} {mode}')
            child = subprocess.Popen(args, pass_fds=(lock_fd,))
            while child.poll() is None:
                check_cancel()
                memory_gate(32 << 30, 8 << 30)
                assert model.poll() is None
                time.sleep(.5)
            assert child.returncode == 0
            child = None
            path = BASE / 'results' / label / 'result.json'
            rows = read_arm_measurement(path, plan, arm_env)
            arm.update(rows=rows, measurement=str(path), measurement_sha256=sha256(path),
                background_after_measurement=background(model.pid))
            if baseline is None: baseline = rows
            arm['outputs_and_counts_match'] = all(rows[kind][key] == baseline[kind][key]
                for kind in ['prose', 'code'] for key in
                ['output_sha256', 'generated_tokens', 'draft_tokens', 'accepted_draft_tokens', 'cache_tokens'])
            save()
            assert arm['outputs_and_counts_match'], 'R8 candidate changed output or speculative counts'
            guard.assert_idle()
            model.terminate()
            try: model.wait(timeout=45)
            except subprocess.TimeoutExpired: model.kill(); model.wait(timeout=15)
            arm['owned_qwen_exit'] = model.returncode
            dispatch = dispatch_log(arm_dir)
            arm['dispatch_cleanup_valid'] = sorted(g for g, r in dispatch['created']) == sorted(dispatch['destroyed'])
            arm['finished'] = time.time()
            save()
            assert arm['dispatch_cleanup_valid'] and not inference_snapshot()
            model = None
            print(json.dumps(dict(completed_arm=index, mode=mode, rows=rows, dispatch_clean=True)), flush=True)
        summaries = []
        for kind in ['prose', 'code']:
            before = [arm['rows'][kind] for arm in result['trials'] if arm['mode'] == 'off']
            after = [arm['rows'][kind] for arm in result['trials'] if arm['mode'] == 'both']
            bp, cp = [statistics.mean(row['tok_s'] for row in values) for values in [before, after]]
            bb, cb = [statistics.mean(row['adjusted_gb_s'] for row in values) for values in [before, after]]
            qualifies = all(row['counters']['adjacent_idle_qualifies'] for row in before + after)
            summaries.append(dict(workload=kind, off_mean_tok_s=bp, on_mean_tok_s=cp,
                speed_change_percent=100 * (cp / bp - 1), off_mean_gb_s=bb, on_mean_gb_s=cb,
                all_attribution_valid=qualifies,
                repeated_bandwidth_target_met=qualifies and all(row['adjusted_gb_s'] >= 250 for row in after),
                repeated_tok_s_target_met=qualifies and all(row['tok_s'] >= 40 for row in after)))
        result.update(summaries=summaries, all_outputs_match=True,
            target_reached=all(row['repeated_bandwidth_target_met'] for row in summaries),
            tok_s_target_reached=all(row['repeated_tok_s_target_met'] for row in summaries))
        verify_sources(plan)
        result['passed'] = True

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
            dispatch = dispatch_log(arm_dir)
            result['dispatch_cleanup_valid'] = sorted(g for g,r in dispatch['created']) == sorted(dispatch['destroyed'])
        if handoff_started and not inference_snapshot():
            save('restoring selected Flash Q4')
            try:
                restored,info = load(original['command'],environment,saved_cwd,18131,OUT/'restored-flash.log',260_000_000_000,60_000_000_000)
                assert mapped_libraries(restored.pid) == set(plan['peer_libraries'])
                assert sha256(SELECTED) == plan['source_sha256'][str(SELECTED)]
                current = dict(configuration,pid=restored.pid,info=info,log=str(OUT/'restored-flash.log'))
                manager.state['current'] = current
                manager.record('flash_restored_after_qwen_r8_abba',pid=restored.pid,port=18131)
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
