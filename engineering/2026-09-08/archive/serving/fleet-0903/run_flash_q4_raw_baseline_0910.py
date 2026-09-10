#!/usr/bin/env python3
"""Load Flash Q4 without a draft model, repeat raw decode, and restore the exact selected MTP2 service."""
import argparse
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
from model_measurement_guard import ModelMeasurementGuard, read_service
from qwen_high_quant_trial import atomic_json, port_available, set_option, unit_state
from qwen_split_trial import inference_snapshot, process_info, process_environment, runtime_environment, sha256
from select_flash_q4_0910c import Manager, SELECTED, mapped_libraries
from trace_qwen_shared_dispatch_ops_0909 import output_record

BASE = Path(__file__).resolve().parent
OUT = BASE / 'results/flash-q4-raw-baseline-0910'
PORT = 18162


def memory_gate(global_bytes, node_bytes):
    memory, nodes = memory_status(), node_memory_status()
    assert memory['MemAvailable'] > global_bytes, ('Global RAM reserve', memory)
    assert len(nodes) == 4 and all(row['estimated_available'] > node_bytes for row in nodes.values()), ('NUMA RAM reserve', nodes)
    return dict(memory=memory, nodes=nodes)


def common_libraries(pid):
    return {line.split(maxsplit=5)[-1] for line in Path(f'/proc/{pid}/maps').read_text().splitlines()
            if '/libllama-common.so.' in line}


def verify_sources(plan):
    assert all(sha256(path) == digest for path, digest in plan['source_sha256'].items())
    for item in plan['model_records']:
        stat = Path(item['path']).stat()
        assert (stat.st_size, stat.st_ino, stat.st_dev, stat.st_mtime_ns) == (item['size'], item['inode'], item['device'], item['mtime_ns'])
    assert unit_state()['ActiveState'] == 'inactive'


def read_measurement(path, plan):
    data = json.loads(path.read_text())
    assert data['finished'] and data['input_integrity_verified'] and not data.get('error')
    assert len(data['checks']) == 2 and all(c['pass_check'] and not c['abort'] for c in data['checks'])
    assert data['server_command'] == plan['command'] and data['runtime_env'] == plan['runtime_env']
    assert all(sha256(p) == digest for p, digest in data['input_sha256'].items())
    rows = {}
    for row in data['measurements']:
        assert not row['abort'] and not row['inference_churn'] and not row['other_inference']
        assert row['draft_n'] == 0 and row['timings']['cache_n'] == 0
        sample = path.parent / (row['kind'] + '-draft0')
        counters = validate_counters(sample, row)
        _, timings, digest = output_record(json.loads((sample / 'chunks.json').read_text()))
        assert timings == row['timings'] and 32 <= timings['predicted_n'] <= 512
        assert timings.get('draft_n', 0) == 0 and timings.get('draft_n_accepted', 0) == 0
        rows[row['kind']] = dict(tok_s=timings['predicted_per_second'], adjusted_gb_s=row['background_subtracted_gb_s'],
            counters=counters, generated_tokens=timings['predicted_n'], output_sha256=digest,
            draft_tokens=timings.get('draft_n', 0), accepted_draft_tokens=timings.get('draft_n_accepted', 0),
            cache_tokens=timings['cache_n'], completed_answer=row['completed_answer'])
    assert set(rows) == {'prose', 'code'}
    return rows


def prepare():
    assert not OUT.exists() and port_available(PORT)
    manager = Manager()
    peer = manager.validate_current()
    assert peer['quant'] == 'UD-Q4_K_XL' and peer['drafts'] == 2 and peer['workers'] == 15
    assert set(inference_snapshot()) == {str(peer['pid'])}
    selection_path = BASE / 'results/glm-flash-q4-selection-0910c/plan.json'
    checks_path = BASE / 'results/flash-q4-raw-controller-checks-0910.json'
    baseline_path = BASE / 'results/flash-q4-quiet-mtp2-0910/result.json'
    selection, checks, baseline = [json.loads(p.read_text()) for p in [selection_path, checks_path, baseline_path]]
    assert checks['passed'] and checks['sources'][str(Path(__file__))] == sha256(__file__)
    assert baseline['passed'] and baseline['selected_flash_preserved'] and baseline['current'] == peer
    assert baseline['all_outputs_and_counts_match']
    chosen = selection['configurations']['raw']
    assert chosen['quant'] == peer['quant'] and chosen['drafts'] == 0
    assert chosen['runtime_env'] == peer['runtime_env']
    command = list(chosen['command'])
    assert not any(x.startswith('--spec-') for x in command)
    for flag, value in [('--host', '127.0.0.1'), ('--port', PORT), ('--alias', 'glm-flash-q4-raw-private')]:
        command = set_option(command, flag, value)
    command += ['--spec-type', 'none']
    assert '--no-cache-prompt' in command and '--spec-draft-model' not in command
    libraries = {stem: next(p for p in mapped_libraries(peer['pid']) if '/' + stem in p)
        for stem in ['libggml-cpu.so.', 'libggml-base.so.', 'libllama.so.']}
    records = []
    for old in selection['model_records']:
        path = Path(old['path']); stat = path.stat()
        assert (stat.st_size, stat.st_ino, stat.st_mtime_ns) == (old['bytes'], old['inode'], old['mtime_ns'])
        records.append(dict(path=str(path), size=stat.st_size, inode=stat.st_ino,
            device=stat.st_dev, mtime_ns=stat.st_mtime_ns, publisher_sha256=old['sha256']))
    assert len(records) == 7
    paths = [Path(__file__), BASE / 'check_flash_q4_raw_controller_0910.py', BASE / 'measure-model-bandwidth.py',
        BASE / 'run_full_raw_baseline_0910.py', BASE / 'benchmark_flash_q4_quiet_0910.py',
        BASE / 'benchmark_flash_q4_selected_0910.py', BASE / 'model_measurement_guard.py',
        BASE / 'guarded_inference_request.py', BASE / 'dram_bandwidth.py', BASE / 'qwen_split_trial.py',
        BASE / 'qwen_high_quant_trial.py', BASE / 'select_flash_q4_0910c.py', SELECTED,
        selection_path, checks_path, baseline_path, Path(command[0]), *map(Path, mapped_libraries(peer['pid']))]
    sources = {str(p): sha256(p) for p in paths}
    # The established Q4 admission policy includes target repacking, scratch and
    # reserves. After stopping MTP2, raw decode needs less memory than its restore.
    plan = dict(prepared=time.time(), peer=peer, peer_libraries=sorted(mapped_libraries(peer['pid'])),
        command=command, runtime_env=chosen['runtime_env'], libraries=libraries, source_sha256=sources,
        model_records=records, global_admission_bytes=260_000_000_000, node_admission_bytes=60_000_000_000,
        repetitions=2, target_gb_s=250, capacity_gb_s=380, context=4096, workers_per_socket=15,
        raw_model_quant='UD-Q4_K_XL', native_runtime_precision_preserved=True,
        scope='One fresh Flash Q4 load with no draft model, two same-process raw prose/code pairs. '
            'Request overrides are not used to select raw mode. Max reasoning and no prompt reuse. '
            'The selected Q4/Q8-MTP2 command, full environment, cwd, affinity and libraries are restored.')
    verify_sources(plan)
    OUT.mkdir(); atomic_json(OUT / 'plan.json', plan)
    print(json.dumps(dict(prepared=True, peer_pid=peer['pid'], raw_port=PORT, repetitions=2)), flush=True)


def execute(lock_fd):
    plan = json.loads((OUT / 'plan.json').read_text())
    assert not (OUT / 'result.json').exists()
    verify_sources(plan)
    manager = Manager()
    original = manager.validate_current()
    assert original == plan['peer'] and set(inference_snapshot()) == {str(original['pid'])} and port_available(PORT)
    environment = process_environment(original['pid'])
    saved_cwd = process_info(original['pid'])['cwd']
    atomic_json(OUT / 'flash-restore-context.private.json', dict(environment=environment, cwd=saved_cwd, current=original))
    configuration = {k: v for k, v in original.items() if k not in {'pid', 'info', 'log'}}
    result = dict(started=time.time(), passed=False, controller_pid=os.getpid(), plan_sha256=sha256(OUT / 'plan.json'),
        stage='prepared', model_started=False, runs=[], restored=False, target_reached=False, runtime_promoted=False)
    cancelled, recovering, handoff_started = False, False, False
    model, child = None, None
    def save(stage=None):
        if stage: result['stage'] = stage
        atomic_json(OUT / 'result.json', result)
    def cancel(*_):
        nonlocal cancelled
        if not recovering: cancelled = True
    def check_cancel():
        if cancelled and not recovering: raise InterruptedError('Unload the owned Flash raw baseline and restore Flash Q4')
    for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGHUP]: signal.signal(sig, cancel)
    def wait_idle(guard, label):
        started = time.monotonic()
        while time.monotonic() - started < 15:
            check_cancel(); guard.assert_idle()
            atomic_json(OUT / (label + '.json'), dict(time=time.time(), quiet_seconds=time.monotonic() - started))
            time.sleep(1)

    def load(command,env,cwd,port,log_path,threshold,node_threshold):
        assert not inference_snapshot() and port_available(port)
        memory_gate(threshold,node_threshold)
        with log_path.open('w') as log:
            proc = subprocess.Popen(['taskset','-c','0-127',*command],env=env,cwd=cwd,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        deadline, report = time.monotonic()+3600, 0
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
        guard = ModelMeasurementGuard(original['pid'], {original['pid']: 18131}, inference_snapshot)
        wait_idle(guard, 'flash-idle')
        result['background_before_handoff'] = background(original['pid'])
        manager.validate_current(); verify_sources(plan); check_cancel()
        handoff_started = True
        save('unloading selected Flash Q4')
        manager.stop_flash(); check_cancel()
        result['memory_after_flash_stop'] = memory_gate(plan['global_admission_bytes'], plan['node_admission_bytes'])
        raw_environment = {k: v for k, v in os.environ.items()
            if k not in runtime_environment(os.environ) and not k.startswith('LLAMA_ARG_') and k != 'LD_PRELOAD'}
        raw_environment.update(plan['runtime_env'])
        save('loading Flash Q4 raw baseline')
        model, info = load(plan['command'], raw_environment, BASE, PORT, OUT / 'raw-model.log',
                           plan['global_admission_bytes'], plan['node_admission_bytes'])
        result.update(model_started=True, raw_pid=model.pid, raw_info=info)
        save()
        mapped = mapped_libraries(model.pid)
        for stem, path in plan['libraries'].items():
            actual = common_libraries(model.pid) if stem == 'libllama-common.so.' else {p for p in mapped if '/' + stem in p}
            assert actual == {str(Path(path).resolve())}, (stem, actual)
        result['mapped_libraries_verified'] = True
        result['loaded_libraries'] = sorted(mapped | common_libraries(model.pid))
        guard = ModelMeasurementGuard(model.pid, {model.pid: PORT}, inference_snapshot)
        baseline = None
        for index in range(plan['repetitions']):
            verify_sources(plan); wait_idle(guard, f'raw-{index}-idle')
            run = dict(index=index, started=time.time(), background_before=background(model.pid))
            result['runs'].append(run)
            label = f'flash-q4-raw-measure-0910-{index}'
            args = [sys.executable, '-u', str(BASE / 'measure-model-bandwidth.py'), label,
                '--port', str(PORT), '--pid', str(model.pid), '--alias', 'glm-flash-q4-raw-private', '--drafts', '0',
                '--tokens', '512', '--request-timeout-seconds', '600', '--allowed-idle-pids', '', '--skip-idle-gate',
                '--bandwidth-target-gb-s', '250', '--bandwidth-capacity-gb-s', '380',
                '--chat-template-kwargs', '{"reasoning_effort":"max"}', '--check-reasoning-budget-tokens', '0']
            save(f'measuring Flash Q4 raw repetition {index}')
            child = subprocess.Popen(args, pass_fds=(lock_fd,))
            while child.poll() is None:
                check_cancel(); memory_gate(32 << 30, 8 << 30)
                assert model.poll() is None
                time.sleep(.5)
            assert child.returncode == 0
            child = None
            path = BASE / 'results' / label / 'result.json'
            rows = read_measurement(path, plan)
            run.update(rows=rows, measurement=str(path), measurement_sha256=sha256(path),
                       background_after=background(model.pid), finished=time.time())
            if baseline is None: baseline = rows
            run['outputs_and_counts_match'] = all(rows[kind][key] == baseline[kind][key]
                for kind in ['prose', 'code'] for key in ['output_sha256', 'generated_tokens', 'draft_tokens', 'accepted_draft_tokens', 'cache_tokens'])
            save()
            assert run['outputs_and_counts_match'], 'Raw Flash raw repeat changed output or counts'
            print(json.dumps(dict(completed_repetition=index, rows=rows)), flush=True)
        result['target_reached'] = all(row['adjusted_gb_s'] >= 250 and row['counters']['adjacent_idle_qualifies']
            for run in result['runs'] for row in run['rows'].values())
        result['all_attribution_valid'] = all(row['counters']['adjacent_idle_qualifies'] for run in result['runs'] for row in run['rows'].values())
        result['all_outputs_match'] = True
        verify_sources(plan); guard.assert_idle()
        result['passed'] = True
    except BaseException as error:
        result.update(error=repr(error), error_traceback=traceback.format_exc())
        raise
    finally:
        recovering = True
        if child is not None and child.poll() is None:
            child.terminate()
            try: child.wait(timeout=20)
            except subprocess.TimeoutExpired: child.kill(); child.wait(timeout=10)
        if model is not None:
            if model.poll() is None:
                model.terminate()
                try: model.wait(timeout=45)
                except subprocess.TimeoutExpired: model.kill(); model.wait(timeout=15)
            result['owned_raw_exit'] = model.returncode
            result['owned_raw_stopped'] = model.poll() is not None
        if handoff_started and not inference_snapshot():
            save('restoring selected Flash Q4')
            try:
                restored, info = load(original['command'], environment, saved_cwd, 18131, OUT / 'restored-flash.log',
                                     260_000_000_000, 60_000_000_000)
                assert mapped_libraries(restored.pid) == set(plan['peer_libraries'])
                assert sha256(SELECTED) == plan['source_sha256'][str(SELECTED)]
                manager.state['current'] = dict(configuration, pid=restored.pid, info=info, log=str(OUT / 'restored-flash.log'))
                manager.record('flash_mtp2_restored_after_raw_baseline', pid=restored.pid, port=18131)
                manager.validate_current()
                result.update(restored=True, restored_flash_pid=restored.pid, restored_flash_start=info['start'],
                    full_environment_restored=True, restored_command_and_affinity=True, restored_libraries_match=True,
                    restored_service=read_service(18131))
            except BaseException as error:
                result.update(passed=False, restore_error=repr(error), finished=time.time())
                save('Flash restoration failed')
                raise
        elif handoff_started:
            try: manager.validate_current(); result['original_flash_preserved'] = True
            except BaseException as error:
                result.update(passed=False, restore_error='Competing or unidentified inference prevents restoration: ' + repr(error))
        result['finished'] = time.time()
        save('finished')


if __name__ == '__main__':
    os.umask(0o077)
    assert os.sched_getaffinity(0) == {127}
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['prepare', 'execute']); args = parser.parse_args()
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == 'prepare': prepare()
        else: execute(lock.fileno())
