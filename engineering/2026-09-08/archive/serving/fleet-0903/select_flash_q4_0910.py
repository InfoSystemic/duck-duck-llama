#!/usr/bin/env python3
"""Validate and select the explicitly requested Flash Q4 model, with rollback."""
import argparse
import copy
import fcntl
import hashlib
import http.client
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request

from analyze_flash_bandwidth250_0910 import validate_counters
from benchmark_qwen_q6 import wait_background
from flash_hugepages250_trial_0910 import output_record, stop_owned_flash
from glm_flash_q8_trial import Manager as FlashManager, memory_status, node_memory_status
from model_measurement_guard import ModelMeasurementGuard, read_service
from qwen_high_quant_trial import atomic_json, expected, port_available, set_option, unit_state
from qwen_split_trial import ExactProcess, inference_snapshot, process_environment, process_info, runtime_environment, sha256

BASE = Path(__file__).resolve().parent
OUT = BASE / 'results/glm-flash-q4-selection-0910'
STATE = OUT / 'state.json'
PORT = 18131
SELECTED = BASE / 'glm-flash-selected.json'


class Manager(FlashManager):
    def __init__(self):
        super().__init__()
        self.state = json.loads(STATE.read_text()) if STATE.exists() else dict(events=[], current=None)

    def check_cancel(self):
        if self.cancelled and not self.recovering:
            raise InterruptedError('Flash Q4 selection cancelled; restore Qwen')

    def record(self, event, **values):
        row = dict(time=time.time(), event=event, **values)
        self.state['events'].append(row)
        atomic_json(STATE, self.state)
        print(json.dumps(row), flush=True)

    def quiet(self):
        if self.state.get('current'):
            current = self.validate_current()
            guard = ModelMeasurementGuard(current['pid'], {current['pid']: PORT}, inference_snapshot)
            guard.wait_idle(OUT / 'waiting-for-idle.json', quiet_seconds=15)
            guard.assert_idle()
        else:
            assert not inference_snapshot()
        self.check_cancel()

    def start_flash(self, configuration, environment):
        assert not inference_snapshot() and port_available(PORT)
        assert self.qwen.state['full_stopped'] and unit_state()['ActiveState'] == 'inactive'
        before = memory_status()
        # Includes target repacking, the draft, work buffers, and a reserve.
        # tmpfs file backing is already excluded from MemAvailable.
        assert before['MemAvailable'] > 260000000000, before
        node_before = node_memory_status()
        assert len(node_before) == 4
        assert all(v['estimated_available'] > 60000000000 for v in node_before.values()), node_before
        self.check_cancel()
        log_path = OUT / f'flash-{time.time_ns()}.log'
        with log_path.open('w') as log:
            proc = subprocess.Popen(['taskset', '-c', ','.join(map(str, configuration['affinity'])), *configuration['command']],
                env=environment, cwd=BASE, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
        self.record('loading_flash', pid=proc.pid, batching=configuration['batching'],
                    drafts=configuration['drafts'], log=str(log_path), memory_before=before, node_memory_before=node_before)
        try:
            deadline, report = time.monotonic() + 3600, 0
            while True:
                self.check_cancel()
                assert proc.poll() is None, ('Flash exited while loading', proc.returncode)
                assert not (set(inference_snapshot()) - {str(proc.pid)}), 'Another inference process loaded'
                memory = memory_status()
                assert memory['MemAvailable'] > (32 << 30), ('Flash load exhausted its RAM reserve', memory)
                node_memory = node_memory_status()
                assert all(v['estimated_available'] > (8 << 30) for v in node_memory.values()), ('Flash reached a per-node loading reserve', node_memory)
                with log_path.open('rb') as log:
                    log.seek(max(0, log_path.stat().st_size - 32768))
                    tail = log.read()
                assert not (b'GGML_ASSERT' in tail and b'failed' in tail), 'Flash asserted while loading'
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=2) as response:
                        if json.load(response).get('status') == 'ok':
                            break
                except OSError:
                    pass
                now = time.monotonic()
                assert now < deadline, 'Flash load exceeded one hour'
                if now - report >= 30:
                    self.record('waiting_for_flash', pid=proc.pid, memory=memory, node_memory=node_memory, log=str(log_path))
                    report = now
                time.sleep(1)
            info = process_info(proc.pid)
            assert info['command'] == configuration['command']
            assert info['affinity'] == configuration['affinity']
            assert runtime_environment(process_environment(proc.pid)) == configuration['runtime_env']
            mapped = Path(f'/proc/{proc.pid}/maps').read_text().splitlines()
            for stem, wanted in [('libggml-cpu.so.', Path(configuration['cpu_library'])),
                                 ('libllama.so.', Path(configuration['pinned_directory']) / 'libllama.so.0.3.0')]:
                actual = {row.split()[-1] for row in mapped if '/' + stem in row}
                assert actual == {str(wanted.resolve())}, actual
            assert sha256(configuration['cpu_library']) == configuration['cpu_sha256']
            current = dict(configuration, pid=proc.pid, info=info, log=str(log_path))
            self.state['current'] = current
            self.record('flash_ready', pid=proc.pid, port=PORT, memory=memory_status(), node_memory=node_memory_status(), service=read_service(PORT))
        except BaseException as error:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=30)
            self.state['current'] = None
            self.record('flash_load_failed', pid=proc.pid, exit_code=proc.returncode, error=repr(error), log=str(log_path))
            raise



def mapped_libraries(pid):
    return {row.split()[-1] for row in Path(f'/proc/{pid}/maps').read_text().splitlines()
            if any('/' + stem in row for stem in ('libggml-', 'libggml.so.', 'libllama.so.'))}


def verify_records(records):
    for record in records:
        st = Path(record['path']).stat()
        assert (st.st_size, st.st_ino, st.st_mtime_ns) == (record['bytes'], record['inode'], record['mtime_ns']), record['path']


def verify_plan(plan):
    assert all(sha256(path) == digest for path, digest in plan['source_sha256'].items())
    verify_records(plan['model_records'])
    assert unit_state()['ActiveState'] == 'inactive'


def verify_peer(plan):
    pid = plan['qwen_pid']
    actual = process_info(pid)
    assert all(actual[k] == plan['qwen'][k] for k in ('start', 'exe', 'command', 'cwd', 'affinity'))
    assert runtime_environment(process_environment(pid)) == plan['qwen_runtime_env']
    assert mapped_libraries(pid) == set(plan['qwen_libraries'])
    assert set(inference_snapshot()) == {str(pid)}


def prepare():
    assert not OUT.exists() and not SELECTED.exists()
    download_path = BASE / 'results/glm-flash-q4-download-0910/status.json'
    download = json.loads(download_path.read_text())
    assert download['complete'] and not download['errors'] and len(download['verified']) == 6
    assert download['quant'] == 'UD-Q4_K_XL' and download['expected_bytes'] == 199707321347
    manifest = BASE / 'results/glm-flash-q4-manifest-0910.json'
    assert sha256(manifest) == download['manifest_sha256']
    records = [download['verified'][record['name']] for record in download['records']]
    assert all(row['sha256'] == item['sha256'] and row['bytes'] == item['bytes']
               for row, item in zip(records, download['records']))
    verify_records(records)
    mtp_path = BASE / 'results/glm-flash-q8-mtp-0908/result.json'
    mtp = json.loads(mtp_path.read_text())
    assert mtp['passed'] and mtp['revision'] == download['revision']
    draft = dict(path=mtp['destination'], bytes=mtp['bytes'], inode=mtp['inode'],
                 mtime_ns=mtp['mtime_ns'], sha256=mtp['sha256'])
    records.append(draft)
    verify_records(records)
    parent_path = BASE / 'results/flash-bandwidth250-comparison-0909/plan.json'
    parent_result = BASE / 'results/flash-bandwidth250-assessment-0910.json'
    assessment = json.loads(parent_result.read_text())
    assert assessment['passed']
    parent = json.loads(parent_path.read_text())
    baseline = parent['configurations']['mtp_control']
    assert baseline['model_revision'] == download['revision'] and baseline['draft_sha256'] == mtp['sha256']
    configs = {}
    for name, drafts in [('raw', 0), ('mtp2', 2)]:
        config = copy.deepcopy(baseline)
        command = set_option(config['command'], '--model', records[0]['logical_path'])
        command = set_option(command, '--alias', 'glm-flash-goal,glm-flash-q4,GLM-5.3-Flash')
        if not drafts:
            for flag in [value for value in command if value.startswith('--spec-')]:
                command = set_option(command, flag, None)
        config.update(command=command, drafts=drafts, quant='UD-Q4_K_XL',
                      model_manifest_sha256=download['manifest_sha256'])
        assert config['workers'] == 15 and config['affinity'] == list(range(128))
        assert not any('REQUANT' in k or 'PROFILE' in k for k in config['runtime_env'])
        assert config['runtime_env']['GGML_CPU_NUMA_HUGEPAGES'] == '0' and not config['dissemination']
        configs[name] = config
    handoff_path = BASE / 'results/flash-hugepages250-comparison-0910/result.json'
    handoff = json.loads(handoff_path.read_text())
    assert handoff['finished'] and handoff['qwen_restored'] and not handoff.get('restoration_error')
    pid = handoff['restored_qwen_pid']
    peer = process_info(pid)
    assert peer['start'] == handoff['restored_qwen']['start']
    libraries = mapped_libraries(pid)
    runtime_paths = {Path(baseline['cpu_library']), Path(baseline['command'][0]).resolve()}
    runtime_paths.update(path.resolve() for path in Path(baseline['pinned_directory']).iterdir()
                         if path.is_file() and ('.so' in path.name or path.name == 'llama-server'))
    assert sha256(baseline['cpu_library']) == baseline['cpu_sha256']
    inputs = [Path(__file__).resolve(), download_path, manifest, mtp_path, parent_path, parent_result, handoff_path,
              BASE / 'glm_flash_q8_trial.py', BASE / 'flash_hugepages250_trial_0910.py',
              BASE / 'analyze_flash_bandwidth250_0910.py', BASE / 'benchmark_qwen_q6.py',
              BASE / 'measure-model-bandwidth.py', BASE / 'dram_bandwidth.py',
              BASE / 'guarded_inference_request.py', BASE / 'model_measurement_guard.py',
              BASE / 'inference_contention_guard.py', BASE / 'qwen_high_quant_trial.py',
              BASE / 'qwen_split_trial.py', BASE / 'launch-glm-flash-validated.py',
              *runtime_paths, *map(Path, libraries)]
    plan = dict(time=time.time(), user_authorized_quant='UD-Q4_K_XL', configurations=configs,
                sequence=['raw', 'mtp2', 'mtp2'], selected_configuration='mtp2',
                model_records=records, runtime_sha256={str(p): sha256(p) for p in runtime_paths},
                qwen_pid=pid, qwen=peer, qwen_runtime_env=runtime_environment(process_environment(pid)),
                qwen_libraries={p: sha256(p) for p in libraries},
                source_sha256={str(p): sha256(p) for p in inputs}, target_gb_s=250, capacity_gb_s=380,
                scope='Explicitly selected Q4 target, retained Q8 draft, fresh single-conversation decode. '
                      'One raw run and two MTP2 repeats. Preserve Qwen Q6 configuration, stop it for Flash, '
                      'and restore its exact process context if selection fails. Keep Flash Q4 running on success. '
                      'Q4 task quality is not certified equivalent to Q8 or the released checkpoint.')
    verify_plan(plan)
    verify_peer(plan)
    assert port_available(PORT) and Manager().state['current'] is None
    OUT.mkdir()
    atomic_json(OUT / 'plan.json', plan)
    print(json.dumps(dict(prepared=True, sequence=plan['sequence'], qwen_pid=pid,
                         model_bytes=download['expected_bytes'], selected_quant='UD-Q4_K_XL')), flush=True)


def measurement_rows(path, current):
    data = json.loads(path.read_text())
    assert data['finished'] and data['input_integrity_verified'] and not data.get('error')
    assert data['server_command'] == current['command'] and data['runtime_env'] == current['runtime_env']
    assert data['bandwidth_target_gb_s'] == 250 and data['bandwidth_capacity_gb_s'] == 380
    assert len(data['checks']) == 2 and all(check['pass_check'] and not check['abort'] for check in data['checks'])
    assert all(sha256(source) == digest for source, digest in data['input_sha256'].items())
    rows = {}
    for row in data['measurements']:
        assert row['counter_metadata']['valid'] and row['counter_metadata']['exit_code'] == 0
        assert not row['abort'] and not row['inference_churn'] and not row['other_inference']
        assert all(row[key]['valid'] for key in ['baseline_before', 'baseline_after', 'decode'])
        assert row['timings']['cache_n'] == 0 and 128 <= row['timings']['predicted_n'] <= 512
        assert row['draft_n'] == current['drafts']
        assert bool(row['timings'].get('draft_n', 0)) == bool(current['drafts'])
        before, after = row['baseline_before']['total_gb_s'], row['baseline_after']['total_gb_s']
        assert max(before, after) <= 19 and abs(before-after) <= 9.5, 'Idle traffic does not qualify attribution'
        chunks = path.parent / f"{row['kind']}-draft{row['draft_n']}" / 'chunks.json'
        rows[row['kind']] = dict(tok_s=row['timings']['predicted_per_second'],
            adjusted_gb_s=row['background_subtracted_gb_s'],
            over_250_gb_s=row['background_subtracted_gb_s'] >= 250,
            output_sha256=output_record(json.loads(chunks.read_text())), chunks_sha256=sha256(chunks),
            generated_tokens=row['timings']['predicted_n'], draft_tokens=row['timings'].get('draft_n', 0),
            accepted_draft_tokens=row['timings'].get('draft_n_accepted', 0),
            completed_answer=row['completed_answer'])
    assert set(rows) == {'prose', 'code'}
    return rows


def request_checks(current):
    guard = ModelMeasurementGuard(current['pid'], {current['pid']: PORT}, inference_snapshot)

    def request(endpoint, payload):
        guard.assert_idle()
        guard.reset_activity()
        connection = http.client.HTTPConnection('127.0.0.1', PORT, timeout=180)
        connection.connect()
        own_socket = connection.sock
        stop, failures = threading.Event(), []
        def monitor():
            while not stop.wait(.5):
                try:
                    reason = guard.abort_reason()
                except Exception as error:
                    reason = repr(error)
                if reason:
                    failures.append(reason)
                    try:
                        own_socket.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    return
        watcher = threading.Thread(target=monitor, daemon=True)
        watcher.start()
        try:
            connection.request('POST', '/' + endpoint, json.dumps(payload), {'Content-Type': 'application/json'})
            response = connection.getresponse()
            assert response.status == 200
            data = json.load(response)
            assert not failures
            return data
        finally:
            stop.set()
            connection.close()
            watcher.join(timeout=5)
            assert not watcher.is_alive()

    formatted = request('apply-template', dict(messages=[dict(role='user', content='Explain how a refrigerator moves heat. Give a detailed explanation in plain English.')],
                         chat_template_kwargs={'reasoning_effort': 'max'}))['prompt']
    prompt = request('tokenize', dict(content=formatted, add_special=False, parse_special=True))['tokens']
    common = dict(temperature=0, seed=42, return_tokens=True, n_predict=16, cache_prompt=False)
    prime = request('completion', dict(common, prompt=prompt, n_predict=24))
    full = prompt + prime['tokens']
    checks = []
    for name, tokens in [('repeat', prompt), ('extend', full), ('trim_one', full[:-1]), ('trim_three', full[:-3])]:
        fresh = request('completion', dict(common, prompt=tokens))
        request('completion', dict(common, prompt=prompt, n_predict=24))
        payload = dict(common, prompt=tokens)
        payload.pop('cache_prompt')
        default = request('completion', payload)
        passed = bool(fresh['tokens']) and fresh['tokens'] == default['tokens'] and fresh['timings']['cache_n'] == default['timings']['cache_n'] == 0
        checks.append(dict(name=name, passed=passed, fresh=fresh, default=default))
        atomic_json(OUT / 'request-checks.json', dict(checks=checks, current=current))
        print(json.dumps(dict(request_check=name, passed=passed)), flush=True)
        assert passed, 'Default request differs from fresh evaluation'
    return checks


def execute(lock_fd):
    plan = json.loads((OUT / 'plan.json').read_text())
    assert not (OUT / 'result.json').exists() and not SELECTED.exists()
    verify_plan(plan)
    verify_peer(plan)
    manager = Manager()
    result = dict(started=time.time(), controller_pid=os.getpid(), passed=False, selected=False,
                  runs=[], plan_sha256=sha256(OUT / 'plan.json'))
    def save(stage):
        result['stage'] = stage
        atomic_json(OUT / 'result.json', result)
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda *_: setattr(manager, 'cancelled', True))
    original, environment = None, None
    success = False
    try:
        guard = ModelMeasurementGuard(plan['qwen_pid'], {plan['qwen_pid']: 18095}, inference_snapshot)
        save('waiting for idle Qwen')
        guard.wait_idle(OUT / 'qwen-idle.json')
        wait_background(guard, plan['qwen_pid'], 4, OUT / 'background-wait.json')
        verify_plan(plan)
        verify_peer(plan)
        environment = process_environment(plan['qwen_pid'])
        with (OUT / 'qwen-restore-context.private.json').open('x') as handle:
            json.dump(dict(info=plan['qwen'], environment=environment), handle)
            handle.write('\n')
        assert (OUT / 'qwen-restore-context.private.json').stat().st_mode & 0o777 == 0o600
        original = ExactProcess(plan['qwen_pid'], expected(plan['qwen']), set())
        guard.assert_idle()
        manager.check_cancel()
        original.terminate()
        save('stopping idle Qwen for the selected Flash model')
        deadline = time.monotonic() + 180
        while not original.exited() or inference_snapshot():
            assert time.monotonic() < deadline
            time.sleep(.5)
        assert port_available(18095)
        for index, name in enumerate(plan['sequence']):
            manager.check_cancel()
            desired = plan['configurations'][name]
            if manager.state['current'] is None or manager.state['current']['drafts'] != desired['drafts']:
                stop_owned_flash(manager)
                verify_plan(plan)
                env = {k: v for k, v in os.environ.items() if k not in runtime_environment(os.environ)}
                env.update(desired['runtime_env'])
                manager.start_flash(desired, env)
            current = manager.validate_current()
            guard = ModelMeasurementGuard(current['pid'], {current['pid']: PORT}, inference_snapshot)
            guard.wait_idle(OUT / 'decode-idle.json', quiet_seconds=15)
            wait_background(guard, current['pid'], 4, OUT / 'decode-background.json')
            label = f'glm-flash-q4-selected-0910-{index:02d}-{name}'
            command = [sys.executable, '-u', str(BASE / 'measure-model-bandwidth.py'), label,
                       '--port', str(PORT), '--pid', str(current['pid']), '--alias', 'glm-flash-q4',
                       '--drafts', str(current['drafts']), '--tokens', '512', '--request-timeout-seconds', '240',
                       '--allowed-idle-pids', '', '--skip-idle-gate', '--bandwidth-target-gb-s', '250',
                       '--bandwidth-capacity-gb-s', '380', '--chat-template-kwargs', '{"reasoning_effort":"max"}',
                       '--check-reasoning-budget-tokens', '0']
            run = dict(index=index, configuration=name, model_pid=current['pid'], started=time.time(), command=command)
            result['runs'].append(run)
            save('measuring ' + label)
            child = subprocess.Popen(command, pass_fds=(lock_fd,))
            try:
                while child.poll() is None:
                    manager.check_cancel()
                    time.sleep(.5)
                assert child.returncode == 0
            finally:
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=15)
            path = BASE / 'results' / label / 'result.json'
            rows = measurement_rows(path, current)
            data = json.loads(path.read_text())
            counters = {row['kind']: validate_counters(path.parent / f"{row['kind']}-draft{row['draft_n']}", row)
                        for row in data['measurements']}
            prior = next((x for x in result['runs'] if x['configuration'] == name and 'rows' in x), None)
            if prior:
                assert all(rows[k][field] == prior['rows'][k][field] for k in rows
                           for field in ('output_sha256', 'generated_tokens', 'draft_tokens', 'accepted_draft_tokens'))
            run.update(finished=time.time(), measurement=str(path), measurement_sha256=sha256(path), rows=rows, counters=counters)
            save('measured ' + label)
            print(json.dumps(dict(configuration=name, rows=rows)), flush=True)
        result['request_checks'] = request_checks(manager.validate_current())
        verify_plan(plan)
        current = manager.validate_current()
        assert current['drafts'] == 2 and set(inference_snapshot()) == {str(current['pid'])}
        assert port_available(18095)
        config = {k: copy.deepcopy(v) for k, v in current.items() if k not in ('pid', 'info', 'log')}
        config.update(runtime_sha256=plan['runtime_sha256'], model_records=plan['model_records'],
                      selection_time=time.time(), evidence=str(OUT / 'result.json'),
                      user_authorized_quant='UD-Q4_K_XL',
                      quality_scope='Q4 selected by user; checkpoint-equivalent task quality is not established.')
        atomic_json(SELECTED, config)
        result.update(passed=True, selected=True, selected_config=str(SELECTED), selected_config_sha256=sha256(SELECTED),
                      selected_pid=current['pid'], selected_start=current['info']['start'],
                      qwen_q6_configuration_preserved=True, qwen_stopped_for_flash=True,
                      target_reached=all(row['over_250_gb_s'] for run in result['runs'] if run['configuration'] == 'mtp2' for row in run['rows'].values()),
                      all_model_goal_complete=False, memory=memory_status(), node_memory=node_memory_status())
        save('Flash Q4 selected and running')
        success = True
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        manager.recovering = True
        try:
            if not success:
                if SELECTED.exists() and result.get('selected_config_sha256') == sha256(SELECTED):
                    SELECTED.unlink()
                stop_owned_flash(manager)
                if original is not None and original.termination_sent:
                    save('restoring exact Qwen configuration after incomplete selection')
                    if not original.exited():
                        verify_peer(plan)
                        result['qwen_original_still_present'] = True
                    else:
                        assert not inference_snapshot() and port_available(18095) and port_available(PORT)
                        assert environment is not None and memory_status()['MemAvailable'] > 230000000000
                        with (OUT / 'qwen-restored.log').open('w') as log:
                            restored = subprocess.Popen(['taskset', '-c', ','.join(map(str, plan['qwen']['affinity'])), *plan['qwen']['command']],
                                env=environment, cwd=plan['qwen']['cwd'], stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                        result['restored_qwen_pid'] = restored.pid
                        save('restoring Qwen')
                        deadline, report = time.monotonic() + 1200, 0
                        while True:
                            assert restored.poll() is None and time.monotonic() < deadline
                            assert set(inference_snapshot()) == {str(restored.pid)}
                            assert memory_status()['MemAvailable'] > 32 << 30
                            try:
                                with urllib.request.urlopen('http://127.0.0.1:18095/health', timeout=2) as response:
                                    if json.load(response).get('status') == 'ok':
                                        break
                            except OSError:
                                pass
                            if time.monotonic() - report >= 30:
                                print(json.dumps(dict(restoring_qwen_pid=restored.pid)), flush=True)
                                report = time.monotonic()
                            time.sleep(1)
                        actual = process_info(restored.pid)
                        assert all(actual[k] == plan['qwen'][k] for k in ('exe', 'command', 'cwd', 'affinity'))
                        assert process_environment(restored.pid) == environment
                        assert mapped_libraries(restored.pid) == set(plan['qwen_libraries'])
                        result.update(qwen_restored=True, restored_qwen=actual)
        except BaseException as error:
            result['restoration_error'] = repr(error)
            raise
        finally:
            if original is not None:
                original.close()
            result['finished'] = time.time()
            save('Flash Q4 selected and running' if success else 'selection incomplete; inspect restoration')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run'])
    args = parser.parse_args()
    os.umask(0o077)
    assert os.sched_getaffinity(0) == {127}
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare() if args.action == 'prepare' else execute(lock.fileno())
