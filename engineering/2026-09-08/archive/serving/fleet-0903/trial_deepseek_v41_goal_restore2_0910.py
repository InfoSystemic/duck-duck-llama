#!/usr/bin/env python3
"""Run one serialized DeepSeek CPU trial and restore its exact selected endpoint.

The full process environment stays in memory. A pidfd targets only the selected
DeepSeek process. An independent controller lock serializes the handoff of the
inherited fleet lifecycle lock from old server to controller to restored server.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import urllib.request

from glm_flash_q8_trial import memory_status
from model_measurement_guard import ModelMeasurementGuard
from qwen_high_quant_trial import atomic_json, port_available
from qwen_split_trial import ExactProcess, inference_snapshot, process_environment, process_info, sha256
from select_flash_q4_0910c import Manager

BASE = Path(__file__).resolve().parent
SELECTED = BASE / 'deepseek-v41-selected.json'
FLEET_LOCK = BASE / 'results/qwen-q6-trial-0907/lifecycle.lock'


def health():
    with urllib.request.urlopen('http://127.0.0.1:18170/health', timeout=3) as response:
        value = json.load(response)
    assert value['status'] == 'ok' and value['model'] == 'DeepSeek-V4.1-Flash'
    return value


def stop_child(proc):
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill(); proc.wait(timeout=30)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runner', type=Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--timeout', type=int, default=1200)
    args = parser.parse_args()
    assert os.sched_getaffinity(0) == {127}
    assert re.fullmatch(r'deepseek-v41-[a-z0-9-]+', args.label)
    runner = args.runner.resolve()
    assert runner.parent == BASE and runner.is_file() and '.private.' not in runner.name
    out = BASE / 'results' / args.label
    assert not out.exists() and 30 <= args.timeout <= 3600
    with (BASE / 'results/deepseek-v41-controller.lock').open('a') as controller_lock:
        fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        selected = json.loads(SELECTED.read_text())
        old = process_info(selected['pid'])
        assert old['start'] == selected['start']
        assert all(sha256(p) == h for p, h in selected['source_sha256'].items())
        command = list(old['command'])
        assert command[1] in [str(BASE / name) for name in ['deepseek_v41_server_0910.py', 'deepseek_v41_resident_server_0910.py', 'deepseek_v41_goal_server_0910.py']]
        assert command[command.index('--port') + 1] == '18170'
        assert command[command.index('--cache') + 1] == selected['temporary_ram_cache']
        server_fd = int(command[command.index('--lifecycle-lock-fd') + 1])
        assert os.stat(f'/proc/{selected["pid"]}/fd/{server_fd}').st_ino == FLEET_LOCK.stat().st_ino
        environment = process_environment(selected['pid'])
        peer = Manager().validate_current()
        guard = ModelMeasurementGuard(peer['pid'], {peer['pid']: 18131}, inference_snapshot)
        guard.assert_idle(); assert not health()['busy']
        out.mkdir(); (out / 'trial').mkdir()
        result = dict(passed=False, started=time.time(), old_server=dict(pid=selected['pid'], **old),
                      old_selected_sha256=sha256(SELECTED), peer_pid=peer['pid'], peer_start=peer['info']['start'],
                      source_sha256={str(p): sha256(p) for p in [Path(__file__), runner]},
                      restored=False, memory_before=memory_status())
        atomic_json(out / 'result.json', result)
        owned = None; restored = None; keep = False; stopped = False; recovering = False

        def cancel(*_):
            if not recovering:
                raise InterruptedError('Cancel owned DeepSeek trial and restore endpoint')

        for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGHUP]:
            signal.signal(sig, cancel)
        exact = ExactProcess(selected['pid'], dict(start_ticks=old['start'], exe=old['exe']), {peer['pid']})
        with FLEET_LOCK.open('a') as lock:
            acquired = False
            try:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    raise AssertionError('Selected server must hold the fleet lock')
                except BlockingIOError:
                    pass
                assert not health()['busy']; guard.assert_idle()
                exact.terminate()
                deadline = time.monotonic() + 30
                while not exact.exited():
                    assert time.monotonic() < deadline, 'Selected server did not stop'
                    time.sleep(.1)
                stopped = True
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); acquired = True
                guard.assert_idle(); assert port_available(18170)
                trial_command = ['taskset', '-c', ','.join(map(str, old['affinity'])), command[0], str(runner),
                                 '--cache', selected['temporary_ram_cache'], '--output', str(out / 'trial')]
                with (out / 'trial.log').open('w') as log:
                    owned = subprocess.Popen(trial_command, cwd=old['cwd'], env=environment, stdin=subprocess.DEVNULL,
                                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                result['trial_pid'] = owned.pid
                atomic_json(out / 'result.json', result)
                deadline = time.monotonic() + args.timeout; report = 0
                while owned.poll() is None:
                    guard.assert_idle()
                    assert memory_status()['MemAvailable'] > 100 << 30
                    assert time.monotonic() < deadline, 'Owned trial timed out'
                    if time.monotonic() - report >= 30:
                        print(json.dumps(dict(trial_pid=owned.pid, running=args.label)), flush=True)
                        report = time.monotonic()
                    time.sleep(.5)
                result['trial_exit'] = owned.returncode
                trial = json.loads((out / 'trial/result.json').read_text())
                assert owned.returncode == 0 and trial['passed'], 'Owned trial failed'
                assert all(sha256(p) == h for p, h in result['source_sha256'].items())
                result['trial_passed'] = True
            except BaseException as error:
                result['error'] = str(error) if isinstance(error, AssertionError) else type(error).__name__
            finally:
                recovering = True
                stop_child(owned)
                if not stopped and exact.termination_sent:
                    stopped = exact.exited()
                exact.close()
                try:
                    if stopped:
                        if not acquired:
                            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); acquired = True
                        guard.assert_idle(); assert port_available(18170)
                        assert all(sha256(p) == h for p, h in selected['source_sha256'].items())
                        destination = out / 'server'; destination.mkdir()
                        command[command.index('--output') + 1] = str(destination)
                        command[command.index('--lifecycle-lock-fd') + 1] = str(lock.fileno())
                        with (destination / 'server.log').open('w') as log:
                            restored = subprocess.Popen(['taskset', '-c', ','.join(map(str, old['affinity'])), *command],
                                cwd=old['cwd'], env=environment, stdin=subprocess.DEVNULL, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock.fileno(),))
                        result['restored_pid'] = restored.pid
                        atomic_json(out / 'result.json', result)
                        deadline = time.monotonic() + 600; report = 0
                        while not (destination / 'ready.json').exists():
                            assert restored.poll() is None and time.monotonic() < deadline, 'Endpoint restart failed'
                            guard.assert_idle()
                            if time.monotonic() - report >= 30:
                                print(json.dumps(dict(restoring_endpoint=restored.pid)), flush=True); report = time.monotonic()
                            time.sleep(.5)
                        info = process_info(restored.pid)
                        assert info['command'] == command and info['affinity'] == old['affinity']
                        assert not health()['busy']
                        request = urllib.request.Request('http://127.0.0.1:18170/v1/chat/completions',
                            data=json.dumps(dict(model='DeepSeek-V4.1-Flash', messages=[dict(role='user', content='Hi.')],
                                                 temperature=0, max_tokens=16)).encode(), headers={'Content-Type': 'application/json'})
                        print(json.dumps(dict(validating_restored_endpoint=restored.pid)), flush=True)
                        with urllib.request.urlopen(request, timeout=600) as response:
                            reply = json.load(response)
                        assert reply['choices'][0]['message']['content'] == 'Hello! How can I help you today?'
                        assert reply['choices'][0]['finish_reason'] == 'stop' and reply['usage']['completion_tokens'] == 10
                        assert reply['timings']['downloaded_bytes'] == 0
                        guard.assert_idle(); assert not health()['busy']
                        assert process_info(peer['pid'])['start'] == peer['info']['start']
                        result.update(restored=True, restored_server=info, validation=reply,
                                      peer_preserved=True, memory_after=memory_status())
                        selected.update(pid=restored.pid, start=info['start'], evidence=str(out / 'result.json'),
                                        request_record=str(destination / 'last-request.json'),
                                        restoration_sources=result['source_sha256'])
                        atomic_json(SELECTED, selected)
                        keep = True
                    else:
                        result['original_server_preserved'] = process_info(selected['pid'])['start'] == old['start']
                except BaseException as error:
                    result['restoration_error'] = str(error) if isinstance(error, AssertionError) else type(error).__name__
                    raise
                finally:
                    if not keep:
                        stop_child(restored)
                    result['passed'] = bool(result.get('trial_passed') and result['restored'])
                    result['finished'] = time.time()
                    atomic_json(out / 'result.json', result)
                    print(json.dumps({k: result[k] for k in ['passed', 'restored', 'restored_pid', 'error', 'restoration_error'] if k in result}), flush=True)
        if not result['passed']:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
