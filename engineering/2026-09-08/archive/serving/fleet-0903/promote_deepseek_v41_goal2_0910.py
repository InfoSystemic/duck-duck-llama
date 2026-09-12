#!/usr/bin/env python3
"""Promote the fastest exact quiet-trial goal2 configuration after live validation.

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
from goal2_lifecycle_safety_0910 import settle_termination, wait_peer_idle, wait_endpoint_idle
from goal2_promotion_selection_0910 import FLAGS, FLAG_EXACT16, FLAG_SPARSE, select_candidate

BASE = Path(__file__).resolve().parent
SELECTED = BASE / 'deepseek-v41-selected.json'
FLEET_LOCK = BASE / 'results/qwen-q6-trial-0907/lifecycle.lock'
QUIET_EVIDENCE = BASE / 'results/deepseek-v41-goal-exact16-quiet-0910/trial/result.json'
GOAL2_ENTRY = BASE / 'deepseek_v41_goal2_server_0910.py'


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
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--timeout', type=int, default=1200)
    args = parser.parse_args()
    assert os.sched_getaffinity(0) == {32}, 'Promotion must avoid model cores and their SMT siblings'
    assert re.fullmatch(r'deepseek-v41-[a-z0-9-]+', args.label)
    evidence = args.evidence.resolve()
    assert evidence == QUIET_EVIDENCE, 'Only the completed quiet exact16 trial can authorize this promotion'
    validated = json.loads(evidence.read_text())
    choice = select_candidate(validated)
    assert all(sha256(p) == h for p,h in validated['source_sha256'].items())
    assert all(sha256(p) == h for p,h in validated['protected_selected_sources'].items())
    controller_evidence = evidence.parent.parent / 'result.json'
    completed = json.loads(controller_evidence.read_text())
    assert completed['passed'] and completed['restored'] and completed['peer_preserved'], 'Quiet controller has not completed restoration'
    golden_path = BASE / 'results/deepseek-v41-checkpoint-run-0910b/generation.json'
    golden = json.loads(golden_path.read_text())['runs'][0]
    for row in validated['runs']:
        assert row['token_ids'] == golden['token_ids']
        assert row['logits_sha256'] == [step['logits_sha256'] for step in golden['steps']]
    baseline_tps = choice['baseline_decode_tok_s']
    winner = choice['winner']
    candidate_tps = winner['decode_tok_s']
    out = BASE / 'results' / args.label
    assert not out.exists() and 30 <= args.timeout <= 3600
    with (BASE / 'results/deepseek-v41-controller.lock').open('a') as controller_lock:
        print(json.dumps(dict(waiting_for_deepseek_promotion_lock=True, candidate=winner['name'])), flush=True)
        fcntl.flock(controller_lock, fcntl.LOCK_EX)
        selected = json.loads(SELECTED.read_text())
        old = process_info(selected['pid'])
        assert old['start'] == selected['start']
        assert validated['affinity'] == old['affinity'], 'Quiet measurement used a different CPU set'
        assert all(sha256(p) == h for p, h in selected['source_sha256'].items())
        command = list(old['command'])
        assert command[1] in [str(BASE / name) for name in ['deepseek_v41_server_0910.py','deepseek_v41_resident_server_0910.py','deepseek_v41_goal_server_0910.py','deepseek_v41_lattice16_server_0910.py','deepseek_v41_goal2_server_0910.py']]
        assert command[command.index('--port') + 1] == '18170'
        assert command[command.index('--cache') + 1] == selected['temporary_ram_cache']
        server_fd = int(command[command.index('--lifecycle-lock-fd') + 1])
        assert os.stat(f'/proc/{selected["pid"]}/fd/{server_fd}').st_ino == FLEET_LOCK.stat().st_ino
        environment = process_environment(selected['pid'])
        assert environment.get('OMP_NUM_THREADS') == environment.get('MKL_NUM_THREADS') == '16'
        assert environment.get('OMP_WAIT_POLICY') == 'PASSIVE'
        candidate_environment = dict(environment, **winner['environment'])
        old_goal2_flags = {key: environment[key] for key in FLAGS if key in environment}
        peer = Manager().validate_current()
        guard = ModelMeasurementGuard(peer['pid'], {peer['pid']: 18131}, inference_snapshot)
        guard.assert_idle(); assert not health()['busy']
        out.mkdir(); (out / 'trial').mkdir()
        candidate_sources = [GOAL2_ENTRY, BASE/'goal2_promotion_selection_0910.py', BASE/'trial_deepseek_v41_goal2_queue_0910.py', BASE/'goal2_lifecycle_safety_0910.py'] + [Path(p) for p in validated['source_sha256']]
        candidate_hashes = {str(p): sha256(p) for p in candidate_sources}
        result = dict(passed=False, started=time.time(), candidate_source_sha256=candidate_hashes, old_server=dict(pid=selected['pid'], **old),
                      old_selected_sha256=sha256(SELECTED), peer_pid=peer['pid'], peer_start=peer['info']['start'],
                      source_sha256={str(p): sha256(p) for p in [Path(__file__), evidence, controller_evidence, golden_path]},
                      selection=choice, candidate_environment=winner['environment'], old_goal2_environment=old_goal2_flags,
                      restored=False, memory_before=memory_status())
        atomic_json(out / 'result.json', result)
        owned = None; restored = None; keep = False; stopped = False; recovering = False; promote = False

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
                assert all(sha256(p) == h for p,h in validated['source_sha256'].items())
                result.update(trial_passed=True, benchmark_evidence=str(evidence), baseline_decode_tok_s=baseline_tps,
                              candidate_decode_tok_s=candidate_tps, speedup=candidate_tps/baseline_tps)
                promote=True
            except BaseException as error:
                result['error'] = str(error) if isinstance(error, AssertionError) else type(error).__name__
            finally:
                recovering = True
                stop_child(owned)
                stopped = settle_termination(exact, stopped)
                exact.close()
                try:
                    if stopped:
                        if not acquired:
                            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); acquired = True
                        wait_peer_idle(guard); assert port_available(18170)
                        assert all(sha256(p) == h for p, h in selected['source_sha256'].items())
                        for attempt, choose_candidate in enumerate([True, False] if promote else [False]):
                            promote = choose_candidate
                            try:
                                destination = out / ('server' if attempt == 0 else 'fallback-server'); destination.mkdir()
                                if promote:
                                    assert all(sha256(p) == h for p, h in candidate_hashes.items())
                                command = list(old['command'])
                                command[1] = str(GOAL2_ENTRY) if promote else old['command'][1]
                                attempt_environment = candidate_environment if promote else environment
                                command[command.index('--output') + 1] = str(destination)
                                command[command.index('--lifecycle-lock-fd') + 1] = str(lock.fileno())
                                with (destination / 'server.log').open('w') as log:
                                    restored = subprocess.Popen(['taskset', '-c', ','.join(map(str, old['affinity'])), *command],
                                        cwd=old['cwd'], env=attempt_environment, stdin=subprocess.DEVNULL, stdout=log,
                                        stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock.fileno(),))
                                result['restored_pid'] = restored.pid
                                atomic_json(out / 'result.json', result)
                                deadline = time.monotonic() + 600; report = 0
                                while not (destination / 'ready.json').exists():
                                    assert restored.poll() is None and time.monotonic() < deadline, 'Endpoint restart failed'
                                    wait_peer_idle(guard)
                                    if time.monotonic() - report >= 30:
                                        print(json.dumps(dict(restoring_endpoint=restored.pid)), flush=True); report = time.monotonic()
                                    time.sleep(.5)
                                info = process_info(restored.pid)
                                assert info['command'] == command and info['affinity'] == old['affinity']
                                assert info['exe'] == old['exe'] and info['cwd'] == old['cwd']
                                assert process_environment(restored.pid) == attempt_environment, 'Endpoint environment differs from selected attempt'
                                goal2 = Path(command[1]).name == GOAL2_ENTRY.name
                                if goal2:
                                    config = json.loads((destination / 'goal2-config.json').read_text())
                                    expected_flags = winner['environment'] if promote else old_goal2_flags
                                    assert config['environment'] == expected_flags, 'Constructor selection differs from promotion'
                                    result['constructor_configuration'] = config
                                wait_endpoint_idle(health)
                                request = urllib.request.Request('http://127.0.0.1:18170/v1/chat/completions',
                                    data=json.dumps(dict(model='DeepSeek-V4.1-Flash', messages=[dict(role='user', content='Hi.')],
                                                         temperature=0, max_tokens=16)).encode(), headers={'Content-Type': 'application/json'})
                                print(json.dumps(dict(validating_restored_endpoint=restored.pid)), flush=True)
                                with urllib.request.urlopen(request, timeout=600) as response:
                                    reply = json.load(response)
                                assert reply['choices'][0]['message']['content'] == 'Hello! How can I help you today?'
                                assert reply['choices'][0]['finish_reason'] == 'stop' and reply['usage']['completion_tokens'] == 10
                                assert reply['timings']['downloaded_bytes'] == 0
                                result['first_validation'] = reply
                                if goal2:
                                    first_metrics = json.loads((destination / 'goal2-last-request.json').read_text())
                                    result['first_goal2_metrics'] = first_metrics
                                wait_peer_idle(guard); wait_endpoint_idle(health)
                                with urllib.request.urlopen(request, timeout=120) as response:
                                    reply = json.load(response)
                                assert reply['choices'][0]['message']['content'] == 'Hello! How can I help you today?'
                                assert reply['choices'][0]['finish_reason'] == 'stop' and reply['usage']['completion_tokens'] == 10
                                assert reply['timings']['downloaded_bytes'] == 0
                                if goal2:
                                    warm_metrics = json.loads((destination / 'goal2-last-request.json').read_text())
                                    assert warm_metrics['completed'] == first_metrics['completed'] + 1
                                    assert warm_metrics['pack_count_delta'] == 0, 'Second HTTP validation still packed weights'
                                    assert warm_metrics['environment'] == expected_flags
                                    assert warm_metrics['flags_verified'] and warm_metrics['downloaded_bytes'] == 0
                                    assert (warm_metrics['sparse_native_calls_delta'] > 0) == (expected_flags[FLAG_SPARSE] == '1')
                                    if expected_flags[FLAG_EXACT16] == '1':
                                        assert warm_metrics['exact16_metrics_delta']['grouped_native_calls'] > 0
                                        assert warm_metrics['exact16_metrics_delta']['cap_fallbacks'] == 0
                                    result['warm_goal2_metrics'] = warm_metrics
                                live_tps = 9 / reply['timings']['decode_seconds']
                                if promote:
                                    assert live_tps > baseline_tps * 1.05, 'Live warm endpoint did not retain the 5% matched gain'
                                wait_peer_idle(guard); wait_endpoint_idle(health)
                                assert process_info(peer['pid'])['start'] == peer['info']['start']
                                assert all(sha256(p) == h for p,h in selected['source_sha256'].items())
                                if promote:
                                    assert all(sha256(p) == h for p,h in candidate_hashes.items())
                                    assert all(sha256(p) == h for p,h in result['source_sha256'].items())
                                result.update(restored=True, restored_server=info, validation=reply,
                                              peer_preserved=True, memory_after=memory_status())
                                replacement = json.loads(json.dumps(selected))
                                replacement.update(pid=restored.pid, start=info['start'], evidence=str(out / 'result.json'),
                                                request_record=str(destination / 'last-request.json'),
                                                restoration_sources=result['source_sha256'])
                                if promote:
                                    replacement['source_sha256'].update(candidate_hashes)
                                    replacement['retain_cached_expert_mappings'] = True
                                    replacement['promotion_evidence'] = str(evidence)
                                    replacement['cpu_optimizations'] = ['native_quant','quant_reuse','grouped_moe','native_hc','scripted_sparse_attention']
                                    if winner['environment'][FLAG_EXACT16] == '1':
                                        replacement['cpu_optimizations'].append('exact16_grouped_moe')
                                    if winner['environment'][FLAG_SPARSE] == '1':
                                        replacement['cpu_optimizations'].append('native_sparse_attention')
                                    replacement['goal2_environment'] = dict(winner['environment'])
                                    replacement['goal2_candidate'] = winner['name']
                                    replacement['recommended_trial_controller'] = str(BASE/'trial_deepseek_v41_goal2_queue_0910.py')
                                    replacement['measured_decode_tok_s'] = live_tps
                                atomic_json(SELECTED, replacement)
                                keep = True
                                result['optimized_promoted'] = promote
                                break
                            except BaseException as error:
                                stop_child(restored)
                                if not choose_candidate:
                                    raise
                                result['promotion_fallback_reason'] = str(error) if isinstance(error, AssertionError) else type(error).__name__

                    else:
                        result['original_server_preserved'] = process_info(selected['pid'])['start'] == old['start']
                except BaseException as error:
                    result['restoration_error'] = str(error) if isinstance(error, AssertionError) else type(error).__name__
                    raise
                finally:
                    if not keep:
                        stop_child(restored)
                    result['passed'] = bool(result.get('trial_passed') and result['restored'] and result.get('optimized_promoted'))
                    result['finished'] = time.time()
                    atomic_json(out / 'result.json', result)
                    print(json.dumps({k: result[k] for k in ['passed', 'restored', 'restored_pid', 'error', 'restoration_error'] if k in result}), flush=True)
        if not result['passed']:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
