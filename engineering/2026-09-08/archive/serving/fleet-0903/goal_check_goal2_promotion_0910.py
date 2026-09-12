#!/usr/bin/env python3
"""Pure selection and mocked promotion/fallback tests; never signal or launch."""
import argparse
import ast
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import tempfile
import time
import types
import urllib.request

from goal2_promotion_selection_0910 import FLAGS, FLAG_EXACT16, FLAG_SPARSE, select_candidate
from goal2_lifecycle_safety_0910 import settle_termination, wait_peer_idle, wait_endpoint_idle


BASE = Path(__file__).resolve().parent


def evidence():
    rows = []
    for labels, exact16, sparse, rate in (
        (('baseline_before', 'baseline_after'), False, False, 2.0),
        (('exact16_1', 'exact16_2'), True, False, 2.6),
        (('sparse_1', 'sparse_2'), False, True, 2.3),
        (('exact16_sparse_1', 'exact16_sparse_2'), True, True, 3.0),
    ):
        for label in labels:
            rows.append(dict(label=label, measured=True, exact16=exact16, native_sparse=sparse,
                exact_golden=True, tokens_match=True, token_ids=list(range(10)),
                logits_sha256=['logits'+str(i) for i in range(10)],
                timings=dict(decode_seconds=9/rate, downloaded_bytes=0),
                exact16_metrics_delta=dict(pack_count=0, cap_fallbacks=0,
                                           grouped_native_calls=720 if exact16 else 0),
                sparse_calls_delta=dict(native=360 if sparse else 0)))
    return dict(passed=True, all_exact_golden=True, zero_measured_downloads=True,
        zero_measured_packing=True, selected_sources_preserved=True, torch_workers=16,
        native_workers=16, omp_wait_policy='PASSIVE', packed_cap_bytes=64 << 30,
        affinity=list(range(48, 64)),
        runs=rows, source_sha256={}, protected_selected_sources={})


def expect_rejection(value):
    try:
        select_candidate(value)
    except (AssertionError, KeyError):
        return
    raise AssertionError('Invalid evidence was accepted')


def mock_lifecycle(fail_candidate):
    """Execute the real main body with every process, signal, lock and HTTP API fake."""
    with tempfile.TemporaryDirectory(prefix='goal2-promotion-fixture-') as temporary:
        root = Path(temporary)
        selected_path = root / 'deepseek-v41-selected.json'
        fleet = root / 'results/qwen-q6-trial-0907/lifecycle.lock'
        fleet.parent.mkdir(parents=True)
        fleet.touch()
        quiet = root / 'results/deepseek-v41-goal-exact16-quiet-0910/trial/result.json'
        quiet.parent.mkdir(parents=True)
        quiet.write_text(json.dumps(evidence()))
        (quiet.parent.parent / 'result.json').write_text(json.dumps(
            dict(passed=True, restored=True, peer_preserved=True)))
        golden = root / 'results/deepseek-v41-checkpoint-run-0910b/generation.json'
        golden.parent.mkdir()
        golden.write_text(json.dumps(dict(runs=[dict(token_ids=list(range(10)),
            steps=[dict(logits_sha256='logits'+str(i)) for i in range(10)])])))
        entry = root / 'deepseek_v41_goal2_server_0910.py'
        old_environment = dict(OMP_NUM_THREADS='16', MKL_NUM_THREADS='16', OMP_WAIT_POLICY='PASSIVE',
            DEEPSEEK_GOAL2_EXACT16='1', DEEPSEEK_GOAL2_NATIVE_SPARSE='0',
            DEEPSEEK_GOAL2_PACKED_CAP_GIB='64', PRIVATE_FIXTURE_TOKEN='must-not-be-recorded')
        old_flags = {key: old_environment[key] for key in FLAGS}
        old = dict(start='11', exe='/fake/python', cwd=str(root), affinity=list(range(48, 64)),
            command=['/fake/python', str(entry), '--cache', '/owned-cache', '--output', '/old-output',
                     '--port', '18170', '--lifecycle-lock-fd', '9'])
        old_selected = dict(pid=101, start='11', source_sha256={str(entry): 'digest'},
                            temporary_ram_cache='/owned-cache', goal2_environment=old_flags,
                            goal2_candidate='exact16', cpu_optimizations=['existing'])
        selected_path.write_text(json.dumps(old_selected))
        state = dict(stopped=False, processes=[], current=None, signals=[], http=[]) 

        class Exact:
            def __init__(self, pid, expected, forbidden):
                assert pid == 101 and expected == dict(start_ticks='11', exe='/fake/python')
                assert forbidden == {102}
                self.termination_sent = False
            def terminate(self):
                self.termination_sent = True
                state['stopped'] = True
                state['signals'].append(101)
            def exited(self):
                return state['stopped']
            def close(self):
                pass

        class Process:
            def __init__(self, command, **kwargs):
                assert command[:2] == ['taskset', '-c'] and kwargs['pass_fds']
                self.pid = 200 + len(state['processes'])
                self.command = command[3:]
                self.environment = dict(kwargs['env'])
                self.destination = Path(self.command[self.command.index('--output') + 1])
                self.dead = False
                self.requests = 0
                self.info = dict(old, command=self.command, start=str(self.pid))
                state['processes'].append(self)
                state['current'] = self
                (self.destination / 'ready.json').write_text('{}')
                (self.destination / 'goal2-config.json').write_text(json.dumps(
                    dict(environment={key: self.environment[key] for key in FLAGS})))
            def poll(self):
                return 0 if self.dead else None
            def terminate(self):
                self.dead = True
            def kill(self):
                self.dead = True
            def wait(self, timeout=None):
                self.dead = True
                return 0

        def info(pid):
            if pid == 101:
                return old
            if pid == 102:
                return dict(start='12')
            return next(p.info for p in state['processes'] if p.pid == pid)

        def environment(pid):
            if pid == 101:
                return dict(old_environment)
            return dict(next(p.environment for p in state['processes'] if p.pid == pid))

        def flock(handle, operation):
            if handle.name == str(fleet) and not state['stopped']:
                raise BlockingIOError('Selected fake process owns fleet lock')

        def urlopen(request, timeout=None):
            assert request.full_url == 'http://127.0.0.1:18170/v1/chat/completions'
            p = state['current']
            p.requests += 1
            state['http'].append(p.pid)
            flags = {key: p.environment[key] for key in FLAGS}
            metrics = dict(completed=p.requests, environment=flags, pack_count_delta=1 if p.requests==1 else 0,
                           flags_verified=True, downloaded_bytes=0,
                           sparse_native_calls_delta=360 if flags[FLAG_SPARSE]=='1' else 0,
                           exact16_metrics_delta=dict(grouped_native_calls=720, cap_fallbacks=0))
            (p.destination / 'goal2-last-request.json').write_text(json.dumps(metrics))
            # Low warm speed fails promotion after both real-path HTTP validations.
            speed = 1.0 if fail_candidate and p is state['processes'][0] else 4.0
            reply = dict(choices=[dict(message=dict(content='Hello! How can I help you today?'), finish_reason='stop')],
                         usage=dict(completion_tokens=10), timings=dict(downloaded_bytes=0, decode_seconds=9/speed))
            return io.StringIO(json.dumps(reply))

        def atomic_json(path, value):
            Path(path).write_text(json.dumps(value))

        arguments = ['--evidence', str(quiet), '--label', 'deepseek-v41-goal2-fixture']
        class Parser(argparse.ArgumentParser):
            def parse_args(self):
                return super().parse_args(arguments)

        source = (BASE / 'promote_deepseek_v41_goal2_0910.py').read_text()
        tree = ast.parse(source)
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ('main', 'stop_child')]
        namespace = dict(__file__=str(BASE/'promote_deepseek_v41_goal2_0910.py'),
            BASE=root, SELECTED=selected_path, FLEET_LOCK=fleet, QUIET_EVIDENCE=quiet, GOAL2_ENTRY=entry,
            argparse=types.SimpleNamespace(ArgumentParser=Parser), json=json, Path=Path,
            re=__import__('re'), time=time, select_candidate=select_candidate,
            settle_termination=settle_termination, wait_peer_idle=wait_peer_idle,
            wait_endpoint_idle=wait_endpoint_idle,
            FLAGS=FLAGS, FLAG_EXACT16=FLAG_EXACT16, FLAG_SPARSE=FLAG_SPARSE,
            os=types.SimpleNamespace(sched_getaffinity=lambda pid:{32},
                stat=lambda path:types.SimpleNamespace(st_ino=fleet.stat().st_ino)),
            fcntl=types.SimpleNamespace(flock=flock, LOCK_EX=2, LOCK_NB=4),
            signal=types.SimpleNamespace(SIGINT=signal.SIGINT, SIGTERM=signal.SIGTERM,
                                         SIGHUP=signal.SIGHUP, signal=lambda *args:None),
            subprocess=types.SimpleNamespace(Popen=Process, DEVNULL=-3, STDOUT=-2,
                                              TimeoutExpired=RuntimeError),
            urllib=types.SimpleNamespace(request=types.SimpleNamespace(Request=urllib.request.Request, urlopen=urlopen)),
            process_info=info, process_environment=environment, sha256=lambda path:'digest',
            ExactProcess=Exact, health=lambda:dict(status='ok', model='DeepSeek-V4.1-Flash', busy=False),
            port_available=lambda port:True, inference_snapshot=lambda:{},
            memory_status=lambda:dict(MemAvailable=200 << 30), atomic_json=atomic_json,
            Manager=lambda:types.SimpleNamespace(validate_current=lambda:dict(pid=102,info=dict(start='12'))),
            ModelMeasurementGuard=lambda *args:types.SimpleNamespace(assert_idle=lambda:None,
                inspect=lambda:(dict(busy=False), {})))
        exec(compile(ast.Module(body=functions, type_ignores=[]), '<mocked promotion>', 'exec'), namespace)
        try:
            namespace['main']()
        except SystemExit as error:
            assert fail_candidate and error.code == 1
        selected = json.loads(selected_path.read_text())
        result = json.loads((root/'results/deepseek-v41-goal2-fixture/result.json').read_text())
        assert state['signals'] == [101]
        assert all(p.requests == 2 for p in state['processes'])
        assert result['restored'] and result['peer_preserved']
        if fail_candidate:
            assert len(state['processes']) == 2 and state['processes'][0].dead
            assert not state['processes'][1].dead
            assert state['processes'][1].environment == old_environment
            assert selected['goal2_environment'] == old_flags
            assert selected['source_sha256'] == old_selected['source_sha256']
            assert selected['cpu_optimizations'] == ['existing']
            assert not result['passed'] and not result['optimized_promoted']
        else:
            assert len(state['processes']) == 1 and not state['processes'][0].dead
            assert selected['goal2_environment'][FLAG_SPARSE] == '1'
            assert selected['goal2_candidate'] == 'exact16_sparse'
            assert result['passed'] and result['optimized_promoted']
        assert all('must-not-be-recorded' not in path.read_text() for path in root.rglob('*.json'))
        return dict(candidate_failure=fail_candidate, launches_simulated=len(state['processes']),
                    http_requests_simulated=len(state['http']), old_environment_restored=fail_candidate,
                    full_environment_not_serialized=True, passed=True)


def main():
    assert os.sched_getaffinity(0) == {32}, 'Run fixture under taskset -c 32'
    paths = [Path(__file__), BASE/'goal2_promotion_selection_0910.py', BASE/'goal2_lifecycle_safety_0910.py',
             BASE/'promote_deepseek_v41_goal2_0910.py', BASE/'trial_deepseek_v41_goal2_queue_0910.py',
             BASE/'deepseek_v41_goal2_server_0910.py']
    hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    source = evidence()
    assert select_candidate(source)['winner']['name'] == 'exact16_sparse'
    for winner, labels in [('exact16', ('exact16_1','exact16_2')), ('sparse', ('sparse_1','sparse_2'))]:
        value = copy.deepcopy(source)
        for row in value['runs']:
            if row['label'] in labels:
                row['timings']['decode_seconds'] = 1
        assert select_candidate(value)['winner']['name'] == winner
    rejected = []
    for name, edit in [
        ('partial', lambda x:x.update(passed=False)),
        ('golden', lambda x:x['runs'][0].update(exact_golden=False)),
        ('cold_download', lambda x:x['runs'][2]['timings'].update(downloaded_bytes=1)),
        ('cold_pack', lambda x:x['runs'][2]['exact16_metrics_delta'].update(pack_count=1)),
        ('cap_fallback', lambda x:x['runs'][2]['exact16_metrics_delta'].update(cap_fallbacks=1)),
        ('wrong_threads', lambda x:x.update(torch_workers=8)),
        ('sparse_never_ran', lambda x:x['runs'][4]['sparse_calls_delta'].update(native=0)),
        ('duplicate_label', lambda x:x['runs'].append(copy.deepcopy(x['runs'][0]))),
        ('no_gain', lambda x:[r['timings'].update(decode_seconds=9/2.05) for r in x['runs'] if r['exact16'] or r['native_sparse']]),
    ]:
        value = copy.deepcopy(source)
        edit(value)
        expect_rejection(value)
        rejected.append(name)
    # Busy recovery and asynchronous SIGTERM completion must wait, not declare
    # the old process preserved or abandon restoration.
    exits = iter((False, True))
    assert settle_termination(types.SimpleNamespace(termination_sent=True, exited=lambda:next(exits)), False)
    assert not settle_termination(types.SimpleNamespace(termination_sent=False), False)
    peer_states = iter((True, False))
    wait_peer_idle(types.SimpleNamespace(inspect=lambda:(dict(busy=next(peer_states)), {})))
    endpoint_states = iter((True, False))
    wait_endpoint_idle(lambda:dict(busy=next(endpoint_states)))
    def identity_error():
        raise RuntimeError('Model process identity changed')
    try:
        wait_peer_idle(types.SimpleNamespace(inspect=identity_error))
    except RuntimeError as error:
        assert 'identity changed' in str(error)
    else:
        raise AssertionError('Peer identity failure was ignored')
    result = dict(passed=True, selection_winners=3, rejected=rejected,
                          lifecycle=[mock_lifecycle(False), mock_lifecycle(True)],
                          asynchronous_termination_waited=True, busy_peer_recovery_waited=True,
                          endpoint_reply_lock_race_waited=True, peer_identity_failure_fatal=True,
                          real_processes_started=False, actual_http_requests=False,
                          actual_signals_sent=False, selected_manifest_changed=False,
                          source_sha256=hashes)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    output = BASE/'results/goal2-promotion-0910'
    output.mkdir(exist_ok=True)
    (output/'fixture-check.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
