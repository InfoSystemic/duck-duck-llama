#!/usr/bin/env python3
"""Exercise the actual four-arm lifecycle using fake processes and synthetic state."""
import contextlib
import copy
import io
import json
from pathlib import Path
import signal
import tempfile
from unittest.mock import patch

import run_qwen_r8_projection_0910 as trial
from qwen_split_trial import sha256


def check(case):
    original = dict(pid=1001, info=dict(start='111'), command=['fake-flash', '--port', '18131'],
        runtime_env={}, affinity=list(range(128)), log='fake-original.log', quant='UD-Q4_K_XL', drafts=2)
    environment = {'PROFILE_FIXTURE_ONLY': 'true'}
    live = {1001: dict(env=environment, command=original['command'], cwd='/fake')}
    events, callbacks, clock, counts = [], {}, [0.0], {'qwen': 0, 'measure': 0}
    libraries = ['/fixture/libggml-cpu.so.0.22.0', '/fixture/libggml-base.so.0.22.0', '/fixture/libllama.so.0.3.0']
    class FakeManager:
        def __init__(self): self.state = {'current': copy.deepcopy(original), 'events': []}
        def validate_current(self):
            current = self.state['current']
            assert current and current['pid'] in live
            return current
        def stop_flash(self):
            events.append('stop-flash')
            live.pop(1001)
            self.state['current'] = None
            if case == 'cancel-after-stop': callbacks[signal.SIGINT](signal.SIGINT, None)
        def record(self, *args, **kwargs): events.append('record-restore')
    manager = FakeManager()
    class Guard:
        def __init__(self, *args): pass
        def assert_idle(self): pass
    class Proc:
        next_pid = 2000
        def __init__(self, args, env=None, cwd=None, **kwargs):
            self.pid = Proc.next_pid
            Proc.next_pid += 1
            self.is_qwen = False
            if args[:3] != ['taskset', '-c', '0-127']:
                assert args[2].endswith('measure-model-bandwidth.py')
                counts['measure'] += 1
                events.append('measure')
                self.returncode = 7 if case == 'second-measurement-exits' and counts['measure'] == 2 else 0
                return
            executable = args[3]
            self.is_qwen = executable == 'fake-qwen'
            if self.is_qwen: counts['qwen'] += 1
            fail = self.is_qwen and (case == 'qwen-load-exits' or
                (case == 'second-load-exits' and counts['qwen'] == 2))
            self.returncode = 7 if fail else None
            events.append('spawn-' + executable)
            if not fail: live[self.pid] = dict(env=env, command=args[3:], cwd=str(cwd))
        def poll(self): return self.returncode
        def terminate(self):
            self.returncode = 0
            live.pop(self.pid, None)
            if self.is_qwen and case == 'cancel-after-first-arm' and counts['qwen'] == 1:
                callbacks[signal.SIGINT](signal.SIGINT, None)
        def kill(self): self.terminate()
        def wait(self, timeout=None): return self.returncode
    def info(pid):
        row = live[pid]
        return dict(pid=pid, start='111' if pid == 1001 else '222', command=row['command'],
                    cwd=row['cwd'], affinity=list(range(128)))
    def memory(global_bytes, node_bytes):
        if global_bytes == 220_000_000_000 and case in ['memory-after-stop', 'restore-reserve-fails']:
            raise AssertionError('fixture Qwen admission reserve')
        if global_bytes == 260_000_000_000 and case == 'restore-reserve-fails':
            raise AssertionError('fixture Flash restoration reserve')
        return {}
    def dispatch(_):
        active = any(row['command'][0] == 'fake-qwen' for row in live.values())
        destroyed = [] if active or case == 'dispatch-cleanup-missing' else ['fixture-group']
        return dict(created=[('fixture-group', '4')], attached=[('fixture-group', '4')] * 2, destroyed=destroyed)
    def measurement(path, plan, env):
        enabled = env['GGML_CPU_Q8_R8_K160_PREP'] == '1'
        assert env['GGML_CPU_Q8_R8_SSM_TILE8'] == str(int(enabled))
        changed = case == 'outputs-change' and counts['measure'] == 2
        return {kind: dict(tok_s=25.0 if enabled else 22.0, adjusted_gb_s=140.0 if enabled else 135.0,
            counters=dict(adjacent_idle_qualifies=case != 'unqualified-background'),
            output_sha256=kind + ('changed' if changed else '-stable'),
            generated_tokens=512 if kind == 'prose' else 318, draft_tokens=586 if kind == 'prose' else 290,
            accepted_draft_tokens=359 if kind == 'prose' else 244, cache_tokens=0, completed_answer=True)
            for kind in ['prose', 'code']}
    def sleep(seconds): clock[0] += seconds
    with tempfile.TemporaryDirectory(prefix='qwen-r8-lifecycle-') as directory:
        out = Path(directory)
        peer = copy.deepcopy(original)
        if case == 'changed-identity': peer['info']['start'] = 'wrong'
        plan = dict(peer=peer, peer_libraries=[], arm_file=str(out / 'arm'), command=['fake-qwen'],
            runtime_env={}, source_sha256={str(trial.SELECTED): 'fixture-sha'},
            library=libraries[0], base=libraries[1], llama=libraries[2], common='/fixture/libllama-common.so.0.3.0',
            schedule=['off', 'both', 'both', 'off'])
        (out / 'plan.json').write_text(json.dumps(plan))
        changes = dict(OUT=out, Manager=lambda: manager, verify_sources=lambda _: None,
            inference_snapshot=lambda: {str(pid): {} for pid in live}, port_available=lambda _: True,
            process_environment=lambda pid: live[pid]['env'], process_info=info,
            mapped_libraries=lambda pid: set(libraries) if live[pid]['command'][0] == 'fake-qwen' else set(),
            loaded_common_libraries=lambda pid: {plan['common']}, sha256=lambda _: 'fixture-sha',
            memory_gate=memory, ModelMeasurementGuard=Guard, background=lambda _: {},
            read_service=lambda _: {'status': 'ok'}, dispatch_log=dispatch, read_arm_measurement=measurement)
        error = None
        with patch.multiple(trial, **changes), patch.object(trial.subprocess, 'Popen', Proc), \
             patch.object(trial.signal, 'signal', lambda sig, fn: callbacks.update({sig: fn})), \
             patch.object(trial.time, 'sleep', sleep), patch.object(trial.time, 'monotonic', lambda: clock[0]), \
             patch.object(trial.urllib.request, 'urlopen', lambda *a, **kw: io.BytesIO(b'{"status":"ok"}')), \
             contextlib.redirect_stdout(io.StringIO()):
            try: trial.execute(-1)
            except (AssertionError, InterruptedError) as caught: error = type(caught).__name__
        path = out / 'result.json'
        result = json.loads(path.read_text()) if path.exists() else None
        succeeds = case in ['success', 'unqualified-background']
        assert (error is None) == succeeds, (case, error)
        if case == 'changed-identity':
            assert not events and set(live) == {1001} and result is None
        elif case == 'restore-reserve-fails':
            assert not live and result['restore_error'] and result['finished'] and not result['passed']
        else:
            assert result['restored'] and result['finished'] and result['passed'] == succeeds
            assert len(live) == 1 and 1001 not in live
            assert next(iter(live.values())) == dict(env=environment, command=original['command'], cwd='/fake')
            assert events.count('stop-flash') == events.count('spawn-fake-flash') == 1
            if succeeds:
                assert counts == {'qwen': 4, 'measure': 4} and len(result['trials']) == 4
                assert all(row['outputs_and_counts_match'] and row['dispatch_cleanup_valid'] for row in result['trials'])
                assert all(row['off_mean_tok_s'] == 22 and row['on_mean_tok_s'] == 25 for row in result['summaries'])
                assert not result['target_reached'] and not result['tok_s_target_reached']
                assert all(row['all_attribution_valid'] == (case == 'success') for row in result['summaries'])
        return dict(case=case, passed=True, restored=bool(result and result.get('restored')),
                    launches=counts, events=events)


if __name__ == '__main__':
    destination = trial.BASE / 'results/qwen-r8-projection-controller-checks-0910.json'
    assert not destination.exists()
    cases = ['changed-identity', 'cancel-after-stop', 'memory-after-stop', 'qwen-load-exits',
        'restore-reserve-fails', 'second-load-exits', 'second-measurement-exits',
        'cancel-after-first-arm', 'outputs-change', 'dispatch-cleanup-missing', 'success', 'unqualified-background']
    rows = [check(case) for case in cases]
    result = dict(passed=True, checks=rows, sources={str(p): sha256(p) for p in [Path(__file__), Path(trial.__file__)]},
        scope='Fake processes and synthetic state exercise actual multi-arm success, failure, cancellation, and exact restoration paths. No real inference or service mutation.')
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(passed=True, cases=len(rows), checks=[row['case'] for row in rows])))
