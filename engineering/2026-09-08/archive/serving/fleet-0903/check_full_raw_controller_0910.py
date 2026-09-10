#!/usr/bin/env python3
"""Fault-test Full's actual pause/load/repeat/restore controller with synthetic processes."""
import contextlib
import copy
import io
import json
from pathlib import Path
import signal
import tempfile
from unittest.mock import patch

import run_full_raw_baseline_0910 as trial
from qwen_split_trial import sha256


def check(case):
    original = dict(pid=1001, info=dict(start='111'), command=['fake-flash', '--port', '18131'],
        runtime_env={}, affinity=list(range(128)), log='fake-original.log', quant='UD-Q4_K_XL', drafts=2)
    environment = {'LIFECYCLE_FIXTURE': 'true'}
    live = {1001: dict(env=environment, command=original['command'], cwd='/fake')}
    events, callbacks, clock, counts = [], {}, [0.0], {'full': 0, 'measure': 0}
    libraries = {stem: '/fixture/' + stem + '0.20.2' for stem in ['libggml-cpu.so.', 'libggml-base.so.', 'libllama.so.']}
    libraries['libllama-common.so.'] = '/fixture/libllama-common.so.0.1.2'
    class FakeManager:
        def __init__(self): self.state = {'current': copy.deepcopy(original), 'events': []}
        def validate_current(self):
            current = self.state['current']; assert current and current['pid'] in live
            return current
        def stop_flash(self):
            events.append('stop-flash'); live.pop(1001); self.state['current'] = None
            if case == 'cancel-after-stop': callbacks[signal.SIGINT](signal.SIGINT, None)
        def record(self, *args, **kwargs): events.append('record-restore')
    manager = FakeManager()
    class Guard:
        def __init__(self, *args): pass
        def assert_idle(self): pass
    class Proc:
        next_pid = 2000
        def __init__(self, args, env=None, cwd=None, **kwargs):
            self.pid = Proc.next_pid; Proc.next_pid += 1
            if args[:3] != ['taskset', '-c', '0-127']:
                assert args[2].endswith('measure_full_raw_0910.py')
                counts['measure'] += 1; events.append('measure')
                fail = (case == 'first-measurement-exits' or (case == 'second-measurement-exits' and counts['measure'] == 2))
                self.returncode = 7 if fail else 0
                return
            executable = args[3]
            is_full = executable == 'fake-full'
            if is_full: counts['full'] += 1
            self.returncode = 7 if is_full and case == 'full-load-exits' else None
            events.append('spawn-' + executable)
            if self.returncode is None: live[self.pid] = dict(env=env, command=args[3:], cwd=str(cwd))
        def poll(self): return self.returncode
        def terminate(self): self.returncode = 0; live.pop(self.pid, None)
        def kill(self): self.terminate()
        def wait(self, timeout=None): return self.returncode
    def info(pid):
        row = live[pid]
        return dict(pid=pid, start='111' if pid == 1001 else '222', command=row['command'],
                    cwd=row['cwd'], affinity=list(range(128)))
    def memory(global_bytes, node_bytes):
        if global_bytes == 502_698_003_456 and case in ['memory-after-stop', 'restore-reserve-fails']:
            raise AssertionError('fixture Full admission reserve')
        if global_bytes == 260_000_000_000 and case == 'restore-reserve-fails':
            raise AssertionError('fixture Flash restoration reserve')
        return {}
    def maps(pid):
        if live[pid]['command'][0] != 'fake-full': return set()
        return set() if case == 'library-mismatch' else {p for k, p in libraries.items() if k != 'libllama-common.so.'}
    def measurement(path, plan):
        if case == 'cancel-after-first-measurement' and counts['measure'] == 1:
            callbacks[signal.SIGINT](signal.SIGINT, None)
        changed = case == 'outputs-change' and counts['measure'] == 2
        return {kind: dict(tok_s=7.0, adjusted_gb_s=245.0,
            counters=dict(adjacent_idle_qualifies=case != 'unqualified-background'),
            output_sha256=kind + ('changed' if changed else '-stable'), generated_tokens=512,
            draft_tokens=0, accepted_draft_tokens=0, cache_tokens=0, completed_answer=True)
            for kind in ['prose', 'code']}
    def sleep(seconds): clock[0] += seconds
    with tempfile.TemporaryDirectory(prefix='full-raw-lifecycle-') as directory:
        out = Path(directory)
        peer = copy.deepcopy(original)
        if case == 'changed-identity': peer['info']['start'] = 'wrong'
        plan = dict(peer=peer, peer_libraries=[], command=['fake-full'], runtime_env={}, libraries=libraries,
            source_sha256={str(trial.SELECTED): 'fixture-sha'}, repetitions=2,
            global_admission_bytes=502_698_003_456, node_admission_bytes=126_467_450_880)
        (out / 'plan.json').write_text(json.dumps(plan))
        changes = dict(OUT=out, Manager=lambda: manager, verify_sources=lambda _: None,
            inference_snapshot=lambda: {str(pid): {} for pid in live}, port_available=lambda _: True,
            process_environment=lambda pid: live[pid]['env'], process_info=info,
            mapped_libraries=maps, common_libraries=lambda pid: {libraries['libllama-common.so.']},
            sha256=lambda _: 'fixture-sha', memory_gate=memory, ModelMeasurementGuard=Guard,
            background=lambda _: {}, read_service=lambda _: {'status': 'ok'}, read_measurement=measurement)
        error = None
        with patch.multiple(trial, **changes), patch.object(trial.subprocess, 'Popen', Proc), \
             patch.object(trial.signal, 'signal', lambda sig, fn: callbacks.update({sig: fn})), \
             patch.object(trial.time, 'sleep', sleep), patch.object(trial.time, 'monotonic', lambda: clock[0]), \
             patch.object(trial.urllib.request, 'urlopen', lambda *args, **kwargs: io.BytesIO(b'{"status":"ok"}')), \
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
                assert counts == {'full': 1, 'measure': 2} and len(result['runs']) == 2
                assert result['all_outputs_match'] and not result['target_reached']
                assert result['all_attribution_valid'] == (case == 'success')
        return dict(case=case, passed=True, restored=result['restored'] if result else False,
                    launches=counts, events=events)


def main():
    destination = trial.BASE / 'results/full-raw-controller-checks-0910.json'
    assert not destination.exists()
    cases = ['changed-identity', 'cancel-after-stop', 'memory-after-stop', 'full-load-exits',
        'library-mismatch', 'first-measurement-exits', 'second-measurement-exits',
        'cancel-after-first-measurement', 'outputs-change', 'restore-reserve-fails', 'success', 'unqualified-background']
    report = dict(passed=True, checks=[check(case) for case in cases],
        sources={str(p): sha256(p) for p in [Path(__file__), Path(trial.__file__)]},
        scope='Synthetic processes and state exercise the actual controller. No real server is loaded, stopped or signaled.')
    destination.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(passed=True, lifecycle_cases=len(cases))))


if __name__ == '__main__':
    main()
