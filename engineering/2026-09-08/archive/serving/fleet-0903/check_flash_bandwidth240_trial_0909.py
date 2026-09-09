#!/usr/bin/env python3
"""Fault-inject model switching without starting, stopping, or querying a model."""
import io
import json
import os
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

import flash_bandwidth240_trial_0909 as trial


def check(mode):
    events = []
    live = dict(peer=True, flash=False, restored=False, foreign=False, measurement=False)
    original = dict(start='birth', exe='/fixture/llama-server', command=['/fixture/llama-server'],
                    cwd='/fixture', affinity=[0, 1])
    environment = {'GGML_EXAMPLE': '1', 'PRIVATE_RESTORE_TEST': 'retained only in the private context'}
    config = dict(command=['/fixture/flash'], runtime_env={}, drafts=0, dissemination=False)
    plan = dict(qwen_pid=10, qwen=original, qwen_runtime_env={'GGML_EXAMPLE':'1'},
                qwen_libraries={'/fixture/libllama.so.0': 'unused'},
                configurations={'raw_control': config}, sequence=['raw_control'])

    class Guard:
        def __init__(self, *args, **kwargs):
            pass
        def wait_idle(self, *args, **kwargs):
            events.append('idle_gate')
            if mode == 'busy_before_pause':
                raise RuntimeError('busy peer')
            return {'idle':True}
        def assert_idle(self):
            assert not live['foreign']
            return {'idle':True}

    class Manager:
        def __init__(self):
            self.state = {'current':None}
            self.qwen = type('Qwen', (), {'state':{'current':None}})()
            self.cancelled = self.recovering = False
        def check_cancel(self):
            if mode == 'cancel_during_measurement' and live['measurement'] and not self.recovering:
                raise InterruptedError('cancel injected during measurement')
        def quiet(self):
            events.append('flash_idle_gate')
        def candidate(self, **kwargs):
            return dict(config), {}
        def start_flash(self, candidate, env):
            assert not live['peer'] and not live['flash']
            events.append('flash_start')
            if mode in ['flash_load_failure', 'foreign_during_flash_failure']:
                live['foreign'] = mode == 'foreign_during_flash_failure'
                raise RuntimeError('injected Flash loading failure')
            live['flash'] = True
            self.state['current'] = dict(candidate, pid=20)
        def stop_flash(self):
            assert live['flash'] and self.state['current']['pid'] == 20
            events.append('flash_stop')
            live['flash'] = False
            self.state['current'] = None
        def validate_current(self):
            return self.state['current']

    class Original:
        def __init__(self, pid, expected, forbidden):
            assert pid == 10
            self.termination_sent = False
        def terminate(self):
            events.append('peer_sigterm')
            if mode == 'peer_signal_failure':
                raise RuntimeError('injected pidfd signal failure')
            self.termination_sent = True
            live['peer'] = False
        def exited(self):
            return not live['peer']
        def close(self):
            events.append('pidfd_close')

    class Measurement:
        def __init__(self):
            self.returncode = None if mode == 'cancel_during_measurement' else 1
        def poll(self):
            return self.returncode
        def terminate(self):
            events.append('measurement_stop')
            self.returncode = -15
            live['measurement'] = False
        def wait(self, **kwargs):
            return self.returncode
        def kill(self):
            raise AssertionError('The cancellable fixture should not require SIGKILL')

    def popen(command, **kwargs):
        if command[0] == 'taskset':
            assert not any(live[key] for key in ['peer', 'flash', 'foreign'])
            assert kwargs['env'] == environment and kwargs['cwd'] == original['cwd']
            assert command[3:] == original['command']
            events.append('qwen_restore')
            live['restored'] = True
            return type('Restored', (), {'pid':30, 'poll':lambda self:None})()
        assert live['flash'] and not live['peer']
        events.append('measurement_start')
        live['measurement'] = True
        return Measurement()

    def snapshot():
        return {str(pid):(0,'birth','llama-server') for key,pid in
                [('peer',10),('flash',20),('restored',30),('foreign',40)] if live[key]}

    read_text = Path.read_text
    def read(path, *args, **kwargs):
        if str(path) == '/proc/30/maps':
            return 'r-xp /fixture/libllama.so.0\n'
        return read_text(path, *args, **kwargs)

    with tempfile.TemporaryDirectory(prefix='flash240-lifecycle-') as directory:
        out = Path(directory)
        (out/'plan.json').write_text(json.dumps(plan))
        with patch.multiple(trial, Manager=Manager, ExactProcess=Original,
                ModelMeasurementGuard=Guard, wait_background=lambda *a,**k:{'idle':True},
                assert_sources=lambda p:None, assert_peer=lambda p:original,
                process_environment=lambda pid:environment, process_info=lambda pid:dict(original),
                inference_snapshot=snapshot, port_available=lambda port:True,
                memory_status=lambda:{'MemAvailable':500000000000},
                read_service=lambda port:{'processing':0,'queued':0}), \
             patch.object(trial.signal, 'signal'), \
             patch.object(trial.subprocess, 'Popen', side_effect=popen), \
             patch.object(trial.os, 'waitpid', side_effect=lambda pid,flags:events.append('flash_reap')), \
             patch.object(trial.urllib.request, 'urlopen', side_effect=lambda *a,**k:io.BytesIO(b'{"status":"ok"}')), \
             patch.object(Path, 'read_text', read):
            try:
                trial.execute(out, mode != 'approval_absent', 99)
            except (AssertionError, RuntimeError, InterruptedError):
                pass
            else:
                raise AssertionError('The injected failure must not be reported successful')
        result = json.loads((out/'result.json').read_text()) if (out/'result.json').exists() else None
        assert result is None or not result['passed']
        if mode == 'approval_absent':
            assert not events and result is None and live['peer']
        elif mode == 'busy_before_pause':
            assert 'peer_sigterm' not in events and live['peer']
        elif mode == 'peer_signal_failure':
            assert live['peer'] and 'qwen_restore' not in events and 'flash_start' not in events
        elif mode == 'foreign_during_flash_failure':
            assert live['foreign'] and 'qwen_restore' not in events and result.get('restoration_error')
        else:
            assert live['restored'] and not live['flash'] and not live['peer']
            assert result['qwen_restored']
            if mode in ['measurement_failure', 'cancel_during_measurement']:
                assert events.index('flash_stop') < events.index('flash_reap') < events.index('qwen_restore')
            if mode == 'cancel_during_measurement':
                assert events.index('measurement_stop') < events.index('flash_stop')
        return dict(case=mode, passed=True, events=events,
                    original_preserved=live['peer'], exact_restore=live['restored'],
                    foreign_preserved=live['foreign'], owned_flash_absent=not live['flash'])


if __name__ == '__main__':
    os.umask(0o077)
    cases = [check(mode) for mode in ['approval_absent', 'busy_before_pause', 'peer_signal_failure',
             'flash_load_failure', 'measurement_failure', 'cancel_during_measurement', 'foreign_during_flash_failure']]
    result = dict(time=time.time(), passed=True, cases=cases,
                  sources={str(path):trial.sha256(path) for path in [Path(__file__).resolve(), Path(trial.__file__).resolve()]},
                  scope='Seven deterministic lifecycle fault injections. No real process, model, queue, or hardware counter was queried or changed.')
    out = trial.BASE / 'results/flash-bandwidth240-lifecycle-0909.json'
    with out.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps(result, indent=2))
