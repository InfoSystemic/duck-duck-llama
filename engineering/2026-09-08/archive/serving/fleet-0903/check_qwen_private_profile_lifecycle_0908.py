#!/usr/bin/env python3
"""Exercise private profile cleanup and post-load reserves without real model work."""
import io
import json
from pathlib import Path
import signal
import tempfile
import time
from unittest.mock import patch

import qwen_private_profile_trial_0908 as trial
from qwen_split_trial import sha256


class OwnedProcess:
    def __init__(self, pid, code=None):
        self.pid, self.returncode = pid, code
        self.signals = []

    def poll(self):
        return self.returncode

    def terminate(self):
        self.signals.append(signal.SIGTERM)
        self.returncode = -signal.SIGTERM

    def send_signal(self, value):
        self.signals.append(value)
        self.returncode = -value

    def wait(self, timeout):
        assert self.returncode is not None
        return self.returncode

    def kill(self):
        raise AssertionError('A responsive owned child should not be killed')


def main():
    rows = []
    for scenario in ('memory-while-waiting-after-load', 'peer-active-during-profile', 'profile-child-fails'):
        peer_pid = 812345
        peer = dict(start='123', exe='/test/peer-server', command=['/test/peer-server'], affinity=[0, 1])
        info = dict(start='124', exe='/test/private-server', command=['/test/private-server'], affinity=list(range(128)))
        model, measure = OwnedProcess(912345), OwnedProcess(912346, 0)
        profiler = OwnedProcess(912347, 1 if scenario == 'profile-child-fails' else None)
        created, peer_checks, postload = [], [], []

        class FakeGuard:
            def __init__(self, target, *unused):
                self.target = target

            def inspect(self, *args, **kwargs):
                return {'synthetic': True}, {}

            def wait_idle(self, *unused, **kwargs):
                if self.target == model.pid:
                    postload.append(True)
                    self.inspect()
                return {'synthetic': True}

            def assert_idle(self):
                peer_checks.append(True)
                if scenario == 'peer-active-during-profile' and len(created) == 3:
                    raise RuntimeError('Synthetic peer request during owned profile')

        def memory(loading=False):
            if scenario == 'memory-while-waiting-after-load' and postload:
                raise AssertionError('Synthetic reserve crossed after load')
            return {'synthetic': True}

        def create(command, **kwargs):
            child = [model, measure, profiler][len(created)]
            created.append(child.pid)
            if child is profiler:
                assert kwargs['pass_fds'] == (77,)
                assert '--lifecycle-lock-fd' in command
            return child

        with tempfile.TemporaryDirectory(prefix='qwen-private-profile-check-') as temp:
            base = Path(temp)
            out = base / 'results/qwen-private-fake'
            out.mkdir(parents=True)
            measurement_dir = base / 'results/qwen-private-fake-decode'
            measurement_dir.mkdir()
            maps = base / 'fake-maps'
            maps.write_text('0 0 0 0 0 /test/libggml-cpu.so.0\n0 0 0 0 0 /test/libllama.so.0\n')
            plan = dict(peer_pid=peer_pid, peer=peer, command=info['command'], runtime_env={},
                        cpu='/test/libggml-cpu.so.0', llama='/test/libllama.so.0', label='qwen-private-fake',
                        drafts=4, profile=True)
            (out / 'plan.json').write_text(json.dumps(plan))
            measured = dict(input_integrity_verified=True, checks=[dict(pass_check=True)],
                            server_command=info['command'], runtime_env={}, measurements=[
                                dict(kind=kind, abort=[], counter_metadata=dict(valid=True),
                                     timings=dict(cache_n=0, predicted_per_second=22), background_subtracted_gb_s=125)
                                for kind in ('prose', 'code')])
            (measurement_dir / 'result.json').write_text(json.dumps(measured))

            def path(value):
                return maps if str(value) == f'/proc/{model.pid}/maps' else Path(value)

            with patch.multiple(trial, BASE=base, Path=path, verify_plan=lambda plan:None,
                    inference_snapshot=lambda **kwargs:{str(peer_pid):(0,'123','llama-server')},
                    port_available=lambda port:True, ModelMeasurementGuard=FakeGuard,
                    wait_background=lambda *args:{'synthetic':True}, memory_gate=memory,
                    process_info=lambda pid:info if pid == model.pid else peer,
                    process_environment=lambda pid:{}, read_service=lambda port:dict(processing=0,queued=0)), \
                 patch.object(trial.subprocess, 'Popen', side_effect=create), \
                 patch.object(trial.signal, 'signal'), \
                 patch.object(trial.urllib.request, 'urlopen', side_effect=lambda *args, **kwargs:io.BytesIO(b'{"status":"ok"}')):
                try:
                    trial.execute(out, lifecycle_lock_fd=77)
                except (AssertionError, RuntimeError):
                    pass
                else:
                    raise AssertionError('Failure scenario unexpectedly succeeded')
            result = json.loads((out / 'result.json').read_text())
            assert result['error'] and not result['passed'] and not result['target_reached']
            assert result['peer_preserved'] and model.signals == [signal.SIGTERM]
            assert measure.signals == []
            assert profiler.signals == ([signal.SIGINT] if scenario == 'peer-active-during-profile' else [])
            assert len(created) == (1 if scenario == 'memory-while-waiting-after-load' else 3)
            rows.append(dict(scenario=scenario, passed=True, owned_processes_created=len(created),
                             model_signals=model.signals, profiler_signals=profiler.signals, external_peer_untouched=True))
    destination = trial.BASE / 'results/qwen-private-profile-lifecycle-check-0908.json'
    assert not destination.exists()
    sources = [Path(__file__).resolve(), trial.BASE / 'qwen_private_profile_trial_0908.py',
               trial.BASE / 'profile_qwen_private_0908.py']
    result = dict(time=time.time(), passed=True, synthetic_only=True, checks=rows,
                  source_sha256={str(path):sha256(path) for path in sources})
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
