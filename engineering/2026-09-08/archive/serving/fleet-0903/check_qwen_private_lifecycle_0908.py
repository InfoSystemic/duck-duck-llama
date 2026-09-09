#!/usr/bin/env python3
"""Exercise refusal and cleanup paths with fake processes; never load a model."""
import json
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

import qwen_private_trial_0908 as trial
from qwen_split_trial import sha256


class FakeModel:
    def __init__(self, exited=False):
        self.pid = 912345
        self.returncode = 1 if exited else None
        self.terminations = 0

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminations += 1
        self.returncode = -15

    def wait(self, timeout):
        assert self.returncode is not None
        return self.returncode

    def kill(self):
        raise AssertionError('A responding owned process must not be killed')


def main():
    rows = []
    for scenario in ('memory-before-load', 'peer-active-during-load', 'memory-during-load', 'owned-model-exits'):
        peer = dict(start='123', exe='/test/peer-server', command=['/test/peer-server'], affinity=[0, 1])
        model = FakeModel(exited=scenario == 'owned-model-exits')
        created = []
        class FakeGuard:
            def __init__(self, *unused):
                pass
            def wait_idle(self, *unused, **kwargs):
                return {'synthetic':True}
            def assert_idle(self):
                if scenario == 'peer-active-during-load':
                    raise RuntimeError('Synthetic peer request arrived')
        def memory(loading=False):
            if (not loading and scenario == 'memory-before-load') or (loading and scenario == 'memory-during-load'):
                raise AssertionError('Synthetic memory reserve failure')
            return {'synthetic':True}
        def create(*args, **kwargs):
            created.append(model.pid)
            return model
        with tempfile.TemporaryDirectory(prefix='qwen-private-lifecycle-') as temporary:
            out = Path(temporary)
            plan = dict(peer_pid=812345, peer=peer, command=['/test/private-server'], runtime_env={})
            (out / 'plan.json').write_text(json.dumps(plan))
            with patch.multiple(trial, verify_plan=lambda plan:None, inference_snapshot=lambda **kwargs:{'812345':(0,'123','llama-server')},
                                port_available=lambda port:True, ModelMeasurementGuard=FakeGuard,
                                wait_background=lambda *args:{'synthetic':True}, memory_gate=memory,
                                process_info=lambda pid:peer, read_service=lambda port:dict(processing=0,queued=0)), \
                 patch.object(trial.subprocess, 'Popen', side_effect=create), \
                 patch.object(trial.signal, 'signal'), \
                 patch.object(trial.urllib.request, 'urlopen', side_effect=AssertionError('No HTTP request should be reached')):
                try:
                    trial.execute(out)
                except (AssertionError, RuntimeError):
                    pass
                else:
                    raise AssertionError('Failure scenario unexpectedly succeeded')
            result = json.loads((out / 'result.json').read_text())
            assert not result['passed'] and not result['target_reached'] and result['peer_preserved']
            assert bool(created) == (scenario != 'memory-before-load')
            assert model.terminations == int(scenario in ('peer-active-during-load', 'memory-during-load'))
            assert 'error' in result
            rows.append(dict(scenario=scenario, passed=True, private_process_created=bool(created),
                             owned_terminations=model.terminations, external_process_untouched=True))
    destination = trial.BASE / 'results/qwen-private-lifecycle-check-0908.json'
    assert not destination.exists()
    result = dict(time=time.time(), passed=True, checks=rows, synthetic_only=True, model_loaded=False,
                  source_sha256={str(path):sha256(path) for path in [Path(__file__).resolve(), trial.BASE/'qwen_private_trial_0908.py']})
    destination.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
