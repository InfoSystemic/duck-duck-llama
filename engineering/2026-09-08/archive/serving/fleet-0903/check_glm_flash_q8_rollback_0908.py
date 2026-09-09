#!/usr/bin/env python3
"""Exercise model-swap failure paths using fakes; never inspect or signal a PID."""
import copy
import json
from pathlib import Path
import tempfile
import time

import glm_flash_q8_trial as trial

BASE = Path(__file__).resolve().parent
OUT = BASE / 'results/glm-flash-q8-rollback-check-0908.json'


def main():
    assert not OUT.exists()
    selected = json.loads((BASE / 'results/qwen-q6-trial-0907/state.json').read_text())['current']
    # Avoid reading or writing the actual mutable thread-count file in these fakes.
    selected = copy.deepcopy(selected)
    selected['runtime_env'].pop('GGML_CPU_NUMA_THREADS_FILE', None)
    result = dict(started=time.time(), passed=False, checks=[], scope='Fake service identities only; no model actions')
    original_snapshot, original_context = trial.inference_snapshot, trial.CONTEXT
    try:
        for failure in ('preflight', 'idle_gate', 'stop', 'load', 'foreign_during_load', None):
            calls = []
            live = {'qwen'}
            def fail(where):
                if failure == where:
                    raise RuntimeError(where)
            class Qwen:
                def __init__(self):
                    self.state = dict(current=copy.deepcopy(selected), full_stopped=True)
                    self.recovering = False
                def stop_current(self):
                    calls.append('stop_qwen')
                    fail('stop')
                    live.discard('qwen')
                    self.state['current'] = None
                def start_process(self, config, env, role):
                    assert not live
                    assert config == selected
                    assert trial.runtime_environment(env) == config['runtime_env']
                    assert env['PRESERVED_TEST_VALUE'] == 'retain'
                    assert 'GGML_CPU_OP_PROFILE' not in env
                    assert role == 'post-flash-q8'
                    calls.append('restore_selected_q6')
                    live.add('qwen')
                    self.state['current'] = copy.deepcopy(config)
            manager = trial.Manager.__new__(trial.Manager)
            manager.state = dict(events=[], current=None)
            manager.cancelled = manager.recovering = False
            manager.qwen = Qwen()
            manager.record = lambda *args, **kwargs: None
            def candidate(*_):
                fail('preflight')
                return {'new': 'flash'}, {}
            manager.candidate = candidate
            manager.quiet = lambda: fail('idle_gate')
            def stop_flash():
                assert manager.state['current'] is None
            manager.stop_flash = stop_flash
            def start_flash(*_):
                assert not live
                calls.append('start_flash')
                if failure == 'foreign_during_load':
                    live.add('foreign')
                    raise RuntimeError(failure)
                fail('load')
                live.add('flash')
                manager.state['current'] = {'fake': 'flash'}
            manager.start_flash = start_flash
            trial.inference_snapshot = lambda: {name: [] for name in live}
            with tempfile.TemporaryDirectory(prefix='flash-q8-rollback-') as directory:
                trial.CONTEXT = Path(directory) / 'fake-context.json'
                trial.CONTEXT.write_text(json.dumps({'environment': {
                    'PRESERVED_TEST_VALUE': 'retain', 'GGML_CPU_OP_PROFILE': '*', 'LD_LIBRARY_PATH': 'stale'}}))
                raised = None
                try:
                    manager.launch(False, 2, 15)
                except RuntimeError as error:
                    raised = str(error)
                assert raised == failure, (failure, raised)
            expected_calls = {
                'preflight': [], 'idle_gate': [], 'stop': ['stop_qwen'],
                'load': ['stop_qwen', 'start_flash', 'restore_selected_q6'],
                'foreign_during_load': ['stop_qwen', 'start_flash'],
                None: ['stop_qwen', 'start_flash'],
            }[failure]
            assert calls == expected_calls, (failure, calls)
            assert live == ({'flash'} if failure is None else {'foreign'} if failure == 'foreign_during_load' else {'qwen'})
            result['checks'].append(dict(failure=failure, passed=True, actions=calls))
        result['passed'] = True
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        trial.inference_snapshot, trial.CONTEXT = original_snapshot, original_context
        result['finished'] = time.time()
        OUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(passed=True, cases=len(result['checks']))))


if __name__ == '__main__':
    main()
