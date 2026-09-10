#!/usr/bin/env python3
"""Bounded fault checks for model identity, launch admission, and cancellation."""
import copy
import hashlib
import json
import os
from pathlib import Path
import runpy
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch

import select_flash_q4_0910 as trial

BASE = Path(__file__).resolve().parent


def main():
    checks = []
    with tempfile.TemporaryDirectory(prefix='flash-q4-selection-check-') as temp:
        root = Path(temp)
        model, library = root / 'tiny.gguf', root / 'tiny.so'
        model.write_bytes(b'model-fixture')
        library.write_bytes(b'library-fixture')
        st = model.stat()
        record = dict(path=str(model), bytes=st.st_size, inode=st.st_ino, mtime_ns=st.st_mtime_ns)
        trial.verify_records([record])
        checks.append('matching model metadata accepted')
        bad = dict(record, bytes=st.st_size + 1)
        try:
            trial.verify_records([bad])
        except AssertionError:
            checks.append('changed model metadata rejected')
        else:
            raise AssertionError('Model identity gate did not reject changed size')

        manager = trial.Manager.__new__(trial.Manager)
        manager.qwen = SimpleNamespace(state={'full_stopped': True})
        manager.state = {'current': None}
        manager.cancelled, manager.recovering = False, False
        manager.record = Mock()
        good_nodes = {f'node{i}': {'estimated_available': 70000000000} for i in range(4)}
        for name, available, nodes, inference in [
            ('global reserve rejects load before spawning', 259000000000, good_nodes, {}),
            ('per-node reserve rejects load before spawning', 280000000000, dict(good_nodes, node0={'estimated_available': 59000000000}), {}),
            ('competing model rejects load before spawning', 280000000000, good_nodes, {'123': (0, '1', 'llama-server')}),
        ]:
            with patch.object(trial, 'inference_snapshot', return_value=inference), patch.object(trial, 'port_available', return_value=True), \
                 patch.object(trial, 'unit_state', return_value={'ActiveState': 'inactive'}), \
                 patch.object(trial, 'memory_status', return_value={'MemAvailable': available}), \
                 patch.object(trial, 'node_memory_status', return_value=nodes), patch.object(trial.subprocess, 'Popen') as spawn:
                try:
                    manager.start_flash({}, {})
                except AssertionError:
                    pass
                else:
                    raise AssertionError(name)
                spawn.assert_not_called()
                checks.append(name)

        child = Mock(pid=12345, returncode=-15)
        child.poll.return_value = None
        with patch.object(trial, 'OUT', root), patch.object(trial, 'inference_snapshot', return_value={}), \
             patch.object(trial, 'port_available', return_value=True), patch.object(trial, 'unit_state', return_value={'ActiveState': 'inactive'}), \
             patch.object(trial, 'memory_status', return_value={'MemAvailable': 280000000000}), \
             patch.object(trial, 'node_memory_status', return_value=good_nodes), \
             patch.object(trial.subprocess, 'Popen', return_value=child), \
             patch.object(manager, 'check_cancel', side_effect=[None, InterruptedError('fixture')]), \
             patch.object(trial.os, 'killpg') as kill:
            try:
                manager.start_flash(dict(affinity=[0], command=['fixture-server'], batching=True, drafts=2), {})
            except InterruptedError:
                pass
            else:
                raise AssertionError('Cancellation did not propagate')
            kill.assert_called_once_with(child.pid, trial.signal.SIGTERM)
            child.wait.assert_called_once_with(timeout=30)
            assert manager.state['current'] is None
            checks.append('cancelled load terminates and reaps only its owned child')

        launcher = root / 'launch-glm-flash-validated.py'
        launcher.write_bytes((BASE / launcher.name).read_bytes())
        (root / 'results/qwen-q6-trial-0907').mkdir(parents=True)
        legacy = dict(command=['legacy-server'], runtime_env={'GGML_TEST': 'legacy'})
        (root / 'glm-flash-validated.json').write_text(json.dumps(legacy))
        chosen = dict(command=['selected-server'], runtime_env={'GGML_TEST': 'selected'}, model_records=[record],
                      runtime_sha256={str(library): hashlib.sha256(library.read_bytes()).hexdigest()}, affinity=[0])
        class Executed(Exception):
            pass
        def launch(config, inference=None, mutate=None):
            selection = root / 'glm-flash-selected.json'
            if config is None:
                selection.unlink(missing_ok=True)
            else:
                selection.write_text(json.dumps(config))
            if mutate:
                mutate()
            with patch.dict(os.environ, {'KEEP': 'yes', 'GGML_FOREIGN': 'bad', 'LLAMA_MTP_FOREIGN': 'bad', 'LD_LIBRARY_PATH': 'bad'}, clear=True), \
                 patch('glm_flash_q8_trial.memory_status', return_value={'MemAvailable': 280000000000}), \
                 patch('glm_flash_q8_trial.node_memory_status', return_value=good_nodes), \
                 patch('qwen_split_trial.inference_snapshot', return_value=inference or {}), \
                 patch('os.sched_setaffinity') as affinity, patch('os.execvpe', side_effect=Executed) as execute, \
                 patch('fcntl.flock'), patch('sys.argv', ['fixture-launcher']):
                try:
                    runpy.run_path(str(launcher), run_name='__main__')
                except Executed:
                    return execute.call_args, affinity.call_args

        called, affinity = launch(chosen)
        assert called.args[0] == 'selected-server' and called.args[2] == {'KEEP': 'yes', 'GGML_TEST': 'selected'}
        assert affinity.args == (0, [0])
        checks.append('selected launcher verifies identities, isolates runtime environment, and sets full configured affinity')
        called, affinity = launch(None)
        assert called.args[0] == 'legacy-server' and affinity is None
        checks.append('legacy launcher remains available before selection')
        for name, config, other in [
            ('launcher rejects changed model identity', dict(chosen, model_records=[bad]), None),
            ('launcher rejects changed runtime bytes', dict(chosen, runtime_sha256={str(library): '0' * 64}), None),
            ('launcher rejects another loaded model', chosen, {'123': (0, '1', 'llama-server')}),
        ]:
            try:
                launch(config, other)
            except RuntimeError:
                checks.append(name)
            else:
                raise AssertionError(name)
    result = dict(passed=True, checks=checks, sources={str(BASE / name): trial.sha256(BASE / name) for name in
                  ('check_flash_q4_selection_0910.py', 'select_flash_q4_0910.py', 'launch-glm-flash-validated.py')},
                  scope='Small synthetic fault checks; no model load, download, process signal, or generated inference.')
    out = BASE / 'results/flash-q4-selection-checks-0910.json'
    assert not out.exists()
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
