#!/usr/bin/env python3
"""Reconstruct the pinned base library without editing its recorded source tree."""
import difflib
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time

from model_measurement_guard import ModelMeasurementGuard
from qwen_split_trial import inference_snapshot, process_info, sha256

BASE = Path(__file__).resolve().parent
ENGINE = BASE.parents[1] / 'engines/llama.cpp-q4e-goal-0904'
OUT = BASE / 'results/qwen-pinned-base-recovery-0909b'


def recover(text):
    block = '''                            int n_tasks = GGML_N_TASKS_MAX;
                            const char * task_hint = getenv("GGML_CPU_NUMA_FUSED_REDUCE_TASK_HINT");
                            if (task_hint && atoi(task_hint) == 1 && ud->desc->single_task_max_elements > 0) {
                                n_tasks = 1;
                                for (int r = 0; r < ud->desc->n_red; ++r) {
                                    if (ggml_nelements(ud->desc->nodes[r][j]) > ud->desc->single_task_max_elements) {
                                        n_tasks = GGML_N_TASKS_MAX;
                                        break;
                                    }
                                }
                            }
'''
    call = 'nullptr, 0, ggml_backend_meta_fused_reduce_op, n_tasks, ud.get());'
    assert text.count(block) == text.count(call) == 1
    return text.replace(block, '').replace(call, call.replace('n_tasks', 'GGML_N_TASKS_MAX'))


def main():
    assert os.sched_getaffinity(0) == {127}
    assert process_info(1219506)['start'] == '103969952'
    prior_path = BASE / 'results/qwen-shared-dispatch-build-0909/result.json'
    prior = json.loads(prior_path.read_text())
    assert not prior['passed'] and prior['baseline_libraries'] == {'cpu': True, 'base': False}
    assert all(sha256(path) == value for path, value in prior['input_sha256'].items())
    original_path = ENGINE / 'ggml/src/ggml-backend-meta.cpp'
    original = original_path.read_text()
    recovered = recover(original)
    database = json.loads((ENGINE / 'build-goal/compile_commands.json').read_text())
    rows = [row for row in database if row['file'] == str(original_path)]
    assert len(rows) == 1
    row = rows[0]
    directory = Path(row['directory'])
    compile_command = shlex.split(row['command'])
    original_object = (directory / compile_command[compile_command.index('-o') + 1]).resolve()
    link = shlex.split((directory / 'CMakeFiles/ggml-base.dir/link.txt').read_text())
    link = [str((directory / value).resolve()) if value.endswith('.o') and not Path(value).is_absolute() else value for value in link]
    assert str(original_object) in link
    pinned = ENGINE / 'validated-iq-batch3-bin/libggml-base.so.0.22.0'
    assert sha256(pinned) == 'a8d6ff25ffab12b993c98f11c6d5b4146fc1dd5bd7bebb8c43303842cd959d36'
    guard = ModelMeasurementGuard(1219506, {1219506: 18095}, inference_snapshot)
    owned = None

    def interrupted(signum, frame):
        raise InterruptedError('Release only the owned pinned-base reconstruction')

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupted)
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guard.assert_idle()
        OUT.mkdir(exist_ok=False)
        source = OUT / original_path.name
        source.write_text(recovered)
        (OUT / 'recovery.patch').write_text(''.join(difflib.unified_diff(
            original.splitlines(True), recovered.splitlines(True), fromfile=str(original_path), tofile=str(source))))
        obj = OUT / 'ggml-backend-meta.cpp.o'
        library = OUT / 'libggml-base.so.0.22.0'
        compile_command[compile_command.index('-o') + 1] = str(obj)
        compile_command[compile_command.index('-c') + 1] = str(source)
        compile_command.insert(1, '-fmacro-prefix-map=' + str(source) + '=' + str(original_path))
        compile_command.insert(1, '-fmacro-prefix-map=' + str(ENGINE) + '/ggml/src/./ggml-impl.h=' + str(ENGINE / 'ggml/src/ggml-impl.h'))
        link = [str(obj) if value == str(original_object) else value for value in link]
        link[link.index('-o') + 1] = str(library)
        inputs = dict(prior['input_sha256'])
        inputs.update({str(p): sha256(p) for p in [Path(__file__).resolve(), prior_path, source]})
        result = dict(started=time.time(), controller_pid=os.getpid(), passed=False,
                      input_sha256=inputs, original_source=str(original_path), recovered_source=str(source),
                      pinned_library=str(pinned), steps=[], model_loaded=False,
                      scope='Remove only the optional task-hint block, preserving the original file macro and all other base objects. Require byte-identical pinned library reconstruction.')

        def save():
            (OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')

        def run(command, label):
            nonlocal owned
            guard.assert_idle()
            with (OUT / (label + '.log')).open('w') as log:
                owned = subprocess.Popen(command, cwd=directory, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                result['owned'] = dict(pid=owned.pid, label=label)
                save()
                deadline = time.monotonic() + 600
                while owned.poll() is None:
                    guard.assert_idle()
                    assert time.monotonic() < deadline, label
                    time.sleep(.25)
            result['steps'].append(dict(label=label, command=command, directory=str(directory), exit_code=owned.returncode))
            save()
            assert owned.returncode == 0, label
            print(json.dumps(dict(completed=label)), flush=True)

        save()
        try:
            run(compile_command, 'compile-recovered-source')
            run(link, 'link-recovered-base')
            result['library_sha256'] = sha256(library)
            result['object_sha256'] = sha256(obj)
            result['pinned_sha256'] = sha256(pinned)
            result['byte_identical'] = result['library_sha256'] == result['pinned_sha256']
            assert result['byte_identical'], 'Recovered library is not the pinned base'
            assert all(sha256(path) == value for path, value in inputs.items())
            result.update(passed=True, library=str(library), object=str(obj))
            print(json.dumps(dict(passed=True, byte_identical=True, library_sha256=result['library_sha256'])), flush=True)
        except BaseException as error:
            result['error'] = repr(error)
            raise
        finally:
            if owned is not None and owned.poll() is None:
                os.killpg(owned.pid, signal.SIGTERM)
                owned.wait(timeout=10)
            result['finished'] = time.time()
            result['peer_preserved'] = process_info(1219506)['start'] == '103969952'
            save()


if __name__ == '__main__':
    main()
