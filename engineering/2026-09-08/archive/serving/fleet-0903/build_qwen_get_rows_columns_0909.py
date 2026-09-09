#!/usr/bin/env python3
"""Reproduce the current CPU library, then build the isolated gather correction."""
import argparse
import difflib
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import time

from model_measurement_guard import ModelMeasurementGuard, read_service
from qwen_get_rows_columns_transform_0909 import transform
from qwen_split_trial import inference_snapshot, process_info, sha256

BASE = Path(__file__).resolve().parent
ENGINE = BASE.parents[1] / 'engines/llama.cpp-q4e-goal-0904'


def main(label):
    assert os.sched_getaffinity(0) == {127}
    assert process_info(1219506)['start'] == '103969952'
    parent_path = BASE / 'results/qwen-shared-dispatch-build-0909b/private-runtime/manifest.json'
    database_path = ENGINE / 'build-goal/compile_commands.json'
    parent = json.loads(parent_path.read_text())
    assert parent['libraries']['cpu']['sha256'] == 'e68c0c543aa3334a1f1ed732c11f8f4fe8ecf097ea816eb89f82fe9ae1ed6a0a'
    assert all(sha256(row['path']) == row['sha256'] for row in parent['libraries'].values())
    assert all(sha256(path) == digest for path, digest in parent['input_sha256'].items())
    assert all(sha256(path) == digest for path, digest in parent['private_source_sha256'].items())
    record, = [row for row in json.loads(database_path.read_text()) if row['file'].endswith('/ggml-cpu/ops.cpp')]
    compile_command = shlex.split(record['command'])
    directory = Path(record['directory'])
    original_path = Path(record['file'])
    original_object = (directory / compile_command[compile_command.index('-o')+1]).resolve()
    assert sha256(original_object) == '9384fa80213c78f7ab1364e925b5a4ee921f36c2c3ab02ce3c8794dc1e62bbaf'
    link_command = parent['link_commands']['cpu']
    assert link_command.count(str(original_object)) == 1
    original = original_path.read_text()
    changed = transform(original)
    inputs = {str(path):sha256(path) for path in [Path(__file__).resolve(),
              BASE / 'qwen_get_rows_columns_transform_0909.py', BASE / 'model_measurement_guard.py',
              BASE / 'qwen_split_trial.py', parent_path, database_path, original_path, original_object]}
    inputs.update(parent['input_sha256'])
    inputs.update(parent['private_source_sha256'])
    inputs.update({row['path']:row['sha256'] for row in parent['libraries'].values()})
    guard = ModelMeasurementGuard(1219506, {1219506:18095}, inference_snapshot)
    owned = None
    def interrupted(signum, frame):
        raise InterruptedError('Release only the owned gather build')
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupted)
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guard.assert_idle()
        out = BASE / 'results' / label
        out.mkdir()
        private = out / 'private-cpu'
        private.mkdir()
        source = private / 'ops.cpp'
        source.write_text(changed)
        (private / 'get-rows-columns.patch').write_text(''.join(difflib.unified_diff(
            original.splitlines(True), changed.splitlines(True), fromfile=str(original_path), tofile='private/ops.cpp')))
        result = dict(started=time.time(), passed=False, build_completed=False, model_loaded=False,
                      controller_pid=os.getpid(), input_sha256=inputs, private_source_sha256={str(source):sha256(source)},
                      parent_sha256=parent['libraries']['cpu']['sha256'], steps=[],
                      scope='Only ops.cpp.o changes. Restore canonical dequantization for wide quantized gathers and apply the existing column-copy implementation to FP32/I32 gathers. No model weights or matrix arithmetic are changed.')
        def save():
            (out / 'result.json').write_text(json.dumps(result, indent=2)+'\n')
        def run(command, name):
            nonlocal owned
            guard.assert_idle()
            with (out / (name + '.log')).open('w') as log:
                owned = subprocess.Popen(command, cwd=directory, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                result['owned'] = dict(pid=owned.pid, step=name)
                save()
                deadline = time.monotonic()+600
                while owned.poll() is None:
                    guard.assert_idle()
                    assert time.monotonic() < deadline, name
                    time.sleep(.25)
            result['steps'].append(dict(label=name, command=command, exit_code=owned.returncode))
            save()
            assert owned.returncode == 0, name
            print(json.dumps(dict(completed=name)), flush=True)
        save()
        try:
            obj = out / 'baseline-ops.cpp.o'
            command = list(compile_command)
            command[command.index('-o')+1] = str(obj)
            run(command, 'baseline-compile')
            result['baseline_object_identical'] = sha256(obj) == sha256(original_object)
            assert result['baseline_object_identical'], 'Current ops.cpp does not reproduce the measured runtime object'
            baseline = out / 'baseline-libggml-cpu.so'
            link = [str(obj) if value == str(original_object) else value for value in link_command]
            link[link.index('-o')+1] = str(baseline)
            run(link, 'baseline-link')
            result['baseline_library_identical'] = sha256(baseline) == parent['libraries']['cpu']['sha256']
            assert result['baseline_library_identical']
            obj = private / 'ops.cpp.o'
            command = list(compile_command)
            command[command.index('-c')+1] = str(source)
            command[command.index('-o')+1] = str(obj)
            run(command, 'private-compile')
            library = private / 'libggml-cpu.so.0.22.0'
            link = [str(obj) if value == str(original_object) else value for value in link_command]
            link[link.index('-o')+1] = str(library)
            run(link, 'private-link')
            (private / 'libggml-cpu.so.0').symlink_to(library.name)
            (private / 'libggml-cpu.so').symlink_to('libggml-cpu.so.0')
            assert all(sha256(path) == digest for path, digest in {**inputs, **result['private_source_sha256']}.items())
            result.update(passed=True, build_completed=True, library=str(library), library_sha256=sha256(library))
            manifest = dict(input_sha256=inputs, private_source_sha256=result['private_source_sha256'],
                parent_manifest=str(parent_path), parent_sha256=result['parent_sha256'],
                library=str(library), library_sha256=sha256(library), compile_command=command, link_command=link,
                base=parent['libraries']['base'], scope=result['scope'])
            (private / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
            print(json.dumps(dict(passed=True, library=str(library), sha256=sha256(library))), flush=True)
        except BaseException as error:
            result['error'] = repr(error)
            raise
        finally:
            if owned is not None and owned.poll() is None:
                os.killpg(owned.pid, signal.SIGTERM)
                owned.wait(timeout=15)
            result.update(finished=time.time(), peer_preserved=process_info(1219506)['start']=='103969952',
                          peer_service=read_service(18095))
            save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('label')
    args = parser.parse_args()
    assert re.fullmatch(r'qwen-get-rows-columns-build-[A-Za-z0-9_-]+', args.label)
    os.umask(0o077)
    main(args.label)
