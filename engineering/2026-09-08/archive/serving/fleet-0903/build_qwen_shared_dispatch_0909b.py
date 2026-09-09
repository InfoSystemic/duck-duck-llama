#!/usr/bin/env python3
"""Build isolated CPU/base libraries for the shared-dispatch experiment."""
import difflib
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time

from model_measurement_guard import ModelMeasurementGuard, read_service
from qwen_shared_dispatch_transform_0909 import cpu_source, meta_source
from qwen_split_trial import inference_snapshot, process_info, sha256

BASE = Path(__file__).resolve().parent
ENGINE = BASE.parents[1] / 'engines/llama.cpp-q4e-goal-0904'
OUT = BASE / 'results/qwen-shared-dispatch-build-0909b'
PINNED_BASE = ENGINE / 'validated-iq-batch3-bin/libggml-base.so.0.22.0'


def main():
    assert os.sched_getaffinity(0) == {127}
    assert process_info(1219506)['start'] == '103969952'
    parent_path = BASE / 'results/qwen-q6-q8-wide-batch-0907/private-cpu/manifest.json'
    database_path = ENGINE / 'build-goal/compile_commands.json'
    base_link_path = ENGINE / 'build-goal/ggml/src/CMakeFiles/ggml-base.dir/link.txt'
    parent = json.loads(parent_path.read_text())
    assert sha256(parent['library']) == parent['library_sha256'] == 'c79e19e38acfd6ada3f0637c136cf9cb10253127778cb214528b2f966fb6dfcf'
    assert sha256(PINNED_BASE) == 'a8d6ff25ffab12b993c98f11c6d5b4146fc1dd5bd7bebb8c43303842cd959d36'
    assert all(sha256(path) == value for path, value in parent['input_sha256'].items())
    assert all(sha256(path) == value for path, value in parent['private_source_sha256'].items())
    recovery_path = BASE / 'results/qwen-pinned-base-recovery-0909b/result.json'
    recovery = json.loads(recovery_path.read_text())
    assert recovery['passed'] and recovery['byte_identical']
    assert sha256(recovery['library']) == recovery['library_sha256'] == sha256(PINNED_BASE)
    assert all(sha256(path) == value for path, value in recovery['input_sha256'].items())
    database = json.loads(database_path.read_text())
    units = []
    for name, transform in [('ggml-cpu.cpp', cpu_source), ('ggml-backend-meta.cpp', meta_source)]:
        matches = [row for row in database if Path(row['file']).name == name]
        assert len(matches) == 1
        row = matches[0]
        command = shlex.split(row['command'])
        directory = Path(row['directory'])
        source = Path(row['file'])
        obj = (directory / command[command.index('-o') + 1]).resolve()
        if name == 'ggml-backend-meta.cpp':
            source = Path(recovery['recovered_source'])
            obj = Path(recovery['object'])
            command = list(recovery['steps'][0]['command'])
        original = source.read_text()
        units.append(dict(name=name, directory=directory, command=command, source=source,
                          object=obj, original=original, changed=transform(original)))
    assert units[0]['directory'] == units[1]['directory']
    directory = units[0]['directory']
    base_link = shlex.split(base_link_path.read_text())
    for index, value in enumerate(base_link):
        if value.endswith('.o') and not Path(value).is_absolute():
            base_link[index] = str((directory / value).resolve())
    original_meta_object = str((directory / 'CMakeFiles/ggml-base.dir/ggml-backend-meta.cpp.o').resolve())
    assert original_meta_object in base_link
    base_link = [str(units[1]['object']) if value == original_meta_object else value for value in base_link]
    assert str(units[0]['object']) in parent['link_command']
    assert str(units[1]['object']) in base_link
    header = BASE / 'qwen-shared-dispatch-0909.h'
    paths = [Path(__file__).resolve(), BASE / 'qwen_shared_dispatch_transform_0909.py',
             header, parent_path, database_path, base_link_path, PINNED_BASE, recovery_path, Path(recovery['library']),
             Path(parent['library']), BASE / 'model_measurement_guard.py', BASE / 'qwen_split_trial.py']
    paths += [unit['source'] for unit in units]
    paths += [Path(value) for link in (base_link, parent['link_command'])
              for value in link if Path(value).is_absolute() and Path(value).is_file()]
    paths += list((ENGINE / 'ggml/src').rglob('*.h')) + list((ENGINE / 'ggml/include').rglob('*.h'))
    inputs = {str(path): sha256(path) for path in paths}
    guard = ModelMeasurementGuard(1219506, {1219506: 18095}, inference_snapshot)
    owned = None

    def interrupted(signum, frame):
        raise InterruptedError('Stop only the owned shared-dispatch build')

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupted)
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guard.assert_idle()
        OUT.mkdir(exist_ok=False)
        private = OUT / 'private-runtime'
        private.mkdir()
        (private / header.name).write_bytes(header.read_bytes())
        for unit in units:
            source = private / unit['name']
            source.write_text(unit['changed'])
            (private / (unit['name'] + '.patch')).write_text(''.join(difflib.unified_diff(
                unit['original'].splitlines(True), unit['changed'].splitlines(True),
                fromfile=str(unit['source']), tofile='private/' + unit['name'])))
        private_sources = {str(path): sha256(path) for path in
                           [private / header.name, *[private / unit['name'] for unit in units]]}
        result = dict(started=time.time(), controller_pid=os.getpid(), passed=False,
                      build_completed=False, input_sha256=inputs, private_source_sha256=private_sources,
                      parent_cpu_sha256=parent['library_sha256'], parent_base_sha256=sha256(PINNED_BASE),
                      pinned_base_recovery=str(recovery_path),
                      baseline_objects={}, baseline_libraries={}, steps=[], model_loaded=False,
                      scope='Only ggml-cpu.cpp.o and ggml-backend-meta.cpp.o change. Default-off shared ordered Meta dispatch and synchronous NUMA CPU execution; numeric kernels and weights are unchanged.')

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
            for unit in units:
                command = list(unit['command'])
                obj = OUT / ('baseline-' + unit['name'] + '.o')
                command[command.index('-o') + 1] = str(obj)
                run(command, 'baseline-' + unit['name'])
                identical = sha256(obj) == sha256(unit['object'])
                result['baseline_objects'][unit['name']] = identical
                save()
                assert identical, unit['name']
            for name, link, unit, parent_hash in [
                ('cpu', parent['link_command'], units[0], parent['library_sha256']),
                ('base', base_link, units[1], sha256(PINNED_BASE))]:
                command = [str(OUT / ('baseline-' + unit['name'] + '.o')) if value == str(unit['object']) else value for value in link]
                library = OUT / ('parent-' + name + '.so')
                command[command.index('-o') + 1] = str(library)
                run(command, 'parent-link-' + name)
                result['baseline_libraries'][name] = sha256(library) == parent_hash
                save()
                assert result['baseline_libraries'][name], name
            compiles = []
            for unit in units:
                command = list(unit['command'])
                command[command.index('-o') + 1] = str(private / (unit['name'] + '.o'))
                command[command.index('-c') + 1] = str(private / unit['name'])
                run(command, 'private-' + unit['name'])
                compiles.append(command)
            libraries, links = {}, {}
            for name, link, unit in [('base', base_link, units[1]), ('cpu', parent['link_command'], units[0])]:
                command = [str(private / (unit['name'] + '.o')) if value == str(unit['object']) else value for value in link]
                library = private / ('libggml-' + name + '.so.0.22.0')
                command[command.index('-o') + 1] = str(library)
                if name == 'cpu':
                    command = [libraries['base']['path'] if value == str(PINNED_BASE) else value for value in command]
                run(command, 'private-link-' + name)
                (private / ('libggml-' + name + '.so.0')).symlink_to(library.name)
                (private / ('libggml-' + name + '.so')).symlink_to('libggml-' + name + '.so.0')
                libraries[name] = dict(path=str(library), sha256=sha256(library))
                links[name] = command
            assert all(sha256(path) == value for path, value in {**inputs, **private_sources}.items())
            result.update(passed=True, build_completed=True, libraries=libraries)
            manifest = dict(input_sha256=inputs, private_source_sha256=private_sources,
                            pinned_base_recovery=str(recovery_path),
                            parent_cpu_manifest=str(parent_path), parent_base=str(PINNED_BASE),
                            baseline_objects=result['baseline_objects'], baseline_libraries=result['baseline_libraries'],
                            compile_commands=compiles, link_commands=links, libraries=libraries,
                            scope=result['scope'])
            (private / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
            print(json.dumps(dict(passed=True, libraries=libraries)), flush=True)
        except BaseException as error:
            result['error'] = repr(error)
            raise
        finally:
            if owned is not None and owned.poll() is None:
                os.killpg(owned.pid, signal.SIGTERM)
                owned.wait(timeout=10)
            result['finished'] = time.time()
            result['peer_preserved'] = process_info(1219506)['start'] == '103969952'
            result['peer_after'] = read_service(18095)
            save()


if __name__ == '__main__':
    main()
