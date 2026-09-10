#!/usr/bin/env python3
"""Compose the Q6 single-activation helper with the corrected gather runtime."""
import difflib
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from model_measurement_guard import ModelMeasurementGuard, read_service
from qwen_q6_single_transform_0910b import transform
from qwen_split_trial import inference_snapshot, process_info, sha256

BASE = Path(__file__).resolve().parent
ENGINE = BASE.parents[1] / 'engines/llama.cpp-q4e-goal-0904'
OUT = BASE / 'results/qwen-q6-single-build-0910b'


def main():
    assert os.sched_getaffinity(0) == {127}
    assert process_info(1219506)['start'] == '103969952'
    paths = [BASE / 'results' / name for name in (
        'qwen-q6-q8-wide-batch-0907/private-cpu/manifest.json',
        'qwen-get-rows-columns-build-0909/private-cpu/manifest.json',
        'qwen-get-rows-runtime-0909/result.json',
        'qwen-q6-packed-single-proof-0908/result.json',
        'qwen-q6-packed-single-component-qualified-0910/result.json')]
    original, parent, runtime, proof, timing = [json.loads(path.read_text()) for path in paths]
    assert runtime['passed'] and proof['passed'] and timing['passed'] and timing['peer_preserved']
    assert sha256(parent['library']) == parent['library_sha256'] == runtime['cpu_sha256'] == '12c61b337736ca9210433f57c64ce7fffbf7e4b66920aba9a03f97eaef3fd9b7'
    inputs = {}
    for record in (original, parent, runtime, proof, timing):
        for key in ('input_sha256', 'private_source_sha256', 'source_sha256', 'sources'):
            for path, digest in record.get(key, {}).items():
                assert sha256(path) == digest, path
                inputs[path] = digest
    qualified = []
    for case in timing['cases']:
        row = case['attempts'][case['accepted_attempt']]
        assert row['background_within_gate'] and row['other_host_cores'] <= 4
        assert row['output_equality_checked'] and row['weight_storage_unchanged']
        assert case['activations'] == 1 and row['packed_batch_speed_ratio'] > 1.05
        qualified.append(dict(rows=case['rows'], matrices=case['matrices'], speed_ratio=row['packed_batch_speed_ratio']))
    assert {(row['rows'], row['matrices']) for row in qualified} == {(32, 1), (64, 1), (32, 512), (64, 512)}
    command = list(original['compile_commands'][1])
    original_source = Path(command[command.index('-c')+1])
    original_object = command[command.index('-o')+1]
    assert original_object in parent['link_command']
    header = BASE / 'qwen-q6-packed-single-0908.h'
    paths += [Path(__file__).resolve(), BASE / 'qwen_q6_single_transform_0910b.py', original_source, Path(original_object), header]
    inputs.update({str(path): sha256(path) for path in paths})
    guard = ModelMeasurementGuard(1219506, {1219506: 18095}, inference_snapshot)
    owned = None

    def interrupt(signum, frame):
        raise InterruptedError('Release only the owned private Q6 build')

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupt)
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guard.assert_idle()
        OUT.mkdir(exist_ok=False)
        private = OUT / 'private-cpu'
        private.mkdir()
        source = private / 'repack-x86.cpp'
        source.write_text(transform(original_source.read_text()))
        (private / header.name).write_bytes(header.read_bytes())
        (private / 'repack-x86.cpp.patch').write_text(''.join(difflib.unified_diff(
            original_source.read_text().splitlines(True), source.read_text().splitlines(True),
            fromfile=str(original_source), tofile='private/repack-x86.cpp')))
        result = dict(started=time.time(), passed=False, build_completed=False, model_loaded=False,
                      controller_pid=os.getpid(), input_sha256=inputs, steps=[],
                      private_source_sha256={str(path): sha256(path) for path in (source, private / header.name)},
                      parent_sha256=parent['library_sha256'], component_qualification=qualified,
                      scope='Only repack-x86.cpp.o changes. Opt-in NR=1 Q6 arithmetic uses the exact proven helper. Packed weights, expert scheduling, MTP grouping and all other objects remain unchanged. Component results do not establish model speed or bandwidth.')

        def save():
            (OUT / 'result.json').write_text(json.dumps(result, indent=2)+'\n')

        def run(args, name):
            nonlocal owned
            guard.assert_idle()
            log_path = OUT / (name+'.log')
            with log_path.open('w') as log:
                owned = subprocess.Popen(args, cwd=ENGINE/'build-goal', stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                result['owned'] = dict(pid=owned.pid, step=name)
                save()
                deadline = time.monotonic()+600
                while owned.poll() is None:
                    guard.assert_idle()
                    assert time.monotonic() < deadline, name
                    time.sleep(.25)
            result['steps'].append(dict(name=name, command=args, exit_code=owned.returncode, log_sha256=sha256(log_path)))
            save()
            assert owned.returncode == 0, name
            print(json.dumps(dict(completed=name)), flush=True)

        save()
        try:
            baseline_path = BASE / 'results/qwen-q6-single-build-0910/result.json'
            baseline = json.loads(baseline_path.read_text())
            assert baseline['finished'] and not baseline['passed'] and baseline['error'] == "AssertionError('private-compile')"
            assert baseline['baseline_object_identical'] and baseline['baseline_library_identical']
            assert baseline['parent_sha256'] == parent['library_sha256']
            assert all(sha256(path) == digest for path, digest in baseline['input_sha256'].items())
            assert sha256(baseline_path.parent/'baseline-repack-x86.cpp.o') == sha256(original_object)
            assert sha256(baseline_path.parent/'baseline-libggml-cpu.so') == parent['library_sha256']
            inputs[str(baseline_path)] = sha256(baseline_path)
            result.update(baseline_object_identical=True, baseline_library_identical=True,
                          baseline_proof=str(baseline_path), baseline_proof_sha256=sha256(baseline_path))
            obj = private / 'repack-x86.cpp.o'
            command[command.index('-c')+1] = str(source)
            command[command.index('-o')+1] = str(obj)
            command.insert(1, '-I'+str(original_source.parent))
            run(command, 'private-compile')
            library = private / 'libggml-cpu.so.0.22.0'
            link = [str(obj) if value == original_object else value for value in parent['link_command']]
            link[link.index('-o')+1] = str(library)
            run(link, 'private-link')
            (private / 'libggml-cpu.so.0').symlink_to(library.name)
            (private / 'libggml-cpu.so').symlink_to('libggml-cpu.so.0')
            assert all(sha256(path) == digest for path, digest in {**inputs, **result['private_source_sha256']}.items())
            manifest = dict(input_sha256=inputs, private_source_sha256=result['private_source_sha256'],
                            library=str(library), library_sha256=sha256(library), compile_command=command, link_command=link,
                            parent_manifest=str(paths[1]), parent_sha256=parent['library_sha256'], base=parent['base'],
                            baseline_object_identical=True, baseline_library_identical=True, scope=result['scope'])
            (private / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
            result.update(passed=True, build_completed=True, library=str(library), library_sha256=sha256(library))
            print(json.dumps(dict(passed=True, cpu_sha256=result['library_sha256'])), flush=True)
        except BaseException as error:
            result['error'] = repr(error)
            raise
        finally:
            if owned is not None and owned.poll() is None:
                os.killpg(owned.pid, signal.SIGTERM)
                owned.wait(timeout=15)
            result.update(finished=time.time(), peer_preserved=process_info(1219506)['start'] == '103969952', peer_service=read_service(18095))
            save()


if __name__ == '__main__':
    os.umask(0o077)
    main()
