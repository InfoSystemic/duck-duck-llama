#!/usr/bin/env python3
"""Build opt-in block scheduling on the selected Qwen CPU library."""
import difflib
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from model_measurement_guard import ModelMeasurementGuard, read_service
from qwen_quantize_blocks_transform_0909 import transform
from qwen_split_trial import inference_snapshot, process_info, sha256

BASE = Path(__file__).resolve().parent
ENGINE = BASE.parents[1] / 'engines/llama.cpp-q4e-goal-0904'
OUT = BASE / 'results/qwen-quantize-blocks-build-0909'


def main():
    assert os.sched_getaffinity(0) == {127}
    assert process_info(1219506)['start'] == '103969952'
    parent_path = BASE / 'results/qwen-q6-q8-wide-batch-0907/private-cpu/manifest.json'
    proof_path = BASE / 'results/qwen-quantize-blocks-proof-0909/result.json'
    parent, proof = [json.loads(path.read_text()) for path in (parent_path, proof_path)]
    assert sha256(parent['library']) == parent['library_sha256'] == 'c79e19e38acfd6ada3f0637c136cf9cb10253127778cb214528b2f966fb6dfcf'
    assert proof['passed'] and proof['checks'][0]['cases'] == 720
    assert all(sha256(path) == value for record in (parent, proof) for path, value in record['input_sha256'].items())
    assert all(sha256(path) == value for path, value in parent['private_source_sha256'].items())
    command = list(parent['compile_commands'][0])
    source = Path(command[-1])
    original_object = command[command.index('-o') + 1]
    assert source.name == 'repack.cpp' and original_object in parent['link_command']
    original = source.read_text()
    changed = transform(original)
    header = BASE / 'qwen-quantize-blocks-0909.h'
    paths = [Path(__file__).resolve(), BASE / 'qwen_quantize_blocks_transform_0909.py',
             parent_path, proof_path, source, header, Path(parent['library']),
             BASE / 'model_measurement_guard.py', BASE / 'qwen_split_trial.py']
    paths += [Path(value) for value in parent['link_command'] if Path(value).is_absolute() and Path(value).is_file()]
    paths += list((ENGINE / 'ggml/src').rglob('*.h')) + list((ENGINE / 'ggml/include').rglob('*.h'))
    inputs = {str(path):sha256(path) for path in paths}
    guard = ModelMeasurementGuard(1219506, {1219506:18095}, inference_snapshot)
    owned = None

    def interrupt(signum, frame):
        raise InterruptedError('Stop only the owned block-quantization build')

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupt)
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guard.assert_idle()
        OUT.mkdir(exist_ok=False)
        private = OUT / 'private-cpu'
        private.mkdir()
        (private / 'repack.cpp').write_text(changed)
        (private / header.name).write_bytes(header.read_bytes())
        (private / 'quantize-blocks.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True), changed.splitlines(True), fromfile=str(source), tofile='private/repack.cpp')))
        private_sources = {str(path):sha256(path) for path in (private / 'repack.cpp', private / header.name)}
        result = dict(started=time.time(), controller_pid=os.getpid(), passed=False, build_completed=False,
                      input_sha256=inputs, private_source_sha256=private_sources, steps=[],
                      parent_sha256=parent['library_sha256'], model_loaded=False,
                      scope='Only repack.cpp.o changes. Complete activation blocks use the original Q8_0/Q8_K quantizers before the existing barrier. Other objects, weights, and selected profiles are unchanged.')

        def save():
            (OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')

        def run(args, label):
            nonlocal owned
            guard.assert_idle()
            with (OUT / (label + '.log')).open('w') as log:
                owned = subprocess.Popen(args, cwd=ENGINE, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                result['owned'] = dict(pid=owned.pid, label=label)
                save()
                deadline = time.monotonic() + 600
                while owned.poll() is None:
                    guard.assert_idle()
                    assert time.monotonic() < deadline, label
                    time.sleep(.25)
            result['steps'].append(dict(label=label, command=args, exit_code=owned.returncode))
            save()
            assert owned.returncode == 0, label
            print(json.dumps(dict(completed=label)), flush=True)

        save()
        try:
            baseline = OUT / 'baseline-repack.cpp.o'
            baseline_command = list(command)
            baseline_command[baseline_command.index('-o') + 1] = str(baseline)
            run(baseline_command, 'baseline-compile')
            result['baseline_object_identical'] = sha256(baseline) == sha256(original_object)
            assert result['baseline_object_identical']
            baseline_link = [str(baseline) if value == original_object else value for value in parent['link_command']]
            baseline_link[baseline_link.index('-o') + 1] = str(OUT / 'parent-link.so')
            run(baseline_link, 'parent-link')
            result['baseline_link_identical'] = sha256(OUT / 'parent-link.so') == parent['library_sha256']
            assert result['baseline_link_identical']
            command[-1] = str(private / 'repack.cpp')
            command[command.index('-o') + 1] = str(private / 'repack.cpp.o')
            run(command, 'private-compile')
            library = private / 'libggml-cpu.so.0.22.0'
            link = [str(private / 'repack.cpp.o') if value == original_object else value for value in parent['link_command']]
            link[link.index('-o') + 1] = str(library)
            run(link, 'private-link')
            (private / 'libggml-cpu.so.0').symlink_to(library.name)
            (private / 'libggml-cpu.so').symlink_to('libggml-cpu.so.0')
            assert all(sha256(path) == value for path, value in {**inputs, **private_sources}.items())
            result.update(passed=True, build_completed=True, library=str(library), library_sha256=sha256(library))
            manifest = dict(input_sha256=inputs, private_source_sha256=private_sources,
                            parent_manifest=str(parent_path), parent_sha256=parent['library_sha256'],
                            compile_commands=[command], link_command=link, library=str(library), library_sha256=sha256(library),
                            baseline_object_identical=True, baseline_link_identical=True, scope=result['scope'])
            (private / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
            print(json.dumps(dict(passed=True, library_sha256=sha256(library))), flush=True)
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
