#!/usr/bin/env python3
"""Build and check a private narrow Q8 projection implementation."""
import difflib
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time

from flash_q8_r8_ordered_k_transform_0908 import transform_source, transform_fixture
from glm_flash_q8_trial import BASE, Manager, PORT
from model_measurement_guard import ModelMeasurementGuard
from qwen_split_trial import inference_snapshot, sha256

OUT = BASE / 'results/glm-flash-q8-r8-ordered-k-0908'
PRIVATE = OUT / 'private-cpu'
ENGINE = BASE.parents[1] / 'engines/llama.cpp-glm5n-goal-0904'


def main():
    os.umask(0o077)
    manager = Manager()
    current = manager.validate_current()
    guard = ModelMeasurementGuard(current['pid'], {current['pid']: PORT}, inference_snapshot)
    guard.assert_idle()
    assert current['rms_guard'] and current['drafts'] == 0
    parent_path = BASE / 'results/glm-flash-rms-guard-0908/private-cpu/manifest.json'
    source_manifest_path = BASE / 'results/glm-flash-q8-clamp-0908/private-cpu/manifest.json'
    probe_path = BASE / 'results/glm-flash-q8-r8-ordered-k-probe-0908b/result.json'
    parent = json.loads(parent_path.read_text())
    source_manifest = json.loads(source_manifest_path.read_text())
    probe = json.loads(probe_path.read_text())
    assert probe['passed'] and all(sha256(p) == h for p, h in probe['input_sha256'].items())
    assert sha256(parent['library']) == parent['library_sha256'] == current['cpu_sha256'] == probe['cpu_sha256']
    original_command = list(source_manifest['compile_command'])
    source_path = Path(original_command[-1])
    old_object = original_command[original_command.index('-o') + 1]
    assert old_object in parent['link_command']
    original = source_path.read_text()
    changed = transform_source(original)
    fixture_path = BASE / 'flash-rms-matmul-check-0908.cpp'
    fixture = transform_fixture(fixture_path.read_text())
    header = BASE / 'flash-q8-r8-ordered-k-0908.h'
    sources = [Path(__file__), BASE / 'flash_q8_r8_ordered_k_transform_0908.py', parent_path,
               source_manifest_path, probe_path, source_path, fixture_path, header, Path(parent['library'])]
    sources += [Path(value) for value in parent['link_command']
                if value.endswith(('.o', '.a', '.so.0.22.0')) and Path(value).exists()]
    for directory in ('ggml/src', 'ggml/include'):
        sources += list((ENGINE / directory).rglob('*.h'))
    inputs = {str(path): sha256(path) for path in sources}
    OUT.mkdir(exist_ok=False)
    PRIVATE.mkdir()
    (PRIVATE / 'repack.cpp').write_text(changed)
    (PRIVATE / header.name).write_bytes(header.read_bytes())
    (PRIVATE / 'repack.cpp.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True), changed.splitlines(True))))
    fixture_out = OUT / 'ordered-k-graph-check.cpp'
    fixture_out.write_text(fixture)
    inputs[str(fixture_out)] = sha256(fixture_out)
    for path in [Path(__file__), BASE / 'flash_q8_r8_ordered_k_transform_0908.py', BASE / 'glm_flash_q8_trial.py']:
        (OUT / path.name).write_bytes(path.read_bytes())
    result = dict(started=time.time(), passed=False, parent_sha256=parent['library_sha256'],
                  input_sha256=inputs, steps=[], checks=[], timings=[])
    def save():
        (OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    def run(command, label, env=None):
        manager.validate_current()
        guard.assert_idle()
        log = OUT / (label + '.log')
        with log.open('w') as stream:
            process = subprocess.Popen(command, cwd=ENGINE / 'build-goal/ggml/src', env=env,
                                       stdout=stream, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 600
                while process.poll() is None:
                    guard.assert_idle()
                    assert time.monotonic() < deadline, label
                    time.sleep(.5)
                assert process.returncode == 0, (label, process.returncode)
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=10)
        result['steps'].append(dict(label=label, command=command))
        save()
        print(json.dumps(dict(completed=label)), flush=True)
        return log.read_text()
    save()
    try:
        link = list(parent['link_command'])
        link[link.index('-o') + 1] = str(PRIVATE / 'parent-link.so')
        run(link, 'parent-link')
        assert sha256(PRIVATE / 'parent-link.so') == parent['library_sha256']
        command = list(original_command)
        command[command.index('-o') + 1] = str(PRIVATE / 'baseline.o')
        run(command, 'baseline-compile')
        for label, path in [('original', old_object), ('rebuilt', str(PRIVATE / 'baseline.o'))]:
            run(['objcopy', '--dump-section', '.text=' + str(PRIVATE / (label + '.text')), path,
                 str(PRIVATE / (label + '.copy.o'))], label + '-text')
        assert (PRIVATE / 'original.text').read_bytes() == (PRIVATE / 'rebuilt.text').read_bytes()
        command[-1] = str(PRIVATE / 'repack.cpp')
        command[command.index('-o') + 1] = str(PRIVATE / 'repack.cpp.o')
        command[1:1] = ['-I' + str(PRIVATE)]
        run(command, 'private-compile')
        library = PRIVATE / 'libggml-cpu.so.0.22.0'
        link[link.index(old_object)] = command[command.index('-o') + 1]
        link[link.index('-o') + 1] = str(library)
        run(link, 'private-link')
        (PRIVATE / 'libggml-cpu.so.0').symlink_to(library.name)
        (PRIVATE / 'libggml-cpu.so').symlink_to('libggml-cpu.so.0')
        manifest = dict(library=str(library), library_sha256=sha256(library), parent_manifest=str(parent_path),
                        parent_sha256=parent['library_sha256'], compile_command=command, link_command=link,
                        input_sha256=inputs, private_source_sha256={str(p): sha256(p) for p in [PRIVATE / 'repack.cpp', PRIVATE / header.name]},
                        baseline_link_identical=True, unpatched_text_identical=True)
        (PRIVATE / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        result.update(library=str(library), library_sha256=sha256(library), baseline_link_identical=True, unpatched_text_identical=True)
        pinned = Path(current['pinned_directory'])
        binary = OUT / 'ordered-k-graph-check'
        flags = ['c++', '-O3', '-std=c++17', '-march=native', '-fopenmp']
        flags += ['-I' + str(ENGINE / p) for p in ('include', 'ggml/include', 'ggml/src', 'ggml/src/ggml-cpu')]
        flags += [str(fixture_out), '-L' + str(PRIVATE), '-L' + str(pinned),
                  '-Wl,-rpath,' + str(PRIVATE) + ':' + str(pinned), '-lggml-cpu', '-lggml-base', '-ldl', '-pthread', '-o', str(binary)]
        run(flags, 'fixture-compile')
        result['binary_sha256'] = sha256(binary)
        env = {k: v for k, v in os.environ.items() if not k.startswith(('GGML_', 'LLAMA_', 'OMP_', 'GOMP_', 'REPACK_TEST_'))}
        env.update(LD_LIBRARY_PATH=str(PRIVATE) + ':' + str(pinned), GGML_CPU_SINGLE_TASK_MAX_ELEMENTS='4096',
                   GGML_CPU_SOFTMAX_POOL_FUSION='1', GGML_CPU_X16_Q8_BATCH='1', GGML_CPU_Q8_FAST_SUM='1', GGML_CPU_RMS_F64_SIMD='1')
        outputs = []
        for label, enabled, disabled, is_parent in [('parent', '0', '0', True), ('off', '0', '0', False),
                                                    ('on', '1', '0', False), ('no-fusion', '1', '1', False)]:
            trial = dict(env, GGML_CPU_Q8_R8_ORDERED_K=enabled, GGML_CPU_Q8_R8_ORDERED_K_AUDIT='1', GGML_CPU_DISABLE_FUSION=disabled)
            expected_cpu = Path(parent['library']) if is_parent else library
            if is_parent:
                trial['LD_LIBRARY_PATH'] = str(expected_cpu.parent) + ':' + str(pinned)
            output = OUT / (label + '.bin')
            log = run(['taskset', '-c', '48-62', str(binary), str(output)], 'check-' + label, trial)
            loaded, = re.findall(r'^CPU_LIBRARY (.+)$', log, re.M)
            assert Path(loaded).resolve() == expected_cpu.resolve()
            assert 'SUMMARY cases=94 failures=0' in log
            counts = re.findall(r'^PASS .* fused=(\d+) ', log, re.M)
            assert len(counts) == 94
            selected_calls = sum(map(int, counts))
            assert selected_calls > 0 if enabled == '1' else selected_calls == 0
            outputs.append(output)
            result['checks'].append(dict(label=label, cases=94, samples_per_case=3, selected_calls=selected_calls,
                                         output_bytes=output.stat().st_size, output_sha256=sha256(output), cpu_library=loaded))
            save()
        assert all(path.read_bytes() == outputs[0].read_bytes() for path in outputs[1:])
        result['bit_exact'] = True
        for index, enabled in enumerate(('0', '1', '1', '0')):
            log = run(['taskset', '-c', '48-62', str(binary), '--timing'], f'timing-{index}-{enabled}',
                      dict(env, GGML_CPU_Q8_R8_ORDERED_K=enabled))
            rows = re.findall(r'^PASS .* tokens=(\d+) .* weighted=(\d+) .* ms=([\d.]+)$', log, re.M)
            assert len(rows) == 4
            result['timings'].append(dict(enabled=enabled == '1', graph_ms={f'{w}-{t}': float(ms) for t, w, ms in rows}))
        assert all(sha256(path) == digest for path, digest in inputs.items())
        manager.validate_current()
        guard.assert_idle()
        result['passed'] = True
        print(json.dumps(dict(passed=True, library_sha256=sha256(library), checks=result['checks'], timings=result['timings'])), flush=True)
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        result['finished'] = time.time()
        save()


if __name__ == '__main__':
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
