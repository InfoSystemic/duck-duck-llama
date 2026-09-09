#!/usr/bin/env python3
"""Privately test single-worker scheduling for the traced small FP32 matrices."""
import difflib
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

from model_measurement_guard import ModelMeasurementGuard
from qwen_split_trial import inference_snapshot, sha256

BASE = Path(__file__).resolve().parent
ENGINE = BASE.parents[1] / 'engines/llama.cpp-q4e-goal-0904'
PINNED = ENGINE / 'validated-iq-batch3-bin'
PARENT = BASE / 'results/qwen-q6-q8-wide-batch-0907'
OUT = BASE / 'results/qwen-q6-small-f32-0908b'
PRIVATE = OUT / 'private-cpu'


def main():
    OUT.mkdir(exist_ok=False)
    PRIVATE.mkdir()
    parent_path = PARENT / 'private-cpu/manifest.json'
    parent = json.loads(parent_path.read_text())
    assert sha256(parent['library']) == parent['library_sha256']
    state = json.loads((BASE / 'results/qwen-q6-trial-0907/state.json').read_text())
    pid = state['current']['pid']
    guard = ModelMeasurementGuard(pid, {pid: 18095}, inference_snapshot)
    guard.assert_idle()
    source = ENGINE / 'ggml/src/ggml-cpu/ggml-cpu.c'
    original = source.read_text()
    changed = original.replace('static struct ggml_state g_state = {0};',
        'static struct ggml_state g_state = {0};\nstatic bool ggml_cpu_single_small_f32_mm = false;', 1)
    marker = '    const int64_t max_el = ggml_cpu_single_task_max_elements();'
    assert changed.count(marker) == 1
    changed = changed.replace(marker, '''    if (ggml_cpu_single_small_f32_mm && node->op == GGML_OP_MUL_MAT) {
        const struct ggml_tensor * a = node->src[0];
        const struct ggml_tensor * b = node->src[1];
        if (a->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 &&
                a->ne[0] >= 4096 && a->ne[0] <= 16384 && a->ne[1] <= 4 && b->ne[1] <= 8 &&
                a->ne[2] == 1 && a->ne[3] == 1 && b->ne[2] == 1 && b->ne[3] == 1 &&
                a->ne[0] * a->ne[1] * b->ne[1] <= 262144 &&
                ggml_is_contiguous(a) && ggml_is_contiguous(b)) {
            return true;
        }
    }
''' + marker)
    marker = '        {\n            const char * env = getenv("GGML_CPU_DISABLE_FUSION");'
    assert changed.count(marker) == 1
    changed = changed.replace(marker, '''        {
            const char * env = getenv("GGML_CPU_SINGLE_SMALL_F32_MM");
            ggml_cpu_single_small_f32_mm = (env != NULL && atoi(env) == 1);
        }
''' + marker)
    start = changed.index('void ggml_compute_forward_mul_mat(')
    end = changed.index('\nstatic void ', start + 1)
    matmul = changed[start:end]
    assert matmul.count('    ggml_barrier(params->threadpool);') == 1
    matmul = matmul.replace('    ggml_barrier(params->threadpool);',
                           '    if (nth > 1) {\n        ggml_barrier(params->threadpool);\n    }')
    changed = changed[:start] + matmul + changed[end:]
    private_source = PRIVATE / 'ggml-cpu.c'
    private_source.write_text(changed)
    (PRIVATE / 'small-f32.patch').write_text(''.join(difflib.unified_diff(
        original.splitlines(True), changed.splitlines(True), fromfile=str(source), tofile=str(private_source))))
    commands = json.loads((ENGINE / 'build-goal/compile_commands.json').read_text())
    entry, = [e for e in commands if e['file'] == str(source)]
    command = shlex.split(entry['command'])
    old_object = str((Path(entry['directory']) / command[command.index('-o') + 1]).resolve())
    assert old_object in parent['link_command']
    sgemm_source = ENGINE / 'ggml/src/ggml-cpu/llamafile/sgemm.cpp'
    sgemm_original = sgemm_source.read_text()
    start = sgemm_original.index('class tinyBLAS {')
    end = sgemm_original.index('\n};', start) + 3
    sgemm_class = sgemm_original[start:end]
    assert sgemm_class.count('        ggml_barrier(params->threadpool);') == 2
    sgemm_class = sgemm_class.replace('        ggml_barrier(params->threadpool);',
        '        if (params->nth > 1) {\n            ggml_barrier(params->threadpool);\n        }')
    sgemm_changed = sgemm_original[:start] + sgemm_class + sgemm_original[end:]
    private_sgemm = PRIVATE / 'sgemm.cpp'
    private_sgemm.write_text(sgemm_changed)
    (PRIVATE / 'sgemm-single-worker.patch').write_text(''.join(difflib.unified_diff(
        sgemm_original.splitlines(True), sgemm_changed.splitlines(True), fromfile=str(sgemm_source), tofile=str(private_sgemm))))
    sgemm_entry, = [e for e in commands if e['file'] == str(sgemm_source)]
    assert sgemm_entry['directory'] == entry['directory']
    sgemm_command = shlex.split(sgemm_entry['command'])
    old_sgemm_object = str((Path(entry['directory']) / sgemm_command[sgemm_command.index('-o') + 1]).resolve())
    assert old_sgemm_object in parent['link_command']
    inputs = dict(parent['input_sha256'])
    fixture = BASE / 'qwen-small-f32-check.cpp'
    for p in (source, sgemm_source, parent_path, Path(parent['library']), Path(old_object), Path(old_sgemm_object), fixture, Path(__file__)):
        if str(p) in inputs:
            assert sha256(p) == inputs[str(p)], str(p)
        inputs[str(p)] = sha256(p)
    assert all(sha256(p) == h for p, h in inputs.items())
    (OUT / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    (OUT / fixture.name).write_bytes(fixture.read_bytes())
    result = dict(passed=False, started=time.time(), steps=[], graph_checks=[], timings=[],
                  parent_cpu_sha256=parent['library_sha256'])
    environment = {k: v for k, v in os.environ.items() if not k.startswith(('GGML_', 'LLAMA_', 'OMP_', 'GOMP_', 'REPACK_TEST_'))}
    environment.update(LD_LIBRARY_PATH=str(PRIVATE) + ':' + str(PINNED),
                       GGML_CPU_SINGLE_TASK_MAX_ELEMENTS='4096')

    def save():
        (OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')

    def run(cmd, label, env=None):
        guard.assert_idle()
        log = OUT / (label + '.log')
        with log.open('w') as stream:
            proc = subprocess.Popen(cmd, cwd=entry['directory'], env=env,
                                    stdout=stream, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 240
                while proc.poll() is None:
                    guard.assert_idle()
                    assert time.monotonic() < deadline, label
                    time.sleep(0.5)
                assert proc.returncode == 0, (label, proc.returncode)
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=10)
        result['steps'].append(dict(label=label, command=cmd))
        save()
        print(json.dumps({'completed': label}), flush=True)
        return log.read_text()

    save()
    try:
        baseline_obj = PRIVATE / 'baseline.c.o'
        baseline_cmd = list(command)
        baseline_cmd[baseline_cmd.index('-o') + 1] = str(baseline_obj)
        run(baseline_cmd, 'baseline-compile')
        for obj, name in ((old_object, 'original'), (baseline_obj, 'rebuilt')):
            run(['objcopy', '--dump-section', '.text=' + str(PRIVATE / (name + '.text')),
                 str(obj), str(PRIVATE / (name + '.copy.o'))], name + '-text')
        assert (PRIVATE / 'original.text').read_bytes() == (PRIVATE / 'rebuilt.text').read_bytes()
        result['unpatched_text_identical'] = True
        obj = PRIVATE / 'ggml-cpu.c.o'
        command[command.index('-o') + 1] = str(obj)
        command[command.index(str(source))] = str(private_source)
        command.insert(1, '-I' + str(source.parent))
        run(command, 'private-compile')
        sgemm_object = PRIVATE / 'sgemm.cpp.o'
        sgemm_command[sgemm_command.index('-o') + 1] = str(sgemm_object)
        sgemm_command[sgemm_command.index(str(sgemm_source))] = str(private_sgemm)
        sgemm_command.insert(1, '-I' + str(sgemm_source.parent))
        run(sgemm_command, 'private-sgemm-compile')
        library = PRIVATE / 'libggml-cpu.so.0.22.0'
        link = [str(obj) if x == old_object else str(sgemm_object) if x == old_sgemm_object else x for x in parent['link_command']]
        link[link.index('-o') + 1] = str(library)
        run(link, 'private-link')
        (PRIVATE / 'libggml-cpu.so.0').symlink_to(library.name)
        (PRIVATE / 'libggml-cpu.so').symlink_to('libggml-cpu.so.0')
        manifest = dict(input_sha256=inputs, library=str(library), library_sha256=sha256(library),
            parent_cpu_sha256=parent['library_sha256'], parent_manifest=str(parent_path),
            private_source_sha256={str(p): sha256(p) for p in (private_source, private_sgemm)},
            compile_commands=[command, sgemm_command], link_command=link,
            scope='Q6 and wide Q8 CPU with opt-in single-worker small FP32 matrix scheduling.')
        (PRIVATE / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        result.update(library=str(library), library_sha256=sha256(library))
        binary = OUT / 'small-f32-check'
        flags = ['/usr/bin/c++', '-O3', '-std=c++17', '-march=native', '-fopenmp']
        flags += ['-I' + str(ENGINE / p) for p in ('include', 'ggml/include', 'ggml/src', 'ggml/src/ggml-cpu')]
        links = ['-L' + str(PRIVATE), '-L' + str(PINNED), '-Wl,-rpath,' + environment['LD_LIBRARY_PATH'],
                 '-lggml', '-lggml-cpu', '-lggml-base', '-ldl', '-pthread']
        run(flags + [str(fixture)] + links + ['-o', str(binary)], 'fixture-compile')
        hashes = []
        for enabled in ('0', '1'):
            environment['GGML_CPU_SINGLE_SMALL_F32_MM'] = enabled
            log = run(['taskset', '-c', '48-62', str(binary)], 'check-' + enabled, environment)
            values = re.findall(r'^PASS .* hash=([0-9a-f]+) ', log, re.M)
            assert len(values) == 44
            hashes.append(values)
        assert hashes[0] == hashes[1], 'FP32 outputs changed with the scheduling option'
        result['graph_checks'] = dict(bit_exact_cases=44, canonical_reference_passed=True)
        for index, enabled in enumerate(('0', '1', '1', '0')):
            environment['GGML_CPU_SINGLE_SMALL_F32_MM'] = enabled
            log = run(['taskset', '-c', '48-62', str(binary), '--timing'], f'timing-{index}-{enabled}', environment)
            samples = re.findall(r'^PASS .* tokens=(\d+).* ms=([\d.]+)$', log, re.M)
            assert len(samples) == 2
            result['timings'].append(dict(enabled=enabled == '1', graph_ms={n: float(ms) for n, ms in samples}))
        assert all(sha256(p) == h for p, h in inputs.items()), 'An original input changed'
        result['passed'] = True
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
