#!/usr/bin/env python3
"""Privately build and check an opt-in atomic graph barrier for Qwen Q6."""
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
PARENT = BASE / 'results/qwen-q6-q8-batch-0907'
OUT = BASE / 'results/qwen-q6-simple-barrier-0907'
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
        'static struct ggml_state g_state = {0};\n\n#ifdef GGML_USE_OPENMP\n'
        'static bool ggml_cpu_omp_simple_barrier = false;\n#endif', 1)
    start = changed.index('void ggml_barrier(')
    end = changed.index('\nvoid ggml_threadpool_chunk_set', start)
    barrier = changed[start:end]
    assert barrier.count('#ifdef GGML_USE_OPENMP\n    #pragma omp barrier\n#else') == 1
    barrier = barrier.replace('#ifdef GGML_USE_OPENMP\n    #pragma omp barrier\n#else',
        '#ifdef GGML_USE_OPENMP\n    if (!ggml_cpu_omp_simple_barrier) {\n'
        '        #pragma omp barrier\n        return;\n    }\n#endif')
    assert barrier.endswith('#endif\n}\n')
    barrier = barrier[:-len('#endif\n}\n')] + '}\n'
    changed = changed[:start] + barrier + changed[end:]
    marker = '        {\n            const char * env = getenv("GGML_CPU_DISABLE_FUSION");'
    assert changed.count(marker) == 1
    changed = changed.replace(marker,
        '#ifdef GGML_USE_OPENMP\n        {\n'
        '            const char * env = getenv("GGML_CPU_OMP_SIMPLE_BARRIER");\n'
        '            ggml_cpu_omp_simple_barrier = (env != NULL && atoi(env) == 1);\n'
        '        }\n#endif\n' + marker)
    private_source = PRIVATE / 'ggml-cpu.c'
    private_source.write_text(changed)
    (PRIVATE / 'simple-barrier.patch').write_text(''.join(difflib.unified_diff(
        original.splitlines(True), changed.splitlines(True), fromfile=str(source), tofile=str(private_source))))
    commands = json.loads((ENGINE / 'build-goal/compile_commands.json').read_text())
    entry, = [e for e in commands if e['file'] == str(source)]
    command = shlex.split(entry['command'])
    old_object = str((Path(entry['directory']) / command[command.index('-o') + 1]).resolve())
    assert old_object in parent['link_command']
    inputs = dict(parent['input_sha256'])
    for p in (str(source), str(parent_path), parent['library'], old_object):
        if p in inputs:
            assert sha256(p) == inputs[p], p
        inputs[p] = sha256(p)
    assert all(sha256(p) == h for p, h in inputs.items())
    result = dict(passed=False, started=time.time(), steps=[], graph_checks=[], numa_checks=[],
                  parent_cpu_sha256=parent['library_sha256'])
    env = {k: v for k, v in os.environ.items() if not k.startswith(('GGML_', 'LLAMA_', 'OMP_', 'GOMP_', 'REPACK_TEST_'))}
    env.update(LD_LIBRARY_PATH=str(PRIVATE) + ':' + str(PINNED), GOMP_SPINCOUNT='1000')

    def save():
        (OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')

    def run(cmd, label, environment=None, timeout=240):
        guard.assert_idle()
        log = OUT / (label + '.log')
        with log.open('w') as stream:
            proc = subprocess.Popen(cmd, cwd=entry['directory'], env=environment,
                                    stdout=stream, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + timeout
                while proc.poll() is None:
                    guard.assert_idle()
                    assert time.monotonic() < deadline, label
                    time.sleep(0.5)
                assert proc.returncode == 0, (label, proc.returncode)
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=10)
        result['steps'].append(dict(label=label, command=cmd, exit_code=proc.returncode))
        save()
        print(json.dumps({'completed': label}), flush=True)
        return log

    try:
        baseline_obj = PRIVATE / 'baseline.c.o'
        baseline_cmd = list(command)
        baseline_cmd[baseline_cmd.index('-o') + 1] = str(baseline_obj)
        run(baseline_cmd, 'baseline-compile')
        # Relocation-free instruction bytes must match the installed build input.
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
        library = PRIVATE / 'libggml-cpu.so.0.22.0'
        link = [str(obj) if x == old_object else x for x in parent['link_command']]
        link[link.index('-o') + 1] = str(library)
        run(link, 'private-link')
        (PRIVATE / 'libggml-cpu.so.0').symlink_to(library.name)
        (PRIVATE / 'libggml-cpu.so').symlink_to('libggml-cpu.so.0')
        manifest = dict(input_sha256=inputs, library=str(library), library_sha256=sha256(library),
            parent_cpu_sha256=parent['library_sha256'], parent_manifest=str(parent_path),
            private_source_sha256={str(private_source): sha256(private_source)},
            compile_commands=[command], link_command=link,
            scope='Validated Q6 capacity and Q8 batching with opt-in existing atomic graph barrier.')
        (PRIVATE / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        result.update(library=str(library), library_sha256=sha256(library))
        flags = ['/usr/bin/c++', '-O3', '-std=c++17', '-march=native', '-fopenmp']
        flags += ['-I' + str(ENGINE / p) for p in ('include', 'ggml/include', 'ggml/src', 'ggml/src/ggml-cpu')]
        links = ['-L' + str(PRIVATE), '-L' + str(PINNED), '-Wl,-rpath,' + env['LD_LIBRARY_PATH'],
                 '-lggml', '-lggml-cpu', '-lggml-base', '-ldl', '-pthread']
        env.update(GGML_CPU_X16_Q8_0='1', GGML_CPU_X16_Q6_K='1', GGML_CPU_X16_Q8_BATCH='1',
                   REPACK_TEST_DENSE_WORK_SHARING='1', REPACK_TEST_SMALL_BATCHES='1', REPACK_TEST_REPEATS='1')
        for kind in ('q8', 'q6'):
            fixture = (PARENT / 'graph-check.cpp').read_text()
            if kind == 'q6':
                fixture = fixture.replace('if (type == GGML_TYPE_Q5_K) {',
                                           'if (type == GGML_TYPE_Q6_K) {')
                fixture = fixture.replace('quantize_row_q5_K_ref(source, reinterpret_cast<block_q5_K *>(block), QK_K);',
                                           'quantize_row_q6_K_ref(source, reinterpret_cast<block_q6_K *>(block), QK_K);')
                fixture = fixture.replace('q8 ? std::vector<ggml_type>{GGML_TYPE_Q8_0}',
                                           'q8 ? std::vector<ggml_type>{GGML_TYPE_Q6_K}')
                fixture = fixture.replace('{{320, 64}, {640, 160}, {1536, 256}, {2560, 320}, {6144, 64}, {10240, 64}}',
                                           '{{256, 64}, {768, 160}, {1536, 256}, {2560, 320}, {6144, 64}, {10240, 64}}')
            src, binary = OUT / (kind + '-check.cpp'), OUT / (kind + '-check')
            src.write_text(fixture)
            run(flags + [str(src)] + links + ['-o', str(binary)], kind + '-compile')
            for padded in (False, True):
                env.pop('REPACK_TEST_PADDED', None)
                if padded:
                    env['REPACK_TEST_PADDED'] = '1'
                hashes = []
                for enabled in ('0', '1'):
                    env['GGML_CPU_OMP_SIMPLE_BARRIER'] = enabled
                    label = f'{kind}-padded{int(padded)}-barrier{enabled}'
                    log = run(['taskset', '-c', '48-62', str(binary), 'q8'], label, env)
                    hashes.append(re.findall(r'^PASS .* hash=([^\n]+)', log.read_text(), re.M))
                assert len(hashes[0]) == 216 and hashes[0] == hashes[1], (kind, padded)
                result['graph_checks'].append(dict(kind=kind, padded=padded, bit_exact_cases=216))
                save()
        env['GGML_CPU_OMP_SIMPLE_BARRIER'] = '1'
        env['GGML_CPU_NUMA_DEVICES'] = '1'
        for fixture, expected in (('numa-reduce-check', 20), ('numa-thread-limit-check', 40)):
            binary = OUT / fixture
            run(flags + [str(BASE / (fixture + '.cpp'))] + links + ['-o', str(binary)], fixture + '-compile')
            cmd = ['taskset', '-c', '0-127', str(binary)]
            if fixture == 'numa-thread-limit-check':
                cmd += [str(OUT / 'fixture-threads')]
            log = run(cmd, fixture, env).read_text()
            pattern = r' PASS$' if fixture == 'numa-reduce-check' else r'^PASS '
            assert len(re.findall(pattern, log, re.M)) == expected
            result['numa_checks'].append(dict(fixture=fixture, cases=expected))
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
