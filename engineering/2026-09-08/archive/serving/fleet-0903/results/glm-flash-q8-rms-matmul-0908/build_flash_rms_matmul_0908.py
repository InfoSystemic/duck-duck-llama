#!/usr/bin/env python3
"""Build and validate private normalization-to-repacked-Q8 fusion."""
import difflib
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time

from flash_rms_matmul_transform_0908 import transform
from glm_flash_q8_trial import Manager, PORT
from model_measurement_guard import ModelMeasurementGuard
from qwen_split_trial import inference_snapshot, sha256

BASE = Path(__file__).resolve().parent
ENGINE = BASE.parents[1]/'engines/llama.cpp-glm5n-goal-0904'
PINNED = ENGINE/'validated-chunk16-bin'
OUT = BASE/'results/glm-flash-q8-rms-matmul-0908'
PRIVATE = OUT/'private-cpu'


def main():
    os.umask(0o077)
    manager = Manager()
    current = manager.validate_current()
    assert current['pooling'] and current['workers'] == 15 and current['drafts'] == 0
    guard = ModelMeasurementGuard(current['pid'], {current['pid']: PORT}, inference_snapshot)
    guard.assert_idle()
    parent_path = BASE/'results/glm-flash-q8-pool-0908c/private-cpu/manifest.json'
    parent = json.loads(parent_path.read_text())
    repack_path = BASE/'results/glm-flash-q8-clamp-0908/private-cpu/manifest.json'
    repack = json.loads(repack_path.read_text())
    assert parent['library_sha256'] == current['cpu_sha256'] == sha256(parent['library'])
    commands = [list(parent['compile_commands'][0]), list(repack['compile_command'])]
    source_paths = {'ggml-cpu.c': Path(commands[0][-1]), 'repack.cpp': Path(commands[1][-1]),
                    'ops.h': parent_path.parent/'ops.h'}
    originals = {name: path.read_text() for name, path in source_paths.items()}
    changed = transform(originals['ggml-cpu.c'], originals['repack.cpp'], originals['ops.h'])
    inputs = dict(parent['input_sha256'])
    inputs.update(parent['private_source_sha256'])
    inputs.update(repack['private_source_sha256'])
    fixture = BASE/'flash-rms-matmul-check-0908.cpp'
    for p in (parent_path, repack_path, Path(parent['library']), Path(__file__), fixture,
              BASE/'flash_rms_matmul_transform_0908.py', *source_paths.values()):
        if str(p) in inputs: assert inputs[str(p)] == sha256(p)
        inputs[str(p)] = sha256(p)
    for value in parent['link_command']:
        if value.endswith(('.o', '.so.0.22.0', '.a')) and Path(value).exists():
            if value in inputs: assert inputs[value] == sha256(value)
            inputs[value] = sha256(value)
    assert all(sha256(p) == digest for p, digest in inputs.items())
    OUT.mkdir(exist_ok=False)
    PRIVATE.mkdir()
    for name, text in changed.items():
        (PRIVATE/name).write_text(text)
        (PRIVATE/(name+'.patch')).write_text(''.join(difflib.unified_diff(
            originals[name].splitlines(True), text.splitlines(True),
            fromfile=str(source_paths[name]), tofile=str(PRIVATE/name))))
    for p in (Path(__file__), fixture, BASE/'flash_rms_matmul_transform_0908.py'):
        (OUT/p.name).write_bytes(p.read_bytes())
    result = dict(started=time.time(), passed=False, parent_sha256=parent['library_sha256'],
                  input_sha256=inputs, steps=[], checks=[], timings=[])
    cwd = ENGINE/'build-goal/ggml/src'

    def save():
        (OUT/'result.json').write_text(json.dumps(result, indent=2)+'\n')

    def run(command, label, env=None):
        manager.validate_current()
        guard.assert_idle()
        log = OUT/(label+'.log')
        with log.open('w') as stream:
            process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic()+600
                while process.poll() is None:
                    guard.assert_idle()
                    assert time.monotonic() < deadline, label
                    time.sleep(0.5)
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
        baseline_library = PRIVATE/'parent-link.so'
        link[link.index('-o')+1] = str(baseline_library)
        run(link, 'parent-link')
        assert sha256(baseline_library) == parent['library_sha256']
        result['baseline_link_identical'] = True
        compiled = []
        for original_command in commands:
            command = list(original_command)
            source = Path(command[-1])
            old_object = command[command.index('-o')+1]
            assert old_object in link
            baseline = PRIVATE/(source.name+'.baseline.o')
            command[command.index('-o')+1] = str(baseline)
            run(command, source.name+'-baseline-compile')
            sections = []
            for path, label in ((old_object, 'original'), (str(baseline), 'rebuilt')):
                section = PRIVATE/(source.name+'.'+label+'.text')
                sections.append(section)
                run(['objcopy', '--dump-section', '.text='+str(section), path,
                     str(PRIVATE/(source.name+'.'+label+'.copy.o'))], source.name+'-'+label+'-text')
            assert sections[0].read_bytes() == sections[1].read_bytes(), source.name
            command[command.index('-o')+1] = str(PRIVATE/(source.name+'.o'))
            command[-1] = str(PRIVATE/source.name)
            command[1:1] = ['-I'+str(PRIVATE)]
            run(command, source.name+'-private-compile')
            compiled.append(command)
            link[link.index(old_object)] = command[command.index('-o')+1]
        library = PRIVATE/'libggml-cpu.so.0.22.0'
        link[link.index('-o')+1] = str(library)
        run(link, 'private-link')
        (PRIVATE/'libggml-cpu.so.0').symlink_to(library.name)
        (PRIVATE/'libggml-cpu.so').symlink_to('libggml-cpu.so.0')
        manifest = dict(library=str(library), library_sha256=sha256(library), parent_manifest=str(parent_path),
                        parent_sha256=parent['library_sha256'], input_sha256=inputs,
                        private_source_sha256={str(PRIVATE/n): sha256(PRIVATE/n) for n in changed},
                        compile_commands=compiled, link_command=link, baseline_link_identical=True,
                        unpatched_text_identical=True)
        (PRIVATE/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        result.update(library=str(library), library_sha256=sha256(library), unpatched_text_identical=True)
        environment = {k: v for k, v in os.environ.items() if not k.startswith(('GGML_', 'LLAMA_', 'OMP_', 'GOMP_', 'REPACK_TEST_'))}
        environment.update(LD_LIBRARY_PATH=str(PRIVATE)+':'+str(PINNED), GGML_CPU_SINGLE_TASK_MAX_ELEMENTS='4096',
                           GGML_CPU_SOFTMAX_POOL_FUSION='1', GGML_CPU_X16_Q8_BATCH='1')
        binary = OUT/'rms-matmul-check'
        flags = ['/usr/bin/c++', '-O3', '-std=c++17', '-march=native', '-fopenmp']
        flags += ['-I'+str(ENGINE/p) for p in ('include', 'ggml/include', 'ggml/src', 'ggml/src/ggml-cpu')]
        flags += [str(fixture), '-L'+str(PRIVATE), '-L'+str(PINNED), '-Wl,-rpath,'+environment['LD_LIBRARY_PATH'],
                  '-lggml-cpu', '-lggml-base', '-ldl', '-pthread', '-o', str(binary)]
        run(flags, 'fixture-compile')
        outputs = []
        for label, enabled, disabled, is_parent in [('parent','0','0',True), ('off','0','0',False),
                                                    ('on','1','0',False), ('disabled','1','1',False)]:
            env = dict(environment, GGML_CPU_RMS_MATMUL_FUSION=enabled, GGML_CPU_DISABLE_FUSION=disabled,
                       GGML_CPU_RMS_MATMUL_AUDIT='1')
            if is_parent: env['LD_LIBRARY_PATH'] = str(Path(parent['library']).parent)+':'+str(PINNED)
            output = OUT/(label+'.bin')
            log = run(['taskset','-c','48-62',str(binary),str(output)], 'check-'+label, env)
            assert 'SUMMARY cases=86 failures=0' in log
            outputs.append(output)
            result['checks'].append(dict(label=label, cases=86, samples_per_case=3,
                output_bytes=output.stat().st_size, output_sha256=sha256(output),
                fused_computations=sum(int(x) for x in re.findall(r' fused=(\d+)',log)),
                max_scaled_error=max(float(x) for x in re.findall(r' max_error=([^ ]+)',log))))
            save()
        assert all(p.read_bytes() == outputs[0].read_bytes() for p in outputs[1:])
        assert next(x for x in result['checks'] if x['label']=='on')['fused_computations'] > 0
        result['bit_exact'] = True
        for index, enabled in enumerate(('0','1','1','0')):
            env = dict(environment, GGML_CPU_RMS_MATMUL_FUSION=enabled)
            log = run(['taskset','-c','48-62',str(binary),'--timing'], f'timing-{index}-{enabled}', env)
            rows = re.findall(r'^PASS .* tokens=(\d+) .* weighted=(\d+) .* ms=([\d.]+)$',log,re.M)
            assert len(rows) == 4
            result['timings'].append(dict(enabled=enabled=='1', graph_ms={f'{w}-{t}':float(ms) for t,w,ms in rows}))
        assert all(sha256(p) == digest for p, digest in inputs.items())
        manager.validate_current()
        guard.assert_idle()
        result['passed'] = True
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        result['finished'] = time.time()
        save()


if __name__ == '__main__':
    with (BASE/'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
