#!/usr/bin/env python3
"""Build a private Qwen barrier candidate without loading any model."""
import argparse
import difflib
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import traceback

from flash_dissemination_transform_0908 import fixture_audit
from qwen_dissemination_transform_0908 import transform
from qwen_split_trial import sha256

BASE = Path(__file__).resolve().parent
ENGINE = BASE.parents[1] / 'engines/llama.cpp-q4e-goal-0904'
PINNED = ENGINE / 'validated-iq-batch3-bin'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('label')
    args = parser.parse_args()
    assert args.label.startswith('qwen-dissemination-') and Path(args.label).name == args.label
    assert os.sched_getaffinity(0) == {127}, 'Compile on the reserved controller CPU'
    os.umask(0o077)
    out = BASE / 'results' / args.label
    private = out / 'private-cpu'
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        parent_path = BASE / 'results/qwen-q6-q8-wide-batch-0907/private-cpu/manifest.json'
        parent = json.loads(parent_path.read_text())
        assert parent['library_sha256'] == sha256(parent['library']) == 'c79e19e38acfd6ada3f0637c136cf9cb10253127778cb214528b2f966fb6dfcf'
        source = ENGINE / 'ggml/src/ggml-cpu/ggml-cpu.c'
        entries = json.loads((ENGINE / 'build-goal/compile_commands.json').read_text())
        entry, = [row for row in entries if row['file'] == str(source)]
        command = shlex.split(entry['command'])
        original_object = str((Path(entry['directory']) / command[command.index('-o') + 1]).resolve())
        assert original_object in parent['link_command']
        if str(source) in parent['input_sha256']:
            assert sha256(source) == parent['input_sha256'][str(source)]
        probe_path = BASE / 'results/glm-flash-local-barrier-probe-0908b/result.json'
        probe = json.loads(probe_path.read_text())
        header = BASE / 'flash-local-barrier-0908.h'
        assert probe['passed'] and sha256(header) == probe['input_sha256'][str(header)]
        out.mkdir(exist_ok=False)
        private.mkdir()
        sources = dict(q8=BASE / 'results/qwen-q6-simple-barrier-0907/q8-check.cpp',
                       q6=BASE / 'results/qwen-q6-simple-barrier-0907/q6-check.cpp',
                       threads=BASE / 'numa-thread-limit-check.cpp', reduce=BASE / 'numa-reduce-check.cpp')
        paths = [Path(__file__).resolve(), BASE / 'qwen_dissemination_transform_0908.py',
                 BASE / 'flash_dissemination_transform_0908.py', source, header, parent_path, probe_path,
                 Path(parent['library']), *sources.values()]
        inputs = {str(path): sha256(path) for path in paths}
        for value in parent['link_command']:
            if value.endswith(('.o', '.a', '.so.0.22.0', '/libgomp.so')) and Path(value).is_file():
                inputs[value] = sha256(value)
        for directory in ('ggml/src', 'ggml/include'):
            for path in (ENGINE / directory).rglob('*.h'):
                inputs[str(path)] = sha256(path)
        original = source.read_text()
        changed = transform(original)
        (private / 'ggml-cpu.c').write_text(changed)
        (private / header.name).write_bytes(header.read_bytes())
        (private / 'cpu.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True), changed.splitlines(True))))
        result = dict(started=time.time(), build_completed=False, validation_completed=False,
                      parent_sha256=parent['library_sha256'], input_sha256=inputs,
                      steps=[], scope='Compilation only; no component or model execution', model_loaded=False)
        def save():
            (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        def run(cmd, label):
            with (out / (label + '.log')).open('w') as log:
                process = subprocess.run(cmd, cwd=entry['directory'], stdout=log, stderr=subprocess.STDOUT, timeout=600)
            result['steps'].append(dict(label=label, command=cmd, exit_code=process.returncode))
            save()
            assert process.returncode == 0, label
            print(json.dumps(dict(completed=label)), flush=True)
        save()
        try:
            link = list(parent['link_command'])
            link[link.index('-o') + 1] = str(private / 'parent-link.so')
            run(link, 'parent-link')
            assert sha256(private / 'parent-link.so') == parent['library_sha256']
            result['baseline_link_identical'] = True
            baseline = list(command)
            baseline[baseline.index('-o') + 1] = str(private / 'baseline.o')
            run(baseline, 'baseline-compile')
            for name, obj in [('original', original_object), ('rebuilt', str(private / 'baseline.o'))]:
                run(['objcopy', '--dump-section', '.text=' + str(private / (name + '.text')),
                     obj, str(private / (name + '.copy.o'))], name + '-text')
            assert (private / 'original.text').read_bytes() == (private / 'rebuilt.text').read_bytes()
            result['unpatched_text_identical'] = True
            command[command.index('-o') + 1] = str(private / 'ggml-cpu.c.o')
            command[command.index(str(source))] = str(private / 'ggml-cpu.c')
            command.insert(1, '-I' + str(source.parent))
            run(command, 'private-compile')
            library = private / 'libggml-cpu.so.0.22.0'
            link = [str(private / 'ggml-cpu.c.o') if item == original_object else item for item in parent['link_command']]
            link[link.index('-o') + 1] = str(library)
            run(link, 'private-link')
            (private / 'libggml-cpu.so.0').symlink_to(library.name)
            (private / 'libggml-cpu.so').symlink_to('libggml-cpu.so.0')
            flags = ['/usr/bin/c++', '-O3', '-std=c++17', '-march=native', '-fopenmp']
            flags += ['-I' + str(ENGINE / p) for p in ('include', 'ggml/include', 'ggml/src', 'ggml/src/ggml-cpu')]
            links = ['-L' + str(private), '-L' + str(PINNED), '-Wl,-rpath,' + str(private) + ':' + str(PINNED),
                     '-lggml', '-lggml-cpu', '-lggml-base', '-ldl', '-pthread']
            fixture_sources = {}
            for name, path in sources.items():
                fixture = path.read_text()
                if name == 'threads':
                    marker = '            failures += !ok;'
                    assert fixture.count(marker) == 1
                    fixture = fixture.replace(marker, '''            uint64_t hash = 14695981039346656037ULL;
            const auto bytes = reinterpret_cast<const unsigned char *>(data);
            for (size_t i = 0; i < ggml_nbytes(product); ++i) hash = (hash ^ bytes[i]) * 1099511628211ULL;
            std::printf("THREAD_HASH node=%d setting=%s hash=%016llx\\n",node,setting.c_str(),(unsigned long long)hash);
''' + marker)
                target = out / (name + '-check.cpp')
                target.write_text(fixture_audit(fixture))
                fixture_sources[str(target)] = sha256(target)
                run(flags + [str(target)] + links + ['-o', str(out / (name + '-check'))], name + '-compile')
            assert all(sha256(path) == digest for path, digest in inputs.items())
            manifest = dict(input_sha256=inputs, parent_sha256=parent['library_sha256'],
                            parent_manifest=str(parent_path), library=str(library), library_sha256=sha256(library),
                            compile_commands=[command], link_command=link,
                            private_source_sha256={str(private / 'ggml-cpu.c'):sha256(private / 'ggml-cpu.c'),
                                                   str(private / header.name):sha256(private / header.name), **fixture_sources},
                            scope='Qwen Q6 wide-Q8 parent with an opt-in dissemination graph barrier; no arithmetic changes')
            (private / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
            result.update(build_completed=True, library=str(library), library_sha256=sha256(library),
                          fixture_sha256=fixture_sources,
                          binary_sha256={name:sha256(out / (name + '-check')) for name in sources})
        except BaseException:
            result['error'] = traceback.format_exc()
            raise
        finally:
            result['finished'] = time.time()
            save()


if __name__ == '__main__':
    main()
