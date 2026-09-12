#!/usr/bin/env python3
"""Build an isolated Full split library from the existing host object recipe."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
ENGINE = HERE.parents[2] / 'engines/llama.cpp-sr950-glm'
BUILD = ENGINE / 'build-dev2'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    original = ENGINE / 'src/llama-model.cpp'
    candidate = HERE / 'llama-model.patched.cpp'
    if original.read_bytes() != (HERE / 'llama-model.parent.cpp').read_bytes():
        raise RuntimeError('Engine source no longer matches the reviewed split parent')
    output = HERE / 'build'
    baseline, tuned = output / 'baseline', output / 'candidate'
    baseline.mkdir(parents=True, exist_ok=True)
    tuned.mkdir(parents=True, exist_ok=True)
    commands = json.loads((BUILD / 'compile_commands.json').read_text())
    compile_entry, = [row for row in commands if Path(row['file']) == original]
    cwd = Path(compile_entry['directory'])
    original_object = cwd / 'CMakeFiles/llama.dir/llama-model.cpp.o'
    compile_args = shlex.split(compile_entry['command'])
    compile_args = [arg.replace('\\"', '"') for arg in compile_args]
    compile_args[compile_args.index('-o') + 1] = str(tuned / 'llama-model.cpp.o')
    compile_args[compile_args.index('-c') + 1] = str(candidate)
    link_args = shlex.split((BUILD / 'src/CMakeFiles/llama.dir/link.txt').read_text())
    inputs = [cwd / arg for arg in link_args
              if not arg.startswith('-') and (arg.endswith('.o') or '.so.' in arg)]
    # The linker output also ends in .so.VERSION; exclude that non-input.
    old_output = link_args[link_args.index('-o') + 1]
    inputs = [path.resolve() for path in inputs if path != cwd / old_output]
    source_inputs = [original, candidate, original_object,
                     BUILD / 'compile_commands.json', BUILD / 'src/CMakeFiles/llama.dir/link.txt']
    before = {str(path): digest(path) for path in set(inputs + source_inputs)}

    baseline_args = list(link_args)
    baseline_lib = baseline / 'libllama.so.0.1.2'
    baseline_args[baseline_args.index('-o') + 1] = str(baseline_lib)
    subprocess.run(baseline_args, cwd=cwd, check=True)
    selected_lib = (BUILD / 'bin/libllama.so.0.1.2').resolve()
    if digest(baseline_lib) != digest(selected_lib):
        raise RuntimeError('Baseline relink differs from the selected runtime library')

    subprocess.run(compile_args, cwd=cwd, check=True)
    candidate_args = list(link_args)
    candidate_lib = tuned / 'libllama.so.0.1.2'
    candidate_args[candidate_args.index('-o') + 1] = str(candidate_lib)
    matches = [i for i, arg in enumerate(candidate_args)
               if arg.endswith('.o') and (cwd / arg).resolve() == original_object]
    if len(matches) != 1:
        raise RuntimeError('Expected exactly one original split object in link recipe')
    candidate_args[matches[0]] = str(tuned / 'llama-model.cpp.o')
    subprocess.run(candidate_args, cwd=cwd, check=True)
    for directory in (baseline, tuned):
        for name, target in [('libllama.so.0', 'libllama.so.0.1.2'), ('libllama.so', 'libllama.so.0')]:
            path = directory / name
            if path.is_symlink() and os.readlink(path) == target:
                continue
            path.symlink_to(target)
    env = {'PATH': '/usr/bin:/bin', 'LD_LIBRARY_PATH': f'{tuned}:{BUILD / "bin"}'}
    executable = BUILD / 'bin/llama-server'
    ldd = subprocess.check_output(['ldd', str(executable)], env=env, text=True)
    if str(tuned / 'libllama.so.0') not in ldd or 'not found' in ldd:
        raise RuntimeError('Candidate library was not selected by the loader')
    version = subprocess.run([str(executable), '--version'], env=env, text=True,
                             capture_output=True, check=True)
    after = {path: digest(Path(path)) for path in before}
    if after != before:
        raise RuntimeError('Original source, objects or link inputs changed during the build')
    report = {
        'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'passed': True, 'baseline_relink_byte_identical': True,
        'source_and_link_inputs_unchanged': True, 'model_loaded': False,
        'source_and_link_inputs_sha256': before,
        'baseline_library_sha256': digest(baseline_lib),
        'candidate_library_sha256': digest(candidate_lib),
        'candidate_object_sha256': digest(tuned / 'llama-model.cpp.o'),
        'compile_command': compile_args, 'link_command': candidate_args,
        'library_path': env['LD_LIBRARY_PATH'], 'ldd': ldd,
        'version': version.stdout + version.stderr,
        'scope': 'Complete candidate library linked from the byte-identical baseline recipe; loader and executable startup checked. Unequal-split full-model inference remains unvalidated.',
    }
    (HERE / 'build-verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ['passed', 'baseline_relink_byte_identical',
                                           'source_and_link_inputs_unchanged', 'candidate_library_sha256',
                                           'model_loaded']}, indent=2))


if __name__ == '__main__':
    main()
