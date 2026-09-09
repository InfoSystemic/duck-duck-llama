#!/usr/bin/env python3
"""Build the bounded 512-expert x16 fix without replacing installed libraries."""
import importlib.util
import json
from pathlib import Path
import subprocess
import time

BASE = Path(__file__).resolve().parent
ENGINE = BASE.parents[1] / 'engines/llama.cpp-q4e-goal-0904'
OUT = BASE / 'results/qwen-q6-expert-capacity-0907'


def transform(source):
    replacements = {'int active[256]; int n_active = 0;': 'int active[512]; int n_active = 0;',
                    'GGML_ASSERT(n_as <= 256);': 'GGML_ASSERT(n_as <= 512);',
                    'GGML_ASSERT(n_experts <= 256);': 'GGML_ASSERT(n_experts <= 512);'}
    for old, new in replacements.items():
        assert source.count(old) == (2 if 'active[' in old else 1), old
        source = source.replace(old, new)
    return source


def main():
    OUT.mkdir(exist_ok=False)
    spec = importlib.util.spec_from_file_location('private_cpu_builder', BASE / 'build-private-qwen-task-rows.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    builder.transform = transform
    result = dict(started=time.time(), passed=False, steps=[])
    def run(command, cwd, label):
        path = OUT / label.replace('task-rows', 'expert-capacity')
        with path.open('w') as log:
            completed = subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, timeout=300)
        result['steps'].append(dict(command=command, cwd=str(cwd), log=str(path), exit_code=completed.returncode))
        assert completed.returncode == 0, path
        print(json.dumps({'completed': path.name}), flush=True)
    try:
        manifest = builder.build(ENGINE, OUT / 'private-cpu', run)
        old = OUT / 'private-cpu/iq-expert-task-rows.patch'
        old.rename(OUT / 'private-cpu/x16-expert-capacity.patch')
        result.update(passed=True, manifest=str(OUT / 'private-cpu/manifest.json'),
                      library=manifest['library'], library_sha256=manifest['library_sha256'])
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        result['finished'] = time.time()
        (OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
