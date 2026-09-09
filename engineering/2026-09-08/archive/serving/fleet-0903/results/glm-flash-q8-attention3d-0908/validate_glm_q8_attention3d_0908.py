#!/usr/bin/env python3
"""Compare pinned and privately batched Q8 attention across head planes."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

BASE = Path(__file__).resolve().parent
ENGINE = BASE.parents[1] / 'engines/llama.cpp-glm5n-goal-0904'
PINNED = ENGINE / 'validated-chunk16-bin'
BATCH = BASE / 'results/glm-flash-q8-batch-0908'
OUT = BASE / 'results/glm-flash-q8-attention3d-0908'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    os.umask(0o077)
    OUT.mkdir()
    result = dict(started=time.time(), passed=False, checks=[])

    def save():
        (OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')

    def run(command, label, env=None):
        with (OUT / (label + '.log')).open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, check=True, timeout=300)
        return (OUT / (label + '.log')).read_text()

    try:
        batch = json.loads((BATCH / 'result.json').read_text())
        assert batch['passed'] and sha(batch['library']) == batch['library_sha256']
        reference = PINNED / 'libggml-cpu.so.0.22.0'
        assert sha(reference) == batch['reference_cpu_sha256']
        result.update(library=batch['library'], library_sha256=batch['library_sha256'],
                      reference_cpu_sha256=sha(reference))
        source = OUT / 'glm-q8-attention3d-check.cpp'
        source.write_bytes((BASE / source.name).read_bytes())
        (OUT / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
        binary = OUT / 'attention3d-check'
        command = ['/usr/bin/c++', '-O3', '-std=c++17', '-march=native', '-fopenmp']
        command += ['-I' + str(ENGINE / p) for p in ('ggml/include', 'ggml/src', 'ggml/src/ggml-cpu')]
        command += [str(source), '-L' + str(PINNED), '-Wl,-rpath,' + str(PINNED),
                    '-lggml', '-lggml-cpu', '-lggml-base', '-ldl', '-o', str(binary)]
        run(command, 'compile')
        result.update(source_sha256=sha(source), binary_sha256=sha(binary), compile_command=command)
        env = {k: v for k, v in os.environ.items() if not k.startswith(('GGML_', 'LLAMA_', 'OMP_', 'GOMP_', 'REPACK_TEST_'))}
        hashes = []
        for mode in ('reference', 'off', 'on'):
            expected = reference if mode == 'reference' else Path(batch['library'])
            env['LD_LIBRARY_PATH'] = str(expected.parent) + ':' + str(PINNED)
            env['GGML_CPU_X16_Q8_BATCH'] = '1' if mode == 'on' else '0'
            log = run(['taskset', '-c', '48-62', str(binary)], mode, env)
            mapped = re.search(r'^CPU_LIBRARY (.+)$', log, re.M)
            assert mapped and Path(mapped[1]).resolve() == expected.resolve()
            rows = re.findall(r'^PASS (.+?) max_scaled=[^ ]+ hash=([^\n]+)', log, re.M)
            assert len(rows) == 144, (mode, len(rows))
            hashes.append(rows)
            result['checks'].append(dict(mode=mode, numeric_cases=len(rows), mapped_cpu=str(expected)))
            save()
            print(json.dumps(dict(mode=mode, cases=len(rows))), flush=True)
        assert hashes[0] == hashes[1] == hashes[2], 'Attention output bits changed'
        assert sha(batch['library']) == batch['library_sha256'] and sha(reference) == batch['reference_cpu_sha256']
        result.update(passed=True, bit_exact_cases=144)
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
