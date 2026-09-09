#!/usr/bin/env python3
"""Build an opt-in Qwen residual-combination graph change in a private llama library."""
import difflib
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from qwen_split_trial import sha256

BASE = Path(__file__).resolve().parent
ENGINE = BASE.parent.parent / 'engines/llama.cpp-q4e-goal-0904'
OUT = BASE / 'results/qwen-hc-combine-build-0908'


def main():
    assert os.sched_getaffinity(0) == {127}
    parent_path = BASE / 'results/qwen-expert-even-split-policy-0906b/private-split/manifest.json'
    proof_path = BASE / 'results/qwen-hc-combine-proof-0908/result.json'
    parent, proof = [json.loads(p.read_text()) for p in (parent_path, proof_path)]
    assert sha256(parent['library']) == parent['library_sha256'] == 'd213ba477ef5d3f2a68568b30a31595ef9578b6dce4b3b81e95da6f83a3dac8b'
    assert all(sha256(p) == h for p,h in parent['input_sha256'].items())
    assert sha256(parent['compile_command'][-1]) == parent['private_source_sha256']
    assert proof['passed'] and len(proof['checks']) == 2
    assert sum(row['cases'] for row in proof['checks']) == 384
    assert all(sha256(p) == h for p,h in proof['input_sha256'].items())
    original_path = ENGINE / 'src/models/qwen4exp.cpp'
    original_object = ENGINE / 'build-goal/src/CMakeFiles/llama.dir/models/qwen4exp.cpp.o'
    header = BASE / 'qwen-hc-combine-0908.h'
    original = original_path.read_text()
    assert original.count('#include "models.h"') == 1
    changed = original.replace('#include "models.h"', '#include "models.h"\n#include "qwen-hc-combine-0908.h"')
    old = '''    b = ggml_repeat_4d(ctx0, b, n_embd, hc, nt, 1);

    ggml_tensor * cur = ggml_add(ctx0, residual, ggml_mul(ctx0, b, w));'''
    new = '''    static const bool fused = [] {
        const char * value = getenv("LLAMA_QWEN_HC_COMBINE_FUSED");
        return value && atoi(value) == 1;
    }();
    ggml_tensor * cur;
    if (fused) {
        cur = ggml_map_custom3(ctx0, residual, b, w, qwen_hc_combine_f32, GGML_N_TASKS_MAX, nullptr);
    } else {
        b = ggml_repeat_4d(ctx0, b, n_embd, hc, nt, 1);
        cur = ggml_add(ctx0, residual, ggml_mul(ctx0, b, w));
    }'''
    assert changed.count(old) == 1
    changed = changed.replace(old, new)
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        OUT.mkdir(exist_ok=False)
        private = OUT / 'private-llama'
        private.mkdir()
        (private / 'qwen4exp.cpp').write_text(changed)
        (private / header.name).write_bytes(header.read_bytes())
        (private / 'hc-combine.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True), changed.splitlines(True), fromfile=str(original_path), tofile='private/qwen4exp.cpp')))
        inputs = [Path(__file__).resolve(), parent_path, proof_path, original_path, original_object, header, Path(parent['library'])]
        inputs += [Path(value) for value in parent['link_command'] if Path(value).is_absolute() and Path(value).is_file()]
        inputs += list((ENGINE / 'src').rglob('*.h')) + list((ENGINE / 'include').rglob('*.h')) + list((ENGINE / 'ggml/include').rglob('*.h'))
        result = dict(started=time.time(), build_completed=False, model_loaded=False, steps=[],
                      input_sha256={str(p):sha256(p) for p in inputs}, parent_sha256=parent['library_sha256'],
                      scope='Only qwen4exp.cpp.o changes; CPU kernels, quantization, and all other model objects stay unchanged.')

        def save():
            (OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')

        def run(command, label):
            with (OUT / (label + '.log')).open('w') as log:
                done = subprocess.run(command, cwd=ENGINE, stdout=log, stderr=subprocess.STDOUT, timeout=300)
            result['steps'].append(dict(label=label, command=command, exit_code=done.returncode))
            save()
            assert done.returncode == 0, label
            print(json.dumps(dict(completed=label)), flush=True)

        save()
        try:
            command = list(parent['compile_command'])
            command[-1] = str(original_path)
            command[command.index('-o') + 1] = str(OUT / 'baseline-qwen4exp.cpp.o')
            run(command, 'baseline-compile')
            baseline_link = [str(OUT / 'baseline-qwen4exp.cpp.o') if v == str(original_object) else v for v in parent['link_command']]
            assert baseline_link.count(str(OUT / 'baseline-qwen4exp.cpp.o')) == 1
            baseline_link[baseline_link.index('-o') + 1] = str(OUT / 'baseline-libllama.so')
            run(baseline_link, 'baseline-link')
            assert sha256(OUT / 'baseline-libllama.so') == parent['library_sha256']
            result['baseline_link_identical'] = True
            command[1:1] = ['-march=native', '-ffp-contract=off', '-I' + str(original_path.parent)]
            command[-1] = str(private / 'qwen4exp.cpp')
            command[command.index('-o') + 1] = str(private / 'qwen4exp.cpp.o')
            run(command, 'private-compile')
            library = private / 'libllama.so.0.3.0'
            link = [str(private / 'qwen4exp.cpp.o') if v == str(original_object) else v for v in parent['link_command']]
            link[link.index('-o') + 1] = str(library)
            run(link, 'private-link')
            (private / 'libllama.so.0').symlink_to(library.name)
            (private / 'libllama.so').symlink_to('libllama.so.0')
            assert all(sha256(p) == h for p,h in result['input_sha256'].items())
            manifest = dict(input_sha256=result['input_sha256'], parent_manifest=str(parent_path), parent_sha256=parent['library_sha256'],
                            library=str(library), library_sha256=sha256(library), compile_command=command, link_command=link,
                            private_source_sha256={str(p):sha256(p) for p in (private / 'qwen4exp.cpp', private / header.name)}, scope=result['scope'])
            (private / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
            result.update(build_completed=True, library=str(library), library_sha256=sha256(library))
        except BaseException as error:
            result['error'] = repr(error)
            raise
        finally:
            result['finished'] = time.time()
            save()


if __name__ == '__main__':
    main()
