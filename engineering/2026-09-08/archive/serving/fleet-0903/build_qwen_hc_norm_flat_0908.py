#!/usr/bin/env python3
"""Build flattened HC normalization on the private HC stream-mean runtime."""
import difflib
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from model_measurement_guard import ModelMeasurementGuard
from qwen_split_trial import inference_snapshot, process_info, sha256

BASE = Path(__file__).resolve().parent
ENGINE = BASE.parents[1] / 'engines/llama.cpp-q4e-goal-0904'
OUT = BASE / 'results/qwen-hc-norm-flat-build-0908'


def main():
    assert os.sched_getaffinity(0) == {127}
    assert process_info(1219506)['start'] == '103969952'
    parent_path = BASE / 'results/qwen-hc-mix-build-0908/private-llama/manifest.json'
    proof_path = BASE / 'results/qwen-hc-norm-flat-proof-0908/result.json'
    parent, proof = [json.loads(path.read_text()) for path in (parent_path, proof_path)]
    assert sha256(parent['library']) == parent['library_sha256'] == '63ff8e9bad975e2c42efd26d890611ee0c915ab87ed3434a7eaef2a2d02f6c90'
    assert all(sha256(path) == value for path, value in parent['input_sha256'].items())
    assert all(sha256(path) == value for path, value in parent['private_source_sha256'].items())
    assert proof['passed'] and len(proof['checks']) == 2
    assert sum(row['cases'] for row in proof['checks']) == 768
    assert all(row['bit_exact'] and row['scalar_exact'] and row['inputs_preserved'] for row in proof['checks'])
    assert all(sha256(path) == value for path, value in proof['input_sha256'].items())
    original_path = Path(parent['compile_command'][-1])
    original_object = Path(parent['compile_command'][parent['compile_command'].index('-o') + 1])
    header = BASE / 'qwen-hc-norm-flat-0908.h'
    original = original_path.read_text()
    assert original.count('#include "models.h"') == 1
    changed = original.replace('#include "models.h"', '#include "models.h"\n#include "qwen-hc-norm-flat-0908.h"')
    start = changed.index('    ggml_tensor * xn = ggml_rms_norm(ctx0, x, hparams.f_norm_rms_eps);')
    end = changed.index('    cb(xn, "hc_norm", il);', start)
    old = changed[start:end]
    assert old.count('ggml_tensor * xn =') == 1
    fallback = old.replace('ggml_tensor * xn =', 'xn =')
    fallback = ''.join('    ' + line if line.strip() else line for line in fallback.splitlines(True))
    new = """    static const bool norm_flat = [] {
        const char * value = getenv("GGML_QWEN_HC_NORM_FLAT");
        return value && atoi(value) == 1;
    }();
    ggml_tensor * xn;
    if (norm_flat) {
        xn = qwen_hc_norm_flat(ctx0, gf, x, w_norm, hparams.f_norm_rms_eps);
    } else {
""" + fallback + '    }\n'
    assert changed.count(old) == 1
    changed = changed.replace(old, new)
    guard = ModelMeasurementGuard(1219506, {1219506:18095}, inference_snapshot)
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guard.assert_idle()
        OUT.mkdir(exist_ok=False)
        private = OUT / 'private-llama'
        private.mkdir()
        (private / original_path.name).write_text(changed)
        for path in [header] + [Path(path) for path in parent['private_source_sha256'] if Path(path).suffix == '.h']:
            (private / path.name).write_bytes(path.read_bytes())
        (private / 'hc-norm-flat.patch').write_text(''.join(difflib.unified_diff(
            original.splitlines(True), changed.splitlines(True), fromfile=str(original_path), tofile='private/qwen4exp.cpp')))
        inputs = {str(path):sha256(path) for path in [Path(__file__).resolve(), parent_path, proof_path,
                  original_path, original_object, header, Path(parent['library'])]}
        inputs.update(parent['input_sha256'])
        inputs.update(parent['private_source_sha256'])
        result = dict(started=time.time(), build_completed=False, model_loaded=False, steps=[],
                      input_sha256=inputs, parent_sha256=parent['library_sha256'],
                      scope='Only qwen4exp.cpp.o changes from HC-mix. CPU kernels, quantization, and other model objects stay unchanged.')

        def save():
            (OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')

        def run(command, label):
            guard.assert_idle()
            with (OUT / (label + '.log')).open('w') as log:
                done = subprocess.run(command, cwd=ENGINE, stdout=log, stderr=subprocess.STDOUT, timeout=300)
            result['steps'].append(dict(label=label, command=command, exit_code=done.returncode))
            save()
            assert done.returncode == 0, label
            guard.assert_idle()
            print(json.dumps(dict(completed=label)), flush=True)

        save()
        try:
            baseline_object = OUT / 'baseline-qwen4exp.cpp.o'
            baseline_compile = list(parent['compile_command'])
            baseline_compile[baseline_compile.index('-o') + 1] = str(baseline_object)
            run(baseline_compile, 'baseline-compile')
            result['baseline_object_identical'] = sha256(baseline_object) == sha256(original_object)
            assert result['baseline_object_identical']
            baseline_library = OUT / 'baseline-libllama.so'
            baseline_link = [str(baseline_object) if value == str(original_object) else value for value in parent['link_command']]
            baseline_link[baseline_link.index('-o') + 1] = str(baseline_library)
            run(baseline_link, 'baseline-link')
            result['baseline_link_identical'] = sha256(baseline_library) == parent['library_sha256']
            assert result['baseline_link_identical']
            command = list(parent['compile_command'])
            command[-1] = str(private / original_path.name)
            command[command.index('-o') + 1] = str(private / original_object.name)
            run(command, 'private-compile')
            library = private / 'libllama.so.0.3.0'
            link = [str(private / original_object.name) if value == str(original_object) else value for value in parent['link_command']]
            link[link.index('-o') + 1] = str(library)
            run(link, 'private-link')
            (private / 'libllama.so.0').symlink_to(library.name)
            (private / 'libllama.so').symlink_to('libllama.so.0')
            assert all(sha256(path) == value for path, value in inputs.items())
            source_files = [private / original_path.name] + sorted(private.glob('*.h'))
            manifest = dict(input_sha256=inputs, parent_manifest=str(parent_path), parent_sha256=parent['library_sha256'],
                            library=str(library), library_sha256=sha256(library), compile_command=command, link_command=link,
                            private_source_sha256={str(path):sha256(path) for path in source_files}, scope=result['scope'])
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
