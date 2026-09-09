#!/usr/bin/env python3
"""Build private, opt-in Q6 expert batching from the selected CPU library."""
import difflib
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from qwen_q6_packed_batch_transform_0908b import transform
from qwen_split_trial import sha256

BASE = Path(__file__).resolve().parent
ENGINE = BASE.parent.parent / 'engines/llama.cpp-q4e-goal-0904'
OUT = BASE / 'results/qwen-q6-packed-batch-build-0908b'


def main():
    assert os.sched_getaffinity(0) == {127}
    parent_path = BASE / 'results/qwen-q6-q8-wide-batch-0907/private-cpu/manifest.json'
    parent = json.loads(parent_path.read_text())
    assert sha256(parent['library']) == parent['library_sha256'] == 'c79e19e38acfd6ada3f0637c136cf9cb10253127778cb214528b2f966fb6dfcf'
    assert all(sha256(path) == digest for path,digest in parent['private_source_sha256'].items())
    proof_path = BASE / 'results/qwen-q6-packed-batch-proof-0908/result.json'
    timing_paths = [BASE / 'results' / label / 'result.json' for label in
                    ('qwen-q6-packed-batch-component-0908','qwen-q6-packed-batch-component-tail-0908',
                     'qwen-q6-packed-batch-component-tail-long-0908')]
    proof = json.loads(proof_path.read_text())
    timings = [json.loads(path.read_text()) for path in timing_paths]
    assert proof['passed']
    assert all(sha256(path) == digest for path,digest in proof['source_sha256'].items())
    qualified = {}
    for path,timing in zip(timing_paths,timings):
        assert timing['finished'] and not timing.get('error') and timing['peer_preserved']
        assert all(sha256(source) == digest for source,digest in timing['input_sha256'].items())
        assert sha256(path.parent / 'benchmark') == timing['binary_sha256'] == timings[0]['binary_sha256']
        for case in timing['cases']:
            if 'accepted_attempt' not in case:
                continue
            measured = case['attempts'][case['accepted_attempt']]
            assert measured['other_host_cores'] <= 4 and measured['background_within_gate']
            assert measured['weight_storage_unchanged'] and measured['output_equality_checked']
            key = (case['rows'],case['matrices'],case['activations'])
            assert key not in qualified
            qualified[key] = dict(case=list(key),result=str(path),attempt=case['accepted_attempt'],
                                  speed_ratio=measured['packed_batch_speed_ratio'],background_cores=measured['other_host_cores'])
    expected = {(64,1,3)} | {(rows,512,count) for rows in (64,32) for count in (1,2,3,5)}
    assert set(qualified) == expected
    assert all(row['speed_ratio'] > (1.1 if key[2] > 1 else .9) for key,row in qualified.items())
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        OUT.mkdir(exist_ok=False)
        private = OUT / 'private-cpu'
        private.mkdir()
        command = list(parent['compile_commands'][0])
        source = Path(command[-1])
        old_object = command[command.index('-o')+1]
        original = source.read_text()
        changed = transform(original)
        header = BASE / 'qwen-q6-packed-batch-0908.h'
        (private / 'repack.cpp').write_text(changed)
        (private / header.name).write_bytes(header.read_bytes())
        (private / 'expert-batch.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),changed.splitlines(True),fromfile=str(source),tofile='private/repack.cpp')))
        paths = [Path(__file__).resolve(),BASE / 'qwen_q6_packed_batch_transform_0908b.py',parent_path,
                 proof_path,*timing_paths,source,header,Path(parent['library'])]
        paths += [Path(value) for value in parent['link_command'] if Path(value).is_absolute() and Path(value).is_file()]
        paths += list((ENGINE / 'ggml/src').rglob('*.h')) + list((ENGINE / 'ggml/include').rglob('*.h'))
        inputs = {str(path):sha256(path) for path in paths}
        result = dict(started=time.time(),build_completed=False,model_loaded=False,steps=[],
                      input_sha256=inputs,parent_sha256=parent['library_sha256'],component_qualification=list(qualified.values()),
                      scope='Private Q6 expert batching; selected weights and all other arithmetic objects unchanged. No model performance claim.')

        def save():
            (OUT / 'result.json').write_text(json.dumps(result,indent=2)+'\n')

        def run(cmd,label):
            with (OUT / (label+'.log')).open('w') as log:
                completed = subprocess.run(cmd,cwd=ENGINE,stdout=log,stderr=subprocess.STDOUT,timeout=300)
            result['steps'].append(dict(label=label,command=cmd,exit_code=completed.returncode))
            save()
            assert completed.returncode == 0,label
            print(json.dumps(dict(completed=label)),flush=True)

        save()
        try:
            baseline_path = BASE / 'results/qwen-q6-packed-batch-build-0908/result.json'
            baseline = json.loads(baseline_path.read_text())
            assert baseline['baseline_link_identical'] and baseline['unpatched_text_identical']
            assert all(sha256(path) == digest for path,digest in baseline['input_sha256'].items())
            result.update(baseline_link_identical=True,unpatched_text_identical=True,
                          baseline_proof=str(baseline_path),baseline_proof_sha256=sha256(baseline_path))
            command[command.index('-o')+1] = str(private / 'repack.cpp.o')
            command[-1] = str(private / 'repack.cpp')
            run(command,'private-compile')
            library = private / 'libggml-cpu.so.0.22.0'
            link = [str(private / 'repack.cpp.o') if value == old_object else value for value in parent['link_command']]
            link[link.index('-o')+1] = str(library)
            run(link,'private-link')
            (private / 'libggml-cpu.so.0').symlink_to(library.name)
            (private / 'libggml-cpu.so').symlink_to('libggml-cpu.so.0')
            assert all(sha256(path) == digest for path,digest in inputs.items())
            manifest = dict(input_sha256=inputs,parent_manifest=str(parent_path),parent_sha256=parent['library_sha256'],
                            library=str(library),library_sha256=sha256(library),compile_commands=[command],link_command=link,
                            private_source_sha256={str(path):sha256(path) for path in [private / 'repack.cpp',private / header.name]},
                            scope=result['scope'])
            (private / 'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            result.update(build_completed=True,library=str(library),library_sha256=sha256(library))
        except BaseException as error:
            result['error'] = repr(error)
            raise
        finally:
            result['finished'] = time.time()
            save()


if __name__ == '__main__':
    main()
