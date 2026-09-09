#!/usr/bin/env python3
"""Prepare a versioned single-activation Q6 primitive and standalone checks."""
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def main():
    origins = {}
    outputs = {}

    def derive(parent, target, transform):
        src, dst = BASE / parent, BASE / target
        assert not dst.exists(), dst
        origins[str(src)] = digest(src)
        dst.write_text(transform(src.read_text()))
        outputs[str(dst)] = digest(dst)

    def header(text):
        text = replace_once(text, 'static_assert(NR >= 2 && NR <= 3);',
                            'static_assert(NR >= 1 && NR <= 3);')
        text = replace_once(text,
            'ggml_gemv_q6_K_x16_q8_K(n, output[r], 0, weights, activation[r], 1, rows);',
            'qwen_q6_packed_batch_impl<1>(n, rows, weights, activation + r, output + r);')
        return text.replace('// Share bit extraction across distinct activation rows. Weight storage is unchanged.',
            '// Apply the same sub-block correction to one, two, or three activation rows.\n'
            '// Six-bit codes, scales, weight storage, and FP32 accumulation order are unchanged.')

    old_header = 'qwen-q6-packed-batch-0908.h'
    new_header = 'qwen-q6-packed-single-0908.h'
    derive(old_header, new_header, header)
    derive('check-qwen-q6-packed-batch-0908.cpp', 'check-qwen-q6-packed-single-0908.cpp',
           lambda s: replace_once(s, old_header, new_header))
    derive('benchmark-qwen-q6-packed-batch-0908.cpp', 'benchmark-qwen-q6-packed-single-0908.cpp',
           lambda s: replace_once(s, old_header, new_header))

    def check_runner(text):
        for old, new in [
            ('Check a standalone packed-Q6 batch primitive; no performance claim.',
             'Check the Q6 single-activation correction variant; no performance claim.'),
            ('results/qwen-q6-packed-batch-proof-0908', 'results/qwen-q6-packed-single-proof-0908'),
            ('check-qwen-q6-packed-batch-0908.cpp', 'check-qwen-q6-packed-single-0908.cpp'),
            (old_header, new_header),
        ]:
            text = replace_once(text, old, new)
        return text

    derive('run_qwen_q6_packed_batch_check_0908.py', 'run_qwen_q6_packed_single_check_0908.py', check_runner)

    def benchmark_runner(text):
        for old, new in [
            ('Qualify packed Q6 batching with distinct activations and a CPU gate per case.',
             'Qualify Q6 single-activation correction on warm and shuffled cold matrices.'),
            ('qwen-q6-packed-batch-component-[A-Za-z0-9_-]+', 'qwen-q6-packed-single-component-[A-Za-z0-9_-]+'),
            ('results/qwen-q6-packed-batch-proof-0908', 'results/qwen-q6-packed-single-proof-0908'),
            ('check-qwen-q6-packed-batch-0908.cpp', 'check-qwen-q6-packed-single-0908.cpp'),
            ('benchmark-qwen-q6-packed-batch-0908.cpp', 'benchmark-qwen-q6-packed-single-0908.cpp'),
            (old_header, new_header),
            ('cases = [(64,1,3)] + [(rows,512,activations) for rows in (64,32) for activations in (1,2,3,5)]',
             'cases = [(rows,matrices,1) for matrices in (1,512) for rows in (64,32)]'),
            ("str(activations),'.6'", "str(activations),'1.5'"),
        ]:
            text = replace_once(text, old, new)
        return text

    derive('run_qwen_q6_packed_batch_benchmark_0908.py',
           'run_qwen_q6_packed_single_benchmark_0908.py', benchmark_runner)
    record = BASE / 'results/qwen-30tps-tools-0908/packed-single-preparation.json'
    assert not record.exists()
    record.write_text(json.dumps(dict(
        scope=__doc__, parent_sha256=origins, generated_sha256=outputs,
        generator_sha256=digest(Path(__file__).resolve()),
        runtime_integrated=False, model_performance_established=False,
    ), indent=2) + '\n')
    print(json.dumps(dict(prepared=len(outputs), record=str(record))))


if __name__ == '__main__':
    main()
