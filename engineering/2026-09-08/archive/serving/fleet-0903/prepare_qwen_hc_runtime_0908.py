#!/usr/bin/env python3
"""Prepare versioned HC-fusion build, loader, and bounded model controllers."""
import ast
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent


def replace(text, old, new, count=1):
    assert text.count(old) == count, (old, text.count(old), count)
    return text.replace(old, new)


def main():
    parents, outputs = {}, {}

    def derive(src, dst, transform):
        original, target = BASE / src, BASE / dst
        assert not target.exists()
        text = transform(original.read_text())
        ast.parse(text)
        target.write_text(text)
        parents[str(original)] = hashlib.sha256(original.read_bytes()).hexdigest()
        outputs[str(target)] = hashlib.sha256(target.read_bytes()).hexdigest()

    def builder(text):
        text = replace(text, "OUT = BASE / 'results/qwen-hc-combine-build-0908'", "OUT = BASE / 'results/qwen-hc-combine-build-0908b'")
        text = replace(text, 'LLAMA_QWEN_HC_COMBINE_FUSED', 'GGML_QWEN_HC_COMBINE_FUSED')
        start = text.index("            command = list(parent['compile_command'])")
        end = text.index("            command[1:1] =", start)
        text = text[:start] + '''            baseline_path = BASE / 'results/qwen-hc-combine-build-0908/result.json'
            baseline = json.loads(baseline_path.read_text())
            assert baseline['build_completed'] and baseline['baseline_link_identical']
            assert all(sha256(p) == h for p,h in baseline['input_sha256'].items())
            assert sha256(baseline_path.parent / 'baseline-libllama.so') == parent['library_sha256']
            result.update(baseline_link_identical=True, baseline_proof=str(baseline_path), baseline_proof_sha256=sha256(baseline_path))
            command = list(parent['compile_command'])
''' + text[end:]
        return text

    derive('build_qwen_hc_combine_0908.py', 'build_qwen_hc_combine_0908b.py', builder)

    def loader(text):
        replacements = [
            ('Package and inspect the Qwen Q6 batching runtime without loading weights.', 'Package and inspect the Qwen HC-fusion runtime without loading weights.'),
            ('results/qwen-q6-packed-batch-runtime-0908', 'results/qwen-hc-combine-runtime-0908'),
            ('results/qwen-q6-packed-batch-validation-0908b/result.json', 'results/qwen-hc-combine-proof-0908/result.json'),
            ('results/qwen-q6-packed-batch-build-0908b/result.json', 'results/qwen-hc-combine-build-0908b/result.json'),
            ('''        assert validation['passed'] and validation['bit_exact'] and len(validation['checks']) == 10
        assert len(validation['numa_checks']) == 10
        assert build['build_completed'] and build['baseline_link_identical'] and build['unpatched_text_identical']
        cpu = Path(build['library'])
        assert sha256(cpu) == build['library_sha256'] == validation['cpu_sha256']
        llama = BASE / 'results/qwen-expert-even-split-policy-0906b/private-split/libllama.so.0.3.0'
        assert sha256(llama) == preset['binary_sha256'][str(llama)]''',
             '''        assert validation['passed'] and len(validation['checks']) == 2
        assert sum(row['cases'] for row in validation['checks']) == 384
        assert all(row['bit_exact'] and row['scalar_exact'] for row in validation['checks'])
        assert build['build_completed'] and build['baseline_link_identical']
        assert all(sha256(p) == h for p,h in validation['input_sha256'].items())
        assert all(sha256(p) == h for p,h in build['input_sha256'].items())
        cpu = BASE / 'results/qwen-q6-q8-wide-batch-0907/private-cpu/libggml-cpu.so.0.22.0'
        assert sha256(cpu) == 'c79e19e38acfd6ada3f0637c136cf9cb10253127778cb214528b2f966fb6dfcf'
        llama = Path(build['library'])
        assert sha256(llama) == build['library_sha256']'''),
            ("GGML_CPU_X16_Q6_EXPERT_BATCH='1'", "GGML_QWEN_HC_COMBINE_FUSED='1'"),
        ]
        for old, new in replacements:
            text = replace(text, old, new)
        return text

    derive('isolate_qwen_q6_packed_batch_runtime_0908.py', 'isolate_qwen_hc_combine_runtime_0908.py', loader)

    def controller(text):
        replacements = [
            ('results/qwen-q6-packed-batch-runtime-0908/result.json', 'results/qwen-hc-combine-runtime-0908/result.json'),
            ('results/qwen-q6-packed-batch-validation-0908b/result.json', 'results/qwen-hc-combine-proof-0908/result.json'),
            ('results/qwen-q6-packed-batch-build-0908b/result.json', 'results/qwen-hc-combine-build-0908b/result.json'),
            ('''    assert bundle['passed'] and checks['passed'] and checks['bit_exact'] and build['build_completed']
    assert len(checks['checks']) == 10 and sum(row['cases'] for row in checks['checks']) == 360
    assert len(checks['numa_checks']) == 10 and checks['bit_exact']
    assert bundle['cpu_sha256'] == checks['cpu_sha256'] == build['library_sha256'] == sha256(bundle['cpu'])
    assert build['baseline_link_identical'] and build['unpatched_text_identical']''',
             '''    assert bundle['passed'] and checks['passed'] and build['build_completed']
    assert len(checks['checks']) == 2 and sum(row['cases'] for row in checks['checks']) == 384
    assert all(row['bit_exact'] and row['scalar_exact'] for row in checks['checks'])
    assert bundle['cpu_sha256'] == sha256(bundle['cpu']) == 'c79e19e38acfd6ada3f0637c136cf9cb10253127778cb214528b2f966fb6dfcf'
    assert bundle['llama_sha256'] == build['library_sha256'] == sha256(bundle['llama'])
    assert build['baseline_link_identical']'''),
            ("checks['source_sha256'].items()", "checks['input_sha256'].items()"),
            ('GGML_CPU_X16_Q6_EXPERT_BATCH', 'GGML_QWEN_HC_COMBINE_FUSED'),
            ('''            rows.append(dict(workload=row['kind'], tok_s=speed, adjusted_gb_s=traffic,
                             both_targets_met=speed > 30 and traffic > 190))''',
             '''            before = row['baseline_before']['total_gb_s']
            after = row['baseline_after']['total_gb_s']
            attribution_valid = max(before, after) <= .05 * plan['capacity_gb_s'] and abs(before - after) <= .025 * plan['capacity_gb_s']
            rows.append(dict(workload=row['kind'], tok_s=speed, adjusted_gb_s=traffic,
                             idle_before_gb_s=before, idle_after_gb_s=after, attribution_valid=attribution_valid,
                             both_targets_met=attribution_valid and speed > 30 and traffic > 190))'''),
            ("    parser.add_argument('--batch', choices=('off', 'on'), default='on')", "    parser.add_argument('--combine', choices=('off', 'on'), default='on')"),
        ]
        for old, new in replacements:
            text = replace(text, old, new)
        text = replace(text, 'args.batch', 'args.combine', count=3)
        text = replace(text, 'batch=args.combine', 'combine=args.combine', count=2)
        return text

    derive('qwen_private_packed_batch_trial_0908.py', 'qwen_private_hc_combine_trial_0908.py', controller)
    record = BASE / 'results/qwen-30tps-tools-0908/hc-combine-runtime-preparation.json'
    assert not record.exists()
    record.write_text(json.dumps(dict(parent_sha256=parents, generated_sha256=outputs,
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        note='Use a GGML-prefixed opt-in so existing environment capture includes the actual setting. Reuse the exact baseline rebuild; no installed runtime change.'), indent=2) + '\n')
    print(json.dumps(dict(prepared=len(outputs), record=str(record))))


if __name__ == '__main__':
    main()
