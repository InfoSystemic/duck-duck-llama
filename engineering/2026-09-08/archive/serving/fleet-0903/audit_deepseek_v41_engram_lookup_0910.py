#!/usr/bin/env python3
"""Reconcile native lookup sources, exact checks, real row extents and live peer identity."""
import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import struct
import time

from model_measurement_guard import ModelMeasurementGuard
from probe_deepseek_v41_native_cpu_0910 import reference_bf16
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import inference_snapshot, sha256
from select_flash_q4_0910c import Manager

BASE = Path(__file__).resolve().parent
OUT = BASE / 'results/deepseek-v41-engram-lookup-audit-0910.json'


def read(path):
    assert '.private.' not in str(path)
    return json.loads(Path(path).read_text())


def hashes(values):
    for path, digest in values.items():
        assert '.private.' not in path and sha256(path) == digest, path


def main():
    assert os.sched_getaffinity(0) == {127} and not OUT.exists()
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manager = Manager()
        current = manager.validate_current()
        guard = ModelMeasurementGuard(current['pid'], {current['pid']: 18131}, inference_snapshot)
        guard.assert_idle()
        directory = BASE / 'results/deepseek-v41-engram-lookup-0910'
        result, check, fixture = [read(directory / name) for name in ['result.json', 'correctness.json', 'real-rows.json']]
        assert result['passed'] and result['finished'] and check['passed']
        assert result['selected_flash_preserved'] and result['selected_flash_pid'] == current['pid']
        hashes(result['input_sha256']); hashes(check['input_sha256'])
        assert result['correctness_sha256'] == sha256(directory / 'correctness.json')
        assert result['library_sha256'] == sha256(directory / 'libdeepseek-v41-engram-lookup.so')
        assert check['real_manifest_sha256'] == sha256(directory / 'real-rows.json')
        assert [s['label'] for s in result['steps']] == ['compile-library', 'official-reference-correctness']
        for step in result['steps']:
            assert step['exit_code'] == 0 and sha256(directory / (step['label'] + '.log')) == step['log_sha256']
        expected_modes = {(p, r, v) for p in [1, 4] for r in [0, 1] for v in [0, 1]}
        assert len(check['cases']) == 68
        for case in check['cases']:
            assert case['exact'] and {tuple(m) for m in case['modes']} == expected_modes
        bf16_count = sum(c['batch'] * c['tokens'] * 48 * 256 * len(c['modes']) for c in check['cases'])
        id_count = sum(c['batch'] * c['tokens'] * 48 * len(c['modes']) for c in check['cases'] if c['ids_returned'])
        assert bf16_count == check['exact_bf16_values_compared'] == result['exact_bf16_values_compared'] == 267583488
        assert id_count == check['exact_row_ids_compared'] == result['exact_row_ids_compared'] == 1044096
        assert len(check['invalid_calls']) == 112 and all(c['rejected'] and c['unchanged'] for c in check['invalid_calls'])
        assert len(check['rejected_configurations']) == 19
        labels = {c['label'] for c in check['cases']}
        assert {'clone-append', 'original-independent-append', 'clone-rewind-append', 'overwrite-suffix',
            'append-after-truncation', 'request-reset', 'append-after-reset', 'optional-ids-omitted'} <= labels
        assert {f'image-mask-{i}' for i in range(16)} <= labels

        intake = BASE / 'results/deepseek-v41-intake-0910'
        sources = read(intake / 'sources.json')
        for suffix in ['inference/engram.py', 'inference/config.json', 'inference/model.py']:
            source, = [s for s in sources if s['file'].endswith('/official/' + suffix)]
            assert fixture['revision'] in source['url']
            assert sha256(source['file']) == source['sha256'] == check['input_sha256'][source['file']]
        nodes = ast.parse((intake / 'official/inference/model.py').read_text()).body
        node, = [n for n in nodes if isinstance(n, ast.ClassDef) and n.name == 'ParallelEngramEmbedding']
        assert hashlib.sha256(ast.dump(node).encode()).hexdigest() == check['official_embedding_class_ast_sha256']

        # Recompute the actual selected IDs with Python integer arithmetic, independent
        # of both native code and the tensor operations used by the correctness checker.
        old_dir = BASE / 'results/deepseek-v41-engram-hash-0910'
        metadata = read(old_dir / 'correctness.json')
        raw_map = (old_dir / 'compressed-token-map.bin').read_bytes()
        assert hashlib.sha256(raw_map).hexdigest() == metadata['token_map_sha256']
        compressed, = struct.unpack_from('<I', raw_map, 42 * 4)
        assert fixture['token_ids'] == [42] and fixture['start'] == 0 and fixture['mask'] is None
        selected = []
        for layer in range(2):
            values, offset = [], 0
            rolling = compressed * metadata['multipliers'][layer][0]
            for shift in range(1, 4):
                rolling ^= metadata['compressed_pad_id'] * metadata['multipliers'][layer][shift]
                for prime in metadata['moduli'][layer][shift-1]:
                    values.append(rolling % prime + offset)
                    offset += prime
            selected.append(values)
        assert fixture['selected_rows'] == [[selected]]
        assert sum(metadata['table_rows']) * 264 == check['sparse_virtual_table_bytes'] == 202758032400
        tensors = read(BASE / 'results/deepseek-v41-tensor-audit-0910/result.json')
        index = read(intake / 'official/model.safetensors.index.json')['weight_map']
        assert tensors['revision'] == fixture['revision'] == 'fb2764a5cf321eaa5070ca8f9e892818f477c16d'
        assert len(fixture['records']) == 96 and len({(r['layer'], r['row'], r['suffix']) for r in fixture['records']}) == 96
        for record in fixture['records']:
            path = Path(record['file'])
            assert path.parent == directory / 'real-rows' and sha256(path) == record['sha256']
            assert path.stat().st_size == record['bytes'] == {'weight': 256, 'scale': 8}[record['suffix']]
            assert record['row'] in selected[record['layer']]
            shard, = [s for s in tensors['shards'] if s['file'] == index[record['tensor']]]
            header_path = BASE / 'results/deepseek-v41-tensor-audit-0910/headers' / (shard['file'] + '.json')
            assert sha256(header_path) == shard['header_sha256'] == record['header_sha256']
            tensor = read(header_path)[record['tensor']]
            assert record['source_url'] == shard['source_url'] and record['total'] == shard['size']
            assert record['publisher_whole_shard_sha256'] == shard['publisher_sha256']
            assert record['start'] == 8 + shard['header_bytes'] + tensor['data_offsets'][0] + record['row'] * record['bytes']
            assert record['end'] == record['start'] + record['bytes'] - 1
        expected_bits = []
        for layer in range(2):
            for row in selected[layer]:
                w, s = [Path(next(r['file'] for r in fixture['records'] if r['layer'] == layer and r['row'] == row and r['suffix'] == suffix)).read_bytes()
                    for suffix in ['weight', 'scale']]
                expected_bits.extend(reference_bf16(code, s[i // 32]) for i, code in enumerate(w))
        assert all((v & 0x7f80) != 0x7f80 for v in expected_bits)
        expected_bytes = struct.pack('<' + 'H' * len(expected_bits), *expected_bits)
        real_case, = [c for c in check['cases'] if c['label'] == 'actual-pinned-token42-lookups']
        assert hashlib.sha256(expected_bytes).hexdigest() == real_case['expected_bf16_sha256']
        assert sum(r['bytes'] for r in fixture['records']) == check['actual_retrieved_bytes'] == result['actual_retrieved_bytes'] == 12672
        assert check['actual_pinned_rows'] == result['actual_pinned_rows'] == 48
        manager.validate_current(); guard.assert_idle()
        report = dict(passed=True, finished=time.time(), source_sha256=sha256(__file__),
            input_sha256={str(p): sha256(p) for p in [directory / 'result.json', directory / 'correctness.json',
                directory / 'real-rows.json', BASE / 'probe_deepseek_v41_native_cpu_0910.py']},
            selected_flash_pid=current['pid'], selected_flash_preserved=True,
            sequence_cases=68, native_variants=8, exact_bf16_values_compared=bf16_count,
            exact_row_ids_compared=id_count, rejected_input_calls=112, rejected_configurations=19,
            actual_selected_rows=48, actual_retrieved_bytes=12672, actual_row_reference_finite=True,
            full_model_loaded=False, model_tok_s_measured=False, imc_bandwidth_measured=False, runtime_promoted=False,
            limitations=['Only hash to native FP8/BF16 lookup is integrated; projection/gate and full graph remain pending.',
                'One/four logical row shards are address-dispatch checks, not multi-socket scaling measurements.',
                'Official 202.8 GB address spans are sparsely populated fixtures, not downloaded full tables.',
                'Whole-shard publisher hashes are recorded, not verified. Retrieved slices are individually hash-bound.'])
        atomic_json(OUT, report)
        print(json.dumps({k: report[k] for k in ['passed', 'sequence_cases', 'exact_bf16_values_compared',
            'exact_row_ids_compared', 'actual_selected_rows', 'selected_flash_pid']}))


if __name__ == '__main__':
    os.umask(0o077)
    main()
