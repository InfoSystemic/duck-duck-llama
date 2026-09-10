#!/usr/bin/env python3
"""Reconcile pinned sources, official hash checks, native binaries and timing logs."""
import fcntl
import json
import os
from pathlib import Path
import statistics
import time

from model_measurement_guard import ModelMeasurementGuard
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import inference_snapshot, sha256
from select_flash_q4_0910c import Manager

BASE = Path(__file__).resolve().parent
OUT = BASE / 'results/deepseek-v41-engram-hash-audit-0910.json'


def read(path):
    assert '.private.' not in str(path)
    return json.loads(Path(path).read_text())


def hashes(items):
    for path, digest in items.items():
        assert '.private.' not in path and sha256(path) == digest, path


def main():
    assert os.sched_getaffinity(0) == {127} and not OUT.exists()
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manager = Manager()
        current = manager.validate_current()
        guard = ModelMeasurementGuard(current['pid'], {current['pid']: 18131}, inference_snapshot)
        guard.assert_idle()
        setup_dir = BASE / 'results/deepseek-v41-hash-reference-setup-0910'
        run_dir = BASE / 'results/deepseek-v41-engram-hash-0910'
        setup, result = read(setup_dir / 'result.json'), read(run_dir / 'result.json')
        correctness = read(run_dir / 'correctness.json')
        assert setup['passed'] and result['passed'] and correctness['passed']
        assert setup['selected_flash_preserved'] and result['selected_flash_preserved']
        assert setup['peer_pid'] == result['selected_flash_pid'] == current['pid']
        hashes(result['input_sha256'])
        hashes(correctness['sources'])
        assert sha256(BASE / 'setup_deepseek_v41_hash_reference_0910.py') == setup['source_sha256']
        sources = read(BASE / 'results/deepseek-v41-intake-0910/sources.json')
        for suffix in ['/official/inference/engram.py', '/official/inference/config.json']:
            source, = [s for s in sources if s['file'].endswith(suffix)]
            assert source['url'].find(setup['revision']) >= 0
            assert sha256(source['file']) == source['sha256'] == correctness['sources'][source['file']]
        for item in setup['retrieved']:
            assert sha256(item['file']) == item['sha256']
            assert Path(item['file']).stat().st_size == item['bytes']
        for directory, data in [(setup_dir, setup), (run_dir, result)]:
            for step in data['steps']:
                assert step['exit_code'] == 0
                assert sha256(directory / (step['label'] + '.log')) == step['log_sha256']
        assert sha256(run_dir / 'correctness.json') == result['correctness_sha256']
        assert sha256(run_dir / 'libdeepseek-v41-engram-hash.so') == result['library_sha256']
        assert sha256(run_dir / 'deepseek-v41-engram-hash-bench') == result['benchmark_sha256']
        assert sha256(run_dir / 'compressed-token-map.bin') == correctness['token_map_sha256']
        assert sha256(run_dir / 'hash-config.bin') == correctness['config_sha256']
        assert len(correctness['cases']) == 120 and all(c['exact'] for c in correctness['cases'])
        values = sum(c['batch'] * c['tokens'] * 48 * len(c['implementations']) for c in correctness['cases'])
        assert values == correctness['exact_hash_values_compared'] == result['exact_hash_values_compared'] == 16091808
        assert correctness['original_vocabulary'] == 129280 and correctness['compressed_vocabulary'] == 99092
        assert correctness['table_rows'] == [384006168, 384016682]
        assert correctness['reciprocal_arithmetic_cases'] == 316946
        assert correctness['rejected_configurations'] == 11 and len(correctness['invalid_calls']) == 16
        assert setup['reference']['versions'] == correctness['versions']
        assert setup['reference']['cuda_version'] is None and not setup['reference']['cuda_available']
        summaries = []
        for tokens in [1, 128, 4096]:
            runs = [r for r in result['runs'] if r['tokens'] == tokens]
            assert [r['mode'] for r in runs] == [0, 1, 1, 0]
            assert len({(r['timing']['output_digest'], r['timing']['guard']) for r in runs}) == 1
            for run in runs:
                assert read(run_dir / f"timing-{tokens}-{run['index']}-{run['mode']}.log") == run['timing']
            baseline, candidate = [statistics.mean(r['timing']['median_ns_per_call'] for r in runs if r['mode'] == mode) for mode in [0, 1]]
            summaries.append(dict(tokens=tokens, baseline_ns_per_call=baseline, reciprocal_ns_per_call=candidate,
                                  baseline_over_reciprocal=baseline / candidate, component_only=True))
        assert summaries == result['summaries']
        manager.validate_current()
        guard.assert_idle()
        audit = dict(passed=True, finished=time.time(), source_sha256=sha256(__file__),
            input_sha256={str(p): sha256(p) for p in [setup_dir / 'result.json', run_dir / 'result.json', run_dir / 'correctness.json']},
            selected_flash_pid=current['pid'], selected_flash_preserved=True, official_reference_cpu=True,
            original_vocabulary=129280, compressed_vocabulary=99092, token_map_sha256=correctness['token_map_sha256'],
            exact_hash_values_compared=values, sequence_cases=120, reciprocal_arithmetic_cases=316946,
            rejected_configurations=11, rejected_input_calls=16, component_benchmark_arms=12, summaries=summaries,
            full_model_loaded=False, model_tok_s_measured=False, imc_bandwidth_measured=False, runtime_promoted=False,
            limits=['Native hash and state management component only; model graph and native gather integration remain pending.',
                'The official reference and versioned tokenizer supply map, prime, and RNG semantics; native row IDs compare exactly.',
                'The independent integer-remainder checks cover unsigned boundaries beyond the model product range.',
                'Timing uses one CPU and records host load; component ratios do not multiply model throughput.'])
        atomic_json(OUT, audit)
        print(json.dumps(dict(passed=True, exact_hash_values=values, timing_arms=12, selected_flash_pid=current['pid'])))


if __name__ == '__main__':
    os.umask(0o077)
    main()
