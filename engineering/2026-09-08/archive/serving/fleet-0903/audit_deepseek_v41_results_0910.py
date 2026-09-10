#!/usr/bin/env python3
"""Reconcile completed DeepSeek component evidence before source publication."""
import fcntl
import json
import os
from pathlib import Path
import statistics
import time

from qwen_split_trial import sha256
from select_flash_q4_0910c import Manager

BASE = Path(__file__).resolve().parent
OUT = BASE / 'results/deepseek-v41-validation-audit-0910.json'


def main():
    assert os.sched_getaffinity(0) == {127} and not OUT.exists()
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        paths = [BASE / 'results' / name / 'result.json' for name in
                 ['deepseek-v41-tensor-audit-0910', 'deepseek-v41-native-cpu-0910', 'deepseek-v41-bringup-0910']]
        records = [json.loads(path.read_text()) for path in paths]
        bound_files = {str(Path(__file__)): sha256(__file__)}
        for path, record in zip(paths, records):
            assert record['passed'] and record['peer_preserved'] and record['finished'] >= record['started']
            assert not record['model_loaded']
            bound_files[str(path)] = sha256(path)
            for name, digest in record['input_sha256'].items():
                assert '.private.' not in name and sha256(name) == digest
                bound_files[name] = digest
        tensors, probe, bringup = records
        payload = 0
        count = 0
        for shard in tensors['shards']:
            path = paths[0].parent / 'headers' / (shard['file'] + '.json')
            assert sha256(path) == shard['header_sha256']
            header = json.loads(path.read_text())
            entries = {name: value for name, value in header.items() if name != '__metadata__'}
            assert len(entries) == shard['tensors']
            payload += sum(value['data_offsets'][1] - value['data_offsets'][0] for value in entries.values())
            count += len(entries)
        assert count == tensors['tensor_count'] == 96085 and payload == tensors['payload_bytes']
        for sample in tensors['samples']:
            assert sha256(sample['file']) == sample['sha256']
        for step in probe['steps']:
            path = paths[1].parent / (step['label'] + '.log')
            assert step['exit_code'] == 0 and sha256(path) == step['log_sha256']
        assert len(probe['runs']) == 8
        comparisons = []
        for rows in [1024, 1048576]:
            runs = [run for run in probe['runs'] if run['table_rows'] == rows]
            assert [run['mode'] for run in runs] == [0, 1, 1, 0]
            assert len({run['checksum'] for run in runs}) == 1
            for run in runs:
                path = paths[1].parent / f'rows{rows}-arm{run["arm"]}-mode{run["mode"]}.log'
                observed = json.loads(path.read_text())
                assert all(run[key] == value for key, value in observed.items())
            scalar = statistics.mean(run['median_us'] for run in runs if run['mode'] == 0)
            vector = statistics.mean(run['median_us'] for run in runs if run['mode'] == 1)
            comparisons.append(dict(table_rows=rows, scalar_mean_median_us=scalar,
                                    avx512_mean_median_us=vector, scalar_over_avx512=scalar/vector))
        state = json.loads((paths[2].parent / 'check.log').read_text())
        assert state == bringup['caller_environment_check'] and state['passed'] and state['environment_cases'] == 16
        assert sha256(paths[1].parent / 'libdeepseek-v41-engram.so') == probe['library_sha256']
        assert sha256(paths[1].parent / 'deepseek-v41-engram-bench') == probe['binary_sha256']
        assert sha256(paths[2].parent / 'check-engram-env') == bringup['check_binary_sha256']
        peer = Manager().validate_current()
        assert peer['pid'] == bringup['selected_flash_pid'] == probe['peer_pid'] == tensors['peer_pid']
        result = dict(passed=True, audited_at=time.time(), input_sha256=bound_files,
            official_model_revision=tensors['revision'], header_count=48, tensor_count=count,
            payload_bytes=payload, partial_weight_samples=len(tensors['samples']),
            exhaustive_fp8_scale_pairs_per_implementation=65536, real_engram_table_rows=128,
            fp4_real_sample_checks=4, fp4_exhaustive_packed_byte_cases=256,
            floating_point_state_cases=state['environment_cases'], unaligned_buffers_checked=True,
            canaries_preserved=True, caller_mxcsr_restored=True,
            benchmark_arms=8, benchmark_output_checksums_match=True, component_comparison=comparisons,
            whole_model_measurement=False, full_weights_downloaded=False, runtime_promoted=False,
            selected_flash_pid=peer['pid'], selected_flash_quant=peer['quant'],
            qwen_40_tok_s_reached=False, all_models_250_gb_s_reached=False)
        OUT.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({k: result[k] for k in ['passed', 'tensor_count', 'partial_weight_samples',
                                               'floating_point_state_cases', 'benchmark_arms', 'component_comparison']}))


if __name__ == '__main__':
    os.umask(0o077)
    main()
