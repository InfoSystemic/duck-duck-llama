#!/usr/bin/env python3
"""Record CPU bring-up readiness and verify the decoder's caller-state contract."""
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time

from model_measurement_guard import ModelMeasurementGuard
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import inference_snapshot, sha256
from select_flash_q4_0910c import Manager

BASE = Path(__file__).resolve().parent
INTAKE = BASE / 'results/deepseek-v41-intake-0910'
OUT = BASE / 'results/deepseek-v41-bringup-0910'
ENGINE = BASE.parents[1] / 'engines/llama.cpp-q4e-goal-0904'


def main():
    assert os.sched_getaffinity(0) == {127} and not OUT.exists()
    paths = [BASE / 'results/deepseek-v41-tensor-audit-0910/result.json',
             BASE / 'results/deepseek-v41-native-cpu-0910/result.json']
    audit, probe = [json.loads(p.read_text()) for p in paths]
    assert audit['passed'] and probe['passed'] and probe['peer_preserved']
    manager = Manager()
    peer = manager.validate_current()
    guard = ModelMeasurementGuard(peer['pid'], {peer['pid']: 18131}, inference_snapshot)
    cpp = BASE / 'check-deepseek-v41-engram-env-0910.cpp'
    lib = BASE / 'results/deepseek-v41-native-cpu-0910/libdeepseek-v41-engram.so'
    assert sha256(lib) == probe['library_sha256']
    config = json.loads((INTAKE / 'official/config.json').read_text())
    result = dict(started=time.time(), passed=False, model_id='deepseek-ai/DeepSeek-V4.1-Flash',
                  revision=audit['revision'], input_sha256={str(p): sha256(p) for p in
                      [Path(__file__), cpp, lib, *paths, INTAKE / 'official/config.json']})
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guard.assert_idle()
        OUT.mkdir()
        try:
            binary = OUT / 'check-engram-env'
            command = ['c++', '-O2', '-std=c++17', str(cpp), '-L' + str(lib.parent),
                       '-Wl,-rpath,' + str(lib.parent), '-ldeepseek-v41-engram', '-o', str(binary)]
            compiled = subprocess.run(command, capture_output=True, text=True, timeout=45)
            (OUT / 'compile.log').write_text(compiled.stdout + compiled.stderr)
            assert compiled.returncode == 0
            guard.assert_idle()
            checked = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30)
            (OUT / 'check.log').write_text(checked.stdout + checked.stderr)
            assert checked.returncode == 0
            result['caller_environment_check'] = json.loads(checked.stdout)
            result['check_binary_sha256'] = sha256(binary)
            result['compile_command'] = command
            support = []
            for label, root in [('local_qwen_engine', ENGINE), ('upstream_snapshot', INTAKE / 'upstream')]:
                entries = []
                for relative in ['src/llama-arch.cpp', 'conversion/deepseek.py']:
                    source = root / relative
                    text = source.read_text()
                    names = sorted(set(re.findall(r'DeepseekV41\w*|deepseek_v41\w*|deepseek41\w*', text)))
                    entries.append(dict(file=str(source), sha256=sha256(source), v41_identifiers=names))
                support.append(dict(runtime=label, registration_present=any(v['v41_identifiers'] for v in entries), sources=entries))
            assert all(not v['registration_present'] for v in support)
            text_config = config['text_config']
            table_elements = sum(text_config['engram_num_embeddings']) * text_config['engram_head_dim']
            native_tables = audit['families']['engram_tables']['bytes']
            bf16_tables = table_elements * 2
            lookups = len(text_config['engram_layer_ids']) * (text_config['engram_max_ngram_size'] - 1) * text_config['engram_n_heads']
            lookup_bytes = lookups * (text_config['engram_head_dim'] + text_config['engram_head_dim'] // 32)
            storage = []
            for path in ['/home/kwebb', '/models']:
                stat = os.statvfs(path)
                storage.append(dict(path=path, available_bytes=stat.f_bavail * stat.f_frsize,
                                    holds_native_checkpoint_with_32gib_reserve=stat.f_bavail * stat.f_frsize >= audit['shard_bytes'] + 32 * 2**30))
            mem = {line.split(':')[0]: int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:', 'MemAvailable:', 'Shmem:'))}
            result.update(
                passed=True, source_and_sample_audit_passed=True, architecture=config['architectures'][0],
                current_runtime_support=support, upstream_commit='4ea6d1bb6dac161f70be728983e2cd58e4d9246f',
                checkpoint_bytes=audit['shard_bytes'], native_engram_table_bytes=native_tables,
                bf16_engram_table_bytes=bf16_tables, native_table_bytes_saved_vs_bf16=bf16_tables-native_tables,
                engram_rows_per_text_token=lookups, engram_table_payload_bytes_per_text_token=lookup_bytes,
                lookup_traffic_scope='One use of each selected row and scale; excludes cache lines, page faults, projections, and inter-socket transfers.',
                routed_expert_native_bytes_per_decode_token=audit['families']['routed_experts']['bytes'] * text_config['num_experts_per_tok'] / text_config['n_routed_experts'],
                routed_expert_traffic_scope='Expert weight and scale payload only; not total bytes per token or measured bandwidth.',
                storage=storage, memory=mem, component_comparison=probe['component_comparison'],
                intended_precision='Preserve publisher FP4 experts and FP8 Engram tables. Audit dense conversion separately. No target quant selected or downloaded.',
                runtime_work_remaining=[
                    'V4.1 architecture registration, nested config and tensor-name conversion',
                    'CED/CSA2 cross-layer KV ownership, reuse and hierarchical index candidates',
                    'Single-pass mHC coefficient handoff across sublayers',
                    'Exact compressed-token map and n-gram state, Engram graph/gather integration',
                    'Text/vision routing bias and native vision path',
                    'Three-block DSpark draft, verification and cache/state lifecycle',
                    'Whole-server memory/load plan, native-precision correctness, then prose/code IMC baselines'],
                tuning_targets=dict(adjusted_decode_gb_s=250, server_capacity_denominator_gb_s=380,
                                    model_tok_s=None, qwen_q6_generated_tok_s=40),
                simultaneous_model_residency_required=False, model_loaded=False,
                model_tok_s_measured=False, imc_bandwidth_measured=False, runtime_promoted=False,
                selected_flash_pid=peer['pid'], selected_flash_quant=peer['quant'])
            assert all(sha256(p) == value for p, value in result['input_sha256'].items())
            manager.validate_current()
            result['peer_preserved'] = True
        except BaseException as error:
            result['error'] = repr(error)
            raise
        finally:
            result['finished'] = time.time()
            atomic_json(OUT / 'result.json', result)
    print(json.dumps({k: result[k] for k in ['passed', 'caller_environment_check', 'checkpoint_bytes',
        'native_table_bytes_saved_vs_bf16', 'engram_rows_per_text_token', 'storage', 'peer_preserved']}))


if __name__ == '__main__':
    os.umask(0o077)
    main()
