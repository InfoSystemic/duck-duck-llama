#!/usr/bin/env python3
"""Independently verify the matched trial, selected endpoint, and scope of claims."""
import fcntl
import json
import os
from pathlib import Path
import time
import urllib.request
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import process_info, sha256
from select_flash_q4_0910c import Manager

BASE = Path(__file__).resolve().parent


def main():
    assert os.sched_getaffinity(0) == {127}
    out = BASE / 'results/deepseek-v41-resident-audit-0910.json'
    assert not out.exists()
    path = BASE / 'results/deepseek-v41-resident-trial-0910'
    handoff = json.loads((path / 'result.json').read_text())
    trial = json.loads((path / 'trial/result.json').read_text())
    golden = json.loads((BASE / 'results/deepseek-v41-checkpoint-run-0910b/generation.json').read_text())['runs'][0]
    selected = json.loads((BASE / 'deepseek-v41-selected.json').read_text())
    assert handoff['passed'] and handoff['restored'] and trial['passed']
    assert handoff['resident_promoted'] == trial['promote']
    assert all(sha256(p) == h for group in [selected['source_sha256'], selected['restoration_sources'],
        handoff['candidate_source_sha256'], handoff['source_sha256'], trial['source_sha256']] for p, h in group.items())
    assert selected['pid'] == handoff['restored_pid']
    info = process_info(selected['pid'])
    assert info['start'] == selected['start'] and info['affinity'] == list(range(48, 64))
    assert info['command'] == handoff['restored_server']['command']
    command = info['command']
    fd = int(command[command.index('--lifecycle-lock-fd') + 1])
    lock = BASE / 'results/qwen-q6-trial-0907/lifecycle.lock'
    assert os.stat(f'/proc/{selected["pid"]}/fd/{fd}').st_ino == lock.stat().st_ino
    with lock.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise AssertionError('Retained server lost the fleet lock')
    with urllib.request.urlopen('http://127.0.0.1:18170/health', timeout=3) as response:
        health = json.load(response)
    assert health['status'] == 'ok' and not health['busy']
    peer = Manager().validate_current()
    assert peer['pid'] == handoff['peer_pid'] and peer['info']['start'] == handoff['peer_start']
    assert [r['label'] for r in trial['runs']] == ['demand_warmup', 'demand_1', 'resident_warmup', 'resident_1', 'resident_2', 'demand_2']
    for run in trial['runs']:
        assert run['token_ids'] == golden['token_ids']
        assert run['logits_sha256'] == [s['logits_sha256'] for s in golden['steps']]
        assert run['timings']['downloaded_bytes'] == 0 and run['usage']['completion_tokens'] == 10
        if run['measured'] and run['resident']:
            assert run['tensor_loads'] == 0 and run['resident_experts'] > 0
    rates = {}
    for enabled, name in [(False, 'demand'), (True, 'resident')]:
        runs = [r for r in trial['runs'] if r['measured'] and r['resident'] == enabled]
        rates[name] = 18 / sum(r['timings']['decode_seconds'] for r in runs)
        assert rates[name] == trial[name + '_decode_tok_s']
    speedup = rates['resident'] / rates['demand']
    assert speedup == trial['speedup'] and trial['promote'] == (speedup > 1.05)
    api = handoff['validation']
    assert api['timings']['downloaded_bytes'] == 0 and api['usage']['completion_tokens'] == 10
    assert api['choices'][0]['message']['content'] == 'Hello! How can I help you today?'
    cycles = json.loads((BASE / 'results/deepseek-v41-cpu-profile-0910/result.json').read_text())
    wall = json.loads((BASE / 'results/deepseek-v41-wall-profile-0910/trial/result.json').read_text())
    assert cycles['passed'] and cycles['all_samples_inside_decode'] and wall['passed'] and wall['exact_logits_all_runs']
    assert trial['cache_check']['passed'] and len(trial['cache_check']['cases']) == 6
    result = dict(passed=True, audited_at=time.time(), source_sha256=sha256(__file__),
        input_sha256={str(p): sha256(p) for p in [path / 'result.json', path / 'trial/result.json', BASE / 'deepseek-v41-selected.json']},
        endpoint=selected['endpoint'], pid=selected['pid'], start=selected['start'], revision=selected['revision'],
        model=selected['model'], native_precision=True, physical_workers=16, affinity=info['affinity'], context=256,
        vision=False, dspark=False, full_checkpoint_resident=False, temporary_ram_cache=True,
        retained_mappings_promoted=handoff['resident_promoted'], exact_logits_compared=60,
        matched_request_threads=True, demand_decode_tok_s=rates['demand'], resident_decode_tok_s=rates['resident'],
        matched_speedup=speedup, warm_endpoint_decode_tok_s=9 / api['timings']['decode_seconds'],
        decode_tokens_per_request=9, measured_requests_per_configuration=2,
        zero_benchmark_downloads=True, warm_resident_tensor_loads=0, cache_check_cases=6,
        cached_tensor_bytes=api['timings']['cache_bytes'], lifecycle_lock_held=True, selected_flash_preserved=True,
        cycle_profile_samples=cycles['samples'], cycle_profile_is_http_decode=True,
        wall_profile_scope='CLI main thread includes DeviceContext overhead absent from HTTP request threads; inclusive timers overlap.',
        broad_quality_evaluation=False, whole_server_tuning_complete=False, imc_bandwidth_measured=False)
    atomic_json(out, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
