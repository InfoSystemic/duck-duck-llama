#!/usr/bin/env python3
"""Audit real generation, the retained endpoint, and the native cache inventory."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import process_info,sha256

BASE=Path(__file__).resolve().parent
OUT=BASE/'results/deepseek-v41-real-run-audit-0910.json'


def main():
    assert os.sched_getaffinity(0)=={127} and not OUT.exists()
    files={name:BASE/'results'/path for name,path in {
        'run':'deepseek-v41-checkpoint-run-0910b/generation.json',
        'controller':'deepseek-v41-checkpoint-run-0910b/result.json',
        'shape':'deepseek-v41-checkpoint-run-0910b/shape-check.json',
        'transport':'deepseek-v41-checkpoint-run-0910b/transport.json',
        'kernel':'deepseek-v41-native-gemm-0910b/kernel-check.json',
        'decode':'deepseek-v41-native-gemm-0910b/decode-check.json',
        'server':'deepseek-v41-server-0910/result.json',
        'cache_check':'deepseek-v41-server-0910/cache-check.json',
        'greeting':'deepseek-v41-server-greeting-0910/result.json'}.items()}
    records={k:json.loads(p.read_text()) for k,p in files.items()}
    assert all(r['passed'] for k,r in records.items() if k!='transport')
    run=records['run'];a,b=run['runs'];tokens=[19923,3,1730,588,342,1694,440,4316,33,1]
    assert a['token_ids']==b['token_ids']==tokens and a['reached_eos'] and b['reached_eos']
    assert [s['logits_sha256'] for s in a['steps']]==[s['logits_sha256'] for s in b['steps']]
    assert all(s['finite_logits'] for r in run['runs'] for s in r['steps'])
    assert all(s['downloaded_bytes']==0 for s in b['steps'])
    expected=json.loads((BASE/'results/deepseek-v41-intake-0910/official/inference/config.json').read_text())
    changed={'n_mtp_layers','dspark_block_size','dspark_target_layer_ids','vision_n_layers'}
    actual=records['shape']['args']
    assert all(actual[k]==v for k,v in expected.items() if k not in changed)
    assert actual['dim']==5120 and actual['n_layers']==40 and actual['n_routed_experts']==384
    assert records['shape']['parameters']==93338
    selected=json.loads((BASE/'deepseek-v41-selected.json').read_text());pid=selected['pid'];info=process_info(pid)
    assert info['start']==selected['start'] and info['affinity']==list(range(48,64))
    assert all(sha256(p)==h for p,h in selected['source_sha256'].items())
    with urllib.request.urlopen('http://127.0.0.1:18170/health',timeout=5) as response:health=json.load(response)
    assert health['status']=='ok' and not health['busy']
    lock_path=BASE/'results/qwen-q6-trial-0907/lifecycle.lock'
    fd=int(info['command'][info['command'].index('--lifecycle-lock-fd')+1])
    assert Path(f'/proc/{pid}/fd/{fd}').resolve()==lock_path
    with lock_path.open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:held=True
        else:held=False;fcntl.flock(lock,fcntl.LOCK_UN)
    assert held
    cache=Path(selected['temporary_ram_cache']);manifest=cache/'tensors.jsonl'
    before=manifest.read_bytes();tensors=[json.loads(l) for l in before.splitlines()]
    assert len({r['name'] for r in tensors})==len(tensors)
    assert all(r['revision']==run['revision'] and not r['whole_shard_sha_verified'] for r in tensors)
    for r in tensors:
        p=cache/(r['name']+'.bin');assert p.is_file() and not p.is_symlink() and p.stat().st_size==r['bytes']
    assert manifest.read_bytes()==before
    decode_seconds=sum(s['seconds_including_downloads'] for s in b['steps'][1:])
    result=dict(passed=True,finished=time.time(),revision=run['revision'],real_released_text_backbone_executed=True,
        full_checkpoint_resident=False,all_declared_text_shapes_match_checkpoint=True,layers=40,dim=5120,
        routed_experts_per_layer=384,selected_experts_per_token=6,parameter_entries=93338,
        native_checkpoint_precision_preserved=True,greedy_tokens=tokens,complete_response='Hello! How can I help you today?',
        repeated_logits_exact=True,initial_warm_decode_tok_s=9/decode_seconds,initial_warm_decode_tokens=9,
        initial_warm_decode_seconds=decode_seconds,initial_warm_prefill_seconds=b['steps'][0]['seconds_including_downloads'],
        initial_cold_prefill_seconds=a['steps'][0]['seconds_including_downloads'],
        current_endpoint_warm_decode_tok_s=records['greeting']['warm_decode_tok_s'],
        current_endpoint_warm_response=records['greeting']['runs'][1]['response'],
        server_pid=pid,server_start=info['start'],endpoint=selected['endpoint'],model='DeepSeek-V4.1-Flash',
        physical_workers=16,affinity=info['affinity'],context=256,vision_enabled=False,dspark_enabled=False,
        server_lifecycle_lock_verified=True,server_healthy=True,api_json_and_streaming_checked=True,
        cached_tensor_count=len(tensors),cached_tensor_bytes=sum(r['bytes'] for r in tensors),
        cached_manifest_sha256=hashlib.sha256(before).hexdigest(),whole_shard_lfs_hashes_verified=False,
        cpu_kernels_exact_matrix_values=records['kernel']['exact_values'],fp8_finite_codes_exact=254,fp8_nan_codes_checked=2,
        cache_eviction_cases=records['cache_check']['eviction_cases'],incremental_decode_prefixes=records['cache_check']['incremental_decode_prefixes'],
        imc_bandwidth_measured=False,whole_server_tuning_complete=False,broad_quality_evaluation=False,
        input_sha256={str(p):sha256(p) for p in [Path(__file__),BASE/'deepseek-v41-selected.json',*files.values()]},
        scope='Verified real text generation and a running local endpoint with temporary native-weight caches. '
              'The short warm greeting measures the initial CPU implementation, not full-server capability. '
              'Checkpoint range integrity, finite logits, and exact repeated CPU logits do not establish GPU-reference or broad task-quality equivalence.')
    atomic_json(OUT,result)
    print(json.dumps({k:result[k] for k in ['passed','server_pid','current_endpoint_warm_decode_tok_s','cached_tensor_bytes','repeated_logits_exact']}),flush=True)


if __name__=='__main__':main()
