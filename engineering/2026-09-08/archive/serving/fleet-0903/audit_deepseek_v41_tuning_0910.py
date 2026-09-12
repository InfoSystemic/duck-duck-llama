#!/usr/bin/env python3
"""Audit the selected live endpoint under the controller lock; optional publication hook."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

from qwen_high_quant_trial import atomic_json
from qwen_split_trial import process_info, sha256, inference_snapshot
from select_flash_q4_0910c import Manager
from model_measurement_guard import ModelMeasurementGuard

BASE = Path(__file__).resolve().parent
SELECTED = BASE / 'deepseek-v41-selected.json'
OUT = BASE / 'results/deepseek-v41-tuning-audit-0910.json'


def health():
    with urllib.request.urlopen('http://127.0.0.1:18170/health', timeout=3) as response:
        value = json.load(response)
    assert value['status'] == 'ok' and value['model'] == 'DeepSeek-V4.1-Flash' and not value['busy']
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--publication-script', type=Path)
    args = parser.parse_args()
    assert os.sched_getaffinity(0) == {127} and not OUT.exists()
    publication = args.publication_script
    if publication is not None:
        publication = publication.resolve()
        assert publication == Path('/tmp/publish_deepseek_tuning_0910.py') and publication.is_file()
    with (BASE / 'results/deepseek-v41-controller.lock').open('a') as lock:
        print(json.dumps(dict(waiting_for_deepseek_audit_lock=True)), flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        selected = json.loads(SELECTED.read_text())
        selected_hash = sha256(SELECTED)
        assert all('.private.' not in name for name in selected['source_sha256'])
        assert all(sha256(p) == h for p, h in selected['source_sha256'].items())
        info = process_info(selected['pid'])
        assert info['start'] == selected['start']
        peer = Manager().validate_current()
        guard = ModelMeasurementGuard(peer['pid'], {peer['pid']:18131}, inference_snapshot)
        guard.assert_idle(); health()
        request_record = Path(selected['request_record'])
        completed_before = json.loads(request_record.read_text())['completed']
        result = dict(passed=False, started=time.time(), pid=selected['pid'], start=selected['start'],
                      endpoint=selected['endpoint'], model=selected['model'], revision=selected['revision'],
                      selected_manifest_sha256=selected_hash, source_sha256=selected['source_sha256'],
                      audit_source_sha256=sha256(__file__), cpu_optimizations=selected.get('cpu_optimizations', []),
                      native_precision=selected['native_precision'], context=selected['context'],
                      vision=selected['vision'], dspark=selected['dspark'], affinity=info['affinity'],
                      peer_pid=peer['pid'], peer_start=peer['info']['start'], requests=[])
        try:
            for index in range(2):
                request = urllib.request.Request('http://127.0.0.1:18170/v1/chat/completions',
                    data=json.dumps(dict(model=selected['model'], messages=[dict(role='user', content='Hi.')],
                                         temperature=0, max_tokens=16)).encode(), headers={'Content-Type':'application/json'})
                with urllib.request.urlopen(request, timeout=180) as response:
                    reply = json.load(response)
                assert reply['choices'][0]['message']['content'] == 'Hello! How can I help you today?'
                assert reply['choices'][0]['finish_reason'] == 'stop' and reply['usage']['completion_tokens'] == 10
                assert reply['timings']['downloaded_bytes'] == 0
                row = dict(index=index, usage=reply['usage'], timings=reply['timings'],
                           decode_tok_s=9 / reply['timings']['decode_seconds'])
                result['requests'].append(row)
                print(json.dumps(dict(endpoint_decode_tok_s=row['decode_tok_s'])), flush=True)
                guard.assert_idle(); health()
            assert json.loads(request_record.read_text())['completed'] == completed_before + 2
            assert sha256(SELECTED) == selected_hash
            assert process_info(selected['pid'])['start'] == selected['start']
            assert process_info(peer['pid'])['start'] == peer['info']['start']
            assert all(sha256(p) == h for p, h in selected['source_sha256'].items())
            result.update(passed=True, selected_preserved=True, peer_preserved=True,
                          warm_decode_tok_s=18 / sum(r['timings']['decode_seconds'] for r in result['requests']),
                          bandwidth_measured=False, broad_quality_measured=False, full_checkpoint_resident=False)
        except BaseException as error:
            result['error'] = type(error).__name__
            raise
        finally:
            result['finished'] = time.time()
            atomic_json(OUT, result)
        if publication is not None:
            subprocess.run(['python3', str(publication)], check=True)
        health()
        assert process_info(selected['pid'])['start'] == selected['start']
        print(json.dumps(dict(passed=True, selected_endpoint_healthy=True, pid=selected['pid'])), flush=True)


if __name__ == '__main__':
    main()
