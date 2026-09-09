#!/usr/bin/env python3
"""Inspect the first MTP/raw token divergence with bounded native requests."""
import fcntl
import http.client
import json
from pathlib import Path
import socket
import threading
import time

from glm_flash_q8_trial import Manager, PORT
from model_measurement_guard import ModelMeasurementGuard
from qwen_split_trial import inference_snapshot, sha256

BASE = Path(__file__).resolve().parent
OUT = BASE / 'results/glm-flash-q8-mtp2-divergence-0908'


def main():
    current = Manager().validate_current()
    assert current['drafts'] == 2 and not current['batching']
    raw_path = BASE / 'results/glm-flash-q8-raw-parity-0908/result.json'
    mtp_path = BASE / 'results/glm-flash-q8-mtp2-parity-0908/result.json'
    raw = next(x for x in json.loads(raw_path.read_text())['responses'] if x['kind'] == 'prose')
    mtp = next(x for x in json.loads(mtp_path.read_text())['responses'] if x['kind'] == 'prose')
    first = next(i for i, pair in enumerate(zip(raw['response']['tokens'], mtp['response']['tokens'])) if pair[0] != pair[1])
    assert raw['prompt_tokens'] == mtp['prompt_tokens']
    OUT.mkdir()
    result = dict(started=time.time(), current=current, first_difference=first,
                  raw_token=raw['response']['tokens'][first], mtp_token=mtp['response']['tokens'][first],
                  raw_source_sha256=sha256(raw_path), mtp_source_sha256=sha256(mtp_path), probes=[])
    (OUT / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    guard = ModelMeasurementGuard(current['pid'], {current['pid']: PORT}, inference_snapshot)
    def save():
        (OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    def request(payload):
        guard.assert_idle()
        guard.reset_activity()
        connection = http.client.HTTPConnection('127.0.0.1', PORT, timeout=180)
        connection.connect()
        own_socket = connection.sock
        stop, failures = threading.Event(), []
        def monitor():
            while not stop.wait(.5):
                try:
                    reason = guard.abort_reason()
                except Exception as error:
                    reason = repr(error)
                if reason:
                    failures.append(reason)
                    try:
                        own_socket.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    return
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        try:
            connection.request('POST', '/completion', json.dumps(payload), {'Content-Type': 'application/json'})
            response = connection.getresponse()
            assert response.status == 200, response.status
            value = json.load(response)
            assert not failures, failures
            return value
        finally:
            stop.set()
            connection.close()
            thread.join(timeout=5)
            assert not thread.is_alive()
    try:
        cases = [('mtp_repeat', 0, 64, 0), ('forced_first_difference', first, 1, -1),
                 ('near_prefix_four', first - 2, 4, -1), ('near_prefix_two', first - 1, 2, -1)]
        for name, prefix, count, temperature in cases:
            payload = dict(prompt=raw['prompt_tokens'] + raw['response']['tokens'][:prefix],
                           n_predict=count, temperature=temperature, seed=42, cache_prompt=False,
                           return_tokens=True, n_probs=10)
            response = request(payload)
            row = dict(name=name, prefix=prefix, request=payload, response=response)
            result['probes'].append(row)
            if name == 'mtp_repeat':
                row['matches_previous_mtp'] = response['tokens'] == mtp['response']['tokens']
            if name == 'forced_first_difference':
                row['matches_raw'] = response['tokens'] == [result['raw_token']]
                row['draft_tokens'] = response['timings'].get('draft_n', 0)
            save()
            print(json.dumps(dict(name=name, tokens=response['tokens'][:8],
                                  matches_raw=row.get('matches_raw'), matches_previous_mtp=row.get('matches_previous_mtp'),
                                  draft_tokens=response['timings'].get('draft_n', 0))), flush=True)
        result['collected'] = True
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        result['finished'] = time.time()
        save()


if __name__ == '__main__':
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
