#!/usr/bin/env python3
"""Record or compare greedy Q8 tokens and default-cache-off continuations."""
import argparse
import fcntl
import http.client
import json
import os
from pathlib import Path
import re
import socket
import threading
import time

from glm_flash_q8_trial import Manager, PORT
from model_measurement_guard import ModelMeasurementGuard
from qwen_split_trial import inference_snapshot, sha256

BASE = Path(__file__).resolve().parent
PROMPTS = [
    ('arithmetic', 'What is 17 * 23? Reply with only the number.'),
    ('prose', 'Explain how a refrigerator moves heat. Give a detailed explanation in plain English.'),
    ('code', 'Return only a Python function merge_sorted(a, b) that merges two sorted lists without modifying them. Use a Python code block. No explanation.'),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('label')
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--cache-checks', action='store_true')
    args = parser.parse_args()
    assert re.fullmatch(r'[A-Za-z0-9_-]+', args.label)
    os.umask(0o077)
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = Manager().validate_current()
        guard = ModelMeasurementGuard(current['pid'], {current['pid']: PORT}, inference_snapshot)
        guard.assert_idle()
        out = BASE / 'results' / args.label
        out.mkdir()
        reference = json.loads(args.reference.read_text()) if args.reference else None
        if reference:
            assert reference['passed'] and reference['current']['model_revision'] == current['model_revision']
        result = dict(started=time.time(), passed=False, current=current, responses=[], cache_checks=[],
                      scope='Bounded runtime equivalence checks, not a near-lossless quality certification',
                      source_sha256=sha256(__file__), reference=str(args.reference) if reference else None,
                      reference_sha256=sha256(args.reference) if reference else None)
        (out / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
        def save():
            (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')

        def request(endpoint, payload):
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
                connection.request('POST', '/' + endpoint, json.dumps(payload), {'Content-Type': 'application/json'})
                response = connection.getresponse()
                assert response.status == 200, (endpoint, response.status)
                value = json.load(response)
                assert not failures, failures
                return value
            finally:
                stop.set()
                connection.close()
                thread.join(timeout=5)
                assert not thread.is_alive()

        try:
            common = dict(temperature=0, seed=42, return_tokens=True, n_predict=64, cache_prompt=False)
            for kind, prompt in PROMPTS:
                formatted = request('apply-template', dict(messages=[dict(role='user', content=prompt)],
                                    chat_template_kwargs={'reasoning_effort': 'max'}))['prompt']
                tokens = request('tokenize', dict(content=formatted, add_special=False, parse_special=True))['tokens']
                response = request('completion', dict(common, prompt=tokens))
                assert response['tokens'] and response['timings']['cache_n'] == 0
                if kind != 'arithmetic' or not current['drafts']:
                    assert bool(response['timings'].get('draft_n')) == bool(current['drafts'])
                row = dict(kind=kind, prompt=prompt, prompt_tokens=tokens, response=response)
                if reference:
                    previous = next(x for x in reference['responses'] if x['kind'] == kind)
                    row['tokens_equal'] = previous['prompt_tokens'] == tokens and previous['response']['tokens'] == response['tokens']
                result['responses'].append(row)
                save()
                print(json.dumps(dict(kind=kind, tokens=len(response['tokens']), tokens_equal=row.get('tokens_equal'))), flush=True)
            if args.cache_checks:
                prompt = next(x for x in result['responses'] if x['kind'] == 'prose')['prompt_tokens']
                prime = request('completion', dict(common, prompt=prompt, n_predict=24))
                full = prompt + prime['tokens']
                for name, tokens in [('repeat', prompt), ('extend', full), ('trim_one', full[:-1]), ('trim_three', full[:-3])]:
                    fresh = request('completion', dict(common, prompt=tokens, n_predict=16))
                    request('completion', dict(common, prompt=prompt, n_predict=24))
                    default = dict(common, prompt=tokens, n_predict=16)
                    default.pop('cache_prompt')
                    repeated = request('completion', default)
                    passed = bool(fresh['tokens']) and fresh['tokens'] == repeated['tokens'] and repeated['timings']['cache_n'] == 0
                    result['cache_checks'].append(dict(name=name, passed=passed, fresh=fresh, default=repeated))
                    save()
                    print(json.dumps(dict(cache=name, passed=passed)), flush=True)
            assert all(x.get('tokens_equal', True) for x in result['responses']), 'Greedy token comparison changed'
            assert all(x['passed'] for x in result['cache_checks']), 'Default-off continuation check failed'
            result['passed'] = True
        except BaseException as error:
            result['error'] = repr(error)
            raise
        finally:
            result['finished'] = time.time()
            save()


if __name__ == '__main__':
    main()
