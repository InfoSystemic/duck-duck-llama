#!/usr/bin/env python3
"""Check generated merge code and cached continuations on the loaded Q6 model."""
import argparse
import ast
import http.client
import json
from pathlib import Path
import re
import socket
import subprocess
import sys
import threading
import time

from model_measurement_guard import ModelMeasurementGuard
from qwen_split_trial import inference_snapshot, process_info

BASE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('measurement')
    parser.add_argument('--diagnose', action='store_true')
    parser.add_argument('--runtime-only', action='store_true',
                        help='Collect live correctness responses without requiring a speed measurement')
    parser.add_argument('--default-cache-off', action='store_true',
                        help='Verify requests omitting cache_prompt on a --no-cache-prompt runtime')
    args = parser.parse_args()
    assert re.fullmatch(r'[A-Za-z0-9_-]+', args.measurement)
    current = json.loads((BASE / 'results/qwen-q6-trial-0907/state.json').read_text())['current']
    source = BASE / 'results' / args.measurement / 'result.json'
    measurement = None if args.runtime_only else json.loads(source.read_text())
    pid = current['pid'] if args.runtime_only else measurement['target_pid']
    assert pid == current['pid'] and not current['original']
    assert process_info(pid)['start'] == current['info']['start']
    if args.default_cache_off:
        assert '--no-cache-prompt' in current['command']
    if measurement:
        assert all(check['pass_check'] for check in measurement['checks'])
    out = BASE / 'results' / (args.measurement + '-quality')
    out.mkdir(exist_ok=False)
    result = dict(started=time.time(), source=None if args.runtime_only else str(source),
                  current=current, cache_checks=[], passed=False, default_cache_off=args.default_cache_off,
                  scope='Runtime correctness checks; not a near-lossless quality certification')
    def save():
        (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    guard = ModelMeasurementGuard(pid, {pid: 18095}, inference_snapshot)

    def request(endpoint, payload):
        guard.assert_idle()
        guard.reset_activity()
        connection = http.client.HTTPConnection('127.0.0.1', 18095, timeout=180)
        connection.connect()
        own_socket = connection.sock
        stop = threading.Event()
        failures = []
        def monitor():
            while not stop.wait(0.5):
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
        if args.runtime_only:
            result['live_checks'] = []
            for prompt, expected in [('What is 17 * 23? Reply with only the number.', '391'),
                                     ('Name the capital city of France. Reply with one word.', 'Paris')]:
                response = request('v1/chat/completions', dict(model='qwen-q6-trial',
                    messages=[dict(role='user', content=prompt)], temperature=0, seed=42,
                    cache_prompt=False, max_tokens=16, chat_template_kwargs={'enable_thinking': False}))
                answer = response['choices'][0]['message']['content']
                passed = answer.strip().rstrip('.') == expected
                result['live_checks'].append(dict(prompt=prompt, response=response, passed=passed))
                save()
                assert passed
                print(json.dumps({'live_check': expected, 'passed': passed}), flush=True)
            response = request('v1/chat/completions', dict(model='qwen-q6-trial',
                messages=[dict(role='user', content='Return only a Python function merge_sorted(a, b) that merges two sorted lists without modifying them. Use a Python code block. No explanation.')],
                temperature=0, seed=42, cache_prompt=False, max_tokens=256,
                chat_template_kwargs={'enable_thinking': False}))
            result['live_code_response'] = response
            content = response['choices'][0]['message']['content']
            save()
        else:
            code_entry = next(item for item in measurement['measurements'] if item['kind'] == 'code')
            chunks = json.loads((source.parent / f'code-draft{code_entry["draft_n"]}' / 'chunks.json').read_text())
            content = ''.join(choice.get('delta', {}).get('content') or ''
                              for chunk in chunks for choice in chunk.get('choices', []))
        (out / 'generated-code-answer.md').write_text(content)
        functions = []
        for block in re.findall(r'```(?:python|py)?\s*\n(.*?)```', content, re.S):
            try:
                functions += [node for node in ast.parse(block).body if isinstance(node, ast.FunctionDef)
                              and len(node.args.args) == 2]
            except SyntaxError:
                pass
        assert functions, 'No complete two-argument Python function found in the measured answer'
        function = functions[0]
        function.returns = None
        for arg in function.args.args:
            arg.annotation = None
        assert not function.decorator_list and not function.args.defaults
        for node in ast.walk(function):
            assert not isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal)), 'Unexpected scope/import'
            if isinstance(node, ast.Name):
                assert not node.id.startswith('__')
            if isinstance(node, ast.Attribute):
                assert node.attr in ('append', 'extend', 'copy'), node.attr
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id in ('len', 'range', 'list'), node.func.id
            if isinstance(node, ast.Call):
                assert isinstance(node.func, (ast.Name, ast.Attribute))
        generated = ast.unparse(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])))
        cases = [([], []), ([], [1, 2]), ([1, 2], []), ([1, 3, 5], [2, 4, 6]),
                 ([1, 1, 3], [1, 2, 2]), ([-9, -3, 0], [-8, -2, 4]),
                 ([0], list(range(1, 30))), ([4, 4], [4, 4, 4])]
        script = ('import json,resource\nresource.setrlimit(resource.RLIMIT_CPU,(3,3))\n'
                  'resource.setrlimit(resource.RLIMIT_AS,(268435456,268435456))\n'
                  'namespace={"__builtins__":{"len":len,"range":range,"list":list}}\n'
                  + 'exec(' + repr(generated) + ',namespace)\n'
                  + 'function=namespace[' + repr(function.name) + ']\n'
                  + 'cases=' + repr(cases) + '\n'
                  + 'for left,right in cases:\n a,b=left.copy(),right.copy()\n'
                    ' assert function(a,b)==sorted(left+right)\n assert (a,b)==(left,right)\n'
                    'print(json.dumps({"passed":True,"cases":len(cases)}))\n')
        code_test = subprocess.run([sys.executable, '-I', '-c', script], capture_output=True, text=True, timeout=10)
        result['code_check'] = dict(exit_code=code_test.returncode, output=code_test.stdout,
                                   stderr=code_test.stderr, function=function.name)
        save()
        assert code_test.returncode == 0, 'Generated merge code failed behavioral checks'
        guard.assert_idle()
        formatted = request('apply-template', dict(messages=[dict(role='user', content=
            'Explain how a refrigerator moves heat. Give a detailed explanation in plain English.')],
            chat_template_kwargs={'enable_thinking': False}))['prompt']
        prompt = request('tokenize', dict(content=formatted, add_special=False, parse_special=True))['tokens']
        common = dict(temperature=0, seed=42, return_tokens=True, n_predict=24)
        prime = request('completion', dict(common, prompt=prompt, cache_prompt=False))
        full = prompt + prime['tokens']
        for name, tokens in [('repeat', prompt), ('extend', full), ('trim_one', full[:-1]), ('trim_three', full[:-3])]:
            reference = request('completion', dict(common, prompt=tokens, cache_prompt=False))
            request('completion', dict(common, prompt=prompt, cache_prompt=False))
            cache_setting = {} if args.default_cache_off else {'cache_prompt': True}
            cached = request('completion', dict(common, prompt=tokens, **cache_setting))
            passed = bool(reference['tokens']) and reference['tokens'] == cached['tokens']
            if args.default_cache_off:
                passed = passed and cached['timings']['cache_n'] == 0 and reference['timings']['cache_n'] == 0
            result['cache_checks'].append(dict(name=name, passed=passed, reference=reference, cached=cached,
                                               explicit_cache_request=not args.default_cache_off))
            save()
            print(json.dumps({'cache': name, 'passed': passed}), flush=True)
            if not passed and args.diagnose:
                fresh_probe = request('completion', dict(common, prompt=tokens, cache_prompt=False,
                                                         temperature=-1, n_probs=8))
                request('completion', dict(common, prompt=prompt, cache_prompt=False))
                cached_probe = request('completion', dict(common, prompt=tokens, cache_prompt=True,
                                                          temperature=-1, n_probs=8))
                result.setdefault('cache_diagnostics', []).append(dict(name=name,
                    fresh=fresh_probe, cached=cached_probe,
                    note='Negative temperature keeps greedy selection and exposes softmax probabilities.'))
                # Accepted MTP tokens do not report target probabilities. Force
                # the common prefix and request just the first differing token.
                first = next((i for i, pair in enumerate(zip(reference['tokens'], cached['tokens']))
                              if pair[0] != pair[1]), None)
                if first is not None and name == 'extend':
                    forced = tokens + reference['tokens'][:first]
                    fresh_one = request('completion', dict(common, prompt=forced,
                        cache_prompt=False, temperature=-1, n_predict=1, n_probs=8))
                    primed = request('completion', dict(common, prompt=prompt,
                        cache_prompt=False, n_predict=len(forced) - len(prompt)))
                    expected_prime = forced[len(prompt):]
                    diagnostic = dict(position=first, prime_matches=primed['tokens'] == expected_prime,
                                      fresh=fresh_one)
                    if diagnostic['prime_matches']:
                        diagnostic['cached'] = request('completion', dict(common, prompt=forced,
                            cache_prompt=True, temperature=-1, n_predict=1, n_probs=8))
                    result['cache_diagnostics'][-1]['forced_first_difference'] = diagnostic
                save()
            if not args.diagnose:
                assert passed, ('Cached continuation changed output tokens', name)
        assert all(check['passed'] for check in result['cache_checks']), 'Cached token equality did not pass for every case'
        result['passed'] = True
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        result['finished'] = time.time()
        save()


if __name__ == '__main__':
    main()
