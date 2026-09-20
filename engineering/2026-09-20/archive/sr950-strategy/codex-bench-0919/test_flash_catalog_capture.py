#!/usr/bin/env python3
"""Capture Codex HTTP request metadata only; the stub never runs inference."""
import copy
import hashlib
import http.server
import json
import os
import pathlib
import queue
import select
import shutil
import subprocess
import tempfile
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parent
WRAPPER = '/home/user/.local/libexec/paseo/codex-smeagol'
CAT_BEFORE = ROOT / 'glm-5.3-flash.catalog.before.json'
CAT_AFTER = ROOT / 'glm-5.3-flash.catalog.proposed.json'
captures = queue.Queue()

def schema_shape(value):
    if isinstance(value, dict):
        return {k: schema_shape(v) for k, v in value.items() if k != 'description'}
    if isinstance(value, list):
        return [schema_shape(v) for v in value]
    return value



def tool_labels(tools):
    out = []
    for tool in tools:
        kind = tool.get('type', '')
        name = tool.get('name') or tool.get('function', {}).get('name')
        if kind == 'namespace':
            for child in tool.get('tools', []):
                out.append(name + '.' + (child.get('name') or child.get('type', '')))
        else:
            out.append(name or kind)
    return sorted(out)


class Recorder(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        # Only wrapper health probes can reach production; this stub handles
        # harmless health requests if app-server sends any to its custom URL.
        body = b'{"status":"ok"}'
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get('Content-Length', '0')))
        try:
            request = json.loads(body)
            instructions = request.get('instructions', '')
            if not isinstance(instructions, str):
                instructions = json.dumps(instructions, sort_keys=True)
            tools = request.get('tools', [])
            safe = {
                'path': self.path,
                'model': request.get('model'),
                'instructions_chars': len(instructions),
                'instructions_sha256': hashlib.sha256(instructions.encode()).hexdigest(),
                'tool_names': tool_labels(tools),
                'tools_sha256': hashlib.sha256(json.dumps(tools, sort_keys=True).encode()).hexdigest(),
                'tool_schema_sha256': hashlib.sha256(json.dumps(schema_shape(tools), sort_keys=True).encode()).hexdigest(),
                'reasoning': request.get('reasoning'),
                'stream': request.get('stream'),
                'input_item_count': len(request.get('input', [])),
            }
            captures.put(safe)
        except Exception as exc:
            captures.put({'capture_error': type(exc).__name__})
        body = b'{"error":{"message":"Expected offline capture stop; no inference was run","type":"invalid_request_error","code":"capture_only"}}'
        self.send_response(400)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_case(port, catalog, rpc_model, collaboration_effort):
    with tempfile.TemporaryDirectory(prefix='capture-home-', dir=ROOT) as home:
        shutil.copyfile('/home/user/.codex-glm/config.toml', pathlib.Path(home) / 'config.toml')
        env = dict(os.environ)
        env['CODEX_LOCAL_HOME'] = home
        env.pop('CODEX_GLM_HOME', None)
        cmd = [
            WRAPPER,
            '-c', 'model_catalog_json=' + json.dumps(str(catalog)),
            '-c', f'model_providers.smeagol.base_url="http://127.0.0.1:{port}/v1"',
            '-c', 'model_providers.smeagol.request_max_retries=0',
            '-c', 'model_providers.smeagol.stream_max_retries=0',
            'app-server', '--enable', 'goals',
        ]
        p = subprocess.Popen(cmd, env=env, cwd=ROOT, stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             text=True, bufsize=1)
        responses = queue.Queue()

        def read_responses():
            for line in p.stdout:
                try:
                    responses.put(json.loads(line))
                except json.JSONDecodeError:
                    pass

        reader = threading.Thread(target=read_responses, daemon=True)
        reader.start()
        sequence = 0

        def send(method, params, notification=False):
            nonlocal sequence
            sequence += 1
            frame = {'method': method, 'params': params}
            if not notification:
                frame['id'] = sequence
            p.stdin.write(json.dumps(frame) + '\n')
            p.stdin.flush()
            if notification:
                return None
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                msg = responses.get(timeout=max(0.1, deadline - time.monotonic()))
                if msg.get('id') == sequence:
                    if 'error' in msg:
                        raise RuntimeError(json.dumps(msg['error']))
                    return msg.get('result')
            raise TimeoutError(method)

        try:
            send('initialize', {'clientInfo': {'name': 'glm-local-capture', 'version': '1'},
                                'capabilities': {'experimentalApi': True}})
            send('initialized', {}, notification=True)
            models = send('model/list', {'includeHidden': True})
            thread = send('thread/start', {
                'model': rpc_model, 'cwd': str(ROOT), 'ephemeral': True,
                'approvalPolicy': 'on-request', 'sandbox': 'workspace-write',
            })
            thread_id = thread['thread']['id']
            settings = {'model': rpc_model}
            if collaboration_effort is not None:
                settings['reasoning_effort'] = collaboration_effort
            send('turn/start', {
                'threadId': thread_id,
                'input': [{'type': 'text', 'text': 'Reply only OK. Do not use tools.'}],
                'model': rpc_model, 'effort': 'low',
                'collaborationMode': {'mode': 'default', 'settings': settings},
            })
            result = captures.get(timeout=20)
            result.update({
                'catalog': catalog.name, 'rpc_model': rpc_model,
                'collaboration_effort': collaboration_effort,
                'advertised_models': [
                    {'id': x.get('id'), 'defaultReasoningEffort': x.get('defaultReasoningEffort')}
                    for x in models.get('data', [])
                    if 'glm' in x.get('id', '').lower()
                ],
            })
            return result
        finally:
            p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=5)
            p.stdin.close()
            p.stdout.close()


def main():
    before = json.loads(CAT_BEFORE.read_text())
    after = json.loads(CAT_AFTER.read_text())
    original = before['models'][0]
    assert len(after['models']) == 2
    for model in after['models']:
        expected = copy.deepcopy(original)
        expected['slug'] = model['slug']
        expected['default_reasoning_level'] = 'low'
        assert model == expected
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Recorder)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    results = []
    try:
        for catalog in [CAT_BEFORE, CAT_AFTER]:
            for model in ['GLM-5.3-Flash', 'glm-5.3-flash']:
                result = run_case(server.server_port, catalog, model, None)
                results.append(result)
                print(json.dumps(result), flush=True)
        explicit = run_case(server.server_port, CAT_AFTER, 'glm-5.3-flash', 'low')
        results.append(explicit)
        print(json.dumps(explicit), flush=True)
    finally:
        server.shutdown()
        server.server_close()
    expected_hash = hashlib.sha256(original['base_instructions'].encode()).hexdigest()
    # Actual request contents are the authority; no token generation is involved.
    assert results[0]['instructions_sha256'] == expected_hash
    assert results[1]['instructions_sha256'] != expected_hash
    for result in results[2:]:
        assert result['instructions_sha256'] == expected_hash
        assert result['tool_names'] == results[0]['tool_names']
        assert result['tool_schema_sha256'] == results[0]['tool_schema_sha256']
    report = {
        'status': 'PASS', 'inference_requests': 0,
        'stub_http_requests': len(results), 'capture_metadata_only': True,
        'capability_fields_preserved': True, 'results': results,
    }
    (ROOT / 'flash-catalog-capture-results.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PASS: catalog capabilities preserved; alias selects intended instructions; uppercase tool schema preserved.')


if __name__ == '__main__':
    main()
