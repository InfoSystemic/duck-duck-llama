#!/usr/bin/env python3
"""Measure the same lowercase app-server model selection used by Paseo.

This is a serial apply_patch integration probe. It launches the installed Paseo Codex
wrapper, leaves its normal home/skills/context settings intact, and records timing
and token counters. It does not create a Paseo managed agent or alter services.
"""
import argparse
import collections
import datetime
import http.client
import hashlib
import http.server
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
import urllib.request
import urllib.parse

from summarize_metrics import summarize

ROOT = Path(__file__).resolve().parent
PROMPT = 'This is a bounded coding integration test. Use apply_patch to create the single file /home/user/sr950-strategy/codex-bench-0919/tool-smoke-0919/lru_cache.py. Do not run shell commands or read other files. Implement a Python class LRUCache(capacity) with get(key) returning the stored value or -1 on miss, and put(key,value) updating or inserting an entry. Both operations must be O(1). A successful get and every put mark an entry most recently used. Evict the least recently used entry when capacity is exceeded. Capacity zero stores nothing; negative capacity raises ValueError. Store None and -1 as normal values. Use collections.OrderedDict with move_to_end and popitem(last=False) to guarantee constant-time recency and eviction; avoid pop(next(iter(dict))). After the file is written, finish with the exact marker GLM_CODEX_TOOL_OK.'


def metrics(endpoint):
    text = urllib.request.urlopen(endpoint + '/metrics', timeout=10).read().decode()
    return {line.rsplit(None, 1)[0]: float(line.rsplit(None, 1)[1]) for line in text.splitlines()
            if line and not line.startswith('#')}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--catalog', type=Path, required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--endpoint', default='http://127.0.0.1:18131')
    ap.add_argument('--timeout', type=float, default=900)
    ap.add_argument('--warm', action='store_true', help='Allow prompt cache; default forces cold processing')
    ap.add_argument('--target-name', default='lru_cache', help='Unique Python filename inside tool-smoke-0919')
    ap.add_argument('--approval-file', type=Path, help='Wait for a reviewed decision tied to the exact pending request hash')
    args = ap.parse_args()
    if args.approval_file and args.approval_file.exists(): raise RuntimeError('Approval file must initially be absent')
    if not args.target_name.isidentifier(): raise ValueError('target-name must be a Python identifier')
    global PROMPT
    PROMPT = PROMPT.replace('tool-smoke-0919/lru_cache.py', 'tool-smoke-0919/'+args.target_name+'.py')
    if (ROOT/'tool-smoke-0919'/(args.target_name+'.py')).exists():
        raise RuntimeError('Preserve the existing smoke-test file; choose a fresh target-name')
    slots = json.load(urllib.request.urlopen(args.endpoint + '/slots', timeout=10))
    if any(s.get('is_processing') for s in slots):
        raise RuntimeError('Another request is active; run the benchmark serially')
    result = {'tag': args.tag, 'catalog': str(args.catalog.resolve()),
              'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'prompt': PROMPT, 'rpc_model': 'glm-5.3-flash', 'cold_prompt': not args.warm}
    endpoint = urllib.parse.urlsplit(args.endpoint)
    class ColdRequestProxy(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            body['cache_prompt'] = bool(args.warm)
            result['request_instructions_chars'] = len(body.get('instructions', ''))
            result['request_reasoning'] = body.get('reasoning')
            result['request_input_chars'] = len(json.dumps(body.get('input')))
            connection = http.client.HTTPConnection(endpoint.hostname, endpoint.port, timeout=args.timeout)
            try:
                connection.request('POST', self.path, json.dumps(body).encode(), {'Content-Type': 'application/json'})
                upstream = connection.getresponse()
                self.send_response(upstream.status)
                self.send_header('Content-Type', upstream.getheader('Content-Type', 'text/event-stream'))
                self.send_header('Connection', 'close')
                self.end_headers()
                while chunk := upstream.read1(65536):
                    self.wfile.write(chunk)
                    self.wfile.flush()
            finally:
                connection.close()
    proxy = http.server.ThreadingHTTPServer(('127.0.0.1', 0), ColdRequestProxy)
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    cmd = ['/home/user/.local/libexec/paseo/codex-smeagol',
           '-c', 'model_catalog_json=' + json.dumps(str(args.catalog.resolve())),
           '-c', 'model_providers.smeagol.base_url=' + json.dumps(f'http://127.0.0.1:{proxy.server_port}/v1'),
           'app-server', '--enable', 'goals']
    with open(ROOT / f'appserver-{args.tag}.stderr.log', 'w') as err:
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=err, text=True, bufsize=1, cwd=ROOT)
        incoming = queue.Queue()
        def reader():
            for line in p.stdout:
                try:
                    incoming.put(json.loads(line))
                except json.JSONDecodeError:
                    pass
            incoming.put({'method': '_process_exit'})
        threading.Thread(target=reader, daemon=True).start()
        request_id = 0
        events = collections.Counter()
        def send(method, params, notification=False):
            nonlocal request_id
            request_id += 1
            frame = {'method': method, 'params': params}
            if not notification:
                frame['id'] = request_id
            p.stdin.write(json.dumps(frame) + '\n'); p.stdin.flush()
            return request_id
        def response(ident):
            deadline = time.monotonic() + 30
            while True:
                msg = incoming.get(timeout=max(.1, deadline - time.monotonic()))
                if msg.get('id') == ident:
                    if 'error' in msg:
                        raise RuntimeError(msg['error'])
                    return msg.get('result')
                if msg.get('method') == '_process_exit':
                    raise RuntimeError('app-server exited')
        try:
            response(send('initialize', {'clientInfo': {'name': 'glm-paseo-bench', 'version': '1'},
                                        'capabilities': {'experimentalApi': True}}))
            send('initialized', {}, notification=True)
            thread = response(send('thread/start', {'model': 'glm-5.3-flash', 'cwd': str(ROOT),
                                'ephemeral': True, 'approvalPolicy': 'on-request', 'sandbox': 'workspace-write'}))
            result['thread_start_model'] = thread.get('model')
            result['thread_start_reasoning_effort'] = thread.get('reasoningEffort')
            before = metrics(args.endpoint)
            started = time.monotonic(); last_report = started
            ident = send('turn/start', {'threadId': thread['thread']['id'],
                         'input': [{'type': 'text', 'text': PROMPT}], 'model': 'glm-5.3-flash',
                         'effort': 'low', 'collaborationMode': {'mode': 'default', 'settings': {'model': 'glm-5.3-flash'}}})
            text = []
            pending_approval = None
            while True:
                now = time.monotonic()
                if pending_approval and args.approval_file and args.approval_file.exists():
                    decision=json.loads(args.approval_file.read_text())
                    assert decision['request_id']==pending_approval['id']
                    assert decision['pending_sha256']==pending_approval['sha256']
                    assert decision['decision'] in ('accept','decline','cancel')
                    p.stdin.write(json.dumps({'id':pending_approval['id'],'result':{'decision':decision['decision']}})+'\n')
                    p.stdin.flush()
                    result.setdefault('reviewed_decisions',[]).append(decision)
                    args.approval_file.unlink()
                    pending_approval=None
                if now - started > args.timeout:
                    raise TimeoutError('inference probe exceeded deadline')
                if now - last_report > 30:
                    print(f'{args.tag}: active {now-started:.0f}s, model delta received={"first_model_delta_seconds" in result}', flush=True)
                    last_report = now
                try:
                    msg = incoming.get(timeout=1)
                except queue.Empty:
                    continue
                method = msg.get('method', '')
                events[method] += 1
                params = msg.get('params', {})
                if msg.get('id') == ident and 'error' in msg:
                    raise RuntimeError(msg['error'])
                if method == '_process_exit':
                    raise RuntimeError('app-server exited during turn')
                if method.endswith('/delta') and ('reasoning' in method.lower() or 'agentMessage' in method):
                    result.setdefault('first_model_delta_seconds', time.monotonic()-started)
                if method == 'item/agentMessage/delta':
                    result.setdefault('first_visible_delta_seconds', time.monotonic()-started)
                    text.append(params.get('delta', ''))
                if method == 'thread/tokenUsage/updated':
                    result['token_usage'] = params.get('tokenUsage')
                if method == 'item/started':
                    item = params.get('item', {})
                    if item.get('type')=='fileChange':
                        result.setdefault('proposed_file_changes',{})[item['id']]=item.get('changes',[])
                    if item.get('type') in ('commandExecution', 'mcpToolCall', 'webSearch'):
                        raise RuntimeError('Coding probe used a disallowed tool')
                if method == 'item/fileChange/patchUpdated':
                    result.setdefault('proposed_file_changes',{})[params['itemId']]=params['changes']
                if method == 'item/completed' and params.get('item', {}).get('type') == 'fileChange':
                    result.setdefault('file_changes', []).append(params['item'])
                if method.endswith('/requestApproval'):
                    if method!='item/fileChange/requestApproval' or not args.approval_file:
                        result.setdefault('approval_requests', []).append({'method': method, 'params': params})
                        raise RuntimeError('Unexpected permission request; stop for explicit review')
                    pending={'id':msg['id'],'method':method,'params':params,
                             'changes':result.get('proposed_file_changes',{}).get(params['itemId'],[])}
                    pending_bytes=(json.dumps(pending,indent=2)+'\n').encode()
                    pending_path=ROOT/f'appserver-{args.tag}.pending-approval.json'
                    pending_path.write_bytes(pending_bytes)
                    pending_approval={'id':msg['id'],'sha256':hashlib.sha256(pending_bytes).hexdigest()}
                    result.setdefault('approval_requests',[]).append(pending)
                    print('Pending exact-edit review:',pending_path,flush=True)
                if method == 'turn/completed':
                    result['wall_seconds'] = time.monotonic()-started
                    result['turn_status'] = params.get('turn', {}).get('status')
                    result['turn_error'] = params.get('turn', {}).get('error')
                    break
            result['server_metrics'] = summarize(before, metrics(args.endpoint))
            result['output_text'] = ''.join(text)
            result['marker_pass'] = result['output_text'].rstrip().endswith('GLM_CODEX_TOOL_OK')
            result['event_counts'] = dict(events)
            if result['turn_status'] != 'completed' or not result['marker_pass'] or not result.get('file_changes'):
                raise RuntimeError('Benchmark response did not complete with expected marker')
        except Exception as exc:
            result['error'] = repr(exc)
            raise
        finally:
            (ROOT / f'appserver-{args.tag}.json').write_text(json.dumps(result, indent=2) + '\n')
            p.terminate()
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill(); p.wait(timeout=10)
            proxy.shutdown()
            proxy.server_close()
    print(json.dumps({k: v for k, v in result.items() if k not in ('output_text', 'prompt', 'event_counts')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
