#!/usr/bin/env python3
"""Check cancellation of blocked HTTP reads against a temporary loopback server."""
import http.server
import json
import threading
import time
import urllib.request

from guarded_inference_request import stream_completion

ready = threading.Event()
release = threading.Event()


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'healthy')

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        mode = body['mode']
        if mode == 'before_headers':
            ready.set()
            release.wait(5)
            return
        self.send_response(200)
        self.end_headers()
        if mode in ('blocked_line', 'monitor_error'):
            self.wfile.write(b'data: {')
            self.wfile.flush()
            ready.set()
            release.wait(5)
            return
        self.wfile.write(b'data: {"choices":[{"delta":{"content":"391"}}]}\n\n')
        if mode != 'truncated':
            self.wfile.write(b'data: [DONE]\n\n')
        self.wfile.flush()


server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
worker = threading.Thread(target=server.serve_forever, daemon=True)
worker.start()
port = server.server_address[1]
try:
    chunks = stream_completion(port, {'mode': 'complete'}, lambda: None, [], interval=.05)
    assert chunks[0]['choices'][0]['delta']['content'] == '391'
    try:
        stream_completion(port, {'mode': 'truncated'}, lambda: None, [], interval=.05)
    except RuntimeError as error:
        assert 'without [DONE]' in str(error)
    else:
        raise AssertionError('Truncated stream accepted')
    for mode in ('blocked_line', 'before_headers', 'monitor_error'):
        ready.clear()
        release.clear()
        abort = []
        def check_abort():
            if not ready.is_set():
                return None
            if mode == 'monitor_error':
                raise ValueError('synthetic queue monitor failure')
            return dict(reason='Synthetic queued user request')
        started = time.monotonic()
        try:
            stream_completion(port, {'mode': mode}, check_abort, abort, interval=.05)
        except (OSError, RuntimeError, http.client.HTTPException):
            assert abort, 'Failure did not originate from request guard'
        else:
            raise AssertionError('Guarded request was not cancelled')
        assert time.monotonic() - started < 2, 'Cancellation waited for socket timeout'
        release.set()
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2) as response:
            assert response.read() == b'healthy'
    print('PASS: complete SSE, truncated SSE rejection, cancellation before headers and during a blocked line, monitor failure, server remains healthy')
finally:
    release.set()
    server.shutdown()
    server.server_close()
    worker.join(timeout=2)
