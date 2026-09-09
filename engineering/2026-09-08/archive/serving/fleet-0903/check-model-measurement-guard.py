#!/usr/bin/env python3
"""Exercise request cancellation against a disposable local HTTP server."""
import http.server
import json
import threading
import time
import urllib.request

from guarded_inference_request import stream_completion
from model_measurement_guard import ModelMeasurementGuard


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        body = b'{"status":"ok"}'
        self.send_response(200)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Connection', 'close')
        self.end_headers()
        try:
            for _ in range(1 if payload['normal'] else 100):
                self.wfile.write(b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n')
                self.wfile.flush()
                time.sleep(0.02)
            self.wfile.write(b'data: [DONE]\n\n')
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        self.close_connection = True


def run_case(case):
    processes = {'100': (0, 'target-start', 'llama-server'), '200': (0, 'peer-start', 'llama-server')}
    states = {'100': dict(processing=0, queued=0), '200': dict(processing=0, queued=0)}
    failed_reader = False
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.01}, daemon=True)
    thread.start()
    port = server.server_port
    def read(port_to_read):
        if failed_reader:
            raise OSError('simulated metrics failure')
        return dict(states['100' if port_to_read == port else '200'])
    guard = ModelMeasurementGuard(100, {'100': port, '200': 1}, lambda: dict(processes), read)
    guard.assert_idle()
    states['100']['processing'] = 1
    assert guard.abort_reason() is None, 'Own generation must be allowed'
    changed = False
    def on_chunk(chunk):
        nonlocal changed, failed_reader
        if changed:
            return
        changed = True
        if case == 'peer_queue': states['200']['queued'] = 1
        elif case == 'peer_active': states['200']['processing'] = 1
        elif case == 'target_queue': states['100']['queued'] = 1
        elif case == 'parallel_user': states['100']['processing'] = 2
        elif case == 'peer_cpu': processes['200'] = (100, 'peer-start', 'llama-server')
        elif case == 'identity': processes['200'] = (0, 'replaced-peer', 'llama-server')
        elif case == 'foreign': processes['300'] = (0, 'foreign-start', 'llama-server')
        elif case == 'monitor_failure': failed_reader = True
    aborts = []
    error = None
    try:
        try:
            chunks = stream_completion(port, {'normal': case == 'normal'}, guard.abort_reason,
                                       aborts, on_chunk, timeout=5, interval=0.01)
            assert case == 'normal' and len(chunks) == 1
        except RuntimeError as caught:
            error = str(caught)
            assert case != 'normal' and aborts, (case, error)
        assert bool(aborts) == (case != 'normal'), (case, aborts)
        if case in ('identity', 'monitor_failure'):
            assert aborts[0]['reason'] == 'Request monitor failed'
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2) as response:
            assert json.load(response) == {'status': 'ok'}
        assert thread.is_alive(), 'The service must survive release of the request socket'
        return dict(case=case, passed=True, request_released=bool(aborts), server_still_responsive=True)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive()


if __name__ == '__main__':
    cases = ('normal', 'peer_queue', 'peer_active', 'target_queue', 'parallel_user',
             'peer_cpu', 'identity', 'foreign', 'monitor_failure')
    print(json.dumps(dict(cases=[run_case(case) for case in cases], real_model_requests=0)))
