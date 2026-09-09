"""Stream one HTTP request and release its socket if the host needs the server."""
import http.client
import json
import socket
import threading
import time


def stream_completion(port, payload, check_abort, abort_events, on_chunk=None,
                      timeout=600, interval=2):
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=timeout)
    response = None
    watcher = None
    client_socket = None
    stop = threading.Event()
    chunks = []

    def watch():
        while not stop.wait(interval):
            try:
                reason = check_abort()
            except Exception as error:
                reason = dict(reason='Request monitor failed', error=repr(error))
            if reason:
                abort_events.append(dict(time=time.time(), **reason))
                try:
                    client_socket.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                return

    try:
        connection.connect()
        client_socket = connection.sock
        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
        connection.request('POST', '/v1/chat/completions', json.dumps(payload),
                           {'Content-Type': 'application/json'})
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(f'Completion HTTP status {response.status}')
        done = False
        for raw in response:
            if abort_events:
                raise RuntimeError(f'Benchmark request released: {abort_events}')
            line = raw.decode().strip()
            if not line.startswith('data:'):
                continue
            data = line[5:].strip()
            if data == '[DONE]':
                done = True
                break
            chunk = json.loads(data)
            chunks.append(chunk)
            if on_chunk is not None:
                on_chunk(chunk)
        if abort_events:
            raise RuntimeError(f'Benchmark request released: {abort_events}')
        if not done:
            raise RuntimeError('Completion stream ended without [DONE]')
        return chunks
    finally:
        stop.set()
        if client_socket is not None:
            try:
                client_socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if response is not None:
            response.close()
        connection.close()
        if watcher is not None:
            watcher.join(timeout=5)
            if watcher.is_alive():
                raise RuntimeError('Request monitor did not stop')
