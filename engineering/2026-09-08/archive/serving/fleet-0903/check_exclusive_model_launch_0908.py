#!/usr/bin/env python3
"""Verify launch exclusion without executing a model or changing a live service."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import time
from exclusive_model_launch import acquire_exclusive_model_launch, inference_processes, ModelLaunchConflict

BASE = Path(__file__).resolve().parent


def conflict(call, expected):
    try:
        call()
    except ModelLaunchConflict as error:
        assert expected in str(error), str(error)
        return str(error)
    raise AssertionError('Launch was unexpectedly allowed')


def main(pid):
    checks = []
    with tempfile.TemporaryDirectory(prefix='model-launch-check-') as directory:
        root = Path(directory)
        proc = root / 'proc'
        proc.mkdir()
        lease = acquire_exclusive_model_launch(root, proc)
        assert not os.get_inheritable(lease.transaction.fileno()) and os.get_inheritable(lease.residency.fileno())
        conflict(lambda: acquire_exclusive_model_launch(root, proc), 'transaction')
        checks.append('concurrent startup transaction rejected')
        fd = lease.residency.fileno()
        code = 'import os,sys; os.fstat(int(sys.argv[1])); print("ready",flush=True); sys.stdin.readline()'
        child = subprocess.Popen([sys.executable, '-c', code, str(fd)], pass_fds=(fd,),
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        try:
            assert child.stdout.readline().strip() == 'ready'
            lease.close()
            conflict(lambda: acquire_exclusive_model_launch(root, proc), 'guarded whole-server model')
            checks.append('residency lock remains held by exec child after parent closes its handles')
            child.stdin.write('finish\n')
            child.stdin.flush()
            assert child.wait(timeout=5) == 0
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=5)
            lease.close()
        with acquire_exclusive_model_launch(root, proc):
            pass
        checks.append('process exit releases residency lock despite remaining lock file')
        fake = proc / '321'
        fake.mkdir()
        (fake / 'comm').write_text('llama-server\n')
        (fake / 'stat').write_text('321 (llama-server) D 1 2 3\n')
        (fake / 'cmdline').write_bytes(b'\0'.join([b'llama-server', b'--model', b'/models/GLM-Flash-Q8.gguf', b'--port', b'18131']) + b'\0')
        message = conflict(lambda: acquire_exclusive_model_launch(root, proc), 'PID 321')
        assert '18131' in message and 'GLM-Flash-Q8.gguf' in message
        checks.append('loading resident model blocks launch before it has an HTTP service')
        (fake / 'stat').write_text('321 (llama-server) Z 1 2 3\n')
        with acquire_exclusive_model_launch(root, proc):
            pass
        checks.append('dead zombie does not reserve the server')
    tree = ast.parse((BASE / 'pin-qwen-flash-20tps.py').read_text())
    templates = [node.args[0].value for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == 'write_text' and node.args
                 and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)
                 and node.args[0].value.startswith('#!/usr/bin/env python3')]
    assert templates == [(BASE / 'launch-qwen-flash-20tps.py').read_text()]
    checks.append('launcher generator preserves the same exclusion check')
    peers = inference_processes()
    assert any(row['pid'] == pid for row in peers)
    original_exec = os.execvpe
    def never_exec(*args, **kwargs):
        raise AssertionError('The test must never execute a model')
    os.execvpe = never_exec
    try:
        try:
            runpy.run_path(str(BASE / 'launch-qwen-flash-20tps.py'), run_name='__main__')
        except SystemExit as error:
            assert 'Qwen launch refused:' in str(error) and f'PID {pid}' in str(error)
            refusal = str(error)
        else:
            raise AssertionError('Actual launcher failed to refuse the resident server')
    finally:
        os.execvpe = original_exec
    assert any(row['pid'] == pid for row in inference_processes())
    checks.append('actual Qwen launcher rejects a duplicate start while the existing process remains alive')
    paths = [Path(__file__), BASE / 'exclusive_model_launch.py', BASE / 'launch-qwen-flash-20tps.py', BASE / 'pin-qwen-flash-20tps.py']
    result = dict(time=time.time(), passed=True, checks=checks, existing_pid=pid,
                  refusal=refusal, source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    out = BASE / 'results/exclusive-model-launch-0908/result.json'
    assert not out.exists()
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--existing-pid', type=int, required=True)
    main(parser.parse_args().existing_pid)
