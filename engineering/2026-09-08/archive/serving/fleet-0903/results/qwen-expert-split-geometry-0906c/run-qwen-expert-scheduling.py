#!/usr/bin/env python3
"""Guarded private Qwen expert checks; preserve all production services/files."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
import urllib.request

from inference_contention_guard import activity


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', required=True)
    args = parser.parse_args()
    assert re.fullmatch(r'[A-Za-z0-9_-]+', args.label)
    base = Path(__file__).resolve().parent
    engine = base.parents[1] / 'engines/llama.cpp-q4e-goal-0904'
    pinned = engine / 'validated-iq-batch3-bin'
    out = base / 'results' / args.label
    out.mkdir(exist_ok=False)
    allowed = {'4005448': 18091, '2308651': 18095}
    result = dict(started=time.time(), pid=os.getpid(), config=vars(args), runs=[], completed=False,
                  scope='Private metadata and component checks, not model bandwidth.')
    owned = []
    stop = threading.Event()
    failure = []

    def save():
        temporary = out / 'result.json.tmp'
        temporary.write_text(json.dumps(result, indent=2) + '\n')
        temporary.replace(out / 'result.json')

    def snapshot():
        state = {}
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():
                continue
            try:
                name = (proc / 'comm').read_text().strip()
                if name != 'llama-server' and not name.startswith(('glm-mtp-head', 'qwen-mtp-head')):
                    continue
                fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
                command = (proc / 'cmdline').read_bytes().split(b'\0')
                ports = [int(command[i + 1]) for i, a in enumerate(command[:-1]) if a == b'--port']
                state[proc.name] = (int(fields[11]) + int(fields[12]), fields[19], ports[-1] if ports else None)
            except (OSError, ValueError, IndexError):
                pass
        return state

    def services(state=None):
        state = snapshot() if state is None else state
        assert set(allowed) <= set(state), 'Protected inference process missing'
        statuses = {}
        for pid, port in allowed.items():
            assert state[pid][2] == port, 'Protected service port changed'
            def get(path):
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/{path}', timeout=2) as response:
                    return response.read().decode()
            slots = json.loads(get('slots'))
            queue = [float(l.split()[-1]) for l in get('metrics').splitlines()
                     if not l.startswith('#') and 'requests_deferred' in l]
            assert len(queue) == 1, 'Missing queue metric'
            statuses[pid] = dict(processing=any(s['is_processing'] for s in slots), queue=queue[0])
        foreign = sorted(set(state) - set(allowed))
        return dict(servers=statuses, foreign=foreign, busy=bool(foreign) or
                    any(s['processing'] or s['queue'] for s in statuses.values()))

    def terminate_owned(sig=signal.SIGTERM):
        for process in list(owned):
            if process.poll() is None:
                try:
                    os.killpg(process.pid, sig)
                except ProcessLookupError:
                    pass

    def monitor():
        try:
            before, then = snapshot(), time.monotonic()
            while not stop.wait(0.5):
                after, now = snapshot(), time.monotonic()
                statuses = services(after)
                samples, churn = activity(before, after, now - then)
                assert not statuses['busy'] and not churn and not any(s['cpu_percent'] > 20 for s in samples), (statuses, samples, churn)
                before, then = after, now
        except BaseException as error:
            failure.append(dict(time=time.time(), reason=repr(error), action='Stop only owned private subprocess groups.'))
            (out / 'contention.json').write_text(json.dumps(failure, indent=2) + '\n')
            terminate_owned()

    def check():
        assert not failure, failure
        assert not services()['busy'], 'Inference began or queued'

    def run(command, label, env, cwd=None, timeout=600):
        check()
        info = dict(command=command, label=label, started=time.time())
        result['runs'].append(info)
        save()
        with (out / (label + '.log')).open('w') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       env=env, cwd=cwd, start_new_session=True)
            owned.append(process)
            info['pid'] = process.pid
            deadline = time.monotonic() + timeout
            try:
                while process.poll() is None:
                    check()
                    assert time.monotonic() < deadline, 'Private command timed out'
                    time.sleep(0.2)
                check()
                info['exit_code'] = process.returncode
                assert process.returncode == 0, (label, process.returncode)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                info['finished'] = time.time()
                save()
        print(label + ' completed', flush=True)
        return (out / (label + '.log')).read_text()

    thread = None
    save()
    try:
        before, then = snapshot(), time.monotonic()
        quiet_since, last_report = None, 0
        while True:
            state = services(before)
            now = time.monotonic()
            quiet_since = None if state['busy'] else now if quiet_since is None else quiet_since
            quiet = 0 if quiet_since is None else now - quiet_since
            result['idle_gate'] = dict(time=time.time(), quiet_seconds=quiet, services=state)
            save()
            if quiet >= 60:
                break
            if now - last_report >= 30:
                print(json.dumps(dict(waiting_for_idle=True, quiet_seconds=quiet, busy=state['busy'])), flush=True)
                last_report = now
            # Average over five seconds: subsecond probes can make the service's
            # own metrics requests look like non-idle inference at 1% CPU.
            time.sleep(5)
            after, now = snapshot(), time.monotonic()
            samples, churn = activity(before, after, now - then)
            result['idle_activity'] = dict(samples=samples, churn=churn, elapsed=now - then)
            if churn or any(s['cpu_percent'] > 1 for s in samples):
                quiet_since = None
            before, then = after, now
        result['before_state'] = services()
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        source = base / 'qwen-expert-split-check.cpp'
        inputs = [source, Path(__file__).resolve(), engine / 'src/llama-model.cpp',
                  engine / 'src/llama-model.h', engine / 'src/llama-hparams.h', engine / 'include/llama.h']
        inputs += [p for p in pinned.glob('lib*.so.*') if p.is_file()]
        result['input_sha256'] = {str(p): digest(p) for p in inputs}
        for p in (source, Path(__file__).resolve()):
            (out / p.name).write_bytes(p.read_bytes())
        env = {k:v for k,v in os.environ.items() if not k.startswith(('GGML_', 'OMP_', 'GOMP_'))}
        env.update(LD_LIBRARY_PATH=str(pinned), OMP_NUM_THREADS='1')
        binary = out / 'qwen-expert-split-check'
        command = ['g++', '-O2', '-std=c++17', str(source)]
        command += ['-I' + str(engine / p) for p in ('include', 'src', 'ggml/include', 'ggml/src')]
        command += ['-L' + str(pinned), '-lllama', '-lggml', '-lggml-base', '-ldl', '-pthread', '-o', str(binary)]
        run(command, 'build-geometry', env)
        log = run([str(binary)], 'geometry', env)
        lines = [json.loads(l) for l in log.splitlines() if l.startswith('{')]
        library = [l for l in lines if l['event'] == 'library']
        assert len(library) == 1 and Path(library[0]['path']).resolve() == (pinned / 'libllama.so').resolve()
        splits = [l for l in lines if l['event'] == 'split']
        assert len(splits) == 144 and len({s['tensor'] for s in splits}) == 144
        assert all(sorted(s['slices']) == [128, 128, 128, 256] for s in splits), 'Split hypothesis was not confirmed'
        assert all(s['axis'] == (0 if '.ffn_down_' in s['tensor'] else 1) for s in splits)
        result['geometry'] = dict(library=library[0], splits=splits, all_slices_confirmed=True,
                                  scope='Deployed split function evaluated on weight-free metadata for all 48 expert layers.')
        result['binary_sha256'] = digest(binary)
        assert all(digest(p) == checksum for p, checksum in result['input_sha256'].items())
        result['after_state'] = services()
        check()
        result['completed'] = True
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        terminate_owned()
        for process in owned:
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        stop.set()
        if thread:
            thread.join(timeout=10)
            assert not thread.is_alive(), 'Guard thread did not stop'
        result.update(finished=time.time(), contention=failure, owned_exit_codes={p.pid:p.returncode for p in owned})
        save()


if __name__ == '__main__':
    main()
