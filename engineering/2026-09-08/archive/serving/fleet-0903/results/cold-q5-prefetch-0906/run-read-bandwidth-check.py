#!/usr/bin/env python3
"""Guarded, NUMA-local read calibration; never a model-throughput result."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import select
import signal
import subprocess
import time
import urllib.request

from dram_bandwidth import PerfDramRecorder, summarize_samples
from inference_contention_guard import InferenceContentionGuard, activity, wait_for_idle

parser = argparse.ArgumentParser()
parser.add_argument('--label', required=True)
parser.add_argument('--seconds', type=float, default=10)
parser.add_argument('--workers-per-socket', type=int, default=15)
parser.add_argument('--mib-per-worker', type=int, default=32)
parser.add_argument('--kind', choices=('read', 'q4', 'q5'), default='read')
parser.add_argument('--k', type=int, default=4096)
parser.add_argument('--tokens', type=int, default=1)
parser.add_argument('--modes', help='Comma-separated measurement order; defaults to paired repetitions.')
args = parser.parse_args()
assert re.fullmatch(r'[a-zA-Z0-9_-]+', args.label)
assert 3 <= args.seconds <= 30 and 1 <= args.workers_per_socket <= 15
assert 16 <= args.mib_per_worker <= 64
assert 256 <= args.k <= 8192 and args.k % 256 == 0 and 1 <= args.tokens <= 3
modes = args.modes.split(',') if args.modes else (['cached', 'stream', 'stream', 'cached'] if args.kind == 'read'
        else ['full', 'copy', 'pf1', 'pf2', 'pf4', 'pf4', 'pf2', 'pf1', 'copy', 'full'])
assert modes and all(mode in (('cached', 'stream') if args.kind == 'read' else ('full', 'copy', 'pf1', 'pf2', 'pf4')) for mode in modes)
base = Path(__file__).resolve().parent
out = base / 'results' / args.label
out.mkdir(exist_ok=False)
source = base / 'read-bandwidth-check.cpp'
allowed = {'4005448': 18091, '2308651': 18095}
result = dict(started=time.time(), pid=os.getpid(), config=vars(args), runs=[],
              source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
              scope='Synthetic read-only calibration, not model-attributable bandwidth or goal completion.')
result['modes'] = modes
environment = dict(os.environ)
engine = base.parents[1] / 'engines/llama.cpp-sr950-glm'
libraries = engine / 'build-dev2/bin'


def save():
    temporary = out / 'result.json.tmp'
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(out / 'result.json')


def snapshot():
    current = {}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            name = (p / 'comm').read_text().strip()
            if name != 'llama-server' and not name.startswith(('glm-mtp-head', 'qwen-mtp-head')):
                continue
            fields = (p / 'stat').read_text().rsplit(')', 1)[1].split()
            command = (p / 'cmdline').read_bytes().split(b'\0')
            ports = [command[i + 1].decode() for i, x in enumerate(command[:-1]) if x == b'--port']
            current[p.name] = (int(fields[11]) + int(fields[12]), fields[19], ports[-1] if ports else None)
        except (OSError, ValueError, IndexError):
            pass
    return current


def server_state():
    current = snapshot()
    assert set(allowed) <= set(current), 'An expected inference service is missing'
    state = {}
    for pid, port in allowed.items():
        assert current[pid][2] == str(port), 'An expected server port changed'
        def get(path):
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/{path}', timeout=2) as response:
                return response.read().decode()
        slots = json.loads(get('slots'))
        queue = [float(line.split()[-1]) for line in get('metrics').splitlines()
                 if not line.startswith('#') and 'requests_deferred' in line]
        assert len(queue) == 1, 'Missing queue metric'
        state[pid] = dict(processing=any(s['is_processing'] for s in slots), queued=queue[0])
    return dict(servers=state, foreign_pids=sorted(set(current) - set(allowed)),
                busy=bool(set(current) - set(allowed)) or any(s['processing'] or s['queued'] for s in state.values()))


class ProcessGroup:
    def __init__(self, process):
        self.process = process
        self.pid = process.pid

    def poll(self):
        return self.process.poll()

    def terminate(self):
        if self.poll() is None:
            os.killpg(self.pid, signal.SIGTERM)

    def kill(self):
        if self.poll() is None:
            os.killpg(self.pid, signal.SIGKILL)


def ensure_idle(guard=None):
    state = server_state()
    assert not state['busy'], f'Inference active or queued: {state}'
    assert guard is None or guard.info is None, 'Other inference used CPU during the benchmark'
    return state


def background(seconds, guard):
    start = time.monotonic()
    while time.monotonic() - start < seconds:
        ensure_idle(guard)
        time.sleep(0.5)
    return start, time.monotonic()


def event(process, guard, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        ensure_idle(guard)
        readable, _, _ = select.select([process.stdout], [], [], 0.5)
        if readable:
            line = process.stdout.readline()
            assert line, f'Fixture exited unexpectedly: {process.poll()}'
            return json.loads(line)
    raise TimeoutError('Fixture did not produce the expected event')


def host_processes():
    state = {}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            fields = (p / 'stat').read_text().rsplit(')', 1)[1].split()
            state[p.name] = (int(fields[11]) + int(fields[12]), fields[19], (p / 'comm').read_text().strip())
        except (OSError, ValueError, IndexError):
            pass
    return state


save()
try:
    while True:
        result['idle_gate'] = wait_for_idle(snapshot, out / 'waiting-for-idle.json', allowed_idle_pids=tuple(allowed))
        if not server_state()['busy']:
            break
    result['before_state'] = ensure_idle()
    result['memory_before'] = [line for line in Path('/proc/meminfo').read_text().splitlines()
                               if line.startswith(('MemFree:', 'MemAvailable:', 'SwapFree:'))]
    node_bytes = args.workers_per_socket * args.mib_per_worker * 1024**2
    result['node_memory_before'] = {}
    for node in range(4):
        text = Path(f'/sys/devices/system/node/node{node}/meminfo').read_text()
        free_kib = int(next(line for line in text.splitlines() if 'MemFree:' in line).split()[-2])
        result['node_memory_before'][str(node)] = free_kib
        assert free_kib * 1024 > node_bytes + 512 * 1024**2, f'Insufficient free memory on node {node}'
    binary = out / 'read-bandwidth-check'
    command = ['g++', '-O3', '-std=c++17', '-march=native', '-pthread', str(source), '-o', str(binary)]
    if args.kind != 'read':
        generator = base / 'make-cold-x16-candidates.py'
        spec = importlib.util.spec_from_file_location('cold_x16_generator', generator)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result['candidate_source'] = module.generate(engine, out / 'cold-x16-candidates.h')
        result['fixture_dependencies'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in (generator, base / 'cold-x16-workload.h')}
        result['libraries'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in libraries.glob('libggml*.so.*') if p.is_file()}
        command += ['-DCOLD_X16', '-I' + str(out)]
        command += ['-I' + str(engine / p) for p in ('include', 'ggml/include', 'ggml/src', 'ggml/src/ggml-cpu')]
        command += ['-L' + str(libraries), '-lggml-cpu', '-lggml-base']
        environment.update(LD_LIBRARY_PATH=str(libraries), COLD_X16_KIND=args.kind[-1],
                           COLD_X16_K=str(args.k), COLD_X16_TOKENS=str(args.tokens))
    result['build_command'] = command
    save()
    with (out / 'build.log').open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        owned = ProcessGroup(process)
        guard = InferenceContentionGuard(owned, snapshot, out / 'build-contention.json',
                                         interval=0.5, threshold=20, consecutive=1, grace=1)
        guard.start()
        try:
            deadline = time.monotonic() + 120
            while process.poll() is None:
                ensure_idle(guard)
                assert time.monotonic() < deadline, 'Build timed out'
                time.sleep(0.5)
            assert process.returncode == 0, 'Build failed'
        finally:
            owned.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                owned.kill()
                process.wait()
            result['build_contention'] = guard.stop()
    ensure_idle()
    result['binary_sha256'] = hashlib.sha256(binary.read_bytes()).hexdigest()
    for index, mode in enumerate(modes):
        run = dict(index=index, mode=mode, started=time.time(), before_state=ensure_idle())
        result['runs'].append(run)
        save()
        before = snapshot()
        before_time = time.monotonic()
        command = [str(binary), mode, str(args.workers_per_socket), str(args.mib_per_worker), str(args.seconds)]
        recorder = None
        with (out / f'arm{index}-{mode}.log').open('w') as log:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=log, text=True, bufsize=1, start_new_session=True, env=environment)
            owned = ProcessGroup(process)
            guard = InferenceContentionGuard(owned, snapshot, out / f'arm{index}-contention.json',
                                             interval=0.5, threshold=20, consecutive=1, grace=1)
            guard.start()
            try:
                run['ready'] = event(process, guard, 60)
                assert run['ready']['event'] == 'ready' and run['ready']['pid'] == process.pid
                assert run['ready']['workers'] == 4 * args.workers_per_socket
                recorder = PerfDramRecorder(out / f'arm{index}-imc').start()
                run['background_before'] = background(5, guard)
                host_before = host_processes()
                host_start = time.monotonic()
                process.stdin.write('go\n')
                process.stdin.flush()
                run['measurement'] = measured = event(process, guard, args.seconds + 30)
                host_elapsed = time.monotonic() - host_start
                host_activity, host_churn = activity(host_before, host_processes(), host_elapsed, own_pid=process.pid)
                run['other_host_activity'] = sorted(host_activity, key=lambda s: s['cpu_percent'], reverse=True)[:12]
                run['host_process_churn'] = host_churn
                assert measured['event'] == 'done' and measured['checksums_exact']
                run['background_after'] = background(5, guard)
                samples, metadata = recorder.stop()
                recorder = None
                run['counter_metadata'] = metadata
                assert metadata['valid'] and metadata['exit_code'] == 0
                start = max(t['start'] for t in measured['threads']) + 1
                end = min(t['end'] for t in measured['threads']) - 1
                run['stable_read_window'] = [start, end]
                run['dram'] = dram = summarize_samples(samples, start, end)
                run['background'] = backgrounds = [summarize_samples(samples, a + 0.5, b - 0.5)
                                                   for a, b in (run['background_before'], run['background_after'])]
                assert dram['valid'] and all(b['valid'] for b in backgrounds)
                run['adjusted_read_gb_s'] = dram['read_gb_s'] - max(b['read_gb_s'] for b in backgrounds)
                run['adjusted_total_gb_s'] = dram['total_gb_s'] - max(b['total_gb_s'] for b in backgrounds)
                run['read_counter_over_logical'] = run['adjusted_read_gb_s'] / measured['logical_gb_s']
                process.stdin.write('exit\n')
                process.stdin.flush()
                assert process.wait(timeout=10) == 0
            finally:
                owned.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    owned.kill()
                    process.wait()
                if recorder is not None:
                    recorder.stop()
                run['contention'] = guard.stop()
                run['finished'] = time.time()
                save()
        overlap, churn = activity(before, snapshot(), time.monotonic() - before_time)
        run.update(other_inference=overlap, inference_churn=churn, after_state=ensure_idle())
        assert not run['contention'] and not churn and not any(s['cpu_percent'] > 5 for s in overlap)
        run['complete'] = True
        save()
        print(json.dumps({k: run[k] for k in ('index', 'mode', 'adjusted_read_gb_s', 'adjusted_total_gb_s',
                                             'read_counter_over_logical')}), flush=True)
    assert result['source_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert result['binary_sha256'] == hashlib.sha256(binary.read_bytes()).hexdigest()
    if args.kind != 'read':
        for collection in ('fixture_dependencies', 'libraries'):
            assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest
                       for p, digest in result[collection].items()), f'{collection} changed during the measurement'
        candidate = result['candidate_source']
        assert candidate['source_sha256'] == hashlib.sha256(Path(candidate['source']).read_bytes()).hexdigest()
        assert candidate['generated_sha256'] == hashlib.sha256((out / 'cold-x16-candidates.h').read_bytes()).hexdigest()
    result['after_state'] = ensure_idle()
except BaseException as error:
    result['error'] = repr(error)
    raise
finally:
    result['finished'] = time.time()
    save()
