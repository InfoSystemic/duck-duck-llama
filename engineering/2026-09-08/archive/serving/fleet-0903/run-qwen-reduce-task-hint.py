#!/usr/bin/env python3
"""Check NUMA reduction scheduling without loading or reconfiguring a model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import time
import urllib.request

from inference_contention_guard import InferenceContentionGuard, activity, wait_for_idle

parser = argparse.ArgumentParser()
parser.add_argument('--label', required=True)
parser.add_argument('--timing', action='store_true')
parser.add_argument('--engine', choices=('qwen', 'full'), default='qwen')
parser.add_argument('--flag', choices=('task-hint', 'single-worker'), default='task-hint')
parser.add_argument('--single-worker-limit', type=int, default=65536)
options = parser.parse_args()
assert re.fullmatch(r'[a-zA-Z0-9_-]+', options.label)
assert options.single_worker_limit > 0
base = Path(__file__).resolve().parent
root = base.parents[1]
engine = root / 'engines' / ('llama.cpp-q4e-goal-0904' if options.engine == 'qwen' else 'llama.cpp-sr950-glm')
build = engine / ('build-goal' if options.engine == 'qwen' else 'build-dev2')
assert options.engine != 'full' or options.flag == 'single-worker', 'The protected Full library has no experimental task hint'
out = base / 'results' / options.label
out.mkdir(exist_ok=False)
result = dict(started=time.time(), pid=os.getpid(), config=vars(options), runs=[], timings=[])
environment = {k: v for k, v in os.environ.items() if not k.startswith(('GGML_', 'OMP_', 'GOMP_', 'REPACK_TEST_', 'NUMA_REDUCE_TEST_'))}
environment['LD_LIBRARY_PATH'] = str(build / 'bin')
if options.engine == 'full':
    baseline = json.loads((base / 'results/glm53-draft-sweep-bandwidth-0905/result.json').read_text())
    assert baseline.get('finished') and not baseline.get('error')
    result['numa_environment'] = {k:v for k,v in baseline['runtime_env'].items() if k.startswith('GGML_CPU_NUMA_')}
    environment.update(result['numa_environment'])
allowed = {'4005448': 18091, '2308651': 18095}


def save():
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')


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
            args = (proc / 'cmdline').read_bytes().split(b'\0')
            ports = [args[i + 1].decode() for i, value in enumerate(args[:-1]) if value == b'--port']
            state[proc.name] = (int(fields[11]) + int(fields[12]), fields[19], ports[-1] if ports else None)
        except (OSError, ValueError, IndexError):
            pass
    return state


def assert_servers_idle():
    state = snapshot()
    assert set(state) == set(allowed), 'Inference process inventory changed'
    for pid, port in allowed.items():
        assert state[pid][2] == str(port)
        def get(path):
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/{path}', timeout=3) as response:
                return response.read().decode()
        assert not any(s['is_processing'] for s in json.loads(get('slots'))), 'Inference is active'
        queue = [float(line.split()[-1]) for line in get('metrics').splitlines()
                 if not line.startswith('#') and 'requests_deferred' in line]
        assert queue == [0.0], 'Inference queue is not empty'


class ProcessGroup:
    def __init__(self, process): self.process = process; self.pid = process.pid
    def poll(self): return self.process.poll()
    def terminate(self):
        if self.poll() is None: os.killpg(self.pid, signal.SIGTERM)
    def kill(self):
        if self.poll() is None: os.killpg(self.pid, signal.SIGKILL)


def run(command, label, env):
    assert_servers_idle()
    info = dict(label=label, command=command, started=time.time())
    result['runs'].append(info); save()
    before = snapshot(); start = time.monotonic()
    with (out / (label + '.log')).open('w') as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        owned = ProcessGroup(process)
        guard = InferenceContentionGuard(owned, snapshot, out / (label + '-contention.json'),
                                         interval=0.5, threshold=20, consecutive=1, grace=1)
        guard.start()
        try:
            info['exit'] = process.wait(timeout=600)
        except BaseException:
            owned.terminate()
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired: owned.kill(); process.wait()
            raise
        finally:
            samples, churn = activity(before, snapshot(), time.monotonic() - start)
            info.update(contention=guard.stop(), other_inference=samples, churn=churn, finished=time.time())
            save()
    assert not info['contention'] and not churn and not any(s['cpu_percent'] > 5 for s in samples), 'Inference interrupted the check'
    assert_servers_idle()
    assert info['exit'] == 0, (label, info['exit'])
    print(label, 'completed', flush=True)
    return (out / (label + '.log')).read_text()


save()
try:
    result['idle'] = wait_for_idle(snapshot, out / 'waiting-for-idle.json', allowed_idle_pids=tuple(allowed))
    result['source_sha256'] = hashlib.sha256((engine / 'ggml/src/ggml-backend-meta.cpp').read_bytes()).hexdigest()
    result['fixture_sha256'] = hashlib.sha256((base / 'numa-reduce-check.cpp').read_bytes()).hexdigest()
    if options.engine == 'qwen':
        run(['cmake', '--build', str(build), '--target', 'ggml', 'ggml-cpu', '-j', '3'], 'build-library', environment)
    result['library_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (build / 'bin').glob('libggml*.so.*') if p.is_file()}
    binary = out / 'numa-reduce-check'
    command = ['g++', '-O2', '-std=c++17', '-I' + str(engine / 'ggml/include'), str(base / 'numa-reduce-check.cpp'),
               '-L' + str(build / 'bin'), '-lggml', '-lggml-cpu', '-lggml-base', '-pthread', '-o', str(binary)]
    run(command, 'build-check', environment)
    hashes = []
    arms = (0, 1, 1, 0) if options.timing else (0, 1)
    for arm, hint in enumerate(arms):
        env = dict(environment)
        if options.flag == 'task-hint':
            env['GGML_CPU_NUMA_FUSED_REDUCE_TASK_HINT'] = str(hint)
        else:
            env['GGML_CPU_NUMA_FUSED_REDUCE_SINGLE_MAX_ELEMENTS'] = str(options.single_worker_limit) if hint else '0'
        if options.engine == 'full':
            env['GGML_CPU_SINGLE_TASK_MAX_ELEMENTS'] = '32768'
            env['NUMA_REDUCE_TEST_FULL_SHAPES'] = '1'
        if options.timing:
            env['NUMA_REDUCE_TEST_TIMING_REPEATS'] = '1000'
        log = run([str(binary), '--fused-graph'], f'check-{arm}-hint-{hint}', env)
        assert 'fused in-graph all-reduce active' in log, 'The intended reduction path was not exercised'
        lines = [line for line in log.splitlines() if line.startswith('PASS fused_graph ')]
        assert len(lines) == (8 if options.timing else 32), len(lines)
        hashes.append([line.split(' hash=')[1] for line in lines])
        if options.timing:
            result['timings'].append(dict(arm=arm, hint=hint, cases=[dict(re.findall(r'(\w+)=([^ ]+)', line)) for line in lines]))
            save()
    assert all(h == hashes[0] for h in hashes), 'Task hint changed graph outputs'
    result['exact_graph_cases'] = len(hashes[0])
    result['checked_input_iterations_per_arm'] = len(hashes[0]) * 5
    if options.timing:
        result['ratios'] = []
        for i in range(8):
            reference = result['timings'][0]['cases'][i]
            ms = {hint: statistics.median(float(arm['cases'][i]['graph_ms']) for arm in result['timings'] if arm['hint'] == hint) for hint in (0, 1)}
            item = {k: int(reference[k]) for k in ('rows', 'tokens', 'merged')}
            item.update(ms=ms, off_over_on=ms[0] / ms[1])
            result['ratios'].append(item)
            print(json.dumps(item), flush=True)
    assert result['source_sha256'] == hashlib.sha256((engine / 'ggml/src/ggml-backend-meta.cpp').read_bytes()).hexdigest()
    assert result['fixture_sha256'] == hashlib.sha256((base / 'numa-reduce-check.cpp').read_bytes()).hexdigest()
    assert all(hashlib.sha256((build / 'bin' / name).read_bytes()).hexdigest() == digest for name, digest in result['library_sha256'].items())
except Exception as error:
    result['error'] = repr(error)
    raise
finally:
    result['finished'] = time.time(); save()
