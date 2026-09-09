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
from model_measurement_guard import ModelMeasurementGuard

parser = argparse.ArgumentParser()
parser.add_argument('--label', required=True)
parser.add_argument('--seconds', type=float, default=10)
parser.add_argument('--workers-per-socket', type=int, default=15)
parser.add_argument('--mib-per-worker', type=int, default=32)
parser.add_argument('--kind', choices=('read', 'q4', 'q5', 'graph'), default='read')
parser.add_argument('--k', type=int, default=4096)
parser.add_argument('--tokens', type=int, default=1)
parser.add_argument('--modes', help='Comma-separated measurement order; defaults to paired repetitions.')
parser.add_argument('--graph-rows', type=int, default=6144)
parser.add_argument('--graph-matrices', type=int, default=8)
parser.add_argument('--graph-controllers', type=int, choices=(1, 4), default=4)
parser.add_argument('--profile-cycles', action='store_true', help='Profile private graph cycles instead of measuring IMC bandwidth')
args = parser.parse_args()
assert re.fullmatch(r'[a-zA-Z0-9_-]+', args.label)
assert 3 <= args.seconds <= 30 and 1 <= args.workers_per_socket <= 15
assert 16 <= args.mib_per_worker <= 64
assert 256 <= args.k <= 8192 and args.k % 256 == 0 and 1 <= args.tokens <= 3
default_modes = {'read': ['cached', 'stream', 'stream', 'cached'], 'graph': ['64', '32', '16', '16', '32', '64']}
modes = args.modes.split(',') if args.modes else default_modes.get(args.kind, ['full', 'copy', 'pf1', 'pf2', 'pf4', 'pf4', 'pf2', 'pf1', 'copy', 'full'])
valid_modes = {'read': ('cached', 'stream'), 'graph': ('16', '32', '64', '128', '256', '64g1', '64g2', '64g4', '64c1', '64c4')}.get(args.kind, ('full', 'small', 'copy', 'pf1', 'pf2', 'pf4', 'batch3', 'batch3c2'))
assert modes and all(mode in valid_modes for mode in modes)
assert 'batch3' not in modes or args.kind == 'q5', 'The multirow candidate supports Q5 only'
assert 'batch3c2' not in modes or (args.kind == 'q5' and args.tokens == 3), 'Two chains target three-row Q5 only'
assert not args.profile_cycles or (args.kind == 'graph' and args.seconds >= 8 and len(modes) <= 2)
assert 64 <= args.graph_rows <= 16384 and args.graph_rows % 64 == 0 and 1 <= args.graph_matrices <= 64
graph_weight_bytes = args.graph_matrices * (args.graph_rows // 16) * (4 * args.k // 256) * 2880
assert args.kind != 'graph' or graph_weight_bytes <= 2 * 1024**3, 'Graph weight pool exceeds the fixture memory budget'
base = Path(__file__).resolve().parent
out = base / 'results' / args.label
out.mkdir(exist_ok=False)
source = base / ('cold-graph-check.cpp' if args.kind == 'graph' else 'read-bandwidth-check.cpp')
allowed = {'4005448': 18091, '2308651': 18095}
result = dict(started=time.time(), pid=os.getpid(), config=vars(args), runs=[],
              source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
              scope='Synthetic read-only calibration, not model-attributable bandwidth or goal completion.')
result['modes'] = modes
guard_paths = [Path(__file__).resolve(), base / 'model_measurement_guard.py',
               base / 'inference_contention_guard.py', base / 'dram_bandwidth.py']
result['guard_sources'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in guard_paths}
for path in guard_paths:
    (out / path.name).write_bytes(path.read_bytes())
expected_identities = {}
if args.profile_cycles:
    result['scope'] = 'Private finite-graph cycle profile. Profiled timing is not a throughput or bandwidth result.'
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
        assert current[pid][1] == expected_identities[pid], 'An expected server identity changed'
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
            assert line.lstrip().startswith('{'), f'Unexpected fixture stdout: {line!r}'
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


def graph_placement(pid):
    nodes = {str(j): dict(pages=0, nonlocal_pages=0) for j in range(4)}
    for line in Path(f'/proc/{pid}/numa_maps').read_text().splitlines():
        fields = line.split()
        if len(fields) < 2 or fields[1] not in ('bind:0', 'bind:1', 'bind:2', 'bind:3'):
            continue
        expected = fields[1][5:]
        for field in fields[2:]:
            match = re.fullmatch(r'N(\d+)=(\d+)', field)
            if match:
                node, count = match.groups()
                nodes[expected]['pages'] += int(count)
                nodes[expected]['nonlocal_pages'] += int(count) if node != expected else 0
    page_size = os.sysconf('SC_PAGE_SIZE')
    assert all(n['pages'] * page_size >= graph_weight_bytes // 4 and not n['nonlocal_pages'] for n in nodes.values()), nodes
    return dict(page_size=page_size, nodes=nodes, scope='Node-bound mappings after complete reference checks, before timed graph execution.')


def profile_graph(process, guard, directory):
    directory.mkdir()
    data = directory / 'perf.data'
    command = ['sudo', '-n', 'perf', 'record', '--per-thread', '--no-inherit',
               '--clockid', 'mono', '--timestamp', '--sample-cpu',
               '-F', '199', '-e', 'cycles:u', '-p', str(process.pid),
               '-o', str(data), '--', '/bin/cat']
    info = dict(command=command, target_pid=process.pid, scope='Only the private fixture; no IMC collection.')
    with (directory / 'record.log').open('w') as log:
        profiler = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        info['profiler_pid'] = profiler.pid
        try:
            background(1, guard)
            assert profiler.poll() is None, 'Cycle profiler exited during setup'
            process.stdin.write('go\n')
            process.stdin.flush()
            measured = event(process, guard, args.seconds + 30)
            assert measured['event'] == 'done' and measured['checksums_exact']
            measured['profiled'] = True
        finally:
            if profiler.stdin and not profiler.stdin.closed:
                profiler.stdin.close()
            try:
                info['exit_code'] = profiler.wait(timeout=15)
            except subprocess.TimeoutExpired:
                # The isolated profiler process group contains only this
                # sudo/perf/cat invocation, never an inference service.
                subprocess.run(['sudo', '-n', 'kill', '-INT', '--', f'-{profiler.pid}'],
                               check=True, timeout=5)
                info['exit_code'] = profiler.wait(timeout=5)
                info['required_interrupt'] = True
    assert info['exit_code'] == 0 and not info.get('required_interrupt')
    start = max(t['start'] for t in measured['threads']) + 1
    end = min(t['end'] for t in measured['threads']) - 1
    window = f'{start:.9f},{end:.9f}'
    info['selected_window_monotonic'] = [start, end]
    times = subprocess.run(['sudo', '-n', 'perf', 'script', '--show-lost-events',
                            '--time', window, '-F', 'time', '-i', str(data)],
                           capture_output=True, text=True, timeout=30)
    (directory / 'sample-time-errors.txt').write_text(times.stderr)
    assert times.returncode == 0 and 'LOST' not in times.stdout.upper(), 'Lost or unreadable cycle samples'
    timestamps = [float(line.strip().rstrip(':')) for line in times.stdout.splitlines() if line.strip()]
    assert len(timestamps) >= 1000 and start <= min(timestamps) <= max(timestamps) <= end
    assert min(timestamps) <= start + 1 and max(timestamps) >= end - 1
    info.update(selected_samples=len(timestamps), sample_window_monotonic=[min(timestamps), max(timestamps)])
    report = subprocess.run(['sudo', '-n', 'perf', 'report', '--stdio', '--no-children',
                             '--time', window, '--sort', 'dso,symbol', '--percent-limit', '0.5',
                             '-i', str(data)], capture_output=True, text=True, timeout=30)
    (directory / 'report.txt').write_text(report.stdout + report.stderr)
    assert report.returncode == 0
    trace = subprocess.run(['sudo', '-n', 'perf', 'script', '--time', window,
                            '-F', 'pid,tid,cpu,time,period,ip,sym,dso', '-i', str(data)],
                           capture_output=True, text=True, timeout=30)
    (directory / 'cycles.txt').write_text(trace.stdout)
    (directory / 'cycle-errors.txt').write_text(trace.stderr)
    assert trace.returncode == 0
    info['report_file'] = str(directory / 'report.txt')
    info['trace_file'] = str(directory / 'cycles.txt')
    ensure_idle(guard)
    (directory / 'profile.json').write_text(json.dumps(info, indent=2) + '\n')
    return measured, info


save()
try:
    initial = snapshot()
    expected_identities = {pid: initial[pid][1] for pid in allowed}
    result['protected_identities'] = expected_identities
    idle_guard = ModelMeasurementGuard(next(iter(allowed)), allowed, snapshot)
    result['idle_gate'] = idle_guard.wait_idle(out / 'waiting-for-idle.json')
    result['before_state'] = ensure_idle()
    result['memory_before'] = [line for line in Path('/proc/meminfo').read_text().splitlines()
                               if line.startswith(('MemFree:', 'MemAvailable:', 'SwapFree:'))]
    node_bytes = args.workers_per_socket * args.mib_per_worker * 1024**2
    if args.kind == 'graph':
        node_bytes = max(node_bytes, 2 * graph_weight_bytes // 4 + graph_weight_bytes // args.graph_matrices)
    result['node_memory_before'] = {}
    for node in range(4):
        text = Path(f'/sys/devices/system/node/node{node}/meminfo').read_text()
        free_kib = int(next(line for line in text.splitlines() if 'MemFree:' in line).split()[-2])
        result['node_memory_before'][str(node)] = free_kib
        assert free_kib * 1024 > node_bytes + 512 * 1024**2, f'Insufficient free memory on node {node}'
    binary = out / 'read-bandwidth-check'
    command = ['g++', '-O3', '-std=c++17', '-march=native', '-pthread', str(source), '-o', str(binary)]
    if args.kind in ('q4', 'q5'):
        generator = base / 'make-cold-x16-candidates.py'
        spec = importlib.util.spec_from_file_location('cold_x16_generator', generator)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result['candidate_source'] = module.generate(engine, out / 'cold-x16-candidates.h')
        result['fixture_dependencies'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in (generator, base / 'cold-x16-workload.h', base / 'cold-q5-batch.h', base / 'cold-q5-two-chains.h')}
        result['libraries'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in libraries.glob('libggml*.so.*') if p.is_file()}
        command += ['-DCOLD_X16', '-I' + str(out)]
        command += ['-I' + str(engine / p) for p in ('include', 'ggml/include', 'ggml/src', 'ggml/src/ggml-cpu')]
        command += ['-L' + str(libraries), '-lggml-cpu', '-lggml-base']
        environment.update(LD_LIBRARY_PATH=str(libraries), COLD_X16_KIND=args.kind[-1],
                           COLD_X16_K=str(args.k), COLD_X16_TOKENS=str(args.tokens))
    elif args.kind == 'graph':
        baseline = json.loads((base / 'results/glm53-draft-sweep-bandwidth-0905/result.json').read_text())
        assert baseline.get('finished') and not baseline.get('error')
        environment = {k:v for k,v in os.environ.items() if not k.startswith(('GGML_', 'LLAMA_', 'OMP_', 'GOMP_'))}
        environment.update({k:v for k,v in baseline['runtime_env'].items() if k.startswith('GGML_')})
        environment.update(LD_LIBRARY_PATH=str(libraries), OMP_NUM_THREADS='1',
                           COLD_GRAPH_K_PER_SOCKET=str(args.k), COLD_GRAPH_ROWS=str(args.graph_rows),
                           COLD_GRAPH_TOKENS=str(args.tokens), COLD_GRAPH_MATRICES=str(args.graph_matrices),
                           COLD_GRAPH_GROUP='1', COLD_GRAPH_CONTROLLERS=str(args.graph_controllers))
        result['fixture_environment'] = {k:v for k,v in environment.items() if k.startswith(('GGML_', 'OMP_', 'COLD_GRAPH_'))}
        result['libraries'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in libraries.glob('libggml*.so.*') if p.is_file()}
        result['fixture_dependencies'] = {str(engine / p): hashlib.sha256((engine / p).read_bytes()).hexdigest()
            for p in ('ggml/src/ggml-backend-meta.cpp', 'ggml/src/ggml-cpu/repack.cpp', 'ggml/src/ggml-cpu/arch/x86/repack.cpp')}
        command += ['-I' + str(engine / p) for p in ('include', 'ggml/include', 'ggml/src', 'ggml/src/ggml-cpu')]
        command += ['-L' + str(libraries), '-lggml', '-lggml-cpu', '-lggml-base']
    result['build_command'] = command
    for path in (source, Path(__file__).resolve()):
        (out / path.name).write_bytes(path.read_bytes())
    if args.kind in ('q4', 'q5'):
        for filename in ('cold-x16-workload.h', 'make-cold-x16-candidates.py', 'cold-q5-batch.h', 'cold-q5-two-chains.h'):
            (out / filename).write_bytes((base / filename).read_bytes())
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
        arm_environment = dict(environment)
        if args.kind in ('q4', 'q5'):
            arm_environment['COLD_X16_SMALL_SCALES'] = '1' if mode == 'small' else '0'
            if mode == 'small':
                command[1] = 'full'
        if args.kind == 'graph':
            chunk = re.match(r'\d+', mode).group()
            group = int(mode.split('g')[1]) if 'g' in mode else 1
            controllers = int(mode.split('c')[1]) if 'c' in mode else args.graph_controllers
            assert args.graph_matrices % group == 0
            command[1] = chunk
            arm_environment['COLD_GRAPH_GROUP'] = str(group)
            arm_environment['COLD_GRAPH_CONTROLLERS'] = str(controllers)
            command = ['taskset', '-c', '0-14,16-30,32-46,48-62'] + command
        recorder = None
        with (out / f'arm{index}-{mode}.log').open('w') as log:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=log, text=True, bufsize=1, start_new_session=True, env=arm_environment)
            owned = ProcessGroup(process)
            guard = InferenceContentionGuard(owned, snapshot, out / f'arm{index}-contention.json',
                                             interval=0.5, threshold=20, consecutive=1, grace=1)
            guard.start()
            try:
                run['ready'] = event(process, guard, 180 if args.kind == 'graph' else 60)
                assert run['ready']['event'] == 'ready' and run['ready']['pid'] == process.pid
                assert run['ready']['workers'] == 4 * args.workers_per_socket
                if args.kind == 'graph':
                    run['numa_placement'] = graph_placement(process.pid)
                    assert run['ready']['k'] == 4 * args.k and run['ready']['tokens'] == args.tokens
                    assert run['ready']['rows'] == args.graph_rows and run['ready']['matrices'] == args.graph_matrices
                    save()
                if args.profile_cycles:
                    run['measurement'], run['profile'] = profile_graph(process, guard, out / f'arm{index}-cycles')
                else:
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
        if args.kind == 'graph':
            log_text = (out / f'arm{index}-{mode}.log').read_text()
            assert 'fused in-graph all-reduce active' in log_text and 'x16: repacking' in log_text
            assert run['ready']['chunk'] == int(chunk) and run['ready']['group'] == group and run['ready']['bytes_per_pass'] == graph_weight_bytes
            assert run['ready']['controller_cores'] == controllers
            boundary_match = re.search(r'fused in-graph all-reduce active: (\d+) boundaries \((\d+) tensors\)', log_text)
            assert boundary_match, 'Missing fused reduction boundary counts'
            run['reduction_boundaries'], run['reduction_tensors'] = map(int, boundary_match.groups())
            assert run['reduction_boundaries'] == args.graph_matrices // group and run['reduction_tensors'] == args.graph_matrices
            assert all(run['ready'][key] == result['runs'][0]['ready'][key] for key in ('weight_hash', 'output_hash'))
        run['complete'] = True
        save()
        if args.profile_cycles:
            print(json.dumps(dict(index=index, mode=mode, profile=run['profile'])), flush=True)
        else:
            print(json.dumps({k: run[k] for k in ('index', 'mode', 'adjusted_read_gb_s', 'adjusted_total_gb_s',
                                                 'read_counter_over_logical')}), flush=True)
    assert result['source_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert result['binary_sha256'] == hashlib.sha256(binary.read_bytes()).hexdigest()
    if args.kind != 'read':
        for collection in ('fixture_dependencies', 'libraries'):
            assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest
                       for p, digest in result[collection].items()), f'{collection} changed during the measurement'
        if args.kind in ('q4', 'q5'):
            candidate = result['candidate_source']
            assert candidate['source_sha256'] == hashlib.sha256(Path(candidate['source']).read_bytes()).hexdigest()
            assert candidate['generated_sha256'] == hashlib.sha256((out / 'cold-x16-candidates.h').read_bytes()).hexdigest()
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
               for path, digest in result['guard_sources'].items()), 'A guard or runner input changed'
    result['after_state'] = ensure_idle()
except BaseException as error:
    result['error'] = repr(error)
    raise
finally:
    result['finished'] = time.time()
    save()
