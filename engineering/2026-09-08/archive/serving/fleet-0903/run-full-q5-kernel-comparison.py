#!/usr/bin/env python3
"""Compare Q5 kernels one case at a time, yielding to existing inference."""
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
parser.add_argument('--cases', default=','.join(map(str, range(24))))
parser.add_argument('--repeats', type=int, default=50)
parser.add_argument('--socket', type=int, choices=range(4), default=0)
options = parser.parse_args()
assert re.fullmatch(r'[a-zA-Z0-9_-]+', options.label)
selected_cases = list(map(int, options.cases.split(',')))
assert selected_cases and len(set(selected_cases)) == len(selected_cases) and all(0 <= i < 24 for i in selected_cases)
assert 1 <= options.repeats <= 1000
base = Path(__file__).resolve().parent
root = base.parents[1]
out = base / 'results' / options.label
out.mkdir(exist_ok=False)
engines = {'full': root / 'engines/llama.cpp-sr950-glm',
           'new': root / 'engines/llama.cpp-glm5n-goal-0904'}
bins = {'full': engines['full'] / 'build-dev2/bin', 'new': engines['new'] / 'build-goal/bin'}
allowed = {'4005448': 18091, '2308651': 18095}
prior = json.loads((base / 'results/glm53-decode-profile-0905c/result.json').read_text())
assert prior.get('finished') and not prior.get('error') and len(prior['profiles']) == 2
arms = ['full', 'legacy', 'compact', 'compact', 'legacy', 'full']
primary_indices = [0, 2, 10, 14]
case_order = [i for i in primary_indices if i in selected_cases] + [i for i in selected_cases if i not in primary_indices]
result = dict(started=time.time(), pid=os.getpid(),
              scope='Single-socket Q5 kernels with identical synthetic canonical weights; no full-model performance claim.',
              fixture_sha256=hashlib.sha256((base / 'iq2-repack-check.cpp').read_bytes()).hexdigest(),
              config=vars(options), binaries={}, runs=[], records=[], ratios=[], case_attempts=[], idle_gates=[])


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
            args = (proc / 'cmdline').read_bytes().split(b'\0')
            ports = [args[i + 1].decode() for i, x in enumerate(args[:-1]) if x == b'--port']
            state[proc.name] = (int(fields[11]) + int(fields[12]), fields[19], ports[-1] if ports else None)
        except (OSError, ValueError, IndexError):
            pass
    return state


def server_state():
    current = snapshot()
    assert set(allowed) <= set(current), 'An expected protected server is missing'
    states = {}
    for pid, port in allowed.items():
        assert current[pid][2] == str(port), 'An expected server port changed'
        def get(path):
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/{path}', timeout=3) as response:
                return response.read().decode()
        slots = json.loads(get('slots'))
        metrics = get('metrics').splitlines()
        queue = [float(line.split()[-1]) for line in metrics
                 if not line.startswith('#') and 'requests_deferred' in line]
        assert len(queue) == 1, 'Queue metric missing'
        states[pid] = dict(processing=any(slot['is_processing'] for slot in slots), queued=queue[0])
    return dict(servers=states, foreign_pids=sorted(set(current) - set(allowed)),
                busy=bool(set(current) - set(allowed)) or any(s['processing'] or s['queued'] for s in states.values()))


def idle_gate():
    while True:
        status = wait_for_idle(snapshot, out / 'waiting-for-idle.json', allowed_idle_pids=tuple(allowed))
        state = server_state()
        result['idle_gates'].append(dict(status=status, state=state))
        save()
        if not state['busy']:
            return


class Contention(Exception):
    pass


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


def run(command, label, env):
    info = dict(label=label, command=command, started=time.time(), before_state=server_state())
    result['runs'].append(info)
    save()
    if info['before_state']['busy']:
        info.update(discarded='Inference was active before launch', finished=time.time())
        save()
        raise Contention(info['discarded'])
    before = snapshot()
    before_time = time.monotonic()
    with (out / (label + '.log')).open('w') as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        owned = ProcessGroup(process)
        guard = InferenceContentionGuard(owned, snapshot, out / (label + '-contention.json'),
                                         interval=0.5, threshold=20, consecutive=1, grace=1)
        guard.start()
        try:
            info['exit'] = process.wait(timeout=900)
        except BaseException:
            owned.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                owned.kill()
                process.wait()
            raise
        finally:
            info.update(contention=guard.stop(), finished=time.time())
            samples, churn = activity(before, snapshot(), time.monotonic() - before_time)
            info.update(other_inference=samples, process_churn=churn)
            save()
    info['after_state'] = server_state()
    if info['contention'] or churn or any(s['cpu_percent'] > 5 for s in samples) or info['after_state']['busy']:
        info['discarded'] = 'Inference activity overlapped this subprocess'
        save()
        raise Contention(info['discarded'])
    save()
    assert info['exit'] == 0, (label, info['exit'])
    return (out / (label + '.log')).read_text()


environment = {k: v for k, v in os.environ.items()
               if not k.startswith(('GGML_', 'LLAMA_GRAPH_PHASE', 'LLAMA_MTP_', 'OMP_', 'GOMP_', 'REPACK_TEST_'))}
environment.update(GGML_CPU_NUMA_DEVICES='0', GGML_CPU_X16_Q5_K='1', GGML_CPU_Q5_K_REPACK_MOE_DOWN='1',
                   GGML_CPU_X16_Q5_BYTES='0', GGML_CPU_X16_Q5_COMPACT_P2='0', GGML_CPU_X16_CHUNK_MAX='64',
                   REPACK_TEST_FULL_GLM_Q5='1', REPACK_TEST_DOWN='1', REPACK_TEST_THREADS='15',
                   REPACK_TEST_PERSISTENT_POOL='1', REPACK_TEST_PIN_POOL='1',
                   REPACK_TEST_REPEATS=str(options.repeats), REPACK_TEST_TIMING_MEDIAN='1')
save()
try:
    idle_gate()
    for name, engine in engines.items():
        binary = out / (name + '-q5-check')
        command = ['g++', '-O2', '-std=c++17']
        command += ['-I' + str(engine / p) for p in ('include', 'ggml/include', 'ggml/src', 'ggml/src/ggml-cpu')]
        command += [str(base / 'iq2-repack-check.cpp'), '-L' + str(bins[name]), '-lggml', '-lggml-cpu',
                    '-lggml-base', '-ldl', '-pthread', '-o', str(binary)]
        attempt = 0
        while True:
            try:
                run(command, f'build-{name}-{attempt}', dict(environment, LD_LIBRARY_PATH=str(bins[name])))
                break
            except Contention:
                attempt += 1
                idle_gate()
        result['binaries'][name] = dict(executable=str(binary), libraries=str(bins[name]),
            sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in bins[name].glob('libggml*.so.*') if p.is_file()})
        save()
    for padded in (False, True):
        for case_index in case_order:
            attempt = 0
            while True:
                progress = dict(padded=padded, case_index=case_index, attempt=attempt, started=time.time())
                result['case_attempts'].append(progress)
                records = []
                try:
                    for index, arm in enumerate(arms):
                        name = 'full' if arm == 'full' else 'new'
                        env = dict(environment, LD_LIBRARY_PATH=str(bins[name]), REPACK_TEST_FULL_CASE_INDEX=str(case_index),
                                   GGML_CPU_X16_Q5_COMPACT_P4='1' if arm == 'compact' else '0',
                                   GGML_CPU_X16_Q5_BYTES_BATCH3='1' if arm == 'compact' else '0')
                        if padded:
                            env['REPACK_TEST_PADDED'] = '1'
                        label = f'p{int(padded)}-c{case_index}-a{attempt}-{index}-{arm}'
                        first_cpu = 16 * options.socket
                        log = run(['numactl', f'--physcpubind={first_cpu}-{first_cpu + 14}', f'--membind={options.socket}',
                                   str(out / (name + '-q5-check')), 'q5-pair'], label, env)
                        outputs = [dict(re.findall(r'(\w+)=([^ ]+)', line)) for line in log.splitlines() if line.startswith('PASS ')]
                        inputs = [dict(re.findall(r'(\w+)=([^ ]+)', line)) for line in log.splitlines() if line.startswith('FULL_INPUT ')]
                        assert len(outputs) == len(inputs) == 1, (label, len(outputs), len(inputs))
                        values, canonical = outputs[0], inputs[0]
                        assert int(canonical['case_index']) == case_index
                        assert int(values['threads']) == 15
                        for key in ('k', 'rows', 'threads', 'tokens', 'moe'):
                            assert values[key] == canonical[key]
                        records.append(dict(arm=arm, index=index, padded=padded, case_index=case_index,
                            k=int(values['k']), rows=int(values['rows']), tokens=int(values['tokens']), moe=int(values['moe']),
                            ms=float(values['packed_ms']), output_hash=values['hash'], weight_digest=canonical['digest']))
                    assert len({r['weight_digest'] for r in records}) == 1, 'Canonical weights differ across libraries'
                    assert len({r['output_hash'] for r in records}) == 1, 'Packed outputs differ across libraries'
                    medians = {arm: statistics.median(r['ms'] for r in records if r['arm'] == arm) for arm in set(arms)}
                    ratio = {k: records[0][k] for k in ('padded', 'case_index', 'k', 'rows', 'tokens', 'moe')}
                    ratio.update(ms=medians, full_over_compact=medians['full'] / medians['compact'],
                                 legacy_over_compact=medians['legacy'] / medians['compact'])
                    result['records'].extend(records)
                    result['ratios'].append(ratio)
                    result['exact_cases'] = len(result['ratios'])
                    progress.update(finished=time.time(), complete=True)
                    save()
                    print('Case complete', json.dumps(ratio), flush=True)
                    break
                except Contention as error:
                    progress.update(finished=time.time(), discarded=str(error))
                    save()
                    print(f'Case {case_index} interrupted; completed cases retained. Waiting for inference.', flush=True)
                    attempt += 1
                    idle_gate()
    assert result['exact_cases'] == 2 * len(selected_cases)
    primary = [r for r in result['ratios'] if r['case_index'] in primary_indices]
    assert len(primary) == 2 * len(set(primary_indices) & set(selected_cases))
    if primary:
        result['primary_full_over_compact_median'] = statistics.median(r['full_over_compact'] for r in primary)
        print('Primary Full/compact median ratio', result['primary_full_over_compact_median'], flush=True)
    assert result['fixture_sha256'] == hashlib.sha256((base / 'iq2-repack-check.cpp').read_bytes()).hexdigest()
    for name, manifest in result['binaries'].items():
        assert all(hashlib.sha256((bins[name] / filename).read_bytes()).hexdigest() == digest
                   for filename, digest in manifest['sha256'].items()), 'A compared library changed during the experiment'
except Exception as error:
    result['error'] = repr(error)
    raise
finally:
    result['finished'] = time.time()
    save()
