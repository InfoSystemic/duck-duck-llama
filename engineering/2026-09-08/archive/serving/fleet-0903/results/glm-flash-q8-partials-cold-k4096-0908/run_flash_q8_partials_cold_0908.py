#!/usr/bin/env python3
"""Guarded cold Q8 kernel comparison; never model bandwidth or goal completion."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import select
import signal
import statistics
import subprocess
import time

from benchmark_qwen_q6 import wait_background, host_cpu
from dram_bandwidth import PerfDramRecorder, summarize_samples
from flash_q8_partials_fixture_0908 import generate
from glm_flash_q8_trial import Manager, PORT, memory_status, node_memory_status, unit_state
from inference_contention_guard import InferenceContentionGuard, activity
from model_measurement_guard import ModelMeasurementGuard
from qwen_split_trial import inference_snapshot, sha256

BASE = Path(__file__).resolve().parent
ENGINE = BASE.parents[1] / 'engines/llama.cpp-glm5n-goal-0904'


class OwnedGroup:
    def __init__(self, process):
        self.process, self.pid = process, process.pid

    def poll(self):
        return self.process.poll()

    def terminate(self):
        if self.poll() is None:
            os.killpg(self.pid, signal.SIGTERM)

    def kill(self):
        if self.poll() is None:
            os.killpg(self.pid, signal.SIGKILL)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('label')
    parser.add_argument('--k', type=int, choices=(512, 2048, 4096), default=4096)
    parser.add_argument('--seconds', type=int, choices=range(6, 21), default=8)
    args = parser.parse_args()
    assert re.fullmatch(r'[A-Za-z0-9_-]+', args.label)
    os.umask(0o077)
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manager = Manager()
        current = manager.validate_current()
        assert current['q8_clamp'] and current['drafts'] == 0 and current['op_profile_count'] == 0
        assert not manager.qwen.state.get('current') and unit_state()['ActiveState'] == 'inactive'
        cpu = Path(current['cpu_library'])
        assert sha256(cpu) == current['cpu_sha256']
        assert current['cpu_sha256'] == '0f2feaaa6a173a23bf5adde440ec5541348f1cb7ac2edee7d77bebc15af9ec23'
        guard = ModelMeasurementGuard(current['pid'], {current['pid']: PORT}, inference_snapshot)
        guard.assert_idle()
        out = BASE / 'results' / args.label
        out.mkdir(exist_ok=False)
        result = dict(started=time.time(), config=vars(args), runs=[], passed=False,
            scope='Synthetic NR=1 cold kernel streaming, not model throughput, model bandwidth, or goal completion.',
            server_pid=current['pid'], cpu_library=str(cpu), cpu_sha256=current['cpu_sha256'],
            workers_per_socket=15, mib_per_worker=32, tokens=1,
            modes=['full', 'copy', 'p2', 'p4', 'p4', 'p2', 'copy', 'full'])

        def save():
            temporary = out / 'result.json.tmp'
            temporary.write_text(json.dumps(result, indent=2) + '\n')
            temporary.replace(out / 'result.json')

        def assert_maps(pid):
            paths = sorted({line.split()[-1] for line in Path(f'/proc/{pid}/maps').read_text().splitlines()
                            if 'libggml-cpu.so' in line})
            assert paths == [str(cpu)], paths
            return paths

        def ensure_idle(contention=None):
            state = guard.assert_idle()
            assert contention is None or contention.info is None, 'Inference contention detected'
            return state

        def pause(seconds, contention):
            start = time.monotonic()
            while time.monotonic() - start < seconds:
                ensure_idle(contention)
                time.sleep(0.5)
            return [start, time.monotonic()]

        def event(process, contention, timeout):
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                ensure_idle(contention)
                ready, _, _ = select.select([process.stdout], [], [], 0.5)
                if ready:
                    line = process.stdout.readline()
                    assert line, f'Fixture exited: {process.poll()}'
                    return json.loads(line)
            raise TimeoutError('Private fixture event timeout')

        def stop(owned):
            owned.terminate()
            try:
                owned.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                owned.kill()
                owned.process.wait(timeout=3)

        save()
        try:
            result['model_maps'] = assert_maps(current['pid'])
            result['idle_gate'] = guard.wait_idle(out / 'idle.json', quiet_seconds=15)
            result['background_gate'] = wait_background(guard, current['pid'], 5, out / 'background.json')
            result['memory_before'] = memory_status()
            result['nodes_before'] = node_memory_status()
            assert result['memory_before']['MemAvailable'] > 32 * 1024**3
            assert all(v['estimated_available'] > 8 * 1024**3 for v in result['nodes_before'].values())
            result['candidate_source'] = generate(out, sha256)
            source = out / 'cold-q8-check.cpp'
            original = BASE / 'read-bandwidth-check.cpp'
            contents = original.read_text()
            assert contents.count('#include "cold-x16-workload.h"') == 1
            source.write_text(contents.replace('#include "cold-x16-workload.h"', '#include "cold-q8-workload.h"'))
            fixture = BASE / 'cold-q8-workload.h'
            (out / fixture.name).write_bytes(fixture.read_bytes())
            inputs = [Path(__file__).resolve(), BASE / 'flash_q8_partials_fixture_0908.py', original,
                      fixture, BASE / 'dram_bandwidth.py', BASE / 'model_measurement_guard.py',
                      BASE / 'inference_contention_guard.py', BASE / 'benchmark_qwen_q6.py']
            result['input_sha256'] = {str(p): sha256(p) for p in inputs}
            for p in inputs:
                if p.name != fixture.name:
                    (out / p.name).write_bytes(p.read_bytes())
            result['headers_sha256'] = {str(p): sha256(p) for subdir in ('ggml/src', 'ggml/include')
                                       for p in (ENGINE / subdir).rglob('*.h')}
            binary = out / 'cold-q8-check'
            pinned = Path(current['pinned_directory'])
            base_library = (pinned / 'libggml-base.so').resolve()
            result['base_library'] = dict(path=str(base_library), sha256=sha256(base_library))
            command = ['g++', '-O3', '-DNDEBUG', '-std=gnu++17', '-march=native', '-fPIC', '-pthread',
                       '-DCOLD_X16', '-I' + str(out)]
            command += ['-I' + str(ENGINE / p) for p in ('include', 'ggml/include', 'ggml/src', 'ggml/src/ggml-cpu')]
            command += [str(source), str(cpu), str(base_library), '-o', str(binary)]
            result['build_command'] = command
            save()
            with (out / 'build.log').open('w') as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                owned = OwnedGroup(process)
                contention = InferenceContentionGuard(owned, inference_snapshot, out / 'build-contention.json',
                                                       interval=0.5, threshold=20, consecutive=1, grace=1)
                contention.start()
                try:
                    deadline = time.monotonic() + 120
                    while process.poll() is None:
                        ensure_idle(contention)
                        assert time.monotonic() < deadline, 'Build timeout'
                        time.sleep(0.5)
                    result['build_exit_code'] = process.returncode
                    assert process.returncode == 0, 'Build failed; see build.log'
                finally:
                    stop(owned)
                    result['build_contention'] = contention.stop()
                    save()
            result['binary_sha256'] = sha256(binary)
            environment = {k: v for k, v in os.environ.items()
                           if not k.startswith(('GGML_', 'LLAMA_', 'OMP_', 'GOMP_', 'COLD_'))}
            environment.update(LD_LIBRARY_PATH=str(cpu.parent) + ':' + str(pinned),
                               OMP_NUM_THREADS='1', COLD_Q8_K=str(args.k))
            for index, mode in enumerate(result['modes']):
                before = inference_snapshot()
                before_time = time.monotonic()
                run = dict(index=index, mode=mode, started=time.time(), state_before=ensure_idle())
                result['runs'].append(run)
                save()
                recorder = None
                with (out / f'arm{index}-{mode}.log').open('w') as log:
                    process = subprocess.Popen([str(binary), mode, '15', '32', str(args.seconds)],
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True,
                        bufsize=1, start_new_session=True, env=environment)
                    owned = OwnedGroup(process)
                    contention = InferenceContentionGuard(owned, inference_snapshot, out / f'arm{index}-contention.json',
                                                           interval=0.5, threshold=20, consecutive=1, grace=1)
                    contention.start()
                    try:
                        run['ready'] = event(process, contention, 90)
                        assert run['ready']['event'] == 'ready' and run['ready']['pid'] == process.pid
                        assert run['ready']['workers'] == 60 and run['ready']['verified_page_samples'] == 3840
                        run['cpu_maps'] = assert_maps(process.pid)
                        recorder = PerfDramRecorder(out / f'arm{index}-imc').start()
                        run['background_before'] = pause(5, contention)
                        host_before, host_start = {str(p): v for p, v in host_cpu().items()}, time.monotonic()
                        process.stdin.write('go\n')
                        process.stdin.flush()
                        run['measurement'] = measured = event(process, contention, args.seconds + 20)
                        other, churn = activity(host_before, {str(p): v for p, v in host_cpu().items()}, time.monotonic() - host_start,
                                                own_pid=process.pid)
                        run['other_host_activity'] = sorted(other, key=lambda x: -x['cpu_percent'])[:12]
                        run['host_churn'] = churn
                        assert measured['event'] == 'done' and measured['checksums_exact']
                        run['background_after'] = pause(5, contention)
                        samples, run['counter_metadata'] = recorder.stop()
                        recorder = None
                        assert run['counter_metadata']['valid'] and run['counter_metadata']['exit_code'] == 0
                        start = max(t['start'] for t in measured['threads']) + 1
                        end = min(t['end'] for t in measured['threads']) - 1
                        run['stable_window'] = [start, end]
                        run['dram'] = dram = summarize_samples(samples, start, end)
                        run['backgrounds'] = backgrounds = [summarize_samples(samples, a + .5, b - .5)
                            for a, b in (run['background_before'], run['background_after'])]
                        assert dram['valid'] and all(b['valid'] for b in backgrounds)
                        run['adjusted_read_gb_s'] = dram['read_gb_s'] - max(b['read_gb_s'] for b in backgrounds)
                        run['adjusted_total_gb_s'] = dram['total_gb_s'] - max(b['total_gb_s'] for b in backgrounds)
                        process.stdin.write('exit\n')
                        process.stdin.flush()
                        assert process.wait(timeout=10) == 0
                    finally:
                        stop(owned)
                        if recorder is not None:
                            recorder.stop()
                        run['contention'] = contention.stop()
                        run['finished'] = time.time()
                        save()
                other, churn = activity(before, inference_snapshot(), time.monotonic() - before_time)
                run.update(other_inference=other, inference_churn=churn, state_after=ensure_idle())
                assert not run['contention'] and not churn and not any(s['cpu_percent'] > 5 for s in other)
                run['complete'] = True
                save()
                print(json.dumps(dict(index=index, mode=mode, logical_gb_s=measured['logical_gb_s'],
                    adjusted_read_gb_s=run['adjusted_read_gb_s'], checksums_exact=True)), flush=True)
            for key in ('input_sha256', 'headers_sha256'):
                assert all(sha256(Path(p)) == digest for p, digest in result[key].items())
            assert sha256(cpu) == result['cpu_sha256'] and sha256(base_library) == result['base_library']['sha256']
            assert sha256(binary) == result['binary_sha256']
            manager.validate_current()
            assert_maps(current['pid'])
            result['state_after'] = ensure_idle()
            result['memory_after'] = memory_status()
            result['summary'] = {mode: dict(
                logical_gb_s=statistics.median(r['measurement']['logical_gb_s'] for r in result['runs'] if r['mode'] == mode),
                adjusted_read_gb_s=statistics.median(r['adjusted_read_gb_s'] for r in result['runs'] if r['mode'] == mode))
                for mode in ('full', 'copy', 'p2', 'p4')}
            result['passed'] = True
            print(json.dumps(dict(passed=True, summary=result['summary'], model_result=False)), flush=True)
        except BaseException as error:
            result['error'] = repr(error)
            raise
        finally:
            result['finished'] = time.time()
            save()


if __name__ == '__main__':
    main()
