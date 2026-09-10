#!/usr/bin/env python3
"""Use Full's existing allocation simulator; preserve the restored Qwen service."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from glm_flash_q8_trial import memory_status, node_memory_status
from model_measurement_guard import ModelMeasurementGuard
from qwen_high_quant_trial import unit_state
from qwen_split_trial import inference_snapshot, process_environment, process_info, runtime_environment, sha256

BASE = Path(__file__).resolve().parent
ENGINE = BASE.parents[1] / 'engines/llama.cpp-sr950-glm'
OUT = BASE / 'results/full-memory-plan-0910b'


def main():
    assert os.sched_getaffinity(0) == {127}
    previous_path = BASE / 'results/flash-bandwidth250-assessment-0910.json'
    assets_path = BASE / 'results/full-bandwidth250-assets-0910/result.json'
    previous, assets = [json.loads(path.read_text()) for path in (previous_path, assets_path)]
    assert previous['passed'] and assets['passed'] and not assets.get('error')
    pid = previous['restored_qwen_pid']
    peer = process_info(pid)
    assert peer['start'] == previous['restored_qwen_start']
    peer_env = runtime_environment(process_environment(pid))
    assert all(sha256(path) == digest for path, digest in assets['libraries'].items())
    target = assets['target_shards'][0]['path']
    for record in assets['target_shards']:
        stat = Path(record['path']).stat()
        assert (stat.st_size, stat.st_ino, stat.st_dev, stat.st_mtime_ns) == (
            record['size'], record['inode'], record['device'], record['mtime_ns'])
    library_dir = Path(assets['reference_command'][0]).parent
    source = BASE / 'full-memory-plan-0910.cpp'
    assert '#include "llama-ext.h"' in source.read_text()
    assert 'mp.no_alloc = true;' in source.read_text()
    assert 'mp.load_mode = LLAMA_LOAD_MODE_NONE;' in source.read_text()
    assert 'llama_decode(' not in source.read_text()
    paths = [Path(__file__).resolve(), source, previous_path, assets_path,
             BASE / 'qwen_split_trial.py', BASE / 'glm_flash_q8_trial.py',
             BASE / 'model_measurement_guard.py', BASE / 'qwen_high_quant_trial.py']
    paths += [ENGINE / name for name in ('include/llama.h', 'src/llama-ext.h',
              'ggml/include/ggml-backend.h', 'src/llama-model.cpp', 'src/llama-context.cpp', 'common/fit.cpp')]
    inputs = {str(path): sha256(path) for path in paths}
    inputs.update(assets['libraries'])
    environment = {key: value for key, value in os.environ.items()
                   if key not in runtime_environment(os.environ) and not key.startswith('LLAMA_ARG_')
                   and key != 'LD_PRELOAD'}
    environment.update(assets['runtime_env'])
    guard = ModelMeasurementGuard(pid, {pid: 18095}, inference_snapshot)

    def verify_peer():
        actual = process_info(pid)
        assert all(actual[key] == peer[key] for key in ('start', 'exe', 'command', 'cwd', 'affinity'))
        assert runtime_environment(process_environment(pid)) == peer_env
        assert set(inference_snapshot()) == {str(pid)}
        assert unit_state()['ActiveState'] == 'inactive'
        guard.assert_idle()

    def run(command, name):
        verify_peer()
        log_path = OUT / (name + '.log')
        with log_path.open('w') as log:
            child = subprocess.Popen(command, cwd=BASE, env=environment, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + 180
            while child.poll() is None:
                verify_peer()
                assert time.monotonic() < deadline, name + ' timed out'
                time.sleep(.5)
            assert child.returncode == 0, (name, child.returncode)
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=10)
        verify_peer()
        return dict(name=name, command=command, exit_code=child.returncode,
                    log_sha256=sha256(log_path), text=log_path.read_text())

    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_peer()
        assert all(sha256(path) == digest for path, digest in inputs.items())
        OUT.mkdir(exist_ok=False)
        result = dict(started=time.time(), passed=False, source_sha256=inputs, peer_pid=pid,
                      peer_start=peer['start'], source=source.name, model_started=False, steps=[])
        try:
            binary = OUT / 'full-memory-plan'
            compile_command = ['/usr/bin/c++', '-O2', '-std=c++17', '-DNDEBUG',
                '-I' + str(ENGINE / 'include'), '-I' + str(ENGINE / 'src'),
                '-I' + str(ENGINE / 'ggml/include'), str(source), '-L' + str(library_dir),
                '-Wl,-rpath,' + str(library_dir), '-lllama', '-lggml', '-lggml-base',
                '-pthread', '-o', str(binary)]
            compiled = run(compile_command, 'compile')
            compiled.pop('text')
            result['steps'].append(compiled)
            result['helper_sha256'] = sha256(binary)
            command = ['prlimit', '--as=8589934592', '--data=8589934592', '--cpu=120', '--',
                       'taskset', '-c', '0-127', str(binary), str(library_dir), target]
            simulated = run(command, 'simulation')
            records = [json.loads(line) for line in simulated.pop('text').splitlines()
                       if line.startswith('{"event":"full_memory_plan"')]
            result['steps'].append(simulated)
            estimate, = records
            assert estimate['no_alloc'] and not estimate['load_mtp'] and estimate['context'] == 32768
            assert estimate['peak_rss_bytes'] < 4 * 1024**3
            assert estimate['model_bytes'] > 300_000_000_000
            assert estimate['total_bytes'] == sum(estimate[key] for key in ('model_bytes', 'context_bytes', 'compute_bytes'))
            for key in ('model_bytes', 'context_bytes', 'compute_bytes'):
                assert estimate[key] == sum(row[key] for row in estimate['buffers'])
            status = Path(f'/proc/{pid}/status').read_text().splitlines()
            anonymous = next(int(line.split()[1]) * 1024 for line in status if line.startswith('RssAnon:'))
            memory, nodes = memory_status(), node_memory_status()
            estimated_available_after_pause = memory['MemAvailable'] + anonymous
            reserve = 32 * 1024**3
            result.update(passed=True, allocation_estimate=estimate, memory=memory, nodes=nodes,
                peer_anonymous_bytes=anonymous, estimated_available_after_peer_pause=estimated_available_after_pause,
                global_reserve_bytes=reserve,
                estimated_global_shortfall_bytes=max(0, estimate['total_bytes'] + reserve - estimated_available_after_pause),
                peer_preserved=True, full_inactive=True, no_weight_payload_loaded=True,
                scope='The existing Full no_alloc path constructs model/context descriptors and simulates buffer sizes. '
                      'The helper has an 8 GiB virtual-address limit and never calls decode. '
                      'This estimates raw target weights, 32768-token Q8 KV, and work buffers under the existing runtime. '
                      'It excludes a separately loaded MTP model, transient load copies and host bookkeeping. '
                      'Reported sizes are allocation estimates, not actual residency or throughput. '
                      'The selected Qwen process remains loaded and idle; no model files are changed.')
            assert all(sha256(path) == digest for path, digest in inputs.items())
            verify_peer()
        except BaseException as error:
            result['error'] = repr(error)
            raise
        finally:
            result['finished'] = time.time()
            (OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps({key: result.get(key) for key in ('passed', 'error', 'allocation_estimate',
                  'estimated_available_after_peer_pause', 'estimated_global_shortfall_bytes', 'peer_preserved')}, indent=2))


if __name__ == '__main__':
    os.umask(0o077)
    main()
