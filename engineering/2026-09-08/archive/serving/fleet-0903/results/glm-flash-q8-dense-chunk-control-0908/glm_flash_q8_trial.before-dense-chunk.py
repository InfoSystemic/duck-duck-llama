#!/usr/bin/env python3
"""Manage whole-server Flash Q8 trials, retaining the selected Qwen Q6 rollback."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

from extract_glm_flash_q8_mtp_0908 import verified_sources
from model_measurement_guard import ModelMeasurementGuard, read_service
from qwen_high_quant_trial import (Manager as QwenManager, CONTEXT, atomic_json,
                                  expected, port_available, set_option, unit_state)
from qwen_split_trial import (ExactProcess, inference_snapshot, process_environment,
                             process_info, runtime_environment, sha256)

BASE = Path(__file__).resolve().parent
OUT = BASE / 'results/glm-flash-q8-trial-0908'
STATE = OUT / 'state.json'
PORT = 18131


def memory_status():
    values = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        if key in ('MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree', 'Shmem'):
            values[key] = int(value.split()[0]) * 1024
    return values


def node_memory_status():
    result = {}
    for path in sorted(Path('/sys/devices/system/node').glob('node[0-9]*')):
        values = {}
        for line in (path / 'meminfo').read_text().splitlines():
            fields = line.split()
            key = fields[2].rstrip(':')
            if key in ('MemFree', 'FilePages', 'Shmem', 'SReclaimable', 'AnonPages'):
                values[key] = int(fields[3]) * 1024
        values['estimated_available'] = values['MemFree'] + max(0, values['FilePages'] - values['Shmem']) + values['SReclaimable']
        result[path.name] = values
    return result


class Manager:
    def __init__(self):
        self.state = json.loads(STATE.read_text()) if STATE.exists() else dict(events=[], current=None)
        self.cancelled = False
        self.recovering = False
        self.qwen = QwenManager()

    def check_cancel(self):
        if self.cancelled and not self.recovering:
            raise InterruptedError('Flash Q8 trial cancelled')

    def record(self, event, **values):
        row = dict(time=time.time(), event=event, **values)
        self.state['events'].append(row)
        atomic_json(STATE, self.state)
        print(json.dumps(row), flush=True)

    def candidate(self, batching, drafts, workers, q8_experts=False, op_profile_count=0, q8_clamp=False, pooling=False, expert_tile_rows=0):
        assert not q8_experts or batching
        assert not q8_clamp or q8_experts
        assert not pooling or q8_clamp
        assert expert_tile_rows in (0, 32, 48, 64) and (not expert_tile_rows or pooling)
        assert 0 <= op_profile_count <= 256
        download = verified_sources()
        mtp = json.loads((BASE / 'results/glm-flash-q8-mtp-0908/result.json').read_text())
        assert mtp['passed'] and mtp['revision'] == download['revision']
        draft_path = Path(mtp['destination'])
        info = draft_path.stat()
        assert info.st_size == mtp['bytes'] and info.st_ino == mtp['inode'] and info.st_mtime_ns == mtp['mtime_ns']
        assert mtp['type_counts'] == {'Q8_0': 18, 'F32': 14}
        preset = json.loads((BASE / 'glm-flash-validated.json').read_text())
        pinned = Path(preset['command'][0]).parent
        for name, digest in preset['binary_sha256'].items():
            assert sha256(pinned / name) == digest, name
        command = set_option(preset['command'], '--model', download['records'][0]['logical_path'])
        command = set_option(command, '--spec-draft-model', draft_path)
        command = set_option(command, '--spec-draft-n-max', drafts)
        command = set_option(command, '--threads', workers)
        command = set_option(command, '--threads-batch', workers)
        command = set_option(command, '--alias', 'glm-flash-q8-trial,glm-flash-goal')
        if drafts == 0:
            for flag in [value for value in command if value.startswith('--spec-')]:
                command = set_option(command, flag, None)
        # Validate default-off continuations before enabling reusable prompt state.
        command.append('--no-cache-prompt')
        env = {k: v for k, v in os.environ.items() if k not in runtime_environment(os.environ)}
        env.update(preset['runtime_env'])
        env['GGML_CPU_NUMA_THREADS'] = str(workers)
        cpu = pinned / 'libggml-cpu.so.0.22.0'
        cpu_sha = preset['binary_sha256'][cpu.name]
        if batching:
            batch = json.loads((BASE / 'results/glm-flash-q8-batch-0908/result.json').read_text())
            attention = json.loads((BASE / 'results/glm-flash-q8-attention3d-0908/result.json').read_text())
            assert batch['passed'] and attention['passed']
            assert sum(x['bit_exact_cases'] for x in batch['checks'] if x['type'] == 'q8') == 432
            assert attention['bit_exact_cases'] == 144
            assert batch['library_sha256'] == attention['library_sha256'] == sha256(batch['library'])
            assert batch['reference_cpu_sha256'] == cpu_sha
            cpu, cpu_sha = Path(batch['library']), batch['library_sha256']
            env['LD_LIBRARY_PATH'] = str(cpu.parent) + ':' + str(pinned)
            env['GGML_CPU_X16_Q8_BATCH'] = '1'
        if q8_experts:
            expert = json.loads((BASE / 'results/glm-flash-q8-experts-0908/result.json').read_text())
            prefill = json.loads((BASE / 'results/glm-flash-q8-experts-prefill-0908/result.json').read_text())
            assert expert['passed'] and prefill['passed']
            assert expert['experts'] == prefill['experts'] == 288 and expert['capacity'] == 512
            assert expert['parent_sha256'] == cpu_sha
            assert prefill['tokens'] == 64 and all(x['all_288_active'] for x in prefill['runs'])
            assert len(expert['comparisons']) == 45 and len(prefill['comparisons']) == 3
            assert expert['library_sha256'] == prefill['library_sha256'] == sha256(expert['library'])
            cpu, cpu_sha = Path(expert['library']), expert['library_sha256']
            env['LD_LIBRARY_PATH'] = str(cpu.parent) + ':' + str(pinned)
            env['GGML_CPU_X16_Q8_EXPERTS'] = '1'
        if q8_clamp:
            clamp = json.loads((BASE / 'results/glm-flash-q8-clamp-0908/result.json').read_text())
            prefill = json.loads((BASE / 'results/glm-flash-q8-clamp-prefill-0908/result.json').read_text())
            assert clamp['passed'] and prefill['passed']
            assert clamp['parent_sha256'] == prefill['baseline_sha256'] == cpu_sha
            assert clamp['experts'] == prefill['experts'] == 288 and clamp['capacity'] == 512
            assert prefill['tokens'] == 64 and all(x['all_288_active'] for x in prefill['runs'])
            assert len(clamp['comparisons']) == 63 and len(prefill['comparisons']) == 3
            assert all(x['bit_exact'] for x in clamp['comparisons'] + prefill['comparisons'])
            assert sum(x['values'] for x in clamp['comparisons'] + prefill['comparisons']) == 4915200
            assert clamp['library_sha256'] == prefill['library_sha256'] == sha256(clamp['library'])
            cpu, cpu_sha = Path(clamp['library']), clamp['library_sha256']
            env['LD_LIBRARY_PATH'] = str(cpu.parent) + ':' + str(pinned)
            assert env['GGML_CPU_MOE_CLAMP_FUSION'] == '1'
            env['GGML_CPU_X16_Q8_CLAMP_FUSION'] = '1'
            bundle = json.loads((BASE / 'results/glm-flash-q8-clamp-runtime-0908/result.json').read_text())
            assert bundle['passed'] and bundle['cpu_sha256'] == cpu_sha
            assert bundle['server_sha256'] == preset['binary_sha256']['llama-server'] == sha256(bundle['server'])
            private_probe = next(x for x in bundle['runs'] if x['mode'] == 'private')
            assert private_probe['cpu_maps'] == [str(cpu.resolve())]
            assert all(f'CPU-NUMA{i}' in private_probe['devices'] for i in range(4))
            assert all(Path(path).is_symlink() and str(Path(path).resolve()) == target
                       for path, target in bundle['symlinks'].items())
            command[0] = bundle['server']
        if pooling:
            root = BASE / 'results/glm-flash-q8-pool-0908c'
            pool = json.loads((root / 'result.json').read_text())
            manifest = json.loads((root / 'private-cpu/manifest.json').read_text())
            assert pool['passed'] and pool['bit_exact'] and pool['unpatched_text_identical']
            assert pool['parent_sha256'] == manifest['parent_sha256'] == cpu_sha
            assert {x['label'] for x in pool['checks']} == {'parent', 'off', 'on', 'disabled'}
            assert all(x['cases'] == 28 and x['samples_per_case'] == 3 and x['output_bytes'] == 8506368
                       for x in pool['checks'])
            assert len({x['output_sha256'] for x in pool['checks']}) == 1
            assert all(sha256(root / (x['label'] + '.bin')) == x['output_sha256'] for x in pool['checks'])
            assert all(sha256(path) == digest for path, digest in manifest['input_sha256'].items())
            assert all(sha256(path) == digest for path, digest in manifest['private_source_sha256'].items())
            assert pool['library_sha256'] == manifest['library_sha256'] == sha256(pool['library'])
            cpu, cpu_sha = Path(pool['library']), pool['library_sha256']
            env['LD_LIBRARY_PATH'] = str(cpu.parent) + ':' + str(pinned)
            env['GGML_CPU_SOFTMAX_POOL_FUSION'] = '1'
            bundle = json.loads((BASE / 'results/glm-flash-q8-pool-runtime-0908/result.json').read_text())
            assert bundle['passed'] and bundle['cpu_sha256'] == cpu_sha
            assert bundle['server_sha256'] == preset['binary_sha256']['llama-server'] == sha256(bundle['server'])
            assert bundle['runtime_env'] == runtime_environment(env)
            private_probe, = bundle['runs']
            assert private_probe['mode'] == 'private' and private_probe['cpu_maps'] == [str(cpu.resolve())]
            assert all(f'CPU-NUMA{i}' in private_probe['devices'] for i in range(4))
            assert all(Path(path).is_symlink() and str(Path(path).resolve()) == target
                       for path, target in bundle['symlinks'].items())
            command[0] = bundle['server']
        if expert_tile_rows:
            root = BASE / 'results/glm-flash-q8-expert-tiles-0908'
            tiles = json.loads((root / 'result.json').read_text())
            manifest = json.loads((root / 'private-cpu/manifest.json').read_text())
            rotating = json.loads((BASE / 'results/glm-flash-q8-expert-tiles-rotating-0908/result.json').read_text())
            assert tiles['passed'] and rotating['passed']
            assert tiles['parent_sha256'] == manifest['parent_sha256'] == cpu_sha
            assert len(tiles['comparisons']) == 90 and len(rotating['comparisons']) == 432
            assert all(x['bit_exact'] for x in tiles['comparisons'] + rotating['comparisons'])
            assert all(sha256(path) == digest for path, digest in manifest['input_sha256'].items())
            assert all(sha256(path) == digest for path, digest in manifest['private_source_sha256'].items())
            assert all(sha256(path) == digest for path, digest in rotating['input_sha256'].items())
            assert tiles['library_sha256'] == manifest['library_sha256'] == sha256(tiles['library'])
            cpu, cpu_sha = Path(tiles['library']), tiles['library_sha256']
            env['LD_LIBRARY_PATH'] = str(cpu.parent) + ':' + str(pinned)
            env['GGML_CPU_Q8_MOE_TILE_ROWS'] = str(expert_tile_rows)
            bundle = json.loads((BASE / 'results/glm-flash-q8-expert-tiles-runtime-0908/result.json').read_text())
            assert bundle['passed'] and bundle['cpu_sha256'] == cpu_sha
            assert bundle['server_sha256'] == preset['binary_sha256']['llama-server'] == sha256(bundle['server'])
            probe, = [x for x in bundle['runs'] if x['tile_rows'] == expert_tile_rows]
            assert probe['mode'] == 'private' and probe['cpu_maps'] == [str(cpu.resolve())]
            assert probe['runtime_env'] == runtime_environment(env)
            assert all(f'CPU-NUMA{i}' in probe['devices'] for i in range(4))
            assert all(Path(path).is_symlink() and str(Path(path).resolve()) == target
                       for path, target in bundle['symlinks'].items())
            command[0] = bundle['server']
        assert not any('REQUANT' in key or key == 'GGML_CPU_ROUTER_F16' for key in runtime_environment(env))
        assert not any('PROFILE' in key for key in runtime_environment(env))
        if op_profile_count:
            arm_file = OUT / 'op-profile.arm'
            assert not arm_file.exists(), 'An operation capture is already armed'
            env.update(GGML_CPU_OP_PROFILE='*', GGML_CPU_OP_PROFILE_ARM_FILE=str(arm_file),
                       GGML_CPU_OP_PROFILE_SKIP='0', GGML_CPU_OP_PROFILE_COUNT=str(op_profile_count))
        # NUMA backends pin their own workers. Preserve the whole-host discovery
        # mask so loader/controller threads can also use the reserved cores.
        affinity = list(self.qwen.state['original_qwen']['affinity'])
        assert affinity == list(range(128))
        configuration = dict(command=command, runtime_env=runtime_environment(env),
            batching=batching, q8_experts=q8_experts, drafts=drafts, workers=workers, affinity=affinity,
            op_profile_count=op_profile_count, q8_clamp=q8_clamp, pooling=pooling, expert_tile_rows=expert_tile_rows,
            cpu_library=str(cpu), cpu_sha256=cpu_sha, pinned_directory=str(pinned),
            runtime_directory=str(Path(command[0]).parent),
            model_revision=download['revision'], model_manifest_sha256=download['manifest_sha256'],
            draft_sha256=mtp['sha256'], temporary_storage=True)
        return configuration, env

    def validate_current(self):
        current = self.state['current']
        info = process_info(current['pid'])
        assert expected(info) == expected(current['info'])
        assert info['command'] == current['command']
        assert runtime_environment(process_environment(current['pid'])) == current['runtime_env']
        return current

    def quiet(self):
        if self.state.get('current'):
            current = self.validate_current()
            guard = ModelMeasurementGuard(current['pid'], {current['pid']: PORT}, inference_snapshot)
            guard.wait_idle(OUT / 'waiting-for-idle.json', quiet_seconds=15)
            guard.assert_idle()
        elif self.qwen.state.get('current'):
            self.qwen.quiet()
        else:
            assert not inference_snapshot(), 'Another inference process is loaded'
        self.check_cancel()

    def stop_flash(self):
        if not self.state.get('current'):
            return
        current = self.validate_current()
        ModelMeasurementGuard(current['pid'], {current['pid']: PORT}, inference_snapshot).assert_idle()
        proc = ExactProcess(current['pid'], expected(current['info']), set())
        try:
            proc.terminate()
            self.record('stopping_idle_flash', pid=current['pid'])
            deadline = time.monotonic() + 180
            while not proc.exited():
                if time.monotonic() > deadline:
                    raise TimeoutError('Idle Flash shutdown did not finish')
                time.sleep(.5)
        finally:
            proc.close()
        self.state['current'] = None
        self.record('flash_stopped', pid=current['pid'])

    def start_flash(self, configuration, environment):
        assert not inference_snapshot() and port_available(PORT)
        assert self.qwen.state['full_stopped'] and unit_state()['ActiveState'] == 'inactive'
        before = memory_status()
        # Includes target repacking, the draft, work buffers, and a reserve.
        # tmpfs file backing is already excluded from MemAvailable.
        assert before['MemAvailable'] > 410000000000, before
        node_before = node_memory_status()
        assert len(node_before) == 4
        assert all(v['estimated_available'] > 106000000000 for v in node_before.values()), node_before
        self.check_cancel()
        log_path = OUT / f'flash-{time.time_ns()}.log'
        with log_path.open('w') as log:
            proc = subprocess.Popen(['taskset', '-c', ','.join(map(str, configuration['affinity'])), *configuration['command']],
                env=environment, cwd=BASE, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
        self.record('loading_flash', pid=proc.pid, batching=configuration['batching'],
                    drafts=configuration['drafts'], log=str(log_path), memory_before=before, node_memory_before=node_before)
        try:
            deadline, report = time.monotonic() + 3600, 0
            while True:
                self.check_cancel()
                assert proc.poll() is None, ('Flash exited while loading', proc.returncode)
                assert not (set(inference_snapshot()) - {str(proc.pid)}), 'Another inference process loaded'
                memory = memory_status()
                assert memory['MemAvailable'] > (32 << 30), ('Flash load exhausted its RAM reserve', memory)
                node_memory = node_memory_status()
                assert all(v['estimated_available'] > (8 << 30) for v in node_memory.values()), ('Flash reached a per-node loading reserve', node_memory)
                with log_path.open('rb') as log:
                    log.seek(max(0, log_path.stat().st_size - 32768))
                    tail = log.read()
                assert not (b'GGML_ASSERT' in tail and b'failed' in tail), 'Flash asserted while loading'
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=2) as response:
                        if json.load(response).get('status') == 'ok':
                            break
                except OSError:
                    pass
                now = time.monotonic()
                assert now < deadline, 'Flash load exceeded one hour'
                if now - report >= 30:
                    self.record('waiting_for_flash', pid=proc.pid, memory=memory, node_memory=node_memory, log=str(log_path))
                    report = now
                time.sleep(1)
            info = process_info(proc.pid)
            assert info['command'] == configuration['command']
            assert info['affinity'] == configuration['affinity']
            assert runtime_environment(process_environment(proc.pid)) == configuration['runtime_env']
            mapped = Path(f'/proc/{proc.pid}/maps').read_text().splitlines()
            for stem, wanted in [('libggml-cpu.so.', Path(configuration['cpu_library'])),
                                 ('libllama.so.', Path(configuration['pinned_directory']) / 'libllama.so.0.3.0')]:
                actual = {row.split()[-1] for row in mapped if '/' + stem in row}
                assert actual == {str(wanted.resolve())}, actual
            assert sha256(configuration['cpu_library']) == configuration['cpu_sha256']
            current = dict(configuration, pid=proc.pid, info=info, log=str(log_path))
            self.state['current'] = current
            self.record('flash_ready', pid=proc.pid, port=PORT, memory=memory_status(), node_memory=node_memory_status(), service=read_service(PORT))
        except BaseException as error:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=30)
            self.state['current'] = None
            self.record('flash_load_failed', pid=proc.pid, exit_code=proc.returncode, error=repr(error), log=str(log_path))
            raise

    def restore_qwen(self):
        assert self.state.get('qwen_rollback'), 'No saved selected Qwen configuration'
        if self.qwen.state.get('current'):
            assert not self.state.get('current')
            self.qwen.ports()
            self.record('qwen_already_present', pid=self.qwen.state['current']['pid'])
            return
        self.quiet()
        self.stop_flash()
        configuration = copy.deepcopy(self.state['qwen_rollback'])
        context = json.loads(CONTEXT.read_text())
        env = {k: v for k, v in context['environment'].items() if k not in runtime_environment(context['environment'])}
        env.update(configuration['runtime_env'])
        self.qwen.recovering = True
        self.qwen.start_process(configuration, env, 'post-flash-q8')
        self.record('selected_qwen_restored', pid=self.qwen.state['current']['pid'])

    def launch(self, batching, drafts, workers, q8_experts=False, op_profile_count=0, q8_clamp=False, pooling=False, expert_tile_rows=0):
        configuration, environment = self.candidate(batching, drafts, workers, q8_experts, op_profile_count, q8_clamp, pooling, expert_tile_rows)
        self.quiet()
        if self.qwen.state.get('current'):
            previous = copy.deepcopy(self.qwen.state['current'])
            assert previous.get('no_cache_prompt') and previous.get('q8_wide_batch') and previous.get('drafts') == 4
            assert not previous.get('op_profile_count')
            if previous['runtime_env'].get('GGML_CPU_NUMA_THREADS_FILE'):
                previous['active_threads'] = int(Path(previous['runtime_env']['GGML_CPU_NUMA_THREADS_FILE']).read_text())
            self.state['qwen_rollback'] = previous
        assert self.state.get('qwen_rollback'), 'No Qwen rollback saved'
        self.record('transition_prepared', configuration=configuration)
        try:
            self.stop_flash()
            if self.qwen.state.get('current'):
                self.qwen.stop_current()
            self.check_cancel()
            self.start_flash(configuration, environment)
        except BaseException:
            self.recovering = True
            try:
                if not self.state.get('current') and not inference_snapshot():
                    self.restore_qwen()
            finally:
                self.recovering = False
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'launch', 'restore-qwen', 'status'))
    parser.add_argument('--batching', action='store_true')
    parser.add_argument('--q8-experts', action='store_true', help='Use the validated 288-expert Q8 layout; requires --batching')
    parser.add_argument('--q8-clamp', action='store_true', help='Use validated Q8 clamped expert fusion; requires --q8-experts')
    parser.add_argument('--pooling', action='store_true', help='Use the validated four-member pooling fusion; requires --q8-clamp')
    parser.add_argument('--expert-tile-rows', type=int, choices=(32, 48, 64), default=0,
                        help='Use the validated Q8 gate/up tile library; requires --pooling')
    parser.add_argument('--drafts', type=int, choices=range(0, 9), default=2)
    parser.add_argument('--workers', type=int, choices=range(1, 17), default=15)
    parser.add_argument('--op-profile-count', type=int, choices=range(257), default=0,
                        help='Enable bounded, file-armed operation capture; diagnostic runs only')
    args = parser.parse_args()
    os.umask(0o077)
    OUT.mkdir(exist_ok=True)
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manager = Manager()
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(signum, lambda *_: setattr(manager, 'cancelled', True))
        if args.action == 'launch':
            manager.launch(args.batching, args.drafts, args.workers, args.q8_experts, args.op_profile_count, args.q8_clamp, args.pooling, args.expert_tile_rows)
        elif args.action == 'restore-qwen':
            manager.restore_qwen()
        elif args.action == 'check':
            configuration, _ = manager.candidate(args.batching, args.drafts, args.workers, args.q8_experts, args.op_profile_count, args.q8_clamp, args.pooling, args.expert_tile_rows)
            print(json.dumps(dict(configuration=configuration, memory=memory_status()), indent=2))
        else:
            print(json.dumps(dict(current=manager.state.get('current'), inference=inference_snapshot(), memory=memory_status()), indent=2))


if __name__ == '__main__':
    main()
