#!/usr/bin/env python3
"""Manage the authorized Q6 trial, unloading Full and retaining a Qwen rollback."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.request

from model_measurement_guard import ModelMeasurementGuard, read_service
from qwen_split_trial import (ExactProcess, inference_snapshot,
                             process_environment, process_info, runtime_environment,
                             sha256)

BASE = Path(__file__).resolve().parent
OUT = BASE / 'results/qwen-q6-trial-0907'
STATE = OUT / 'state.json'
CONTEXT = OUT / 'original-context.private.json'
FULL_UNIT = 'glm53-sr950.service'
ORIGINAL_QWEN = 2308651
ORIGINAL_FULL = 4005448
PORT = 18095


def port_available(port):
    # Match the HTTP server's reuse behavior after a clean shutdown. An active
    # listener still prevents this bind; expired connections need not block it.
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(('127.0.0.1', port))
        except OSError:
            return False
    return True


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def set_option(command, flag, value):
    result = []
    index = 0
    while index < len(command):
        if command[index] == flag:
            index += 2
        else:
            result.append(command[index])
            index += 1
    if value is not None:
        result += [flag, str(value)]
    return result


def unit_state():
    output = subprocess.check_output(['systemctl', '--user', 'show', FULL_UNIT,
        '-p', 'MainPID', '-p', 'ActiveState', '-p', 'SubState', '-p', 'FragmentPath',
        '-p', 'DropInPaths'], text=True)
    return dict(line.split('=', 1) for line in output.splitlines() if '=' in line)


def expected(info):
    return {'start_ticks': info['start'], 'exe': info['exe']}


class Manager:
    def __init__(self):
        self.state = json.loads(STATE.read_text()) if STATE.exists() else {}
        self.cancelled = False
        self.recovering = False

    def check_cancel(self):
        if self.cancelled and not self.recovering:
            raise InterruptedError('Trial cancelled; restore Qwen after any completed shutdown')

    def record(self, event, **values):
        entry = dict(time=time.time(), event=event, **values)
        self.state.setdefault('events', []).append(entry)
        atomic_json(STATE, self.state)
        print(json.dumps(entry), flush=True)

    def prepare(self):
        assert not STATE.exists(), 'Preparation already exists; inspect status instead'
        assert set(inference_snapshot()) == {str(ORIGINAL_QWEN), str(ORIGINAL_FULL)}
        qwen = process_info(ORIGINAL_QWEN)
        full = process_info(ORIGINAL_FULL)
        assert qwen['command'][-4:] == ['--port', '18095', '--alias',
            'qwen-goal,qwen3.8-flash-next,flash-next']
        unit = unit_state()
        assert int(unit['MainPID']) == ORIGINAL_FULL and unit['ActiveState'] == 'active'
        unit_files = [Path(unit['FragmentPath']), *map(Path, unit['DropInPaths'].split())]
        environment = process_environment(ORIGINAL_QWEN)
        old_plan = json.loads((BASE / 'results/qwen-even-split-model-trial-0906-staging/plan.json').read_text())
        assert runtime_environment(environment) == old_plan['original_environment']
        binary_dir = Path(qwen['exe']).parent
        hashes = {str(binary_dir / name): digest for name, digest in
                  old_plan['baseline_binary_sha256'].items()}
        for path, digest in hashes.items():
            assert sha256(path) == digest, path
        assert b'GGML_CPU_NUMA_THREADS_FILE' in (binary_dir / 'libggml-cpu.so').read_bytes()
        atomic_json(CONTEXT, dict(qwen=qwen, environment=environment))
        self.state = dict(prepared=time.time(), original_qwen=qwen, original_full=full,
            full_unit=unit, full_unit_sha256={str(p): sha256(p) for p in unit_files},
            full_stopped=False, binary_sha256=hashes,
            current=dict(pid=ORIGINAL_QWEN, info=qwen, original=True,
                         command=qwen['command'], runtime_env=runtime_environment(environment)),
            events=[])
        self.record('prepared', qwen_pid=ORIGINAL_QWEN, full_pid=ORIGINAL_FULL,
                    full_policy='Unload the idle service for the whole-server trial')

    def validate_inputs(self):
        for path, digest in self.state['binary_sha256'].items():
            assert sha256(path) == digest, path

    def ports(self):
        ports = {}
        current = self.state.get('current')
        if current:
            info = process_info(current['pid'])
            assert info['start'] == current['info']['start'] and info['exe'] == current['info']['exe']
            assert info['command'] == current['command']
            ports[current['pid']] = PORT
        if not self.state['full_stopped']:
            full = process_info(ORIGINAL_FULL)
            assert expected(full) == expected(self.state['original_full'])
            ports[ORIGINAL_FULL] = 18091
        return ports

    def quiet(self):
        ports = self.ports()
        if not ports:
            assert not inference_snapshot(), 'Another model has been loaded'
            return
        guard = ModelMeasurementGuard(next(iter(ports)), ports, inference_snapshot)
        guard.wait_idle(OUT / 'waiting-for-idle.json')
        guard.assert_idle()
        self.check_cancel()

    def unload_full(self):
        if self.state['full_stopped']:
            return
        unit = unit_state()
        assert int(unit['MainPID']) == ORIGINAL_FULL and unit['ActiveState'] == 'active'
        for path, digest in self.state['full_unit_sha256'].items():
            assert sha256(path) == digest, path
        ports = self.ports()
        ModelMeasurementGuard(next(iter(ports)), ports, inference_snapshot).assert_idle()
        self.record('stopping_idle_full', unit=FULL_UNIT, pid=ORIGINAL_FULL)
        subprocess.run(['systemctl', '--user', 'stop', FULL_UNIT], check=True, timeout=180)
        unit = unit_state()
        assert unit['ActiveState'] == 'inactive' and int(unit['MainPID']) == 0
        assert str(ORIGINAL_FULL) not in inference_snapshot()
        self.state['full_stopped'] = True
        self.record('full_stopped', unit=FULL_UNIT)

    def stop_current(self):
        current = self.state.get('current')
        if not current:
            return
        ports = self.ports()
        ModelMeasurementGuard(current['pid'], ports, inference_snapshot).assert_idle()
        proc = ExactProcess(current['pid'], expected(current['info']), set())
        try:
            proc.terminate()
            self.record('stopping_qwen', pid=current['pid'], original=current['original'])
            deadline = time.monotonic() + 120
            while not proc.exited():
                if time.monotonic() > deadline:
                    raise TimeoutError('Qwen did not finish its idle shutdown')
                time.sleep(0.5)
        finally:
            proc.close()
        self.state['current'] = None
        self.record('qwen_stopped', pid=current['pid'])

    def candidate(self, drafts, even_split, q8_batch=False, q8_fused_batch=False, draft_p_min=0.0, workers=15, q8_experts=False, q8_dense_extra=False, simple_barrier=False, gomp_spin_count=1000, q8_wide_batch=False, no_cache_prompt=False, op_profile_count=0):
        assert 0.0 <= draft_p_min <= 1.0
        assert 1 <= workers <= 16
        assert sum((q8_experts, q8_fused_batch, q8_dense_extra, q8_wide_batch)) <= 1
        assert not simple_barrier or not any((q8_experts, q8_fused_batch, q8_dense_extra, q8_wide_batch))
        assert 0 <= gomp_spin_count <= 1000000000
        assert 0 <= op_profile_count <= 256
        q8_batch = q8_batch or q8_fused_batch or q8_experts or q8_dense_extra or simple_barrier or q8_wide_batch
        download = json.loads((BASE / 'results/qwen-q6-download-0907/status.json').read_text())
        manifest = json.loads((BASE / 'results/qwen-higher-quant-selection-0907.json').read_text())
        assert download['complete'] and download['revision'] == manifest['revision']
        records = manifest['candidates']['UD-Q6_K_XL']['files']
        for record in records:
            path = Path(download['destination']) / record['name']
            verified = download['verified'][record['name']]
            stat = path.stat()
            assert verified['sha256'] == record['sha256']
            assert stat.st_size == record['bytes'] and stat.st_ino == verified['inode']
            assert stat.st_mtime_ns == verified['mtime_ns'], path
        context = json.loads(CONTEXT.read_text())
        command = context['qwen']['command']
        command = set_option(command, '--model', Path(download['destination']) / records[0]['name'])
        command = set_option(command, '--port', PORT)
        command = set_option(command, '--alias', 'qwen-goal,qwen3.8-flash-next,flash-next,qwen-q6-trial')
        command = set_option(command, '--spec-draft-n-max', drafts)
        command = set_option(command, '--spec-draft-p-min', draft_p_min)
        if no_cache_prompt:
            assert not any(x in command for x in ('--cache-prompt', '--no-cache-prompt'))
            command.append('--no-cache-prompt')
        if drafts == 0:
            for flag in [arg for arg in command if arg.startswith('--spec-')]:
                command = set_option(command, flag, None)
        environment = dict(context['environment'])
        if op_profile_count:
            arm_file = OUT / 'op-profile.arm'
            assert not arm_file.exists(), 'Remove the owned profile arm file after the previous capture'
            environment.update(GGML_CPU_OP_PROFILE='*', GGML_CPU_OP_PROFILE_ARM_FILE=str(arm_file),
                               GGML_CPU_OP_PROFILE_SKIP='0', GGML_CPU_OP_PROFILE_COUNT=str(op_profile_count))
        environment['GGML_CPU_NUMA_THREADS'] = str(workers)
        threads_file = OUT / 'numa-threads'
        environment['GGML_CPU_NUMA_THREADS_FILE'] = str(threads_file)
        cpu_manifest = BASE / 'results/qwen-q6-expert-capacity-0907/private-cpu/manifest.json'
        cpu = json.loads(cpu_manifest.read_text())
        cpu_validation = json.loads((BASE / 'results/qwen-q6-512-expert-validation-0907/result.json').read_text())
        assert cpu_validation['passed'] and cpu_validation['experts'] == 512
        assert cpu_validation['cpu_sha256'] == cpu['library_sha256'] == sha256(cpu['library'])
        assert cpu_validation['unfused_prefill']['experts'] == 512
        if q8_batch:
            batch_root = BASE / ('results/qwen-q6-q8-wide-batch-0907' if q8_wide_batch else
                                 'results/qwen-q6-q8-dense-extra-0907' if q8_dense_extra else
                                 'results/qwen-q6-q8-experts-0907' if q8_experts else
                                 'results/qwen-q6-q8-fused-batch-0907' if q8_fused_batch
                                 else 'results/qwen-q6-q8-batch-0907')
            validation = json.loads((batch_root / 'result.json').read_text())
            batch_cpu = json.loads((batch_root / 'private-cpu/manifest.json').read_text())
            assert validation['passed'] and batch_cpu['parent_cpu_sha256'] == cpu['library_sha256']
            assert validation['library_sha256'] == batch_cpu['library_sha256'] == sha256(batch_cpu['library'])
            assert validation['kernel_check']['bit_exact_values'] == (842400 if q8_wide_batch else 168480)
            assert sum(c['bit_exact_cases'] for c in validation['graph_checks']) == (648 if q8_wide_batch else 432)
            assert bool(validation.get('fused')) == q8_fused_batch
            assert bool(validation.get('experts')) == q8_experts
            assert bool(validation.get('dense_extra')) == q8_dense_extra
            assert bool(validation.get('wide')) == q8_wide_batch
            if q8_dense_extra:
                check = json.loads((batch_root / 'selector-check/result.json').read_text())
                assert check['passed'] and check['library_sha256'] == batch_cpu['library_sha256']
                assert sum(x['bit_exact_cases'] for x in check['checks']) == 432
                environment['GGML_CPU_X16_Q8_DENSE_EXTRA'] = '1'
            if q8_experts:
                for kind in ('q6', 'q8'):
                    check = json.loads((BASE / f'results/qwen-x16-q8-experts-{kind}-512-0907/result.json').read_text())
                    assert check['passed'] and check['experts'] == 512 and check['x16_q8_experts']
                    assert check['cpu_sha256'] == batch_cpu['library_sha256']
                    assert check['weight_type'] == ('q6_K' if kind == 'q6' else 'q8_0')
                    assert check['unfused_prefill']['experts'] == 512
                environment['GGML_CPU_X16_Q8_EXPERTS'] = '1'
            cpu = batch_cpu
            environment['GGML_CPU_X16_Q8_BATCH'] = '1'
        if simple_barrier:
            barrier_root = BASE / 'results/qwen-q6-simple-barrier-0907'
            validation = json.loads((barrier_root / 'result.json').read_text())
            barrier_cpu = json.loads((barrier_root / 'private-cpu/manifest.json').read_text())
            assert validation['passed'] and validation['unpatched_text_identical']
            assert barrier_cpu['parent_cpu_sha256'] == cpu['library_sha256']
            assert validation['library_sha256'] == barrier_cpu['library_sha256'] == sha256(barrier_cpu['library'])
            assert sum(x['bit_exact_cases'] for x in validation['graph_checks']) == 864
            assert sum(x['cases'] for x in validation['numa_checks']) == 60
            cpu = barrier_cpu
            environment['GGML_CPU_OMP_SIMPLE_BARRIER'] = '1'
            environment['GOMP_SPINCOUNT'] = str(gomp_spin_count)
        environment['LD_LIBRARY_PATH'] = str(Path(cpu['library']).parent) + ':' + environment['LD_LIBRARY_PATH']
        if even_split:
            plan = json.loads((BASE / 'results/qwen-even-split-model-trial-0906-staging/plan.json').read_text())
            assert sha256(plan['candidate_library']) == plan['candidate_library_sha256']
            validation = json.loads((BASE / 'results/qwen-q6-expert-validation-0907/result.json').read_text())
            assert validation['passed'] and validation['candidate_sha256'] == plan['candidate_library_sha256']
            environment['LD_LIBRARY_PATH'] = str(Path(plan['candidate_library']).parent) + ':' + environment['LD_LIBRARY_PATH']
            environment['GGML_Q4E_EXPERT_EVEN_SPLIT'] = '1'
        return dict(command=command, runtime_env=runtime_environment(environment), original=False,
                    drafts=drafts, draft_p_min=draft_p_min, even_split=even_split, q8_batch=q8_batch,
                    q8_fused_batch=q8_fused_batch, q8_experts=q8_experts,
                    q8_dense_extra=q8_dense_extra, active_threads=workers,
                    simple_barrier=simple_barrier, gomp_spin_count=gomp_spin_count if simple_barrier else None,
                    q8_wide_batch=q8_wide_batch,
                    no_cache_prompt=no_cache_prompt,
                    op_profile_count=op_profile_count,
                    cpu_library=cpu['library'], cpu_sha256=cpu['library_sha256']), environment

    def start_process(self, configuration, environment, role):
        assert not inference_snapshot(), 'Another model has been loaded'
        assert port_available(PORT)
        self.validate_inputs()
        if configuration.get('cpu_library'):
            assert sha256(configuration['cpu_library']) == configuration['cpu_sha256']
        context = json.loads(CONTEXT.read_text())
        self.check_cancel()
        if environment.get('GGML_CPU_NUMA_THREADS_FILE'):
            Path(environment['GGML_CPU_NUMA_THREADS_FILE']).write_text(
                str(configuration.get('active_threads', 15)) + '\n')
        affinity = ','.join(map(str, context['qwen']['affinity']))
        log_path = OUT / f'{role}-{time.time_ns()}.log'
        with log_path.open('w') as log:
            proc = subprocess.Popen(['taskset', '-c', affinity, *configuration['command']],
                env=environment, cwd=context['qwen']['cwd'], stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        self.record('loading_qwen', pid=proc.pid, log=str(log_path),
                    drafts=configuration.get('drafts'), even_split=configuration.get('even_split'))
        try:
            deadline, report = time.monotonic() + 3600, 0
            while True:
                self.check_cancel()
                assert proc.poll() is None, ('Server exited while loading', proc.returncode)
                with log_path.open('rb') as log:
                    log.seek(max(0, log_path.stat().st_size - 16384))
                    tail = log.read()
                assert not (b'GGML_ASSERT' in tail and b'failed' in tail), 'Model asserted during startup; see its log'
                foreign = set(inference_snapshot()) - {str(proc.pid)}
                assert not foreign, ('Another model was loaded during this trial', foreign)
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=2) as response:
                        if json.load(response).get('status') == 'ok':
                            break
                except OSError:
                    pass
                now = time.monotonic()
                assert now < deadline, 'Model load exceeded one hour'
                if now - report >= 30:
                    self.record('waiting_for_load', pid=proc.pid, log=str(log_path))
                    report = now
                time.sleep(1)
            info = process_info(proc.pid)
            assert info['command'] == configuration['command']
            assert info['affinity'] == context['qwen']['affinity']
            assert runtime_environment(process_environment(proc.pid)) == configuration['runtime_env']
            mapped = Path(f'/proc/{proc.pid}/maps').read_text().splitlines()
            expected_library = Path(context['qwen']['exe']).parent / 'libllama.so.0.3.0'
            if configuration.get('even_split'):
                plan = json.loads((BASE / 'results/qwen-even-split-model-trial-0906-staging/plan.json').read_text())
                expected_library = Path(plan['candidate_library'])
            libraries = {line.split()[-1] for line in mapped if '/libllama.so.' in line}
            assert libraries == {str(expected_library.resolve())}, libraries
            expected_cpu = Path(configuration.get('cpu_library',
                str(Path(context['qwen']['exe']).parent / 'libggml-cpu.so.0.22.0')))
            cpu_libraries = {line.split()[-1] for line in mapped if '/libggml-cpu.so.' in line}
            assert cpu_libraries == {str(expected_cpu.resolve())}, cpu_libraries
            current = dict(configuration, pid=proc.pid, info=info, log=str(log_path))
            self.state['current'] = current
            self.record('qwen_ready', pid=proc.pid, port=PORT, drafts=current.get('drafts'),
                        even_split=current.get('even_split'), service=read_service(PORT))
        except BaseException:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=30)
            self.record('failed_load_terminal', pid=proc.pid, exit_code=proc.returncode,
                        log=str(log_path))
            raise

    def launch(self, drafts, even_split, q8_batch=False, q8_fused_batch=False, draft_p_min=0.0, workers=15, q8_experts=False, q8_dense_extra=False, simple_barrier=False, gomp_spin_count=1000, q8_wide_batch=False, no_cache_prompt=False, op_profile_count=0):
        configuration, environment = self.candidate(drafts, even_split, q8_batch, q8_fused_batch, draft_p_min, workers, q8_experts, q8_dense_extra, simple_barrier, gomp_spin_count, q8_wide_batch, no_cache_prompt, op_profile_count)
        self.validate_inputs()
        self.quiet()
        previous = self.state.get('current') or self.state.get('rollback')
        assert previous, 'No current or saved Qwen context'
        if previous['runtime_env'].get('GGML_CPU_NUMA_THREADS_FILE'):
            previous['active_threads'] = int(Path(previous['runtime_env']['GGML_CPU_NUMA_THREADS_FILE']).read_text())
        self.state['rollback'] = previous
        self.record('transition_prepared', drafts=drafts, draft_p_min=draft_p_min, even_split=even_split,
                    q8_batch=configuration['q8_batch'], q8_fused_batch=q8_fused_batch,
                    workers=workers, q8_experts=q8_experts, q8_dense_extra=q8_dense_extra,
                    simple_barrier=simple_barrier, gomp_spin_count=gomp_spin_count if simple_barrier else None,
                    q8_wide_batch=q8_wide_batch, no_cache_prompt=no_cache_prompt,
                    op_profile_count=op_profile_count)
        self.unload_full()
        self.check_cancel()
        try:
            self.stop_current()
            self.check_cancel()
            self.start_process(configuration, environment, 'q6')
        except BaseException:
            if self.state.get('current') is not None:
                self.record('transition_stopped_before_qwen_exit')
                raise
            context = json.loads(CONTEXT.read_text())
            rollback_env = dict(context['environment'])
            rollback_env.update(previous['runtime_env'])
            self.record('restoring_previous_qwen')
            self.recovering = True
            try:
                self.start_process(previous, rollback_env, 'rollback')
            finally:
                self.recovering = False
            raise

    def restore(self):
        assert self.state['full_stopped'], 'Restore is for an already started whole-server trial'
        self.quiet()
        self.stop_current()
        context = json.loads(CONTEXT.read_text())
        environment = context['environment']
        configuration = dict(command=context['qwen']['command'], original=True,
                             runtime_env=runtime_environment(environment))
        self.start_process(configuration, environment, 'original-q2')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'launch', 'restore', 'status'])
    parser.add_argument('--drafts', type=int, default=2, choices=range(0, 9))
    parser.add_argument('--even-split', action='store_true')
    parser.add_argument('--q8-batch', action='store_true')
    parser.add_argument('--q8-fused-batch', action='store_true')
    parser.add_argument('--draft-p-min', type=float, default=0.0)
    parser.add_argument('--workers', type=int, choices=range(1, 17), default=15)
    parser.add_argument('--q8-experts', action='store_true')
    parser.add_argument('--q8-dense-extra', action='store_true')
    parser.add_argument('--simple-barrier', action='store_true')
    parser.add_argument('--gomp-spin-count', type=int, default=1000)
    parser.add_argument('--q8-wide-batch', action='store_true')
    parser.add_argument('--no-cache-prompt', action='store_true')
    parser.add_argument('--op-profile-count', type=int, default=0, choices=range(0, 257),
                        help='Bounded operation trace, armed only by the trial profile helper')
    args = parser.parse_args()
    os.umask(0o077)
    OUT.mkdir(exist_ok=True, mode=0o700)
    with (OUT / 'lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manager = Manager()
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(signum, lambda *_: setattr(manager, 'cancelled', True))
        if args.action == 'prepare':
            manager.prepare()
        elif args.action == 'launch':
            manager.launch(args.drafts, args.even_split, args.q8_batch, args.q8_fused_batch, args.draft_p_min, args.workers, args.q8_experts, args.q8_dense_extra, args.simple_barrier, args.gomp_spin_count, args.q8_wide_batch, args.no_cache_prompt, args.op_profile_count)
        elif args.action == 'restore':
            manager.restore()
        else:
            print(json.dumps(manager.state, indent=2))


if __name__ == '__main__':
    main()
