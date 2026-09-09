#!/usr/bin/env python3
"""Run a bounded private Q6 trial while preserving one identified Qwen service."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import urllib.request

from benchmark_qwen_q6 import wait_background
from benchmark_qwen_resident_0908 import normalized_command
from glm_flash_q8_trial import memory_status, node_memory_status
from model_measurement_guard import ModelMeasurementGuard, read_service
from qwen_high_quant_trial import port_available, set_option, unit_state
from qwen_split_trial import inference_snapshot, process_environment, process_info, runtime_environment, sha256

BASE = Path(__file__).resolve().parent
PORT = 18155


def memory_gate(loading=False):
    memory, nodes = memory_status(), node_memory_status()
    assert len(nodes) == 4
    assert memory['MemAvailable'] > ((64 << 30) if loading else 300_000_000_000), memory
    assert all(row['estimated_available'] > ((16 << 30) if loading else 70_000_000_000)
               for row in nodes.values()), nodes
    return dict(memory=memory, nodes=nodes)


def model_records(preset):
    download_path = BASE / 'results/qwen-q6-download-0907/status.json'
    manifest_path = BASE / 'results/qwen-higher-quant-selection-0907.json'
    download, manifest = [json.loads(path.read_text()) for path in (download_path, manifest_path)]
    assert download['complete'] and download['revision'] == manifest['revision']
    records = []
    for row in manifest['candidates']['UD-Q6_K_XL']['files']:
        path = Path(download['destination']) / row['name']
        verified = download['verified'][row['name']]
        stat = path.stat()
        assert verified['sha256'] == row['sha256'] and stat.st_size == row['bytes']
        assert stat.st_ino == verified['inode'] and stat.st_mtime_ns == verified['mtime_ns']
        records.append(dict(path=str(path), size=stat.st_size, inode=stat.st_ino, mtime_ns=stat.st_mtime_ns, sha256=row['sha256']))
    assert len(records) == 6 and sum(row['size'] for row in records) == 169165382688
    command = preset['command']
    assert command[command.index('--model') + 1] == records[0]['path']
    draft = Path(command[command.index('--spec-draft-model') + 1])
    stat = draft.stat()
    records.append(dict(path=str(draft), size=stat.st_size, inode=stat.st_ino, mtime_ns=stat.st_mtime_ns,
                        provenance='Unchanged selected Q8 draft path; metadata snapshot'))
    return records, [download_path, manifest_path]


def verify_plan(plan):
    assert all(sha256(path) == digest for path, digest in plan['source_sha256'].items())
    for row in plan['model_records']:
        stat = Path(row['path']).stat()
        assert (stat.st_size, stat.st_ino, stat.st_mtime_ns) == (row['size'], row['inode'], row['mtime_ns'])
    peer = process_info(plan['peer_pid'])
    assert all(peer[key] == plan['peer'][key] for key in ('start', 'exe', 'command', 'affinity'))
    assert runtime_environment(process_environment(plan['peer_pid'])) == plan['peer_runtime_env']
    assert unit_state()['ActiveState'] == 'inactive'


def prepare(args, out):
    assert set(inference_snapshot()) == {str(args.peer_pid)} and port_available(PORT)
    peer = process_info(args.peer_pid)
    assert peer['start'] == args.start_ticks
    preset_path = BASE / 'qwen-flash-20tps.json'
    bundle_path = BASE / 'results/qwen-hc-combine-runtime-0908/result.json'
    check_path = BASE / 'results/qwen-hc-combine-proof-0908/result.json'
    build_path = BASE / 'results/qwen-hc-combine-build-0908b/result.json'
    preset, bundle, checks, build = [json.loads(path.read_text()) for path in (preset_path, bundle_path, check_path, build_path)]
    assert bundle['passed'] and checks['passed'] and build['build_completed']
    assert len(checks['checks']) == 2 and sum(row['cases'] for row in checks['checks']) == 384
    assert all(row['bit_exact'] and row['scalar_exact'] for row in checks['checks'])
    assert bundle['cpu_sha256'] == sha256(bundle['cpu']) == 'c79e19e38acfd6ada3f0637c136cf9cb10253127778cb214528b2f966fb6dfcf'
    assert bundle['llama_sha256'] == build['library_sha256'] == sha256(bundle['llama'])
    assert build['baseline_link_identical']
    assert all(sha256(path) == digest for path, digest in build['input_sha256'].items())
    assert all(sha256(path) == digest for path, digest in checks['input_sha256'].items())
    assert all(sha256(path) == digest for path, digest in bundle['sources'].items())
    peer_env = runtime_environment(process_environment(args.peer_pid))
    assert peer_env == preset['runtime_env']
    assert normalized_command(peer['command']) == normalized_command(preset['command'])
    command = list(preset['command'])
    command[0] = bundle['server']
    command = set_option(command, '--port', PORT)
    command = set_option(command, '--alias', 'qwen-q6-private')
    command = set_option(command, '--spec-draft-n-max', args.drafts)
    if args.drafts == 0:
        for flag in [value for value in command if value.startswith('--spec-')]:
            command = set_option(command, flag, None)
    env = dict(bundle['runtime_env'], GGML_QWEN_HC_COMBINE_FUSED=str(int(args.combine == 'on')))
    assert not any('AUDIT' in key or 'PROFILE' in key or 'REQUANT' in key for key in env)
    records, source_paths = model_records(preset)
    source_paths += [Path(__file__).resolve(), preset_path, bundle_path, check_path, build_path,
                     BASE / 'measure-model-bandwidth.py', BASE / 'model_measurement_guard.py',
                     BASE / 'guarded_inference_request.py', BASE / 'dram_bandwidth.py',
                     BASE / 'benchmark_qwen_q6.py', BASE / 'glm_flash_q8_trial.py',
                     BASE / 'benchmark_qwen_resident_0908.py',
                     Path(bundle['cpu']), Path(bundle['llama']), Path(bundle['server'])]
    plan = dict(prepared=time.time(), label=args.label, peer_pid=args.peer_pid, peer=peer, peer_runtime_env=peer_env,
                port=PORT, command=command, runtime_env=env, drafts=args.drafts, combine=args.combine,
                cpu=bundle['cpu'], cpu_sha256=bundle['cpu_sha256'], llama=bundle['llama'],
                server_sha256=bundle['server_sha256'], model_records=records,
                source_sha256={str(path):sha256(path) for path in source_paths},
                memory_preflight=memory_gate(), target_tok_s=30, target_gb_s=190, capacity_gb_s=380,
                no_model_loaded=True,
                scope='Temporary second Q6 instance with explicit peer identity and memory reserves; the public exclusion guard remains unchanged')
    verify_plan(plan)
    out.mkdir(exist_ok=False)
    (out / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps(dict(prepared=True, path=str(out / 'plan.json'), combine=args.combine,
                          drafts=args.drafts, private_port=PORT, peer_preserved=args.peer_pid)), flush=True)


def execute(out):
    plan = json.loads((out / 'plan.json').read_text())
    assert not (out / 'result.json').exists(), 'Preserve the previous trial result'
    verify_plan(plan)
    assert set(inference_snapshot()) == {str(plan['peer_pid'])} and port_available(PORT)
    owned_pid = None
    peer_guard = ModelMeasurementGuard(plan['peer_pid'], {plan['peer_pid']:18095},
                                      lambda: inference_snapshot(exclude=owned_pid))
    model, measurement = None, None
    result = dict(started=time.time(), passed=False, controller_pid=os.getpid(), plan_sha256=sha256(out / 'plan.json'),
                  peer_pid=plan['peer_pid'], model_started=False, target_reached=False)
    def save(stage=None):
        if stage:
            result['stage'] = stage
        (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    def interrupted(signum, frame):
        raise InterruptedError('Stop owned trial processes and preserve external Qwen')
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupted)
    save('waiting for exclusive CPU use')
    try:
        result['idle_gate'] = peer_guard.wait_idle(out / 'waiting-for-idle.json')
        result['background_gate'] = wait_background(peer_guard, plan['peer_pid'], 4, out / 'background-wait.json')
        verify_plan(plan)
        result['memory_before_load'] = memory_gate()
        assert port_available(PORT)
        environment = {key:value for key,value in os.environ.items() if key not in runtime_environment(os.environ)}
        environment.update(plan['runtime_env'])
        with (out / 'model.log').open('w') as log:
            model = subprocess.Popen(['taskset', '-c', '0-127', *plan['command']], cwd=BASE, env=environment,
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        owned_pid = model.pid
        result.update(model_pid=model.pid, model_started=True)
        save('loading private Qwen')
        deadline, report = time.monotonic() + 900, 0
        while True:
            assert model.poll() is None, 'Private model exited while loading'
            peer_guard.assert_idle()
            memory_gate(loading=True)
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=1) as response:
                    if json.load(response).get('status') == 'ok':
                        break
            except OSError:
                pass
            now = time.monotonic()
            assert now < deadline, 'Private load timeout'
            if now - report >= 30:
                print(json.dumps(dict(loading_private_pid=model.pid, peer_preserved=plan['peer_pid'])), flush=True)
                report = now
            time.sleep(.5)
        current = process_info(model.pid)
        assert current['command'] == plan['command'] and current['affinity'] == list(range(128))
        assert runtime_environment(process_environment(model.pid)) == plan['runtime_env']
        mapped = Path(f'/proc/{model.pid}/maps').read_text().splitlines()
        for stem, path in [('libggml-cpu.so.', plan['cpu']), ('libllama.so.', plan['llama'])]:
            assert {row.split()[-1] for row in mapped if '/' + stem in row} == {str(Path(path).resolve())}
        result['current'] = current
        class ReservedMemoryGuard(ModelMeasurementGuard):
            def inspect(self, *args, **kwargs):
                memory_gate(loading=True)
                return super().inspect(*args, **kwargs)
        guard = ReservedMemoryGuard(model.pid, {model.pid:PORT, plan['peer_pid']:18095}, inference_snapshot)
        result['decode_idle_gate'] = guard.wait_idle(out / 'decode-idle.json', quiet_seconds=15)
        result['decode_background_gate'] = wait_background(guard, model.pid, 4, out / 'decode-background.json')
        label = plan['label'] + '-decode'
        command = [sys.executable, '-u', str(BASE / 'measure-model-bandwidth.py'), label,
                   '--port', str(PORT), '--pid', str(model.pid), '--alias', 'qwen-q6-private',
                   '--drafts', str(plan['drafts']), '--tokens', '512', '--request-timeout-seconds', '180',
                   '--allowed-idle-pids', str(plan['peer_pid']), '--skip-idle-gate',
                   '--bandwidth-capacity-gb-s', '380', '--bandwidth-target-gb-s', '190']
        save('measuring private Qwen')
        measurement = subprocess.Popen(command, cwd=BASE, start_new_session=True)
        while measurement.poll() is None:
            peer_guard.assert_idle()
            memory_gate(loading=True)
            assert model.poll() is None, 'Private model exited during measurement'
            time.sleep(.5)
        assert measurement.returncode == 0, 'Measurement did not complete'
        path = BASE / 'results' / label / 'result.json'
        measured = json.loads(path.read_text())
        assert measured['input_integrity_verified'] and not measured.get('error')
        assert all(row['pass_check'] for row in measured['checks']) and len(measured['measurements']) == 2
        assert measured['server_command'] == plan['command'] and measured['runtime_env'] == plan['runtime_env']
        rows = []
        for row in measured['measurements']:
            assert not row['abort'] and row['counter_metadata']['valid'] and row['timings']['cache_n'] == 0
            speed, traffic = row['timings']['predicted_per_second'], row['background_subtracted_gb_s']
            before = row['baseline_before']['total_gb_s']
            after = row['baseline_after']['total_gb_s']
            attribution_valid = max(before, after) <= .05 * plan['capacity_gb_s'] and abs(before - after) <= .025 * plan['capacity_gb_s']
            rows.append(dict(workload=row['kind'], tok_s=speed, adjusted_gb_s=traffic,
                             idle_before_gb_s=before, idle_after_gb_s=after, attribution_valid=attribution_valid,
                             both_targets_met=attribution_valid and speed > 30 and traffic > 190))
        verify_plan(plan)
        result.update(passed=True, rows=rows, measurement=str(path), measurement_sha256=sha256(path),
                      target_reached=all(row['both_targets_met'] for row in rows))
        print(json.dumps(dict(passed=True, rows=rows, target_reached=result['target_reached'])), flush=True)
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        if measurement is not None and measurement.poll() is None:
            measurement.send_signal(signal.SIGINT)
            try:
                measurement.wait(timeout=20)
            except subprocess.TimeoutExpired:
                measurement.terminate()
                measurement.wait(timeout=10)
        if model is not None and model.poll() is None:
            model.terminate()
            try:
                model.wait(timeout=60)
            except subprocess.TimeoutExpired:
                model.kill()
                model.wait(timeout=20)
        result['owned_model_exit'] = model.returncode if model is not None else None
        try:
            peer_after = process_info(plan['peer_pid'])
            result['peer_preserved'] = all(peer_after[key] == plan['peer'][key] for key in ('start', 'exe', 'command', 'affinity'))
            result['peer_service_after'] = read_service(18095)
        except Exception as error:
            result['peer_preserved'] = False
            result['peer_observation_error'] = repr(error)
        result['finished'] = time.time()
        save('finished; private model released')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run'))
    parser.add_argument('label')
    parser.add_argument('--peer-pid', type=int, default=1219506)
    parser.add_argument('--start-ticks', default='103969952')
    parser.add_argument('--combine', choices=('off', 'on'), default='on')
    parser.add_argument('--drafts', type=int, choices=range(5), default=4)
    args = parser.parse_args()
    assert re.fullmatch(r'qwen-private-[A-Za-z0-9_-]+', args.label)
    os.umask(0o077)
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        out = BASE / 'results' / args.label
        prepare(args, out) if args.action == 'prepare' else execute(out)
