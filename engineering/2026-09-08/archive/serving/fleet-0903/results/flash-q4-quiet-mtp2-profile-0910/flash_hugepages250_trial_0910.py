#!/usr/bin/env python3
"""Compare Flash Q8 page backing with a reversible whole-server model switch."""
import argparse
import fcntl
import hashlib
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
from glm_flash_q8_trial import Manager as FlashManager, PORT, memory_status
from qwen_numa_page_audit_0909 import audit_pages
from model_measurement_guard import ModelMeasurementGuard, read_service
from qwen_high_quant_trial import expected, port_available, unit_state
from qwen_private_shared_dispatch_trial_0909c import model_records
from qwen_split_trial import ExactProcess, inference_snapshot, process_environment, process_info, runtime_environment, sha256

BASE = Path(__file__).resolve().parent
PREVIOUS = BASE / 'results/flash-bandwidth250-assessment-0910.json'
OPTIONS = dict(batching=True, workers=15, q8_experts=True, q8_clamp=True,
               pooling=True, sum16=True, rms_guard=True, ordered_k=True)


class Manager(FlashManager):
    def candidate(self, *args, huge_pages=False, **kwargs):
        config, env = super().candidate(*args, **kwargs)
        env['GGML_CPU_NUMA_HUGEPAGES'] = str(int(huge_pages))
        config.update(runtime_env=runtime_environment(env), huge_pages=huge_pages)
        return config, env

    def start_flash(self, configuration, environment):
        super().start_flash(configuration, environment)
        current = self.validate_current()
        pages = audit_pages(current['pid'])
        current['page_backing'] = pages
        self.record('flash_page_backing', pid=current['pid'], huge_pages=configuration['huge_pages'], pages=pages)
        fraction = pages['anonymous_hugepage_fraction']
        assert fraction >= .5 if configuration['huge_pages'] else fraction <= .05, 'Page backing does not qualify the intended comparison'


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def assert_sources(plan):
    assert all(sha256(path) == digest for path, digest in plan['source_sha256'].items())
    for record in plan['qwen_model_records']:
        stat = Path(record['path']).stat()
        assert (stat.st_size, stat.st_ino, stat.st_mtime_ns) == (record['size'], record['inode'], record['mtime_ns'])
    assert unit_state()['ActiveState'] == 'inactive'


def assert_peer(plan):
    actual = process_info(plan['qwen_pid'])
    assert all(actual[key] == plan['qwen'][key] for key in ('start', 'exe', 'command', 'cwd', 'affinity'))
    assert runtime_environment(process_environment(plan['qwen_pid'])) == plan['qwen_runtime_env']
    assert set(inference_snapshot()) == {str(plan['qwen_pid'])}
    return actual


def prepare(out):
    assert not out.exists()
    previous = json.loads(PREVIOUS.read_text())
    assert previous['passed'] and previous['only_restored_qwen_loaded']
    peer_pid, peer_start = previous['restored_qwen_pid'], previous['restored_qwen_start']
    controls = previous['configurations']['raw_control']
    candidates = previous['configurations']['raw_candidate']
    barrier = all(candidates[k]['tok_s_range'][0] > controls[k]['tok_s_range'][1]
                  and candidates[k]['mean_tok_s'] / controls[k]['mean_tok_s'] > 1.01
                  for k in ('prose', 'code'))
    manager = Manager()
    assert manager.state['current'] is None and manager.qwen.state['current'] is None
    assert port_available(PORT) and unit_state()['ActiveState'] == 'inactive'
    peer = process_info(peer_pid)
    assert peer['start'] == peer_start and peer['affinity'] == list(range(128))
    preset_path = BASE / 'qwen-flash-20tps.json'
    preset = json.loads(preset_path.read_text())
    assert normalized_command(peer['command']) == normalized_command(preset['command'])
    peer_env = runtime_environment(process_environment(peer_pid))
    assert peer_env == preset['runtime_env']
    records, record_sources = model_records(preset)
    configs = {}
    for name, huge_pages in [('raw_control', False), ('raw_candidate', True)]:
        configs[name], _ = manager.candidate(**OPTIONS, drafts=0, dissemination=barrier, huge_pages=huge_pages)
    control, candidate = configs['raw_control'], configs['raw_candidate']
    assert {k for k in control if control[k] != candidate[k]} == {'runtime_env', 'huge_pages'}
    assert {k for k in control['runtime_env'] if control['runtime_env'][k] != candidate['runtime_env'][k]} == {'GGML_CPU_NUMA_HUGEPAGES'}
    for name, config in configs.items():
        assert config['model_revision'] == '621d456e93e926e4b52f85cff5f634358c1828f9'
        assert config['workers'] == 15 and not config['op_profile_count']
        assert not any('PROFILE' in key or 'AUDIT' in key or 'REQUANT' in key for key in config['runtime_env'])
    libraries = {line.split()[-1] for line in Path(f'/proc/{peer_pid}/maps').read_text().splitlines()
                 if any('/' + stem in line for stem in ['libggml-', 'libggml.so.', 'libllama.so.'])}
    assert any('libggml-cpu.so.' in path for path in libraries) and any('libllama.so.' in path for path in libraries)
    paths = [Path(__file__).resolve(), preset_path, PREVIOUS, BASE / 'qwen_numa_page_audit_0909.py',
             BASE / 'glm_flash_q8_trial.py', BASE / 'benchmark_qwen_q6.py',
             BASE / 'benchmark_qwen_resident_0908.py', BASE / 'qwen_private_shared_dispatch_trial_0909c.py',
             BASE / 'qwen_high_quant_trial.py', BASE / 'qwen_split_trial.py',
             BASE / 'measure-model-bandwidth.py', BASE / 'dram_bandwidth.py',
             BASE / 'guarded_inference_request.py', BASE / 'model_measurement_guard.py',
             BASE / 'inference_contention_guard.py', BASE / 'extract_glm_flash_q8_mtp_0908.py',
             *record_sources, Path(peer['exe']), *map(Path, libraries)]
    for config in configs.values():
        paths += [Path(config['command'][0]), Path(config['cpu_library'])]
    plan = dict(prepared=time.time(), label=out.name, target_gb_s=250, capacity_gb_s=380,
                qwen_pid=peer_pid, qwen=peer, qwen_runtime_env=peer_env,
                qwen_libraries={path:sha256(path) for path in libraries}, qwen_model_records=records,
                configurations=configs, source_sha256={str(path):sha256(path) for path in paths},
                sequence=['raw_control', 'raw_candidate', 'raw_candidate', 'raw_control'],
                dissemination=barrier, previous_assessment_sha256=sha256(PREVIOUS),
                tokens=512, max_background_cores=4, model_loaded=False,
                temporary_model_switch_authorized=True,
                scope='Repeated fresh single-conversation raw Q8 prose/code, differing only in the existing NUMA huge-page advice. Read page backing before measurements, with no global policy changes. Qwen is temporarily stopped under the existing whole-server task authorization and its exact command, full environment, working directory, and affinity restored afterward.')
    assert_sources(plan)
    assert_peer(plan)
    out.mkdir()
    write_json(out / 'plan.json', plan)
    print(json.dumps(dict(prepared=True, plan=str(out / 'plan.json'), target_gb_s=250,
                         configurations=list(configs), sequence=plan['sequence'],
                         qwen_service=read_service(18095), model_started=False, model_stopped=False)), flush=True)


def output_record(chunks):
    content, reasoning, finishes = [], [], []
    for chunk in chunks:
        for choice in chunk.get('choices', []):
            delta = choice.get('delta', {})
            content.append(delta.get('content') or '')
            reasoning.append(delta.get('reasoning_content') or delta.get('reasoning') or '')
            if choice.get('finish_reason'):
                finishes.append(choice['finish_reason'])
    return hashlib.sha256(json.dumps([''.join(content), ''.join(reasoning), finishes], ensure_ascii=False).encode()).hexdigest()


def stop_owned_flash(manager):
    """Release a manager-owned child and reap it before the next model scan."""
    current = manager.state['current']
    if current is None:
        return
    manager.quiet()
    manager.stop_flash()
    try:
        os.waitpid(current['pid'], 0)
    except ChildProcessError:
        # An intervening subprocess call may already have reaped this child.
        pass


def measurement_rows(path, current):
    data = json.loads(path.read_text())
    assert data['finished'] and data['input_integrity_verified'] and not data.get('error')
    assert data['server_command'] == current['command'] and data['runtime_env'] == current['runtime_env']
    assert data['bandwidth_target_gb_s'] == 250 and data['bandwidth_capacity_gb_s'] == 380
    assert len(data['checks']) == 2 and all(check['pass_check'] and not check['abort'] for check in data['checks'])
    assert all(sha256(source) == digest for source, digest in data['input_sha256'].items())
    rows = {}
    for row in data['measurements']:
        assert row['counter_metadata']['valid'] and row['counter_metadata']['exit_code'] == 0
        assert not row['abort'] and not row['inference_churn'] and not row['other_inference']
        assert all(row[key]['valid'] for key in ['baseline_before', 'baseline_after', 'decode'])
        assert row['timings']['cache_n'] == 0 and row['timings']['predicted_n'] == 512
        assert row['draft_n'] == current['drafts']
        assert bool(row['timings'].get('draft_n', 0)) == bool(current['drafts'])
        before, after = row['baseline_before']['total_gb_s'], row['baseline_after']['total_gb_s']
        assert max(before, after) <= 19 and abs(before-after) <= 9.5, 'Idle traffic does not qualify attribution'
        chunks = path.parent / f"{row['kind']}-draft{row['draft_n']}" / 'chunks.json'
        rows[row['kind']] = dict(tok_s=row['timings']['predicted_per_second'],
            adjusted_gb_s=row['background_subtracted_gb_s'],
            over_250_gb_s=row['background_subtracted_gb_s'] >= 250,
            output_sha256=output_record(json.loads(chunks.read_text())), chunks_sha256=sha256(chunks),
            generated_tokens=row['timings']['predicted_n'], draft_tokens=row['timings'].get('draft_n', 0),
            accepted_draft_tokens=row['timings'].get('draft_n_accepted', 0),
            completed_answer=row['completed_answer'])
    assert set(rows) == {'prose', 'code'}
    return rows


def execute(out, pause_authorized, lock_fd):
    assert pause_authorized, 'Execution must explicitly opt into the authorized temporary model switch'
    plan = json.loads((out / 'plan.json').read_text())
    assert not (out / 'result.json').exists()
    assert_sources(plan)
    assert_peer(plan)
    manager = Manager()
    assert manager.state['current'] is None and manager.qwen.state['current'] is None
    result = dict(started=time.time(), passed=False, stage='waiting for idle',
                  plan_sha256=sha256(out / 'plan.json'), runs=[], target_reached=False,
                  controller_pid=os.getpid(), pause_authorized=True, qwen_termination_sent=False)
    def save(stage=None):
        if stage:
            result['stage'] = stage
        write_json(out / 'result.json', result)
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, lambda *_: setattr(manager, 'cancelled', True))
    original = None
    environment = None
    try:
        save()
        guard = ModelMeasurementGuard(plan['qwen_pid'], {plan['qwen_pid']:18095}, inference_snapshot)
        result['idle_gate'] = guard.wait_idle(out / 'waiting-for-idle.json')
        result['background_gate'] = wait_background(guard, plan['qwen_pid'], 4, out / 'background-wait.json')
        assert_sources(plan)
        assert_peer(plan)
        environment = process_environment(plan['qwen_pid'])
        private = out / 'qwen-restore-context.private.json'
        with private.open('x') as handle:
            json.dump(dict(info=plan['qwen'], environment=environment), handle)
            handle.write('\n')
        assert private.stat().st_mode & 0o777 == 0o600
        result['restore_context_sha256'] = sha256(private)
        original = ExactProcess(plan['qwen_pid'], expected(plan['qwen']), set())
        guard.assert_idle()
        manager.check_cancel()
        original.terminate()
        result['qwen_termination_sent'] = True
        save('waiting for identified idle Qwen to exit')
        deadline = time.monotonic() + 180
        while not original.exited():
            assert time.monotonic() < deadline, 'Identified Qwen has not finished shutdown'
            time.sleep(.5)
        # Its separate parent may need a moment to reap the exited process.
        while inference_snapshot() and time.monotonic() < deadline:
            time.sleep(.5)
        assert not inference_snapshot() and port_available(18095)
        result['qwen_paused'] = True
        save('exclusive Flash trials')
        for index, name in enumerate(plan['sequence']):
            manager.check_cancel()
            desired = plan['configurations'][name]
            if manager.state['current'] is None or any(manager.state['current'][key] != value for key, value in desired.items()):
                stop_owned_flash(manager)
                actual, env = manager.candidate(**OPTIONS, drafts=desired['drafts'], dissemination=desired['dissemination'], huge_pages=desired['huge_pages'])
                assert actual == desired
                assert_sources(plan)
                manager.start_flash(actual, env)
            current = manager.validate_current()
            guard = ModelMeasurementGuard(current['pid'], {current['pid']:PORT}, inference_snapshot)
            guard.wait_idle(out / 'decode-idle.json', quiet_seconds=15)
            wait_background(guard, current['pid'], 4, out / 'decode-background.json')
            pages_before = audit_pages(current['pid'])
            fraction = pages_before['anonymous_hugepage_fraction']
            assert fraction >= .5 if desired['huge_pages'] else fraction <= .05
            label = f'{out.name}-{index:02d}-{name.replace("_", "-")}'
            command = [sys.executable, '-u', str(BASE / 'measure-model-bandwidth.py'), label,
                       '--port', str(PORT), '--pid', str(current['pid']), '--alias', 'glm-flash-q8-trial',
                       '--drafts', str(current['drafts']), '--tokens', '512', '--request-timeout-seconds', '240',
                       '--allowed-idle-pids', '', '--skip-idle-gate', '--bandwidth-target-gb-s', '250',
                       '--bandwidth-capacity-gb-s', '380', '--chat-template-kwargs', '{"reasoning_effort":"max"}',
                       '--check-reasoning-budget-tokens', '0']
            record = dict(index=index, configuration=name, model_pid=current['pid'], started=time.time(), command=command,
                          page_backing=current['page_backing'], pages_before=pages_before)
            result['runs'].append(record)
            save('measuring ' + label)
            child = subprocess.Popen(command, pass_fds=(lock_fd,))
            try:
                while child.poll() is None:
                    manager.check_cancel()
                    time.sleep(.5)
                assert child.returncode == 0, ('Measurement failed', child.returncode)
            finally:
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=15)
            manager.validate_current()
            pages_after = audit_pages(current['pid'])
            record['pages_after'] = pages_after
            save()
            assert abs(pages_after['anonymous_hugepage_fraction'] - fraction) <= .01, 'Page backing changed during measurement'
            path = BASE / 'results' / label / 'result.json'
            rows = measurement_rows(path, current)
            reference_name = 'raw_control' if name.startswith('raw_') else 'mtp_control'
            reference = next((run['rows'] for run in result['runs'] if run['configuration'] == reference_name and 'rows' in run), None)
            if reference:
                for kind, row in rows.items():
                    assert all(row[key] == reference[kind][key] for key in ['output_sha256', 'generated_tokens', 'draft_tokens', 'accepted_draft_tokens']), 'Candidate or repeat output differs from its matching draft-mode control'
            record.update(finished=time.time(), measurement=str(path), measurement_sha256=sha256(path), rows=rows)
            save()
            print(json.dumps(dict(configuration=name, rows=rows)), flush=True)
        assert_sources(plan)
        result['passed'] = True
        result['configuration_targets'] = {
            name: len([run for run in result['runs'] if run['configuration'] == name]) == 2 and
                  all(row['over_250_gb_s'] for run in result['runs'] if run['configuration'] == name for row in run['rows'].values())
            for name in plan['configurations']}
        result['target_reached'] = any(result['configuration_targets'].values())
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        manager.recovering = True
        try:
            stop_owned_flash(manager)
        finally:
            try:
                if original is not None and original.termination_sent:
                    save('restoring exact Qwen configuration')
                    if not original.exited():
                        assert_peer(plan)
                        result['qwen_original_still_present'] = True
                    else:
                        assert not inference_snapshot() and port_available(18095) and port_available(PORT)
                        assert environment is not None and memory_status()['MemAvailable'] > 300000000000
                        assert_sources(plan)
                        with (out / 'qwen-restored.log').open('w') as log:
                            restored = subprocess.Popen(['taskset', '-c', ','.join(map(str, plan['qwen']['affinity'])), *plan['qwen']['command']],
                                env=environment, cwd=plan['qwen']['cwd'], stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                        result['restored_qwen_pid'] = restored.pid
                        save()
                        deadline, report = time.monotonic() + 1200, 0
                        while True:
                            assert restored.poll() is None, 'Restored Qwen exited during loading'
                            assert set(inference_snapshot()) == {str(restored.pid)}, 'Foreign model appeared during restoration'
                            assert time.monotonic() < deadline, 'Qwen restoration timed out'
                            try:
                                with urllib.request.urlopen('http://127.0.0.1:18095/health', timeout=2) as response:
                                    if json.load(response).get('status') == 'ok':
                                        break
                            except OSError:
                                pass
                            if time.monotonic()-report >= 30:
                                print(json.dumps(dict(restoring_qwen_pid=restored.pid)), flush=True)
                                report = time.monotonic()
                            time.sleep(1)
                        actual = process_info(restored.pid)
                        assert all(actual[key] == plan['qwen'][key] for key in ['exe', 'command', 'cwd', 'affinity'])
                        assert process_environment(restored.pid) == environment
                        mapped = {line.split()[-1] for line in Path(f'/proc/{restored.pid}/maps').read_text().splitlines()
                                  if any('/' + stem in line for stem in ['libggml-', 'libggml.so.', 'libllama.so.'])}
                        assert mapped == set(plan['qwen_libraries'])
                        result.update(qwen_restored=True, restored_qwen=actual, restored_service=read_service(18095))
            except BaseException as error:
                result['restoration_error'] = repr(error)
                result['passed'] = False
                raise
            finally:
                if original is not None:
                    original.close()
                result['finished'] = time.time()
                save('finished' if result.get('qwen_restored') else 'stopped; inspect restoration state')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run'])
    parser.add_argument('label')
    parser.add_argument('--pause-identified-qwen', action='store_true')
    args = parser.parse_args()
    assert re.fullmatch(r'flash-hugepages250-[A-Za-z0-9_-]+', args.label)
    os.umask(0o077)
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        out = BASE / 'results' / args.label
        prepare(out) if args.action == 'prepare' else execute(out, args.pause_identified_qwen, lock.fileno())
