#!/usr/bin/env python3
"""Measure the selected resident Q6 service without adopting or reconfiguring it."""
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

from benchmark_qwen_q6 import wait_background
from model_measurement_guard import ModelMeasurementGuard, read_service
from qwen_high_quant_trial import unit_state
from qwen_split_trial import inference_snapshot, process_environment, process_info, runtime_environment, sha256

BASE = Path(__file__).resolve().parent


def normalized_command(command):
    # The public wrapper may repeat port/alias options; their values are checked
    # separately. Request cache_prompt=false overrides its cache default.
    values = []
    index = 0
    while index < len(command):
        value = command[index]
        if value in ('--alias', '--port'):
            index += 2
            continue
        if value not in ('--cache-prompt', '--no-cache-prompt'):
            values.append(value)
        index += 1
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('label')
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--start-ticks', required=True)
    parser.add_argument('--drafts', default='4')
    parser.add_argument('--tokens', type=int, default=512)
    parser.add_argument('--max-background-cores', type=float, default=4.0)
    args = parser.parse_args()
    assert re.fullmatch(r'[A-Za-z0-9_-]+', args.label)
    drafts = [int(value) for value in args.drafts.split(',')]
    assert drafts == [4], 'This pinned build ignores request draft overrides; measure only its verified launch limit'
    assert 256 <= args.tokens <= 2048
    out = BASE / 'results' / (args.label + '-controller.json')
    assert not out.exists() and not (BASE / 'results' / args.label).exists()
    result = dict(started=time.time(), passed=False, config=vars(args), adopted_server=False,
                  server_reconfigured=False, server_reloaded=False, target_tok_s=30,
                  bandwidth_target_gb_s=190, bandwidth_capacity_gb_s=380, strict_thresholds=True,
                  sent_request_p_min=0, effective_p_min=0.3, effective_draft_limit=4,
                  request_speculative_overrides_supported=False,
                  note='The unchanged measurement helper sends request speculative settings, but this pinned server ignores them. Only the verified launch MTP4/p_min=0.3 configuration is measured.')
    def save():
        out.write_text(json.dumps(result, indent=2) + '\n')
    def interrupted(signum, frame):
        raise InterruptedError('Release the owned benchmark request; preserve resident Qwen')
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupted)
    child = None
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            preset_path = BASE / 'qwen-flash-20tps.json'
            preset = json.loads(preset_path.read_text())
            info = process_info(args.pid)
            assert info['start'] == args.start_ticks
            assert normalized_command(info['command']) == normalized_command(preset['command'])
            alias_values = [info['command'][i + 1].split(',') for i, v in enumerate(info['command']) if v == '--alias']
            live_aliases = alias_values[-1]
            ports = [info['command'][i + 1] for i, v in enumerate(info['command']) if v == '--port']
            assert ports and all(value == '18095' for value in ports)
            assert 'qwen-goal' in live_aliases
            assert all(set(values) <= set(preset['command'][preset['command'].index('--alias') + 1].split(',')) for values in alias_values)
            environment = runtime_environment(process_environment(args.pid))
            assert environment == preset['runtime_env']
            assert not any('PROFILE' in k or 'AUDIT' in k for k in environment)
            assert environment['GGML_CPU_NUMA_THREADS'] == '15'
            assert unit_state()['ActiveState'] == 'inactive'
            assert set(inference_snapshot()) == {str(args.pid)}
            mapped = {line.split()[-1] for line in Path(f'/proc/{args.pid}/maps').read_text().splitlines()
                      if len(line.split()) >= 6 and line.split()[-1].startswith('/')}
            libraries = {}
            for stem, expected_hash in [('libggml-cpu.so.', 'c79e19e38acfd6ada3f0637c136cf9cb10253127778cb214528b2f966fb6dfcf'),
                                        ('libllama.so.', 'd213ba477ef5d3f2a68568b30a31595ef9578b6dce4b3b81e95da6f83a3dac8b')]:
                actual, = [path for path in mapped if '/' + stem in path]
                assert sha256(actual) == expected_hash
                libraries[actual] = expected_hash
            assert sha256(info['exe']) == preset['binary_sha256'][info['exe']]
            result.update(process=info, runtime_env=environment, mapped_libraries=libraries,
                          permitted_live_differences=dict(aliases=alias_values, ports=ports,
                              cache_flags=[v for v in info['command'] if v in ('--cache-prompt', '--no-cache-prompt')]),
                          preset_sha256=sha256(preset_path), server_sha256=sha256(info['exe']))
            result['source_sha256'] = {str(path): sha256(path) for path in [Path(__file__).resolve(),
                BASE / 'measure-model-bandwidth.py', BASE / 'model_measurement_guard.py',
                BASE / 'guarded_inference_request.py', BASE / 'dram_bandwidth.py', preset_path]}
            guard = ModelMeasurementGuard(args.pid, {args.pid: 18095}, inference_snapshot)
            result['idle_gate'] = guard.wait_idle(BASE / 'results' / (args.label + '-idle.json'))
            result['background_gate'] = wait_background(guard, args.pid, args.max_background_cores,
                BASE / 'results' / (args.label + '-background.json'))
            save()
            command = [sys.executable, '-u', str(BASE / 'measure-model-bandwidth.py'), args.label,
                       '--port', '18095', '--pid', str(args.pid), '--alias', 'qwen-goal',
                       '--drafts', args.drafts, '--tokens', str(args.tokens),
                       '--request-timeout-seconds', str(max(120, args.tokens / 3)),
                       '--allowed-idle-pids', '', '--skip-idle-gate',
                       '--bandwidth-capacity-gb-s', '380', '--bandwidth-target-gb-s', '190']
            guard.assert_idle()
            child = subprocess.Popen(command, cwd=BASE, start_new_session=True)
            assert child.wait() == 0, 'Owned measurement did not complete'
            path = BASE / 'results' / args.label / 'result.json'
            measured = json.loads(path.read_text())
            assert measured['input_integrity_verified'] and not measured.get('error')
            assert measured['server_command'] == info['command'] and measured['runtime_env'] == environment
            assert all(row['pass_check'] for row in measured['checks'])
            assert len(measured['measurements']) == 2 * len(drafts)
            rows = []
            for row in measured['measurements']:
                assert row['counter_metadata']['valid'] and row['decode']['valid'] and not row['abort']
                assert row['timings']['cache_n'] == 0, 'Fresh-prompt benchmark unexpectedly reused prompt tokens'
                speed = row['timings']['predicted_per_second']
                traffic = row['background_subtracted_gb_s']
                rows.append(dict(workload=row['kind'], drafts=row['draft_n'], tok_s=speed,
                                 adjusted_gb_s=traffic, capacity_percent=traffic / 380 * 100,
                                 both_targets_met=speed > 30 and traffic > 190))
            after = process_info(args.pid)
            assert all(after[k] == info[k] for k in ('start', 'exe', 'command', 'affinity'))
            assert runtime_environment(process_environment(args.pid)) == environment
            assert all(sha256(path) == digest for path, digest in result['source_sha256'].items())
            result.update(passed=True, measurement=str(path), measurement_sha256=sha256(path), rows=rows,
                          service_after=read_service(18095), target_reached=all(row['both_targets_met'] for row in rows))
            print(json.dumps(dict(passed=True, rows=rows, target_reached=result['target_reached']), indent=2), flush=True)
        except BaseException as error:
            result['error'] = repr(error)
            raise
        finally:
            if child is not None and child.poll() is None:
                child.send_signal(signal.SIGINT)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    child.wait(timeout=10)
            result['finished'] = time.time()
            save()


if __name__ == '__main__':
    main()
