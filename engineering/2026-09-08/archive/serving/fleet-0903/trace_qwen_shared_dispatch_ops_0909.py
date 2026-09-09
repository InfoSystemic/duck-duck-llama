#!/usr/bin/env python3
"""Capture bounded operation traces on the validated shared-dispatch runtime."""
import argparse
import copy
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
from compare_qwen_private_shared_dispatch_0909 import read_run
from guarded_inference_request import stream_completion
from model_measurement_guard import ModelMeasurementGuard, read_service
from profile_qwen_ops_0908 import parse_trace
from qwen_high_quant_trial import port_available
from qwen_private_shared_dispatch_trial_0909b import dispatch_log, memory_gate, verify_plan
from qwen_split_trial import inference_snapshot, process_environment, process_info, runtime_environment, sha256

BASE = Path(__file__).resolve().parent
PORT = 18155
REFERENCE = 'qwen-private-shared-dispatch-on-0909c'
COUNT = 80


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def prepare(out):
    reference = read_run(REFERENCE)
    plan = copy.deepcopy(reference['plan'])
    verify_plan(plan)
    assert set(inference_snapshot()) == {str(plan['peer_pid'])} and port_available(PORT)
    assert plan['shared_dispatch'] == 'on' and plan['drafts'] == 4
    assert plan['logging_verbosity'] == 4
    arm = out / 'op-profile.arm'
    assert not out.exists() and not arm.exists()
    env = dict(plan['runtime_env'], GGML_CPU_OP_PROFILE='*', GGML_CPU_OP_PROFILE_COUNT=str(COUNT),
               GGML_CPU_OP_PROFILE_SKIP='0', GGML_CPU_OP_PROFILE_ARM_FILE=str(arm))
    plan.update(prepared=time.time(), label=out.name, runtime_env=env, profile=False,
                diagnostic_only=True, op_profile_count=COUNT, op_profile_arm_file=str(arm),
                reference_label=REFERENCE, reference_outputs=reference['rows'],
                memory_preflight=memory_gate(), no_model_loaded=True,
                scope='Diagnostic-only operation traces of the validated shared-dispatch Q6/Q8 MTP4 stack. Timers include following barriers; no throughput or bandwidth claim is made from these instrumented requests.')
    sources = [Path(__file__).resolve(), BASE / 'qwen_private_shared_dispatch_trial_0909b.py',
               BASE / 'profile_qwen_ops_0908.py', BASE / 'compare_qwen_private_shared_dispatch_0909.py',
               BASE / 'qwen_split_trial.py', BASE / 'inference_contention_guard.py',
               BASE / 'results' / REFERENCE / 'plan.json', BASE / 'results' / REFERENCE / 'result.json']
    plan['source_sha256'].update({str(path): sha256(path) for path in sources})
    verify_plan(plan)
    out.mkdir()
    write_json(out / 'plan.json', plan)
    print(json.dumps(dict(prepared=True, plan=str(out / 'plan.json'), diagnostic_only=True,
                          graphs_per_request=COUNT, peer_preserved=plan['peer_pid'])), flush=True)


def output_record(chunks):
    content, reasoning, finishes = [], [], []
    for chunk in chunks:
        for choice in chunk.get('choices', []):
            delta = choice.get('delta', {})
            content.append(delta.get('content') or '')
            reasoning.append(delta.get('reasoning_content') or delta.get('reasoning') or '')
            if choice.get('finish_reason'):
                finishes.append(choice['finish_reason'])
    value = [''.join(content), ''.join(reasoning), finishes]
    timings = [chunk['timings'] for chunk in chunks if chunk.get('timings')]
    assert timings
    return value, timings[-1], hashlib.sha256(json.dumps(value, ensure_ascii=False).encode()).hexdigest()


def payload(prompt):
    return dict(model='qwen-q6-private', messages=[dict(role='user', content=prompt)],
                temperature=0, seed=42, max_tokens=512, cache_prompt=False, stream=True,
                chat_template_kwargs={'enable_thinking': False})


def execute(out):
    plan = json.loads((out / 'plan.json').read_text())
    assert plan['diagnostic_only'] and not (out / 'result.json').exists()
    verify_plan(plan)
    assert set(inference_snapshot()) == {str(plan['peer_pid'])} and port_available(PORT)
    arm = Path(plan['op_profile_arm_file'])
    assert arm == out / 'op-profile.arm' and not arm.exists()
    owned_pid = None
    model = None
    peer_guard = ModelMeasurementGuard(plan['peer_pid'], {plan['peer_pid']: 18095},
                                      lambda: inference_snapshot(exclude=owned_pid))
    result = dict(started=time.time(), passed=False, controller_pid=os.getpid(),
                  plan_sha256=sha256(out / 'plan.json'), peer_pid=plan['peer_pid'],
                  diagnostic_only=True, target_reached=False, model_started=False, checks=[], traces=[])
    def save(stage=None):
        if stage:
            result['stage'] = stage
        write_json(out / 'result.json', result)
    def interrupted(signum, frame):
        raise InterruptedError('Release only the owned diagnostic model')
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupted)
    save('waiting for exclusive CPU use')
    try:
        result['idle_gate'] = peer_guard.wait_idle(out / 'waiting-for-idle.json')
        result['background_gate'] = wait_background(peer_guard, plan['peer_pid'], 4, out / 'background-wait.json')
        verify_plan(plan)
        result['memory_before_load'] = memory_gate()
        assert port_available(PORT)
        environment = {key: value for key, value in os.environ.items() if key not in runtime_environment(os.environ)}
        environment.update(plan['runtime_env'])
        with (out / 'model.log').open('w') as log:
            model = subprocess.Popen(['taskset', '-c', '0-127', *plan['command']], cwd=BASE,
                                     env=environment, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        owned_pid = model.pid
        result.update(model_pid=model.pid, model_started=True)
        save('loading private Qwen for operation tracing')
        deadline, report = time.monotonic() + 900, 0
        while True:
            assert model.poll() is None, 'Owned model exited during loading'
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
                print(json.dumps(dict(loading_private_pid=model.pid, diagnostic_only=True,
                                      peer_preserved=plan['peer_pid'])), flush=True)
                report = now
            time.sleep(.5)
        current = process_info(model.pid)
        assert current['command'] == plan['command'] and current['affinity'] == list(range(128))
        assert runtime_environment(process_environment(model.pid)) == plan['runtime_env']
        mapped = Path(f'/proc/{model.pid}/maps').read_text().splitlines()
        for stem, path in [('libggml-cpu.so.', plan['cpu']), ('libggml-base.so.', plan['base']), ('libllama.so.', plan['llama'])]:
            assert {row.split()[-1] for row in mapped if '/' + stem in row} == {str(Path(path).resolve())}
        result['current'] = current
        dispatch = dispatch_log(out)
        result['dispatch_after_load'] = dispatch
        assert any(ranks == '4' and dispatch['attached'].count((group, ranks)) >= 2 for group, ranks in dispatch['created'])
        assert 'CPU_OP_PROFILE index=' not in (out / 'model.log').read_text()
        class ReservedMemoryGuard(ModelMeasurementGuard):
            def inspect(self, *args, **kwargs):
                memory_gate(loading=True)
                return super().inspect(*args, **kwargs)
        guard = ReservedMemoryGuard(model.pid, {model.pid: PORT, plan['peer_pid']: 18095}, inference_snapshot)
        save('waiting for decode idle')
        result['decode_idle_gate'] = guard.wait_idle(out / 'decode-idle.json', quiet_seconds=15)
        result['decode_background_gate'] = wait_background(guard, model.pid, 4, out / 'decode-background.json')
        for prompt, expected in [('What is 17 * 23? Reply with only the number.', '391'),
                                 ('Name the capital city of France. Reply with one word.', 'Paris')]:
            guard.assert_idle()
            guard.reset_activity()
            check = dict(expected=expected, abort=[])
            result['checks'].append(check)
            chunks = stream_completion(PORT, payload(prompt), guard.abort_reason, check['abort'], interval=.5, max_elapsed=180)
            value, timings, digest = output_record(chunks)
            check.update(chunks=chunks, passed=value[0].strip().rstrip('.') == expected,
                         timings=timings, output_sha256=digest)
            save('short checks')
            assert check['passed'] and not check['abort'] and timings['cache_n'] == 0
            print('short check', check['passed'], expected, flush=True)
        prompts = [('prose', 'Explain how a refrigerator moves heat. Give a detailed explanation in plain English.'),
                   ('code', 'Write a Python function that merges two sorted lists. Include an explanation of its time complexity.')]
        for kind, prompt in prompts:
            assert model.poll() is None and not arm.exists()
            guard.assert_idle()
            guard.reset_activity()
            log = out / 'model.log'
            offset = log.stat().st_size
            entry = dict(kind=kind, chunks=[], abort=[], events_before_arm=32, started=time.time())
            result['traces'].append(entry)
            events = 0
            def on_chunk(chunk):
                nonlocal events
                entry['chunks'].append(chunk)
                if any(any(choice.get('delta', {}).get(key) for key in ('content', 'reasoning_content', 'reasoning'))
                       for choice in chunk.get('choices', [])):
                    events += 1
                    if events == 32:
                        with arm.open('x') as handle:
                            handle.write('')
                        entry['armed_monotonic'] = time.monotonic()
                        print(json.dumps(dict(tracing=kind, armed_after_events=events, graphs=COUNT)), flush=True)
            save('tracing ' + kind)
            try:
                stream_completion(PORT, payload(prompt), guard.abort_reason, entry['abort'], on_chunk,
                                  interval=.5, max_elapsed=180)
            finally:
                arm.unlink(missing_ok=True)
                entry['stream_events'] = events
                save()
            deadline, stable_since, size = time.monotonic() + 10, time.monotonic(), log.stat().st_size
            while time.monotonic() - stable_since < .5:
                assert model.poll() is None and time.monotonic() < deadline
                peer_guard.assert_idle()
                time.sleep(.1)
                current_size = log.stat().st_size
                if current_size != size:
                    size, stable_since = current_size, time.monotonic()
            with log.open('rb') as handle:
                handle.seek(offset)
                text = handle.read().decode()
            trace_path = out / (kind + '.log')
            trace_path.write_text(text)
            assert not entry['abort'] and 'armed_monotonic' in entry
            parsed = parse_trace(text, COUNT)
            for graph in parsed['graphs']:
                indices = [node['index'] for node in graph['nodes']]
                assert len(indices) == len(set(indices))
                assert all(0 <= index < graph['node_count'] for index in indices)
            operations = out / (kind + '-operations.json')
            write_json(operations, parsed)
            value, timings, digest = output_record(entry['chunks'])
            reference = plan['reference_outputs'][kind]
            assert digest == reference['output_sha256'], 'Instrumented output differs from the reference'
            assert timings['cache_n'] == 0
            for actual, expected in [('predicted_n', 'generated_tokens'), ('draft_n', 'draft_tokens'), ('draft_n_accepted', 'accepted_draft_tokens')]:
                assert timings[actual] == reference[expected], actual
            entry.update(timings=timings, output_sha256=digest, output_and_draft_counts_match=True,
                         operations=str(operations), operations_sha256=sha256(operations),
                         trace_sha256=sha256(trace_path), summary=parsed['groups'], finished=time.time())
            save()
            print(json.dumps(dict(kind=kind, graphs=len(parsed['graphs']), output_matches=True,
                                  groups=[dict(first=g['first'], last=g['last'], graphs=g['graphs'],
                                               total_ms=g['total_ms'], operations=g['operations']) for g in parsed['groups']])), flush=True)
        guard.assert_idle()
        verify_plan(plan)
        result.update(passed=True, health_after=read_service(PORT))
    except BaseException as error:
        result['passed'] = False
        result['error'] = repr(error)
        raise
    finally:
        arm.unlink(missing_ok=True)
        if model is not None and model.poll() is None:
            model.terminate()
            try:
                model.wait(timeout=60)
            except subprocess.TimeoutExpired:
                model.kill()
                model.wait(timeout=20)
        result['owned_model_exit'] = model.returncode if model is not None else None
        if (out / 'model.log').exists():
            dispatch = dispatch_log(out)
            result['dispatch_after_release'] = dispatch
            result['dispatch_cleanup_valid'] = sorted(group for group, ranks in dispatch['created']) == sorted(dispatch['destroyed'])
        try:
            peer = process_info(plan['peer_pid'])
            result['peer_preserved'] = all(peer[key] == plan['peer'][key] for key in ('start', 'exe', 'command', 'affinity'))
            result['peer_environment_preserved'] = runtime_environment(process_environment(plan['peer_pid'])) == plan['peer_runtime_env']
            result['peer_service_after'] = read_service(18095)
        except Exception as error:
            result.update(peer_preserved=False, peer_observation_error=repr(error))
        result['finished'] = time.time()
        save('finished; private diagnostic model released')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run'])
    parser.add_argument('label')
    args = parser.parse_args()
    assert re.fullmatch(r'qwen-private-[A-Za-z0-9_-]+', args.label)
    os.umask(0o077)
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        out = BASE / 'results' / args.label
        prepare(out) if args.action == 'prepare' else execute(out)
