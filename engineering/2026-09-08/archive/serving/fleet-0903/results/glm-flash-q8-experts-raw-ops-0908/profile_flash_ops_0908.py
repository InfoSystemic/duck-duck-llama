#!/usr/bin/env python3
"""Capture bounded Flash Q8 decode stages with the existing file-armed profiler."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import time

from benchmark_qwen_q6 import wait_background
from glm_flash_q8_trial import BASE, OUT, PORT, Manager
from guarded_inference_request import stream_completion
from model_measurement_guard import ModelMeasurementGuard, read_service
from profile_qwen_ops_0908 import parse_trace
from qwen_high_quant_trial import atomic_json
from qwen_split_trial import inference_snapshot, sha256


def text_fields(chunks):
    return {key: ''.join(choice.get('delta', {}).get(key) or ''
                        for chunk in chunks for choice in chunk.get('choices', []))
            for key in ('content', 'reasoning_content')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('label')
    parser.add_argument('--max-background-cores', type=float, default=5.0)
    args = parser.parse_args()
    assert re.fullmatch(r'[A-Za-z0-9_-]+', args.label)
    destination = BASE / 'results' / args.label
    baseline_path = BASE / 'results/glm-flash-q8-experts-raw-0908/result.json'
    os.umask(0o077)
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, lambda sig, frame: (_ for _ in ()).throw(InterruptedError(f'Signal {sig}')))
    with (BASE / 'results/qwen-q6-trial-0907/lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        destination.mkdir(exist_ok=False)
        manager = Manager()
        current = manager.validate_current()
        assert manager.qwen.state['full_stopped'] and not manager.qwen.state.get('current')
        assert current['drafts'] == 0 and current['workers'] == 15
        assert current['batching'] and current['q8_experts']
        count = current['op_profile_count']
        assert 1 <= count <= 256
        environment = current['runtime_env']
        arm = Path(environment['GGML_CPU_OP_PROFILE_ARM_FILE'])
        assert arm == OUT / 'op-profile.arm' and not arm.exists()
        assert environment['GGML_CPU_OP_PROFILE_COUNT'] == str(count)
        assert environment['GGML_CPU_OP_PROFILE'] == '*' and environment['GGML_CPU_OP_PROFILE_SKIP'] == '0'
        baseline = json.loads(baseline_path.read_text())
        assert baseline['input_integrity_verified'] and baseline.get('finished') and not baseline.get('error')
        assert baseline['server_command'] == current['command']
        assert baseline['runtime_env'] == {k: v for k, v in environment.items() if 'PROFILE' not in k}
        guard = ModelMeasurementGuard(current['pid'], {current['pid']: PORT}, inference_snapshot)
        paths = [Path(__file__), BASE / 'guarded_inference_request.py', BASE / 'model_measurement_guard.py',
                 BASE / 'benchmark_qwen_q6.py', BASE / 'glm_flash_q8_trial.py', BASE / 'qwen_split_trial.py',
                 BASE / 'profile_qwen_ops_0908.py', baseline_path]
        result = dict(started=time.time(), current=current, source_sha256={str(p): sha256(p) for p in paths},
                      traces=[], passed=False,
                      note='Instrumented stage times include following barriers; throughput is not a benchmark.')
        for path in paths[:-1]:
            (destination / path.name).write_bytes(path.read_bytes())
        save = lambda: atomic_json(destination / 'result.json', result)
        save()
        owned_arm_inode = None

        def disarm():
            nonlocal owned_arm_inode
            if owned_arm_inode is not None:
                assert arm.stat().st_ino == owned_arm_inode
                arm.unlink()
                owned_arm_inode = None

        try:
            result['idle_gate'] = guard.wait_idle(destination / 'waiting-for-idle.json', quiet_seconds=15)
            result['background_gate'] = wait_background(guard, current['pid'], args.max_background_cores,
                                                        destination / 'background-wait.json')
            for measurement in baseline['measurements']:
                kind, prompt = measurement['kind'], measurement['prompt']
                guard.assert_idle()
                guard.reset_activity()
                log = Path(current['log'])
                offset = log.stat().st_size
                entry = dict(kind=kind, prompt=prompt, chunks=[], abort=[], events_before_arm=32)
                result['traces'].append(entry)
                events = 0

                def on_chunk(chunk):
                    nonlocal events, owned_arm_inode
                    entry['chunks'].append(chunk)
                    if any(c.get('delta', {}).get('content') or c.get('delta', {}).get('reasoning_content')
                           for c in chunk.get('choices', [])):
                        events += 1
                        if events == 32:
                            with arm.open('x') as handle:
                                owned_arm_inode = os.fstat(handle.fileno()).st_ino
                            entry['armed_monotonic'] = time.monotonic()

                body = dict(model='glm-flash-q8-trial', messages=[dict(role='user', content=prompt)],
                            temperature=0, seed=42, max_tokens=256, cache_prompt=False, stream=True,
                            chat_template_kwargs={'reasoning_effort': 'max'})
                entry['payload'] = body
                try:
                    stream_completion(PORT, body, guard.abort_reason, entry['abort'], on_chunk,
                                      interval=0.5, max_elapsed=120)
                finally:
                    disarm()
                    with log.open('rb') as handle:
                        handle.seek(offset)
                        trace_text = handle.read().decode(errors='replace')
                    (destination / f'{kind}.log').write_text(trace_text)
                    entry['stream_events'] = events
                    save()
                assert not entry['abort'] and 'armed_monotonic' in entry
                parsed = parse_trace(trace_text, count)
                atomic_json(destination / f'{kind}-operations.json', parsed)
                entry['summary'] = parsed['groups']
                reference_path = baseline_path.parent / f'{kind}-draft0/chunks.json'
                reference = text_fields(json.loads(reference_path.read_text()))
                observed = text_fields(entry['chunks'])
                entry['stream_prefix_checks'] = {key: reference[key].startswith(value) for key, value in observed.items()}
                entry['stream_characters'] = {key: len(value) for key, value in observed.items()}
                entry['reference_chunks_sha256'] = sha256(reference_path)
                assert sum(entry['stream_characters'].values()) > 0 and all(entry['stream_prefix_checks'].values())
                save()
                print(json.dumps(dict(kind=kind, captured_graphs=len(parsed['graphs']),
                    stream_prefix_checks=entry['stream_prefix_checks'],
                    groups=[dict(first=g['first'], last=g['last'], graphs=g['graphs'],
                                 total_ms=g['total_ms'], operations=g['operations']) for g in parsed['groups']])), flush=True)
            guard.assert_idle()
            manager.validate_current()
            assert all(sha256(Path(path)) == digest for path, digest in result['source_sha256'].items())
            result['health_after'] = read_service(PORT)
            result['passed'] = True
        except BaseException as error:
            result['error'] = repr(error)
            raise
        finally:
            disarm()
            result['finished'] = time.time()
            save()


if __name__ == '__main__':
    main()
