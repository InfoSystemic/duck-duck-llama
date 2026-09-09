#!/usr/bin/env python3
"""Compare Full replay request profiles with IMC counters and exact file-edit checks."""
import argparse
import ast
import json
import math
import os
from pathlib import Path
import re
import sys
import time

from dram_bandwidth import PerfDramRecorder, summarize_samples
from guarded_inference_request import stream_completion
from inference_contention_guard import activity
from model_measurement_guard import ModelMeasurementGuard
from qwen_split_trial import (inference_snapshot, identity_matches, process_info,
                              process_environment, runtime_environment, sha256)


def host_snapshot():
    result = {}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            stat = (proc / 'stat').read_text()
            fields = stat[stat.rfind(')') + 2:].split()
            result[proc.name] = (int(fields[11]) + int(fields[12]), fields[19],
                                 stat[stat.find('(') + 1:stat.rfind(')')])
        except (OSError, ValueError, IndexError):
            pass
    return result


def exact_code(content, expected):
    matched = re.fullmatch(r'\s*```(?:python|py)?\r?\n(.*?)\r?\n```\s*', content, re.S)
    if not matched:
        return dict(passed=False, reason='Expected exactly one Python code block')
    code = matched.group(1)
    try:
        ast.parse(code)
    except SyntaxError as error:
        return dict(passed=False, reason='Invalid Python syntax', error=str(error))
    return dict(passed=code.rstrip('\n') == expected.rstrip('\n'),
                reason='Exact source comparison; only trailing newlines are ignored',
                code_characters=len(code), expected_characters=len(expected))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', required=True)
    args = parser.parse_args()
    assert re.fullmatch(r'[A-Za-z0-9_-]+', args.label)
    base = Path(__file__).resolve().parent
    out = base / 'results' / args.label
    out.mkdir(exist_ok=False)
    dataset_path = base / 'full-replay-workloads-0906.json'
    dataset = json.loads(dataset_path.read_text())
    assert len(dataset['workloads']) == 3
    assert sha256(dataset['source_path']) == dataset['source_sha256']
    plan_path = base / 'results/qwen-even-split-model-trial-0906-staging/plan.json'
    plan = json.loads(plan_path.read_text())
    for pid, expected in plan['protected'].items():
        assert identity_matches(process_info(int(pid)), expected)
    runtime_path = base / 'results/glm53-draft-sweep-bandwidth-0905/result.json'
    runtime = json.loads(runtime_path.read_text())
    command = process_info(4005448)['command']
    environment = runtime_environment(process_environment(4005448))
    assert command == runtime['server_command'] and environment == runtime['runtime_env']
    sources = [Path(__file__).resolve(), dataset_path, Path(dataset['source_path']), plan_path,
               base / 'dram_bandwidth.py', base / 'guarded_inference_request.py',
               base / 'inference_contention_guard.py', base / 'model_measurement_guard.py',
               base / 'qwen_split_trial.py']
    inputs = {str(path): sha256(path) for path in sources}
    for path in sources:
        (out / path.name).write_bytes(path.read_bytes())
    result = dict(started=time.time(), pid=os.getpid(), target_pid=4005448,
                  source_sha256=inputs, measurements=[], server_command=command, runtime_env=environment,
                  target_gb_s=304, note='Systemwide IMC counters; background subtraction is an attribution estimate.')
    def save():
        temporary = out / 'result.json.tmp'
        temporary.write_text(json.dumps(result, indent=2) + '\n')
        temporary.replace(out / 'result.json')
    guard = ModelMeasurementGuard(4005448, {4005448: 18091, 2308651: 18095}, inference_snapshot)
    save()
    try:
        result['idle_gate'] = guard.wait_idle(out / 'waiting-for-idle.json')
        # Alternate the order within adjacent workload pairs.
        order = [(1, 2, 0), (1, 18, .75), (2, 18, .75), (2, 2, 0), (3, 2, 0), (3, 18, .75)]
        result['order'] = order
        for workload_id, draft_n, p_min in order:
            workload = dataset['workloads'][workload_id - 1]
            guard.assert_idle()
            case = out / f'task{workload_id}-n{draft_n}p{int(p_min * 100):02d}'
            payload = dict(model='GLM-5.3', messages=[dict(role='user', content=workload['prompt'])],
                           max_tokens=400, temperature=0, seed=7100, reasoning_effort='low',
                           cache_prompt=False, stream=True,
                           **{'speculative.n_max': draft_n, 'speculative.p_min': p_min})
            entry = dict(workload=workload_id, draft_n=draft_n, p_min=p_min, started=time.time(),
                         payload=payload, abort=[])
            result['measurements'].append(entry)
            save()
            recorder = PerfDramRecorder(case).start()
            first = last = None
            chunks = []
            before_host = host_snapshot()
            before_inference = inference_snapshot()
            cpu_start = time.monotonic()
            try:
                before_start = time.monotonic()
                guard.pause_idle(5)
                before_end = time.monotonic()
                guard.assert_idle()
                guard.reset_activity()
                def on_chunk(chunk):
                    nonlocal first, last
                    chunks.append(chunk)
                    if any(c.get('delta', {}).get(k) for c in chunk.get('choices', [])
                           for k in ('content', 'reasoning_content', 'reasoning')):
                        now = time.monotonic()
                        first = now if first is None else first
                        last = now
                print(json.dumps(dict(measuring_workload=workload_id, draft_n=draft_n, p_min=p_min)), flush=True)
                stream_completion(18091, payload, guard.abort_reason, entry['abort'], on_chunk, interval=.5)
                assert not entry['abort'] and first is not None and last - first >= 5
                after_start = time.monotonic()
                guard.pause_idle(5)
                after_end = time.monotonic()
            finally:
                samples, metadata = recorder.stop()
                (case / 'chunks.json').write_text(json.dumps(chunks, indent=2) + '\n')
                entry.update(counter_metadata=metadata, first_content_monotonic=first,
                             last_content_monotonic=last)
            assert metadata['valid'] and metadata['exit_code'] == 0
            before = summarize_samples(samples, before_start, before_end)
            after = summarize_samples(samples, after_start, after_end)
            decode = summarize_samples(samples, first + .5, last - .5)
            assert all(x['valid'] for x in (before, after, decode)) and decode['sampled_seconds'] >= 4
            timings = [chunk['timings'] for chunk in chunks if chunk.get('timings')][-1]
            assert timings['draft_n'] > 0 and timings['predicted_n'] > 1
            assert math.isclose(timings['predicted_per_second'],
                                1000 * (timings['predicted_n'] - 1) / timings['predicted_ms'])
            content = ''.join(c.get('delta', {}).get('content') or ''
                              for chunk in chunks for c in chunk.get('choices', []))
            reasoning = ''.join(c.get('delta', {}).get('reasoning_content') or ''
                                for chunk in chunks for c in chunk.get('choices', []))
            quality = exact_code(content, workload['expected_code'])
            elapsed = time.monotonic() - cpu_start
            loads, _ = activity(before_host, host_snapshot(), elapsed, own_pid=4005448)
            other, churn = activity(before_inference, inference_snapshot(), elapsed, own_pid=4005448)
            assert not churn and all(x['cpu_percent'] < 1 for x in other)
            adjusted = max(0, decode['total_gb_s'] - max(before['total_gb_s'], after['total_gb_s']))
            finish = [c['finish_reason'] for chunk in chunks for c in chunk.get('choices', []) if c.get('finish_reason')]
            quality['finished_naturally'] = finish == ['stop']
            quality['passed'] = quality['passed'] and quality['finished_naturally']
            entry.update(timings=timings, quality=quality, content=content, reasoning_characters=len(reasoning),
                         finish_reasons=finish, before=before, after=after, decode=decode,
                         adjusted_gb_s=adjusted, utilization_percent=adjusted / 3.8,
                         approx_gb_per_generated_token=adjusted / timings['predicted_per_second'],
                         other_inference=other, inference_churn=churn,
                         other_host_cpu=sorted(loads, key=lambda x: -x['cpu_percent'])[:12], finished=time.time())
            save()
            print(json.dumps(dict(workload=workload_id, draft_n=draft_n, p_min=p_min,
                                  tok_s=timings['predicted_per_second'], adjusted_gb_s=adjusted,
                                  tokens=timings['predicted_n'], quality=quality)), flush=True)
        assert all(sha256(path) == digest for path, digest in inputs.items())
        assert process_info(4005448)['command'] == command
        assert runtime_environment(process_environment(4005448)) == environment
        result['input_integrity_verified'] = True
        result['all_outputs_exact'] = all(x['quality']['passed'] for x in result['measurements'])
        result['completed'] = True
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        result['finished'] = time.time()
        save()


if __name__ == '__main__':
    main()
