#!/usr/bin/env python3
"""Capture bounded decode operation traces from the existing armed profiler."""
import argparse
from collections import defaultdict
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import time

from benchmark_qwen_q6 import wait_background
from guarded_inference_request import stream_completion
from model_measurement_guard import ModelMeasurementGuard, read_service
from qwen_high_quant_trial import BASE, OUT, STATE, atomic_json
from qwen_split_trial import inference_snapshot, process_environment, process_info, runtime_environment, sha256


HEADER = re.compile(r"CPU_OP_PROFILE index=(\d+) cpu=(\d+) graph=(\S+) nodes=(\d+) total=([\d.]+) ms first='([^']*)' last='([^']*)'")
NODE = re.compile(r"CPU_OP_PROFILE cpu=(\d+) graph=(\S+) node=(\d+) op=(\S+) time=([\d.]+) ms name='([^']*)' src0_type=(\S+) src0_ne=\[([^]]+)\] src0_name='([^']*)'")


def parse_trace(text, count):
    graphs, active = [], {}
    for line in text.splitlines():
        header = HEADER.search(line)
        if header:
            index, cpu, pointer, nodes, total, first, last = header.groups()
            graph = dict(index=int(index), cpu=int(cpu), pointer=pointer, node_count=int(nodes),
                         total_ms=float(total), first=first, last=last, nodes=[])
            graphs.append(graph)
            active[(cpu, pointer)] = graph
        node = NODE.search(line)
        if node:
            cpu, pointer, index, op, elapsed, name, kind, shape, source = node.groups()
            graph = active[(cpu, pointer)]
            graph['nodes'].append(dict(index=int(index), op=op, ms=float(elapsed), name=name,
                                       src0_type=kind, src0_ne=list(map(int, shape.split(','))), src0_name=source))
    assert sorted(g['index'] for g in graphs) == list(range(count)), 'Incomplete graph capture'
    assert f'CPU_OP_PROFILE complete count={count}' in text
    groups = defaultdict(lambda: dict(graphs=0, total_ms=0.0, operations=defaultdict(float), families=defaultdict(float)))
    for graph in graphs:
        group = groups[(graph['first'], graph['last'])]
        group['graphs'] += 1
        group['total_ms'] += graph['total_ms']
        for node in graph['nodes']:
            group['operations'][node['op']] += node['ms']
            family = re.sub(r'\d+', '#', node['name'])
            group['families'][family] += node['ms']
    summary = []
    for (first, last), group in groups.items():
        summary.append(dict(first=first, last=last, graphs=group['graphs'], total_ms=group['total_ms'],
                            operations=dict(sorted(group['operations'].items(), key=lambda x: -x[1])),
                            families=dict(sorted(group['families'].items(), key=lambda x: -x[1])[:40])))
    return dict(graphs=graphs, groups=summary,
                note='Stage times include the following barrier. Summed socket times are not elapsed wall time. Fused work is charged to its first node.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('label')
    parser.add_argument('--max-background-cores', type=float, default=5.0)
    args = parser.parse_args()
    assert re.fullmatch(r'[A-Za-z0-9_-]+', args.label)
    destination = BASE / 'results' / args.label
    os.umask(0o077)
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, lambda sig, frame: (_ for _ in ()).throw(InterruptedError(f'Signal {sig}')))
    with (OUT / 'lifecycle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        destination.mkdir(exist_ok=False)
        state = json.loads(STATE.read_text())
        current = state['current']
        assert state['full_stopped'] and not current['original']
        assert current['drafts'] == 4 and current['draft_p_min'] == 0.3
        assert current['q8_wide_batch'] and current['no_cache_prompt']
        count = current['op_profile_count']
        assert 1 <= count <= 256
        pid = current['pid']
        info = process_info(pid)
        assert info['start'] == current['info']['start'] and info['command'] == current['command']
        environment = runtime_environment(process_environment(pid))
        assert environment == current['runtime_env']
        arm = Path(environment['GGML_CPU_OP_PROFILE_ARM_FILE'])
        assert arm == OUT / 'op-profile.arm' and not arm.exists()
        assert environment['GGML_CPU_OP_PROFILE_COUNT'] == str(count)
        assert environment['GGML_CPU_OP_PROFILE'] == '*'
        assert int(Path(environment['GGML_CPU_NUMA_THREADS_FILE']).read_text()) == 15
        guard = ModelMeasurementGuard(pid, {pid: 18095}, inference_snapshot)
        paths = [Path(__file__), BASE / 'guarded_inference_request.py', BASE / 'model_measurement_guard.py',
                 BASE / 'benchmark_qwen_q6.py', BASE / 'qwen_high_quant_trial.py', BASE / 'qwen_split_trial.py']
        result = dict(started=time.time(), current=current, source_sha256={str(p): sha256(p) for p in paths},
                      traces=[], passed=False, note='Instrumented decode trace; reported throughput is not a benchmark result.')
        for path in paths:
            (destination / path.name).write_bytes(path.read_bytes())
        save = lambda: atomic_json(destination / 'result.json', result)
        save()
        try:
            result['idle_gate'] = guard.wait_idle(destination / 'waiting-for-idle.json')
            result['background_gate'] = wait_background(guard, pid, args.max_background_cores)
            prompts = [('prose', 'Explain how a refrigerator moves heat. Give a detailed explanation in plain English.'),
                       ('code', 'Write a Python function that merges two sorted lists. Include an explanation of its time complexity.')]
            for kind, prompt in prompts:
                guard.assert_idle()
                guard.reset_activity()
                log = Path(current['log'])
                offset = log.stat().st_size
                entry = dict(kind=kind, prompt=prompt, chunks=[], abort=[], events_before_arm=32)
                result['traces'].append(entry)
                events = 0
                def on_chunk(chunk):
                    nonlocal events
                    entry['chunks'].append(chunk)
                    if any(c.get('delta', {}).get('content') or c.get('delta', {}).get('reasoning_content')
                           for c in chunk.get('choices', [])):
                        events += 1
                        if events == 32:
                            with arm.open('x') as handle:
                                handle.write('')
                            entry['armed_monotonic'] = time.monotonic()
                body = dict(model='qwen-q6-trial', messages=[dict(role='user', content=prompt)],
                            temperature=0, seed=42, max_tokens=256, cache_prompt=False, stream=True,
                            chat_template_kwargs={'enable_thinking': False})
                try:
                    stream_completion(18095, body, guard.abort_reason, entry['abort'], on_chunk,
                                      interval=0.5, max_elapsed=120)
                finally:
                    arm.unlink(missing_ok=True)
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
                save()
                print(json.dumps(dict(kind=kind, captured_graphs=len(parsed['graphs']),
                                      groups=[dict(first=g['first'], last=g['last'], graphs=g['graphs'],
                                                   total_ms=g['total_ms'], operations=g['operations'])
                                              for g in parsed['groups']])), flush=True)
            guard.assert_idle()
            assert process_info(pid)['start'] == info['start']
            assert all(sha256(path) == digest for path, digest in result['source_sha256'].items())
            result['health_after'] = read_service(18095)
            result['passed'] = True
        except BaseException as error:
            result['error'] = repr(error)
            raise
        finally:
            arm.unlink(missing_ok=True)
            result['finished'] = time.time()
            save()


if __name__ == '__main__':
    main()
