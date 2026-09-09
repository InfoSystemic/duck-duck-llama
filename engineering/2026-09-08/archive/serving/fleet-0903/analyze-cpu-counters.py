#!/usr/bin/env python3
"""Validate attached perf stat CSV and summarize process CPU events."""
import argparse
import csv
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('result', type=Path)
options = parser.parse_args()
result = json.loads(options.result.read_text())
events = result['config']['counter_events'].split(',')
assert len(events) == len(set(events)) == 5
assert not result.get('error')
for profile in result['profiles']:
    if not profile.get('finished'):
        continue
    assert profile['perf_exit'] == 0 and not profile['abort']
    start, end = profile['collection_window_monotonic']
    assert profile['first_content_monotonic'] <= start <= end <= profile['last_content_monotonic']
    path = Path(profile['counter_log'])
    threads = {}
    for columns in csv.reader(path.read_text().splitlines()):
        if not columns:
            continue
        assert len(columns) >= 6 and columns[3] in events, columns
        thread, value, unit, event, runtime, running = columns[:6]
        assert unit == '' and value != '<not supported>', columns
        entry = dict(value=None if value == '<not counted>' else int(value),
                     runtime_ns=float(runtime), running_percent=float(running))
        assert entry['runtime_ns'] >= 0 and 0 <= entry['running_percent'] <= 100
        assert entry['value'] is None or entry['value'] >= 0
        values = threads.setdefault(thread, {})
        assert event not in values, (thread, event)
        values[event] = entry
    assert threads and all(set(values) == set(events) for values in threads.values())
    active, inactive = {}, []
    for thread, values in threads.items():
        if all(value['value'] is None or value['value'] == 0 for value in values.values()):
            inactive.append(thread)
            continue
        assert all(v['value'] is not None and v['runtime_ns'] > 0 and v['running_percent'] >= 99
                   for v in values.values()), ('Incomplete active-thread counters', thread, values)
        active[thread] = values
    assert active
    totals = {event: sum(values[event]['value'] for values in active.values()) for event in events}
    cycles = totals['cycles:u']
    instructions = totals['instructions:u']
    assert cycles > 0 and instructions > 0
    summary = dict(valid=True, draft_n=profile['draft_n'], active_threads=len(active), inactive_threads=len(inactive),
                   collection_seconds=end-start, totals=totals, instructions_per_cycle=instructions/cycles,
                   load_walk_active_percent_of_cycles=100*totals['dtlb_load_misses.walk_active:u']/cycles,
                   completed_load_walks_per_1000_instructions=1000*totals['dtlb_load_misses.walk_completed:u']/instructions,
                   retired_l3_load_misses_per_1000_instructions=1000*totals['mem_load_retired.l3_miss:u']/instructions,
                   minimum_running_percent=min(v['running_percent'] for t in active.values() for v in t.values()),
                   note='Process event totals. Walk-active cycles are activity, not a direct measure of stall time; retired L3 misses exclude prefetch traffic. IPC includes spin waits.',
                   threads=threads)
    (path.parent / 'cpu-counter-analysis.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({key:value for key,value in summary.items() if key != 'threads'}))
