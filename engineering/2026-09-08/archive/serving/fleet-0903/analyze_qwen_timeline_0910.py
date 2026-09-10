"""Parse diagnostic graph timelines without summing concurrent sockets as wall time."""
from collections import Counter, defaultdict
import re

from profile_qwen_ops_0908 import HEADER, NODE, parse_trace

TIMELINE = re.compile(r'CPU_OP_TIMELINE index=(\d+) cpu=(\d+) graph=(\S+) threads=(\d+) start_us=(\d+) end_us=(\d+)')
TIMES = re.compile(r'start_us=(\d+) work_end_us=(\d+) end_us=(\d+) dst_ne=\[([0-9,]+)\]')


def role(graph):
    if graph['first'] == 'hc_init':
        return 'target'
    if graph['first'] == 'norm-48' and graph['last'] == 'result_output':
        return 'draft'
    return 'input'


def union_us(intervals):
    total, end = 0, None
    for start, stop in sorted(intervals):
        assert start <= stop
        if end is None or start > end:
            total += stop-start
            end = stop
        elif stop > end:
            total += stop-end
            end = stop
    return total


def analyze(text, count):
    parsed = parse_trace(text, count)
    graphs = {g['index']: g for g in parsed['graphs']}
    active, headers = {}, {}
    for line in text.splitlines():
        match = TIMELINE.search(line)
        if match:
            index, cpu, pointer, threads, start, end = match.groups()
            index = int(index)
            assert index not in headers
            headers[index] = dict(cpu=int(cpu), pointer=pointer, threads=int(threads), start_us=int(start), end_us=int(end))
        match = HEADER.search(line)
        if match:
            index, cpu, pointer, *_ = match.groups()
            active[(int(cpu), pointer)] = int(index)
        match = NODE.search(line)
        if match:
            cpu, pointer, index, *_ = match.groups()
            graph = graphs[active[(int(cpu), pointer)]]
            node, = [n for n in graph['nodes'] if n['index'] == int(index)]
            timings = TIMES.search(line)
            assert timings, 'Missing monotonic node timestamps'
            start, work_end, end, shape = timings.groups()
            node.update(start_us=int(start), work_end_us=int(work_end), end_us=int(end), dst_ne=list(map(int, shape.split(','))))
    assert len(headers) == count
    for index, graph in graphs.items():
        graph.update(headers[index])
        graph['role'] = role(graph)
        assert graph['start_us'] <= graph['end_us']
        prior_end = graph['start_us']
        for node in graph['nodes']:
            assert prior_end <= node['start_us'] <= node['work_end_us'] <= node['end_us'] <= graph['end_us']
            assert abs(node['ms']*1000-(node['end_us']-node['start_us'])) < .001
            prior_end = node['end_us']
    summaries = []
    for name in ('target', 'draft', 'input'):
        selected = [g for g in graphs.values() if g['role'] == name]
        intervals = [(g['start_us'],g['end_us']) for g in selected]
        summaries.append(dict(role=name, graphs=len(selected), cpus=dict(Counter(g['cpu'] for g in selected)),
            summed_socket_us=sum(stop-start for start,stop in intervals), interval_union_us=union_us(intervals)))
    # Connected overlapping intervals delimit fully captured four-socket graph waves.
    waves = []
    for name in ('target', 'draft'):
        groups = []
        for graph in sorted((g for g in graphs.values() if g['role'] == name), key=lambda g:g['start_us']):
            if not groups or graph['start_us'] >= max(g['end_us'] for g in groups[-1]):
                groups.append([graph])
            else:
                groups[-1].append(graph)
        for group in groups:
            if len(group) != 4 or {g['cpu'] for g in group} != {0,16,32,48}:
                continue
            last = max(group,key=lambda g:g['end_us'])
            elapsed = max(g['end_us'] for g in group)-min(g['start_us'] for g in group)
            families, operations = defaultdict(int), defaultdict(int)
            for node in last['nodes']:
                duration = node['end_us']-node['start_us']
                families[re.sub(r'\d+', '#', node['name'])] += duration
                operations[node['op']] += duration
            waves.append(dict(role=name, graph_indices=[g['index'] for g in group], elapsed_envelope_us=elapsed,
                last_finishing_socket=last['cpu'], last_finishing_graph_us=last['end_us']-last['start_us'],
                last_finishing_graph_ops_us=dict(sorted(operations.items(),key=lambda p:-p[1])),
                last_finishing_graph_families_us=dict(sorted(families.items(),key=lambda p:-p[1])[:15])))
    assert any(w['role']=='target' for w in waves) and any(w['role']=='draft' for w in waves)
    return dict(graphs=list(graphs.values()), summaries=summaries, complete_four_socket_waves=waves,
        all_graph_interval_union_us=union_us((g['start_us'],g['end_us']) for g in graphs.values()),
        timestamp_order_valid=True,
        limitations=[
            'Worker-zero stage timestamps include internal kernel waits; work_end is before the following outer barrier.',
            'The last-finishing socket is descriptive and does not prove a removable critical-path fraction.',
            'Graph logging and profiler overhead occur outside reported graph intervals; these intervals are not full request latency.',
            'Short input graphs and incomplete first/last waves are excluded from four-socket wave attribution.',
            'Recorded background CPU load can delay individual sockets. Instrumented throughput is excluded from speed claims.'])
