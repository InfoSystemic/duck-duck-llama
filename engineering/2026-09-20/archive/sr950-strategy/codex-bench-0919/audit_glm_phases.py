#!/usr/bin/env python3
"""Audit the archived sequential GLM MTP2 phase runs with explicit graph shapes."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
from statistics import mean

ROOT = Path('/home/user/InfoSystemic/AI-Server/serving/fleet-0912-ctx')
HERE = Path(__file__).resolve().parent
PHASE = re.compile(
    r'^(\d+)\.(\d+)\.(\d+)\.(\d+) W GRAPH_PHASE '
    r'tokens=(\d+) nodes=(\d+) reused=(\d) '
    r'apply=([\d.]+) prepare=([\d.]+) inputs=([\d.]+) '
    r'compute=([\d.]+) total=([\d.]+) ms')
FIELDS = ('apply', 'prepare', 'inputs', 'compute', 'total')
VERIFY_NODES, DRAFT_NODES = 7152, 89


def provenance(path):
    return {'path': str(path), 'sha256': sha256(path.read_bytes()).hexdigest()}


def parse(path, allowed_nodes=(VERIFY_NODES, DRAFT_NODES)):
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if 'GRAPH_PHASE' not in line:
            continue
        m = PHASE.search(line)
        if not m:
            raise ValueError(f'Unparsed phase at {path}:{line_number}')
        a = m.groups()
        row = dict(line=line_number,
                   end_ms=int(a[0])*60000 + int(a[1])*1000 + int(a[2]) + int(a[3])/1000,
                   tokens=int(a[4]), nodes=int(a[5]), reused=int(a[6]))
        row.update(zip(FIELDS, map(float, a[7:])))
        assert row['nodes'] in allowed_nodes, row
        assert abs(sum(row[f] for f in FIELDS[:-1])-row['total']) <= .0031, row
        if rows:
            assert row['end_ms'] > rows[-1]['end_ms']
        rows.append(row)
    assert rows
    return rows


def totals(rows):
    return {f: sum(r[f] for r in rows) for f in FIELDS}


def audit(mode, stamp):
    log = ROOT/'results'/f'glm-flash-native-20260914-{stamp}.log'
    rows = parse(log)
    starts = [i for i, r in enumerate(rows) if r['nodes'] == VERIFY_NODES and r['tokens'] > 4]
    assert len(starts) == 3 and starts[0] == 0
    requests = []
    for number, (lo, hi) in enumerate(zip(starts, starts[1:] + [len(rows)]), 1):
        bench = ROOT/'results'/f'bench3-flashfix-{mode}-phase-20260914-133740-{number}.json'
        t = json.loads(bench.read_text())['timings']
        rr = rows[lo:hi]
        prompt_tokens, boundary = 0, None
        for i, r in enumerate(rr):
            if r['nodes'] != VERIFY_NODES:
                continue
            prompt_tokens += r['tokens']
            assert prompt_tokens <= t['prompt_n']
            assert rr[i+1]['nodes'] == DRAFT_NODES and rr[i+1]['tokens'] == r['tokens']
            if prompt_tokens == t['prompt_n']:
                boundary = i+2
                break
        assert boundary is not None
        prompt, decode = rr[:boundary], rr[boundary:]
        verify = [r for r in decode if r['nodes'] == VERIFY_NODES]
        draft = [r for r in decode if r['nodes'] == DRAFT_NODES]
        expected_verifications = t['predicted_n'] - 1 - t['draft_n_accepted']
        assert len(verify) == expected_verifications
        assert abs(sum(r['tokens'] for r in verify) - (t['draft_n'] + expected_verifications)) <= 2
        vi = [i for i, r in enumerate(decode) if r['nodes'] == VERIFY_NODES]
        cycles = []
        excluded = Counter()
        for a, b in zip(vi, vi[1:]):
            cycle = decode[a+1:b+1]
            shape = [(r['nodes'], r['tokens']) for r in cycle]
            if decode[a]['tokens'] != 3 or decode[b]['tokens'] != 3:
                excluded['target_token_shape'] += 1
                continue
            if not decode[a]['reused'] or not decode[b]['reused']:
                excluded['target_rebuild'] += 1
                continue
            if shape != [(DRAFT_NODES, 3), (DRAFT_NODES, 1), (DRAFT_NODES, 1), (VERIFY_NODES, 3)]:
                excluded['draft_sequence'] += 1
                continue
            wall = decode[b]['end_ms']-decode[a]['end_ms']
            graph = sum(r['total'] for r in cycle)
            assert wall >= graph-.01
            cycles.append(dict(
                from_line=decode[a]['line'], to_line=decode[b]['line'], wall_ms=wall,
                graph_ms=graph, outside_graph_ms=wall-graph,
                verify_ms=cycle[-1]['total'], draft_ms=sum(r['total'] for r in cycle[:-1]),
                draft_prepare_ms=sum(r['prepare'] for r in cycle[:-1])))
        assert cycles
        average = {k: mean(c[k] for c in cycles) for k in cycles[0] if k.endswith('_ms')}
        requests.append(dict(
            benchmark=provenance(bench), timing=t, prompt_phase_count=len(prompt),
            prompt_tokens_accounted=prompt_tokens,
            decode_shapes=dict(sorted(Counter(f"{r['tokens']} tokens / {r['nodes']} nodes" for r in decode).items())),
            target_verifications=len(verify), expected_target_verifications=expected_verifications,
            target_phase_ms=totals(verify), draft_phase_ms=totals(draft),
            decode_phase_ms=totals(decode),
            wall_minus_decode_phases_ms=t['predicted_ms']-sum(r['total'] for r in decode),
            steady_cycle_count=len(cycles), excluded_cycle_counts=dict(excluded),
            steady_cycle_mean=average, steady_cycles=cycles))
    cycles = [c for r in requests for c in r['steady_cycles']]
    return dict(mode=mode, log=provenance(log), phase_count=len(rows), requests=requests,
                steady_cycle_count=len(cycles),
                steady_cycle_mean={k: mean(c[k] for c in cycles) for k in cycles[0] if k.endswith('_ms')})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=HERE/'glm-phase-audit.json')
    args = ap.parse_args()
    results = [audit('base', '133740'), audit('fix', '134507')]
    out = dict(scope='Archived September 14 GLM phase runs; not current production timing',
               accounting='Explicit 7152-node target and 89-node draft graphs; prompt token sum checked against each response; every decode phase included',
               wall_convention='Completed target verifications equal predicted_n - 1 - draft_n_accepted; predicted_n includes the first token sampled from prefill',
               steady_convention='Adjacent reused three-token target completions, with one three-token draft catch-up and two one-token draft passes between them',
               caveats=['Instrumented logs may include logging and synchronization overhead.',
                        'Source timing hooks confirm CPU_OP_PROFILE total excludes its separately reported barrier time.',
                        'No projection from these old timings is a measured gain in the current wider-pool runtime.'],
               results=results)
    args.output.write_text(json.dumps(out, indent=2)+'\n')
    for arm in results:
        print(arm['mode'], arm['steady_cycle_count'], json.dumps(arm['steady_cycle_mean']))
    print(args.output)


if __name__ == '__main__':
    main()
