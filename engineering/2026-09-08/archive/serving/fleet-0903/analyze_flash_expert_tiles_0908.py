#!/usr/bin/env python3
"""Summarize the fixed-route and rotating-route expert tile comparisons."""
import json
from pathlib import Path
from statistics import geometric_mean

from qwen_split_trial import sha256

BASE=Path(__file__).resolve().parent


def main():
    warm_path=BASE/'results/glm-flash-q8-expert-tiles-0908/result.json'
    rotating_path=BASE/'results/glm-flash-q8-expert-tiles-rotating-0908/result.json'
    warm=json.loads(warm_path.read_text());rotating=json.loads(rotating_path.read_text())
    assert warm['passed'] and rotating['passed']
    assert len(warm['comparisons'])==90 and len(rotating['comparisons'])==432
    assert all(x['bit_exact'] for x in warm['comparisons']+rotating['comparisons'])
    fixed={str(tile):{} for tile in (64,32,48)}
    for run in warm['runs']:
        if not run['label'].startswith('normal-'):continue
        for row in run['rows']:
            if row['k']=='4096' and row['rows']=='512' and row['fused']=='1':
                fixed[str(run['tile'])].setdefault(row['tokens'],[]).append(float(row['packed_ms']))
    changing={str(tile):[] for tile in (64,32,48)}
    for run in rotating['runs']:
        if run['tile'] is not None:
            changing[str(run['tile'])].append(float(run['rows'][0]['packed_ms']))
    assert all(len(values)==2 for tokens in fixed.values() for values in tokens.values())
    assert all(len(values)==2 for values in changing.values())
    result=dict(passed=True,source_sha256={str(p):sha256(p) for p in (Path(__file__),warm_path,rotating_path)},
        library_sha256=warm['library_sha256'],fixed_route_gate_up_ms=fixed,rotating_route_gate_up_ms=changing,
        fixed_route_time_change_percent={tile:{n:100*(geometric_mean(values)/geometric_mean(fixed['64'][n])-1)
            for n,values in tokens.items()} for tile,tokens in fixed.items() if tile!='64'},
        rotating_route_time_change_percent={tile:100*(geometric_mean(values)/geometric_mean(changing['64'])-1)
            for tile,values in changing.items() if tile!='64'},
        bit_exact_comparisons=522,values_compared=sum(x['values'] for x in warm['comparisons']+rotating['comparisons']),
        notes=['Positive time changes mean slower execution.',
               'The first and last 64-row arms bracket the two 32-row and two 48-row arms.',
               'These are one-socket component timings, not complete-model speed or DRAM utilization.'])
    path=BASE/'results/glm-flash-q8-expert-tiles-analysis-0908.json'
    with path.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
