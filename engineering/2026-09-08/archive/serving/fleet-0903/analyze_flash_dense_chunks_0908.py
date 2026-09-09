#!/usr/bin/env python3
"""Summarize the Q8 dense component sweep without estimating model bandwidth."""
import json
import math
from pathlib import Path

from qwen_split_trial import sha256

BASE = Path(__file__).resolve().parent
SOURCE = BASE/'results/glm-flash-q8-dense-chunks-0908b/result.json'
OUT = BASE/'results/glm-flash-q8-dense-chunks-analysis-0908.json'


def main():
    data = json.loads(SOURCE.read_text())
    assert data['passed'] and all(x['bit_exact'] for x in data['runs'])
    limits = (16,32,64,128)
    medians = {limit:{key:math.prod(x['graph_ms'][key] for x in data['runs'] if x['chunk']==limit)**.5
                     for key in data['runs'][0]['graph_ms']} for limit in limits}
    rows = []
    for k,n,name in data['shapes']:
        proposed = (((n+29)//30)+15)//16*16
        effective = {str(limit):min(max(proposed,16),limit) for limit in limits}
        for tokens in (1,3):
            key = f'{k}-{n}-{tokens}'
            rows.append(dict(k=k,rows=n,tokens=tokens,tensor_name=name,effective_chunks=effective,
                             graph_ms={str(limit):medians[limit][key] for limit in limits},
                             relative_time_percent={str(limit):100*(medians[limit][key]/medians[16][key]-1)
                                                    for limit in limits[1:]},
                             same_effective_chunk_for_all_limits=len(set(effective.values()))==1))
    result = dict(passed=True,cpu_sha256=data['cpu_sha256'],rows=rows,
                  bit_exact_cases_vs_first_arm=26*7,output_values_per_arm=data['runs'][0]['output_bytes']//4,
                  source_sha256={str(p):sha256(p) for p in (Path(__file__),SOURCE)},
                  notes=['All arms use the unchanged pooling CPU library and two measurements per limit.',
                         'These are isolated graph times, not model timings or IMC measurements.',
                         'Matrix input and weight cache state differ from continuous model decode.',
                         'Physical page placement and phase background load were not measured.',
                         'Timing changes on shapes whose effective chunk is unchanged demonstrate noise; do not attribute them to the limit.',
                         'The effective chunk formula shown applies to the two-dimensional matrices in this probe.'])
    assert not OUT.exists()
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,path=str(OUT),rows=len(rows),exact_cases=182,values_per_arm=result['output_values_per_arm'])))


if __name__ == '__main__':
    main()
