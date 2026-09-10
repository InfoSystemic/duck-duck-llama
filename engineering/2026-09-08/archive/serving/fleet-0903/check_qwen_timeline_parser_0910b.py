#!/usr/bin/env python3
"""Check overlap accounting and reject malformed diagnostic timestamps."""
import json
from pathlib import Path

from analyze_qwen_timeline_0910b import analyze
from qwen_split_trial import sha256


def fixture():
    lines=[]
    for index in range(8):
        rank=index%4;cpu=rank*16;pointer='0x'+str(index+10)
        target=index<4;start=(10 if target else 200)+rank;end=start+(100 if target else 20)
        node_start=start+1;node_end=end-1;work_end=node_start+1
        first='hc_init' if target else 'norm-48'
        lines.append(f'CPU_OP_TIMELINE index={index} cpu={cpu} graph={pointer} threads=15 start_us={start} end_us={end}')
        lines.append(f"CPU_OP_PROFILE index={index} cpu={cpu} graph={pointer} nodes=1 total={(node_end-node_start)/1000:.3f} ms first='{first}' last='result_output'")
        lines.append(f"CPU_OP_PROFILE cpu={cpu} graph={pointer} node=0 op=MUL_MAT time={(node_end-node_start)/1000:.3f} ms name='result_output' src0_type=Q8_0 src0_ne=[2560,248320,1] src0_name='output.weight' start_us={node_start} work_end_us={work_end} end_us={node_end} dst_ne=[248320,1,1,1]")
    lines.append('CPU_OP_PROFILE complete count=8')
    return '\n'.join(lines)


if __name__=='__main__':
    base=Path(__file__).resolve().parent
    destination=base/'results/qwen-timeline-parser-checks-0910b.json'
    assert not destination.exists()
    text=fixture();parsed=analyze(text,8)
    assert parsed['all_graph_interval_union_us']==126
    assert [(r['summed_socket_us'],r['interval_union_us']) for r in parsed['summaries'][:2]]==[(400,103),(80,23)]
    assert len(parsed['complete_four_socket_waves'])==2
    checks=[dict(case='concurrent sockets counted once',passed=True)]
    for name,bad in [
        ('missing graph timestamp',text.replace('CPU_OP_TIMELINE index=0','MISSING index=0',1)),
        ('reversed work timestamp',text.replace('work_end_us=12','work_end_us=1',1)),
        ('missing node timestamp',text.replace('start_us=11 work_end_us=12','missing_start=11 work_end_us=12',1)),
        ('missing completed graph',text.replace('CPU_OP_PROFILE index=7','MISSING index=7',1))]:
        try:analyze(bad,8)
        except (AssertionError,KeyError):checks.append(dict(case=name,passed=True))
        else:raise AssertionError(name)
    result=dict(passed=True,checks=checks,sources={str(p):sha256(p) for p in [Path(__file__),base/'analyze_qwen_timeline_0910b.py']},
        scope='Synthetic parser fixtures only. Proves interval-union arithmetic and malformed-capture rejection, not model speed.')
    destination.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
