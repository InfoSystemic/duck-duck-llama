#!/usr/bin/env python3
"""Summarize the completed KV rollback window and its separate phase traces."""
from collections import Counter
import json
from pathlib import Path
from statistics import mean
from audit_glm_phases import parse, VERIFY_NODES, DRAFT_NODES

HERE=Path(__file__).resolve().parent


def aggregate(rows):
    return sum(r['server_metrics']['generated_tokens'] for r in rows)/sum(r['server_metrics']['decode_seconds'] for r in rows)


def profile(item):
    rows=parse(Path(item['path']))
    m=item['benchmark']['server_metrics']
    prompt=0
    boundary=None
    for i,row in enumerate(rows):
        if row['nodes']!=VERIFY_NODES:continue
        prompt+=row['tokens']
        assert prompt<=m['prompt_tokens']
        assert rows[i+1]['nodes']==DRAFT_NODES and rows[i+1]['tokens']==row['tokens']
        if prompt==m['prompt_tokens']:
            boundary=i+2;break
    assert boundary is not None
    decode=rows[boundary:]
    verify_indices=[i for i,r in enumerate(decode) if r['nodes']==VERIFY_NODES]
    assert len(verify_indices)==m['draft_verification_steps'],(len(verify_indices),m['draft_verification_steps'])
    cycles=[]
    for a,b in zip(verify_indices,verify_indices[1:]):
        cycle=decode[a+1:b+1]
        if not all(decode[i]['tokens']==3 and decode[i]['reused'] for i in [a,b]):continue
        if [(r['nodes'],r['tokens']) for r in cycle]!=[(DRAFT_NODES,3),(DRAFT_NODES,1),(DRAFT_NODES,1),(VERIFY_NODES,3)]:continue
        wall=decode[b]['end_ms']-decode[a]['end_ms']
        graph=sum(r['total'] for r in cycle)
        cycles.append(dict(from_line=decode[a]['line'],to_line=decode[b]['line'],wall_ms=wall,graph_ms=graph,
            outside_graph_ms=wall-graph,verify_ms=cycle[-1]['total'],draft_ms=sum(r['total'] for r in cycle[:-1]),
            verify_prepare_ms=cycle[-1]['prepare'],draft_prepare_ms=sum(r['prepare'] for r in cycle[:-1])))
    assert cycles
    return dict(source=item['path'],graph_count=len(rows),prompt_tokens_accounted=prompt,
        decode_shapes=dict(Counter(f"{r['tokens']} tokens / {r['nodes']} nodes" for r in decode)),
        target_verifications=len(verify_indices),reported_verifications=m['draft_verification_steps'],
        graph_decode_ms=sum(r['total'] for r in decode),reported_decode_ms=m['decode_seconds']*1000,
        steady_cycle_count=len(cycles),steady_mean={k:mean(c[k] for c in cycles) for k in cycles[0] if k.endswith('_ms')},
        cycles=cycles)


def main():
    r=json.loads((HERE/'kv-range-r2-window-report.json').read_text())
    assert r['completed'] and r['restored_healthy'] and 'production_codex_after' in r
    warm=[r['arms']['warm-'+str(i)] for i in range(8)]
    work_keys=['prompt_tokens','cached_prompt_tokens','generated_tokens','draft_tokens','accepted_draft_tokens','draft_verification_steps']
    work=lambda x:{k:x['server_metrics'][k] for k in work_keys}
    assert all(x['output_text']==warm[0]['output_text'] and work(x)==work(warm[0]) for x in warm)
    arms={str(value):[x for x in warm if x['bounded']==bool(value)] for value in [0,1]}
    off,on=aggregate(arms['0']),aggregate(arms['1'])
    out=dict(candidate_sha256=r['candidate_sha256'],restored_pid=r['restored_pid'],promoted=False,completed=r['completed'],restored_healthy=r['restored_healthy'],
        window_seconds=r['window_seconds'],stateful_regression_passed=r['stateful_regression_passed'],
        intrinsic_cache_consistency=r['stateful_intrinsic_consistency'],
        matched_work=work(warm[0]),warm_original_loop_tps=off,warm_bounded_tps=on,warm_gain_percent=100*(on/off-1),
        warm_rows=[dict(order=x['order'],bounded=x['bounded'],tps=x['server_metrics']['decode_tokens_per_second']) for x in warm],
        blocks=[],cold={str(v):r['arms']['cold-'+str(v)]['server_metrics'] for v in [0,1]},
        native={str(v):r['arms']['native-'+str(v)] for v in [0,1]},
        production_controls={k:r[k]['server_metrics'] for k in ['production_codex_before','production_codex_after']},
        profiles={},caveats=['Flag-off retains original source but compiler optimization differs from the original library.',
                            'Production controls bracket separate process loads; same-process repeated cached runs are the primary bound comparison.',
                            'Phase profiles are instrumented and excluded from throughput estimates.',
                            'Pre-existing cached/fresh output inconsistency is not fixed.'])
    for begin in [0,4]:
        block=warm[begin:begin+4]
        a=aggregate([x for x in block if not x['bounded']]);b=aggregate([x for x in block if x['bounded']])
        out['blocks'].append(dict(first_order=begin,original_loop_tps=a,bounded_tps=b,gain_percent=100*(b/a-1)))
    for key,value in r['profiles'].items():
        try:out['profiles'][key]=profile(value)
        except Exception as exc:out['profiles'][key]={'source':value['path'],'analysis_error':repr(exc)}
    (HERE/'kv-range-r2-window-summary.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:out[k] for k in ['warm_original_loop_tps','warm_bounded_tps','warm_gain_percent','blocks','restored_pid']},indent=2))
    for key,value in out['profiles'].items():print('profile',key,value.get('steady_mean',value.get('analysis_error')))


if __name__=='__main__':main()
