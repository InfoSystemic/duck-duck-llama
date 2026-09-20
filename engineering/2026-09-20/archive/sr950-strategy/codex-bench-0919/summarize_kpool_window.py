#!/usr/bin/env python3
"""Separate actual decode rates from profiles, context caching, and draft acceptance."""
import json
from summarize_optrace import HEADER,ROW
from pathlib import Path
D=Path(__file__).resolve().parent
report=json.loads((D/'kpool-window-report.json').read_text())
result={'completed':report.get('completed',False),'restored_healthy':report.get('restored_healthy',False),
        'candidate_sha256':report['candidate_sha256'],'rows':[],
        'known_cache_consistency_issue_unresolved':True}
keys=['generated_tokens','draft_tokens','accepted_draft_tokens','draft_verification_steps','prompt_tokens','cached_prompt_tokens']
for label in ['short','long']:
    cold=[report.get('arms',{}).get(f'{label}-cold-{m}') for m in [0,1]]
    warm=[report.get('arms',{}).get(f'{label}-warm-{i}') for i in range(4)]
    if all(cold):
        a,b=[x['server_metrics'] for x in cold]
        comparable=all(a[k]==b[k] for k in keys)
        result['rows'].append({'context':label,'cache':'cold','scalar_tps':a['decode_tokens_per_second'],
            'wide_tps':b['decode_tokens_per_second'],'speedup':b['decode_tokens_per_second']/a['decode_tokens_per_second'],
            'same_generated_text':cold[0]['output_text']==cold[1]['output_text'],
            'same_work_counters':comparable,'counter_values':{k:[a[k],b[k]] for k in keys},
            'scalar_prefill_seconds':a['prompt_seconds'],'wide_prefill_seconds':b['prompt_seconds']})
    if all(warm):
        arms={mode:[x['server_metrics'] for x in warm if x['wide']==bool(mode)] for mode in [0,1]}
        speed={m:sum(x['generated_tokens'] for x in rows)/sum(x['decode_seconds'] for x in rows) for m,rows in arms.items()}
        reference=warm[0]['server_metrics']
        result['rows'].append({'context':label,'cache':'warm ABBA','scalar_tps':speed[0],'wide_tps':speed[1],
            'speedup':speed[1]/speed[0],'same_generated_text':all(x['output_text']==warm[0]['output_text'] for x in warm),
            'same_work_counters':all(x['server_metrics'][k]==reference[k] for x in warm for k in keys),
            'counter_values':{k:[x['server_metrics'][k] for x in warm] for k in keys},
            'order_tps':[x['server_metrics']['decode_tokens_per_second'] for x in warm],
            'scalar_end_to_start':arms[0][1]['decode_tokens_per_second']/arms[0][0]['decode_tokens_per_second'],
            'warm_equals_cold':[x['cold_output_equal'] for x in warm]})
result['production_controls']={side:report['production_codex_'+side]['server_metrics'] for side in ['before','after'] if 'production_codex_'+side in report}
result['native']={k:{'aggregate_tps':v['aggregate_tps'],'all_parity':v['all_parity'],
    'counts':[{'prompt':r['prompt'], **{field:r['timings'].get(field) for field in ['predicted_n','draft_n','draft_n_accepted']}} for r in v['rows']]}
    for k,v in report.get('arms',{}).items() if k.startswith('native-')}
result['profiles']={}
for arm,row in report.get('profiles',{}).items():
    prof=json.loads(Path(row['summary']).read_text())
    totals={};current={};shapes={}
    log=Path(row['summary'].replace('.optrace-summary.json','.optrace.log'))
    for line in log.read_text().splitlines():
        if match:=HEADER.search(line):
            index,cpu,ptr,_,_,_=match.groups();current[(cpu,ptr)]=int(index);totals[int(index)]=0.0;shapes[int(index)]=set()
        elif match:=ROW.search(line):
            cpu,ptr,_,op,ms,_,name,typ,shape,src=match.groups()
            if name.startswith('indexer_pool_members') and (cpu,ptr) in current:
                totals[current[(cpu,ptr)]]+=float(ms)
            if (cpu,ptr) in current and shape and (name.startswith('indexer_pool_') or (src or '').startswith('indexer_pool_')):
                shapes[current[(cpu,ptr)]].add((op,typ,shape))
    result['profiles'][arm]=[
        {'index':g['index'],'nodes':g['declared_nodes'],'total_ms':g['total_ms'],'barrier_ms':g['reported_barrier_ms'],
         'pool_ms':totals[g['index']], 'pool_shape_evidence':sorted(shapes[g['index']]), 'operations_ms':g['operation_ms']}
        for g in prof['graphs'] if g['declared_nodes']>1000]
if len(result['profiles'])==2:
    a={tuple(tuple(x) for x in g['pool_shape_evidence']) for g in result['profiles']['0']}
    b={tuple(tuple(x) for x in g['pool_shape_evidence']) for g in result['profiles']['1']}
    result['long_profile_pool_shapes_match']=(a==b)
if 'error' in report:result['error']=report['error']
(D/'kpool-window-summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
