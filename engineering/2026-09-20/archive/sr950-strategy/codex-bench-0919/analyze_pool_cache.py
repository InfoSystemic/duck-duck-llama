#!/usr/bin/env python3
"""Audit and summarize completed actual-Codex pooled-result cache comparisons."""
import argparse,hashlib,json,re
from pathlib import Path
HERE=Path(__file__).resolve().parent
def aggregate(rows):return sum(x['server_metrics']['generated_tokens'] for x in rows)/sum(x['server_metrics']['decode_seconds'] for x in rows)
def accounted(x):
    m=x['server_metrics'];u=x['token_usage']['total'];return m['prompt_tokens']+m['cached_prompt_tokens']==u['inputTokens'] and m['cached_prompt_tokens']==u['cachedInputTokens'] and m['generated_tokens']==u['outputTokens']
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--context',choices=['short','long'],required=True);a=ap.parse_args();prefix='pool-cache-r2-'+a.context
    path=HERE/(prefix+'-window-report.json');r=json.loads(path.read_text());assert r['completed'] and r['restored_healthy'] and 'production_codex_after' in r and 'error' not in r
    rows=[r['arms']['warm-'+str(i)] for i in range(8)];keys=['prompt_tokens','cached_prompt_tokens','generated_tokens','draft_tokens','accepted_draft_tokens','draft_verification_steps']
    work=lambda row:{k:row['server_metrics'][k] for k in keys}
    assert all(x['output_text']==rows[0]['output_text'] and work(x)==work(rows[0]) for x in rows)
    assert [int(x['cache']) for x in rows]==[0,1,1,0,1,0,0,1]
    accounting={k:accounted(v) for k,v in r['arms'].items() if 'server_metrics' in v}
    accounting.update({k:accounted(r[k]) for k in ['production_codex_before','production_codex_after']})
    accounting.update({'profile-'+k:accounted(v['benchmark']) for k,v in r['profiles'].items()});assert all(accounting.values())
    off=aggregate([x for x in rows if not x['cache']]);on=aggregate([x for x in rows if x['cache']])
    out={'source':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'candidate_sha256':r['candidate_sha256'],'completed':True,'restored_healthy':True,'restored_pid':r['restored_pid'],'promoted':False,'context':a.context,'window_seconds':r['window_seconds'],'matched_work':work(rows[0]),'off_tps':off,'on_tps':on,'gain_percent':100*(on/off-1),'request_accounting':accounting,'rows':[{'order':x['order'],'cache':x['cache'],'tps':x['server_metrics']['decode_tokens_per_second'],'decode_seconds':x['server_metrics']['decode_seconds'],'cold_output_equal':x['cold_output_equal']} for x in rows],'blocks':[],'cold':{k:v['server_metrics'] for k,v in r['arms'].items() if k.startswith('cold-')},'production_controls':{k:r[k]['server_metrics'] for k in ['production_codex_before','production_codex_after']},'profiles':{},'stateful_regression_passed':r.get('stateful_regression_passed'),'stateful_intrinsic_consistency':r.get('stateful_intrinsic_consistency'),'caveats':['The disabled candidate retains the original arithmetic but has different compiler layout from the production library.','Balanced same-process measurements are primary; separate production loads can drift.','Diagnostic counters and operation profiling are disabled for primary measurements; profiles are excluded from rates.','Existing cached/fresh output discrepancy remains unresolved.','A short fixture does not establish a workload-wide throughput floor.']}
    for i in [0,4]:
        block=rows[i:i+4];b_off=aggregate([x for x in block if not x['cache']]);b_on=aggregate([x for x in block if x['cache']]);out['blocks'].append({'first_order':i,'off_tps':b_off,'on_tps':b_on,'gain_percent':100*(b_on/b_off-1)})
    for mode,item in r['profiles'].items():
        summary=json.loads(Path(item['summary']).read_text());targets=[x for x in summary['graphs'] if x['declared_nodes']>1000];assert targets
        trace=Path(item['trace']).read_text();observed=[]
        for g in targets:
            pool_lines=[x for x in trace.splitlines() if 'graph='+g['graph']+' ' in x and "name='indexer_pool_members-" in x and 'op=GET_ROWS ' in x]
            assert len(pool_lines)==11,('Missing pooling attribution',mode,g['graph'],len(pool_lines))
            ms=sum(float(re.search(r'time=([0-9.eE+-]+) ms',x).group(1)) for x in pool_lines)
            observed.append({'cpu':g['cpu'],'nodes':g['declared_nodes'],'operation_ms':g['total_ms'],'barrier_ms':g['reported_barrier_ms'],'pool_ms':ms,'pool_layers':len(pool_lines)})
        out['profiles'][mode]={'source':item['summary'],'trace':item['trace'],'excluded_from_throughput':True,'target_graphs':observed,'cache_hits_logged':'GLM_POOL_CACHE_HIT' in trace}
    assert out['profiles']['1']['cache_hits_logged']
    result=HERE/(prefix+'-window-summary.json');result.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:out[k] for k in ['off_tps','on_tps','gain_percent','blocks','restored_pid','profiles']},indent=2))
if __name__=='__main__':main()
