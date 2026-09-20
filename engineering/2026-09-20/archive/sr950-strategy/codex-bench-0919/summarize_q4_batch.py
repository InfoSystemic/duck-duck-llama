#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
B=Path(__file__).resolve().parent
F=Path('/home/user/InfoSystemic/AI-Server/serving/fleet-0912-ctx')
D=F/'glm-q4-batch-cpu-0919';M=F/'glm-q4-batch-0919'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks={}
for suite in ['small','real','dense','down-real','speed','down-speed']:
 r=json.loads((D/('validation-'+suite)/'report.json').read_text())
 assert r['completed'] and r['all_output_bit_equal'] and r['all_activation_counts_expected'],suite
 checks[suite]={'cases_per_variant':len(next(iter(r['variants'].values()))['rows']),'exact_output_parity':True,'engagement_verified':True,'report':str(D/('validation-'+suite)/'report.json')}
perf={}
for suite in ['speed','down-speed']:
 r=json.loads((D/('validation-'+suite)/'report.json').read_text())
 assert r['production_task_ids_before']==r['production_task_ids_after'] and r['production_idle_after']
 rows={name:{x['key']:x for x in data['rows']} for name,data in r['variants'].items()}
 results=[]
 for key,x in rows['off1'].items():
  off=[rows[v][key]['median_ms'] for v in ['off1','off2']];on=[rows[v][key]['median_ms'] for v in ['on1','on2']]
  results.append({'key':key,'type':x['type'],'k':x['k'],'rows':x['rows'],'tokens':x['tokens'],'route_pattern':x['pattern'],'off_ms':off,'on_ms':on,'speedup':sum(off)/sum(on),'control_drift_fraction':off[1]/off[0]-1})
 perf[suite]=results
micro=[json.loads(l) for l in (M/'speed-r1.jsonl').read_text().splitlines()]
summary={'status':'STANDALONE VALIDATED; full model untested; not deployed','production_changed':False,'candidate_sha256':sha(D/'libggml-cpu.so.0.22.0'),'parent_sha256':sha(F/'glm-cpu-fast-0919/libggml-cpu.so.0.22.0'),'direct_kernel_cases':1080,'distinct_graph_cases':sum(checks[s]['cases_per_variant'] for s in ['small','real','dense','down-real']),'validation':checks,'graph_performance':perf,'micro_performance':micro[1:],'micro_scope':'One pinned physical core; 64-row production expert tiles plus exploratory 512-row tiles. Alternating pairs, exact production CPU and base libraries.','graph_timing_scope':'15 pinned physical cores, paired off/on/off/on, counters disabled, 61 repeats per median. Synthetic clamped gate/up plus SwiGLU graphs at gate and down shapes; these are not whole-model tok/s.','conclusion':'Q4 batching improves shared-route cases; mostly divergent routes show small gains near control drift. Worth a scoped full-model test, but no prediction that this reaches 18 tok/s.','full_model_tok_s':None,'cache_issue_fixed':False,'test_expectation_corrections':['Dense counter expectation updated to actual GGML_CPU_X16_CHUNK_MAX=16.','Dense fused-path eligibility requires src1 plane count (weight planes times broadcast) to equal one.','All saved dense numerical outputs were bit-identical before and after correcting counter expectations.'],'preliminary_library_control':'First direct correctness run used the validated-bin base library. The identity guard caught this before timing; correctness was repeated with exact production base and gave the same digest. No timing results from the mismatched base are used.'}
assert summary['distinct_graph_cases']==1304
(B/'q4batch-standalone-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
man=json.loads((D/'manifest.json').read_text());man.update(status=summary['status'],standalone_summary=str(B/'q4batch-standalone-summary.json'),standalone_exact_graph_cases=1304,standalone_exact_kernel_cases=1080,full_model_validated=False,deployed=False)
(D/'manifest.json').write_text(json.dumps(man,indent=2)+'\n')
print(json.dumps({k:summary[k] for k in ['status','candidate_sha256','direct_kernel_cases','distinct_graph_cases','conclusion']},indent=2))
