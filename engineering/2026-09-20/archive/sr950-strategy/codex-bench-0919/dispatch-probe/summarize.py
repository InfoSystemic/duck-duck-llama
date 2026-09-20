#!/usr/bin/env python3
"""Summarize standalone results without extrapolating synthetic speedups to GLM."""
from pathlib import Path
import collections,hashlib,json,statistics
P=Path(__file__).resolve().parent
cases=[];failed=[];fingerprints=collections.defaultdict(set);byte_groups=collections.defaultdict(list)
for path in sorted(P.glob('*.json')):
 d=json.loads(path.read_text())
 if not isinstance(d,dict) or not d.get('command') or d['command'][0]!=str(P/'probe'):continue
 if not d.get('completed'):
  failed.append({'tag':d['tag'],'returncode':d.get('returncode'),'error':d.get('error')});continue
 row=d['rows'][-1]
 assert row['completed'] and d['production_task_ids_before']==d['production_task_ids_after']
 fingerprints[d['size']].add((row['target_hash'],row['draft_hash']))
 entry={k:d.get(k) for k in ['tag','workers','spin_us','size','gomp_spin','omp_wait','simple_barrier','loader_team','peak_threads','peak_rss_kib','peak_vm_kib','source_sha256','binary_sha256','production_task_ids_before','production_task_ids_after']}
 entry.update({k:v for k,v in row.items() if k!='samples_ms'})
 if 'output_sha256' in d:
  entry['saved_output_sha256']=d['output_sha256'];byte_groups[d['size']].append(d['tag'])
 cases.append(entry)
assert all(len(s)==1 for s in fingerprints.values()),'Output fingerprint changed'
for size,tags in byte_groups.items():
 for kind in ['target','draft']:
  baseline=(P/(tags[0]+'.'+kind+'.bin')).read_bytes()
  assert all((P/(t+'.'+kind+'.bin')).read_bytes()==baseline for t in tags)
by_tag={r['tag']:r for r in cases}
def compare(off,on):
 a=[by_tag[t] for t in off];b=[by_tag[t] for t in on]
 aw=sum(r['wall_seconds'] for r in a);bw=sum(r['wall_seconds'] for r in b)
 ar=sum(r['rounds'] for r in a);br=sum(r['rounds'] for r in b)
 return {'control_tags':off,'candidate_tags':on,'control_mean_of_run_medians_ms':statistics.mean(r['median_cycle_ms'] for r in a),'candidate_mean_of_run_medians_ms':statistics.mean(r['median_cycle_ms'] for r in b),'median_speed_ratio':statistics.mean(r['median_cycle_ms'] for r in a)/statistics.mean(r['median_cycle_ms'] for r in b),'aggregate_cycle_speed_ratio':(aw/ar)/(bw/br),'control_system_cpu_per_wall_second':sum(r['system_seconds'] for r in a)/aw,'candidate_system_cpu_per_wall_second':sum(r['system_seconds'] for r in b)/bw}
loaded=compare([f'exact-long-15-barrier0-gompNone-waitNone-{s}' for s in 'ab'],[f'exact-long-15-barrier1-gompNone-waitNone-{s}' for s in 'ab'])
unloaded=compare([f'exact-long-15-barrier0-unloaded-{s}' for s in 'ab'],[f'exact-long-15-barrier1-unloaded-{s}' for s in 'ab'])
report={'completed':True,'production_changed':False,'production_outage':False,'promote_any_setting':False,'all_successful_fingerprints_match_by_shape':True,'all_saved_output_bytes_equal_by_shape':True,'successful_cases':len(cases),'failed_cases':failed,'saved_byte_cases':sum(len(t) for t in byte_groups.values()),'saved_byte_shapes':list(byte_groups),'saved_byte_case_measured_cycles':sum(by_tag[t]['rounds'] for ts in byte_groups.values() for t in ts),'barrier_with_artificial_openmp_loader':loaded,'barrier_without_artificial_openmp_loader':unloaded,'interpretation':['The 8 GiB virtual-memory failure was an allocation failure, fixed only in the probe runner with a 24 GiB address-space limit and a 6 GiB sampled resident-memory guard.','An extra retained 128-thread OpenMP team produced a synthetic barrier speedup, but the same improvement was absent without that artificial team.','Server source defaults to 127 HTTP workers on this host; 128 unbound threads were created within 0.09 seconds of production startup. This supports HTTP workers plus listener as the idle group, not a retained OpenMP team; user-space call stacks were not sampled.','Current other workloads were observed on physical cores 15,31,47,63 and their SMT siblings. Both loaded and no-loader 16-worker cases slowed severely during this activity. This is a current confound, not proof of the cause of the earlier full-model sweep.','No new full-model outage is warranted by these measurements. No standalone result establishes a production speedup or a route to 18 tokens/s.'],'cases':cases}
(P/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['cases','interpretation']},indent=2))
