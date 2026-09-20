#!/usr/bin/env python3
"""Summarize all candidate and production controls without discarding slow runs."""
import hashlib,json,statistics
from pathlib import Path
H=Path(__file__).resolve().parent

def native_score(groups):
 rows=[r for g in groups for r in g['rows']]
 return sum(r['timings']['predicted_n']-1 for r in rows)/sum(r['timings']['predicted_ms']/1000 for r in rows)
def codex_row(label,x):
 m=x['server_metrics']
 return {'label':label,'tag':x['tag'],'decode_tps':m['decode_tokens_per_second'],
  'first_visible_seconds':x['first_visible_delta_seconds'],'wall_seconds':x['wall_seconds'],
  'workload':{k:m.get(k) for k in ['generated_tokens','cached_prompt_tokens','processed_prompt_tokens','draft_tokens','accepted_draft_tokens','draft_verification_steps']},
  'marker_pass':x['marker_pass'],'output_sha256':hashlib.sha256(x['output_text'].encode()).hexdigest()}
def main():
 source=H/'q4batch-window-report.json';d=json.loads(source.read_text())
 assert d.get('experiment_completed') and d.get('restored_healthy') and 'codex_after' in d and not d.get('error')
 before,after=d['native_before'],d['native_after'];candidate=d['native_candidate']
 rows=[codex_row('production cold before',d['codex_before']['cold']),codex_row('production warm before',d['codex_before']['warm']),
  codex_row('candidate cold',d['codex_candidate_cold'])]
 rows += [codex_row('candidate warm '+str(i+1),x) for i,x in enumerate(d['codex_candidate'])]
 rows += [codex_row('production cold after',d['codex_after']['cold']),codex_row('production warm after',d['codex_after']['warm'])]
 control_warm=statistics.mean(x['server_metrics']['decode_tokens_per_second'] for x in [d['codex_before']['warm'],d['codex_after']['warm']])
 candidate_warm=statistics.mean(x['server_metrics']['decode_tokens_per_second'] for x in d['codex_candidate'])
 control_cold=statistics.mean(x['server_metrics']['decode_tokens_per_second'] for x in [d['codex_before']['cold'],d['codex_after']['cold']])
 report={'source':str(source),'candidate_sha256':d['candidate_sha256'],'production_restored_pid':d['restored_pid'],'outage_seconds':d['outage_seconds'],
  'promoted':d['promoted'],'all_native_outputs_match':all(g['all_parity'] for g in [before,*candidate,after]),
  'native_tps':{'before':before['aggregate_tps'],'candidate_groups':[g['aggregate_tps'] for g in candidate],'candidate_all_runs':native_score(candidate),
   'after':after['aggregate_tps'],'controls_combined':native_score([before,after]),'control_after_before_ratio':after['aggregate_tps']/before['aggregate_tps']},
  'codex_rows':rows,'codex_warm_control_mean':control_warm,'codex_warm_candidate_mean':candidate_warm,
  'codex_warm_gain_fraction':candidate_warm/control_warm-1,
  'codex_cold_gain_fraction':d['codex_candidate_cold']['server_metrics']['decode_tokens_per_second']/control_cold-1,
  'codex_control_warm_after_before_ratio':d['codex_after']['warm']['server_metrics']['decode_tokens_per_second']/d['codex_before']['warm']['server_metrics']['decode_tokens_per_second'],
  'stateful_regression_passed':d['stateful_regression_passed'],'stateful_intrinsic_consistency':d['stateful_intrinsic_consistency'],
  'full_model_graph_structure':d['full_model_graph_structure'],
  'interpretation_limits':['Small benchmark sample; before/after controls measure drift, not a confidence interval.',
   'All candidate native groups are included; no favorable-run selection.',
   'Cached and fresh outputs are compared to matching production scenarios; the known cache inconsistency is unresolved.',
   'No production promotion is implied by these measurements.']}
 output=H/'q4batch-window-summary.json';output.write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:report[k] for k in ['native_tps','codex_warm_control_mean','codex_warm_candidate_mean','codex_warm_gain_fraction','codex_cold_gain_fraction','codex_control_warm_after_before_ratio','production_restored_pid','outage_seconds']},indent=2))
if __name__=='__main__':main()
