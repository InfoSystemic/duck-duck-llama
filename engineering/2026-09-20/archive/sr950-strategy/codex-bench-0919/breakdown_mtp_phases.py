#!/usr/bin/env python3
"""Break the already validated MTP A/B cycles into catch-up and two predictions."""
import argparse
from pathlib import Path
import json
from statistics import mean
from audit_glm_phases import parse,VERIFY_NODES,DRAFT_NODES
HERE=Path(__file__).resolve().parent
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--report',type=Path,default=HERE/'mtp-kv-only-r2-window-report.json')
args=ap.parse_args()
report=json.loads(args.report.read_text())
assert report.get('completed') and report.get('restored_healthy')
result={'excluded_from_throughput_results':True,'modes':{}}
for mode,item in report['profiles'].items():
    rows=parse(Path(item['path']),allowed_nodes=(VERIFY_NODES,DRAFT_NODES,15,16) if item['kv_only'] else (VERIFY_NODES,DRAFT_NODES))
    catchup_nodes={15,16} if item['kv_only'] else {DRAFT_NODES}
    selected=[]
    for i,previous in enumerate(rows):
        if previous['nodes']!=VERIFY_NODES or previous['tokens']!=3 or not previous['reused']:continue
        cycle=rows[i+1:i+5]
        if len(cycle)!=4 or cycle[0]['nodes'] not in catchup_nodes or cycle[0]['tokens']!=3:continue
        if [(x['nodes'],x['tokens']) for x in cycle[1:]]!=[(DRAFT_NODES,1),(DRAFT_NODES,1),(VERIFY_NODES,3)]:continue
        if not cycle[-1]['reused']:continue
        selected.append(cycle)
    assert selected
    mode_result={'cycles':len(selected),'passes':{}}
    for j,name in enumerate(['catch_up','first_prediction','second_prediction','target_verification']):
        mode_result['passes'][name]={field+'_ms':mean(c[j][field] for c in selected) for field in ['apply','prepare','inputs','compute','total']}
        mode_result['passes'][name]['reused_fraction']=mean(c[j]['reused'] for c in selected)
    result['modes'][mode]=mode_result
(HERE/'mtp-kv-only-r2-phase-breakdown.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
