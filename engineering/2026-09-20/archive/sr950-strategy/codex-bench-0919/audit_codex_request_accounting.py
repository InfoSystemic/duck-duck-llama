#!/usr/bin/env python3
"""Compare each ephemeral Codex thread's own cumulative usage with the surrounding global metrics."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
rows=[]
for path in sorted(HERE.glob('appserver-*.json')):
    r=json.loads(path.read_text())
    if 'server_metrics' not in r or 'token_usage' not in r:continue
    usage=r['token_usage'].get('total')
    if not usage:continue
    m=r['server_metrics']
    if not all(k in m for k in ['prompt_tokens','cached_prompt_tokens','generated_tokens']):continue
    actual={'input':m['prompt_tokens']+m['cached_prompt_tokens'],'cached_input':m['cached_prompt_tokens'],'output':m['generated_tokens']}
    expected={'input':usage['inputTokens'],'cached_input':usage.get('cachedInputTokens',0),'output':usage['outputTokens']}
    differences={k:actual[k]-expected[k] for k in actual if actual[k]!=expected[k]}
    rows.append({'path':str(path),'tag':r.get('tag'),'client':expected,'global':actual,'counter_differences':differences,'matched':not differences})
report={'usage_scope':'Thread total, including multiple model requests in a tool smoke test; each fixture starts a fresh ephemeral thread.','note':'Unmatched global counters cannot supply a per-request throughput result. Matched counts do not independently prove absence of unrelated host load.','requests':rows,'matched':sum(r['matched'] for r in rows),'mismatched':sum(not r['matched'] for r in rows)}
(HERE/'codex-request-accounting-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['matched','mismatched']}))
for row in rows:
    if not row['matched']:print(row['tag'],row['counter_differences'])
