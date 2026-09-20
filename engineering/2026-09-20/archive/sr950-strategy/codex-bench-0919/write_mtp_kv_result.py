#!/usr/bin/env python3
from pathlib import Path
import json
HERE=Path(__file__).resolve().parent
ROOT=Path('/home/user/InfoSystemic/AI-Server/serving/fleet-0912-ctx/glm-mtp-kv-only-0919')
s=json.loads((HERE/'mtp-kv-only-r2-window-summary.json').read_text())
r=json.loads((HERE/'mtp-kv-only-r2-validation-report.json').read_text())
p=HERE/'mtp-kv-only-promotion-report.json'
promotion=json.loads(p.read_text()) if p.exists() else {}
texts=[r['production_codex_before']['output_text'],r['production_codex_after']['output_text']]
texts += [v['output_text'] for k,v in r['arms'].items() if k.startswith(('cold','warm'))]
texts += [v['benchmark']['output_text'] for v in r['profiles'].values()]
assert len(texts)==14 and len(set(texts))==1
assert all(s['request_accounting'].values())
work=['prompt_tokens','cached_prompt_tokens','generated_tokens','draft_tokens','accepted_draft_tokens','draft_verification_steps']
assert all(s['cold']['0'][k]==s['cold']['1'][k] for k in work)
a=s['profiles']['0']['steady_mean'];b=s['profiles']['1']['steady_mean']
cold0=s['cold']['0']['decode_tokens_per_second'];cold1=s['cold']['1']['decode_tokens_per_second']
lead=f"Deployed and verified on the normal endpoint as PID {promotion['production_pid']}." if promotion.get('promoted') else 'Full-model validation passed; the separate production deployment is in progress.'
lines=[
'# GLM MTP cache-only catch-up result','',lead,'',
'The new path produces another small, repeatable gain. The measured short Codex fixture remains below 17 tok/s; no consistent 17+ or 18+ result is established.','',
'| Actual Codex input: 3,998 tokens | Full catch-up graph | Cache-only catch-up | Gain |',
'|---|---:|---:|---:|',
f"| Cold, matched work | {cold0:.2f} tok/s | {cold1:.2f} tok/s | {100*(cold1/cold0-1):.2f}% |",
f"| Cached, eight balanced repeats | {s['warm_full_graph_tps']:.2f} tok/s | {s['warm_cache_only_tps']:.2f} tok/s | {s['warm_gain_percent']:.2f}% |",'',
'The primary comparison uses ABBA then BAAB order in one loaded model on private port 18141. Both blocks improved: '+', '.join(f"{x['gain_percent']:.2f}%" for x in s['blocks'])+'. All eight outputs and all input/cache, generated-token, draft-token, accepted-draft and verification counts match. Each cached request uses four new plus 3,994 cached input tokens, emits 270 tokens, and records 210 drafted tokens, 165 accepted drafts and 105 verification steps. The installed Paseo Codex wrapper/app-server path runs at low reasoning, temperature zero and seed 42. Rates exclude prefill and client work.','',
f"Original-production cold controls were {r['production_codex_before']['server_metrics']['decode_tokens_per_second']:.2f} and {r['production_codex_after']['server_metrics']['decode_tokens_per_second']:.2f} tok/s. These separate process loads have approximately 3.2% control drift and a one-accepted-draft difference; they are not the primary comparison. The candidate's disabled branch retains the original graph but differs in compiler layout and adds a mode guard. The balanced in-process result is the measured effect of selecting the fast path in the final library.",'',
'## Implementation and exact gates','',
'The MTP catch-up batch requests no output rows. The guarded fast path keeps the token/hidden normalization, combined projection, attention-input normalization, KV projection and normalization, and identical cache writes. It omits the unused query/attention path. It applies only to a single MTP layer with zero outputs, masked next-token embeddings, no ordinary embeddings, no backend samplers and no requested intermediate layer outputs. Other cases keep the full graph. No retained floating-point arithmetic changes.','',
'Only models/glm5next.cpp.o changes over the byte-identical current production parent. Candidate libllama SHA256: `'+s['candidate_sha256']+'`. Existing wider-pool and bounded-rollback improvements remain enabled.','',
'- Standalone production/off/on/alternating-switch gates at 1 and 15 workers per socket match all 170 records and 28,400,396 bytes per arm, including actual cache tensors, sequence metadata, logits and hidden rows.',
'- All six native A/B outputs and all 14 Codex outputs, including instrumented and bracketing controls, match their references.',
'- All nine stateful output comparisons match production. The pre-existing cached/fresh consistency issue remains unresolved.',
'- Every primary Codex request matches its own cumulative token usage. The separate first shared-port window has a contaminated control and is excluded.','',
'## Timing attribution','',
f"Both saved traces contain 422 graphs and reconcile all 105 target verifications, with 101 steady cycles each. Draft work falls from {a['draft_ms']:.3f} to {b['draft_ms']:.3f} ms/cycle. The catch-up pass itself falls from 7.398 to 4.411 ms. Enabled draft preparation still costs {b['draft_prepare_ms']:.3f} ms; the catch-up and first prediction rebuild, while the second prediction reuses its graph.",'',
f"Target time in these separate instrumented traces is {a['verify_ms']:.3f} versus {b['verify_ms']:.3f} ms; outside-graph time is {a['outside_graph_ms']:.3f} versus {b['outside_graph_ms']:.3f} ms. These profiles confirm the mechanism and are excluded from throughput estimates. No new long-context throughput result is claimed.",'',
'## Restoration and harness repair','',
f"The private test window lasted {s['window_seconds']/60:.2f} minutes and restored the exact original libraries, inference environment and command before the separate deployment. All measurements and both traces finished, but the controller's imported parser rejected the expected new 15-node graph because its allowed set contained only 7152/89. The raw failed controller report is preserved unchanged. The parser now accepts an explicit mode-specific node set, while its historical default stays strict. Saved traces were validated offline, and a separate final production control completed without another benchmark outage. See the derived validation report and its source hashes; this is not recorded as a clean controller execution.",'',
'An earlier shared-port attempt stopped before enabling the candidate because an unrelated request was active. It restored production and is not included in the A/B. A standalone reference-only separate-stream sequence-copy crash is preserved in the candidate directory; normal production uses unified KV and the final exact gate avoids that unrelated failing operation. Neither existing issue is claimed fixed.','',
'The production drop-in sets `GGML_GLM5N_MTP_KV_ONLY=1` and selects the immutable library. The benchmark control and diagnostic probe are absent. The deployment controller rolls back only that new drop-in if runtime identity or output checks fail.','']
if promotion.get('promoted'):
 c=promotion['mtp-kv-only-promoted-cold']['server_metrics']['decode_tokens_per_second']
 w=promotion['mtp-kv-only-promoted-warm']['server_metrics']['decode_tokens_per_second']
 lines += [f"Final native output parity passed. Post-deployment Codex cold/cached smoke rates are {c:.2f} / {w:.2f} tok/s, with exact reference text and clean per-request accounting. These are smoke checks, not a workload-wide throughput floor. Production PID {promotion['production_pid']} is healthy; profiling is disarmed.",'']
lines += ['Evidence: [A/B summary](mtp-kv-only-r2-window-summary.json), [derived validated report](mtp-kv-only-r2-validation-report.json), [preserved raw controller report](mtp-kv-only-r2-window-report.json), [phase breakdown](mtp-kv-only-r2-phase-breakdown.json), [deployment report](mtp-kv-only-promotion-report.json), [source and standalone gates]('+str(ROOT/'README.md')+').','']
(HERE/'MTP-KV-RESULT.md').write_text('\n'.join(lines))
print(HERE/'MTP-KV-RESULT.md')
