#!/usr/bin/env python3
import json
from pathlib import Path
D=Path(__file__).resolve().parent
s=json.loads((D/'kpool-window-summary.json').read_text())
r=json.loads((D/'kpool-window-report.json').read_text())
p=D/'kpool-promotion-report.json'
promotion=json.loads(p.read_text()) if p.exists() else None
if promotion and promotion.get('promoted'):
    status=f"Deployed and verified on the normal endpoint, production PID {promotion['production_pid']}."
elif json.loads((D/'kpool-promotion-manifest.json').read_text()).get('state')=='DEPLOYMENT IN PROGRESS':
    status='Production deployment is in progress. The completed A/B first restored and verified the original runtime.'
elif r.get('restored_healthy'):
    status=f"A/B complete; original production restored and healthy, PID {r['restored_pid']}. Wider kernel staged separately."
else:
    status='A/B complete; restoration of the original production runtime is in progress.'
lines=['# GLM wider kpool kernel result', '',status,'',
       'The new inner loop improves measured throughput, but does not reach 18+ tok/s on the standard Codex/Paseo benchmark.', '',
       '| Actual Codex input | Scalar control | Wider loop | Gain |',
       '|---|---:|---:|---:|']
for label,n in [('short',3998),('long',29930)]:
    row=next(x for x in s['rows'] if x['context']==label and x['cache']=='warm ABBA')
    lines.append(f"| {n:,} tokens | {row['scalar_tps']:.2f} tok/s | {row['wide_tps']:.2f} tok/s | {100*(row['speedup']-1):.1f}% |")
lines += ['', 'Primary results use two scalar and two wider-kernel requests in ABBA order. Each comparison has identical generated text, input/cache counts, generated-token counts, draft-token counts, accepted drafts, and verification steps. Decode rates exclude prefill and client work. Both modes run in the same loaded model on :18131 with a switch changed only while idle.', '',
          'The separate original-service controls measured 14.46 tok/s before the window and 14.78 afterward, with one accepted-draft difference. Within the same loaded model, short ABBA controls drifted about 1.0%; long controls drifted about 0.1%. The long fixture is deterministic synthetic filler plus the standard LRU explanation task, not a real [client] conversation. Its user prompt has 26,000 tokens; actual Codex input is 29,930 tokens.', '',
          '## Cold requests and profiling', '',
          'Cold Codex decode was 14.39 → 15.29 tok/s at 3,998 input tokens and 8.98 → 10.26 tok/s at 29,930 tokens. The long cold pair differs by one speculative verification, so the matched ABBA pair above is the primary estimate. Both cold outputs matched exactly. Cold long-context ingestion took about 12 minutes; this kernel does not remove that cost.', '',
          'Separately instrumented long decode graphs used identical pool shapes: 29,952 cache cells, 7,490 pools, and three target verification tokens. Attributed fused pooling time fell from 54.714 ms to 24.284 ms. These are single sampled graph observations on different CPU workers; other operation timings varied. They are not substituted for unprofiled end-to-end measurements. Instrumented output also matched the unprofiled control.', '',
          '## Correctness and implementation', '',
          'The original first pooling fusion was already deployed. The old design document’s +50% projection compared against an older unfused runtime, not current approximately 14 tok/s performance. Its stale status is corrected.', '',
          'The new kernel groups eight channels, hoists four member-cell lookups out of the channel loop, and preserves scalar expf, float products, sequential double accumulation, and double reciprocal followed by float conversion. The strict graph matcher, model graph, tensor split, and other CPU objects are retained. The flag is GGML_CPU_GLM_POOL_WIDE=1; it defaults off.', '',
          '- 77 graph cases, each executed twice, match every output byte across five modes and 1/2/3/15 workers. Each arm compares 85,564,960 bytes, including 32K/100K pool sizes and fallback cases.',
          '- Repeated in-process switching at 3/15 workers and the existing copy-regression suite pass exact output checks.',
          '- All six native prompt runs, all twelve Codex A/B outputs, both instrumented outputs, and all nine existing stateful regression outputs match their corresponding references.',
          '- The pre-existing cached/fresh consistency failure remains unresolved. Regression parity is not a cache fix.', '',
          'Candidate CPU SHA256: `'+r['candidate_sha256']+'`.', '',
          'Source and build: [glm-kpool-wide-0919](/home/user/InfoSystemic/AI-Server/serving/fleet-0912-ctx/glm-kpool-wide-0919/README.md).', '',
          'Evidence: [summary](kpool-window-summary.json), [full window report](kpool-window-report.json), [reviewed input manifest](kpool-window-manifest.json), [production proposal](70-glm-kpool-wide-0919.conf.proposed).', '']
if r.get('window_seconds'):lines += [f"The temporary test window, including model reloads and correctness checks, lasted {r['window_seconds']/60:.1f} minutes. The original runtime’s command, library hashes, and inference environment were verified on restoration.", '']
if promotion:
    lines += ['## Deployment', '', f"Production PID {promotion.get('production_pid')} is healthy and verified. Native text parity and actual Codex cold/cached checks passed after deployment. The final Codex smoke rates were 15.04 and 15.06 tok/s. See [deployment report](kpool-promotion-report.json) and [final runtime snapshot](kpool-promotion-final.json).", '']
(D/'KPOOL-WIDE-RESULT.md').write_text('\n'.join(lines))
print(status)
