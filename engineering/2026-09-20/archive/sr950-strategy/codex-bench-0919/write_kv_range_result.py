#!/usr/bin/env python3
from pathlib import Path
import json

HERE=Path(__file__).resolve().parent
ROOT=Path('/home/user/InfoSystemic/AI-Server/serving/fleet-0912-ctx/glm-kv-range-0919')
summary=json.loads((HERE/'kv-range-r2-window-summary.json').read_text())
window=json.loads((HERE/'kv-range-r2-window-report.json').read_text())
gate=json.loads((ROOT/'validation.json').read_text())
promotion_path=HERE/'kv-range-promotion-report.json'
promotion=json.loads(promotion_path.read_text()) if promotion_path.exists() else {}
texts=[window['production_codex_before']['output_text'],window['production_codex_after']['output_text']]
texts += [row['output_text'] for key,row in window['arms'].items() if key.startswith(('cold','warm'))]
texts += [row['benchmark']['output_text'] for row in window['profiles'].values()]
assert len(texts)==14 and len(set(texts))==1
for rows in [window['arms']['native-'+str(i)]['rows'] for i in [0,1]]:
    assert all(r['parity'] for r in rows)
off=summary['cold']['0']['decode_tokens_per_second']
on=summary['cold']['1']['decode_tokens_per_second']
work_keys=['prompt_tokens','cached_prompt_tokens','generated_tokens','draft_tokens','accepted_draft_tokens','draft_verification_steps']
cold_work_matches=all(summary['cold']['0'][k]==summary['cold']['1'][k] for k in work_keys)
assert cold_work_matches
p0,p1=summary['profiles']['0']['steady_mean'],summary['profiles']['1']['steady_mean']
if promotion.get('promoted'):
    lead=f"Deployed and verified on the normal GLM endpoint as PID {promotion['production_pid']}."
else:
    lead='Full A/B validation passed; production deployment verification is in progress.'
lines=[
'# GLM bounded KV rollback result',
'',lead,'',
'The scan optimization gives another small, measured improvement. It does not reach 18+ tok/s on the standard Codex/Paseo fixture.',
'',
'| Actual Codex input: 3,998 tokens | Original-source loop | Bounded loop | Gain |',
'|---|---:|---:|---:|',
f"| Cold, matched work | {off:.2f} tok/s | {on:.2f} tok/s | {100*(on/off-1):.1f}% |",
f"| Cached, eight balanced repeats | {summary['warm_original_loop_tps']:.2f} tok/s | {summary['warm_bounded_tps']:.2f} tok/s | {summary['warm_gain_percent']:.1f}% |",
'',
'The primary comparison uses four runs per mode, in ABBA then BAAB order, within one loaded model on :18131. The mapped switch changes only between idle requests. Both blocks improved: '+', '.join(f"{row['gain_percent']:.2f}%" for row in summary['blocks'])+'.',
'',
'All cached repeats have identical generated text and identical input/cache, generated-token, draft-token, accepted-draft and verification counters. Each uses four new input tokens plus 3,994 cached tokens, generates 270 tokens, and records 210 drafted tokens, 165 accepted drafts and 105 verification steps. Decode rates include generated reasoning/output tokens and exclude prefill and client work. The fixture runs through the installed Paseo Codex wrapper and app-server path, with low reasoning, 504 instruction characters, temperature zero and seed 42.',
'',
f"Separate original-production cold controls were {window['production_codex_before']['server_metrics']['decode_tokens_per_second']:.2f} tok/s before the window and {window['production_codex_after']['server_metrics']['decode_tokens_per_second']:.2f} afterward. The original-source branch in the rebuilt candidate also receives compiler optimization, so it is not an instruction-identical substitute for the original binary. The repeated in-process result isolates the bound in the final candidate; the production controls bracket process and timing variation.",
'',
'## What changed',
'',
'Rollback used to scan all 1,048,576 allocated KV cells even for a short occupied prefix. The new branch snapshots the highest occupied cell index plus one and stops there. Cells after that bound are empty and cannot match the normalized removal range. It retains sequence handling, mutations and head updates. It changes no floating-point kernel arithmetic.',
'',
'The first step was recovering an exact production libllama build from frozen objects and preserved source copies. The reconstructed parent matches the deployed SHA256 byte for byte: `8c794722eccdcc27aac88def7f1a1d549926a0d6aa3dd00586d6fdca5c8eb08a`. The candidate changes only the KV-cache object relative to that parent. The already deployed wider-pool CPU library is retained.',
'',
'Candidate libllama SHA256: `'+summary['candidate_sha256']+'`.',
'',
'## Correctness and timing attribution',
'',
'- Four standalone modes pass exact metadata-state comparison: production, candidate off, candidate on and alternating mapped control. Sixteen cache configurations exercise 5,650 scripted steps and save 2,800 states; all 2,826,776 bytes match in each mode.',
'- All six native outputs, all ten Codex A/B outputs, both instrumented Codex outputs and both bracketing Codex controls match their references. All 14 Codex texts are also identical to one another.',
'- All nine stateful regression outputs match production, including concurrent streams. The pre-existing cached/fresh consistency failure remains unresolved.',
'',
f"Separately instrumented traces contain {summary['profiles']['0']['graph_count']} graphs each and reconcile all 105 target verifications. Across 101 steady cycles per mode, time outside graph phases falls from {p0['outside_graph_ms']:.3f} to {p1['outside_graph_ms']:.3f} ms/cycle. Total draft-model time remains about {p1['draft_ms']:.2f} ms/cycle, including {p1['draft_prepare_ms']:.2f} ms of draft-graph preparation. Target-graph timing varied too; these traces support the mechanism but are not substituted for the uninstrumented throughput result.",
'',
'This window measures the short 3,998-token fixture. It adds no new 30K-context throughput result. The earlier wider-pool long-context A/B remains recorded separately in [KPOOL-WIDE-RESULT.md](KPOOL-WIDE-RESULT.md).',
'',
'## Deployment and restoration',
'',
f"The successful temporary window lasted {summary['window_seconds']/60:.2f} minutes and restored the exact original runtime before the separate promotion. An earlier {json.loads((HERE/'kv-range-window-report.json').read_text())['window_seconds']/60:.2f}-minute attempt stopped because an INFO-level probe was filtered; it also restored production exactly. Raising the opt-in probe to WARN changed one executable byte plus the build ID, and its entire standalone gate passed again. See [the first-attempt record](KV-RANGE-FIRST-WINDOW-RESULT.md).",
'',
'The production override retains the wider-pool library and sets `LLAMA_KV_SEQ_RM_USED_PREFIX=1`. Its optional coarse profiling trigger is `/dev/shm/glm-graph-phase.arm`, checked absent before and after deployment. Benchmark switching and mode-probe flags are absent from production. The deployment controller restores the prior configuration on identity or output failure.',
'']
if promotion.get('promoted'):
    cold=promotion['kv-range-promoted-cold']['server_metrics']['decode_tokens_per_second']
    warm=promotion['kv-range-promoted-warm']['server_metrics']['decode_tokens_per_second']
    lines += [f"Production PID {promotion['production_pid']} is healthy and verified. Native post-deployment parity passed, and final Codex cold/cached smoke rates were {cold:.2f} and {warm:.2f} tok/s with exact reference text. The coarse profiler is disarmed.",'']
lines += ['Evidence: [A/B summary](kv-range-r2-window-summary.json), [full report](kv-range-r2-window-report.json), [deployment report](kv-range-promotion-report.json), [source/build/gate documentation]('+str(ROOT/'README.md')+').','']
(HERE/'KV-RANGE-RESULT.md').write_text('\n'.join(lines))
print(HERE/'KV-RANGE-RESULT.md')
