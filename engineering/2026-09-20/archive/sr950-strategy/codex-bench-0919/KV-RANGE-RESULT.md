# GLM bounded KV rollback result

Deployed and verified on the normal GLM endpoint as PID 1561764.

The scan optimization gives another small, measured improvement. It does not reach 18+ tok/s on the standard Codex/Paseo fixture.

| Actual Codex input: 3,998 tokens | Original-source loop | Bounded loop | Gain |
|---|---:|---:|---:|
| Cold, matched work | 15.09 tok/s | 15.70 tok/s | 4.0% |
| Cached, eight balanced repeats | 15.24 tok/s | 15.61 tok/s | 2.4% |

The primary comparison uses four runs per mode, in ABBA then BAAB order, within one loaded model on :18131. The mapped switch changes only between idle requests. Both blocks improved: 3.19%, 1.66%.

All cached repeats have identical generated text and identical input/cache, generated-token, draft-token, accepted-draft and verification counters. Each uses four new input tokens plus 3,994 cached tokens, generates 270 tokens, and records 210 drafted tokens, 165 accepted drafts and 105 verification steps. Decode rates include generated reasoning/output tokens and exclude prefill and client work. The fixture runs through the installed Paseo Codex wrapper and app-server path, with low reasoning, 504 instruction characters, temperature zero and seed 42.

Separate original-production cold controls were 15.10 tok/s before the window and 15.24 afterward. The original-source branch in the rebuilt candidate also receives compiler optimization, so it is not an instruction-identical substitute for the original binary. The repeated in-process result isolates the bound in the final candidate; the production controls bracket process and timing variation.

## What changed

Rollback used to scan all 1,048,576 allocated KV cells even for a short occupied prefix. The new branch snapshots the highest occupied cell index plus one and stops there. Cells after that bound are empty and cannot match the normalized removal range. It retains sequence handling, mutations and head updates. It changes no floating-point kernel arithmetic.

The first step was recovering an exact production libllama build from frozen objects and preserved source copies. The reconstructed parent matches the deployed SHA256 byte for byte: `8c794722eccdcc27aac88def7f1a1d549926a0d6aa3dd00586d6fdca5c8eb08a`. The candidate changes only the KV-cache object relative to that parent. The already deployed wider-pool CPU library is retained.

Candidate libllama SHA256: `e77a573df69456745857677e71e4e9cc876b8294384fa7798629f3ebe7c3306a`.

## Correctness and timing attribution

- Four standalone modes pass exact metadata-state comparison: production, candidate off, candidate on and alternating mapped control. Sixteen cache configurations exercise 5,650 scripted steps and save 2,800 states; all 2,826,776 bytes match in each mode.
- All six native outputs, all ten Codex A/B outputs, both instrumented Codex outputs and both bracketing Codex controls match their references. All 14 Codex texts are also identical to one another.
- All nine stateful regression outputs match production, including concurrent streams. The pre-existing cached/fresh consistency failure remains unresolved.

Separately instrumented traces contain 422 graphs each and reconcile all 105 target verifications. Across 101 steady cycles per mode, time outside graph phases falls from 9.616 to 6.032 ms/cycle. Total draft-model time remains about 21.39 ms/cycle, including 4.33 ms of draft-graph preparation. Target-graph timing varied too; these traces support the mechanism but are not substituted for the uninstrumented throughput result.

This window measures the short 3,998-token fixture. It adds no new 30K-context throughput result. The earlier wider-pool long-context A/B remains recorded separately in [KPOOL-WIDE-RESULT.md](KPOOL-WIDE-RESULT.md).

## Deployment and restoration

The successful temporary window lasted 19.79 minutes and restored the exact original runtime before the separate promotion. An earlier 10.54-minute attempt stopped because an INFO-level probe was filtered; it also restored production exactly. Raising the opt-in probe to WARN changed one executable byte plus the build ID, and its entire standalone gate passed again. See [the first-attempt record](KV-RANGE-FIRST-WINDOW-RESULT.md).

The production override retains the wider-pool library and sets `LLAMA_KV_SEQ_RM_USED_PREFIX=1`. Its optional coarse profiling trigger is `/dev/shm/glm-graph-phase.arm`, checked absent before and after deployment. Benchmark switching and mode-probe flags are absent from production. The deployment controller restores the prior configuration on identity or output failure.

Production PID 1561764 is healthy and verified. Native post-deployment parity passed, and final Codex cold/cached smoke rates were 15.58 and 15.71 tok/s with exact reference text. The coarse profiler is disarmed.

Evidence: [A/B summary](kv-range-r2-window-summary.json), [full report](kv-range-r2-window-report.json), [deployment report](kv-range-promotion-report.json), [source/build/gate documentation](/home/user/InfoSystemic/AI-Server/serving/fleet-0912-ctx/glm-kv-range-0919/README.md).
