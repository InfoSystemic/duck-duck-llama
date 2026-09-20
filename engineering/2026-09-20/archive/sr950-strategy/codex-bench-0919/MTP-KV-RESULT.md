# GLM MTP cache-only catch-up result

Deployed and verified on the normal endpoint as PID 3717968.

The new path produces another small, repeatable gain. The measured short Codex fixture remains below 17 tok/s; no consistent 17+ or 18+ result is established.

| Actual Codex input: 3,998 tokens | Full catch-up graph | Cache-only catch-up | Gain |
|---|---:|---:|---:|
| Cold, matched work | 15.28 tok/s | 15.77 tok/s | 3.20% |
| Cached, eight balanced repeats | 15.43 tok/s | 15.82 tok/s | 2.51% |

The primary comparison uses ABBA then BAAB order in one loaded model on private port 18141. Both blocks improved: 2.69%, 2.33%. All eight outputs and all input/cache, generated-token, draft-token, accepted-draft and verification counts match. Each cached request uses four new plus 3,994 cached input tokens, emits 270 tokens, and records 210 drafted tokens, 165 accepted drafts and 105 verification steps. The installed Paseo Codex wrapper/app-server path runs at low reasoning, temperature zero and seed 42. Rates exclude prefill and client work.

Original-production cold controls were 15.85 and 15.34 tok/s. These separate process loads have approximately 3.2% control drift and a one-accepted-draft difference; they are not the primary comparison. The candidate's disabled branch retains the original graph but differs in compiler layout and adds a mode guard. The balanced in-process result is the measured effect of selecting the fast path in the final library.

## Implementation and exact gates

The MTP catch-up batch requests no output rows. The guarded fast path keeps the token/hidden normalization, combined projection, attention-input normalization, KV projection and normalization, and identical cache writes. It omits the unused query/attention path. It applies only to a single MTP layer with zero outputs, masked next-token embeddings, no ordinary embeddings, no backend samplers and no requested intermediate layer outputs. Other cases keep the full graph. No retained floating-point arithmetic changes.

Only models/glm5next.cpp.o changes over the byte-identical current production parent. Candidate libllama SHA256: `65670d4a19178bbdffcfbc79ee953e5319083875309ed52aa4de214393e20ce3`. Existing wider-pool and bounded-rollback improvements remain enabled.

- Standalone production/off/on/alternating-switch gates at 1 and 15 workers per socket match all 170 records and 28,400,396 bytes per arm, including actual cache tensors, sequence metadata, logits and hidden rows.
- All six native A/B outputs and all 14 Codex outputs, including instrumented and bracketing controls, match their references.
- All nine stateful output comparisons match production. The pre-existing cached/fresh consistency issue remains unresolved.
- Every primary Codex request matches its own cumulative token usage. The separate first shared-port window has a contaminated control and is excluded.

## Timing attribution

Both saved traces contain 422 graphs and reconcile all 105 target verifications, with 101 steady cycles each. Draft work falls from 21.269 to 18.152 ms/cycle. The catch-up pass itself falls from 7.398 to 4.411 ms. Enabled draft preparation still costs 3.349 ms; the catch-up and first prediction rebuild, while the second prediction reuses its graph.

Target time in these separate instrumented traces is 136.906 versus 137.666 ms; outside-graph time is 6.033 versus 6.565 ms. These profiles confirm the mechanism and are excluded from throughput estimates. No new long-context throughput result is claimed.

## Restoration and harness repair

The private test window lasted 19.67 minutes and restored the exact original libraries, inference environment and command before the separate deployment. All measurements and both traces finished, but the controller's imported parser rejected the expected new 15-node graph because its allowed set contained only 7152/89. The raw failed controller report is preserved unchanged. The parser now accepts an explicit mode-specific node set, while its historical default stays strict. Saved traces were validated offline, and a separate final production control completed without another benchmark outage. See the derived validation report and its source hashes; this is not recorded as a clean controller execution.

An earlier shared-port attempt stopped before enabling the candidate because an unrelated request was active. It restored production and is not included in the A/B. A standalone reference-only separate-stream sequence-copy crash is preserved in the candidate directory; normal production uses unified KV and the final exact gate avoids that unrelated failing operation. Neither existing issue is claimed fixed.

The production drop-in sets `GGML_GLM5N_MTP_KV_ONLY=1` and selects the immutable library. The benchmark control and diagnostic probe are absent. The deployment controller rolls back only that new drop-in if runtime identity or output checks fail.

Final native output parity passed. Post-deployment Codex cold/cached smoke rates are 15.67 / 15.52 tok/s, with exact reference text and clean per-request accounting. These are smoke checks, not a workload-wide throughput floor. Production PID 3717968 is healthy; profiling is disarmed.

Evidence: [A/B summary](mtp-kv-only-r2-window-summary.json), [derived validated report](mtp-kv-only-r2-validation-report.json), [preserved raw controller report](mtp-kv-only-r2-window-report.json), [phase breakdown](mtp-kv-only-r2-phase-breakdown.json), [deployment report](mtp-kv-only-promotion-report.json), [source and standalone gates](/home/user/InfoSystemic/AI-Server/serving/fleet-0912-ctx/glm-mtp-kv-only-0919/README.md).
