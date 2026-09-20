# Current GLM short operation profile

The deployed wider pooling and bounded rollback runtime was profiled through the actual Codex/Paseo 3,998-token fixture. All eight bounded graph samples completed; output text matched the post-deployment reference. Production PID 1561764, library hashes and configuration remained unchanged, and profiling is disarmed. This request is excluded from throughput measurements.

One sampled target CPU graph reports 117.253 ms of operations plus 12.237 ms of barriers. These are per-worker observations, not an end-to-end cycle or a throughput estimate.

| Target work | Sampled operation time |
|---|---:|
| Expert matrix multiplies | 42.253 ms |
| Dense matrix multiplies | 35.581 ms |
| Attention | 9.476 ms |
| Fused pooling (attributed to GET_ROWS) | 3.847 ms |
| Custom reductions | 4.371 ms |
| Top-k | 2.852 ms |

The 5 draft samples span multiple socket workers and executions. Their output projection is about 1.77-1.83 ms of 4.08-4.20 ms sampled operation time. These five samples must not be added together as a single cycle. The separate coarse trace accounts for the actual three MTP passes.

The next source audit identifies a guarded opportunity in zero-output MTP catch-up: preserve hidden/token normalization, the combined projection, attention-input normalization, KV projection/normalization and the cache write; omit the query/attention result when no logits or hidden rows are requested. That audit subsequently produced the validated and deployed cache-only MTP path. Its matched actual Codex gain is +2.51%; exact state/logit/hidden gates passed. See [MTP-KV-RESULT.md](MTP-KV-RESULT.md).

Evidence: [profile report](kv-deployed-short-profile-report.json), [per-graph summary](profile-kv-deployed-short-profile.optrace-summary.json), [raw trace](profile-kv-deployed-short-profile.optrace.log).
