# GLM wider kpool kernel result

Deployed and verified on the normal endpoint, production PID 3692141.

The new inner loop improves measured throughput, but does not reach 18+ tok/s on the standard Codex/Paseo benchmark.

| Actual Codex input | Scalar control | Wider loop | Gain |
|---|---:|---:|---:|
| 3,998 tokens | 14.60 tok/s | 15.15 tok/s | 3.8% |
| 29,930 tokens | 9.05 tok/s | 10.26 tok/s | 13.4% |

Primary results use two scalar and two wider-kernel requests in ABBA order. Each comparison has identical generated text, input/cache counts, generated-token counts, draft-token counts, accepted drafts, and verification steps. Decode rates exclude prefill and client work. Both modes run in the same loaded model on :18131 with a switch changed only while idle.

The separate original-service controls measured 14.46 tok/s before the window and 14.78 afterward, with one accepted-draft difference. Within the same loaded model, short ABBA controls drifted about 1.0%; long controls drifted about 0.1%. The long fixture is deterministic synthetic filler plus the standard LRU explanation task, not a real [client] conversation. Its user prompt has 26,000 tokens; actual Codex input is 29,930 tokens.

## Cold requests and profiling

Cold Codex decode was 14.39 → 15.29 tok/s at 3,998 input tokens and 8.98 → 10.26 tok/s at 29,930 tokens. The long cold pair differs by one speculative verification, so the matched ABBA pair above is the primary estimate. Both cold outputs matched exactly. Cold long-context ingestion took about 12 minutes; this kernel does not remove that cost.

Separately instrumented long decode graphs used identical pool shapes: 29,952 cache cells, 7,490 pools, and three target verification tokens. Attributed fused pooling time fell from 54.714 ms to 24.284 ms. These are single sampled graph observations on different CPU workers; other operation timings varied. They are not substituted for unprofiled end-to-end measurements. Instrumented output also matched the unprofiled control.

## Correctness and implementation

The original first pooling fusion was already deployed. The old design document’s +50% projection compared against an older unfused runtime, not current approximately 14 tok/s performance. Its stale status is corrected.

The new kernel groups eight channels, hoists four member-cell lookups out of the channel loop, and preserves scalar expf, float products, sequential double accumulation, and double reciprocal followed by float conversion. The strict graph matcher, model graph, tensor split, and other CPU objects are retained. The flag is GGML_CPU_GLM_POOL_WIDE=1; it defaults off.

- 77 graph cases, each executed twice, match every output byte across five modes and 1/2/3/15 workers. Each arm compares 85,564,960 bytes, including 32K/100K pool sizes and fallback cases.
- Repeated in-process switching at 3/15 workers and the existing copy-regression suite pass exact output checks.
- All six native prompt runs, all twelve Codex A/B outputs, both instrumented outputs, and all nine existing stateful regression outputs match their corresponding references.
- The pre-existing cached/fresh consistency failure remains unresolved. Regression parity is not a cache fix.

Candidate CPU SHA256: `3f957a341321b940d93be53c250cdd068825093faa2d9efda142ebe56427b1b3`.

Source and build: [glm-kpool-wide-0919](/home/user/InfoSystemic/AI-Server/serving/fleet-0912-ctx/glm-kpool-wide-0919/README.md).

Evidence: [summary](kpool-window-summary.json), [full window report](kpool-window-report.json), [reviewed input manifest](kpool-window-manifest.json), [production proposal](70-glm-kpool-wide-0919.conf.proposed).

The temporary test window, including model reloads and correctness checks, lasted 46.0 minutes. The original runtime’s command, library hashes, and inference environment were verified on restoration.

## Deployment

Production PID 3692141 is healthy and verified. Native text parity and actual Codex cold/cached checks passed after deployment. The final Codex smoke rates were 15.04 and 15.06 tok/s. See [deployment report](kpool-promotion-report.json) and [final runtime snapshot](kpool-promotion-final.json).
