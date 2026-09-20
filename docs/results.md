# Recorded results

These selected results make the engineering work easy to inspect. Each row retains its own model, quantization, source stack, workload, and scope. They should not be combined into a cross-model ranking.

## Model measurements

| Experiment | Recorded result | Interpretation and source |
| --- | --- | --- |
| GLM Flash Q8, raw dissemination barrier | 11.61–11.68 prose / 11.69 code tok/s; 239.34–241.37 adjusted GB/s | Repeated control/candidate comparison; identical matching outputs. Generated reasoning reaches the 512-token cap. [Report](../engineering/2026-09-08/archive/serving/fleet-0903/FLASH-BANDWIDTH250-20260910.md) |
| GLM Flash Q4 with MTP2, actual Codex turn through Paseo, ~4K input tokens | production 19.2 tok/s greedy fixture (19.17–19.32), 18.2 on requests as Codex sends them (n=12, 17.6–18.9); was 16.24 on 09-19 | Every step measured inside one loaded process; one output text from revision b on. The attention kernel changes output toward the float64 result. [Report](../benchmarks/glm53-flash-paseo-decode-20260920.md) |
| Qwen Q6 with Q8 MTP4, retained configuration | 21.53 prose / 28.46 code tok/s | Historical measurements of the identified selected stack. [Source table and records](../engineering/2026-09-08/README.md#retained-measurements) |
| Qwen corrected gather with HC/shared dispatch and MTP4 | 24.04–24.35 prose / 30.80–31.62 code tok/s; mean gains +3.94% / +2.50% | Parent/corrected/corrected/parent; outputs and draft counts agree. No runtime promotion. [Report](../engineering/2026-09-08/archive/serving/fleet-0903/BANDWIDTH250-20260909.md) |
| GLM Full, Q4-based mixed runtime with MTP2 | 7.89 prose / 9.64 code tok/s | Historical generated-token observations; runtime requantization and a hybrid draft matter. Current loadability is a separate open issue. [Record](../engineering/2026-09-08/measured-status.json) |
| DeepSeek V4.1 native CPU endpoint | 1.800 warm decode tok/s | Two short cached greeting requests; nine decode passes per response. Longer and cold requests unqualified. [Audit and report](../engineering/2026-09-08/archive/serving/fleet-0903/DEEPSEEK-V41-CPU-TUNING-20260910.md) |

## Component and implementation evidence

| Work | Evidence | Limit |
| --- | --- | --- |
| CPU flash attention, F16 cache | stock 6.4e-3 relative RMS error vs float64 (1.3e-2 on real tensors, 3-query batch); F32 kernel 2e-7 to 7e-7, bit-identical alone vs in a batch (stock: 3.4e-4 apart) | Reproduced on unmodified upstream `b23efaa2`. Operator arithmetic only; no model-quality benchmark. [Report](../benchmarks/cpu-flash-attn-f16-accumulation.md) |
| Native Engram lookup | 267,583,488 exact BF16 values; 1,044,096 exact row IDs | Component/fixture and selected real-row validation. [Report](../engineering/2026-09-08/archive/serving/fleet-0903/DEEPSEEK-V41-ENGRAM-LOOKUP-20260910.md) |
| GLM Full split correction | 57,888 metadata checks across 1,809 tensors and 32 schedules; candidate library built | Unequal-split model inference pending. [Report](../engineering/2026-09-12/archive/serving/fleet-0912/glmfull-split/README.md) |
| NUMA relocation utility | 4 MiB fixture, 3 MiB moved, balanced final placement, unchanged complete SHA-256 | Successful bulk model migration not established. [Evidence](../engineering/2026-09-12/archive/serving/fleet-0912/numa/fixture-verification.json) |
| Native cache placement inventory | 5,530 tensors / 36.734 GiB; every page queried and all complete hashes matched before/after | Read-only cache inventory; no relocation or full-checkpoint claim. [Evidence](../engineering/2026-09-12/archive/serving/fleet-0912/numa/deepseek-inventory-summary-20260912-0928.json) |
| Native sparse prefill | 45 cases, 4,998,306 byte-exact BF16 outputs | Mixed timings; unselected. [Report](../engineering/2026-09-12/archive/serving/fleet-0912/deepseek/README.md) |
| Persistent Engram cache recovery | Five failing fault-injection cases corrected; four existing context tests still pass | No full-model extended-context validation. [Evidence](../engineering/2026-09-12/archive/serving/fleet-0912/deepseek-context/recovery-verification.json) |

The newest GLM Flash unary and Qwen unary/ngram recipes also preserve historical measurements and rebuilt libraries. Their fresh A/B/A runs remain pending, so they are documented in the [model guides](models/README.md) rather than presented as newly established speedups.

No result establishes the earlier all-model 250 GB/s target or the requested Qwen 40 tok/s and DeepSeek over-10 tok/s goals. The [methodology](methodology.md) explains the measurement limits; the [roadmap](roadmap.md) keeps the unresolved work explicit.
