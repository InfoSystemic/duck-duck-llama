# Recorded results

These selected results make the engineering work easy to inspect. Each row retains its own model, quantization, source stack, workload, and scope. They should not be combined into a cross-model ranking.

## Model measurements

| Experiment | Recorded result | Interpretation and source |
| --- | --- | --- |
| GLM Flash Q8, raw dissemination barrier | 11.61–11.68 prose / 11.69 code tok/s; 239.34–241.37 adjusted GB/s | Repeated control/candidate comparison; identical matching outputs. Generated reasoning reaches the 512-token cap. [Report](../engineering/2026-09-08/archive/serving/fleet-0903/FLASH-BANDWIDTH250-20260910.md) |
| GLM Flash Q4 with MTP2, actual Codex turn through Paseo, ~4K input tokens | production revision h: 19.8–20.6 tok/s greedy fixture, 18.7–19.9 sampled (revision f: 20.2 greedy, 19.7 sampled over 12 requests); was 16.24 on 09-19 | Every step measured inside one loaded process; one output text from revision b on. The attention kernel changes output toward the float64 result. Four further ideas (F16 router, dense Q6_K, two composite drafters) measured slower and were rolled back. [Report](../benchmarks/glm53-flash-paseo-decode-20260920.md), [revisions g–l](models/glm-flash.md#revisions-gl-september-20-evening-what-24-toks-would-take) |
| GLM Flash Q4, decode and prefill against context depth, one append-only session | decode 19.6 tok/s at 25K, 15.5 at 67K, 12.6 at 128K; prefill of new tokens 64 → 14.5 tok/s from 8K to 128K | The slope is the sparse-attention indexer's bookkeeping, not attention. A +51 ms layer in the first 128K trace was an artifact of a restored session. [Report](../benchmarks/glm53-flash-context-depth-20260920.md) |
| MiMo-V2.6-Pro-RL (1.02T MoE, 518 GiB MXFP4/Q8_0), four-way tensor parallel with Xiaomi's DFlash drafter | 22.8 verbatim / 26.9 counting / 16.3 code / 13.2 list / 8.8 prose tok/s at short context; 7.94 without speculation (1.03 without tensor parallelism); prefill 60–69 tok/s; decode 6.4 tok/s at 64K (was 1.98) | Six defects fixed between the drafter and the target; `p_min` 0.5 is the control; the verify cycle runs at 79% of the memory wall. Golden output identical across builds 0922c–0922j. [Benchmark](../benchmarks/mimo-v26-pro-cpu-20260923.md), [guide](models/mimo-v26-pro.md) |
| Qwen Q6 with Q8 MTP4, retained configuration | 21.53 prose / 28.46 code tok/s | Historical measurements of the identified selected stack. [Source table and records](../engineering/2026-09-08/README.md#retained-measurements) |
| Qwen corrected gather with HC/shared dispatch and MTP4 | 24.04–24.35 prose / 30.80–31.62 code tok/s; mean gains +3.94% / +2.50% | Parent/corrected/corrected/parent; outputs and draft counts agree. No runtime promotion. [Report](../engineering/2026-09-08/archive/serving/fleet-0903/BANDWIDTH250-20260909.md) |
| GLM Full, Q4-based mixed runtime with MTP2 | 7.89 prose / 9.64 code tok/s | Historical generated-token observations; runtime requantization and a hybrid draft matter. Current loadability is a separate open issue. [Record](../engineering/2026-09-08/measured-status.json) |
| DeepSeek V4.1 native CPU endpoint | 1.800 warm decode tok/s | Two short cached greeting requests; nine decode passes per response. Longer and cold requests unqualified. [Audit and report](../engineering/2026-09-08/archive/serving/fleet-0903/DEEPSEEK-V41-CPU-TUNING-20260910.md) |

## Component and implementation evidence

| Work | Evidence | Limit |
| --- | --- | --- |
| CPU flash attention, F16 cache | stock 6.4e-3 relative RMS error vs float64 (1.3e-2 on real tensors, 3-query batch); F32 kernel 2e-7 to 7e-7, bit-identical alone vs in a batch (stock: 3.4e-4 apart) | Reproduced on unmodified upstream `b23efaa2`. Operator arithmetic only; no model-quality benchmark. [Report](../benchmarks/cpu-flash-attn-f16-accumulation.md) |
| Grouped-query split-KV CPU flash attention | 11.2× at 65K cells × 8 verify rows (160.5 → 14.3 ms), later ~14×; relative error 3.8e-2 → 8.6e-7 against float64 | Stock dispatch re-streams the KV cache per query row and Q head for 2–63-row batches, identical in upstream `b23efaa2`. [Report](../benchmarks/cpu-flash-attn-gqa-splitkv.md) |
| Cascade Lake VNNI issue rate | register-form `vpdpbusd` ~1.85/cycle, `{1to16}` memory broadcast ~1.0–1.35; paired-group kernels 1.26× dense, 1.21× MoE gate, bit-identical | Single core, sibling hyperthread held idle; not re-checked on the other models' kernels. [Report](../benchmarks/cascade-lake-vnni-broadcast.md) |
| Qwen-Image-2.1 on CPU | 20-step 512×512 image in ~7.5 min, 1024×1024 in ~35 min, on 45 cores; ggml matmuls slower on three sockets than one at attention shapes | Runs only on cores the model servers leave idle. [Report](../benchmarks/qwen-image-21-cpu.md) |
| Native Engram lookup | 267,583,488 exact BF16 values; 1,044,096 exact row IDs | Component/fixture and selected real-row validation. [Report](../engineering/2026-09-08/archive/serving/fleet-0903/DEEPSEEK-V41-ENGRAM-LOOKUP-20260910.md) |
| GLM Full split correction | 57,888 metadata checks across 1,809 tensors and 32 schedules; candidate library built | Unequal-split model inference pending. [Report](../engineering/2026-09-12/archive/serving/fleet-0912/glmfull-split/README.md) |
| NUMA relocation utility | 4 MiB fixture, 3 MiB moved, balanced final placement, unchanged complete SHA-256 | Successful bulk model migration not established. [Evidence](../engineering/2026-09-12/archive/serving/fleet-0912/numa/fixture-verification.json) |
| Native cache placement inventory | 5,530 tensors / 36.734 GiB; every page queried and all complete hashes matched before/after | Read-only cache inventory; no relocation or full-checkpoint claim. [Evidence](../engineering/2026-09-12/archive/serving/fleet-0912/numa/deepseek-inventory-summary-20260912-0928.json) |
| Native sparse prefill | 45 cases, 4,998,306 byte-exact BF16 outputs | Mixed timings; unselected. [Report](../engineering/2026-09-12/archive/serving/fleet-0912/deepseek/README.md) |
| Persistent Engram cache recovery | Five failing fault-injection cases corrected; four existing context tests still pass | No full-model extended-context validation. [Evidence](../engineering/2026-09-12/archive/serving/fleet-0912/deepseek-context/recovery-verification.json) |

The newest GLM Flash unary and Qwen unary/ngram recipes also preserve historical measurements and rebuilt libraries. Their fresh A/B/A runs remain pending, so they are documented in the [model guides](models/README.md) rather than presented as newly established speedups.

An outside tester reproduced the GLM-5.3-Flash engine on a two-socket Xeon Silver machine: 6.74 tok/s without speculation and 7.27 with MTP at 0.72 acceptance. That run found a bug the development machine never exercised. [Independent verification](independent-verification.md).

No result establishes the earlier all-model 250 GB/s target or the requested Qwen 40 tok/s and DeepSeek over-10 tok/s goals. The [methodology](methodology.md) explains the measurement limits; the [roadmap](roadmap.md) keeps the unresolved work explicit.
