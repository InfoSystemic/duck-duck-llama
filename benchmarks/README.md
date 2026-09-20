# Benchmark reports

Start with the [selected results](../docs/results.md) for measurements with their current interpretation, or the [methodology](../docs/methodology.md) for timing, correctness, and counter checks. This directory preserves the earlier experiment reports, including conclusions revised by later source fixes and controlled comparisons.

## Models and serving

| Report | What it records |
| --- | --- |
| [SR950 model profiles](sr950-model-profiles.md) | Early GLM Full/Flash and Qwen profiles, with later corrections and unreproduced replay rates identified |
| [Qwen 27B CPU-NUMA](qwen38-27b-cpu-numa.md) | The earlier model's placement, thread, collective, and speculative sweeps |
| [Qwen Flash Next NUMA](qwen38-flash-next-numa.md) | Progress from node-local execution to four CPU-NUMA devices, and failed MTP/placement arms |
| [GLM Flash NUMA port](glm53-flash-numa-port.md) | Porting the backend into a model-support source line |
| [GLM Flash decode through Paseo, 09-18 to 09-20](glm53-flash-paseo-decode-20260920.md) | 12.0 to 20.2 tok/s as a Codex agent sees it (19.7 on sampled requests): same-process A/Bs, the cycle budget, two OpenMP teams per core, batch-invariant attention, coupled sampling, and the negative results |
| [Concurrent throughput](concurrent-throughput.md) | Multi-request serving measurements; distinct from single-conversation decode |
| [Ten tokens per second](ten-tokens-per-second.md) | A target-driven experiment log and corrections to earlier replay claims |

## Correctness and unsuccessful experiments

| Report | What it records |
| --- | --- |
| [Qwen tensor-split corruption](qwen4exp-tensor-split-corruption.md) | An earlier fast but incorrect execution path; later source variants repair the underlying support |
| [GLM tensor-parallel blocker](glm5next-tensor-parallel-blocker.md) | Architecture-specific constraints in an earlier integration |
| [Full composite speculation failure](glm53-full-composite-spec-failure.md) | A speculative-decoding combination that did not qualify |
| [CPU flash attention sums V in FP16](cpu-flash-attn-f16-accumulation.md) | About 1% error in the decode path with the default F16 cache, reproduced on stock upstream and measured on real tensors; a minimal fix and its cost |
| [Kernel efficiency ceiling](kernel-efficiency-ceiling.md) | Component measurements and their limits as a predictor of full-model speed |
| [Fixed per-token overhead](fixed-per-token-overhead.md) | Investigation of work that does not disappear with smaller tensor traffic |

The later [engineering archive](../engineering/README.md) contains the September counter audits, repeated model comparisons, numerical fixtures, and native DeepSeek work. Its searchable catalog is the fastest way to find a particular experiment. The [model guides](../docs/models/README.md) connect early reports to those later results.
