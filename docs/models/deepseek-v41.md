# DeepSeek-V4.1-Flash

**Status:** native FP8/FP4 CPU execution is demonstrated on short warm requests. Extended context and the separate llama.cpp port remain unpromoted.

## Native CPU runtime

The audited endpoint uses the released text checkpoint at revision `fb2764a5cf321eaa5070ca8f9e892818f477c16d`, native quantization, grouped MoE, quantization reuse, native HC, and scripted sparse attention. It measured 1.800 warm decode tok/s on two cached greetings. Context was 256 combined tokens, output at most 128 tokens, greedy and text-only. [CPU runtime and comparisons](../../engineering/2026-09-08/archive/serving/fleet-0903/DEEPSEEK-V41-CPU-TUNING-20260910.md).

The bounded cache does not hold the complete approximately 510 GB checkpoint. Missing expert tensors and Engram rows can require network ranges. A warm greeting and a novel long prompt therefore have very different costs. Native storage is retained; this is not a claim of full-checkpoint accelerator parity.

## Component evidence

- [Engram token history and hashing](../../engineering/2026-09-08/archive/serving/fleet-0903/DEEPSEEK-V41-ENGRAM-HASH-20260910.md): position, reset, overwrite, and rollback behavior.
- [Native lookup](../../engineering/2026-09-08/archive/serving/fleet-0903/DEEPSEEK-V41-ENGRAM-LOOKUP-20260910.md): 267,583,488 exact BF16 comparisons, plus a separate check of 48 real checkpoint rows.
- [Compressor and sparse-prefill work](../../engineering/2026-09-12/archive/serving/fleet-0912/deepseek/README.md): component references and unselected sparse-prefill timing experiments.
- [Context and cache recovery](../../engineering/2026-09-12/archive/serving/fleet-0912/deepseek-context/README.md): four context tests and five corrected interruption-recovery cases, without full-model 4K/16K validation.

## Separate llama.cpp port

The JigSaw source at `3b6fcfe4f7e2c282076f0c159278d3acfa3ad4e5` builds CPU benchmark and server targets with the archived Linux fixes. This port has not run the complete checkpoint here. Native activation/cache quantization differences, broad quality, and long context remain open. [Assessment](../../engineering/2026-09-12/archive/serving/fleet-0912/upstream/DEEPSEEK-V41-UPSTREAM-20260912.md) · [Complete patch](../../engineering/2026-09-12/patches/llama.cpp-deepseek41-jigsaw-0912.patch).

Legacy DeepSeek V4/DSpark work is retained for provenance and must not be reported as V4.1 performance. One archived DSpark branch retains an unresolved index merge, with all three merge-stage blobs preserved.

The next milestones are useful cold-cache behavior, varied generation, validated context-boundary behavior, and matched model execution in the new port. The over-10 tok/s objective remains open.
