# Patch attribution

The [September 8 source inventory](../engineering/2026-09-08/source-bundles.json) extends these bundles with exact snapshots of the local engineering source lines. Qwen's goal snapshot uses ggml-org/llama.cpp at `daef7b6874397a5a7c3d7e38b55e2ee0adf7da38`; GLM Flash uses unslothai/llama.cpp at `2e0e57f1008053bae4902a772da85e3eb99d4aff`. The Full, earlier NUMA and dense-server snapshots retain the bases listed in the inventory. Existing upstream notices are preserved. Private CPU/model overlays and archived local fixtures are published with their source hashes; upstream model-support code is not claimed as locally authored.

The repository separates locally developed engineering from changes derived
from upstream work. Patch filenames pin the source revision they target.

## `llama.cpp-b249-cpu-numa.patch`

This experimental Linux CPU-NUMA backend was developed for this repository
against llama.cpp build 249 (`3173a56471c1753650cd806694145ffd6dcace67`).

## `llama.cpp-b249-qwen4exp-mtp.patch`

The Qwen3.8 / Qwen4 experimental model and MTP support is derived from these
public llama.cpp contributions:

- [ggml-org/llama.cpp PR #27836](https://github.com/ggml-org/llama.cpp/pull/27836)
  by Ryan Monsurate, specifically commits `d303eec923f92ccab7109e97d95cb5c1ab83e0d2`,
  `d72620018f612bb05d16e585108e3440947a1f9f`, and
  `1d8de7c1b0c7d2febf8f983174d8e6a711e2b1af`.
- [detached MTP-sidecar fix `a82a58a`](https://github.com/crusaderky/llama.cpp/commit/a82a58a57fc307e5cec0dc68db64d143339be4f2)
  by crusaderky.
- [detached MTP-sidecar fix `57bb668`](https://github.com/crusaderky/llama.cpp/commit/57bb668674d9fb0d382885e5b04911c6437f8e83)
  by drluoto.

The published patch is a narrowly scoped, combined engineering snapshot for
reproducibility. It does not claim original authorship of those upstream
changes.

## `llama.cpp-a302733-glm-sr950.patch`

This integration patch combines llama.cpp compatibility work, CPU-NUMA
execution, compact-quant and x86 kernel work, GLM tensor-parallel execution,
speculative-server plumbing, and tests against
`a30273376ef669023334fc20ad02ae4ed8196a65`. Any upstream-derived code retains
the notices and history present in the patch. See the patch itself and
[`README.md`](README.md) for the exact audited delta and limitations.

## GLM-5.3-Flash decode series, 2026-09-20

`glm5next-kpool-fusion-statecopy.patch`, `glm5next-kpool-wide.patch`, `glm5next-pool-result-cache.patch`,
`glm-kv-seq-rm-used-prefix.patch` and `glm5next-mtp-kv-only-catchup.patch` were developed for this repository on 2026-09-19;
`glm5next-fa-mqa-cellsplit.patch`, `topk-select-tie-fallback.patch` and `upstream-cpu-fattn-f32-accumulate.patch` on 2026-09-20.
All were written with AI coding agents under the repository owner's direction. They modify llama.cpp source files (MIT) and the
glm5next architecture support of the unslothai/llama.cpp line at `2e0e57f1008053bae4902a772da85e3eb99d4aff`, which is not claimed
as locally authored. The attention kernel reuses the tree's `simd_gemm` helper. KTransformers' kt-kernel independently splits
routed-expert weights across NUMA thread pools for GPU-hybrid serving; nothing here derives from its code.

