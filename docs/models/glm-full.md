# GLM-5.3 Full

**Status:** historical real-model inference is recorded; recent loads under the current co-resident cache footprint fail on NUMA node 3. The new unequal-split correction builds but has not completed a full-model trial.

## Runtime and evidence

The Full integration uses the `sr950-glm` source line because its GLM DSA tensor-split path is required by this model. The recorded Q4-based mixed runtime uses load-time attention/shared-expert requantization and a hybrid MTP draft. The GGUF filename therefore does not fully describe runtime precision.

Historical MTP2 generated-token measurements are 7.89 prose and 9.64 code tok/s. A subsequent raw observation reached 7.44–7.57 tok/s but failed output-repeatability checks and exhausted its reasoning cap; it is retained as a rejected trial. [Retained measurements](../../engineering/2026-09-08/measured-status.json) · [Rejected raw trial](../../engineering/2026-09-08/archive/serving/fleet-0903/FULL-RAW-BANDWIDTH-20260910.md).

## Current source work

The [foreground profile](../../engineering/2026-09-12/archive/serving/fleet-0912/glmfull/README.md) preserves the v5b settings and exposes separately identified arms. The original NUMA-repack setting and the later `NUMA_REPACK=0` attempt both encountered node-3 OOM. [Latest inspected failure](../../engineering/2026-09-12/archive/serving/fleet-0912/results/glmfull-no-numa-repack-873660/kernel-oom.txt).

The [split correction](../../engineering/2026-09-12/archive/serving/fleet-0912/glmfull-split/README.md) aligns affected slices to the source/requantization block requirements. It passes 57,888 metadata checks, preserves all balanced splits, and builds as an isolated library. A baseline relink matches the original library byte for byte. The candidate launcher checks its dependency hashes and loader selection before execution.

## Reproduction entry points

- [Complete source bundles](../../engineering/2026-09-12/source-bundles.json): select `llama.cpp-sr950-glm`.
- [Host profile and launcher](../../engineering/2026-09-12/archive/serving/fleet-0912/glmfull/profile.json).
- [Split patch, metadata fixtures, and build recipe](../../engineering/2026-09-12/archive/serving/fleet-0912/glmfull-split/README.md).
- [NUMA placement utility](../../engineering/2026-09-12/archive/serving/fleet-0912/numa/README.md).

The model uses eleven shards. The preferred hybrid draft is RAM-backed and must be staged again after reboot. The next decisive evidence is a completed load with measured per-node headroom, followed by semantic, repeatability, and throughput checks. A metadata or build pass is insufficient for promotion.
