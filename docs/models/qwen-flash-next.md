# Qwen3.8-Flash-Next

**Status:** Q6/MTP inference, repeated corrected-gather comparisons, and request-state checks are recorded. The latest unary/ngram recipe has durable build/preflight evidence; fresh A/B/A performance validation remains pending.

## Runtime and evidence

The project moved through single-node bring-up, a CPU-NUMA integration, detached MTP support, and later Q6/Q8-MTP4 tuning. Earlier zero-acceptance sidecars and single-node conclusions are historical results for their own engine revisions. They are not current limits. [Earlier NUMA campaign](../../benchmarks/qwen38-flash-next-numa.md).

The retained Q6/Q8-MTP4 configuration measured 21.53 prose and 28.46 code tok/s. Corrected-gather experiments with HC/shared dispatch measured 24.04–24.35 and 30.80–31.62, with identical outputs and draft counts in the parent/corrected comparison. [Measurements and source layers](../../engineering/2026-09-08/README.md#retained-measurements) · [Gather report](../../engineering/2026-09-08/archive/serving/fleet-0903/BANDWIDTH250-20260909.md).

The MTP fresh-sequence reset passes eight repeated-request comparisons. The R8 projection and bounded scheduling experiments illustrate rejected candidates: component correctness alone did not produce a qualified serving improvement. [Request-state work](../../engineering/2026-09-08/archive/serving/fleet-0903/QWEN-MTP-STATE-TIMELINES-20260910.md) · [R8 result](../../engineering/2026-09-08/archive/serving/fleet-0903/QWEN-R8-PROJECTION-STATUS-20260910.md).

## Latest prepared recipe

The unary/ngram comparison records historical three-prompt means of 19.90 baseline, 20.32 unary, and 20.80 composite tok/s. The baseline relink is byte-identical, and the candidate launch recipe checks library order and assets. These measurements use a different prompt suite from the earlier prose/code table. [Recipe and rebuild evidence](../../engineering/2026-09-12/archive/serving/fleet-0912/qwen/README.md).

## Reproduction entry points

- [Host-specific baseline/unary/tuned profiles](../../engineering/2026-09-12/archive/serving/fleet-0912/qwen/profile.json).
- [Fresh-server comparison controller](../../engineering/2026-09-12/archive/serving/fleet-0912/AB-PROFILES.md).
- [Parameterized Q6/Q8-MTP4 profile](../../profiles/20260908/qwen-flash-next-q6-mtp4.json).
- [Base source, selected overlays, and gather correction order](../../engineering/2026-09-08/README.md#source-bundles-and-reconstruction).

The requested 40 tok/s target remains open. Further work needs fresh controls, explicit ngram warm-state handling, broader task quality, and measurements at useful context lengths.
