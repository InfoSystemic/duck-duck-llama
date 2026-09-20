# Engineering snapshot, 2026-09-20

GLM-5.3-Flash decode work of September 18-20, measured the way a Codex agent inside Paseo experiences it. A curated subset, not a
full export: 374 files, no engine trees. Start with the [report](../../benchmarks/glm53-flash-paseo-decode-20260920.md) and the
[patch table](../../patches/README-20260920-glm5next.md).

| Area | Entry point |
| --- | --- |
| Chronology of the 09-19 campaign: baseline through Paseo, catalog fix, every window and its restoration | [GLM53-FLASH-CODEX-20260919.md](archive/sr950-strategy/GLM53-FLASH-CODEX-20260919.md) |
| 09-18 windows: MTP depth, fusion switches, first production op trace, decode against context, concurrency | [RESULTS-LOG.md](archive/serving/fleet-0912-ctx/RESULTS-LOG.md) |
| Fused pooling and flat state copy: sources, standalone suites, provenance | [glm-pool-copy-0919](archive/serving/fleet-0912-ctx/glm-pool-copy-0919/README.md) |
| Wider pool loop | [glm-kpool-wide-0919](archive/serving/fleet-0912-ctx/glm-kpool-wide-0919/README.md), [result](archive/sr950-strategy/codex-bench-0919/KPOOL-WIDE-RESULT.md) |
| Bounded KV rollback, including the exact reconstruction of the deployed libllama | [glm-kv-range-0919](archive/serving/fleet-0912-ctx/glm-kv-range-0919/README.md), [result](archive/sr950-strategy/codex-bench-0919/KV-RANGE-RESULT.md) |
| Cache-only MTP catch-up | [glm-mtp-kv-only-0919](archive/serving/fleet-0912-ctx/glm-mtp-kv-only-0919/README.md), [result](archive/sr950-strategy/codex-bench-0919/MTP-KV-RESULT.md) |
| Pooled-result cache | [design](archive/serving/fleet-0912-ctx/glm-pool-cache-r2-0920/DESIGN.md), [gates](archive/serving/fleet-0912-ctx/glm-pool-cache-r2-0920/README.md) |
| Negative results: Q4 batching, Q8 batching, dispatch and barrier probes, phase-accounting correction | [Q4](archive/sr950-strategy/codex-bench-0919/Q4-BATCH-WINDOW-RESULT.md), [Q8](archive/serving/fleet-0912-ctx/glm-q8-batch-0920/RESULT.md), [dispatch](archive/sr950-strategy/codex-bench-0919/DISPATCH-PROBE-RESULT.md), [phases](archive/sr950-strategy/codex-bench-0919/PHASE-AUDIT-RESULT.md) |
| What was judged worth doing next on 09-19 | [NEXT-PRIORITIES.md](archive/sr950-strategy/codex-bench-0919/NEXT-PRIORITIES.md) |
| Cell-split MQA attention, selection top-k, feature switch | [fa-mqa.inc](archive/serving/fleet-0920-flash18/cpu/fa-mqa.inc), [topk-fast.inc](archive/serving/fleet-0920-flash18/cpu/topk-fast.inc), [f18-common.inc](archive/serving/fleet-0920-flash18/cpu/f18-common.inc) |
| Window w1: controller, filtered report, log | [window1.py](archive/serving/fleet-0920-flash18/run/window1.py), [report](archive/serving/fleet-0920-flash18/results/window-w1.json), [log](archive/serving/fleet-0920-flash18/results/window-w1.log) |
| 8-layer proxy launcher, greedy/log-probability parity probe, op-trace helpers | [proxy.sh](archive/serving/fleet-0920-flash18/run/proxy.sh), [parity.py](archive/serving/fleet-0920-flash18/run/parity.py), [opsum.py](archive/serving/fleet-0920-flash18/run/opsum.py) |
| Cell-split MQA attention revision 2 (batch-invariant) and revision 1, gated delta net by state row | [fa-mqa.inc](archive/serving/fleet-0920-flash18/cpu/fa-mqa.inc), [fa-mqa.v1.inc](archive/serving/fleet-0920-flash18/cpu/fa-mqa.v1.inc), [invariance test](archive/serving/fleet-0920-flash18/cpu/test/test_fa_invariance.cpp), [gdn-rows.inc](archive/serving/fleet-0920-flash18/cpu/gdn-rows.inc) |
| MTP draft loop: merge, direct pick, constant-shape batches (generators over `speculative.cpp`) | [make_f18.py](archive/serving/fleet-0920-flash18/common/make_f18.py), [make_f18_pad.py](archive/serving/fleet-0920-flash18/common/make_f18_pad.py) |
| Coupled draft/verifier sampling and the sampler's host cost | [make_f18_couple.py](archive/serving/fleet-0920-flash18/common/make_f18_couple.py), [f18-coupling.inc](archive/serving/fleet-0920-flash18/common/f18-coupling.inc), [f18-topk-scan.inc](archive/serving/fleet-0920-flash18/common/f18-topk-scan.inc), [exactness test](archive/serving/fleet-0920-flash18/common/test/coupled_sampling_check.cpp), [server checks](archive/serving/fleet-0920-flash18/run/couple_check.py) |
| Two OpenMP teams per core: how it was found and the shared-team dispatcher | [phase_timeline.py](archive/serving/fleet-0920-flash18/run/phase_timeline.py), [fast_sampler_timing.py](archive/serving/fleet-0920-flash18/run/fast_sampler_timing.py), [make_dispatch.py](archive/serving/fleet-0920-flash18/cpu/make_dispatch.py) |
| MTP graph: query side only for the predicting row; libllama build with its lineage gate | [make_mtp_qrows.py](archive/serving/fleet-0920-flash18/llama/make_mtp_qrows.py), [build_glm5next.py](archive/serving/fleet-0920-flash18/llama/build_glm5next.py) |
| Negative result: multi-slot graph cache against the Meta backend | [make_ctx.py](archive/serving/fleet-0920-flash18/llama/make_ctx.py) |
| Windows w2-w6: controllers, filtered reports, logs | [w2](archive/serving/fleet-0920-flash18/results/window-w2.log), [w3](archive/serving/fleet-0920-flash18/results/window-w3.log), [w4](archive/serving/fleet-0920-flash18/results/window-w4.log), [w5](archive/serving/fleet-0920-flash18/results/window-w5.log), [w6](archive/serving/fleet-0920-flash18/results/window-w6.log), [window6.py](archive/serving/fleet-0920-flash18/run/window6.py) |
| Production revisions b, c, d: drop-ins, library hashes, deploy scripts with automatic rollback | [b](archive/serving/fleet-0920-flash18/deploy-0920b/95-f18-0920.conf.proposed), [c](archive/serving/fleet-0920-flash18/deploy-0920c/95-f18-0920.conf.proposed), [d](archive/serving/fleet-0920-flash18/deploy-0920d/95-f18-0920.conf.proposed), [deploy.sh](archive/serving/fleet-0920-flash18/deploy-0920d/deploy.sh) |
| Patch headers and the generator that cuts the patch files from the built sources | [make_patches_revd.py](archive/serving/fleet-0920-flash18/publish/make_patches_revd.py) |

## Contract

Same as the [archive contract](../README.md#archive-contract), applied by [assemble_snapshot.py](archive/serving/fleet-0920-flash18/publish/assemble_snapshot.py):
sources and notes byte for byte; evidence JSON through the exporter's filter plus a guard that turns long strings into length/hash
summaries; three notes that named a client are redacted and marked filtered; two routing scripts that describe a client service are
excluded and listed in the manifest. Per-request captures, process and environment snapshots, binaries and raw traces are not shipped.
Full modified copies of engine files above 150 KB are omitted; the [patches](../../patches/README-20260920-glm5next.md) carry those deltas
against parents published in the 09-12 snapshot.

```bash
python3 tools/verify_engineering_snapshot.py --snapshot engineering/2026-09-20
```

Two sessions produced this work: a Codex session on 09-19 (pooling, KV rollback, MTP catch-up, pooled-result cache, the negative
results) and a Claude session on 09-20 (attention kernel and its invariance fix, top-k, draft loop, coupled sampling, the OpenMP
team finding, gated delta net, MTP query rows, windows w1-w6, production revisions b-d, the FP16 accumulation finding, this publication).
