# Engineering snapshot, 2026-09-20

GLM-5.3-Flash decode work of September 18-20, measured the way a Codex agent inside Paseo experiences it. A curated subset, not a
full export: 319 files, no engine trees. Start with the [report](../../benchmarks/glm53-flash-paseo-decode-20260920.md) and the
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
| Draft-loop restructure for the MTP driver: **written, built, never run** | [make_f18.py](archive/serving/fleet-0920-flash18/common/make_f18.py) |

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
results) and a Claude session on 09-20 (attention kernel, top-k, window w1, the FP16 accumulation finding, this publication).
