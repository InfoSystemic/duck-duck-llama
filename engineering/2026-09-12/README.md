# CPU inference engineering through September 12, 2026

This snapshot publishes the GLM-5.3 Full/Flash, Qwen3.8-Flash-Next and DeepSeek-V4.1-Flash work after the [September 8-10 snapshot](../2026-09-08/README.md). It includes the September 11 tuning experiments and September 12 source fixes, foreground launch profiles, comparison controllers, and component validation. The original four-model optimization objective remains open.

## Results and implementation status

| Work | Evidence | Status |
| --- | --- | --- |
| GLM Flash parallel UNARY/SCALE | Historical MTP2 comparison: 15.39 to 15.71 decode tok/s; three greedy outputs identical | Durable library rebuilt; new launcher preflight passes; fresh full-model A/B pending |
| Qwen parallel UNARY/SCALE and ngram/MTP4 | Historical three-prompt means: 19.90 baseline, 20.32 unary, 20.80 composite | Durable library rebuilt, baseline relink byte-identical; new launcher preflight passes; fresh full-model A/B pending |
| GLM Full tensor split granularity | 57,888 metadata checks across all 1,809 tensors and 32 split schedules; all balanced splits unchanged | Source candidate, unapplied; full-model validation pending |
| RAM-cache NUMA rebalance | 4 MiB fixture: 3 MiB moved, four equal page populations, complete external SHA-256 unchanged | Utility built and fixture-tested; no bulk model migration performed |
| DeepSeek V4.1 Linux Engram prefetch | Actual original source fails alignment regression; patched source passes | Isolated JigSaw CPU engine and server compile and start; no checkpoint inference with this port |
| DeepSeek official compressor oracle | 332,928 values checked across ratios 1/2, prefill, decode and reset | Component reference for port validation; subsequent cache quantization is a separate requirement |
| DeepSeek native sparse prefill | 45 cases and 4,998,306 byte-exact BF16 outputs | Mixed component timings; unselected |

The historical speed numbers use their recorded prompts and measurement windows. They are not fresh September 12 results or interchangeable with end-to-end request throughput. DeepSeek-V4-Flash measurements in the September 11 work are labeled V4 and do not establish V4.1 performance. The older native V4.1 CPU endpoint reached 1.800 tok/s on short warm requests; complete checkpoint residency, broad quality, and practical long-context serving remain open.

Full's September 12 balanced load failed with a [kernel-confirmed memory-policy OOM on node 3](archive/serving/fleet-0912/results/glmfull-v5b-4097083/kernel-oom.txt). The staged Flash payload was unevenly distributed across RAM nodes. A second existing controller started another Full load. Neither a completed Full benchmark nor successful bulk NUMA redistribution is claimed by this snapshot. The earlier report's proposed unequal split also exposed a separate quantization-boundary assertion; the new split patch addresses that source defect and has metadata-level validation.

## Source and tuning entry points

- [September 11 measurements and corrections](archive/serving/fleet-0911/FLEET-EXTRACTION-20260911.md), [tuning overview](archive/serving/fleet-0911/README.md), and [bandwidth assessment](archive/serving/fleet-0911/GOAL-70PCT-20260911.md). These historical reports include failed experiments and superseded conclusions; the status table above states the publication's current claims.
- [Qwen foreground profile and rebuild](archive/serving/fleet-0912/qwen/README.md), [GLM Flash foreground profile and rebuild](archive/serving/fleet-0912/glmflash/README.md), and [GLM Full v5b profile](archive/serving/fleet-0912/glmfull/README.md).
- [Fresh-server A/B/A controller](archive/serving/fleet-0912/AB-PROFILES.md) and [real endpoint benchmark](archive/serving/fleet-0912/benchmark_live.py). They separate semantic checks from long-prompt timing, require exact process/listener identity, reject short or cached benchmark completions, and retain live processes when observation expires.
- [Full split correction](archive/serving/fleet-0912/glmfull-split/README.md), including original and candidate source, metadata fixtures, and validation.
- [NUMA page relocation utility](archive/serving/fleet-0912/numa/README.md), including build instructions and fixture evidence. It maps existing tmpfs files read-only and uses Linux page migration; it does not rewrite model bytes.
- [DeepSeek component fixtures and candidate](archive/serving/fleet-0912/deepseek/README.md), and [current upstream port assessment](archive/serving/fleet-0912/upstream/DEEPSEEK-V41-UPSTREAM-20260912.md).
- [Native DeepSeek context and bounded-cache candidate](archive/serving/fleet-0912/deepseek-context/README.md), with four component tests and explicit remaining model-validation and cache-recovery limits. It is unpromoted.
- [Complete engine source patches](source-bundles.json), [source/evidence inventory](archive-manifest.json), and [publication checks](validation.json).

The isolated JigSaw tree is pinned to `3b6fcfe4f7e2c282076f0c159278d3acfa3ad4e5`. Its local changes fix page alignment for `posix_madvise` and add a missing standard `<cmath>` include. The [CPU benchmark build](archive/serving/fleet-0912/upstream/build-result.json) and [server build](archive/serving/fleet-0912/upstream/server-build-result.json) are recorded separately. Passing a build does not establish native FP8 arithmetic, full-model quality, or long-context parity. The upstream assessment describes these remaining differences.

The `llama.cpp-dspark-dsv4` worktree has an unresolved Git index merge for `src/llama-kv-cache-dsv4.cpp`. The snapshot retains all three merge-stage blobs separately; its working file has no conflict markers. This branch remains archival work in progress. Its patch reconstruction check proves faithful export, not compilability. Other legacy DeepSeek V4 branches are preserved for source provenance; they are not V4.1 runtime selections.

## Reproduction and archive contract

Each engine patch names its exact base commit, changed-file hashes, reconstructed Git tree and available public remotes. Apply it to that base in an isolated checkout. Engine patches are alternative source trees, not a stack to apply together. The DSpark current patch includes eight locally committed forward-port changes from public upstream base `876a4321163249c43ca4e986818fab5ab081f282`, plus subsequent working changes. The Full split candidate is a separate overlay in its artifact directory and remains unselected.

The foreground profiles preserve this host's known paths, quantizations, library order and environment. Their rebuild scripts depend on the corresponding archived source layers and existing object-build recipes. They are auditable host recipes, not a claim that all dependencies are packaged as portable binaries. The Full hybrid draft and Flash Q4 payload have volatile RAM-backed dependencies that must be staged after reboot.

Source text and executable configuration JSON are preserved byte for byte. Evidence JSON omits unrelated process inventories, full environment dumps and model response text; large arrays are represented by a count and hash. The manifest distinguishes original and published hashes and lists malformed/incomplete evidence that could not be exported. Unchanged artifacts reference the preceding snapshot instead of being duplicated. Model weights, compiled binaries/libraries, raw performance traces and generated binary arrays are omitted; their generators and evidence hashes are retained.

```sh
python3 tools/verify_engineering_snapshot.py --snapshot engineering/2026-09-08
python3 tools/verify_engineering_snapshot.py --snapshot engineering/2026-09-12
python3 -m unittest discover -s tools -p 'test_quality_and_decode.py'
python3 -m unittest discover -s engineering/2026-09-12/archive/serving/fleet-0912 -p 'test_ab_profiles.py'
```

`tools/export_tuning_snapshot.py` records the source-selection and evidence-filtering policy. It neither launches models nor changes live serving selections. Upstream source retains its original licensing and notices; the engine bundle metadata and earlier [attribution](../../patches/ATTRIBUTION.md) record provenance.
