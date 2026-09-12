# Research archive

The archive is the detailed engineering record behind the [case studies](../docs/README.md#engineering-case-studies). Source artifacts, runtime settings, numerical fixtures, benchmark controllers, measurements, failed experiments, and provenance are kept together.

## Snapshots

| Snapshot | Archived files | Unchanged prior references | Complete engine trees | Separate overlays |
| --- | ---: | ---: | ---: | ---: |
| [2026-09-08](2026-09-08/README.md) | 5,015 | 0 | 8 | 5 |
| [2026-09-12](2026-09-12/README.md) | 664 | 5,011 | 13 | 0 |

The September 8 snapshot includes its September 9–10 follow-ups. The September 12 snapshot adds September 11–12 engineering and references unchanged earlier files. Git history preserves previous publication revisions.

## Find an artifact

```bash
python3 tools/search_archive.py engram --kind notes
python3 tools/search_archive.py "qwen get-rows" --kind evidence
python3 tools/search_archive.py "glmfull split" --kind source
```

Run these commands from the repository root. The [generated catalog](catalog.json) indexes the latest published version of each logical archive path and the dated engine patches. Search is by artifact path; use `rg` inside the source tree for full-text investigation.

## Source and experiment map

| Area | Entry point |
| --- | --- |
| CPU-NUMA, tensor splitting, source integration | [Complete engine bundles](2026-09-12/source-bundles.json), [earlier layer order](2026-09-08/README.md#source-bundles-and-reconstruction) |
| GLM/Qwen scheduling, kernels, counters | [Earlier engineering map](2026-09-08/README.md#engineering-map), [bandwidth assessment](2026-09-08/archive/serving/fleet-0903/MODEL-250GBPS-20260909.md) |
| September 11 comparisons | [Extraction and corrections](2026-09-12/archive/serving/fleet-0911/FLEET-EXTRACTION-20260911.md) |
| Foreground profiles and library rebuilds | [Flash](2026-09-12/archive/serving/fleet-0912/glmflash/README.md), [Qwen](2026-09-12/archive/serving/fleet-0912/qwen/README.md), [Full](2026-09-12/archive/serving/fleet-0912/glmfull/README.md) |
| Benchmark and lifecycle controls | [A/B/A runner](2026-09-12/archive/serving/fleet-0912/AB-PROFILES.md) |
| Full split granularity | [Source, metadata checks, and candidate build](2026-09-12/archive/serving/fleet-0912/glmfull-split/README.md) |
| RAM-cache placement | [NUMA relocation utility](2026-09-12/archive/serving/fleet-0912/numa/README.md) |
| DeepSeek native CPU runtime | [CPU tuning](2026-09-08/archive/serving/fleet-0903/DEEPSEEK-V41-CPU-TUNING-20260910.md), [Engram lookup](2026-09-08/archive/serving/fleet-0903/DEEPSEEK-V41-ENGRAM-LOOKUP-20260910.md) |
| DeepSeek new components and context | [Compressor/sparse prefill](2026-09-12/archive/serving/fleet-0912/deepseek/README.md), [context/cache recovery](2026-09-12/archive/serving/fleet-0912/deepseek-context/README.md) |
| DeepSeek upstream port | [Assessment, fixes, and build evidence](2026-09-12/archive/serving/fleet-0912/upstream/DEEPSEEK-V41-UPSTREAM-20260912.md) |

## Archive contract

Source text and executable configuration are retained byte for byte with their original and published SHA-256 values. Evidence JSON is filtered to remove unrelated process/host inventories, full environment dumps, and model response text. Oversized values may become count/hash summaries; incomplete JSON is listed as excluded evidence. The manifest records the distinction.

Model weights, compiled libraries/binaries, raw performance traces, and generated binary arrays are not shipped. Historical reports can refer to original host paths or excluded raw artifacts. The curated guides link the published evidence needed to assess their claims.

Full source patches are alternatives, not one patch stack. Separate overlays have an explicit order where needed. Reconstruction verifies the exported tree; it does not certify an unresolved branch or unselected candidate as a working model runtime.

```bash
python3 tools/verify_engineering_snapshot.py --snapshot engineering/2026-09-08
python3 tools/verify_engineering_snapshot.py --snapshot engineering/2026-09-12
```

See [reproduction](../docs/reproducing.md) and [provenance](../docs/provenance.md) before rebuilding a measured configuration.
