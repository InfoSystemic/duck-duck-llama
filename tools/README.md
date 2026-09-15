# Tools

The tools are grouped by the evidence they produce. Repository verification is portable; model probes need a compatible running server; hardware/kernel diagnostics need the relevant CPU and build environment.

## Explore and verify

```bash
python3 tools/check_repository.py
python3 tools/search_archive.py engram --kind notes --limit 10
python3 tools/search_archive.py "qwen get-rows" --kind evidence --limit 10
```

| Tool | Purpose |
| --- | --- |
| [check_repository.py](check_repository.py) | Run publication-integrity, documentation, catalog, syntax, and portable regression checks |
| [verify_engineering_snapshot.py](verify_engineering_snapshot.py) | Check one dated snapshot's inventory, inherited references, and patch hashes |
| [search_archive.py](search_archive.py) | Search the generated artifact catalog by path and kind |
| [build_catalog.py](build_catalog.py) | Regenerate or check the catalog from the published manifests |
| [export_tuning_snapshot.py](export_tuning_snapshot.py) | Host-side source/evidence exporter; uses the documented selection/filtering policy |

Catalog searches cover the latest published version of each logical archive path and the dated engine patches. Earlier versions remain in their snapshots and Git history.

## Probe an existing model server

```bash
python3 tools/quality_probe.py --host 127.0.0.1 --port 8080 --model YOUR_MODEL_ALIAS
python3 tools/decode_bench.py --host 127.0.0.1 --port 8080 \
  --model YOUR_MODEL_ALIAS --label baseline --workloads prose,code --reps 2 \
  --out /tmp/decode-baseline.json
```

[quality_probe.py](quality_probe.py) runs a small semantic smoke test. [decode_bench.py](decode_bench.py) records server decode timings, output hashes, finish information, and speculative metrics where available. These are basic probes; the compact answer validators are not a broad quality suite. Inspect failures and response completion rather than treating an aggregate timing alone as a passing result.

The newer [exact-process benchmark](../engineering/2026-09-12/archive/serving/fleet-0912/benchmark_live.py) and [A/B/A controller](../engineering/2026-09-12/archive/serving/fleet-0912/AB-PROFILES.md) add ownership, prompt-cache, completion-length, and comparison gates. Those recipes have their own host/runtime requirements.

## Inspect profiles and hardware behavior

| Tool | Scope |
| --- | --- |
| [render_profile.py](render_profile.py) | Render a parameterized profile without launching it; see [reproduction](../docs/reproducing.md#3-preview-a-profile) |
| [membw.c](membw.c) | AVX-512/OpenMP read-bandwidth diagnostic; build and placement examples are in its header |
| [barrier.c](barrier.c) | OpenMP barrier latency across thread counts and socket spans |
| [kbench.cpp](kbench.cpp), [kbench_id.cpp](kbench_id.cpp) | Dense and routed-expert component harnesses; preserve equal-work comparisons |
| [concurrent_bench.py](concurrent_bench.py) | Earlier concurrent-serving experiment harness |
| [gguf_types.py](gguf_types.py), [active_bytes.py](active_bytes.py) | Historical header/type and active-byte estimators; type tables and repack assumptions are revision-specific |

The older GGUF estimators are exploratory diagnostics, not authoritative parsers for every model or quantization. Use the model-specific metadata checks and pinned upstream readers when validating allocation or precision. Avoid using estimated active bytes times speculative tok/s as a bandwidth measurement.

## Measure distributions, not samples

```bash
./tools/rbench.sh 18083 prod 5            # repeat the 3-prompt decode bench, report median / min / max / spread
python3 tools/lctx.py 18083 prod 200 8000 32000 100000   # decode rate against CONTEXT length, not just short prompts
python3 tools/phase.py <server.log> <bench*.json>        # graph build vs allocation vs compute, against the wall per cycle
```

`rbench.sh` exists because a single run on a 4-socket host carries about 13% spread on prefill and can stall outright: one
outlying 8-slot point in this repository's own measurements produced a wrong conclusion about aggregate throughput saturating.
Quote a median and a spread. `lctx.py` exists because every decode figure here was taken at ~200 tokens of prompt, which
measures weights only; models that keep a growing KV cache on few layers degrade very differently from conventional ones.
`phase.py` parses the engine's `LLAMA_GRAPH_PHASE_ARM_FILE` output, which separates graph construction and scheduler
allocation from actual compute — the two are easily confused when a batch shape changes and forces a rebuild.
