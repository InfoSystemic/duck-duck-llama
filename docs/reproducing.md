# Reproducing the work

There are three different reproduction tasks: verify the published artifacts, reconstruct a particular implementation, and repeat its model experiment. Start at the level you need.

## 1. Verify the repository

On Linux with Python 3.12 and Bash:

```bash
python3 tools/check_repository.py
python3 tools/search_archive.py "qwen get-rows" --limit 10
```

The repository check covers both dated inventories, references to unchanged prior artifacts, source/patch hashes, Python and shell syntax, curated documentation links, the generated catalog, and the portable probe/controller tests. No model weights, inference process, network request, or privileged hardware access is required.

The same scope runs in [GitHub Actions](../.github/workflows/verify.yml). The CPU-specific and model-level checks below are separate from that CI result.

## 2. Reconstruct the right source tree

The [latest bundle inventory](../engineering/2026-09-12/source-bundles.json) records public repositories, exact base commits, patch hashes, changed-source hashes, and reconstructed tree IDs. Each complete patch describes an alternative engine tree. Pick one model/source line; do not stack complete patches for different trees.

For example, the isolated DeepSeek port's source can be reconstructed with:

```bash
PROJECT_ROOT="$PWD"
git clone https://github.com/JigSawPT/llama.cpp.git /path/to/llama-deepseek-work
git -C /path/to/llama-deepseek-work checkout --detach 3b6fcfe4f7e2c282076f0c159278d3acfa3ad4e5
git -C /path/to/llama-deepseek-work apply --check "$PROJECT_ROOT/engineering/2026-09-12/patches/llama.cpp-deepseek41-jigsaw-0912.patch"
git -C /path/to/llama-deepseek-work apply "$PROJECT_ROOT/engineering/2026-09-12/patches/llama.cpp-deepseek41-jigsaw-0912.patch"
```

Use a new checkout for this example. Its [build evidence](../engineering/2026-09-12/archive/serving/fleet-0912/upstream/build-result.json) records the host configuration. It is a build-reproduction example, not a validated full-checkpoint serving recipe.

Some measured GLM/Qwen runtimes also used separately rebuilt private libraries. Their parent source, overlays, link recipes, and binary identities are archived. A base engine patch alone does not reproduce all selected libraries. Use the [September 8 layer order](../engineering/2026-09-08/README.md#source-bundles-and-reconstruction) or the later model-specific [Flash](../engineering/2026-09-12/archive/serving/fleet-0912/glmflash/README.md), [Qwen](../engineering/2026-09-12/archive/serving/fleet-0912/qwen/README.md), and [Full](../engineering/2026-09-12/archive/serving/fleet-0912/glmfull-split/README.md) recipes.

## 3. Preview a profile

The parameterized profiles render a quoted command without executing it:

```bash
python3 tools/render_profile.py profiles/20260908/qwen-flash-next-q6-mtp4.json \
  --set SERVER=/path/to/llama-server \
  --set MODEL=/path/to/qwen-shard-00001-of-000NN.gguf \
  --set DRAFT=/path/to/qwen-mtp.gguf \
  --set LIBRARY_PATH=/path/to/matching/runtime/libraries
```

The newer foreground launchers add asset, port, and library checks. They are host recipes and retain `/home/user/InfoSystemic/AI-Server`, `/models`, and `/dev/shm` paths. Archived scripts that derive sibling engine paths expect the original AI-Server directory layout. Restore/adapt that layout before invoking their build commands; the publication archive is not a drop-in installation tree.

Model shards, drafts, chat templates, runtime libraries, and NUMA topology must match the experiment. The RAM-backed Flash payload and Full hybrid draft require restaging after reboot. A missing or incompatible artifact should fail preflight before a server is started.

## 4. Repeat a model experiment

Read the model guide and original report first. Preserve the recorded source stack, precision, prompts, token budget, cache policy, and worker settings. Check output quality independently from throughput. Record actual loaded libraries and process identity, and compare against fresh controls.

Use the [tool guide](../tools/README.md) for probes and the [A/B/A controller](../engineering/2026-09-12/archive/serving/fleet-0912/AB-PROFILES.md) for the newest Flash/Qwen recipes. Its full-model comparisons require sufficient exclusive memory and free ports. Observation expiry does not mean the model process stopped.

Component fixtures that depend on Torch, official model files, CPU instruction sets, or host build products list those requirements beside their source. Their recorded results remain evidence at that scope. [Measurement methodology](methodology.md).
