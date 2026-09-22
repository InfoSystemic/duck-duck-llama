# Runtime profiles

Profiles connect a model, quantization, source tree, runtime libraries, and launch settings. They capture measured configurations on the four-socket Xeon host; performance and memory requirements must be rechecked on another machine.

## Parameterized commands

The [September 8 profile set](20260908/README.md) replaces machine paths with explicit parameters. The renderer prints a shell-quoted command without executing it.

| Profile | Configuration |
| --- | --- |
| [Qwen Flash Next Q6 / MTP4](20260908/qwen-flash-next-q6-mtp4.json) | Q6 XL target, Q8 draft, four NUMA devices, 15 workers per socket |
| [GLM Flash Q4 / MTP2](20260908/glm-flash-q4-mtp2.json) | Q4 XL target and Q8 draft; the recorded selected Flash precision |
| [GLM Flash Q8 raw](20260908/glm-flash-q8-raw.json) | Q8 target without speculation; the raw-decode experimental stack |
| [GLM Full mixed precision / MTP2](20260908/glm-full-q4-mixed-mtp2.json) | Q4-based target with load-time requantization and a hybrid draft |

```bash
python3 tools/render_profile.py profiles/20260908/qwen-flash-next-q6-mtp4.json \
  --set SERVER=/path/to/llama-server \
  --set LIBRARY_PATH=/path/to/matching/runtime/libraries \
  --set MODEL=/path/to/qwen-shard-00001-of-000NN.gguf \
  --set DRAFT=/path/to/qwen-mtp.gguf
```

Run from the repository root. Each JSON profile declares its parameters and required source layers. The default bind address is `127.0.0.1`, port `8080`. Full also requires its chat template. See [reproduction](../docs/reproducing.md) for the engine reconstruction contract.

## Later launch and rebuild recipes

The September 12 recipes preserve the original host's paths, environment, CPU affinity, and library selection. They include foreground launchers, isolated candidate builds, and dry-run preflight checks.

| Model | Recipe and evidence | Current validation boundary |
| --- | --- | --- |
| GLM Flash | [Launcher and durable unary library](../engineering/2026-09-12/archive/serving/fleet-0912/glmflash/README.md) | Build/preflight complete; fresh full-model A/B/A pending |
| Qwen Flash Next | [Launcher, unary library, and ngram/MTP arms](../engineering/2026-09-12/archive/serving/fleet-0912/qwen/README.md) | Matched baseline relink and preflight complete; fresh A/B/A pending |
| GLM Full | [Launcher and load observations](../engineering/2026-09-12/archive/serving/fleet-0912/glmfull/README.md), [isolated split candidate](../engineering/2026-09-12/archive/serving/fleet-0912/glmfull-split/README.md) | Recorded loads fail on node 3; corrected unequal split has metadata/build evidence |
| DeepSeek V4.1 | [Native runtime record](../engineering/2026-09-08/archive/serving/fleet-0903/DEEPSEEK-V41-CPU-TUNING-20260910.md), [context/cache candidate](../engineering/2026-09-12/archive/serving/fleet-0912/deepseek-context/README.md) | Short warm requests measured; extended context remains unqualified |

These recipes need the matching AI-Server directory layout and build inputs. Archived binaries and volatile RAM-backed model payloads are not included. An isolated build or a passing dry run does not establish model quality, available memory, or a throughput gain.

The [comparison controller](../engineering/2026-09-12/archive/serving/fleet-0912/AB-PROFILES.md) documents fresh-process A/B/A runs and listener ownership checks. Parameterized commands alone do not manage concurrent model residency. The [model guides](../docs/models/README.md) are the starting point for deciding which source and precision settings to investigate.

## Earlier launchers

The four shell launchers in this directory and their [original guide](legacy-README.md) are retained for historical reproduction. Some use single-node workarounds or older MTP behavior that later source variants changed. Read them with their dated benchmark reports; use the guides above to understand the subsequent engineering.

## Portable recipe

[`glm-flash-any-sockets.json`](glm-flash-any-sockets.json) renders the GLM-5.3-Flash Q4 command for any socket count.
Its defaults describe a dual-socket ten-core host; `DEVICES`, `TENSOR_SPLIT`, `THREADS` and `CTX_SIZE` are parameters,
and it sets no `LD_LIBRARY_PATH` because a freshly built engine resolves its own libraries. It is a recipe, not a
measured result — every recorded figure in this repository is four-socket.
