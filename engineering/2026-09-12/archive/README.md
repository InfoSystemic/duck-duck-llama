# AI-Server — the local inference stack

Consolidated 2026-08-08 from roughly 200 loose files that had accumulated at the
top level of `~/`. Nothing was deleted; everything here was moved from `~/`.

## Layout

| Path | What it holds |
|---|---|
| `tuning/glm52/` | GLM-5.2. Split into `bench-runs/` (43 — the `glm-q4km-bench*.sh` series), `launch/` (9), `config/` (3), `results/` (3). `GLM52-TUNING.md` and the three sweep/parallel scripts stay at its root. |
| `tuning/deepseek-v4/` | DeepSeek-V4-Flash tuning + the `dsv4r*` run logs |
| `tuning/kimi-k3/` | Kimi K3 wildspeed work + the `k3r*` run logs |
| `tuning/gemma/` | Gemma setup and warmup probes |
| `benchmarks/` | Cross-model work, split into `runs/` (21 scripts — `exp*.sh`, `run_exp.sh`, the rebuild/smoke helpers) and `results/` (67 — the `*-gen/prime/warmup.json` sets, `*.out`, `r1-10.log`) |
| `serving/` | `llama-swap` configs, litellm proxy config and venv |
| `engines/` | llama.cpp / ik_llama forks (was `~/src/`) plus the `server-context.*.cpp` patches |
| `models/` | `whisper-models/` |
| `host-setup/` | `build-xrdp/` — xrdp/xorgxrdp source builds for this workstation |
| `scripts/` | Utilities, plus the three reorg scripts described below |

`tuning/glm52/` and `benchmarks/` were flat piles of 62 and 88 files after the first
pass — the same "can't find anything" problem one level down. `scripts/subdivide-and-fix.sh`
split them and rewrote every absolute reference that pointed at a moved file.

## Where the model weights actually are

Bulk weights were moved off the root filesystem (it was at 92%) and replaced with
**symlinks**, so every path in these scripts still resolves:

- `tuning/deepseek-v4/deepseek-v4-tuning/models/*.gguf` → `/data/deepseek-v4-tuning/` (the two ~140 GB layer-split variants) and `/models/deepseek-v4-tuning/` (draft quants)
- `tuning/kimi-k3/k3-wildspeed/models/*.gguf` → `/models/kimi-k3/`
- `~/gguf` → `/models/gguf` (unchanged; the `llama-server` systemd unit depends on this path)

`scripts/relocate-model-weights.sh` performed this and is re-runnable — it skips
anything already symlinked, verifies size before unlinking a source, and logs to
`scripts/relocate-model-weights.log`.

### Open question: ~570 GB of derived quants

There are four large layer-split variants of DeepSeek-V4-Flash across `/data` and
`/models` (`L27-28-37-38`, `L28`, `L33-42`, `L34-42`), each ~140 GB. They are all
**derived** from `/models/gguf/DeepSeek-V4-Flash-0731-UD-Q4_K_XL`, so they are
regenerable. Presumably only one is the winner. Deleting the superseded three would
reclaim ~420 GB — but that is a judgment call about which experiment won, so nothing
was deleted.

## Absolute paths inside these scripts

Many of these scripts hardcoded `/home/user/<name>`. `scripts/fix-moved-paths.sh`
rewrote 59 files to the new locations. It is idempotent and safe to re-run; it only
rewrites a reference when the old path no longer exists *and* the new location is
unambiguous, so it will never guess.

A handful of references remain dangling — `~/k3gguf/`, `~/gguf/GLM-5.2-*`,
`~/bench_glm.sh`, `~/k27-download.log`. These were **already broken before the
reorg** (the models and scripts were deleted earlier); they are not move damage.
