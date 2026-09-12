# fleet-0911 — GLM-5.3 / GLM-5.3-Flash / Qwen3.8-Flash-Next / DeepSeek bandwidth work, 2026-09-11

## Read these first

| file | what |
|---|---|
| `GOAL-70PCT-20260911.md` | the answer to "70% of 380 GB/s on all four models": scoreboard, why it is unreachable, recommendations |
| `FLEET-EXTRACTION-20260911.md` | the full log, 42 sections — every measurement, every negative result, every correction |

## Headline

The "380 GB/s" denominator had never been measured. It is **381.6 GB/s** (node-local streaming
read, 4 sockets) — so the figure was right, and 70% = 266 GB/s is a real target.

| model | best | note |
|---|---:|---|
| GLM-5.3-Flash | **63.4%** | spec off + both fixes |
| DeepSeek-V4-Flash | 51.0% | raw TP4 |
| Qwen3.8-Flash-Next | **49.0%** | raw + both fixes |
| GLM-5.3 Full | — | not loadable while the 186 GB Flash tmpfs is resident |
| DeepSeek-V4.1 | — | needs 306 GB of disk + cross-layer-KV work |

**Nothing reaches 70%, and it is not a tuning gap:** the quantised GEMV kernels run at **92% of a
bare AVX-512 read loop**, bytes/token is **within 1% of the GGUF-derived ideal**, NUMA balance is
exact (43.3/43.8/42.8/43.4 GB/s), TLB is 1.2%.

## Shipped (all opt-in, all bit-exact, all with byte-identical baseline verification)

| dir | runtime | models | gain |
|---|---|---|---|
| `parallel-unary-0911/` | glm5n-goal (GLM, DeepSeek) | GLM-5.3-Flash | **+2.1%** |
| `parallel-unary-q4e-0911/` | q4e-goal | Qwen3.8-Flash-Next | **+3.4%** |

Enable with `GGML_CPU_PARALLEL_UNARY=4096` and the build dir prepended to the pinned
`LD_LIBRARY_PATH`. **Only helps models with a linear-attention/SSM state path** — measured 0% on
DeepSeek-V4-Flash, which shares GLM's runtime but has no SSM path. Each dir has `rebuild.sh`
(self-contained, verified) and a `manifest.json` recording every sha256 and verification.

## Harnesses

| file | what |
|---|---|
| `bench2.sh` | 3-prompt tok/s + IMC GB/s, robust longest-contiguous-run window |
| `benchpar.sh` | N concurrent streams, aggregate throughput |
| `scoreboard.py` | recomputes every result from the raw IMC CSVs |
| `parallel-unary-*/mmbench.c` | isolates one `ggml_mul_mat` shape vs thread count |
| `/dev/shm/bwprobe/` | `membw.c` (ceiling), `bw.sh` (IMC series), `bwsum2.py`, `barbench.c`, `gemvbw.c`, `quiet.sh` |

`restore-baseline-0911.sh` restores the production GLM-5.3-Flash server exactly (argv + all 49
env vars incl. `LD_LIBRARY_PATH`). `finish.sh` wraps it and scrubs any non-pinned `GGML_*` var.

## Traps that cost real time (details in the log)

1. A pinned config is **binary AND env AND flags** — substituting one is *silent* (2.4x slow, or `////`).
2. The production library links objects from **four private builds**; resolve the layer **per file**.
3. `free -g` cannot see a NUMA-node-constrained OOM. Use `numastat -m`.
4. Mount RAM payload volumes `mpol=interleave` — first-touch skew is invisible until two models collide.
5. Never kill llama-servers by name; other sessions serve on other ports.
6. `gcc` picks the language from the **extension** — never `foo.c.patched`.
7. Re-measure the control **last**; this box drifts ~5% over a session.
