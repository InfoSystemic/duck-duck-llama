# >12 tok/s on the SR950 — fleet scoreboard + plan (corrected 2026-08-30 evening)

*The first version of this file (earlier today) claimed the live GLM ran single-
socket on 16 threads — WRONG. `--threads 16` is per-NUMA-device in the fork (×4 =
64 cores) and placement is measured-perfect (~125GB/node ±1%). This version is the
truth. Companion docs: `serving/glm-sr950/GLM53-PERFORMANCE-INVESTIGATION.md`,
`serving/glm-sr950/NUMA-REPACK-DESIGN.md`, `serving/model-bundle-20260815/
QWEN38-27B-NUMA-20260829.md`, `FORK-COMPARISON-DUCK-VS-NG.md`.*

## Scoreboard (measured)

| Model | State | tok/s | vs 12 |
|---|---|---:|---|
| **Qwen3.8-27B** UD-Q8_K_XL | RESTORED tonight, port 5810, 08-29 NUMA profile | **16.49 general / 12.3 novel-median / 23.58 replay-warm** (08-29 doc; tonight's 9.4 probe ran during the V4-Flash sweep — re-validate idle) | **✅ MET** |
| **GLM-5.3** UD-Q4_K_XL | live, port 18091, numa-tensor | **6.76** (draft-mtp, 79% acc) | ❌ gap is bytes/kernels, NOT placement |
| Qwen3.8-**Flash-Next** (qwen4exp) | DOWN | duck-documented 8.98 @ 12thr/node (repack path); tensor-split refused: `not implemented for architecture 'qwen4exp'` (proven tonight) | ❌ needs fork work or the 8.98 recipe + memory-coexistence decision |
| DeepSeek-V4-Flash | thread sweep RUNNING tonight (predicted ~12.1, never tested) | *(results below when done)* | ? |

## Why GLM-5.3 is the hard one (byte math, from the investigation doc)

Decode is bandwidth-bound: tok/s = effective GB/s ÷ GB-read/token.
- Q4_K_XL reads **~34.8–38.6 GB/token** → even at the measured 365 GB/s
  NUMA-local aggregate, the ceiling is ~9.4–10.5. **>12 at Q4 is not physical.**
- Q3_K_XL (29.28 GB/tok) ceiling ≈ 12.5 — knife-edge; Q2_K_XL (21.68) ≈ 16.8.
- Measured extraction efficiency is only ~22–31% of local bandwidth → the real
  win is kernel efficiency (the 5.2-era repack work; llama.cpp-ng's K-quant/
  matvec ideas are candidates — see FORK-COMPARISON) plus spec-decode gains.
- Practical GLM paths to >12, in order: (1) speculative throughput tuning on the
  live Q4 (raise effective rate; ngram-mod is FORBIDDEN in composite on 5.3 —
  500s), realistically lands 8–10 not 12; (2) run the speed tier on **Q3/Q2
  quant** (download blocked by disk: /models 25G free — needs the disk cleanup
  first); (3) kernel-efficiency work in the fork.

## Tonight's session log

- 20:0x — measured live GLM 6.76 (96-tok probe, cache off).
- 20:4x — qwen4exp tensor-split attempt with the fresh /dev/shm repack build:
  clean refusal `LLAMA_SPLIT_MODE_TENSOR not implemented for architecture
  'qwen4exp'`. (Public duck README says attempting it on unsupported archs can
  silently corrupt — the local build now refuses instead. Good.)
- 20:5x — launched Sweep 1 (`serving/RUN-12TOKS-SWEEPS.sh 1`, V4-Flash × t∈{16,32,48,64}).
- 21:0x — restored Qwen3.8-27B on 5810 via `model-bundle-20260815/
  launch-qwen38-base.sh` (healthy in 66s alongside live GLM — coexistence proven
  08-29 and again now).
- Fork comparison written: `FORK-COMPARISON-DUCK-VS-NG.md` (verdict: ours ahead
  on results; test ng's generic `--numa split` on qwen4exp as the cheap path to
  Flash-Next coverage).

## Sweep 1 results (V4-Flash) — hypothesis REFUTED (measured 08-30 ~23:30Z)

- `RUN-12TOKS-SWEEPS.sh 1` itself has a silent-skip bug (wait_up failures print
  nothing; two runs produced zero bench lines) — configs were re-run by hand.
- **t=16, interleaved, quiet box: 2.67 tok/s** (150 tok, temp 0). That is
  ~15 GB/s effective — nowhere near the 69 GB/s the 12.1 prediction borrowed
  from Qwen. t=48/64 not pursued: the gap is architectural, not thread count.
- Conclusion: deepseek4 (V4-Flash) is stuck at ~2.7 until it gets the
  CPU-NUMA-device path. Same boat as the two below.

## The one fork project that unlocks everything (confirmed tonight)

`LLAMA_SPLIT_MODE_TENSOR not implemented for architecture 'X'` — clean refusal
reproduced for **qwen4exp** (Flash-Next) AND **glm5next** (GLM-5.3-Flash, tried
with the fresh /dev/shm/build-glm53flash-numa which DOES expose CPU-NUMA
devices). The tensor-split backend supports exactly glm-dsa + qwen35 today.
**Extending it to glm5next / qwen4exp / deepseek4 is THE kernel-efficiency
project** — and the 08-29 doc notes "generalized source work is prepared
locally but not committed" in the `llama.cpp-qwen4exp-numa` worktree. Start
there. Payoff estimate if glm5next lands: Flash at ~5.6 GB/token over the
NUMA-local path (365 GB/s × even 25% extraction ≈ 91 GB/s) ≈ **16+ tok/s** —
the family's >12 without touching full-GLM quality serving.

Interim alternatives measured/blocked tonight:
- Flash single-node (its documented profile): needs ~110GB free on one node —
  impossible while full-GLM holds ~125GB/node. Spilled run read 1.5 (invalid).
  → It's Flash-node-local OR full-GLM-resident: a serving-fleet choice.
- Op-profile capture on live GLM: ran, but the production binary emits no
  CPU_OP_PROFILE lines (profiler not in that build) — needs a
  profiling-enabled relaunch in a maintenance window. (Live server belongs to
  another session — do not bounce it casually.)
- numa_balancing already 0; hugepages already measured harmful for these
  artifacts (qwen 08-29 doc) — both Grok suggestions pre-applied/answered.

## Standing traps
- GLM-5.3 spec-type: `draft-mtp` ALONE (composite 500s, /health stays green).
- Qwen3.8-Flash-Next: **f16 KV only** (q8_0 KV asserts after weights load).
- Qwen thread counts are per-model: -t 64 collapsed 27B 3×; ALWAYS sweep.
- p_min/acceptance optimization is ANTI-correlated with throughput — optimize
  aggregate tok/s only.
- /dev/shm must stay light before any big load (628GB peak trap).
- Both disks ~full: / 97%, /models 99% — no new quants without cleanup.
