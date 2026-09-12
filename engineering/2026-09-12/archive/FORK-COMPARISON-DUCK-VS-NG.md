# llama-llama-duck (ours) vs llama.cpp-ng (mikechambers84) — 2026-08-30

## Verdict in one paragraph

For THIS box and THIS fleet, **ours is meaningfully ahead where it counts**: it has
measured, reproduced wins on the SR950 (direct F32 all-reduce +36.8%, per-device
thread pools, MTP+ngram speculative stack, documented per-arch results up to 16.49
tok/s general / 23.58 replay on Qwen3.8-27B), while ng has **no published benchmarks
yet** ("being prepared"). But ng is not nothing: it's the *same core idea* (NUMA
nodes as devices — their `--numa split` ≈ our `GGML_CPU_NUMA_DEVICES`), it **tracks
upstream master** (ours is pinned to an older base via the unsloth checkout), and it
carries two things worth stealing/testing: block-interleaved K-quant CPU kernels +
cache-line-aware matvec partitioning (could attack our ~22% bandwidth-extraction
problem on GLM), and MoE dispatch thread scheduling. Neither fork solves our actual
open gap by declaration: tensor-split for `qwen4exp`/`glm5next` — ours documents
that gap honestly ("silently corrupts output"; the current local build now refuses
cleanly), ng doesn't say which archs its split supports at all.

## Side by side

| | **llama-llama-duck (ours)** | **llama.cpp-ng (theirs)** |
|---|---|---|
| Core NUMA idea | CPU-NUMA device per node, strict mbind, per-device thread pools | `--numa split`: node = device, tensor-parallel across sockets |
| Base | pinned (b249 / unsloth lineage) | tracks upstream master, upstream-compatible |
| Published numbers | Qwen3.8-27B 11.72 dense / 16.49 MTP / 23.58 replay-warm; GLM-5.3 6.95; Flash-Next 8.98 (12 thr/node); 17.50 aggregate @ 8 concurrent | none yet |
| Speculative | MTP draft head + ngram-mod, per-request config, measured anti-correlation of acceptance vs throughput | not mentioned |
| Repack | NUMA-local repack, measured (dense 2.5–6.7×, MoE only 0.93–1.26×) | block-interleaved K-quant kernels (different angle, kernel-level) |
| Arch coverage for split | glm-dsa, qwen35 explicitly; qwen4exp/glm5next explicitly UNSUPPORTED | unspecified |
| Extras | SR950 server profiles, placement diagnostics, failure-case docs | GPU MMQ/RDNA2 work (irrelevant here), MoE dispatch scheduling |
| Maturity signal | production on this box, findings reproducible | 2 stars, benchmarks pending |

## What to actually do with ng (ranked)

1. **One experiment answers the interesting question**: build ng and run
   Qwen3.8-Flash-Next with `--numa split`. If their generic split works on qwen4exp
   (where our tensor-split refuses), it either beats our 8.98 NUMA-repack number or
   it doesn't — an afternoon, and it would close our biggest coverage gap for free.
   Same test for `glm5next` when relevant.
2. **Cherry-pick candidates**: their cache-line-aware matvec partitioning + MoE
   dispatch scheduling, benchmarked against our PGO build on GLM-5.3. Our measured
   problem is extraction efficiency (~22–31% of local bandwidth), which is exactly
   the layer they claim to improve.
3. **Do not migrate production** to ng on promise alone — ours has the spec stack
   (MTP+ngram is worth 2–3× on agentic replay) and they have no numbers.
4. Longer term, ours should **rebase or forward-port to current upstream** — ng
   proves tracking master is viable for this class of patch, and our pinned base
   will age.

## Context: tonight's fleet measurements (for the same scoreboard)

- GLM-5.3 UD-Q4_K_XL live: **6.76 tok/s** (numa-tensor, placement perfect at
  ~125GB/node, draft-mtp 79% acc). >12 on GLM needs byte reduction (Q3/Q2 tier) or
  kernel-efficiency wins — not more placement work.
- Qwen3.8-27B: server restored tonight on port 5810 with the 08-29 NUMA profile
  (doc'd 16.49 general / 12.3 novel-median). Probe during the V4-Flash sweep read
  9.4 — re-validate on an idle box.
- Qwen3.8-Flash-Next (qwen4exp): tensor-split refused by current build
  (`LLAMA_SPLIT_MODE_TENSOR not implemented`); single-node home blocked by GLM's
  residency; duck's own 8.98 recipe (12 thr/node NUMA-repack path) is the target to
  reproduce once memory policy is decided.
- DeepSeek-V4-Flash thread sweep (predicted ~12.1): running tonight, results land
  in TUNING-PLAN-12TOKS.md.
