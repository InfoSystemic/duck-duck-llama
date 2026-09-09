# GLM-5.3 (non-Flash) SR950 throughput investigation

Working notes, 2026-08-28. Goal: get GLM-5.3 Full decoding at **>= 5 tok/s**,
with >= 10 tok/s as the honest hardware-justified target (see "Ceiling" below).

Companion to `NUMA-REPACK-DESIGN.md`, `KERNEL-CALIBRATION.md`, `GLM53-RUNBOOK.md`.

> **LEFT IN A NON-DEFAULT STATE — read "Current state" before doing anything else.**
> `qwen38-dedicated.service` is STOPPED and needs restarting.
> `glm53-sr950.service` is STOPPED (failed) after two OOM kills.

---

## 1. Baseline (measured, not estimated)

The staged GLM-5.3 service as found, pid 3987105, running since 11:44:

    predicted_per_second : 1.0185 tok/s      <-- decode
    prompt_per_second    : 3.8565 tok/s
    n_predict            : 24, temperature 0, cache_prompt false

Saved to `/tmp/glm53-baseline.json`. Output was coherent, so the staged config
is *correct*, just slow.

## 2. Hardware facts

    Lenovo SR950 "smeagol"
    4x Intel Xeon Gold 6242 (Cascade Lake), 16C/32T each = 64C / 128T
    4 NUMA nodes, ~193 GB each, 755 GiB total
    AVX-512 + AVX512_VNNI.  NO AMX (Cascade Lake predates it).  No GPU.
    node distances: 10 local / 21 remote, uniform (fully connected)

DRAM: 6ch DDR4-2933 per socket = 140.8 GB/s/socket theoretical, 563 GB/s
aggregate. Realistic STREAM ~420 GB/s. `NUMA-REPACK-DESIGN.md` measured
**365 GB/s** local with socket-local shards.

Storage is effectively full: `/` 25 GB free (98%), `/models` 64 GB free (97%).
This constrains everything below.

## 3. Model facts

    general.architecture      glm-dsa          <-- same family as GLM-5.2
    params                    753,864,139,008
    file bytes                342,956,430,336  (319.4 GiB, 9 shards)
    ftype                     Q3_K - Medium (UD-Q3_K_XL)
    block_count               79   (blk.78 is the NextN/MTP block, unused)
    embedding_length          6144
    expert_count              256, expert_used_count 8, expert_shared_count 1
    expert_feed_forward_length 2048, feed_forward_length 12288
    context_length            1048576

Architecture is `glm-dsa`, identical to the GLM-5.2 currently in production, so
the fork's `LLAMA_SPLIT_MODE_TENSOR` support applies. No loader changes needed.

## 4. Root cause of the 1.02 tok/s

Two independent faults, both configuration.

### 4a. The staged config has every optimisation switched off

`model.glm53-staging.env` is a safe bring-up profile, not a performance one:

    GLM_BACKEND_MODE=cpu-repack     (not numa-tensor)
    GGML_CPU_NUMA_DEVICES=0         (NUMA backend OFF)
    GGML_GLM_ATTN_TP=0
    GGML_CPU_NUMA_DIRECT_ALLREDUCE=0
    GGML_CPU_NUMA_REPACK=0, IQ2/IQ3/Q5_K repacks all 0
    GLM_SPEC_TYPE=none              (no speculative decoding)
    GLM_THREADS=64                  (vs 16/device in production)

`NUMA-REPACK-DESIGN.md` already measured what these are worth:

| config | measured |
| --- | ---: |
| `--device CPU-NUMA0..3 --split-mode tensor` | 6.551 tok/s replay |
| plain CPU + repack, interleaved | 2.001 tok/s replay |

### 4b. The model's tmpfs pages are stranded on two of four nodes

`/proc/3987105/numa_maps`, aggregated:

    N0:   8.0 GB
    N1:  10.5 GB
    N2: 161.6 GB   <-- 94% of the model is on nodes 2+3
    N3: 136.7 GB

The downloader ran on nodes 2/3, and tmpfs page placement is fixed at *write*
time. `numactl --interleave=all` in the staged launcher is a **no-op for
weights**, because it only affects new allocations, not pages that already
exist. So half the cores read ~94% of every token across UPI.

Confirming it is memory stall and not a compute shortfall — during generation:

    process CPU: 5873%  (59 of 64 cores pegged)
    node0 54.2%  node1 56.7%  node2 55.6%  node3 51.6%   (of 32 logical each)

Threads are running and evenly spread; they are spinning on stalled remote
reads. `VmSwap: 0`, so swap is not involved.

## 5. Byte accounting — what actually sets the ceiling

Decode at batch 1 is bandwidth-bound, so tok/s = bandwidth / bytes-per-token.
Computed from the real tensor sizes the loader printed for `blk.78`:

    per-layer dense attention   175.3 MB   <-- 47% of the total
    per-layer routed (8 of 256) 143.1 MB
    per-layer shared+idx+gate    56.9 MB
    per-layer TOTAL             375.4 MB
    x 78 layers               29.28 GB read per token

`attn_output` alone is 107 MB/layer (~100.7M params at Q8-ish). The dense
attention costs *more* per token than the routed experts do.

### Ceiling

    420 GB/s / 29.28 GB = 14.3 tok/s   theoretical
    365 GB/s / 29.28 GB = 12.5 tok/s   at the measured local bandwidth

    5 tok/s  needs 146 GB/s  (35% of 420)
    10 tok/s needs 293 GB/s  (70% of 420)

### Where the efficiency actually goes

The prior GLM-5.2 UD-Q2_K_XL control was 3.78 tok/s raw decode at 2.694 bpw,
i.e. 21.68 GB/token, i.e. **~82 GB/s effective = 22% of 365 GB/s**. That 22% is
the real problem. 10 tok/s means roughly tripling extraction efficiency, not
finding more bandwidth.

Scaling that control to this quant (29.28 / 21.68 = 1.35x the bytes) predicts
**~2.8 tok/s** for GLM-5.3 Q3_K_XL on the numa-tensor path alone. Fixing
placement is necessary but NOT sufficient for either target.

## 6. Why not Q4 (asked and answered)

Three separate answers:

1. **Nothing was chosen.** UD-Q3_K_XL is the only GLM-5.3 on the box. A Q4
   download is ~420 GB against 64 GB free on `/models`.
2. **There is no spare RAM right now**, for a non-obvious reason: the model
   lives in `/dev/shm`, which *is* RAM (319.4 GiB), and numa-tensor allocates a
   **second** node-bound copy. The model is billed twice, ~640 GB of 755, before
   KV cache, activations, or the 40 GB Qwen server.
3. **For a speed goal Q4 is the wrong direction.**

   | quant | GB read/token | GB/s for 5 tok/s | GB/s for 10 tok/s |
   | --- | ---: | ---: | ---: |
   | Q2_K_XL (5.2 today) | 21.68 | 108 | 217 |
   | **Q3_K_XL (5.3 now)** | **29.28** | **146** | **293** |
   | Q4_K_XL | 34.8 - 38.6 | 174 - 193 | 348 - 386 |

   The 34.8 figure assumes only the expert side grows (Unsloth UD keeps
   attention high-precision in both tiers); 38.6 assumes uniform scaling.

   Honest counterargument: Q4_K/Q4_0 blocks dequantise far more cheaply than the
   **IQ3_XXS** format that dominates this file (148 of the expert tensors), and
   we are only extracting 22% of available bandwidth — so we are *partly*
   kernel-bound, and Q4 could recover some efficiency. Supporting datapoint from
   `NUMA-REPACK-DESIGN.md`: Qwen3.8-27B **Q4_0** hit 7.58 tok/s socket-local vs
   4.33 interleaved, ~31% extraction — better than the 22% seen on IQ2/IQ3.

   Verdict: a gamble on kernel efficiency against a certain 19-32% byte penalty.
   Hold Q3 for the throughput goal; treat Q4 as a *quality* decision, and only
   after disk is freed.

## 7. The OOM — why the first numa-tensor cutover failed

Cutting over to `numa-tensor` OOM-killed twice, ~40 s and ~50 s into load:

    Aug 28 17:02:17  glm53-sr950.service: A process of this unit has been killed by the OOM killer.
    Aug 28 17:03:29  Main process exited, code=killed, status=9/KILL

Not an aggregate-RAM problem — a *placement* problem. `--tensor-split 1,1,1,1`
needs ~80 GB of node-bound memory on **each** node. At the time:

    node 0 free: 136 GB    node 2 free: 13 GB   <-- cannot supply 80 GB
    node 1 free: 145 GB    node 3 free: 21 GB   <-- cannot supply 80 GB

The `mbind` allocation on nodes 2/3 fails because the stranded tmpfs pages from
section 4b occupy exactly those nodes. **Section 4b and section 7 are the same
bug**: the download's page placement both starves bandwidth and blocks the fix.

Ignore these two warnings, they are spurious for this fork's CPU-NUMA devices
(emitted by common.cpp whenever `llama_supports_gpu_offload()` is false):

    warning: llama.cpp was compiled without support for GPU offload. Setting the split mode has no effect.
    warning: ... Setting a tensor split has no effect.

## 8. Plan

Ordered by expected value. Steps 1-2 are prerequisites for everything else.

1. **Rebalance the tmpfs pages across all 4 nodes.** Script written to
   `/tmp/rebalance-glm53.sh` (NOT yet run). Rewrites each shard via
   `numactl --interleave=all cp`, one at a time (59 GB headroom, largest shard
   48.8 GB), verifying size before removing the original. Non-destructive.
   Result: ~85.7 GB of model per node, ~107 GB free per node, which unblocks the
   80 GB/node tensor-split allocation.
2. **Restart in numa-tensor mode** using `model.glm53.env` (already written).
   Expected ~2.8-3.5 tok/s from the scaling in section 5. Measure, do not assume.
3. **Get the model out of tmpfs onto disk.** This is the highest-leverage
   *structural* fix: it frees 319 GiB, ends the double billing, and is what makes
   step 4 affordable. Needs ~343 GB of disk that does not currently exist —
   **requires a decision from Kaden about what on `/models` is disposable**
   (deepseek-v4-tuning 596G, gguf 317G, awesomo-archive 309G, archive 197G).
   The runbook says do not delete GLM-5.2 first.
4. **Re-enable the VNNI repack path** (`GGML_CPU_NUMA_REPACK=1` + IQ2/IQ3/Q5_K).
   This is what `NUMA-REPACK-DESIGN.md` was built for, and it is currently off
   *only* because it grows the node-bound copy to 381.8 GiB
   (`projected_weight_bytes`) and will not fit alongside tmpfs. Blocked on step 3.
   Note this quant is IQ3_XXS/IQ4_XS-mixed and **there is no IQ4_XS repack gate**,
   so 71 expert tensors get no benefit — the win will be partial.
5. **Raise `GGML_CPU_MOE_SINGLE_TOKEN_THREADS` from 8.** It was tuned to
   minimise single-token latency on 5.2, but 8 workers/node may not generate
   enough outstanding cache misses to saturate a memory controller. Cheap sweep:
   8 / 12 / 16.
6. **MTP speculative decoding — the one lever that beats the bandwidth wall.**
   GLM-5.3 ships its NextN block (`blk.78.nextn.*`, seen loading and being
   ignored). Verifying k tokens per weight-read pass multiplies effective tok/s
   by the acceptance rate, typically 2-2.5x. This is how the 5.2 production
   config reached 6.3-7.5 tok/s replay off a 3.78 tok/s raw decode.
   `extract-glm-mtp-gguf.py` exists for exactly this.
   **The checked-in `GLM_MTP_MODEL` is GLM-5.2's and must never be paired with a
   5.3 target** — a 5.3 MTP has to be extracted first.

Rough path to 10 tok/s: ~3 tok/s (placement) -> ~4-5 (repack + thread sweep)
-> ~8-12 (MTP). Steps 4 and 6 are where the target is actually won.

## 9. Current state / what was changed

Created:
- `serving/glm-sr950/model.glm53.env` — GLM-5.3 numa-tensor production profile,
  derived from the 5.2 `model.env`. Repacks off pending step 3. Spec off pending
  a 5.3 MTP extraction.
- `~/.config/systemd/user/glm53-sr950.service.d/10-numa-tensor.conf` — points
  `GLM_SR950_CONFIG` at the above.
- `/tmp/rebalance-glm53.sh` — written, **not run**.
- `/tmp/glm53-baseline.json` — the 1.02 tok/s control.

Service state:
- `glm53-sr950.service` — **stopped/failed** after the two OOM kills.
- `qwen38-dedicated.service` — **STOPPED BY ME** to free 40 GB for the cutover.
  **Restart it:** `systemctl --user start qwen38-dedicated.service`

Untouched: `model.env` (still the GLM-5.2 production profile),
`model.glm53-staging.env` (the 1.02 tok/s bring-up config).

### Rollback

    rm ~/.config/systemd/user/glm53-sr950.service.d/10-numa-tensor.conf
    systemctl --user daemon-reload
    systemctl --user start glm53-sr950.service   # back to the staged 1.02 tok/s config
    systemctl --user start qwen38-dedicated.service

## 10. Open questions

- Does `--split-mode tensor` behave on this quant's mixed IQ3_XXS/IQ4_XS/Q6_K
  layout? The 5.2 validation was on an IQ2_XS/IQ3_XXS-dominated file.
- `GGML_GLM_ATTN_TP=1` was validated on 5.2's attention geometry. Same arch, but
  unverified on 5.3. If output is wrong, this is the first gate to drop.
- What on `/models` is disposable? Blocks steps 3 and 4, and therefore the
  10 tok/s target.
- Is the 22% extraction efficiency dominated by IQ3_XXS LUT dequant, or by
  cross-socket sync in the tensor-parallel all-reduce? `capture-decode-op-profile.sh`
  and the dormant `GGML_CPU_OP_PROFILE` machinery would answer this directly.

## 11. 2026-08-29 — Q4_K_XL cutover measurements (supersedes section 6 for speed)

The service is now on `/models/GLM-5.3-GGUF/UD-Q4_K_XL` (disk, not tmpfs),
numa-tensor, repacks on (Q5_K/Q8_0 + generic Q4_K path), MTP draft
`/dev/shm/GLM-5.3-MTP-HYBRID-Q4L-Q6H.gguf`. Live measurements (general
harness, seed 42, 128 tokens, co-tenant noise ~±6%):

| arm | tok/s | draft acc |
| --- | ---: | ---: |
| raw (`speculative.n_max=0`) | **4.62** | - |
| production default n64/p0.8 | 5.51 | 83% |
| n64/p0.9 (agentic alias) | 4.25 | 35% |
| n32/p0.8 | 5.22 | 86% |
| **n2/p0** | **6.27-6.69** | 63% |
| n3/p0 | 5.26 | 42% |
| n4/p0 | 4.44 | 41% |
| n6/p0 | 5.01 | 39% |
| n8/p0 | 4.38 | 31% |
| n12/p0 | 2.95 | 20% |
| n16/p0.5 | 4.94 | 44% |

Conclusions:

1. **Section 6's verdict is reversed by measurement: Q4_K_XL is the fastest
   tier on this box.** Raw 4.62 tok/s x ~36 GB/token = ~166 GB/s effective =
   **45% of the 365 GB/s**, double the 22% IQ-era extraction. Q4_K VNNI repack
   kernels are far more efficient per byte than the compact IQ2/IQ3 path
   (5.2 Q2 raw was 3.99 at ~20 GB/token = ~80 GB/s). Down-quanting to Q2/Q3
   would LOWER raw decode. Hold Q4 for both quality and speed.
2. Spec multiplier is the only remaining lever to 10 tok/s: currently 1.45x
   best (n2/p0), needs ~2.2x. Long drafts are counterproductive — draft-model
   passes are expensive relative to the target pass. The 5.2 composite profile
   (n64/p0.8, agentic p0.9) does NOT transfer to 5.3.
3. Draft cost per token is the multiplier bottleneck. Untried: single-node
   draft placement (draft is 8 GB, fits one socket) or a smaller-quant draft
   head to halve draft-pass bytes.
4. `model.glm53-q4.env` defaults updated to n2/p0 (general). The agentic 0.9
   alias is commented out pending a 5.3 replay-suite derivation.
5. Numbers above are generic-prompt. Warm agentic replay with ngram history
   measured +19% on 5.2 (6.31 cold -> 7.50 warm); expect a similar lift here
   on top of n2/p0.

## 12. 2026-08-29 — replay suite on n2/p0, correctness resolved

`benchmark-replay.sh` on the live n2/p0 config:

- 400-token budget: 7.42 cold / 7.02 warm aggregate, but 5/6 checks "failed"
  with `content_chars: 0`, `finish_reason: length`. **Not a model or spec
  fault**: the harness sends `reasoning_effort:"none"`, and the 5.3 chat
  template maps anything outside `['low','high']` to **Max** (line 2 of
  `chat-template-glm-5.3-llamacpp.jinja`). The model reasoned past the cap
  without emitting content.
- 1400-token budget: **3/3 correct, mean 7.16 tok/s, acceptance 84.8%**.
  p_min=0 acceptance is exact on this suite; the n2/p0 defaults are validated.
- Workload 3 with `reasoning_effort:"low"`: 194 tokens (19 reasoning chars),
  correct, **8.29 tok/s** — best replay number so far.
- Harnesses patched to send `"low"`: `benchmark-replay.sh`,
  `benchmark-concurrent.sh`, `capture-decode-op-profile.sh`.
  `start-glm53-staged.sh` already used `"low"`. Template left untouched
  (production clients may rely on the Max default).
- Note for cross-model comparisons: 5.2-era controls were run with `"none"`;
  on this template that means Max reasoning, so token *counts* are not
  comparable across template versions even when tok/s is.

## 13. 2026-08-29 — Full Q4 exceeds 12 tok/s

The final target is GLM-5.3 **Full, non-Flash** UD-Q4_K_XL. All 11 published
SHA-256 hashes passed before the obsolete
`/dev/shm/ai-models/GLM-5.3-GGUF/UD-Q3_K_XL` copy was deleted. The target
remains on disk; the active MTP draft is the Q4-output hybrid
`/dev/shm/GLM-5.3-MTP-HYBRID-Q4L-Q6H-OUTQ4.gguf`.

Two plausible microbenchmark wins were rejected by full-service measurement:

- Original-layout Q5_K MoE-down measured faster in isolation at n=16/17, but
  reduced the n16/p0.7 replay from 11.47 to 10.55 tok/s. Production therefore
  keeps `GGML_CPU_Q5_K_REPACK_MOE_DOWN=1`.
- A shape-aware Q8_0 tile-2 rule was noisy and did not survive service A/B.
  Production retains the prior `GGML_CPU_Q8_0_REPACK_X_TILE=auto` behavior
  (tile 8 for packed GEMM).

The winning request-scoped coding profile is:

```json
{
  "model": "glm-sr950-agentic",
  "speculative.n_max": 18,
  "speculative.p_min": 0.75
}
```

The live server still defaults unspecified requests to n2/p0, the
general-workload winner. The startup cap is 32, so the agentic profile needs no
restart or speculative-buffer resize.

Clean replay results on the restored expanded-Q5/original-Q8 process:

| File | Output | Aggregate tok/s | Acceptance | Correct |
| --- | ---: | ---: | ---: | ---: |
| `/tmp/glm53-restored-n18-p075-clean.jsonl` | 577 | 13.308 | 93.79% | 3/3 |
| `/tmp/glm53-restored-n18-p075-confirm.jsonl` | 565 | 12.546 | 92.35% | 3/3 |
| **combined** | **1,142** | **12.920** | **93.07%** | **6/6** |

Several discarded runs overlapped
`Asha/scripts/warroom/build.py`; their timestamps were correlated and they
are not included above. Both reported runs were checked during every workload,
and no competing build, `llama-bench`, or backend test process appeared.
