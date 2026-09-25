# Two CPU kernel fixes for MiMo-V2.6-Pro (2026-09-22 evening)

Both live in `engines/llama.cpp-mimo-tp` behind env gates (default OFF, so every other build of this tree is
unchanged) and are ON in production via `launch-mimo-production.sh` / `mimo-v26-pro.service` (`build-prod-0922b`).
Patches: `patches/cpu-fa-gqa-grouped-splitkv.patch`, `patches/cpu-x16-multicolumn-gemm.patch`.

## 1. `GGML_CPU_FA_GQA` -- grouped-query split-KV flash attention (ops.cpp)

**The defect.** For query batches under 64 rows (every decode step and every speculative verify), ggml's CPU flash
attention runs `one_chunk` once per (query row, Q head) over the ENTIRE KV range. With grouped-query attention every
K/V row is therefore streamed rk2 x N times per call: under 4-way tensor parallelism MiMo has 16 Q heads per KV head,
and DFlash verifies 8 rows, so each full-attention layer re-reads its cache 128 times. It also accumulates V in FP16
(`VKQ16`, the defect already recorded in cpu-fattn-fp16-accumulation). This is why decode fell 3.9x between 4K and 64K
depth even though only 10 of 73 layers keep full KV.

**The kernel.** All Q heads sharing a KV head x a group of rows (<= 128 queries) form one block that meets each K/V
tile (64 cells) once; scores and P*V are `simd_gemm` calls; online softmax and accumulation in F32; the visible KV span
(from the mask) is split into chunks so 15 threads have work with only 2 KV heads per node; per-chunk (M, S, O) are
merged at the end, then sinks. Planned at run time from the mask so sliding-window layers only touch their window.
Level 1 = decode/verify batches, level 2 = prompt batches too (replaces the 64-row tiled path).

`tools/fa-bench` (MiMo's per-node shapes: 32 Q / 2 KV heads, DK 192, DV 128, node 3 cores, vs float64):

| shape (per layer per node) | old | new | old rel.err | new rel.err |
|---|---:|---:|---:|---:|
| full attn, 65K cells x 8 rows | 160.5 ms | 14.3 ms (11.2x) | 3.8e-2 | 8.6e-7 |
| full attn, 16K x 8 | 30.7 | 4.2 (7.3x) | 1.9e-2 | 4.7e-7 |
| full attn, 4K x 8 | 5.1 | 1.4 (3.6x) | 9.0e-3 | 2.9e-7 |
| full attn, 65K x 1 | 23.1 | 6.0 (3.8x) | 9.6e-3 | 8.5e-7 |
| SWA-128, 768 cells x 8 | 0.274 | 0.197 (1.4x) | 1.6e-3 | 1.7e-7 |
| DFlash SWA-1024 (DK 128) x 8 | 1.194 | 0.323 (3.7x) | 4.8e-3 | 2.0e-7 |
| prompt 512 rows @ 66K / 16.9K / 4.6K | 2068 / 436 / 115 | 1043 / 261 / 70 | 5e-6 | 5e-6 |

**Not byte-identical to the old build**: the old kernel's FP16 rounding is gone. On the 4-layer proxy, old vs new
top-1 logprobs differ by <= 0.024 on short prompts and 0.146 at 8K depth, which is the old kernel's error growing with
context; level 2 vs level 1 is identical (0.0000).

## 2. `GGML_CPU_X16_GEMM` -- multi-column x16 kernels for prompt batches (repack.cpp, arch/x86/repack.cpp)

**The defect.** For batches, `tensor_traits_x16` calls the x16 GEMV once per token over an L2-resident chunk of
weight rows. Every loaded weight vector feeds one token's dot products, so prompt processing ran at ~1.1 TFLOP/s per
node, ~13% of the int8 VNNI peak (op profile of prefill on the proxy: wqkv 24%, MoE gate+up 24%, attn_out 14.5%,
MoE down 12%, flash attention 6%). Changing `--ubatch-size` 512/1024/2048 made no difference -- kernel-bound.

**The kernel.** `ggml_gemm_ptrs_{q8_0,mxfp4}_x16_q8_0`: up to 8 activation rows per call via pointer arrays (MoE token
rows are not contiguous); each weight vector loaded -- and for MXFP4 LUT-decoded -- once for all of them. Wired into
all four batch paths: dense matmul, fused dense gate/up, MoE matmul, fused MoE gate/up+SwiGLU. **Bit-identical**: per
(row, column) the float operations are the GEMV's in the same order, and the integer dot products are exact in any
order. Verified on the proxy: identical tokens, |dlogprob| 0.0000 on every prompt including an 8K one.

Proxy (measured while the production model was loading, so noisy): prefill 390 -> 475-549 tok/s, decode (DFlash
verify batches also hit the dense GEMM) 14.3-14.9 -> 16.6-17.1 tok/s.

## Validation tools
- `tools/fa-bench.cpp` -- FLASH_ATTN_EXT at production shapes vs float64; window arg for SWA.
- `tools/ab-proxy-env.sh` -- proxy golden + 8K probe per env variant, pairwise compare.
- `tools/prof-proxy-prefill.sh` + `tools/opsum.py` -- file-armed op profile of a prefill, summarised per op/tensor.
- Production carries the dormant profiler: `touch /tmp/mimo-prof-arm`, then `journalctl --user -u mimo-v26-pro`.

## Live results (build-prod-0922b, deployed 18:43; results/deploy-0922b/)

Depth probe, one append-only session (prefill = the NEW tokens of each step):

| depth | prefill tok/s before -> now | decode tok/s before -> now | tokens/cycle | acceptance | ms/cycle now |
|---:|---:|---:|---:|---:|---:|
| 4K | 35.5 -> **62.2** | 7.68 -> **18.45** | 5.82 | 0.98 | 315 |
| 16K | 31.4 -> **54.6** | 5.29 -> **8.56** | 2.46 | 0.55 | 288 |
| 32K | -- -> **43.8** | -- -> **9.25** | 2.70 | 0.65 | 292 |
| 64K | 21.0 -> **33.1** | 1.98 -> **6.36** | 2.31 | 0.48 | 364 |

The verify cycle is now nearly flat with depth (288-364 ms from 4K to 64K; it was ~1.06 s at 64K), so decode at depth
is set by draft acceptance, not by attention. The 4K decode figure is flattered by a very predictable continuation
(0.98 acceptance); compare cycle times, not tok/s, across builds.

By workload (draft-probe, short context): verbatim 22.3, counting 25.3, code 15.7, list 12.7, prose 8.6 tok/s
(was 19.2 / 20.4 / 13.4 / 11.7 / 8.8). Smoke 7/7, prefill 60.5 tok/s in the smoke's speed test.

## build-prod-0922c (deployed 19:46): activation block sums computed once per matmul

The x16 GEMV recomputed every activation row's per-block sums/scales for each 16-row chunk it was called on
(`GGML_CPU_X16_CHUNK_MAX=16` in production), about as much work as the dot products for a 16-row call. 0922c computes
them once, right after quantizing the activations, and routes single-token calls through the same kernel with NR=1.

- **Golden vs 0922b: IDENTICAL on all three prompts, |dlogprob| 0.0000** -- bit-exact on the full model.
- By workload (draft-probe, clean box, 19:47): verbatim 22.8, counting 26.9, list 13.2, code 16.3, prose 8.8 tok/s
  (0922b 22.3 / 25.3 / 12.7 / 15.7 / 8.6), identical drafted/accepted counts. Smoke 7/7 (speed-test prefill 69 tok/s
  vs 60.5). Vision 4/4, audio 12/12.
- The 4K/16K depth run at 19:50-19:58 is INVALID: a training job from another project (`experiment.py
  --threads 16`) started at 19:48 pinned to cores 32-47,96-111 -- NUMA node 2, one of MiMo's four tensor-parallel
  nodes -- plus an sd-cli Qwen-Image job and another project's CI. Every op inflated uniformly (~30%, the known cost of one
  contended TP node, see smeagol-periodic-jobs-stall-tensor-parallel). Re-measure depth only on a quiet box.

## Staged, not deployed: vectorised MXFP4 E8M0 scales (source tree + patches/SOURCE-STATE-staged-0922d.diff)

The MXFP4 GEMM converted each block's 16 E8M0 row scales with a scalar table-lookup loop (~48 uops of ~130 per block
at 5 columns). Replaced with ggml_e8m0_to_fp32_half's own bit formula in 6 AVX-512 ops: exhaustively equal on all 256
codes (`/tmp/mimo-vis/e8m0check.c`), and the proxy golden + two 8K probes are IDENTICAL (0.0000) to the GEMV baseline.
Expected ~8% on prefill (MoE is ~32% of a prefill ubatch). Not worth a 25-minute restart on its own: build it into
`build-prod-0922d` the next time the service restarts for any reason, then run the golden (must equal deploy-0922c).
The source tree already carries it, so ANY rebuild of the tree includes it.

## Production prefill profile (0922c, 20:12, contended by a training job from another project on node 2)
Per node per 512-token ubatch (~11.2 s under contention): cross-node reduce 35% (inflated: fast nodes wait for node
2), MoE 32% (gate+up 19.6%, down 12.1%), dense wqkv + attn_out 17% (~2.5 TFLOP/s/node, 2.1x the old GEMV path),
flash attention 4%, router F32 matmul 3%, expert weighting MUL 3%. `--ubatch-size 2048` is slightly slower than 512
on the proxy with the GEMM (803 vs 834-862 tok/s), so 512 stays.

## Speculation re-tune on the faster verify pass (22:41-22:56, build-prod-0922e) -- keep n_max 7 / p_min 0.5
Measured under steady contention (two InfoSystemic experiments pinned to NUMA nodes 2 and 3), interleaved ABBA so
drift cancels; relative numbers only. On the sweep's hard-prose + code prompts, n=4 p_min 0.5 looked +17% on prose,
-4.6% on code; everything else within noise or worse. Across draft-probe's five workloads, though:

| workload | n=7 | n=5 | n=4 |
|---|---:|---:|---:|
| verbatim repeat | base | -14.7% | -18.4% |
| counting | base | -13.3% | -21.2% |
| memorised list | base | +14.0% | +9.6% |
| code | base | -5.5% | -8.6% |
| open prose | base | +2.7% | -19.4% |
| **geometric mean** | **0** | **-3.9%** | **-12.3%** |

The prose gain was prompt-specific. **n_max 7 / p_min 0.5 stays the production setting.**

## build-prod-0922f (2026-09-23 03:31): three more bit-exact speedups

1. **K tiles transposed 16x16 in registers.** The kernel converted and transposed each 64-key tile one scalar at a time
   (12,288 scalar ops per tile). Now 16 keys x 16 dims per step: `_mm512_cvtph_ps` + an in-register transpose
   (unit-tested against a scalar transpose) + 16 row stores. Same floats (F16->F32 is exact).
2. **L1-blocked tile GEMMs** (`fa_gqa_gemm`). `simd_gemm` walks 4-row blocks across all N columns, re-streaming the
   48 KB K panel (bigger than L1) for every 4 query rows. 32-column panels outermost + 12x2 register blocks keep a
   24 KB panel L1-resident. Same FMA sequence per output element, so bit-identical.
3. **`GGML_CPU_X16_CHUNK_MAX_BATCH=64`** (repack.cpp). Production's decode-tuned 16-row chunk cap made a 512-token
   prompt batch re-read its 3.3 MB of quantized activations 424x per matmul. Prompt batches now use 64-row chunks.
   Proxy prefill: 557 -> 600-638 tok/s (+8..14%); 128 was worse (531-554, load imbalance). Decode unchanged.

fa-bench (node 1, min of 12-60 iterations; every output hash identical across old/new):

| shape (per layer-node) | 0922e | +transpose | +blocked GEMM (0922f) |
|---|---:|---:|---:|
| full attn 64K x 8 verify | 16.1 ms | 12.5 | 10.6-11.9 |
| full attn 16K x 8 | 4.15 | 3.19 | 2.77 |
| full attn 4K x 8 | 1.08 | 0.85 | 0.82-0.87 |
| SWA-128 768 x 8 | 0.20 | 0.09-0.14 | 0.10-0.11 |
| DFlash SWA-1024 (DK 128) | 0.32 | 0.28 | 0.27 |
| prompt 512 rows @ 4.6K | 69.3 | 52.6 | 45.0 |

Against the 09-22 morning kernel the 64K x 8 verify op is now ~14x faster. Proxy goldens (short + two 8K probes) are
IDENTICAL to the pre-change GEMV baseline.

**Measuring on a shared box:** use the MIN over many iterations, not the median -- other sessions' jobs preempt the
15 threads in ~12 ms quanta and make medians bimodal (a transpose that is 1.29x faster read as 1.8x SLOWER at 64K on
medians). The cross-node reduce that looked like 35% of a prefill ubatch under contention is ~1% on a quiet box by
arithmetic (19 MB cross-socket per reduce, 140 reduces per ubatch): it was the other nodes waiting for the contended one.

## build-prod-0922g (2026-09-23 04:20): prompt-batch kernel scheduling, all bit-identical

`tools/x16-gemm-bench.cpp` (new): calls the exported x16 GEMM kernels directly on synthetic data at MiMo's per-node
shapes, every output checked against the single-column GEMV (max|diff| 0). Quiet NUMA node 1, 15 threads:

| change | shape | before | after |
|---|---|---:|---:|
| even split, <= 11 cols/call (was 8 + tail) | MoE gate, 11 tok/expert | 2483 GOPS | 2888 (+16%) |
| even split, <= 10 cols/call | dense Q8_0 wqkv, 512 tok | 3223 | 3361 (+4%) |
| 512-row tiles for prompt batches (`GGML_CPU_X16_MOE_TILE_BATCH`) | MoE down (6144 x 512) | 12.45 ms | 10.27 (1.21x) |
| SCALE threaded (`GGML_CPU_PARALLEL_UNARY=4096` gates it) | attn_out_scaled | 1 thread | 15 |

Learned on the way: **12 columns per call is a cliff for Q8_0** (2152 vs 3384 GOPS at 10, register spills) while MXFP4
at 11 is fine; a thin tail call is what costs (MXFP4 16 tokens as 12 + 4 lost to 8 + 8), hence the even split.
Tile size does not matter for the gate/up shape (64-512 within noise). Proxy prefill A/B was inside noise (load ~100
from other sessions' jobs) but golden-identical in all four runs.

## build-prod-0922i (2026-09-23 05:24, deployed): paired-row-group VNNI kernels + exact MoE weighted-sum fusion + 1024-token micro-batches, all bit-identical

Three changes, each golden-identical on the proxy (short prompts AND the 8K-token probe) against 0922g:

1. **Prompt-batch VNNI kernels run 16-row groups in pairs** (below). Source: `patches/cpu-x16-gemm-paired-groups.patch`.
2. **`GGML_CPU_MOE_WEIGHTED_SUM_FUSION=2`** (`patches/cpu-moe-weighted-sum-exact.patch`, ggml-cpu.c). The MoE output
   is a MUL by the router weights over [6144, 8, tokens] followed by 7 ADD nodes; the tree already had a fusion for
   it (mode 1), but it uses `ggml_vec_mad_f32` = FMA, which rounds once where the graph rounds twice -- not
   bit-identical, so production never enabled it. Mode 2 rounds each product and adds in slot order. GCC's default
   `-ffp-contract=fast` fused the FIRST product into the first add anyway (caught in the disassembly: one
   `vfmadd231ps` per loop); an empty asm barrier on every product, the first included, keeps them apart. Proxy op
   profile: MUL + ADD 7.5% -> 2.5% of prefill time; ~460 MB -> ~113 MB of traffic and 8 barriers -> 1 per layer.
3. **`UBATCH=1024`** (launch-mimo-tp.sh now takes `UBATCH`/`NBATCH`). Every MoE expert's weights (1.9 GB per layer per
   node for the 384 experts) are streamed once per micro-batch, so 1024 halves that traffic per prompt token. Output
   is identical at 512, 1024 and 2048 (the kernels are exact per column, FA per row). The proxy (4 layers, one of them
   dense) could not resolve the speed difference under contention (u512 345-387 tok/s, u1024 361-382, u2048 337-410).

### The paired-group kernels

Why the 0922g kernels sat at ~0.64 dot products per cycle when the core can issue ~1.85 (`tools/ubench/peak.cpp`,
sibling hyperthread held idle, AVX-512 heavy clock measured at 3.06 GHz single-core):
- `vpdpbusd zmm, zmm, m32{1to16}` (the embedded-broadcast form, forced in 0922g-dev by inline asm) issues at only
  ~1.0-1.35/cycle on this Cascade Lake; the register form reaches ~1.85. GCC's own code for `_mm512_set1_epi32(*p)` is
  a separate `vpbroadcastd` per dot product, so either way every dot product cost a load.
- llvm-mca (`tools/ubench/loopmca.py`, -mcpu=cascadelake) on the 10-column Q8_0 loop: 203 instructions per
  block, load ports 65 cycles, vector ports 66 -- the loop was load-bound as much as ALU-bound; ~40 of its loads and
  30 of its vector ops were the per-block epilogue (bias `vpsubd`, zeroing moves, pointer reloads).

The new kernels (`arch/x86/repack.cpp`, `ggml_x16_gemm_{q8_0,mxfp4}_nr`): for calls of <= 8 columns on tiles of >= 32
rows, two 16-row groups per pass, each activation dword broadcast ONCE into a register and fed to two register-form
dot products; accumulators start at a precomputed per-block bias (`ggml_x16_q8_0_col_prep_bias`: -128 x block sum for
Q8_0 weights, -12 x for MXFP4) instead of zero + `vpsubd`. An odd last group, or a call wider than 8 columns, runs the
single-group loop. Callers pick 5 (Q8_0) / 6 (MXFP4) columns per call for >= 32-row tiles, 10 / 11 for 16-row chunks
(decode). For every (row, column) the float operations are the GEMV's in the same order: GEMM == GEMV at every width
1-12 on 16-, 48- and 64-row tiles (max|diff| 0, x16-gemm-bench).

Single core, sibling held, same data, old/new library interleaved (min ms): dense Q8_0 512 columns 40.3 -> 32.0
(1.26x), MoE gate 11 tokens/expert 44.3 -> 36.5 (1.21x), MoE down 45.6 -> 42.5 (1.07x); decode widths no slower
(8-column dense 0.78 -> 0.64, 2-token MoE 20.4 -> 16.5, noisy). Tried and dropped: three groups per pass (no better
than two), activation blocks carrying their own bias/scale (40-byte blocks: GCC runs out of GPRs and reloads
pointers per dot product, 1.26x SLOWER).

Measurement note: all four NUMA nodes carried other sessions' 16-thread training jobs (nice 19) during this work; a
nice-0 `pause` spinner on the benchmark core's hyperthread sibling (`tools/ubench/spin.c`) gives a clean core
without stopping anyone's job. Frequency still drifts with the socket's load, so compare only interleaved runs.

Where the paired kernel sits now (single core, sibling held, `tools/ubench/x16-kernel-proto.cpp` + a direct-call
harness): ~0.8 dot products per cycle with the weight chunk streaming from L2, ~1.04 with it L1-resident, against
llvm-mca's 1.23 and the core's 1.85. Ruled out: instruction delivery (a 3.8 KB unrolled loop of the same mix runs
at 1.9/cycle), L3 activation streaming (64 rows x 5..510 columns all ~325 GOPS once measured on a stable clock),
software prefetch of activations (PFD 2-16) or weights (PFW 1-8): no gain. The L2 weight stream (~22%) and the
per-block epilogue + register moves are what is left.

### Proxy op profile after the change (prefill, 2137-token prompt, contended: other sessions' jobs on all nodes)
Per node per 512-token micro-batch, 4-layer proxy (layer 0 has a dense FFN): dense wqkv 13%, MoE gate+up 12.5%, attn_out
9.5%, MoE down 6.8%, cross-node reduces ~29% (mostly nodes waiting for the most contended one), flash attention 9.5%,
dense FFN (layer 0 only) 7.4%, router F32 GEMM (llamafile tinyBLAS) 3.2%, RoPE 2.7%, RMS norm 1.5%, MUL + ADD 2.5%.
The router runs at ~290 GFLOPS under contention; a faster kernel would have to reproduce tinyBLAS's summation order
exactly or expert choices could flip at near-ties -- not attempted.

## build-prod-0922j (09-23 06:36, the unit's build): vectorised row max in the GQA flash-attention softmax, bit-identical

Phase counters (`patches/dev-fa-gqa-phase-counters.patch`, env `GGML_CPU_FA_GQA_PHASES=1`, dev only) on fa-bench,
16K x 8 verify rows, single core: QK^T GEMM 37.5%, PV GEMM 26.5%, **softmax 19.9%**, K tile 8.8%, V tile 5.3%, mask
1.2%. The softmax cost was `ggml_vec_max_f32`: a scalar loop (GCC does not vectorise a float max reduction without
fast-math), i.e. a 64-step dependent vmaxss chain for each of the 128 query rows of every tile. An AVX-512 max
(`patches/cpu-fa-gqa-vector-max.patch`) is exact in any order -- only the sign of a zero maximum could differ, and
x - (+-0) gives identical exps -- and fa-bench's output hash is unchanged at 16K x 8 (1 thread), 4K x 8 and SWA 768 x 8
(15 threads), 64K x 8 (15 threads). Softmax 19.9% -> 10.7%; 16K x 8 single core 46.9-53.4 -> 34.0-37.3 ms (1.4x);
64K x 8 at 15 threads 14.6 -> 11.7 ms best case under contention (1.25x). Prompt batches use the same kernel.

Tried and dropped: software prefetch of the next tile's K/V rows into L2 (slower single-core, a wash at 15
threads); K-blocking the QK^T tile GEMM so the Q block + K panel fit L1 (102-103 vs 107-112 GFLOP/s, bit-identical
but slower); other register-block shapes (8x3, 6x4, 4x6, 14x2 all within ~15% of 12x2 at K=64, all ~0.96
FMA/cycle at K=192). The tile GEMMs run at ~65% of the core's practical FMA rate (~1.65 FMA/cycle measured with 24
register accumulators, `tools/ubench/`: fpeak2.cpp, fma-ukernel-shapes.cpp, fa-tile-gemm-kblock.cpp) -- the same as the 12x2 microkernel achieves on
L1-resident data by itself, so the remaining GEMM headroom needs a different kernel design, not blocking.

Proxy golden, no speculation, 3 prompts x 48 tokens + 8K-token probe: 0922g == 0922i == 0922j (IDENTICAL, max
|dlogprob| 0.0000). Same run, 8031-token prompt prefill on the 4-layer proxy: 0922g 873-889 tok/s, 0922i 1006-1080,
0922j 1050-1128 -- 1.2-1.27x over 0922g with output unchanged.

## 09-23 05:58: 0922i was OOM-killed PER NUMA NODE during its reload; guard + load watchdog

The kernel log: `constraint=CONSTRAINT_MEMORY_POLICY nodemask=3`. MiMo's per-node weight buffers are mbind()-bound
(MPOL_BIND, ~136 GiB per node); another session's benchmark (`tools/bench_mc.py ... --node 3`, a Qwen3-Next-80B
server, ~48 GiB anon + ~42 GiB page cache on node 3) started three minutes after the RAM guard had passed and
restarts its server within a minute of each run. Host-wide MemAvailable stayed > 180 GiB throughout.
- `wait-for-ram.sh` now also requires every node to have `MIMO_MIN_NODE_GIB` (150) free or reclaimable.
- `load-watchdog.sh` (started by the launcher before its exec): until /health answers, every 20 s, if a node's free +
  reclaimable memory + MiMo's per-node share so far (RssAnon / 4) drops below 141 + 4 GiB, it SIGKILLs the load; systemd
  restarts the unit and the guard waits. A doomed reload now costs about a minute instead of 35 minutes of SSD reads
  plus an OOM stall.
- Not done: MPOL_PREFERRED instead of MPOL_BIND would avoid the OOM but, with page cache filling every node during a
  load and zone_reclaim_mode 0, would scatter MiMo's weights onto remote nodes -- a silent slowdown.
MiMo cannot coexist with a >~45 GiB resident process on any one NUMA node. That is a scheduling decision between
workloads, left to the maintainer.

### Decode widths at 15 threads (node 2, contended, min of 10-40)
- Dense wqkv (6784 x 6144 Q8_0) with 8 verify rows: 0.39-0.44 ms whatever the chunking (16-row single group, 32- or
  64-row pairs) = ~108 GB/s -- at the DRAM wall, so the decode chunk cap (`GGML_CPU_X16_CHUNK_MAX=16`) stays.
- MoE gate (384 experts x 512 x 6144 MXFP4), tokens per expert 1 / 2 / 4: 0922g 99 / 76 / 66 GB/s, 0922j 97 / 91 / 81 GB/s.
  The paired kernels also bring 2-4-token experts (common in 8-token verify steps) ~20% closer to the wall.

## 0922j LIVE (09-23 07:31)
The guard let the reload start at 07:11:50 (the benchmark queue's second Qwen3-Next run had ended; its third run, a
Qwen3.8-27B server, started 4 s later on node 3 but holds only ~7 GiB anon there). Healthy 07:31:43 -- 20 minutes
from a warm page cache; the watchdog stood down at 07:31:53. Per-node free/reclaimable after the load: 26 / 37 / 29 /
19 GiB. **Full-model golden vs 0922g: IDENTICAL on all three prompts, max |dlogprob| 0.0000** -- so 0922j (paired-row
kernels, exact MoE weighted-sum fusion, 1024-token micro-batches, vectorised FA max) is bit-identical to 0922g, and by
the chain to 0922c. Then (`results/deploy-0922j/`): tool calls 0/20 malformed at T=1.0, smoke 7/7, real-image vision
4/4, audio 12/12 words (6.0 s). Production 8031-token prompt at 07:36 with the box still contended (load 66: other
sessions' training jobs on nodes 0-2, the benchmark on node 3): **prefill 61.5 tok/s**, decode 9.5 tok/s -- the
quiet-box record for 0922b was 62 / 55 tok/s at 4K / 16K. A quiet-box re-measure is still owed.
