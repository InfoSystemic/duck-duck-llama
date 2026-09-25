# GLM-5.3-Flash at depth: 15+ tok/s at 512K (plan, 2026-09-20 ~20:00)

Goal (09-20 ~16:40): "run GLM-5.3-Flash from Codex in Paseo at 512K context if I want, with 15+ tok/s".
Today: 20.2 tok/s at 4K, 12.6 at 128K measured (`results/depth-curve-prod-revf.json`), ~6 at 512K extrapolated.
Paseo caps are already lifted to 512K (`codex-smeagol`, Kimi); the wrapper is not the limiter, the engine is.

## Where the depth cost is (measured, `results/depth-trace-8k-vs-128k.txt`)

Verify graph on one socket: 105.2 ms at 8K -> 237.6 ms at 128K (+132 ms). Same 7,239 nodes. Growth by op:

| op | 8K | 128K | growth | what it is |
|---|---:|---:|---:|---|
| `LIGHTNING_INDEXER` `indexer_pool_score` x11 | 2.2 | 37.5 | +35.3 | 3 rows x 32 heads x 128 dims against 34,178 pools per layer; `ggml_vec_dot_f32` per (pool, row, head): ~240 GFLOP/s on 15 cores |
| `GET_ROWS` `indexer_pool_members` x11 | 1.3 | 34.6 | +33.3 | this is the FUSED pool kernel (dispatched at the GET_ROWS node by `pool-fusion.inc`); per pool it memcmp-validates 4 x 512 B of cache against its thread-local record (2 KB read per pool = 70 MB/layer) then memcpy's the cached 128-float output |
| `CONT` of `indexer_pool_members-43` x2 | 0 | 22.3 | +22.3 | layer 43 (the LAST DSA layer) was NOT fused at 128K: the unfused chain ran |
| `ADD` `node_6876` [4,128,34178] | 0 | 18.0 | +18.0 | same layer: gate + ape over the whole cache |
| `SOFT_MAX` `indexer_pool_probs-43` | 0 | 10.7 | +10.7 | same layer |
| `TOP_K` x11 | 0.3 | 3.1 | +2.8 | partial_sort over 34,178 |
| `ADD` `kpool_kq_mask` x11, FA, MoE | | | +6 | mask per row over n_kv; cache effects |

So at 128K: ~51 ms is one layer falling off the fused path, ~35 ms is a scalar-ish score kernel, ~34 ms is cache VALIDATION
bandwidth (not recomputation), ~10 ms is the rest. Attention itself grows 1.4 ms. Nothing here is inherent.

## Why layer 43 is unfused (hypothesis, to verify first)

`ggml_cpu_try_glm_pool_fusion` (fleet-0912-ctx/glm-kpool-wide-0919/pool-fusion.inc, compiled into ops.f18.cpp via
pool-kernel.inc) matches the exact 14-node subgraph and rejects, among other things, when the pattern's OUTPUT storage
overlaps `kg`, `cells` or `ape` (`ggml_cpu_pool_overlaps`). The allocator may reuse `pool_cells`' region for the output once the
GET_ROWS (its last consumer) is "done"; which layer that hits depends on tensor sizes, hence depth. The 8K graph fuses all 11.
Verify by adding a reject-reason counter/print behind `GGML_CPU_GLM_POOL_PROBE` and reproducing on the 8-layer proxy at depth
(its DSA layers are 3 and 7; a 128K prefill on the proxy is minutes), or by reading the gallocr placement of `pool_cells`.
Fix that does not depend on the reason: make the fused kernel overlap-proof — each thread copies its own pools' `cells` entries
(4 x I32) and `ape` (2 KB) into thread-local scratch, `ggml_barrier`, then computes/writes; drop the overlap rejection.
Expected: -51 ms per graph at 128K (238 -> ~187 ms, 12.6 -> ~15.9 tok/s at 128K on its own).

## Progress log

- 20:07 The 8-layer proxy at 41K (64K ctx, `GGML_CPU_GLM_POOL_PROBE=1`) fuses both of its DSA layers: no `GLM_POOL_REJECT` line.
  The rejection is specific to the full model's allocation, consistent with the `cells`-overlap theory (pool_cells is consumed by all
  eleven DSA layers, so only after layer 43's GET_ROWS can its storage be reused for that layer's output).
- 20:12 Step 1 implemented: `cpu/pool-kernel.inc` (wide) and `cpu/pool-scalar.inc` (scalar, now shadowing the 0919 copy) snapshot
  each thread's cell indices into a thread-local vector, `ggml_barrier` (fusion only runs with the full team: single-task nodes
  never enter `ggml_cpu_try_fuse_ops`), then read from the snapshot; `cpu/cfuse/pool-fusion.inc` drops the `cells` term of the
  overlap rejection (kg and ape remain). `cpu/build-all-cfuse.sh` rebuilds `ggml-cpu.c.o` from `cpu/cfuse/ggml-cpu.combo.c` with
  the recorded command + `-ffile-prefix-map` (gate: unpatched copy reproduces 1c80cc478a66327e byte for byte) and links it with the
  f18 ops/dispatch/repack objects. Parity: `run/cfuse_parity.sh <libdir> <tag>` (proxy, greedy hashes at 8K and 41K).
  Validation at depth needs production (rev g deploy, then a 33K+ prefill and `depth_trace.py`): the proxy cannot show it.
- Built: `cpu/build-cfuse/libggml-cpu.so.0.22.0` 1d64d50669230775 (ggml-cpu.c.o 91f99507bf8defe1, ops.cpp.o 6cb818b974188c18;
  the cosmetic unused-variable fix rebuilt to identical bytes). Staged `deploy-0920g/` (only libggml-cpu changes; SHA256SUMS,
  deploy.sh rolling back to f, drop-in header). Gate before deploy: `run/cfuse_parity.sh deploy-0920f base` vs
  `run/cfuse_parity.sh cpu/build-cfuse cand` must give the same greedy text hashes at 8K and 41K on the proxy.
- Parity passed (t1 d08306bb4c6b0efd, t5 d07909537be42b01, drafts 5/113 and 4/117 in both runs). Rev g DEPLOYED to production
  (deploy-0920g/deploy.sh: healthy, libs mapped, sanity ok). 4K Codex fixture greedy x3: 19.98 / 20.26 / 20.33 tok/s, acceptance
  0.775 (rev f under the same prompt: 20.15, 0.775) -- no regression.
- 20:54 Depth validation to 41K (`results/depth-curve-prod-revg-40k.json`): greedy ms/cycle 132.1 / 131.2 / 127.6 / 138.7 / 144.2
  at 8K / 17K / 25K / 33K / 42K against rev f's 131.0 / 130.8 / 127.6 / 141.4 / 142.2 -- NO change. So the 25K -> 33K jump is not
  layer 43 unfusing (or it unfuses only deeper), and the +51 ms seen at 128K must be re-measured there: the session is being
  extended to 128K (`results/depth-curve-prod-revg.json`, ~80 min of prefill) with a 41K trace first. Publication of rev g waits
  for that number; the patch is exact either way.
- 20:55 Rev g trace at 41K (`results/optrace-depth-revg-41k.log`): all eleven DSA layers fused; verify graph 128.8 ms vs 105.2 at
  8K: score kernel +9.7, pool kernel +6.2, MoE cache effects +3.0, top-k +1.0, FA +0.9, mask/ADD +1.0.
- 20:59 Step 3 (score kernel, `cpu/li-fast.inc`, `GGML_F18_FEATURES` bit 4, gate `f18_li_fast_supported`): v1 (row in registers,
  queries streamed) is BIT-IDENTICAL to the deployed kernel at 8K/34K/131K rows (`cpu/test/li_fast_check.cpp`, dlopen + control
  word) and 2.0x faster single-thread (9.6 -> 19 GFLOP/s per core): bound by 48 KB of query vectors streamed from L2 per row.
  v2 blocks 16 rows per query pass (same per-dot ladder and reduce, same head order); built into `cpu/build-lifast/`, test pending.
  The per-dot horizontal reduce is the exactness floor (~4x over deployed); a true GEMM would change summation order and is
  NOT bit-exact -- keep it as a separate, quality-gated option.
- 21:07 v2 measured: IDENTICAL at 8K/34K/131K rows, 2.3x per core (12.8 -> 29.8 GFLOP/s). Comment-only edits rebuilt to the same
  bytes (lib 0763328c479a443b). Step 4 done as well: `cpu/topk-fast.inc` selects rows above 16,384 entries with an MSB-first radix
  select over order-preserving key bits (same threshold value, same set, same tie fallback; `cpu/test/topk_radix_check.cpp`: 9/9
  identical sets, ~2.4x per row); it rides the already-deployed bit 1, so no new switch. Both go into revision h
  (`deploy-0920h/`, GGML_F18_FEATURES=31), rebuilt and re-staged; parity on the proxy at CONTROL=31 and the deploy wait for the
  128K run to finish (a restart would destroy its session cache). Patch body: `publish/patches/glm5next-indexer-score-blocked.body`.
  Spare-core timings are inflated 2-4x by the CI job pinned there: trust ratios, not absolutes.

- 22:37 **128K comparison (`results/depth-curve-prod-revg.json` vs rev f): NO gain from rev g anywhere** (128K: 202.3 vs 198.1
  ms/cycle; the deltas across 16 depths are noise, -12.6..+4.2). The +51 ms unfused layer existed only in the TRACED request:
  that request was restored from the RAM prompt cache with a different cell layout (n_kv 136,712 for 128,416 tokens, 287 ms/cycle)
  -> different tensor sizes -> different allocator placement -> the overlap rejection. Steady-state sessions never hit it. Rev g is
  therefore a robustness fix (a restored 128K session no longer pays +51 ms), not a speedup. The steady-state slope at 128K
  (+70 ms/cycle over 8K) is score ~35 + pool validation ~33 + top-k ~3: rev h should take ~-19 ms (13.6 tok/s at 128K), step 2
  the ~33 ms (~16.7), and 512K still needs step 5 or a non-exact GEMM score kernel (~215 ms/cycle without it, ~163 with).
  Lesson: a trace taken from a restored state is not the steady state; trace the session that produced the curve.

## 2026-09-20 22:45 — new target from the maintainer: "as well as possible from Paseo/Codex, like 25+ tok/s"

Revision h is live and reads 20.54 / 20.62 / 20.49 tok/s on the 4K Codex fixture (revision g 19.98-20.33). 25 tok/s needs the
cycle at 98 ms; it is 121. Where the 121 ms goes, per matrix, from the 8K trace (one socket, 3 verify rows, 15 workers, bytes
divided by the 4-way tensor split):

| matrix | type | ms | GB/s per socket |
|---|---|---:|---:|
| expert down x39 | q5_K | 14.5 | at the wall |
| expert up/gate x41 each | q4_K | 11.9 / 11.7 | at the wall |
| mtp_result_output x5 | q8_0 | 8.9 | 23.6 |
| kda_out x34 | q8_0 | 3.6 | 21.2 |
| ffn_moe_logits x47 | f32 | 3.1 | 17.8 |
| dsa_out x11 | q8_0 | 2.8 | 17.7 |
| mtp_eh_proj x5 | q8_0 | 2.7 | 16.6 |
| hc_mixes x90 | q8_0 | 2.6 | **3.6** |
| result_output, ffn_shexp, ffn_gate, ffn_up | q8_0 | 6.1 | 18-23 |

**The experts are at the wall; the dense matmuls are not — they run at 17-23 GB/s per socket (hc_mixes at 3.6) against the ~95
the expert kernels reach.** That is ~27 ms of the cycle doing ~6 ms of work. This corrects the earlier note that said dense was
"also at the wall (84-92 GB/s)": that figure counted the full tensor, not the per-socket slice.

Cause, from the repacked-matmul path: every matmul quantises its src1 rows into wdata before the GEMV and splits that loop by
ROW. A trunk verify batch is 3 rows, so 3 of 15 workers quantise while 12 wait at the barrier that follows; for the wide, short
hyper-connection matrices (16,384 x 24) that quantisation IS the operation. Fix (exact): split the tail-row quantisation by
COLUMN BLOCKS -- q8_0 and q8_K blocks carry their own scale and are independent, so the bytes are identical and only the
assignment of blocks to workers changes. `cpu/repack.f18.cpp`, `GGML_F18_FEATURES` bit 5, A/B with `run/qsplit_ab.sh`.
Ceiling if the dense matmuls reached even half the expert rate: ~14 ms off the cycle = ~23 tok/s; with the 4-row draft path
(1 worker today) and the small-op chain, 25 is in reach without touching precision.

Still open and quality-gated, not started: dense Q8_0 -> Q6_K load-time requantisation (knobs exist:
`GGML_CPU_{ATTN,SHEXP,OUTPUT,DENSE_FFN}_REQUANT`), ~9 ms, needs `run/ppl_glm.py` (built: llama-perplexity for this engine,
production runtime copied from /proc).

## Steps, in order (each gated: bit-identical output vs production, in-process A/B via control words)

1. **Layer-43 fusion** (above). Small; libggml-cpu only (`cpu/ops.f18.cpp` + `pool-fusion.inc` copy); `cpu/build-all.sh`.
2. **Pooled-key persistence with per-cell write epochs instead of content validation.** Today the thread-local cache re-reads
   2 KB per pool per token (4 member cells x 512 B) to prove the pool unchanged: 70 MB per layer at 128K, 34.6 ms per graph, and
   its 2048-record / 64 MiB per-worker limits start colliding at ~30K pools. Design (refined ~20:45 after reading the code):
   - **epochs**: one U32 per indexer-cache cell per DSA layer (`llama_memory_hybrid_idx` allocates it beside the key/gate cache;
     saved and restored with the cache state). The graph writes it right after `cpy_k`: `ggml_set_rows(epoch_l, epoch_now, k_idxs)`
     where `epoch_now` is a per-context 64-bit counter incremented per ubatch and NEVER restored from state, so a value can only
     recur on the same content (a restored checkpoint carries older epochs; new writes get larger ones).
   - **persistent pool_k**: per layer, f32 128 x n_pools_max plus 4 U32 "computed-from" epochs per pool (~200 MB per device at
     128K, 1.5 GB at 1M, allocate lazily per layer). The kernel compares 16 B of epochs per pool instead of 2 KB of content,
     recomputes only pools whose member epochs changed (the trailing 1-3 per token; every pool after a state restore), and the
     score op reads the buffer directly. Same arithmetic per pool -> identical bytes.
   - **one op instead of the 14-node chain**: `ggml_glm_pool(kbuf, pool_cells, ape, epochs)` built by `build_indexer` when the CPU
     backend advertises it (the existing `LLM_FUSED_OP_LIGHTNING_INDEXER` registration is the pattern), which also drops 13 nodes
     and their allocations per layer and makes the allocator-overlap question moot for good. Meta backend: a split rule like
     `GGML_OP_LIGHTNING_INDEXER` (ggml-backend-meta.cpp:1242; all inputs mirrored -> output mirrored).
   - host side unchanged: `llama_kv_cache_set_input_kpool` keeps filling pool_cells / pool_bias.
   Surface: llama (memory alloc + state I/O + graph + one context counter), ggml (op), ggml-cpu (kernel + buffers), meta (rule),
   four libraries through their lineage gates. Expected: -33 ms at 128K and flat with depth; the honest floor becomes the score
   kernel (step 3) and top-k (step 4).
3. **Score kernel as a blocked GEMM.** Q is 96 rows x 128 (3 tokens x 32 heads); K is n_pools x 128 f32. Tile 16 pools x 96 rows
   with AVX-512 FMAs, relu + head-weight sum in registers, mask add at the end. Target >= 700 GFLOP/s per socket:
   0.84 GFLOP/layer -> ~1.2 ms at 128K (from 3.4), ~4.8 ms at 512K. Exact up to summation order (f32 dot products; the current
   kernel already sums heads in f32) — check pool-ranking parity with `scripts/glm5next_pool_integrity.py`-style jaccard, and
   greedy text parity; a changed ranking on near-ties is possible and must be reported, not hidden.
4. **TOP_K O(n) for large rows**: at 512K it is 137K entries x 3 rows x 11 layers; threshold/radix select instead of partial_sort
   (the selection fast path is gated to <= 16,384 today). ~1 ms per layer at 512K otherwise.
5. **Indexer tensor-parallel**: every device scores ALL pools (mirrored work). Splitting pools across the 4 devices cuts steps 2-4
   by 4x at the cost of an allgather of 4 x select_k (2,051) x 3 scores+ids per layer and a merge. Needs a Meta collective; do
   last, only if 512K still misses 15 tok/s after 1-4.
6. **Prefill at depth: sparse attention is computed densely.** MEASURED 21:10 (`results/optrace-prefill-40k.log`: the op profiler
   armed during the 128K run's turn-5 prefill, one 1,024-token ubatch graph on one socket at ~40K depth): total 35.4 s, of which
   `FLASH_ATTN_EXT` 21.6 s (11 DSA layers, ~2.0 s each), `LIGHTNING_INDEXER` 3.1 s, MoE 4.1 s, dense matmuls 1.5 s, hc_mixes 0.6 s,
   GDN 0.5 s. The DSA layer selects 2,048 cells per query but the prefill path runs the tiled dense kernel over ALL n_kv cells with a
   -inf mask: O(n_tokens x n_kv) per layer, i.e. quadratic in depth (1,024 x 35K x 576 x 4 = 83 GFLOP per layer at CPU FA rates).
   The decode path does not have this problem (the cell-split MQA kernel of revision b visits only the selected cells).
   Fix: gather each query's selected K rows (`ggml_get_rows` on the MLA latent cache with the top-k indices: [576, 2048, n_tokens],
   2.4 GB per layer per ubatch, ~25 ms at DRAM rate) and attend over those 2,048 (4.8 GFLOP per layer): map it onto
   `ggml_flash_attn_ext` with n_head_kv = n_tokens (each query row is its own KV group, q reshaped to [d, n_heads x n_tokens, 1]),
   mask from the pool validity. Expected: a 40K ubatch 35 s -> ~14 s, and flat with depth (128K today ~70 s -> ~14 s: prefill
   14.5 -> ~60 tok/s). Meta split rule for the gathered K needed (mirrored cache -> per-device gather). This is the largest
   remaining lever for the "512K session" use case: prefill dominates wall time in real use. The score op at 1,024 rows already
   runs at ~1 TFLOP/s (0.28 s per layer here; 4x that at 512K), so step 3's exact kernel matters less in prefill than a true GEMM.

Projection after 1-4 at 512K: verify graph ~105 (flat part) + score ~50 + pools ~2 + top-k ~5 + mask/FA ~8 = ~170 ms per cycle
= ~14.7 tok/s at 2.5 tokens/cycle; with step 5 ~135 ms = ~18.5. So 15+ at 512K needs 1-4 done well, and 5 makes it comfortable.

## Tools

- `run/depth_curve.py --step 8192 --max-depth 131072` (preemptible; extend `--max-depth` to 262144/524288 to measure further;
  memory grows ~110 MB per 1K tokens of depth in the cgroup — watch `--mem-stop-gib`).
- `run/depth_trace.py --turns N --tag T` (per-op trace at the depth of turn N; arms once `n_decoded >= 1`; needs a final
  instruction the canned history cannot answer in 10 tokens) and `run/opdiff.py a.log b.log` (growth table).
- Lineage gates: `cpu/build-all.sh`, `cpu/build-repack.sh --gate`, `llama/build_glm5next.py --gate`, `base/build.sh --gate`.
- Deploy: copy `deploy-0920f/` to `deploy-0920g/`, update SHA256SUMS + `95-f18-0920.conf.proposed`, `deploy.sh` (waits for idle).

## Where the cycle actually goes, measured — and what "24+ tok/s" would take (2026-09-21 23:00)

Re-aggregated from the existing rev-g traces, one socket only (every socket runs the same mirrored graph, so summing all
four triple-counts) and one graph id at a time (trunk verify and MTP draft are separate graphs and must not be mixed):

**Trunk verify graph at ~41K depth** (`results/optrace-depth-revg-41k.log`, 2,116 nodes, 127.0 ms on socket 0):

| group | ms | n | % | shape |
|---|---:|---:|---:|---|
| `ffn_moe_down/up/gate` | 43.2 | 126 | **34.1%** | MUL_MAT_ID, 288 experts |
| `indexer_pool_score` | 12.5 | 11 | **9.9%** | LIGHTNING_INDEXER [128,32,3] |
| `indexer_pool_members` | 7.6 | 11 | **6.0%** | GET_ROWS [256,41728,1] |
| `kda_out` | 3.7 | 34 | 2.9% | [2048,4096] |
| `ffn_moe_logits` | 2.9 | 42 | 2.3% | [4096,288] |
| `dsa_out` | 2.8 | 11 | 2.2% | [4096,4096] |
| `hc_mixes` | 2.6 | 90 | 2.0% | [16384,24] |
| `ffn_gate` + `ffn_up` | 3.7 | 90 | 2.9% | [4096,3072] |
| `result_output` | 1.8 | 1 | 1.4% | [4096,38720] q8_0 |

**MTP draft graph at 8K** (`results/optrace-depth-8k.log`, 24 nodes, 4.1 ms):
`mtp_result_output` **1.83 ms = 44.5% of the draft pass on its own**, then `mtp_eh_proj` 0.53, the draft's flash-attn 0.49,
one MoE layer 0.57. At depth 2 that is two draft passes = 8.2 ms per cycle, 3.7 ms of it the output head.

### Two things this settles

**1. The output head is already at the wall, so only its BYTES can be cut.** `result_output` reads 4096 x 38720 q8_0 =
168.5 MB per socket in 1.83 ms = **92 GB/s**, against the ~95 GB/s a single socket sustains. There is no op-level win here on
either the trunk or the draft copy; there is only a quantisation win.

That matters because **the draft's output head can be approximated with zero effect on what the server emits.** Speculative
decoding is exact by construction: the draft only proposes, and every proposed token is kept only if it equals what the trunk
would have produced anyway. A cheaper draft head costs acceptance, never correctness — unlike every other requantisation on
this model, it needs no perplexity gate. The shape of the lever: a second, cheap copy of `output.weight` (Q4_K is 0.53x the
bytes, Q2_K 0.32x) bound only to the draft graph, with the top ~32 rows of its argsort rescored exactly in Q8_0 (32 rows is
139 KB, free) so acceptance does not actually fall. Q2_K would take the two draft passes from 3.7 ms to 1.2 ms, ~2% of a
short-context cycle. The work is a load-time second copy (55 MB per socket) plus a draft-only tensor binding — the MTP path
shares the main model's weights (`shares_model = !has_draft`), so the two copies have to be distinguished by graph, not by
model.

**2. 24 tok/s at short context is not reachable by op-level work, and the remaining levers are all at DEPTH.** At 4-8K the
indexer terms are near zero and the cycle is the MoE experts, which are already at the memory wall. Everything left —
draft-head bytes ~2%, `hc_mixes` ~2%, small-op fusion ~2% — totals ~6%, i.e. **20.6 -> ~21.8 tok/s at 4K**, and then the
decoder is reading all the bytes it must read as fast as the machine can read them. Going past that needs more tokens per
verify cycle, not a faster cycle: a second MTP head or a trained draft model. GLM-5.3-Flash ships exactly one MTP head, and
depth 3 on that single head measured 4% SLOWER. So 24+ single-stream at 4K is a model-capability limit, not an engineering
one. (Aggregate throughput across the 2 slots is a different number and already far higher.)

Where the engineering value actually is: **at 33K-128K+, which is where a 512K Codex session lives.** The indexer pool terms
are 15.9% of the graph at 41K and grow linearly with depth — steps 2 and 3 above (epoch-based pool persistence and the blocked
score GEMM) are worth roughly that much, and more at 128K. That is the work that moves 12.6 tok/s at 128K toward the high
teens, and it is exact. Do those before anything on the short-context end.
