# GLM-5.3-Flash: kpool indexer pooling fusion (historical design; first fusion deployed)

> Status corrected 2026-09-19: the first fusion is implemented in
> `glm-pool-0919` and deployed through the combined `glm-cpu-fast-0919` CPU
> library (SHA256 `4793379e6894a9286168f79c4f323985388ec0b0ac65e54013dcd2957488af1f`).
> The active switch is `GGML_CPU_GLM_POOL_FUSION=1`; the production service also
> enables `GGML_CPU_CPY_FLAT=1`. The approximately 14 tok/s Codex baseline
> measurements already included this first fusion. The September 18 timings and projected gains below
> are relative to the older unfused path, not incremental gains over today.
>
> The implemented integration uses strict CPU subgraph matching, not the suggested
> new CUSTOM model node. See [deployed implementation](glm-pool-copy-0919/README.md).
> The further eight-channel inner loop in
> [glm-kpool-wide-0919](glm-kpool-wide-0919/README.md) is now deployed after
> bit-exact and full-model gates. Actual matched Codex A/B gains were 3.8% at
> 3,998 input tokens (14.60 to 15.15 tok/s) and 13.4% at 29,930 tokens
> (9.05 to 10.26 tok/s). These are measured incremental gains; 18+ tok/s was not reached. Pooled-key caching remains a separate, unimplemented
> design; `GGML_GLM5N_KPOOL_REUSE=1` enables graph reuse, not that cache.

## The finding (traced 2026-09-18, windows flash-optrace + flash-trace32k)

Per token, per full-attention layer (12 of 46), `build_indexer` (glm5next.cpp) re-derives
every pool summary from scratch:

1. `members = get_rows(kg_rows [2*d_idx, n_kv] F16, pool_cells [r*n_pools])` — full
   re-gather of all pools' member cells, converted to F32: 2*d_idx*r*n_pools*4 B
2. `keys_t/gate_t = cont(permute(...))` — TWO full transposes of that gather
3. `gate_t = add(gate_t, ape)` — broadcast ape over all pools
4. `probs = soft_max(gate_t)` — d_idx independent r-way softmaxes
5. `pool_k = sum_rows(keys_t * probs)` — per-channel weighted average

Measured growth, verify graph 150.3 ms (190 ctx) -> 357.7 ms (~30K ctx, 7,554 pools):

| op | 190 ctx | 30K ctx | delta |
|---|---|---|---|
| GET_ROWS | 1.6 | 52.5 | +51 |
| CONT | 0.5 | 52.1 | +52 |
| ADD | 0.7 | 44.9 | +44 |
| SOFT_MAX | 0.3 | 25.6 | +26 |
| TOP_K | 0.1 | 4.0 | +4 |

MoE (53), dense (44) and state CPY (20) are all FLAT. TOP_K is 2% — do not chase it.
The pooling chain (GET_ROWS+CONT+ADD+SOFT_MAX+MUL/SUM) is ~190 ms of the +207 ms.

Shapes at 30K (d_idx=128, r=4, 7,554 pools): members is 2*128*4*7554*4 B = 31 MB;
each transpose moves 31 MB read + 31 MB write with strided patterns.

## The change: one fused pool+score preparation op

A single op (suggested: GGML_OP_CUSTOM `glm_kpool_summarize`, mirroring the
meta_fused_reduce precedent — NOT a core ggml op) that, per pool p in parallel:

- gathers its r=4 member key+gate rows straight from the F16 kbuf (no materialized
  `members`, F16->F32 on the fly)
- adds ape, runs the r-way softmax per channel (r=4: fully unrolled)
- emits pool_k[d_idx] (weighted average)

Traffic per token per layer: read 2*d_idx*r*n_pools*2 B (F16, streaming) + write
d_idx*n_pools*4 B. At 30K: 31 MB + 7.7 MB streaming vs ~300+ MB of gather/transpose/add
traffic today with strided patterns. Expected: ~190 ms -> ~2-4 ms at 30K (~50x on the
stage, +50% decode). At 1M ctx (262K pools): ~670 MB streaming ~= 7 ms vs ~1.5-2 s today.

## Why bit-identical is achievable (and required)

The fused op performs the SAME arithmetic in the SAME element order, only without
materializing intermediates: ape add elementwise, softmax as max-then-normalize in slot
order (match ggml_compute_forward_soft_max_f32 loop order exactly), weighted sum in slot
order. Garbage pools (non-resident slots read cell 0) produce deterministic garbage
either way since the same cells are read; pool_bias neutralizes them downstream
unchanged. Promotion bar: bench3 + lctx parity IDENTICAL, then battery. Any deviation
means an ordering bug, not a modeling choice.

## Sharding

Current nodes run under the meta splitter; the fused op must declare a split rule.
Recommended: ROW_SPLIT over pools (socket s computes pools [s*n/4, (s+1)*n/4)) —
embarrassingly parallel, no cross-socket traffic beyond what the downstream score
matmul already does. Mirror (each socket computes all) is the fallback if the row rule
fights the downstream matmul's expectations; it wastes 4x socket-time but keeps wall
time at ~2-4 ms. Either way the op must handle n_pools % 4 != 0 tails.

NUMA note: kbuf members live wherever the KV cache shards are; the fused op reads them
streaming. Per-pool reads touch r=4 scattered rows (pool_cells order) — same pattern as
today's gather, just without the write-back. No new placement work expected, but verify
with a numastat spot check on the arm.

## Cache second (later, past ~100K)

Exactly per the Qwen INDEXER-POOL-CACHE-DESIGN.md correction: fusion first (safe, ~50x),
pooled-summary cache second (update only pools whose membership changed — one pool per
appended token plus rotation tails; wire invalidation to seq_rm). The cache takes ~7 ms
to ~0 at 1M; below ~100K the fused pass alone is sub-ms and the cache is not worth its
coherence risk. Do not build both at once.

## Build/test plan (house pattern)

1. New files only: ggml-cpu custom-op kernel + model-side node swap in build_indexer
   behind GGML_GLM_KPOOL_FUSED=1 (default off). No core ggml changes.
2. Override lib via LIB_PREPEND (copy the glm-hist build recipe), A/B on :18131.
3. Gate 1: bench3 seed-42 parity IDENTICAL (bit-exact required — same arithmetic).
4. Gate 2: pool integrity script + 10/10 battery + lctx 32K point for the money number.
5. Promote: default the flag on in the launcher, keep env kill-switch for one week.

## Open question (one focused window, not on this path)

The ape-broadcast ADD runs at ~8 GB/s (15.5 MB in 4 ms) — ~10x under STREAM for a
contiguous elementwise op. Probably mirrored-write coherence (same disease as the CPY
window) or a thread-count pathology. It vanishes under fusion, so do not chase it
separately unless fusion stalls.

## Current status

Both the original fusion and the subsequent eight-channel inner loop are built, validated and deployed. The measured incremental wider-loop results are recorded at the top of this historical design. The original projected +50% is not a gain available over the current runtime. Pooled-key caching is still a separate unimplemented design.
