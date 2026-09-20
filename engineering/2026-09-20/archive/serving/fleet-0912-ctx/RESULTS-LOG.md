
## window 53 (09-14 20:47) — Q4_K_M under blunt split modes

| arm | GGML_Q4E_SPLIT | result |
|---|---|---|
| split0 | 0 (routed experts + dense FFN + lm_head split, EVERYTHING else mirrored) | **LOADS**, 11.85 / 11.42 / 12.04 tok/s, prefill 32-35 |
| split14 | 14 | aborted, core dumped |

**Q4_K_M is servable today**, just badly: 11.8 tok/s against production Q6's 25.9. That is not a verdict on the
quantisation. `GGML_Q4E_SPLIT=0` mirrors the whole attention and recurrent path, so those sockets do redundant work
instead of parallel work and the shared weights are read four times over. The comparison is also confounded — it is
Q4+blunt against Q6+proper-split, two variables at once.

What it does establish: **the tensor-split rules are worth roughly 2x**, which is the same story the socket-scaling
window told from the other direction (6.37 -> 25.94 tok/s, 1 -> 4 sockets). Window 60 runs Q4 with the proper rule
(`GGML_Q4E_SSM_SPLIT1=1`) and is the arm that actually tests the recorded prediction: 1.13x if decode is op-bound,
1.5x if byte-bound.

## window flash-mtp4 (2026-09-18) — MTP depth 4 is a LOSS for GLM-5.3-Flash: -22%

First MTP-depth test on Flash (Qwen runs MTP4; Flash production is MTP2, never swept).

| arm, bench3 192 tok | p1 | p2 | p3 | mean | draft acc p1/p2/p3 |
|---|---|---|---|---|---|
| MTP2 production | 12.26 | 14.09 | 13.95 | 13.43 | 49 / 65 / 63 % |
| MTP4 arm | 9.60 | 10.87 | 11.08 | 10.52 | 30 / 38 / 39 % |

Method: production stopped clean via systemd, arm launched from a launcher copy with
only --spec-draft-n-max 2->4 (same CTX/PARALLEL/BATCH/UBATCH/libs/quiet.sh/tg warmup).
bench3 seed 42 temp 0 cache_prompt false. Parity IDENTICAL x3 — greedy verification is
exact across draft depths, as expected, so this is pure speed.

Verdict: NOT PROMOTED. Acceptance falls with depth while each cycle costs more: the MTP
draft's positions 3-4 mostly reject. MTP3 not run — monotonic decline makes a win very
unlikely; revisit only with a stronger draft model. Production restored under systemd,
Paseo end-to-end re-verified OK.

Ops note: the arm was slow to die after SIGTERM, so the systemd restore briefly overlapped
two loaded servers. No OOM (192G free throughout), but future windows should port-wait
after kill before starting production.

## window flash-fusion (2026-09-18) — MoE/FFN fusion flags change outputs, no speed win: NOT PROMOTED

First test of the engine fusion env knobs on Flash (all default off, never swept):
GGML_CPU_MOE_GATE_UP_FUSION + MOE_WEIGHTED_SUM + MOE_DOWN_WEIGHTED_SUM + FFN_GATE_UP, all =1.

| arm, bench3 192 tok | p1 | p2 | p3 | mean | acc p1/p2/p3 | parity |
|---|---|---|---|---|---|---|
| base (no fusion) | 12.40 | 14.42 | 14.17 | 13.66 | 49 / 65 / 63 | - |
| fusion arm | 13.23 | 14.18 | 12.18 | 13.20 | 53 / 63 / 47 | DIFFER x3 |

Parity differs on all three prompts (chars 9 / 89 / 1): the fused kernels reorder
floating-point work enough to flip greedy near-ties almost immediately, so every speed
comparison is confounded by changed text paths and acceptance. Same-text signal: none
(p2 shares 89 chars, -1.7%, noise).

Verdict: NOT PROMOTED. No positive speed signal to justify quality-gating changed
outputs (battery/perplexity). No bisect: four more restarts to isolate which flag flips
numerics is not worth it without a win on the table. If revisited, gate on perplexity
first, speed second.

Method note: stop-by-port.sh + port-wait before systemd start = clean handoff, no
overlap this time (cf. flash-mtp4 window).

## window flash-optrace (2026-09-18) — FIRST production op trace for GLM-5.3-Flash + ranked targets

Captured non-disruptively via the launcher hook (touch /dev/shm/flash-optrace.arm, one
24-token completion, disarm). Raw: results/flash-optrace-prod-20260918.log (4710 lines).
Note: profiler emits barrier= fields that optrace.py predates; strip with sed before parse.

Verify cycle, 150.3 ms steady (179.6 first/cold), ~1.96 tok/cycle:

| cost center | ms/cycle | share | rate vs roof | verdict |
|---|---|---|---|---|
| MUL_MAT_ID experts (42 lyr x gate/up q4_K + down q5_K, 288 exps, 8 active) | 53.1 | 35% | ~24 GB/s/socket vs 88 demonstrated | UNDER-OPTIMISED, target 1 |
| MUL_MAT dense (hc_mixes, kda/dsa out, router, lm_head) | 44.5 | 30% | lm_head at 93% local STREAM | mixed, needs per-family cut |
| CPY recurrent state (34 x 3.1 MB cache_s) | 20.4 | 13% | 4.3 GB/s, coherence-bound | target 2 |
| CUSTOM cross-socket reduce | 10.8 | 7% | NT already on | small |
| GATED_DELTA_NET + rest | ~21 | 14% | - | small |
| barriers (in bar= column, not time=) | ~9 | 6% | spread thin | not a lever |
| draft passes (89-node graphs) | ~6 | 4% | - | not a lever |

Target 1 — MUL_MAT_ID kernel (+30% if fixed to dense rates): per-(token,expert)
single-column calls on 512-wide slices; the x16 kernel family (Qwen plus-80% source)
does NOT exist in this engine (zero x16 refs in ggml-cpu.c). Port + retune = days.

Target 2 — mirrored CPY (+10-13%): recurrent state intentionally mirrored, so all 4
sockets write identical 3.1 MB to the same physical pages every token (coherence storm
arithmetic: 48K lines x hops ~= measured 20 ms). Fixes need scheduler/memory surgery:
single-writer CPY (one socket writes, needs barrier proof) or head-sharded state
(zero sharing, best, biggest change). No clean env seam exists; handle_cpy defers to
generic. Hours, medium risk, parity-gated.

Ruled out this window: NUMA placement already ideal (4 x 47.6 GB shards, one per node —
no interleave win); MoE/FFN fusion flags change outputs with no speed win (see
flash-fusion window); MTP4 loses -22% (see flash-mtp4 window); threads swept (w69).

Next session starts at target 1 or 2 with this trace as the baseline.

## window flash-lctx (2026-09-18) — production decode vs context: steep falloff, new data

lctx.py on live :18131 (MTP2, 1M ctx, N_PREDICT 96, non-repeating filler):

| ctx | prefill tok/s | decode tok/s | ms/token | draft acc | tok/cycle |
|---|---|---|---|---|---|
| 190 | 69.2 | 15.89 | 62.9 | 83% | 5.41 |
| 7,600 | 67.3 | 11.57 | 86.5 | 82% | 5.33 |
| 30,400 | 42.1 | 5.91 | 169.1 | 76% | 5.12 |

Context term is ~3.4 us/token, linear across the measured range: 62.9 -> 86.5 -> 169.1 ms.
Naive extrapolation: ~2 tok/s at 128K, ~1.1 at 256K. The old 17.6-at-128K figure (route
script crossover table) is definitively dead for this recipe. Mechanism unknown: KV bytes
alone predict only ~7 ms at 30K (12 layers x 24 KB/token x 30K at 100 GB/s), so the
+106 ms is elsewhere (indexer blocks? seq_rm scans? state traffic?). Worth an op trace
at 32K before believing any mechanism. Routing consequence: smeagol_advise_model and the
Paseo-model note both need re-deriving with this curve.

## window flash-aggregate (2026-09-18) — concurrency sweep: peaks 28.09, spec-off wins past c=2

CTX=262144 PARALLEL=16 arms, 190-token prompts, 64 tokens each, barrier-started wall clock:

| c | spec ON wall agg | spec OFF wall agg | per-stream (winner) |
|---|---|---|---|
| 1 | 14.74 | 11.26 | 14.74 (spec on, 1.31x) |
| 2 | 14.91 | 15.90 | 7.95 |
| 4 | 18.20 | 20.64 | 5.16 |
| 8 | 19.48 | 23.97 | 3.00 |
| 16 | 21.49 | 28.09 | 1.76 |

Same pattern as Qwen w83 (speculation turns liability with concurrency) at smaller
magnitude: Flash peaks 28.09 vs Qwen 46.74. Crossover at c=2. Per-stream latency
collapses (1.76 at c=16), so this is a multi-agent throughput posture, not an
interactive one. Production (PARALLEL=2, MTP2) unchanged: correct for 1-2 agents.

Ops note: --spec-type none WITH leftover --spec-draft-* flags aborts at load
(GGML_ASSERT suffix_fallback in llama-model.cpp:511). Drop all spec-draft flags.

## window flash-topkhist (2026-09-18) — pool_score histograms REFUTE the zero-mass fast path

Env-gated probe in top_k kernel (glm-hist override lib, GGML_TOPK_HIST; relink verified
1026=1026 symbols, bench3 identical 12.38/14.52/14.09). Row = one query, thread-0 only.

Short ctx (12,892 rows): k == N == 66 on EVERY row — the sort is a take-all no-op.
Long ctx (1,804 rows, npool 1026 -> 7554): k = 512 always. Sample row: eq0=0 neg=499
ninf=258 pos=269. Zero NaN. pos < k in 74% of rows.

Verdict: pool scores are general f32 (neg/pos/-inf), NO zero mass point. The ReLU-zero
story is Qwen-specific; glm5next sums head-weighted relus with sign-free weights, so
exact zeros are rare. Approach 2 (threshold + zero fill) is DEAD for this model. Radix
select (Approach 1) remains viable but the prize is small (see 32K trace window).

## window flash-trace32k (2026-09-18) — the context term is the indexer chain, not TOP_K

Same-hook op trace at ~30K ctx (prompt-cache warm + arm). Verify graph 150.3 -> 357.7 ms.
Deltas: GET_ROWS +51, CONT +52, ADD +45, SOFT_MAX +26, FLASH_ATTN_EXT +16,
LIGHTNING_INDEXER +11, CUSTOM +10. MoE/dense/CPY all FLAT. TOP_K = 4.0 ms (2%).

So the +207 ms context term is the kpool indexer re-deriving every pool summary every
token plus attention over the selection — the same disease as Qwen w71-80, in glm5next
form. The fix is a pooled-summary cache (update dirty pools only), NOT top-k
parallelism: top-k is worth 2-4% even if perfected, the cache attacks ~190 ms.
Queued as the next engine project with the Qwen w80 design as the template.

## window flash-splitdbg (2026-09-18) — GGML_META_DEBUG itself aborts; chain is mirrored by timing

Wanted per-node split states for the pool chain. GGML_META_DEBUG=1 aborts at load:
ggml-backend-meta.cpp:1374 GGML_ASSERT(split_state.n_segments == 1) — the debug print
assumes single-segment and dies on the first multi-segment tensor, printing zero lines.
(No rebuild recipe exists for the glm-fix base lib, so patching the print means
reverse-engineering the base relink; not worth it — see below.)

Moot: timing arithmetic says the whole pool chain is MIRRORED. Per-socket GET_ROWS does
31 MB in 4.8 ms (6.5 GB/s) and CONT 31 MB in 2.4 ms — far too slow for local split work,
exactly the mirrored multi-socket coherence signature already measured on CPY (4.3 GB/s)
and consistent with the engine comment that GLM DSA mirrors MLA/indexer/attention while
splitting only FFNs. Each socket redundantly gathers/transposes/scores all pools.

Consequence for the fusion design (GLM5NEXT-POOL-FUSION-DESIGN.md): no split reasoning
needed. A per-socket fused kernel computing pool_k from the same inputs is correct by
construction (same function, same inputs); the 4x redundancy remains but wall time
drops ~190 ms to ~3 ms. Implement as ggml-cpu subgraph fusion (moe_gate_up_fusion
pattern at ggml-cpu.c:3729), NOT a new core op: no meta split-rule changes required.

## window flash-mmidcheck (2026-09-18) — x16 expert port CANCELLED: experts already at bandwidth

Pre-port audit of the flash-optrace expert math, before touching any code. The
"25% efficiency, 53 ms prize" premise is refuted by the trace's own token counts.

Token count: the steady verify graph processes **4 tokens**, not 1. MUL_MAT_ID
trace lines don't show tokens; the MoE CLAMP shapes do:
`ffn_moe_gate_clamped-3 [512,8,4]` = [N-shard, n_used, n_tokens]. (4 = ~2
accepted + 2 MTP drafts, consistent with 1.96 tok/cycle; cold graph shows 6.)

Expert bytes (per socket, per cycle): gate/up q4_K [4096,512] = 1.18 MB,
down q5_K [512,4096] = 1.44 MB per (expert, socket); 4 tok x 8 exp = 32 pairs
x 42 layers = **5.1 GB**. 5.1 GB / 53.14 ms = **96 GB/s/socket ~ 96% of
STREAM** (STREAM ~100, calibrated by lm_head: 168 MB in 1.811 ms = 93 GB/s =
93%). Cross-check: GLM per-layer expert ms is 2.2x Qwen's, exactly the expert
byte ratio (121 vs 55 MB) — same bytes/ms as Qwen's at-bandwidth experts.

Engagement: no port was ever needed. Production's x16 select (fleet-0903
repack.cpp:7815+) covers Q4_K/Q5_K/Q6_K with no nd/name gate, and production
env sets all three flags — gate/up/down are selected. The "zero x16 refs in
ggml-cpu.c" grep was a red herring: x16 lives in repack.cpp/repack-x86.cpp.
Consistent: the 09-14 q8all arm (glm-gemm repack, same x16 mmid generation)
was neutral + parity IDENTICAL = no path change. (MMID_BATCH exists only in
qwen-ksplit, and Qwen window 19 measured it decode-neutral anyway.) Gate/up run as
separate x16 MUL_MAT_ID nodes (the clamped fusion rejects non-Q8_0, so no
swiglu fusion fires — costs one extra activation quant, <0.1 ms).

Verdict: DO NOT PORT. Ceiling is ~53 -> ~48 ms (~3%), not 13.5 -> 19 tok/s.
Analysis only: zero restarts, production untouched. Baseline reconfirmed this
window: bench3 12.44 / 14.47 / 14.23 (mean 13.71).

Live levers instead: CPY target 2 (20.4 ms mirrored-state coherence, unaffected
by this correction), dense per-family cut (44.5 ms "mixed" — K-split/fast paths
may already cover parts), gate/up clamped-K-quant fusion (~1-3 ms, parity-gated,
likely not worth it alone), and the 32K indexer fusion design (separate doc).
