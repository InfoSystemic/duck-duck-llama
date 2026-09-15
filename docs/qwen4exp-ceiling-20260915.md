# Qwen3.8-Flash-Next on a 4-socket Xeon: where the ceiling actually is

*Measured 2026-09-14/15 on a Lenovo SR950, 4x Xeon Gold 6242 (Cascade Lake, AVX-512 VNNI, no AMX), 755 GB DDR4,
381 GB/s aggregate measured. Every figure is marked measured or modelled.*

## 1. The decode curve, which nobody had

Every benchmark in this project — and essentially every published CPU inference number — is taken at ~200 tokens of
prompt. Agents run at 10K-100K. The full curve, production configuration, **measured**:

| context | decode tok/s | ms/token | draft acceptance | prefill tok/s |
|---:|---:|---:|---:|---:|
| 190 | 24.31 | 41.1 | 55% | 111.5 |
| 7,600 | **26.29** | 38.8 | **89%** | 114.0 |
| 30,400 | 14.20 | 70.4 | 73% | 68.9 |
| 95,000 | 8.08 | 123.7 | 75% | 43.4 |
| 121,600 | 6.41 | 156.0 | 70% | 38.0 |

Fitting only the two longest points gives **41.9 ms + 0.939 us per context token**, which extrapolates to 288
ms/token — **3.47 tok/s at the native 262,144**. An independent three-point fit gave 294 ms / 3.40 tok/s. Two fits
agreeing to 2% is why we stopped short of spending 3.2 hours measuring the last point directly, and spent it on
experiments that could change the answer instead.

Two things fall out that the 200-token benchmark cannot show.

**The 200-token figure understates the machine.** Peak throughput is at ~7.6K context, not at 200, because the MTP
draft head needs context to predict well: acceptance climbs 55% -> 89%. Short-prompt benchmarks are not a best case.

**Decode is close to linear in context** beyond that: roughly 30 ms fixed plus ~1.0 us per context token. Extrapolated
to the native 262,144 window that is ~294 ms/token, or **3.4 tok/s** — which is why "native context support" has never
been usable in practice, on any of the levers we tried.

## 1b. 256K has TWO barriers of comparable size

A 256K deployment has to fill the context as well as generate from it, and both are far from their limits:

| barrier | today | distance from physical limit |
|---|---:|---|
| **fill** a cold 256K context | **3.2 hours** (22.5 tok/s prefill, fitted) | 3.7 PFLOP at 5.7 TFLOP/s peak is 11 min -> **6% of peak** |
| **generate** from it | **3.47 tok/s** | context path at 0.0023 us/ctx-token floor vs 0.939 measured -> **0.2%** |

They have different causes. Generation is overhead-bound and software fixes it. Filling is compute-bound, which is
what a GPU has in surplus and this box does not — a V100 at 30% efficiency turns 194 minutes into about 15. So the
two levers address different halves rather than competing, which dissolves the sequencing question.

**Reaching 30 tok/s at 256K requires both.** The fixed per-token cost is 41.9 ms, so even with zero context cost the
ceiling is 23.9 tok/s. The indexer work alone lands near 22; the dense path moved off the CPU alone lands near 24;
together, past 50.

## 2. The machine is overhead-bound, not bandwidth-bound

Read directly from the uncore memory controllers (`tools/dram_per_token.py`), **measured**:

| | speculative | raw |
|---|---:|---:|
| DRAM read per generated token | 5.71 GB | 9.68 GB |
| sustained bandwidth | 134 GB/s = **35% of 381** | 161 GB/s = 42% |
| ceiling from those bytes | 67 tok/s | 39 tok/s |
| achieved | 23.42 (35% extraction) | 17.73 (45%) |

**65% of this machine's memory bandwidth is idle during decode.** There is 2.85x available without moving one byte
less, which is why the remaining programme is about removing work rather than removing bytes.

The measured 1.70x saving from speculation (9.68 -> 5.71 GB/token) matches the tensor-table prediction of 1.68x to
within 1%, so the model in `tools/active_bytes.py` is structurally right; both absolutes run ~39% high because the KV
cache, the indexer cache and mirrored tensors read once per socket are invisible to a tensor table.

## 3. NUMA tensor parallelism is worth 4.07x

Same engine, same model, varying only how many NUMA nodes the model is split across, **measured**:

| sockets | speculative | raw |
|---:|---:|---:|
| 1 | 6.37 tok/s | — |
| 2 | 16.63 | 13.25 |
| 4 | **25.94** | 14.51 |

Raw decode gains only 1.10x from 2 sockets to 4: a batch-1 graph has too little work per barrier. Speculation and
NUMA parallelism are complementary — speculation is what gives the extra sockets something to do. Upstream llama.cpp
has no multi-device CPU tensor split, so this scaling is unavailable there.

## 4. Where the long-context time goes

Per-node trace, raw decode, same 6,156-node graph at both lengths, **measured**:

| op | ms @200 | ms @32,000 | growth | share |
|---|---:|---:|---:|---:|
| GET_ROWS | 1.1 | 36.2 | +35.1 | 40% |
| ROPE | 0.2 | 14.9 | +14.7 | 17% |
| TOP_K | 0.3 | 8.1 | +7.8 | 9% |
| MUL_MAT | 27.7 | 33.6 | +5.9 | 7% |
| FLASH_ATTN_EXT | 0.3 | 4.2 | +3.9 | 4% |

The attention itself is 4.2 ms. The machinery deciding *what to attend to* is 51. The sparse-attention indexer
re-derives all 7,616 block summaries every token — gathering every cached key row, pooling, norming and roping them —
when exactly one block has changed. Cacheability is verified: block position is `b*r`, a pure function of block index
(`llama_memory_hybrid_idx_context::set_input_qsa`), so roped pooled keys cannot go stale from position drift.

**Unresolved:** a standalone benchmark of that same pooling work measures 11.3 ms/token where the trace attributes
~46. Profiling overhead does not explain it (TOP_K agrees between trace and benchmark to 9%), and NUMA placement does
not either (sampled at 1.02-1.11x imbalance throughout). That 4.1x gap is the most interesting open question here.

## 5. What we refuted, including our own work

Publishing these because the failures were more instructive than the successes, and every one cost real time:

| claim | killed by |
|---|---|
| attention FLOPs explain the long-context penalty | arithmetic: would need 0.31% of peak |
| top-k is ~57% of it | microbenchmark: the model was 3-7x pessimistic |
| top-k is ~28% of it | op trace: it is 9% |
| `std::nth_element` is the top-k fix | benchmark: 1.66x *slower* at 262K |
| fusing the indexer pool is ~125x | benchmark: 1.0-1.3x |
| draft acceptance decays with context | measurement: 75% at 95K |
| NUMA placement explains the gather cost | sampler: 1.02x imbalance |
| the histogram top-k is behaviour-preserving | end-to-end A/B: 7-17% *slower* net |

The last one is the sharpest. The histogram selection makes the kernel 4-7% cheaper per cycle and the model 7-17%
slower, because masked (`-INFINITY`) scores make the tie tail arbitrary and the model then attends to different
positions. **Upstream's own `test_top_k` explicitly permits this** — "when there are ties, the indices can be
different but the input values they correspond to should be the same" — so a change can be conformant to the op's
contract and still change model behaviour. That gap is general, not specific to our patch.

## 6. Method notes that earned their keep

- **A control arm in every A/B.** Twice it caught a conclusion that was about a broken library rather than the
  variable under test. An A/B whose arms all fail proves nothing.
- **A lineage gate before building.** Byte-comparing a rebuilt object against the production one caught a build
  pointed at the wrong source, 195 lines different, which would have been blamed on the change under test.
- **Falsifiable predictions written before the data.** Four hypotheses died within an hour of being stated.
- **Silence is not success.** Three self-inflicted silent failures in one day: a verification that reported OK having
  run zero tests, an `awk` error discarded by `nohup`, and a monitor filter that would have missed a crash.
- **Benchmark what the model actually feeds the kernel.** The top-k microbenchmark used pure normals and declared the
  patch exact and safe; real scores are heavily masked, and the patch both crashed and changed behaviour.

---

## 7. Thread count is closed (09-15)

Production libraries, identical configuration, only threads per socket varied. Draft acceptance (73%) and
tokens per speculative cycle (4.00) were identical in every arm, so the arms are behaviourally the same and
only the thread count moved.

| threads/socket | 190 ctx tok/s | 30,400 ctx tok/s |
|---|---:|---:|
| 15 | 24.61 | **15.89** |
| 16 | 24.06 | 15.50 |
| 30 (hyperthreading) | 24.15 | 15.31 |

Total spread is 3.8% at 30K, and hyperthreading is the **worst** arm rather than merely neutral — consistent
with the aggregate bandwidth being real and already saturated by the physical cores. Leaving one core per
socket free edges out using all of them.

This closes thread count as a route to 30 tok/s: it is worth 4%, and roughly 2x is required. It is the ninth
hypothesis to die to measurement here, after attention FLOPs, top-k selection cost, `nth_element` as a
replacement, kernel fusion, acceptance decay with context, NUMA placement, and histogram top-k.

## 8. The indexer's pooled block keys are cacheable, and the storage is already allocated

The sparse-attention indexer rebuilds every block summary on every token. Tracing decode from 200 to 32,000
tokens of context attributes the growth as `GET_ROWS` +35.1 ms, `ROPE` +14.7 ms, `TOP_K` +7.8 ms of 87.2 ms
total — while `FLASH_ATTN_EXT`, the actual attention, costs 4.2 ms. **The machinery that chooses what to
attend to costs an order of magnitude more than attending.**

Three findings make a cache for it much cheaper than expected.

**The roped value is invariant.** Block `b`'s rope position is assigned as `b*ratio` — a pure function of the
block index, independent of the current token position and of `n_kv`. So the pooled, normed, *roped* key for
block `b` changes only when a new token lands in block `b`. The cache can hold the post-rope value, which
puts both the `GET_ROWS` and the `ROPE` growth — about 57% of the measured context-dependent cost — behind
it, rather than only the gather.

**The dirty set needs no bookkeeping.** Blocks are defined over positions (`block b covers
[b*ratio, (b+1)*ratio)`), and appended tokens always hold the highest positions, so the blocks an ubatch can
touch are always the trailing `ceil(n_tokens/r) + 1`. That is derivable from shapes alone: no index tensor,
no `set_input` change, no memory-subsystem change.

**The storage already exists, unused.** The indexer's KV cache is built as a full cache with a `type_v`, but
the graph's entire use of it is `build_input_k_idxs` / `cpy_k` / `get_k` / `get_n_kv`. `cpy_v` and `get_v`
are never called — the V half is allocated and never touched. It holds 256 elements per cell where the
pooled keys need 128 per block at 4 cells per block: **8x headroom**. Reusing it means the pooled keys
inherit the existing cross-socket mirroring rule, the existing state save/restore, and the existing
sequence-removal path, none of which needs new plumbing.

The stage runs on 12 of 48 layers (`compress_ratios` is non-zero only on every fourth layer; the rest are the
linear-attention path), so the cache is 12 layers deep, not 48.

Net effect: per-token indexer work goes from O(context) to O(r).

## 9. Two more method notes

- **A view of a cache keeps the cache's type.** A measurement-only bypass replaced a `ggml_get_rows` with a
  view of the same tensor and aborted during graph reserve. `get_rows` *always* produces F32; the view
  produced F16, and every downstream consumer — norm, rope, the scoring matmul — received the wrong type.
  Identical shapes are not identical tensors.
- **A crash timestamp is not a crash time.** That abort printed its last line 34 ms in, which looked like a
  failure during library load. It actually died about 2.7 minutes in, at graph reserve; the intervening log
  was buffered and lost with the process. Read the wall clock between the surrounding events, not the last
  line of the log.

## 10. The fixed cost, decomposed — and an honest reframing of the target

Everything above concerns how decode degrades with context. This section is the part that does not depend on
context at all, traced from the 200-token verify graph (53.5 ms, 6,156 nodes).

| op | nodes | time | share |
|---|---:|---:|---:|
| MUL_MAT | 580 | 27.7 ms | 52% |
| MUL_MAT_ID (the actual expert FFN) | 96 | 7.2 ms | 13% |
| CUSTOM (cross-socket fused reduces) | 191 | 6.7 ms | 13% |
| all other ops | ~5,289 | ~11.9 ms | 22% |

**The router costs 11% of the graph by itself.** `ffn_moe_logits` (`ffn_gate_inp`, `[2560,512]`) takes 123 µs
per layer, 5.88 ms across 48 layers. `ffn_gate_shexp` at `[2560,640]` — a *larger* matrix — takes 49 µs. The
ratio tracks bytes, not arithmetic: the router is **F32** (5.24 MB) and the shared-expert weight is Q6_K
(~1.34 MB). 3.9x the bytes for 2.5x the time. The twelve indexer `q_proj` projections share the shape and are
slower still (~180 µs each). About 8 ms of the fixed 53.5 — 15% — is a handful of F32 projections.

Converting them would return roughly 4 ms of graph. That is worth having but is not decisive, and it moves
both expert routing and indexer selection, so it inherits the lesson from the refuted top-k patch and needs an
A/B that controls draft acceptance rather than one that only reads tok/s.

**Bandwidth is not yet the binding constraint.** 6.96 GB of active weights per token against 381 GB/s measured
is an 18.3 ms floor — a 55 tok/s ceiling. Decode runs at 41 ms/token, i.e. 34% of the byte-floor pace. The
missing two-thirds is per-node overhead across a 6,156-node graph, not memory traffic. This is the clearest
statement of why this system is op-bound rather than bandwidth-bound.

**Speculation's value grows with context.** The traces above were taken with `--spec-type none`, so each
6,156-node graph is one *raw* decoded token; the four profile records per round are the four sockets
(cpu 0/16/32/48) executing the same graph, not four separate graphs. Comparing those raw figures against the
speculative runs at matching context:

| ctx | raw (traced) | speculative | speedup |
|---|---:|---:|---:|
| 200 | 53.5 ms/token = 18.7 tok/s | 24.31 tok/s | **1.30x** |
| ~32K | 138.7 ms/token = 7.2 tok/s | 15.89 tok/s | **2.20x** |

Speculation returns 1.3x at short context and 2.2x at long, tracking draft acceptance as it rises from 55% to
73%. It pays *better* at length — the opposite of the intuition we started with, and of one of the hypotheses
that died here.

**The reframing.** No configuration in this project has reached 30 tok/s at *any* context length; the best
measured figure is 24.61 (15 threads/socket, 190 tokens). A 30 tok/s target at 256K therefore requires beating
the current short-context number by 22% *and* removing essentially all context scaling. Those are two
independent problems, and only the second currently has a designed fix.

## 11. REFUTED BY MEASUREMENT — the sparse attention is load-bearing, and the reasoning below was circular

> **Read this first.** The section that follows argued that the sparse-attention indexer is a net loss on this
> hardware and that full attention would be worth 4–9x at 256K. **It was tested and it is wrong.** Disabling
> the indexer made decode **2.2x slower** (7.41 vs 16.30 tok/s at 28,880) and collapsed draft acceptance from
> 76% to 37%.
>
> **The error was logical, not arithmetic.** The "18x cost ratio" compared the indexer's cost against
> `FLASH_ATTN_EXT`'s measured cost — but that is what attention costs *after* the indexer has already
> restricted it to 8,192 positions. It compared the price of an optimisation against the price of the thing it
> had already optimised. The correct comparison is the indexer against attention **without** it:
>
> | at 28,880 | measured |
> |---|---:|
> | full attention | **~150 ms/token** |
> | my estimate | 12.4 ms (**12x off**) |
> | KV streamed | 710 MB at **4.7 GB/s — 1.2% of machine bandwidth** |
>
> Flash-attention decode on this CPU runs at roughly one percent of memory bandwidth. Extrapolated to 237,500
> tokens, full attention would stream 5.84 GB/token and cost **~1,242 ms**, against the indexer's ~499 ms —
> so the indexer **saves** ~742 ms/token, a **2.5x win**. On this machine the sparsity is not a mistake
> inherited from GPU design; it is the only thing making long context tractable at all.
>
> A second, independent refutation: acceptance fell 76% → 37%. The MTP draft head's agreement with the target
> depends on the sparse attention pattern, so even a free change of that pattern would cost 1.6x in tokens per
> speculative cycle.
>
> The section is kept below unedited, because the reasoning is a clean example of a failure mode worth
> recognising: every number in it was measured, and the conclusion was still wrong.

## 11a. (REFUTED — retained) The original argument



This is the conclusion the rest of the document has been circling.

A sparse-attention indexer exists to make attention cheap: score every block of the context, select the top
`k`, attend only to those. On this hardware the selecting costs vastly more than the skipping saves.

Of the 87.2 ms/token of traced growth between 200 and 32,000 tokens of context:

| | ms | share |
|---|---:|---:|
| `GET_ROWS` — gather every cached indexer key | 35.1 | 40% |
| `ROPE` — re-rotate every block summary | 14.7 | 17% |
| `TOP_K` — select 2048 of n_blocks | 7.8 | 9% |
| `MUL_MAT` (all 585 nodes, scoring is a slice of this) | 5.9 | 7% |
| `CONT` — the r slice-copies in the pooling loop | 5.7 | 7% |
| `ADD` — the pooling sums | 4.0 | 5% |
| `CUSTOM` — cross-socket reduces | 3.1 | 4% |
| `MUL_MAT_ID` — expert FFN | 1.8 | 2% |
| **the indexer's share of the above** | **~71** | **~82%** |
| `FLASH_ATTN_EXT` — the attention itself | 3.9 | 4.5% |

**The machinery that decides what to attend to costs ~18x the attention it saves.**

> **Correction.** An earlier revision of this table listed "scoring `mul_mat` ~25.7 ms, 29%" and concluded
> 21x. That row was a *residual* — the 87.2 ms total minus the four ops that had been named — attributed to a
> single op that was never measured. Reading the actual op table shows `MUL_MAT` grows **+5.9 ms** across all
> 585 of its nodes, so the scoring step is cheap; the residual was really `CONT` +5.7 and `ADD` +4.0 (the
> slice-copies and sums inside the pooling loop) plus small growth in the reduces and the expert matmuls. The
> conclusion is unchanged, and `GET_ROWS` is if anything a cleaner target: at 40% it is larger than the next
> four items combined.

And the asymmetry compounds in the worst possible direction:

- **Attention cost is capped.** `top_k = 2048` blocks is 8,192 positions whether the context is 32K or 256K,
  so the attention term stops growing past ~32K.
- **The indexer's cost is linear in context.** It re-derives every block summary on every token.

So the longer the context — precisely the regime a 256K model exists for — the worse the trade becomes. By
the fitted raw-decode curve (`53.5 + 2.679e-3·ctx` ms/token, which reproduces the measured 121,600 point
within 10%), the context-dependent term at 262,144 is **702 ms/token**, of which ~670 ms is indexer.

What does that 670 ms buy? Avoiding a full read of the KV cache, which is:

    12 attention layers × 262,144 positions × 2 kv-heads × 256 dims × 2 bytes × 2 (K and V) = 6.44 GB/token

**17 ms** at the measured 381 GB/s aggregate, or **~124 ms** at the much poorer effective rate
`FLASH_ATTN_EXT` achieves today. Either end of that range is far below 670 ms.

**This is not a defect in the model.** On the hardware it was designed for, both premises hold: a GPU has the
bandwidth to make attention the bottleneck, and enough parallelism to make the indexer nearly free. On a
4-socket CPU with 381 GB/s, bandwidth is scarce but a long serial chain of small ops is *expensive*. Both
premises invert, and the optimisation inverts with them.

The test is one line — pass the plain causal mask to the attention call, which makes the entire indexer chain
unreachable so the graph never builds it. The gate is deliberately **not** speed: full attention is a superset
of what the indexer selects, but the model trained with the sparse pattern, so the experiment plants a fact at
25% depth of a long context and asks for it back. Faster *and* retrieves it means the sparsity is a net loss
here. Faster but *misses* it means the sparsity is load-bearing and the idea is dead.

### The pattern across thirteen refuted hypotheses

Everything that has failed here aimed at **arithmetic** or **bytes**: attention FLOPs, top-k selection cost,
`nth_element`, kernel fusion, NUMA placement, the elementwise megakernel, the F32 router requant (tested and
noise — "overhead-bound, not byte-bound"), expert requantisation, draft depth, thread count, hyperthreading.
This system is bound by neither. The only two ideas that have survived scrutiny attack **the amount of work
performed per token at length** — which is the single axis that has ever moved a number.

## 12. Measured at 237,500 tokens — the model validated 8x outside its fitting range

Every 256K figure above this section was extrapolated from traces at 32,000 and 121,600. This one is measured:
a single 237,500-token context was prefilled, decoded, and saved to disk.

| | measured | predicted by the fitted curve | error |
|---|---|---|---|
| decode | **3.40 tok/s** (293.9 ms/token) | 3.19 tok/s | 6% pessimistic |
| prefill | **26.6 tok/s** — 237,500 tokens in 8,919 s (**2.48 h**) | 24.2 tok/s | 9% pessimistic |
| draft acceptance | 63%, 3.56 tokens/cycle | — | — |
| saved slot | **8.16 GB**, written in 6.8 s | — | — |

`raw(ctx) = 53.5 + 2.679e-3·ctx` predicts 690 ms/token raw at this length; the measured speculative 293.9 ms
implies a **2.35x** speculative ratio against the 2.20x assumed. Both the decode ladder and the prefill model
survive validation eight times beyond the range they were fitted on, which is the strongest available
evidence that the projections here are sound rather than merely self-consistent.

Refitting prefill on the measured average (37.6 ms/token) gives `k = 2.271e-4`, so a full 262,144 fill is
**2.94 h** rather than the 3.24 h quoted earlier from the two-point fit.

**Draft acceptance is not monotonic in context.** 55% at 200 tokens, 73% at 30,400, **63% at 237,500**, with
tokens per cycle falling 4.00 → 3.56. Speculation pays increasingly well from short to medium context and
then gives some of it back at extreme length. An earlier section here said simply that it "pays better at
length"; that is true only up to a point.

### The cheapest real win in the project needs no engine change

A context that costs 2.48 hours to build serialises to 8.16 GB — 33.5 KB per token (24.0 KB attention KV,
6.0 KB indexer, the rest recurrent state) — and writes in 6.8 seconds:

    POST /slots/0?action=save     {"filename": "f16-250k.bin"}
    POST /slots/0?action=restore  {"filename": "f16-250k.bin"}

The intent is that for any fixed long prefix — a codebase, a corpus, a document set — the 2.48 h is paid once
and every later session restores it.

> **Correction (tested, 08:48).** The artifact **loads** but was **not usable**, so this claim is withdrawn
> pending a fix. Restore itself works exactly as advertised: 5.2 s, `n_restored: 237595`, 1.56 GB/s off disk,
> and the handler does repopulate the prefix state (`server-context.cpp:2613`,
> `slot->prompt.tokens = std::move(restored)`). But a completion that resent the original prompt with
> `cache_prompt` **never returned in 100 minutes** — roughly two thirds of a cold 237K prefill, so the server
> was rebuilding from scratch rather than reusing what had just been loaded.
>
> The leading hypothesis is that `llama_state_seq_load_file(ctx_tgt, ...)` restores the **target** context
> only, while speculative decoding also needs draft-side state the snapshot never captured. That predicts the
> restore works with speculation disabled and fails with it on — a cheap, decisive test, now queued. Two
> alternatives are not yet excluded: a prefix-comparison mismatch (the slot holds prompt+generated, 237,595,
> against the 237,500 resent), or a tokenisation difference at the head of the prompt.
>
> Until that reports, a 250K snapshot is an 8.16 GB file that loads in five seconds and saves nothing. The
> save/restore *mechanism* is sound; whether it delivers a usable context is unproven.

Two operational notes. The server builds the path as `slot_save_path + filename`, a plain concatenation with
no separator, so the path argument **must** carry a trailing slash or the file lands next to the directory
rather than inside it. And a saved slot is tied to the cache geometry that produced it: a different KV type,
context size, or slot count will not load it.

## 13. GET_ROWS runs on one thread of sixty — worth +12%, bit-exact, and 33x less than projected

`GET_ROWS` is 40% of the context-dependent decode cost, larger than the next four ops combined. Measured at
32,000 tokens it moves 8.2 MB per layer in 2.93 ms — **2.80 GB/s, 0.74% of this machine's 381 GB/s** — at
**91 ns per gathered row**, which is one DRAM latency with no memory-level parallelism. That is the signature
of a single thread, and `ggml_get_n_tasks()` confirms it:

```c
        case GGML_OP_GET_ROWS:
        case GGML_OP_SET_ROWS:
            {
                // FIXME: get_rows can use additional threads, but the cost of launching additional threads
                // decreases performance with GPU offloading
                n_tasks = 1;
            } break;
```

Both ops are pinned to one thread, and the stated reason — GPU-offload launch cost — does not apply on a
CPU-only build. Meanwhile `ggml_compute_forward_get_rows` **already splits by rows** (`dr = (nr + nth - 1)/nth`
in every type variant), so the kernel is parallel-capable and correct for any `nth`. This is the same defect
shape as the UNARY ops being pinned to one thread: the implementation honours `ith/nth` and the scheduler
refuses to supply them.

Threading it above a row threshold, so small gathers keep the single-task batching path:

| arm | 190 ctx | 28,880 ctx | acceptance | tok/cycle | greedy hash |
|---|---:|---:|---:|---:|---|
| production | 24.04 | 16.64 | 76% | 4.13 | `13b7ea22` |
| patched lib, flag off | 24.06 | 16.28 | 76% | 4.13 | `13b7ea22` |
| **threaded above 256 rows** | 24.33 | **18.44** | 76% | 4.13 | `13b7ea22` |

**+12.0%** against the control mean, on a 2.2% noise floor. Identical greedy hash, identical acceptance,
identical tokens-per-cycle — three independent confirmations that the change alters only speed. The 190-token
row is a built-in placebo: at 200 gathered rows the threshold does not engage, and nothing moves.

### The projection was wrong by 33x, and why that matters

Bands recorded *before* the run: ≥20 tok/s a large win, 17.5–19.5 partial, ~16.5 dead. It landed at 18.44 —
partial. Backing out the implied cost, `GET_ROWS` went 31.7 → 17.6 ms: a **1.80x** parallel speedup, not the
8x called realistic or the 60x of full threading.

The diagnosis was right and the remedy was half wrong. 91 ns/row *is* absence of memory-level parallelism —
but the cure for a latency-bound gather is more outstanding loads, and sixty threads each stalling on their
own random 256-byte read only recovers 1.8x. Thread count never recovers a latency problem linearly. The
honest lesson is that "single-threaded" and "slow because single-threaded" are different claims, and only the
first was established by the 91 ns measurement.

At 237,500–262,144 tokens the same 1.80x on the 288 ms gather term takes raw 756 → 628 ms and decode
**3.40 → ~3.74 tok/s**. Real, bit-exact, stackable — and a 10% improvement to a number that needs 8x. It
shaves the context term; it does not change its shape.

## 14. Four ways a measurement looked like a result and was not

This section exists because more time went into invalid measurements today than into the optimisations
themselves, and the failures share a shape worth naming: **the harness produced a clean-looking number while
the mechanism under test was never exercised.** None would have been caught by reading throughput alone.

**1. A flag that was set and inert.** Production already exports `GGML_CPU_PARALLEL_COPY=1`. It has done
nothing since the day it was added, because the code path it enables also requires an F32 source (the KV cache
is f16) and `ne[0] / 4096 > 0` — integer division that is zero for every tensor narrower than 4096 elements.
A set-but-inert flag is indistinguishable from a set-and-unhelpful one unless the code announces itself.
Later patches print what they resolved to on startup, and an arm whose announcement is missing is discarded
rather than interpreted.

**2. A control that could not fail visibly.** A needle-retrieval gate ran the *production* arm to an empty
string, because the request omitted `ignore_eos` and used raw-completion format against a chat-tuned model.
Had the treatment arm also returned empty, the two would have agreed and looked like evidence. An A/B whose
control cannot answer measures nothing.

**3. A scorer that could not tell truncated from wrong.** The replacement gate scored 0/3 on the reply
`'1. CRIMSON-MERID'` — the model had retrieved the correct passphrase and been cut off mid-word. The cause is
that Qwen3.8-Flash-Next is a **reasoning** model: its chat template gates on
`enable_thinking is undefined or enable_thinking is true`, so thinking is on by default and consumes the
token budget before the answer. Fixed with `chat_template_kwargs: {enable_thinking: false}`, prefix matching
rather than full-string, and reporting `finish_reason` and `completion_tokens` so truncation is visible.

**4. A harness race that killed arms silently.** Every window restored the production server and exited 20 s
later, while that server was still loading and not yet listening. The next window's port check found nothing
bound, launched its own, and two servers raced for the port and 136 GB each. One arm died this way with
nothing but a CORS banner in its log. Any arm reporting "failed to load" in a chained run must be re-run, not
interpreted.

The general lesson is the one the refuted top-k branch taught first: **validate that the test can detect the
thing it is testing, at the operating point it will run at.** A microbenchmark on unmasked scores, a needle in
a context short enough that sparse attention already reads 28% of it, and a scorer blind to truncation are all
the same error.

## 15. Two changes that stack: +19.4% measured, ~+32% projected at 256K

Both survived scrutiny independently and patch different libraries — one the CPU scheduler, one the graph — so
they were measured together:

| arm @28,880 ctx | tok/s | acceptance | tok/cycle |
|---|---:|---:|---:|
| production | 16.50 | 76% | 4.13 |
| GET_ROWS threading (`libggml-cpu`) | 18.44 | 76% | 4.13 |
| f16 pooled keys (`libllama`) | 17.12 | 81% | 4.31 |
| **both** | **19.53** | 81% | 4.31 |

**+19.4%** against a six-run control mean of 16.35 with a 3.4% spread. The multiplicative expectation is
19.31 and the measurement is 19.53 — 1.1% apart, so the two compose rather than contending for the same
resource. Both patched libraries were verified as actually mapped by reading `/proc/<pid>/maps`; the
threading patch prints no banner, so an unset flag would otherwise have read as "these do not stack".

**The second change was discovered as a control, not as a treatment.** `GGML_Q4E_POOL_CACHE=2` computes the
pooled indexer keys exactly as production does and then casts them to f16. It was built purely to price a
risk — that f16 storage would move top-k selection — before attempting a cache that must store f16. The risk
does not exist: output is byte-identical and draft acceptance *rises* from 76% to 81%, while an f16 `mul_mat`
source halves the bytes read. It needs no cache, no invalidation and no rollback logic.

**The gain grows with context**, because the gather is a larger share of decode at length — 24% of raw time
at 28,880, 42% at 237,500. Projected on the raw-decode curve that was validated to within 6% against the
measured 237,500-token run:

    raw 690 ms → GET_ROWS 261 → 145 ms (1.80x measured) → f16 saves ~48 ms → raw 526 ms
    ⇒ 4.47 tok/s at 237,500, against 3.40 measured — a 1.32x improvement

### Where that leaves the ceiling

| configuration | 256K tok/s |
|---|---:|
| measured today | **3.40** |
| + both validated changes | **~4.5** |
| + pooled-key cache, if the splitter accepts it | ~10.7 |
| 30 tok/s | not reachable CPU-only on this hardware |

The two banked changes are small, independent and low-risk — one bit-exact with byte-identical output across
four arms, the other with an identical greedy hash and better acceptance. Neither depends on the pooled cache
nor on any engine-level file.

## 16. Prefill traced: the growth is attention-mask scanning and cross-socket reduces

Prefill costs 2.48 h for a 237,500-token context and had never been traced. Two ubatches of 1,024 tokens,
at 2,048 and 114,000 tokens of context:

| op | @2,048 | @114,000 | growth | share of growth |
|---|---:|---:|---:|---:|
| `FLASH_ATTN_EXT` | 646 | **11,046** | +10,400 | **40%** |
| `CUSTOM` (cross-socket reduces) | 1,307 | **10,795** | +9,488 | **36%** |
| `CONT` | 118 | 2,241 | +2,123 | 8% |
| `SUM_ROWS` | — | 1,131 | +1,131 | 4% |
| `TOP_K` | 204 | 1,078 | +874 | 3% |
| `MUL_MAT_ID` (expert FFN) | 1,909 | 1,910 | 0 | **0%** |

Graph total 7,965 → 34,284 ms, i.e. **7.8 → 33.5 ms/token** against 36.5 predicted by the fitted prefill
curve — 8% out at 56 times the range it was fitted on.

**Prefill is compute-bound, not overhead-bound.** At 2.19 TFLOP/s against a ~5.7 peak it runs at 38%, which is
reasonable for quantised GEMMs. An earlier section here described prefill as having a "33x unexplained" cost;
that was an estimation error — only attention and indexer-scoring FLOPs had been counted, omitting the FFN,
which is 47% of a prefill ubatch. There is no large hidden inefficiency.

**The expert FFN is exactly flat** across a 56x context increase (1,909 → 1,910 ms), which confirms it as the
per-token fixed cost and excludes it from the growth entirely.

**A prediction that failed, recorded because the failure is instructive.** From the short-context point alone,
`TOP_K` looked like the growth driver: it runs per token and scales with the block count, implying 61x growth
and ~12 ms/token. Measured growth is **5.3x**, 0.85 ms/token, 3% of the total. `partial_sort` parallelises
across the ~4,096 rows a 1,024-token ubatch provides; decode supplies one row and cannot. **The op that is
crippled in decode is nearly free in prefill** — the ranking between the two phases inverts, and extrapolating
one from the other is invalid.

**What actually grows is attention and the reduces**, 76% between them. Attention is nominally capped at
`top_k · r = 8,192` attended positions, yet it grew 17x where that cap implies 4x — so the cost is not the
positions attended but the scan across `n_kv` deciding what to skip, which is O(n_kv × n_tokens) in the mask.
The reduces growing 8x with context is unexplained: 189 nodes moving activations that should be sized by
n_tokens, not by n_kv.

Neither is addressed by the pooled-key cache, the GET_ROWS threading, or the f16 keys. Prefill optimisation is
a separate problem from decode optimisation on this architecture.

## 17. Where the remaining gain is blocked, and why the blocker is a build-system fact

The pooled-key cache — caching the indexer's block summaries instead of rebuilding all of them every token —
targets 59.5 ms of the 87.2 ms of context-dependent decode cost, and would be worth roughly 3x at 256K
against the ~1.19x banked. It does not work, and the reason is worth recording precisely.

**Four implementation attempts, four different wrong assumptions:**

| attempt | failure | the assumption that was wrong |
|---|---|---|
| 1 | `GGML_ASSERT(cap >= n_blocks)` | `get_v` returns **4D** `[n_embd_head_v, n_head_kv, n_kv, n_stream]`, not 2D; reading `ne[1]` gave capacity 2 instead of 524,288 |
| 2 | same | stream stride is `nb[3]`, not `nb[2]` |
| 3 | `ggml-backend-meta.cpp:3039` | that a `ggml_cpy` into a cache view would be schedulable |
| 4 | same assert | that switching to `ggml_set_rows` — the write the KV cache itself uses — would fix it |

Attempt 4 is the informative one: it **refutes** the idea that the write operation was the problem. Reading
the splitter shows subgraphs close only at a reduce point or at the final node, while nodes that are views of
leaf tensors hit a `continue` *before* the final-node test. `ggml_set_rows` is documented as returning
`view(a)`, so both the cache write and the read-back fall into that skipped class.

**The next step is instrumentation, and instrumentation is blocked by the build system.** The splitter lives
in `libggml-base`, and production's copy cannot be reproduced:

- the production md5 matches `build-L6` and `build-L8` — *not* `build-L5`, despite living in a directory
  named `prod-L5`
- the recorded compile recipe produces a 313,704 byte object where production's is 317,592 — different flags,
  same source
- **no script in the tree compiles the production object at all.** The link script links it; nothing builds
  it. The flags are unrecorded.

So a diagnostic build would differ from production in the diagnostic *plus* an unknown amount, and no
conclusion drawn from it would be attributable. The neighbouring libraries are fine — libllama and
libggml-cpu both have reproducible, gated recipes — which is why every optimisation reported here could be
verified and this one cannot.

This was surfaced only because the lineage gate refused the build. The same gate caught two earlier
wrong-lineage builds in this project. A private-library workflow without one produces A/B results that are
silently about more than the change under test.

## 18. Unblocking the cache: a lost build recipe and a two-line scheduler bug

Section 17 reported the pooled-key cache as blocked, with instrumentation impossible because production's
`libggml-base` could not be reproduced. Both obstacles turned out to be tractable, and the second is a real
defect rather than a local quirk.

### Recovering a compile recipe that no script contained

Production's `libggml-base` (md5 `1ad65995`) comes from the `build-L6`/`build-L8` lineage, and **no script in
the tree compiles that object** — the link scripts link it, nothing builds it. Recovered by bisecting flags
against the 317,592-byte target and matching the instruction mix:

| flags | size | zmm | ymm | xmm | mask regs |
|---|---:|---:|---:|---:|---:|
| **production** | **317,592** | 318 | 147 | 1131 | **15** |
| recorded recipe | 313,704 | — | — | — | — |
| `-march=cascadelake` | 321,808 | 9 | 578 | 1536 | 7 |
| `-mavx512f` | **317,592** | 318 | 147 | 1131 | **7** |
| **`-mavx512f -mavx512bw`** | **317,592** | **318** | **147** | **1131** | **15** |

`-mavx512f` alone reproduces the production **size exactly** while differing in **72,520 bytes of `.text`**.
Size is not evidence. The mask-register count separated them, and adding `-mavx512bw` produced a
byte-identical object whose relink reproduces the production library md5.

### The scheduler bug

With a gated instrumented build possible, the failure named itself in one run:

    META SPLIT FAIL: i_start=5410 n_nodes=5418 -- 8 unconsumed node(s)
      [5414] SET_ROWS  cache_idx_k_l39 (view)         view_src_op=NONE
      [5417] VIEW      leaf_100 (view)                view_src_op=NONE   ← LAST NODE

The tensor-parallel subgraph splitter closes a subgraph at a reduce point **or at the final node**, but nodes
that are views of leaf tensors hit a `continue` *before* the final-node test. A graph whose last node is such
a view therefore never emits its trailing subgraph, and the partition assert fires. Writing a cache and then
viewing it produces exactly that shape.

The fix emits the trailing subgraph. It is reachable **only** on the path that previously aborted, so it
cannot alter any graph that already worked — the safest class of change available.

### Why this took four failed attempts first

Each earlier attempt assumed something about the system instead of measuring it: that `get_v` returned a 2D
shape (it is 4D), that the stream stride was `nb[2]` (it is `nb[3]`), that a `ggml_cpy` into a cache view was
schedulable, and that switching to `ggml_set_rows` — the write the KV cache itself uses — would therefore fix
it. The fourth was the useful failure: it **refuted** the write-operation hypothesis and forced the question
to be settled by reading the splitter rather than by guessing at it again.

## 19. An exact top-k fast path that never fires — and a test that could not detect its own hypothesis

`GGML_CPU_ARGSORT_TOP_K=1` is enabled in production and gates an exact `partial_sort` fast path in
`argsort_descending()`. Its guard is:

```cpp
std::all_of(data, data + n, [](float v){ return std::isfinite(v); })
```

A sparse-attention indexer masks with `-INFINITY`, which is not finite. **The guard therefore fails on every
real indexer row and the O(n log n) full sort runs on every token**, with the optimisation nominally switched
on. Verified directly: the env var is present in the running server's `/proc/<pid>/environ`, and the guard is
at `ops.cpp:8691`.

The repair is sound in principle — masked entries cannot enter the top-k, so ties *among them* cannot change
the selected set. Only finite values need the tie and boundary proofs. One addition is required that the
obvious fix omits: **NaN must still be rejected.** Comparisons on NaN are undefined and corrupt
`partial_sort`; that is precisely how an earlier histogram-selection attempt crashed. `-inf` is safe, NaN is
not, and the original guard conflated them. The patched guard is `none_of(isnan)` plus a check that the k-th
value is finite.

Output is byte-identical, confirming exactness.

### The measurement was underpowered, and the arithmetic was available beforehand

| ctx | n_blocks | k/n | `partial_sort` vs `sort` | share of graph |
|---|---:|---:|---:|---:|
| 28,880 | 7,220 | **28.4%** | 1.17x | **0.8%** |
| 237,500 | 59,375 | 3.4% | 1.44x | 2.7% |

`partial_sort` is O(n log k), so the saving is `log(n)/log(k)`. At 28,880 the indexer selects 2,048 of 7,220
blocks — 28% of them — which is close to the worst case for a partial sort. Expected effect **~0.8%** against
a **3.4%** noise floor measured across seven production control runs. The result (16.00 vs 16.14) is
indistinguishable from zero **in either direction**, so the fix is neither confirmed nor refuted.

Three earlier windows in this session carried pre-registered effect-size predictions and were sized
accordingly. This one did not, and it is the one that needed it: two minutes of arithmetic would have shown
the test could not resolve its own hypothesis and should run at 120K+, where `k/n` falls to 3% and the effect
roughly triples. **An A/B whose expected effect is below its noise floor is not a weak result; it is not a
result.**

## 20. Everything combined: +25.2%, byte-identical

Four changes, each individually gated and individually measured, run together against the same library set
with every flag off:

| arm @28,880 ctx | tok/s | acceptance | tokens/cycle | greedy hash |
|---|---:|---:|---:|---|
| all off | 16.20 | 76% | 4.13 | `13b7ea22` |
| **all on** | **20.29** | **82%** | **4.36** | `13b7ea22` |

**+25.2%, with byte-identical output.** The four:

| change | library | alone |
|---|---|---|
| row-parallel `GET_ROWS`/`SET_ROWS` | libggml-cpu | +12.8% |
| exact top-k un-gated from the `-inf` mask | libggml-cpu | below noise floor, unresolved |
| pooled-key cache + f16 pooled keys | libllama | +13.8% |
| trailing-subgraph fix | libggml-base | enables the cache to build at all |

All three patched libraries were confirmed *mapped* by reading `/proc/<pid>/maps`, not merely requested via
the environment — two of the four patches print no banner, so a silent fallback would have read as "these do
not stack".

### The prediction was wrong, and in the more interesting direction

Recorded before the run: **18.8–19.8**, on the argument that the cache *subsumes* the threading because it
eliminates the very gather the threading accelerates. The naive multiplicative estimate (21.21) was
explicitly dismissed as double-counting.

Measured 20.29 — **4.3% below the estimate called wrong**, and above the predicted range. The cache does not
remove all gathers: it still fetches `r × n_dirty` rows every token, the ~78 non-indexer `GET_ROWS` nodes are
untouched, and the cache's own `set_rows` write benefits from the threading. The overlap correction was real
but about five times too large.

### What it means at 256K, and what it costs

| basis | 256K tok/s |
|---|---:|
| measured | **3.40** |
| flat 1.252× from 28,880 | 4.26 |
| component-wise, scaling with context | 4.72 |
| **honest range** | **4.3 – 5.0** |

**The combined configuration is not uniformly better.** At 190 tokens it *loses* 7.7%, reproducing the
cache's short-context penalty (−10% measured independently). With 48 blocks the cache's fixed bookkeeping
exceeds what it saves. A shipping configuration must gate the cache on context length; the threading and the
top-k fix are safe at every length.

## 21. Aggregate throughput: 46.74 tok/s, and speculation should be off at concurrency

Every number above this section is single-stream. Single-stream decode leaves this machine 65% bandwidth-idle
at 34% of its byte-floor pace, dispatching 6,156 graph nodes to produce one token — a sparse MoE at batch 1
has nothing to spend 381 GB/s on. So the aggregate question matters, and it had never been measured here.

190-token prompts, 64 tokens each, `--parallel 16`, wall-clock aggregate:

| concurrency | speculation on | speculation off | winner |
|---:|---:|---:|---|
| 1 | 16.25 | 14.25 | spec on |
| 2 | 23.89 | 24.72 | tie |
| 4 | 28.28 | 25.80 | spec on |
| 8 | 34.42 | *15.92* | suspect — see below |
| **16** | 29.57 | **46.74** | **spec off, +58%** |

**Peak 46.74 tok/s aggregate at c=16 with speculation disabled — 1.92× the shipped single-stream 24.3.**

Both predictions made in advance elsewhere are confirmed: the `--spec-type none`, `--parallel 16` figure
lands at 46.74 against a predicted 45–79, and speculation stops paying at high concurrency, losing by 58% at
c=16. The mechanism is that MTP drafting and batching are substitutes — both exist to give a dispatch-bound
machine more tokens per graph pass, and paying for both wastes the draft's passes.

### Two reading errors, recorded because each nearly became a conclusion

**Comparing a prediction against the wrong arm.** The 45–79 range was first called refuted by checking it
against the *speculation-on* column (29.57). The prediction named `--spec-type none` explicitly. Misreading a
prediction is the same class of error as mismeasuring one.

**Explaining an anomaly instead of distrusting it.** The speculation-off point at c=8 reads 15.92 — below its
own c=4 (25.80) *and* its own c=16 (46.74). That is not a physical shape. It was briefly used as the basis
for a mechanism ("speculation *is* batching, so the two are complementary") that the c=16 result then
contradicted. **A datum that breaks monotonicity in both directions is a measurement fault until proven
otherwise**, and it needs re-running before the c=8 column is used for anything.

### What it changes, and what it does not

Every optimisation in this document is worth 10–25% on single-stream decode. Batched serving with speculation
off is worth **1.92×**, and it is a configuration change rather than a code change. That does not make the
single-stream work wasted — it is what a per-request latency SLA needs — but the *machine's* capability is an
aggregate number, and this is the first measurement of it.

It costs per-stream latency heavily: 20.32 → 3.76 tok/s per stream between c=1 and c=16. So this is the right
posture for several agents sharing a box and the wrong one for one agent holding a 256K window. The sweep is
also at 190-token context; at 256K each slot needs ~8.8 GB of KV, and the aggregate behaviour there is a
different, unmeasured question.

## 22. The aggregate advantage is a short-context artefact — and it inverts at length

Section 21 measured 46.74 tok/s aggregate at c=16 with speculation off, 1.92× the shipped single-stream. That
number was taken at a **190-token** context, where the context-independent part of a token is **99%** of it.
Batching amortises exactly that part. Each additional stream, meanwhile, brings its own full-context KV scan,
which amortises not at all.

So the multiplier should track the fixed share. Predicted before the run, then measured at 7,600 tokens:

| concurrency | multiplier @190 ctx | multiplier @7,600 ctx |
|---:|---:|---:|
| 1 | 1.00× | 1.00× |
| 2 | 1.73× | **1.22×** |
| 4 | 1.81× | **0.85×** |
| 8 | — | **0.57×** |
| 16 | 3.28× | not run |

| context | fixed share of a token |
|---:|---:|
| 190 | 99.1% |
| 7,600 | 72.4% |
| 28,880 | 40.9% |
| 237,500 | **7.8%** |

At 7,600 tokens aggregate peaks at **c=2** and then falls *below* single-stream: by c=8 it is 4.74 tok/s
against 8.34 at c=1, a 43% loss, with per-stream collapsed to 0.59 tok/s. **Concurrency stops being a
multiplier and becomes a tax**, because the work it adds per stream is the work it cannot share.

At 256K the fixed share is 7.8%. There is essentially nothing left for batching to amortise, and the trend
across three contexts says concurrency there would be pure loss.

### What this settles

**For one agent holding a long context, concurrency is not a lever**, and the single-stream work in this
document — the pooled-key cache, the GET_ROWS threading, the f16 pooled keys, +25.2% combined and
byte-identical — was the correct target after all.

**For several agents on short contexts, batching dominates everything else**: 46.74 tok/s aggregate, from a
configuration change rather than a code change, and with speculation *disabled* because a batch and an MTP
draft are substitutes.

These are two different machines built out of the same hardware, and the choice between them is a serving
decision rather than an optimisation one. The mistake would be to quote either number as *the* capability.

## 23. Speculation and concurrency are substitutes, and the fixed share of a token decides which one pays

Two sweeps, one at 190 tokens and one at 7,600, each with speculation on and off:

| | 190 tokens | 7,600 tokens |
|---|---|---|
| best configuration | c=16, **speculation OFF** | **c=1, speculation ON** |
| best aggregate | **46.74 tok/s** | **16.16 tok/s** |
| speculation at c=1 | 1.14× | **1.94×** |
| speculation at high concurrency | **−58%** | still positive, on a falling curve |
| concurrency, c=1 → best | **3.28×** | **0.57×** at c=8 |
| fixed share of a token | 99.1% | 72.4% |

At 7,600 tokens the full curve declines from its c=1 peak (16.16 → 10.26 → 7.78 → 6.65), while at 190 tokens
it climbs to c=16. **The optimum moves from one corner of the configuration space to the other**, decided by
a single quantity.

**The mechanism.** MTP drafting and batching do the same job: hand a dispatch-bound machine more tokens per
graph pass. This machine spends 6,156 graph nodes to produce one token and sits at ~34% of its byte-floor
pace, so tokens-per-pass is the binding resource. Short context is almost entirely fixed cost, which batching
amortises across streams — and there the draft's extra passes are pure waste. Long context is dominated by
per-stream KV scanning, which batching cannot share and can only multiply — and there speculation still
raises tokens per pass while concurrency only adds work.

Speculation's value grows with context (1.14× → 1.94× → ~2.3× at 190 / 7,600 / 28,880), tracking draft
acceptance rising 55% → 73% → 82%. Concurrency's value shrinks with context, tracking the fixed share
(99.1% → 72.4% → 40.9% → **7.8%** at 256K).

### What this means for a 256K workload

**`--parallel 1`, speculation on**, plus the four validated single-stream changes documented above. The
46.74 tok/s aggregate is real and does not apply. The existing `--parallel 1` default was chosen because two
slots cost 15–20% of single-stream decode; it turns out to be right for a second and much larger reason that
had never been measured.

The general lesson is that "the throughput of this machine" is not one number. It is a configuration choice
whose optimum inverts between short and long context, and quoting either endpoint without the context length
attached is how a benchmark misleads.
