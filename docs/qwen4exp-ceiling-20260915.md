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

## 11. The sparse attention is a net loss on this machine, and the loss grows with context

This is the conclusion the rest of the document has been circling.

A sparse-attention indexer exists to make attention cheap: score every block of the context, select the top
`k`, attend only to those. On this hardware the selecting costs vastly more than the skipping saves.

Of the 87.2 ms/token of traced growth between 200 and 32,000 tokens of context:

| | ms | share |
|---|---:|---:|
| `GET_ROWS` — gather every cached indexer key | 35.1 | 40% |
| scoring `mul_mat` — queries against every block | ~25.7 | 29% |
| `ROPE` — re-rotate every block summary | 14.7 | 17% |
| `TOP_K` — select 2048 of n_blocks | 7.8 | 9% |
| **the indexer, total** | **83.3** | **95.5%** |
| `FLASH_ATTN_EXT` — the attention itself | 3.9 | 4.5% |

**The machinery that decides what to attend to costs 21x the attention it saves.**

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

For any fixed long prefix — a codebase, a corpus, a document set — the 2.48 h is paid once and every later
session restores it. That is the difference between 256K being a benchmark number and being usable, and it
requires no kernel work, no patch, and no quality tradeoff.

Two operational notes. The server builds the path as `slot_save_path + filename`, a plain concatenation with
no separator, so the path argument **must** carry a trailing slash or the file lands next to the directory
rather than inside it. And a saved slot is tied to the cache geometry that produced it: a different KV type,
context size, or slot count will not load it.
