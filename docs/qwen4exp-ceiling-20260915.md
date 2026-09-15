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
