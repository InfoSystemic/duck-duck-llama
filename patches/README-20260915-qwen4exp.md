# Qwen3.8-Flash-Next decode patches, 09-15

Four changes measured on a Lenovo SR950 (4× Xeon Gold 6242, 755 GB DDR4, 381 GB/s aggregate measured),
against `llama.cpp-q4e-goal-0904`. Combined they give **+25.2%** single-stream decode at 28,880 tokens of
context with **byte-identical greedy output** (sha256 of the completion matched production on every arm).

| patch | file it touches | library | flag | gain alone |
|---|---|---|---|---|
| `qwen4exp-getrows-parallel.patch` | `ggml-cpu/ggml-cpu.c` | libggml-cpu | `GGML_CPU_PARALLEL_GET_ROWS=<min_rows>` | **+12.8%** |
| `qwen4exp-pooled-key-cache.patch` | `src/qwen4exp.cpp` | libllama | `GGML_Q4E_POOL_CACHE=1` (cache) / `=2` (f16 only) | **+13.8%** / +4.7% |
| `qwen4exp-topk-unmask-fastpath.patch` | `ggml-cpu/ops.cpp` | libggml-cpu | none (always on) | unresolved, below noise |
| `meta-backend-trailing-subgraph.patch` | `src/ggml-backend-meta.cpp` | libggml-base | none | enables the cache to build |

## What each one is

**`getrows-parallel`** — `ggml_get_n_tasks()` pins `GET_ROWS` to one thread while
`ggml_compute_forward_get_rows` already splits by rows in every type variant. On a model whose
sparse-attention indexer gathers its whole key cache per token, that gather was moving 8.2 MB per layer in
2.93 ms: **2.80 GB/s, 0.74% of the machine**, at 91 ns per row — one DRAM latency with no memory-level
parallelism. Threading it above a row threshold is bit-exact. The gain is **1.80× on the gather**, not
linear, because the access is latency-bound.

**`pooled-key-cache`** — the indexer re-derives every block summary on every token. Block *b*'s rope position
is `b*ratio`, a pure function of the block index, so the pooled+normed+**roped** key is invariant and
cacheable; and because blocks are defined over positions, the dirty set is always a trailing range derivable
from shapes. Stores them in the indexer KV cache's **V half**, which the graph allocates and never writes.
Mode 2 (f16 keys, no caching) exists as a numerics-matched control and is a free +4.7% on its own.

**`topk-unmask-fastpath`** — `GGML_CPU_ARGSORT_TOP_K=1` gates an exact `partial_sort` path behind
`all_of(isfinite)`. A sparse-attention indexer masks with `-INFINITY`, so on real scores the guard fails and
the full O(n log n) sort runs anyway. This admits `-inf` (masked entries cannot enter the top-k, so ties
among them cannot change the selected set) while still rejecting **NaN**, on which comparisons are undefined
and corrupt `partial_sort`. Exactness is preserved.

**`meta-backend-trailing-subgraph`** — the tensor-parallel subgraph splitter closes a subgraph at a reduce
point **or at the final node**, but `continue`s on views of leaf tensors *before* the final-node test. A
graph that writes a cache and then views it ends on such a node, so its trailing subgraph is never emitted
and `GGML_ASSERT(i_start == cgraph->n_nodes)` fires. Reachable only on the path that already aborted.

## Applying them

The cache **must be gated on context length**: at 190 tokens it *loses* 8–10%, because with 48 blocks its
fixed bookkeeping exceeds the saving. The threading and the top-k fix are safe at every length.

Every measurement here used a lineage gate: the unpatched object rebuilt byte-identical to production's, and
the unpatched link reproducing the production library's md5, before any patched build was trusted. That gate
caught three wrong-lineage builds during this work, including one where the recorded compile recipe produced
a different object than the shipped library.
