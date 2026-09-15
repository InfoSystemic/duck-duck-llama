# The case for CPU + multi-channel memory

*Evidence from a 4-socket Xeon Gold 6242 (SR950), 755 GB, 381 GB/s measured aggregate. Every figure below is marked
**measured** or **modelled**. Modelled figures are stated as models, not results.*

## 1. The capacity wall comes first

A consumer flagship cannot run these models at all. Not slowly — at all.

| model | file size | active GB/token (**measured**, from the tensor table) | fits on a 32 GB card? |
|---|---:|---:|---|
| Qwen3.8-Flash-Next UD-Q6_K_XL | 169.2 GB | 6.96 | no |
| Qwen3.8-Flash-Next Q4_K_M | 119.1 GB | 4.62 | no |
| DeepSeek-V4.1-Flash MXFP4 | 501.8 GB | 13.46 | no |
| GLM-5.3-Flash Q4 | ~120 GB | ~5 | no |

Holding Qwen at Q6 takes six 32 GB cards; DeepSeek takes sixteen. At late-2026 street prices that is $42k-$112k of
silicon to hold what one used 4-socket server holds for roughly $5k.

| | $ | GB | GB/s | $ per GB of capacity |
|---|---:|---:|---:|---:|
| RTX 5090 32GB | ~7,000 | 32 | 1800 | 219 |
| 4-socket SR950, 755 GB | ~5,000 | 755 | 381 | **7** |

**The asymmetry that matters:** a sparse MoE must be *held* in full but only *streamed* a few GB per token. Qwen holds
169 GB and streams 6.96. Capacity is the binding constraint, bandwidth is the rate constraint, and CPU + multi-channel
RAM wins the binding one by ~30x.

## 2. Count active bytes, not parameters

Two traps make published "active parameter" figures useless for sizing hardware, both found here by reconciling a
model's reconstructed size against its on-disk size (`tools/active_bytes.py` does this automatically and refuses to
report numbers when the two disagree):

- **Gathered tables are not streamed.** Qwen carries a 55 GB `per_layer_token_embd` and DeepSeek a 203 GB
  `engram_embd`. Both are indexed per token. Counting them as weights gives Qwen a byte floor of 7 tok/s for a model
  that measurably runs at 26.
- **A speculative verify reads the shared weights once**, and the experts for the *union* of what the drafted tokens
  route to. Bytes per accepted token therefore sit well below bytes per pass.

## 3. NUMA tensor parallelism is worth 4x, and mainline has no mechanism for it

Same engine, same model, varying only the number of NUMA nodes the model is split across (**measured**):

| sockets | speculative decode | raw decode |
|---:|---:|---:|
| 1 | 6.37 tok/s | — |
| 2 | 16.63 | 13.25 |
| 4 | **25.94** | 14.51 |

**4.07x from one socket to four.** Note the second column: raw decode gains only 1.10x going from 2 sockets to 4,
because a batch-1 graph has too little work per barrier to pay for the synchronisation. Speculation and NUMA tensor
parallelism are complementary — speculation is what gives the extra sockets something to do.

Upstream llama.cpp has no multi-device CPU tensor split, so this scaling is simply unavailable there. Used
multi-socket servers are the cheapest bulk memory bandwidth available anywhere, and the software cannot currently use
them. **This is the single largest gap between what the hardware offers and what the software extracts.**

## 4. The remaining gap is software, not silicon

On the same box, against a byte floor of 54.7 tok/s raw (**measured** bytes over **measured** bandwidth), we achieve
14.5 raw and 25.9 speculative at short context. Extraction is 26-48%. From a per-node trace of a 6,156-node decode
graph (**measured**): 62% of graph time is in matrix multiplies, 16% in cross-socket reduces, 11% in elementwise glue.
Many small GEMVs run at 12-35 GB/s against a 95 GB/s socket. There is a 2-4x in code before any hardware changes.

## 5. Benchmarks measure the wrong operating point

`llama-bench` defaults to short prompts, and essentially every published CPU number is taken at ~200 tokens of
context. Agents run at 10K-100K. On this box decode **halves** between 234 and 38,056 tokens of context: 29.22 ->
14.21 tok/s (**measured**).

That gap is how a genuine defect survived: the sparse-attention indexer used by Qwen (top_k 2048), DeepSeek (512) and
GLM (2048) selects its keys with `std::partial_sort` over the whole context, on a tensor whose `ggml_nrows()` is 1 —
so it runs on one thread, scaling with context length. At 200 tokens it costs 17 us and is invisible.

**Measured** (best of 20 reps, one row, top_k 2048), and note this corrects an earlier estimate of ours that was 3-7x
too pessimistic:

| n_ctx | partial_sort | nth_element | threshold |
|---:|---:|---:|---:|
| 38,056 | 0.624 ms | 0.134 | **0.170** |
| 100,000 | 0.874 | 0.564 | **0.282** |
| 262,144 | 1.296 | **2.157** | **0.727** |

Across 12 layers and 1.37 passes per token that is ~10 ms/token at 38K context and ~21 at 262,144 — real, worth
removing, but roughly 28% of the long-context penalty rather than most of it. Two things are worth taking from the
table: `std::nth_element` is the intuitive fix and is 1.66x *slower* than what it replaces at 262,144, and a
histogram-threshold selection wins throughout. See `patches/topk-linear-selection.patch`.

## 6. Where CPU honestly loses

**Prefill.** We measure 63-107 tok/s; a modern GPU does thousands. Prompt caching hides this for conversational and
agent use, since a session pays prefill once, but a cold 100K-token context is tens of minutes here and seconds there.
Anyone advocating CPU serving without saying this plainly is selling something.

## 7. The honest conclusion: hybrid, not purity

The right machine is not all-CPU. It is a large multi-channel memory system holding the experts, plus a modest GPU
holding the dense path and doing prefill. On Qwen the dense weights — the shared tensors plus the output head — are
**4.83 GB**, which is 69% of the bytes a token streams and the part that runs worst on CPU. That fits on a card that
costs a small fraction of a flagship.

The thing to stop doing is sizing inference hardware by parameter count. Size it by active bytes per token for rate,
and by total weights for feasibility. On that basis the economics of sparse MoE point away from consumer flagships and
toward exactly the hardware that is currently cheapest per gigabyte.
