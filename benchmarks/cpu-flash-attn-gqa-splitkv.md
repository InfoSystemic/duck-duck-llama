# CPU flash attention streams the KV cache once per query row and head

**Status:** found 2026-09-22 on MiMo-V2.6-Pro. A replacement kernel has been in production here since that evening. Upstream llama.cpp (`b23efaa2`) has the same dispatch.

**What this means:** every CPU user of speculative decoding on a grouped-query-attention model pays this cost, and it grows with context.

## The defect

`ggml_compute_forward_flash_attn_ext_f16` picks a path by the number of query rows in the batch:

| Query rows | Path |
| --- | --- |
| 1 | Split-KV |
| 64 or more | Tiled |
| 2–63 | `one_chunk`, once per (query row, Q head), over the entire KV range |

Every speculative verify batch falls in the 2–63 range.

With grouped-query attention, several Q heads share each KV head. The third path therefore streams every K/V row once per Q head sharing it, times the number of query rows. Under 4-way tensor parallelism, MiMo has 16 Q heads per KV head, and DFlash verifies 8 rows. So each full-attention layer re-read its cache **128 times per call**.

The same path also accumulates V in FP16, which is the [defect already reported](cpu-flash-attn-f16-accumulation.md) for the MQA case.

The symptom looked like a model property. Decode fell 3.9× between 4K and 64K of context, even though only 10 of MiMo's 73 layers keep full KV. Those layers add only about 1.3× to the bytes read per token at 64K.

## The kernel

The work is organised around each KV head:

1. **Grouping.** All the Q heads that share a KV head, times a group of up to 128 query rows, form one block.
2. **One pass per tile.** The block meets each 64-cell K/V tile once. Scores and the P·V product are two small GEMMs.
3. **Softmax.** Online softmax and accumulation run in F32.
4. **Splitting.** The visible KV span is split into chunks, so that 15 threads have work even with only two KV heads per NUMA node. The partial (max, sum, output) of each chunk is merged at the end, and attention sinks are applied last.

The plan is made at run time from the mask. As a result, a sliding-window layer touches only its window.

The first version planned from the KV length alone and was 2.4× *slower* on 128-token-window layers. Always benchmark the windowed shape as well.

Switches:

- `GGML_CPU_FA_GQA=1` covers decode and verify batches.
- `=2` also replaces the tiled prompt path.

The patch is [cpu-fa-gqa-grouped-splitkv.patch](../engineering/2026-09-23/archive/serving/mimo-v26-pro/patches/cpu-fa-gqa-grouped-splitkv.patch).

## Measurements

The benchmark is [fa-bench.cpp](../engineering/2026-09-23/archive/serving/mimo-v26-pro/tools/fa-bench.cpp): the operator at MiMo's per-node production shapes, against a float64 reference. Each shape has 32 Q heads over 2 KV heads, K dimension 192 and V dimension 128, on one node's cores.

| Shape, per layer per node | Old | New | Old relative error | New relative error |
| --- | ---: | ---: | ---: | ---: |
| Full attention, 65K cells × 8 rows | 160.5 ms | 14.3 ms (11.2×) | 3.8e-2 | 8.6e-7 |
| Full attention, 16K × 8 | 30.7 | 4.2 (7.3×) | 1.9e-2 | 4.7e-7 |
| Full attention, 4K × 8 | 5.1 | 1.4 (3.6×) | 9.0e-3 | 2.9e-7 |
| Full attention, 65K × 1 | 23.1 | 6.0 (3.8×) | 9.6e-3 | 8.5e-7 |
| 128-token window, 768 cells × 8 | 0.274 | 0.197 (1.4×) | 1.6e-3 | 1.7e-7 |
| Drafter, 1,024-token window (K dimension 128) × 8 | 1.194 | 0.323 (3.7×) | 4.8e-3 | 2.0e-7 |
| Prompt, 512 rows at 66K / 16.9K / 4.6K | 2,068 / 436 / 115 | 1,043 / 261 / 70 | 5e-6 | 5e-6 |

Three later changes were each bit-identical to the version before (same output hash):

- **K tiles transposed 16×16 in registers.** They had been converted and transposed one scalar at a time.
- **Tile GEMMs blocked by 32-column panels.** This keeps the 24 KB panel resident in L1.
- **A vectorised row max** ([patch](../engineering/2026-09-23/archive/serving/mimo-v26-pro/patches/cpu-fa-gqa-vector-max.patch)). `ggml_vec_max_f32` is a scalar loop, because GCC does not vectorise a float max reduction without fast-math. It had grown to 20% of the kernel.

Measured as the minimum over many iterations:

- **Transpose and blocking:** the 64K × 8 verify went from 16.1 to 10.6–11.9 ms on one node.
- **Vector max, single core, 16K × 8:** 1.4× (46.9–53.4 → 34.0–37.3 ms).
- **Vector max, 15 threads, 64K × 8:** 1.25× (14.6 → 11.7 ms, best case under contention).

Against the original kernel's 160.5 ms, the verify-shaped operation is now about 14× faster.

## What it did to the model

In one append-only session:

| Context | Decode before | Decode after |
| ---: | ---: | ---: |
| 4K | 7.68 tok/s | 18.45 tok/s |
| 16K | 5.29 tok/s | 8.56 tok/s |
| 64K | 1.98 tok/s | 6.36 tok/s |

Prefill at 4K went from 35.5 to 62.2 tok/s. The verify cycle is now nearly flat, at 288–364 ms from 4K to 64K; it was about 1.06 s at 64K. The 4K decode figure is flattered by a very predictable continuation (draft acceptance 0.98), so the cycle times are the fair comparison.

Output is not bit-identical to the old kernel, and it should not be: the FP16 rounding is gone. On the full model, two of three golden prompts are token-identical, and the third flips at a known near-tie. [Benchmark](mimo-v26-pro-cpu-20260923.md).

## Pitfalls met on the way

- **Work buffers.** `ggml_graph_compute_with_ctx` allocates a new work buffer per call. A timing loop exhausts the pool, so plan once and reuse the plan.
- **Scheduler stalls.** On this machine, sub-millisecond timings stall for 10–18 ms unless `OMP_WAIT_POLICY=active` is set. Use the minimum over many iterations; the median goes bimodal when other jobs preempt the threads.
- **Reference cost.** The float64 reference at 512 rows × 66K cells takes minutes. Time the prompt shapes separately.

## Upstream

This is one of the strongest [upstream candidates](../docs/upstream-candidates.md). It fixes the FP16 accumulation and the redundant streaming at once, for every GQA model that uses speculative decoding on a CPU.

The vectorised row max stands alone and is smaller. Upstream calls the same scalar `ggml_vec_max_f32` per row in SOFT_MAX, in its tiled flash attention and in cross-entropy. The pull request should state the NaN and signed-zero semantics: `vmaxps` returns its second operand.

Submission is for a human contributor (see that page).
