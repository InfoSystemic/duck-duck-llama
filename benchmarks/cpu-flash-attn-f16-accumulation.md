# CPU flash attention accumulates V in FP16: about 1% error in the decode path

**Status:** reproduced on unmodified upstream llama.cpp (`b23efaa2`, 2026-09-20) with a standalone program, and measured in situ on
real GLM-5.3-Flash tensors. A replacement kernel for the MQA case is in production here since 2026-09-20; a minimal upstream-style
fix exists and is not submitted.

## The defect

`ggml_compute_forward_flash_attn_ext_f16_one_chunk` keeps two accumulators. When V is F16 it uses the FP16 one:

```c
if (v->type == GGML_TYPE_F16) {
    ...
    ggml_vec_mad_f16(DV, VKQ16, (const ggml_fp16_t *) v_data, vs);   // V += v*expf(s - M), stored back to FP16 each time
```

Attention over N cells adds N terms of weight about 1/N. With N in the thousands each term is comparable to, or below, half the
rounding step of an FP16 accumulator holding values of order 0.1-1 (step 6e-5 to 5e-4), so the tail of the sum is rounded away
term by term. F16 is the default KV cache type, and this function serves every decode and short-batch call; the tiled prefill path
uses F32 accumulators and is not affected.

The one-query path is less wrong than the multi-query path: for one query upstream splits the CELLS over the workers, each worker
sums about 1/15 of them in FP16 and the partial sums are merged in F32. For two or more queries each query-head row is summed whole.
A speculative verify batch is exactly that case.

## Measurements

Relative RMS error of the op's output against a float64 reference computed from the same tensors. 512-wide MQA head, 16 heads,
15 workers on one socket, [tools/fa_mqa_check.cpp](../tools/fa_mqa_check.cpp).

| Case | Stock kernel | Cell-split F32 kernel |
| --- | ---: | ---: |
| synthetic, 2,051 of 4,096 cells visible, 3 queries | 6.4e-3 | 2.4e-7 |
| synthetic, 2,051 of 4,096 visible, 1 query | 1.6e-3 | 2.3e-7 |
| synthetic, 4,000 of 4,096 visible, 1 query | 2.0e-3 | 2.8e-7 |
| same two cases, **pure upstream build** | 6.4e-3 / 1.6e-3 | n/a |
| in situ, real model tensors, 1 query (36 calls) | 9.6e-4 mean, 1.6e-3 max | 7.3e-7 mean |
| in situ, real model tensors, **3 queries** (32 calls) | **1.26e-2 mean, 2.5e-2 max** | 6.4e-7 mean, 1.4e-6 max |
| in situ, 4 queries (8 calls) | 1.22e-2 mean | 5.5e-7 mean |

The in-situ rows come from a diagnostic build (`-DF18_FA_VERIFY`) that, inside the server, computes every attention node with the new
kernel, lets the stock kernel recompute the same node, and compares both with a float64 sum over the node's actual Q, cache and mask.
The model was an 8-layer truncation of GLM-5.3-Flash UD-Q4_K_XL with its real MTP head, 3,861-token prompt; truncation makes the text
meaningless but not the tensors the op receives.

Setting the cache type to F32 did not give a control here: this engine keeps the MLA cache in F16 regardless, so the float64 reference
is the control instead.

## What it does to a model

In the full model, switching the 11 sparse-attention layers and the MTP head from the stock kernel to the F32 kernel changes greedy
output quickly: 2 of 10 short factual prompts produced the same 24-token string, and on a 7K-token prompt the log-probability of the
chosen token moved by 0.15 on average over the three tokens before the texts diverged. Draft acceptance did not move (157 vs 158 of 196).
This is a comparison of two runs of one model, not a quality benchmark; no perplexity tool exists for this runtime, so the claim made
here is about the operator's arithmetic, which is what the float64 comparison establishes.

## A minimal fix, and what it costs

[patches/upstream-cpu-fattn-f32-accumulate.patch](../patches/upstream-cpu-fattn-f32-accumulate.patch) removes the FP16 accumulator:
each V row is converted with the SIMD `ggml_cpu_fp16_to_fp32` and accumulated in F32. Measured on the upstream build:

| Case | Stock | Patched |
| --- | --- | --- |
| 2,051 of 4,096 visible, 3 queries | 6.39e-3, 1.12 ms | 8.55e-5, 1.31 ms |
| 2,051 of 4,096 visible, 1 query | 1.56e-3, 0.32 ms | 8.44e-5, 0.38 ms |
| 2,051 of 32,768 visible, 3 queries | 6.67e-3, 1.65 ms | 8.93e-5, 1.95 ms |
| 4,000 of 4,096 visible, 1 query | 2.03e-3, 0.49 ms | 1.06e-4, 0.59 ms |

75x more accurate for +17-21% in this op on AVX-512. Two details matter: the generic `to_float` type trait is a scalar table walk and
made the op **7x slower**, so the fix must use the CPU backend's converter; and the remaining 8.5e-5 is the F16 Q.K dot product, which
the patch leaves alone. A fused "multiply-add from f16 into an f32 accumulator" would probably recover the time and is not written.
Only x86 was measured.

The 4x faster kernel in [patches/glm5next-fa-mqa-cellsplit.patch](../patches/glm5next-fa-mqa-cellsplit.patch) is a different, larger
change specific to MQA caches and short query batches; it is not the minimal fix and the two should not be presented as one.

## A second way to get attention wrong in a verify batch

The first revision of that kernel was accurate and still broke something: each worker summed a slice of the cells that *any* query
of the batch could see, so the rounding of one query depended on its batch companions. In speculative decoding the companions are
draft tokens, which vary with the drafter's state, and greedy text began to alternate between two outputs from run to run. The fix
is structural (cells dealt by index, row maximum taken before the softmax pass, weight sums in double); the test is
[tools/fa_mqa_invariance_check.cpp](../tools/fa_mqa_invariance_check.cpp): one query alone and inside 3-query batches, compared
bit for bit (INVARIANT, 8 trials).

The stock kernel is not invariant in this sense either, for a different reason: its one-query path splits the cells across the
workers and merges partial sums, its multi-query path sums each row whole, so the same query differs by 3.4e-4 (max abs, same
synthetic case) between a batch of one and a batch of three. Among batches of the same size it is stable. That is the ordinary
batch-size sensitivity of CPU kernels, but it does mean that a token verified in a speculative batch is not numerically the token
plain decoding would have computed.
