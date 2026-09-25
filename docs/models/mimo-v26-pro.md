# MiMo-V2.6-Pro-RL

**Status (2026-09-23):** production build `build-prod-0922j` serves the model with Xiaomi's own DFlash drafter, at a 256K-token window.

With speculation, the rate depends on how predictable the text is. The speeds below are at short context on a quiet box (build 0922c; builds 0922d–j produce bit-identical output and compute faster):

| Workload | Speed (tok/s) |
| --- | ---: |
| Verbatim repetition | 22.8 |
| Counting | 26.9 |
| Code | 16.3 |
| Recalling a memorised list | 13.2 |
| Open prose | 8.8 |

For comparison:

- Without speculation the same stack decodes at 7.94 tok/s.
- Without the tensor-parallel engine it decodes at 1.03 tok/s.

Prefill runs at 60–69 tok/s at short context. Vision, audio and tool calls are verified on every deploy. [Benchmark](../../benchmarks/mimo-v26-pro-cpu-20260923.md) · [records](../../engineering/2026-09-23/README.md)

## The model and the conversion

MiMo-V2.6-Pro-RL is Xiaomi's sparse mixture-of-experts model: 1.02 trillion parameters, 42 billion active per token. Its structure:

- 70 transformer layers: 60 use a 128-token sliding window and 10 attend to the full context.
- 384 routed experts per layer, 8 active per token.
- A 1M-token context.
- Vision and audio encoders.
- A separate 5-layer drafter.

The checkpoint was `XiaomiMiMo/MiMo-V2.6-Pro-RL` at revision `54b10491`, 534 GiB in total. Its tensors ship in three formats:

- Experts in MXFP4, quantization-aware trained.
- Attention in FP8 with 128×128 block scales.
- The output projection and embeddings in BF16.

No GGUF of this version existed, and the upstream converter would have turned the MXFP4 expert codes into float32 numbers. The [converter patch](../../engineering/2026-09-23/archive/serving/mimo-v26-pro/mimo-v26-mxfp4-converter.patch) writes the experts raw, through the existing lossless MXFP4 repack. The other tensors become Q8_0; norms and the router stay F32. The result is 518 GiB.

The nibble order was established on real weights before converting. With element 2i in the low nibble, the per-neuron magnitudes of each expert's up and down projections correlate at +0.54 to +0.95; swapped, the correlation is about 0.

The checkpoint never existed on disk in full: there was no room for it. The conversion ran in three stages:

1. **Non-expert tensors:** fetched and hash-checked (38 GiB).
2. **Skeleton:** the converter wrote all 13 output splits with the expert tensors reserved as sparse holes. This took 12 minutes and used 24 GiB of real disk.
3. **Expert fill:** a [fill job](../../engineering/2026-09-23/archive/serving/mimo-v26-pro/fill_experts.py) processed each of the 128 expert shards in turn, resumably:
   - download the shard and check its hash;
   - repack its 621 tensors and write each at its computed offset;
   - read one back to confirm the bytes;
   - delete the shard.

A final pass checks that no hole remains in any split. [Conversion notes](../../engineering/2026-09-23/archive/serving/mimo-v26-pro/README.md).

## Why speculation is required here

The model's own tensor table shows where the bytes go ([arithmetic](../../engineering/2026-09-23/archive/serving/mimo-v26-pro/BANDWIDTH-CEILING-20260921.md)). One decode step reads 35.0 GB:

| Group | Read per token | Share |
| --- | ---: | ---: |
| Attention weights (Q8_0) | 19.3 GiB | 59% |
| Routed experts (8 of 384, MXFP4) | 10.3 GiB | 31% |
| Router, output head and dense layers | 3.0 GiB | 10% |

**This is the opposite of GLM-5.3-Flash, where the experts are the wall.**

Against the machine's measured 381.6 GB/s, decode without speculation cannot exceed 10.9 tok/s. The tensor-parallel engine reached 7.94, a 73% duty cycle. With 129.5 GiB on each of the four NUMA nodes, placement was already right.

Speculation pays because attention, the router and the output head are read once per verify pass, whatever the number of rows. Only the expert bytes grow with the rows.

## The drafter

The checkpoint's three MTP blocks are a dead end. Drafting with them accepts 4–16% of proposals and makes decode 42% slower (4.58 against 7.94 tok/s).

One hypothesis was that the hidden state reached the heads before the final norm. A patch tested it: output stayed identical and acceptance did not move, so the hypothesis was [refuted](../../engineering/2026-09-23/archive/serving/mimo-v26-pro/patches/mimo2-hnextn-postnorm.patch).

The model card answers the question. The inference drafter is a separate component:

- five sliding-window layers, with a 1,024-token window;
- it drafts an 8-token block per forward pass.

The MTP blocks are training heads.

Six defects lay between that drafter and a working one. None was in the drafter's weights. The [block-drafter case study](../case-studies/block-drafter.md) covers:

- how each defect was found;
- why `p_min`, not a fixed draft length, is the right control;
- a cycle model that fits to 0.7%.

The model puts the verify cycle at 79% of the memory wall. Faster serving code therefore has little left to gain; almost all of it would have to come from accepting more tokens per verify cycle.

Production uses `--spec-draft-n-max 7 --spec-draft-p-min 0.5`. The setting was re-checked after the kernel work made the verify pass cheaper. Across five workloads, shorter drafts lost on the geometric mean:

- n = 5: −3.9%
- n = 4: −12.3%

## CPU kernels, builds 0922b–0922j

[Kernel notes](../../engineering/2026-09-23/archive/serving/mimo-v26-pro/STATE-FA-GQA-20260922.md). Every change below is gated by an environment variable and default-off in the source tree.

**Grouped-query split-KV flash attention** (`GGML_CPU_FA_GQA=2`). ggml's CPU flash attention processed query batches of 2–63 rows (every speculative verify) once per query row and Q head. It streamed the whole KV cache each time, and it summed V in FP16. The new kernel lets all the heads that share a KV head, across a group of rows, meet each K/V tile once, and accumulates in F32.

- Speed: 11× faster at 64K × 8 rows.
- Error: relative error falls from 4e-2 to 1e-6.
- Later builds ([report](../../benchmarks/cpu-flash-attn-gqa-splitkv.md)):
  - K tiles transposed in registers;
  - tile GEMMs blocked for L1;
  - a vectorised row max.

  Together these make the verify-shaped operation about 14× faster than the kernel it replaced.

**Multi-column x16 GEMM** (`GGML_CPU_X16_GEMM=1`). Prompt batches ran one GEMV per token. The new kernels:

- load each weight vector once for up to 8 activation rows;
- compute activation block sums once per matmul;
- convert MXFP4 scales with vector instructions.

Later builds:

- split calls evenly (12 columns per call spills registers for Q8_0);
- tile MoE-down in 512-row tiles;
- run 16-row groups in pairs, so that one register broadcast feeds two dot products ([why](../../benchmarks/cascade-lake-vnni-broadcast.md)).

All of it is bit-identical to the GEMV: per output element, the float operations are the same, in the same order.

**Exact MoE weighted sum** (`GGML_CPU_MOE_WEIGHTED_SUM_FUSION=2`). The existing fusion used FMA and so was not bit-identical. The exact mode rounds each product and adds the products in slot order. GCC's default `-ffp-contract=fast` still fused one multiply-add; an empty asm barrier on every product keeps them apart.

**1024-token micro-batches.** The output is identical at 512, 1024 and 2048 tokens. At 1024, each prompt token pays half as much expert-weight streaming as at 512.

The results:

- **Prefill:** 35 → 62 tok/s at 4K; 31 → 55 tok/s at 16K.
- **Decode at 64K:** 1.98 → 6.36 tok/s.
- **Output:** identical from build 0922c through 0922j on the full-model golden prompts.

It is not identical to the morning build of September 22, because the old attention kernel's FP16 error is gone. Two of three golden prompts match token for token. The third flips at a near-tie at token 11.

## Vision, audio and tool calls

**Vision** answered most images with `?` repeated. At first this was read as a size limit and blamed on the windowed-attention mask. That was wrong.

The real cause: the last ViT block's SwiGLU output reaches 1.1e5. ggml's CPU matmul converts activations to F16 for an F16 weight, so values above 65,504 overflow to infinity and spread as NaN. Small flat test images never produced the outlier and passed, which made the problem look size-dependent.

An **F32 projector** fixes it; it was no slower here. Compared tensor by tensor against Xiaomi's reference module, it reaches cosine 0.9999.

Two related changes:

- `--image-max-tokens` was ignored for this projector type. A one-line fix makes it work, and production caps images at 1,280 tokens.
- The reference applies attention sinks as a key-0 bias; the virtual-column form llama.cpp uses matches Xiaomi's serving stack.

[Vision notes](../../engineering/2026-09-23/archive/serving/mimo-v26-pro/STATE-VISION-20260922.md).

**Tool calls** failed 3 of 20 times at the recommended temperature of 1.0; two of the failures looped to `max_tokens`. The chat layer routes this template to the Qwen3-Coder XML handler. That handler's parser and lazy grammar require newlines that MiMo never emits, so compact calls did not parse and the grammar pushed the model off-format. Making each structural newline optional fixed it: 0 of 20 failures, and streamed multi-argument calls came out exact 5 of 5 times. The same strict handler is in upstream.

**Audio** transcribes a public-domain test clip exactly: 12 of 12 content words.

## Operating it

- **Memory.** The model needs 518 GiB and runs alone.
  - It cannot share the machine with GLM-5.3-Flash or GLM-5.3 Full.
  - Its weights are bound to their NUMA nodes, about 136 GiB each, so another process holding more than ~45 GiB on any one node triggers a per-node out-of-memory kill during load.

  Two scripts guard against this:

  - The [RAM guard](../../engineering/2026-09-23/archive/serving/mimo-v26-pro/wait-for-ram.sh) waits for 575 GiB host-wide and 150 GiB free or reclaimable on every node.
  - The [load watchdog](../../engineering/2026-09-23/archive/serving/mimo-v26-pro/load-watchdog.sh) kills a doomed reload within about 20 seconds instead of letting it run into an out-of-memory kill 30 minutes later.
- **Context.**
  - Only 10 layers keep full KV, so a 256K window costs about 12.5 GiB.
  - Decode stays usable to 64K: 18.5 / 8.6 / 9.3 / 6.4 tok/s at 4K / 16K / 32K / 64K.
  - Draft acceptance falls with depth, from 0.98 to 0.48 on this probe. That fall, not attention cost, now sets the rate at depth.
  - The Codex window is 64K.
- **Loading** takes about 20 minutes from a warm page cache and longer from the SATA SSD.
- **Host-RAM prompt cache.** 32 GiB. A displaced 4.8K-token session resumes in 1.9 s instead of 80 s.
- **Measuring** on a shared machine:
  - Take the minimum over many iterations.
  - Check the load first: another job on one of the four tensor-parallel nodes slows every operation by about 30%.
- **[Production launcher](../../engineering/2026-09-23/archive/serving/mimo-v26-pro/launch-mimo-production.sh).** Every setting is explained in its header.
- **Rebuilding the engine.** [Recipe](../../engineering/2026-09-23/README.md#rebuilding-the-mimo-engine).

## Open

- Quote throughput with its workload. On open prose the drafter accepts little, and nothing in the serving stack changes that.
- Q4_K attention would be faster (code 15.3, prose 10.0 tok/s), but it changes the output: one golden prompt diverges at token 11. It stays off.
- A quiet-machine re-measure of 0922j is still owed; other sessions' jobs held every NUMA node after its deploy.
- Prefill at depth (33 tok/s at 64K) remains slow for long first prompts.
