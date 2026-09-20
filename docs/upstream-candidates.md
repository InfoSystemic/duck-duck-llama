# What is worth taking upstream, and who has to do it

Checked against ggml-org/llama.cpp `b23efaa2ef147f547ee75cbf0c621d61904de80e` (2026-09-20).

**Nothing here has been submitted, and none of it may be submitted by an agent.** llama.cpp's `AGENTS.md` and `CONTRIBUTING.md` require
that a human contributor understands every line, writes the description and the commit messages, and answers review personally;
agent-opened pull requests and AI-written descriptions are closed and can lead to a ban. Private forks are exempt, which is why the
work lives here. This page is preparation: what exists, what the evidence is, and what a submitter must be able to defend.
Large features should start as an upstream discussion, not as a patch.

## 1. Per-socket CPU devices and tensor parallelism across them

The distinct contribution of this repository. Each NUMA node becomes a ggml device (`CPU-NUMA0..N`) with node-bound buffers and a pinned
worker pool, the Meta backend splits tensors across them, and a direct host-memory all-reduce joins the results. Measured 4.07x from
one socket to four on Qwen3.8-Flash-Next with speculation (raw decode gains far less), and the basis of every result here.
Source: [llama.cpp-b249-cpu-numa.patch](../patches/llama.cpp-b249-cpu-numa.patch), [porting notes](../patches/PORTING.md),
[the case for CPU + multi-channel memory](case-for-cpu-multichannel.md), [NUMA case study](case-studies/numa-and-bandwidth.md).

Related work to cite honestly: KTransformers' kt-kernel already splits routed-expert weights across NUMA thread pools
(`--kt-threadpool-count` = number of NUMA nodes). It is an expert backend for GPU-hybrid serving: attention, dense and shared layers run
on the GPU, and its GLM-5.3-Flash recipe lists SM89/SM120 GPUs and has no CPU-only mode
([tutorial](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/GLM-5.3-Flash-Tutorial.md),
[kt-kernel](https://github.com/kvcache-ai/ktransformers/tree/main/kt-kernel)). The difference to state is scope, not priority: this
backend parallelises the WHOLE graph of any GGUF model on a machine with no GPU. On this SR950 the expert phase already runs at the
measured DRAM ceiling, so the remaining work is exactly the part a hybrid engine gives to a GPU.

A submitter needs: the device/threading model, why strict `mbind` and not `numactl --interleave`, the failure modes of the generic Meta
collective that the direct all-reduce avoids, and a plan for platforms without libnuma. This is a large change; ask first.

One design point changed on 09-20 and belongs in any proposal: a backend instance must NOT own its worker team. Each context creates
its own backends, so a server with a draft model ran two OpenMP teams pinned to the same cores, and libgomp's default idle spin
(300,000 iterations, ~15 ms with Skylake's 140-cycle `pause`, not the documented 3 ms) made every hand-off between trunk and draft a
collision. One dispatcher per DEVICE, shared by all backends, removes it ([patch](../patches/cpu-numa-shared-team.patch)). Stock
upstream does not have this problem with OpenMP: both contexts enter their parallel regions from the same server thread and so share
one team.

A second one from the same day: `set_tensor` on a group of NUMA buffers must not start a thread per device for small uploads. It did,
for every graph input, ~93 thread creations per decode cycle, and cost 4.6% of decode
([patch](../patches/meta-backend-small-uploads-blocking-dispatch.patch)). Parallel upload is for weights.

## 2. CPU flash attention sums V in FP16

A correctness defect in the default configuration, reproducible in isolation on stock upstream: 6.4e-3 relative RMS error for a
three-query batch over ~2,000 cells, 1.3e-2 on real model tensors. Small, self-contained, and independent of everything else here.
Evidence and a minimal fix with its measured cost (+17-21% in the op): [report](../benchmarks/cpu-flash-attn-f16-accumulation.md),
[reproducer](../tools/fa_mqa_check.cpp), [candidate patch](../patches/upstream-cpu-fattn-f32-accumulate.patch).
It is a precision fix that costs time, not a speedup; the faster MQA kernel is a separate and much larger change.
A submitter needs: why the one-query path is less affected, why the generic `to_float` trait must not be used (7x slower),
and numbers from at least one non-x86 platform, which do not exist yet. Worth stating in the same breath: the one-query and the
multi-query path sum in different orders, so one query differs by 3.4e-4 between a batch of one and a batch of three
([check](../tools/fa_mqa_invariance_check.cpp)); an F32 accumulator shrinks that gap as well.

## 3. `llama_kv_cache::seq_rm` scans every allocated cell

Upstream still loops `for (i = 0; i < cells.size(); ++i)`. With a 1M-cell cache and a speculative rollback every cycle this was
~0.85 ms per call; stopping at `cells.used_max_p1()` (which upstream already has) removes it with no behaviour change.
+2.4% decode on GLM-5.3-Flash in a same-process A/B, and the scan was found independently at 256K on Qwen.
[q4e-line patch](../patches/kv-seq-rm-bounded-scan.patch), [GLM-line patch](../patches/glm-kv-seq-rm-used-prefix.patch).
A submitter needs: the argument that cells past the used bound cannot match a non-negative range, including after `seq_cp` and shifts.

## 4. Elementwise UNARY ops and SCALE are pinned to one thread

`ggml_get_n_tasks` sets `n_tasks = 1` for SIGMOID/EXP/SOFTPLUS/... and for SCALE although both kernels honour `ith/nth`.
On models with a linear-attention or SSM path this serialises real work: +1.7% (GLM-5.3-Flash) and +3.4% (Qwen3.8-Flash-Next), bit-exact,
behind a size threshold. Single-row tensors need a column split for SCALE to benefit.
[Source, parent and build record](../engineering/2026-09-12/archive/serving/fleet-0911/parallel-unary-0911/ggml-cpu.patch).
A submitter needs: a threshold justified on more than one machine, since small tensors lose to the dispatch cost.

## 5. Top-k by selection, with a fallback when the set is not unique

`ggml_compute_forward_top_k_f32` uses `std::partial_sort` with an indirect comparator. `std::nth_element` on a float copy is 3-4x faster
for rows up to a few thousand entries and returns the same SET whenever the k-th value is not tied across the cut; on a tie, fall back.
Worth less than it looks: above ~25,000 entries `partial_sort` wins again, and an earlier variant without the fallback
[changed model behaviour](../patches/topk-linear-selection.patch) because masked scores tie at `-inf`.
[Patch](../patches/topk-select-tie-fallback.patch), [check](../tools/topk_select_check.cpp).
Upstream's own test accepts any index among ties, so conformance is not the bar; unchanged model behaviour is.

## 6. The sampler rebuilds and sorts the whole vocabulary for every token

`common_sampler::set_logits` writes one 12-byte candidate per vocabulary entry and the top-k sampler then partial-sorts that array:
0.45 + 0.19 ms per sampled token at 154,880 entries, on the server's main thread, between graphs. The speculative path also clones
the sampler before every verify step and the clone copies the candidate array (1.86 MB, 0.58 ms), although nothing reads it.
When the first active sampler of the chain is top-k (the default chain), the k best logits can be selected straight from the logits
with a threshold scan (~50 us) and handed to the chain already sorted: same candidates, same order, same token.
+1.6% decode here, more on faster machines where a graph is shorter. [Patch, part B](../patches/coupled-sampling-fast-sampler.patch),
[check](../tools/topk_scan_check.cpp). Self-contained in `common/sampling.cpp`.
A submitter needs: the conditions under which the shortcut is invalid (logit bias, penalties, DRY or top-n-sigma ahead of top-k,
mirostat, `n_probs`, grammar-first, a forcing reasoning budget) and that every one of them falls back; tie order at the k-th value
can differ from `std::partial_sort`, which only matters for identical logits.

## 7. Speculative decoding with sampled requests: share the verifier's randomness

Upstream verifies a draft by sampling the target and comparing tokens. With temperature > 0 a greedy draft is then accepted with
probability p(argmax q), which is poor exactly where the text is uncertain. If the final `dist` step picks
`argmax(logit + G(salt, position, token))`, with G a counter-based Gumbel variate, the sample is exact (Gumbel-max), it no longer
depends on how many RNG draws speculation consumed, and a drafter that knows (salt, position) can add the same noise to its own
logits. Acceptance on real agent traffic rose from 72.4% to 75.0% here with an MTP head; a better-calibrated drafter gains much more
(synthetic: 0.39 -> 0.88). Same seed gives the same text with and without speculation, which is not true of rejection-sampling
schemes. [Patch, part A](../patches/coupled-sampling-fast-sampler.patch), [exactness test](../tools/coupled_sampling_check.cpp).
This is a design proposal, not a fix: it changes which random stream a seed produces, so it should start as a discussion.
A submitter needs: the argument for exactness including the grammar resample path (it keeps an independent stream, reproducing the
current rejection-resample law), and a cleaner channel between drafter and sampler than the token-tail registry used here.

## Already fixed upstream

The scheduler cut a new split when a split reached `GGML_SCHED_MAX_SPLIT_INPUTS` (30) inputs, which cost 11 ms per token on Qwen and was
worked around here with `-DGGML_SCHED_MAX_SPLIT_INPUTS=64`. At `b23efaa2` the input arrays grow on demand and that cut is gone.
Nothing to submit; the workaround only matters for the older source lines in this repository.

## Not candidates

- `GET_ROWS` on all workers: upstream keeps it single-threaded on purpose (a FIXME cites the cost with GPU offloading). A proposal has to
  answer that, for example by threading only above a row count on CPU-only graphs.
- The glm5next pool fusion, pooled-result cache, MTP catch-up, MTP query rows, the row-split gated delta net and the cell-split MQA
  attention kernel depend on this fork's architecture code and Meta split rules. They belong with the architecture if and when it
  lands upstream. The gated-delta-net idea itself (split by state row when heads do not divide by workers, one fused pass per row)
  applies to upstream's `ggml_compute_forward_gated_delta_net` as it stands and is bit-exact; it is small (under 1% here).
- Constant-shape draft batches are a workaround for single-slot graph reuse. The upstream answer is a small graph cache per context,
  which this fork's Meta backend cannot support yet.
