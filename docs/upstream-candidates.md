# What is worth taking upstream, and who has to do it

Checked against ggml-org/llama.cpp `b23efaa2ef147f547ee75cbf0c621d61904de80e` (2026-09-20). Every "still open upstream" claim below
was re-verified in that tree on 2026-09-20 at 17:00, and items 3a, 3b and the grouped-attention note on 2026-09-22/23; the line
numbers are from it. Upstream moves daily: re-check each item before preparing it.

**Nothing here has been submitted, and none of it may be submitted by an agent.** llama.cpp's `AGENTS.md` and `CONTRIBUTING.md` require
that a human contributor understands every line, writes the description and the commit messages, and answers review personally;
agent-opened pull requests and AI-written descriptions are closed and can lead to a ban. Private forks are exempt, which is why the
work lives here. This page is preparation: what exists, what the evidence is, and what a submitter must be able to defend.
Large features should start as an upstream discussion, not as a patch.

## Where upstream stands on the things this repository touches

| upstream at `b23efaa2` | state | consequence for this list |
|---|---|---|
| `qwen4exp` (Qwen3.8-Flash-Next) | **merged**: `src/models/qwen4exp.cpp`, 1,297 lines; already excluded from `--split-mode tensor` in `llm_arch_supports_sm_tensor` | the Qwen indexer cache becomes a real candidate (item 9); the tensor-split safety patch is moot |
| `deepseek4` (DeepSeek-V4-Flash) | merged | — |
| `glm5next` (GLM-5.3-Flash) | **absent** | every glm5next patch here waits for the architecture (item 12) |
| DeepSeek-V4.1 (`deepseek41`) | absent | the port here is built on JigSaw's third-party diff (item 13) |
| MTP drafting (`draft-mtp`, downloadable MTP sidecar) | present (`COMMON_SPECULATIVE_TYPE_DRAFT_MTP` in `common/arg.cpp`) | the sidecar idea is not novel; only the two MTP graph savings transfer as ideas (item 12) |
| x86 repack kernels | `q4_K`/`q5_K`/`q6_K`/`q2_K` 8x8 and `q4_0`/`q4_K`/`q8_0`/`q2_K` 16x1 AVX-512, `mxfp4` 8x8 | the x16 family here must be benchmarked against those before any claim (item 11) |
| `ggml_get_n_tasks`: 17 unary ops and SCALE forced to one thread | still there (`ggml-cpu.c:2322-2325`, SCALE at 2378) | item 1 open |
| `llama_kv_cache::seq_rm` full-cell scans | still there (`llama-kv-cache.cpp:405, 429, 481`) | item 2 open |
| CPU flash attention sums V in FP16 (`VKQ16`) | still there | item 3 open |
| `GET_ROWS`/`SET_ROWS` on one thread with a FIXME | still there | item 4 open |
| `top_k` by `std::partial_sort` | still there (`ops.cpp:8583`) | item 5 open |
| scheduler split cut at 30 inputs | **fixed**: input arrays grow on demand | nothing to submit |

## Tier 1: small, generic, evidence complete; submit as they are

### 1. Elementwise UNARY ops (and SCALE) are pinned to one thread

`ggml_get_n_tasks` sets `n_tasks = 1` for ABS, SGN, NEG, STEP, TANH, ELU, RELU, SIGMOID, HARDSWISH, HARDSIGMOID, EXP, SOFTPLUS,
EXPM1, FLOOR, CEIL, ROUND, TRUNC although `unary-ops.cpp` already splits rows by `ith/nth` on exactly the path XIELU uses with
`n_threads`. Deleting the four lines is the whole change. Bit-exact (elementwise, no reduction). +2.1% GLM-5.3-Flash, +3.4%
Qwen3.8-Flash-Next, 0.0% DeepSeek-V4-Flash (no SSM/linear-attention path: report that zero, it is the honest shape).
SCALE is the same defect but its hot instance is one row of 262,144 elements, so it needs a column split in the kernel: leave it out
and be ready to say why.
Handoff with every argument and its evidence: [upstream-handoff-unary-threading.md](upstream-handoff-unary-threading.md);
source and build record: [parallel-unary-0911](../engineering/2026-09-12/archive/serving/fleet-0911/parallel-unary-0911/).
Branch `unary-ops-parallel` on `InfoSystemic/llama.cpp` was rebased on master on 09-12; rebase again and rewrite its commit message.
**The most ready item on this page.**

### 2. `llama_kv_cache::seq_rm` scans every allocated cell

Three loops `for (i = 0; i < cells.size(); ++i)` (lines 405, 429, 481). With a 1M-cell cache and a speculative rollback every cycle
that is ~0.85 ms per call; bounding the scan by `cells.used_max_p1()` (which upstream already maintains) removes it with no behaviour
change: +2.4% decode on GLM-5.3-Flash in a same-process A/B, and the same scan was found independently at 256K on Qwen.
[q4e-line patch](../patches/kv-seq-rm-bounded-scan.patch), [GLM-line patch](../patches/glm-kv-seq-rm-used-prefix.patch).
A submitter needs: the argument that cells past the used bound cannot match a non-negative range, including after `seq_cp` and shifts.

### 3. CPU flash attention sums V in FP16

A correctness defect in the default configuration, reproducible in isolation on stock upstream: 6.4e-3 relative RMS error for a
three-query batch over ~2,000 cells, 1.3e-2 on real model tensors. Small, self-contained, and independent of everything else here.
Evidence and a minimal fix with its measured cost (+17-21% in the op): [report](../benchmarks/cpu-flash-attn-f16-accumulation.md),
[reproducer](../tools/fa_mqa_check.cpp), [candidate patch](../patches/upstream-cpu-fattn-f32-accumulate.patch).
Open it as an ISSUE with the reproducer first; the maintainers may prefer a different shape of fix (F32 accumulation only when the
batch has more than one query, or an F32 `V` path selected by type). A submitter needs: why the one-query path is less affected, why
the generic `to_float` trait must not be used (7x slower), and numbers from at least one non-x86 platform, which do not exist yet.
Worth stating in the same breath: the one-query and the multi-query path sum in different orders, so one query differs by 3.4e-4
between a batch of one and a batch of three ([check](../tools/fa_mqa_invariance_check.cpp)); an F32 accumulator shrinks that gap too.

**Stronger since 2026-09-22: the same decode path also re-streams the KV cache once per query row and Q head.** For batches of
2-63 query rows, which is every speculative verify, `ggml_compute_forward_flash_attn_ext_f16` runs `one_chunk` per (row, head) over
the whole KV range. Split-KV applies only to one row and the tiled path only from 64. Under grouped-query attention each K/V row is
therefore read once per Q head that shares it, times the rows: 128 times per layer for MiMo-V2.6-Pro under 4-way tensor parallelism.
Upstream `b23efaa2` has the identical dispatch. A grouped split-KV kernel fixes both defects at once: 11x at 64K cells x 8 rows,
relative error 4e-2 to 1e-6. On the model, decode at 64K went from 1.98 to 6.36 tok/s
([report](../benchmarks/cpu-flash-attn-gqa-splitkv.md),
[patch](../engineering/2026-09-23/archive/serving/mimo-v26-pro/patches/cpu-fa-gqa-grouped-splitkv.patch),
[benchmark tool](../engineering/2026-09-23/archive/serving/mimo-v26-pro/tools/fa-bench.cpp)). File the FP16 issue first, then
offer the kernel as the larger fix; a submitter must be able to explain the run-time plan from the mask, because the first version
planned from the KV length and was 2.4x slower on sliding-window layers.

### 3a. `ggml_vec_max_f32` is a scalar loop

GCC does not vectorise a float max reduction without fast-math, so every softmax row maximum is a dependent `vmaxss` chain. Upstream
`b23efaa2` calls it per row in SOFT_MAX (`ops.cpp:5668`), per KV tile in its tiled flash attention (`ops.cpp:9040`) and in
cross-entropy (`11720`, `11813`). In the grouped attention kernel above it had grown to 20% of the time. An AVX-512 max made the
kernel 1.25-1.4x faster with an unchanged output hash
([patch](../engineering/2026-09-23/archive/serving/mimo-v26-pro/patches/cpu-fa-gqa-vector-max.patch)). Small and generic. The PR
must state the NaN and signed-zero semantics, because `vmaxps` returns its second operand when either input is NaN, and when both
are zeros of opposite sign.

### 3b. The Qwen3-Coder XML tool-call handler requires newlines some models never emit

Templates containing `<function=` and `<parameter=` are routed to `common/parsers/qwen3-coder.cpp`. Its parser and lazy grammar
require a newline after every tag and `\n</parameter>\n` to close a value. MiMo-V2.5/2.6 render, and the model emits, the compact
form `<tool_call><function=f><parameter=a>value</parameter></function></tool_call>`. The consequences:

- Compact calls parse to no tool calls at all.
- With `tool_choice=auto`, the grammar pushes the model off-format.
- Measured at the recommended temperature 1.0: 3 of 20 calls were malformed, two of them looping to `max_tokens`.

Fix: make every structural newline optional, and end a value at either `\n</parameter>` or `</parameter>`, stripping one layout
newline on each side as the reference parsers do. After it, 0 of 20 were malformed and streamed calls were exact
([patch](../engineering/2026-09-23/archive/serving/mimo-v26-pro/patches/chat-xml-toolcall-optional-newlines.patch),
[offline harness](../engineering/2026-09-23/archive/serving/mimo-v26-pro/tools/chat-toolcall-test.cpp)).

### 4. `GET_ROWS` / `SET_ROWS` on all workers above a row threshold

`ggml-cpu.c` keeps `n_tasks = 1` with the comment "the cost of launching additional threads decreases performance with GPU
offloading". On a CPU-only graph with a sparse-attention indexer the gather is the largest single growth term with context:
+12.8% alone on Qwen3.8-Flash-Next at 30K and the largest share of the 1.84x at 237,500 tokens, byte-identical.
[Patch](../patches/qwen4exp-getrows-parallel.patch) (env-gated `GGML_CPU_PARALLEL_GET_ROWS=<min_rows>`; upstream would want a
constant threshold, not an env). A submitter needs: an answer to the FIXME — thread only when the row count exceeds a threshold
(256 here) so small gathers on offloaded graphs keep the single-thread path — and a measurement on a GPU-offload configuration
showing no regression, which does not exist yet.

### 5. Top-k by selection, with a fallback when the set is not unique

`ggml_compute_forward_top_k_f32` uses `std::partial_sort` with an indirect comparator. `std::nth_element` on a float copy is 3-4x faster
for rows up to a few thousand entries and returns the same SET whenever the k-th value is not tied across the cut; on a tie, fall back.
Worth less than it looks: above ~25,000 entries `partial_sort` wins again, and an earlier variant without the fallback
[changed model behaviour](../patches/topk-linear-selection.patch) because masked scores tie at `-inf`.
[Patch](../patches/topk-select-tie-fallback.patch), [check](../tools/topk_select_check.cpp).
Upstream's own test accepts any index among ties, so conformance is not the bar; unchanged model behaviour is.

### 6. The sampler rebuilds and sorts the whole vocabulary for every token

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

### 7. Gated delta net: split by state row when heads do not divide by workers

Upstream's `ggml_compute_forward_gated_delta_net` splits by head; with 16 heads over 15 workers one worker does two in every layer.
Splitting the (seq, head, row) space evenly with one fused pass per row is bit-identical and applies to the upstream op as it stands.
Under 1% here (the op is dominated by state copies, not arithmetic): [patch](../patches/glm5next-gdn-row-split.patch). Lowest priority
in this tier; submit only bundled with a measurement on a model where the op matters more.

## Tier 2: design proposals; start as a discussion

### 8. Speculative decoding with sampled requests: share the verifier's randomness

Upstream verifies a draft by sampling the target and comparing tokens. With temperature > 0 a greedy draft is then accepted with
probability p(argmax q), which is poor exactly where the text is uncertain. If the final `dist` step picks
`argmax(logit + G(salt, position, token))`, with G a counter-based Gumbel variate, the sample is exact (Gumbel-max), it no longer
depends on how many RNG draws speculation consumed, and a drafter that knows (salt, position) can add the same noise to its own
logits. Acceptance on real agent traffic rose from 72.4% to 75.0% here with an MTP head (+3.1% tokens per cycle, paired replay of
4,752 positions); a better-calibrated drafter gains much more (synthetic: 0.39 -> 0.88). Same seed gives the same text with and
without speculation, which is not true of rejection-sampling schemes.
[Patch, part A](../patches/coupled-sampling-fast-sampler.patch), [exactness test](../tools/coupled_sampling_check.cpp),
[offline replay of drafter settings](../tools/couple_fit.py).
This changes which random stream a seed produces, so it should start as a discussion. A submitter needs: the argument for exactness
including the grammar resample path (it keeps an independent stream, reproducing the current rejection-resample law), and a cleaner
channel between drafter and sampler than the token-tail registry used here.

### 9. Qwen3.8-Flash-Next: cache the pooled indexer keys instead of re-pooling the whole cache every token

Now that `qwen4exp` is upstream this is the highest-value model-specific item. Upstream's `qwen4exp.cpp` still says "cached indexer
keys are raw: pooling precedes norm and rotation" (line 598) and pools at read time, i.e. every decode token gathers, pools, norms and
ropes ALL cached indexer keys before scoring: measured here as 51 of the 87 ms of growth between 200 and 32,000 tokens
(GET_ROWS 40%, ROPE 17%, TOP_K 9%). Caching the post-rope pooled keys in the indexer cache's unused V half (allocated, never written)
and refreshing only the trailing `ceil(n_tokens/r)+1` blocks is byte-identical and gave +13.8% at 30K and, with items 4 and the
split-cap fix, 3.40 -> 6.25 tok/s at 237,500 tokens. It loses ~8-10% at 190 tokens (fixed bookkeeping), so it must be gated on context.
[Patch against the PR-era tree](../patches/qwen4exp-pooled-key-cache.patch), [Qwen notes](../patches/README-20260915-qwen4exp.md),
[qwen-flash-next guide](models/qwen-flash-next.md).
Upstream's `llama-memory-hybrid-idx` is not the tree this was written on: the idea transfers, the patch must be redone and
re-measured there. A submitter needs: why post-rope values are cacheable (block position is a pure function of block index), why the
dirty set needs no bookkeeping, and greedy parity on a long prompt.

### 10. Per-socket CPU devices and tensor parallelism across them

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
Two design points from 09-20 belong in any proposal:
- a backend instance must NOT own its worker team. Each context creates its own backends, so a server with a draft model ran two
  OpenMP teams pinned to the same cores, and libgomp's default idle spin (300,000 iterations, ~15 ms with Skylake's 140-cycle `pause`,
  not the documented 3 ms) made every hand-off between trunk and draft a collision. One dispatcher per DEVICE, shared by all backends,
  removes it ([patch](../patches/cpu-numa-shared-team.patch)). Stock upstream does not have this problem with OpenMP: both contexts
  enter their parallel regions from the same server thread and share one team;
- `set_tensor` on a group of NUMA buffers must not start a thread per device for small uploads. It did, for every graph input (~93
  thread creations per decode cycle), 4.6% of decode ([patch](../patches/meta-backend-small-uploads-blocking-dispatch.patch)).
  Upstream's own `ggml-backend-meta.cpp` does not spawn threads there; this only matters for the fork's backend.

## Tier 3: kernels; benchmark against upstream's current kernels before claiming anything

### 11. x16 AVX-512 VNNI GEMV family and the Q5_K sub-block prefetch

This fork carries 16-row VNNI kernels for Q4_K/Q5_K/Q6_K/Q8_0 (measured at 92% of a bare read loop per socket on 09-11) and for
MXFP4 ([16.3 GB/s per core](../engineering/2026-09-12/), `GGML_CPU_X16_MXFP4=1`), plus the 09-20 finding that the Q5_K expert kernel
read at 77 GB/s per socket against 97 for Q4_K because it walks a 2,880-byte block group by sub-block (320-byte stride) and the hardware
streamer does not follow; one `_mm_prefetch` of the next group per line fixed it, +3.1% decode, bit-exact
([patch](../patches/q5k-x16-expert-prefetch.patch); prefetch distance 4 was 65% SLOWER, so the distance is not a free parameter).
Upstream has since added 8x8 kernels for Q4_K/Q5_K/Q6_K/Q2_K and 16x1 AVX-512 kernels for Q4_0/Q4_K/Q8_0/Q2_K; there is no
measurement of those against this family on the same machine, so there is no claim to make yet. First step for a submitter: build
upstream, run `llama-bench` on the same GGUF, compare GB/s per socket per type; if upstream's Q5_K path shows the same sub-block walk,
the prefetch is a two-line PR with a clear mechanism, and Q5_K/Q6_K 16-row kernels are the gap if upstream's 8x8 is measurably slower.

## Blocked on architecture support; the human can help those PRs land

### 12. GLM-5.3-Flash (`glm5next`)

Not in upstream at `b23efaa2`. Two competing PRs existed on 08-26 (#27754 by unsloth with vision, #27752 text-only); their state must
be checked before anything else, and reviewing/testing them is the most useful thing a human can do for GLM-5.3-Flash users. All of
these wait for the architecture and are model code, not ggml:
pooled-indexer kernel fusion and wide pooling ([1](../patches/glm5next-kpool-fusion-statecopy.patch),
[2](../patches/glm5next-kpool-wide.patch), +3.8% at 4K, +13.4% at 30K), the pooled-result cache
([3](../patches/glm5next-pool-result-cache.patch)), the batch-invariant MQA attention kernel
([4](../patches/glm5next-fa-mqa-cellsplit.patch)), MTP catch-up that writes only K/V for accepted rows
([9](../patches/glm5next-mtp-kv-only-catchup.patch), +2.5%) and MTP query rows that build Q/attention only for the predicting row
([10](../patches/glm5next-mtp-query-rows.patch), +2.3%, +4.5% at 13K). The two MTP savings are ideas that transfer to upstream's
`draft-mtp` path for any architecture with a NextN layer; the code does not.
The whole 12.0 -> 20.2 tok/s record is in [the benchmark](../benchmarks/glm53-flash-paseo-decode-20260920.md) and the
[patch notes](../patches/README-20260920-glm5next.md).

### 13. DeepSeek-V4.1

Not in upstream. The port here ([workspace notes](models/deepseek-v41.md)) is JigSaw's third-party llama-level diff (46 files, no ggml
changes) carried onto the NUMA-tuned engine; it is theirs to submit. What is ours and could go as review evidence on their PR: the
converter traps (lazy architecture map, `text_config` flattening before `index_tensors`, `generate_extra_tensors` must chain to
`super()`), the native Engram lookup validation (267,583,488 exact BF16 values), and the observation that V4.1's compression ratio is
one general pooling factor where the V4 port hard-codes two. See [results](results.md).

## Already fixed upstream

The scheduler cut a new split when a split reached `GGML_SCHED_MAX_SPLIT_INPUTS` (30) inputs, which cost 11 ms per token on Qwen and was
worked around here with `-DGGML_SCHED_MAX_SPLIT_INPUTS=64`. At `b23efaa2` the input arrays grow on demand and that cut is gone.
`qwen4exp` is already excluded from `--split-mode tensor` upstream, so the fork's safety patch for that is moot there.

## Worth an issue: F16 vision projectors overflow on CPU

ggml-cpu's `mul_mat` converts its activations to the weight type's `vec_dot_type`, which is F16 for an F16 weight. Any activation
above 65,504 becomes infinity before the dot product. Late ViT blocks reach 1e5 (MiMo-V2.6-Pro's last SwiGLU output: 1.077e5), so an
F16 `mmproj` answers most images with `?` repeated. `ggml_mul_mat_set_prec(GGML_PREC_F32)` does not change this on CPU. The
workaround is an F32 projector (no slower here); BF16 would keep the range but is emulated without AVX512-BF16.

The report should include:

- the per-node statistics that found it: [vision-embd.cpp](../engineering/2026-09-23/archive/serving/mimo-v26-pro/tools/vision-embd.cpp)
  with `VE_STATS=1`;
- the tensor-by-tensor comparison against the vendor's reference module
  ([notes](../engineering/2026-09-23/archive/serving/mimo-v26-pro/STATE-VISION-20260922.md)).

Also small: `clip.cpp` ignored `--image-min/max-tokens` for the `mimovl` projector. The fix is one line
([patch](../engineering/2026-09-23/archive/serving/mimo-v26-pro/patches/mimovl-image-token-limits.patch)). Check whether upstream
carries `mimovl` at all before filing.

## Not candidates

- Non-temporal stores in the cross-socket all-reduce, the Meta trailing-subgraph fix, small-upload/blocking-dispatch: the fork's
  backend only; upstream has no equivalent code path.
- Constant-shape (padded) draft batches, draft merge and fast pick: workarounds for the fork's single-slot graph reuse. The upstream
  answer is a small graph cache per context, which this fork's Meta backend cannot support yet.
- `topk-linear-selection` (changed model behaviour on `-inf` ties), `qwen4exp-ssm-mirror-type-independent` (disproven), load-time
  requantisation (slower, hurt MTP acceptance): refuted here, recorded so nobody re-submits them.
- Worth an ISSUE rather than a patch: `Q4_K_M` quantises Qwen3.8-Flash-Next's `ssm_alpha`/`ssm_beta`/`hc_*_inject` vectors to q4_K
  while Unsloth's UD quants keep them f32; the type difference changed the fork's split rules and may deserve a type rule in
  `llama-quantize`. Quality impact was not measured.

## Suggested order, and the checklist for every submission

Order: 1 (unary threading) → 2 (`seq_rm`) → 3 (FA precision, as an issue first, then the grouped split-KV kernel) → 3a (vector max)
→ 3b (tool-call newlines) → 4 (`GET_ROWS`) → 5/6 → discussion for 8 and 10 → 9 once re-done on upstream's `qwen4exp` → 11 after the
kernel comparison → help 12 and 13 land as a reviewer. The F16 projector overflow is an issue, not a patch.

Before each one (from the 09-12 handoff): search open PRs and issues for the same change and comment there instead of duplicating;
rewrite the commit message and PR description yourself; fill in the AI-usage disclosure; run `test-backend-ops` where a second backend
exists — in `test` mode on a CPU-only box it compares nothing and prints OK having run zero tests, so use `perf` mode there.
