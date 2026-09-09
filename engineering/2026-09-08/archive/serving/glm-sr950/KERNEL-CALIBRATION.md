# SR950 kernel calibration — why GLM-5.2 is slow, measured

Measured 2026-08-25. Separate file from `README.md` to avoid collision with the
concurrent GLM tuning session. Harness: `/tmp/kbench.cpp` (links
`libggml`/`libggml-base`/`libggml-cpu` directly).

## Production status update (2026-08-26)

This file preserves the calibration evidence and the historical reasoning that
selected the implementation. The two central blockers identified below are now
resolved: CPU repack is exposed per NUMA device, exact compact IQ2_XS and
IQ3_XXS VNNI formats are implemented, and the dense fused Q5_K path uses its
repacked traits. The accepted real-shape PGO build is live.

At the selected 256K tier, current clean measurements are 3.9949 tok/s raw,
5.2699 tok/s on the general speculative suite, and 6.3139 tok/s cold / 7.4954
tok/s warm on the final production agentic replay profile. Direct maximum
absolute error is 1.56e-7 for IQ2_XS,
2.01e-7 for IQ3_XXS, and at most 5.50e-4 through the complete fused Q5_K
gate/up graph. See `IQ2-REPACK-DESIGN.md`,
`NUMA-REPACK-DESIGN.md`, and `README.md` for current status. Sections labeled
historical below must not be read as outstanding work.

## Promoted fused x86 Q5_K repack (2026-08-26)

The old profile made Q5_K look like the next obvious production target, so an
exact load-time Q5_K expansion and AVX-512/VNNI GEMV/GEMM path was implemented
behind `GGML_CPU_Q5_K_REPACK=1`. It preserves the
original fp16 scales, fp16 minimum scales, exact 6-bit scale/minimum values, and
all 5-bit weights; only the in-memory layout changes. The expanded layout is
8.625 bpw versus 5.5 bpw on disk and adds about 4.4 GiB for this model's 12.13B
Q5_K parameters.

The isolated checker compared stock and repacked graphs built from identical
Q5_K bytes. All tested shapes passed, with maximum absolute differences from
9.54e-6 to 1.14e-5 and normalized MSE below 3.8e-14. The difference is floating
point reduction order, not weight requantization. Candidate tracing confirmed
that both load-time repack and the AVX-512/VNNI kernel executed.

Pinned 16-core, single-socket microbenchmarks established the kernel ceiling:

| shape | stock | Q5 R8 | speedup |
| --- | ---: | ---: | ---: |
| 2048 rows, batch 1 | 421.19 us | 60.50 us | 6.96x |
| 3072 rows, batch 1 | 345.78 us | 92.60 us | 3.73x |
| 4096 rows, batch 1 | 256.04 us | 139.85 us | 1.83x |
| 2048 rows, batch 4 | 501.99 us | 231.86 us | 2.16x |

The first service candidate did not accelerate the dense fused gate/up path.
The custom repack traits caused the old fusion dispatcher to reject those
tensors, so the candidate paid two separate graph operations. A fresh profile
of the post-IQ/PGO engine made the omission explicit:

| function or runtime region | exclusive CPU samples |
| --- | ---: |
| `ggml_vec_dot_q5_K_q8_K` | 28.63% |
| libgomp static wait | 27.92% |
| `ggml_vec_dot_q8_0_q8_0` | 12.52% |
| compact IQ2_XS R8 | 10.52% |
| compact IQ3_XXS R8 | 7.40% |

Caller attribution put 38.05 percentage points of Q5 activity through
`ggml_cpu_try_fuse_ops`, versus 7.22 through ordinary `MUL_MAT`. The accepted
implementation therefore added a custom-traits `MUL_MAT_SWIGLU` entry point.
It quantizes the common input once, runs both repacked projections, and applies
SwiGLU inside the existing tile. At the exact 512-row dense shape it measured
351.17 us stock versus 46.21 us fused-repacked, a 7.60x kernel speedup.

Direct fused-graph comparisons passed at 512/2048 rows and batch 1/4. Maximum
absolute error was 5.50e-4 and normalized MSE was at most 1.72e-13. Candidate
tracing confirmed that both Q5 tensors entered the repack buffer and the
AVX-512/VNNI kernel executed.

The authority is the matched 256K PGO service A/B, with identical NUMA tensor
split, thread count, model, prompts, and context:

| arm | raw decode | prompt ingest | general speculation decode |
| --- | ---: | ---: | ---: |
| compact IQ2/IQ3 production | 3.7150 tok/s | 10.1251 tok/s | **4.4560 tok/s** |
| Q5 R8 without repacked dense fusion | 3.8235 tok/s | 10.6961 tok/s | 4.2598 tok/s |
| **Q5 R8 with repacked dense fusion** | **3.9949 tok/s** | **10.7384 tok/s** | **5.2699 tok/s** |
| accepted change versus control | **+7.53%** | **+6.06%** | **+18.27%** |

The deterministic replay optimum differs from the general suite. The production
default (`n_max=64`, `p_min=0.8`, `n_match=24`) measured 6.2478 tok/s on that
replay. A request-scoped `p_min=0.9` measured 7.4407 tok/s on a warmed candidate
with all three checks correct. The final production process then measured
6.3139 tok/s on its first replay and 7.4954 tok/s on an immediate warm repeat,
again with every check correct. This cold/warm spread is server-history state
used by `ngram-mod`, not a model correctness difference. Conversely,
`p_min=0.9` reduced the general suite to 4.3894 tok/s. Production therefore
keeps the general default and exposes `glm-sr950-agentic` as an alias-scoped
0.9 default. Explicit request settings take precedence. Q5 R8 and the fused
path are enabled in production.

## Memory system is healthy — the box is not the problem

| scope | read bandwidth |
| --- | ---: |
| single socket, NUMA-local | **95.3 GB/s** (68% of 140.8 theoretical, 6-ch DDR4-2933) |
| 4 sockets concurrent, NUMA-local | **365 GB/s** (93.1 + 94.1 + 93.6 + 83.8) |
| 4 sockets, `--interleave=all` | 138.9 GB/s |

GLM-5.2 at 4.26 tok/s moves ~18.5 GB/token = **79 GB/s = 22% of available**.
Decode is compute-bound, not bandwidth-bound. 13 tok/s would need 240 GB/s
(66% of measured peak), which the memory system can supply.

Benchmark trap: a `s += a[i]` reduction is a serial FP-add dependency chain that
GCC will not vectorize without `-ffast-math`. It reports ~93 GB/s on this host,
which is exactly `2.9 GHz / 4-cycle latency * 8 B * 16 threads` — a coincidence
close enough to the real figure to be believed. Use independent accumulators or
`_mm512_stream_load_si512`.

## Kernel throughput at real expert shape

`[K=6144 x N=2048]` (one GLM-5.2 routed-expert gate/up matrix), M=1 decode GEMV,
16 threads pinned to one socket, best-of-5.

| type | bpw | default ms | repack ms | speedup | effective GB/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| **IQ2_XS** | 2.31 | 0.504 | n/a | - | **7.2** |
| **IQ3_XXS** | 3.06 | 0.377 | n/a | - | **12.8** |
| Q2_K | 2.62 | 0.326 | **0.178** | 1.84x | **23.2** |
| Q4_K | 4.50 | 0.278 | **0.188** | 1.48x | **37.7** |
| Q4_0 | 4.50 | 0.335 | **0.164** | 2.04x | **43.1** |

IQ2_XS sustains **7.6% of single-socket bandwidth**. Q2_K repacked is **2.83x
faster at a modestly larger footprint** (2.62 vs 2.31 bpw). Q2_K beats IQ2_XS by
1.55x even unrepacked, so a large part of the cost is the per-weight grid lookup
itself, not blocking.

## Three stacking defects at the time of calibration

**1. 94.4% of the model can never enter the existing repack path.** Parsing all
seven shards: IQ2_XS is 63.24% of params (137.7 GB) and IQ3_XXS 31.19%
(89.9 GB). The dispatch table contains repack traits for
Q4_0/Q4_K/Q5_K/Q6_K/Q2_K/IQ4_NL/MXFP4/Q8_0, but that list is not portable across
architectures. On x86-64, the current branches cover Q4_0, Q4_K, Q2_K, IQ4_NL,
and MXFP4. Q5_K and Q6_K are ARM-only; Q8_0 is ARM/RISC-V-only. Placing an
IQ2_XS tensor in `CPU_REPACK` **segfaults** —
`ggml_repack_get_optimal_repack_type` returns null, `extra` stays null, and
`get_tensor_traits` dereferences it. Structural, not a flag.

**2. Repack is unreachable for production's tensor-sharded weights.** The model
holds only 7.39B params / ~2.65 GB that the existing x86 dispatcher can repack
(Q2_K 6.44B and Q4_K 0.95B). Q5_K 12.13B and Q6_K 1.31B have traits but no x86
dispatch branch. `--device CPU-NUMA0..3 --split-mode tensor` selects NUMA-local
device buffers before the ordinary CPU extra-buffer fallback, so `buft` never
equals `ggml_backend_cpu_repack_buffer_type()`. Zero journal matches are
consistent with this but are not standalone evidence because the `repack
tensor` message is debug-level while production logs at info. The gprofng
profile's `ggml_vec_dot_q5_K_q8_K` at 27.95% is expected even on the plain CPU
backend until an x86 Q5_K repack kernel is added; it is not evidence by itself
that the global repack buffer failed to engage.

**3. Speculation was clamped.** `server-context.cpp:458` clamps every
implementation to `draft.n_max`, so `ngram-mod`'s 64-token drafts were truncated
to 1 by `GLM_SPEC_DRAFT_N_MAX=1`. Fixed; see `README.md`.

## Backend A/B: locality dominates today's small repack coverage

The deterministic three-case replay was rerun against the same model and
request configuration after removing the `--numa distribute` / external
`numactl` conflict. The plain CPU arm used `numactl --interleave=all`, engaged
the existing x86 repack path for Q2_K/Q4_K, and kept process swap at zero.

| backend | aggregate decode | placement | checks |
| --- | ---: | --- | ---: |
| four `CPU-NUMA` devices, tensor split | **6.551 tok/s** | balanced within ~1% | 3/3 |
| plain CPU + `CPU_REPACK` | **2.001 tok/s** | 69.5/70.8/71.5/83.7 GB | 2/3 |

The global-repack arm is **69.5% slower** despite page placement that is only
18% above the mean on its heaviest node. This rules out severe first-touch skew
as the sole explanation. A single cross-socket CPU backend loses the node-local
allocators, four pinned 16-core worker pools, and direct NUMA reduction, while
only ~2.65 GB of this quant can use the existing x86 repack kernels. The result
does not show that repack kernels are slow; it shows that global repack cannot
pay for losing the fork's four-device execution topology.

The next valid arm was therefore node-local repack: each `CPU-NUMA<N>` device
had to own a repack wrapper over its existing `mbind` allocator. That path is
now implemented behind `GGML_CPU_NUMA_REPACK=1`, fully validated, and enabled
in production.

## Historical projection before the format kernels landed

Re-weighting the gprofng shares by the measured or proxy ratios (q5_K/1.48,
IQ2_XS 2.83x via Q2_K, IQ3_XXS 2.12x, q6_K and q4_K ~1.5x; libgomp sync 18.93%
and tinyBLAS 6.19% unchanged) gives 0.679 of current total time = **~1.47x**, or
about **6.3 tok/s novel / 8.3 tok/s replay**. This is an engineering ceiling,
not a prediction for the existing `CPU_REPACK` switch: reaching it also requires
new x86 Q5_K/Q6_K and IQ2_XS/IQ3_XXS kernels. The current backend A/B can only
exercise the small Q2_K/Q4_K fraction. libgomp sync then becomes ~28% of what
remains and does not move.

The projection was useful for prioritization but was not a production result.
The landed implementation has not produced a measured 12 or 13 tok/s GLM-5.2
result; its best confirmed replay result is 7.0787 tok/s at 32K.

## Active bytes per token is the governing variable

| model | layers | hidden | experts | active bytes/token | quant | repack? |
| --- | ---: | ---: | --- | ---: | --- | --- |
| GLM-5.2 UD-Q2_K_XL | 79 | 6144 | 8 of 256 | ~18.5 GB | IQ2_XS/IQ3_XXS | **yes, compact exact RAM repack** |
| DeepSeek-V4-Flash UD-Q4_K_XL | 43 | 4096 | 6 of 256 | **>=5.8 GB** | 97.4% MXFP4 | **global CPU only; tensor split unsupported** |

The 5.8 GB figure counts routed-expert weights only. It is a lower bound, not a
complete bytes/token measurement: the graph also executes three always-on
shared-expert matrices per layer, hyper-connections, compressed attention, and
other dense tensors. MXFP4 repack still measures 33.5 GB/s versus IQ2_XS at
7.2 GB/s, but a 15+ tok/s prediction cannot be justified by dividing 138.9
GB/s by 5.8 GB. The corrected full-model benchmark is the authority.

Caveat: the fork rejects `--split-mode tensor` for arch `deepseek4`
("LLAMA_SPLIT_MODE_TENSOR not implemented"), so V4-Flash runs the interleaved
backend. The controlled GLM result above shows that balanced interleaving alone
does not guarantee efficient four-socket execution.

## Load-time IQ2/IQ3 repack — corrected design

The grid alphabets are small enough for exact compact representations:
IQ2_XS uses six signed values derived from magnitudes `{8,25,43}` and stores
3-bit codes; IQ3_XXS uses 16 signed values derived from
`{4,12,20,28,36,44,52,62}` and stores 4-bit codes. Keep the original global
fp16 scales and subscale nibbles while replacing grid indices and signs with
interleaved bitplanes. The resulting formats are 3.3125 bpw for IQ2_XS and
4.1875 bpw for IQ3_XXS. The installed GGUF projects to 322.73 GiB for compact
IQ2/IQ3 weights, or 327.14 GiB including the enabled expanded Q5_K path.

See `IQ2-REPACK-DESIGN.md` for the implemented layout, arithmetic, validation,
and measured production result.

## IQ2 unsigned-VNNI calibration (2026-08-26)

The first IQ2 kernel used signed alphabet values and paid absolute-value/sign
correction inside each VNNI tile. A matched real-shape comparator used
`[6144x512]`, 16 routed gate/up matrices, 16 threads pinned to one socket, and
identical packed weights and Q8 activations:

| layout and arithmetic | time | versus signed |
| --- | ---: | ---: |
| 3-bit planes, signed correction | 373.442 us | 1.000x |
| 3-bit planes, unsigned +64 bias | **355.457 us** | **1.051x** |
| 4-bit nibbles, unsigned +64 bias | 348.714 us | 1.071x |

Both alternatives matched the signed path exactly (`max_abs_error=0`). The
nibble layout was rejected: it gains only another 1.9% while adding one bit per
IQ2 weight, about 55.5 GiB on this model. Production now biases the table to
`{21,39,56,72,89,107}`, uses unsigned VNNI, and subtracts
`64 * sum(q8)` from the integer dot product once per 16-value subblock.

The forced `CPU_REPACK` correctness harness now retains a canonical source
shadow only inside the test process, allowing the ordinary reference graph to
read repacked weights without changing production allocation. Normal and
AddressSanitizer runs passed all three IQ2 `MUL_MAT_ID` cases, and the fused
`MUL_MAT_ID_SWIGLU` comparison passed. Post-change production-shape checks
measured 166.33 us for one routed projection and 437.00 us for fused gate/up;
the matched comparator above remains the speedup authority.

## IQ3 routed-down plus router-weight fusion (2026-08-26)

The compact IQ3 path now fuses the routed down projection with the following
router-weight multiply and eight-way reduction.  Each worker keeps a 192-row
tile in local scratch and writes the final weighted output instead of
materializing the `[6144,8]` intermediate.  Routing IDs are copied by thread 0
and published at the same barrier used by the established `MUL_MAT_ID` path;
letting every worker read the live ID tensor caused an assertion under the real
four-backend graph and was rejected.

Matched adjacent measurements on CPU-NUMA2, 16 physical-core threads, real
`[K=2048,N=6144]`, 256 global experts / eight routed experts:

| path | time | region speedup |
| --- | ---: | ---: |
| established down plus weighted-sum fusion | 961.59 us | 1.00x |
| direct down/weighted-sum, tile 192 | **804.16 us** | **1.196x** |

The full 256-expert exact validator reported `max_abs_error=0` at
`n_used=8,n_tokens=1`.  The all-experts llama.cpp warmup topology
(`n_used=256,n_tokens=2`, with the normal eight aggregation views) also passed
with exit 0.  The final three-sample production raw mean was **3.9971 tok/s**
versus **3.9377 tok/s** immediately before the change, a **1.51% end-to-end
gain** under host variance.  An earlier candidate sample set reached 4.0604,
so use the final conservative handoff number.

Production flags:

    GGML_CPU_MOE_DOWN_WEIGHTED_SUM_FUSION=1
    GGML_CPU_MOE_DOWN_WEIGHTED_SUM_TILE=192

Do **not** enable `GGML_CPU_MOE_DOWN_WEIGHTED_SUM_VALIDATE` in the live
four-device service.  That diagnostic mode passed both isolated topologies but
segfaulted during the multi-backend context warmup.  It is intentionally absent
from `model.env`; validation belongs in `test-backend-ops` until its scheduler
workspace plumbing is fixed.
