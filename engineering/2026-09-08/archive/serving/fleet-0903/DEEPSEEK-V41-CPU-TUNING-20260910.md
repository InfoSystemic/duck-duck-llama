# DeepSeek V4.1 Flash: native CPU tuning follow-up

DeepSeek-V4.1-Flash is available at `http://127.0.0.1:18170/v1`. The independent live audit measures **1.800 warm decode tok/s**, with individual requests at 1.799 and 1.800. The selected implementation uses native quantization, quantization reuse, grouped MoE, native HC, and scripted sparse attention. The **over-10 tok/s goal remains unachieved**.

All reported model comparisons use the real released text checkpoint at revision `fb2764a5cf321eaa5070ca8f9e892818f477c16d`. The fixed prompt is `Hi.` and the response is `Hello! How can I help you today?`. Each response contains ten tokens including EOS; decode throughput counts the nine forward passes after prefill. These short cached requests establish matched timing and parity with the existing CPU path. Longer varied generation, official GPU parity, and broad quality remain unmeasured.

## Serving and matched results

The fused CPU configuration previously raised matched decode from 1.097 to 1.781 tok/s, with candidate measurements at 1.773 and 1.789. It combines native activation quantization, scoped expert-input reuse, grouped MoE projections, native HC, and scripted sparse attention. The complete-model evidence preserves exact logits at fixed Torch16/native16 settings.

| Experiment | Measured decode | Result |
| --- | ---: | --- |
| Fused CPU operators | 1.773–1.789 tok/s | Earlier selected baseline; exact compared logits |
| CPU0–63, native16 / Torch16 | 1.340–1.341 tok/s | Slower across sockets |
| CPU0–63, native32 / Torch16 | 1.242–1.249 tok/s | Slower across sockets |
| CPU0–63, native48 / Torch16 | 1.057 tok/s | Slower across sockets |
| CPU0–63, native64 / Torch16 | 0.786–0.850 tok/s | Slower across sockets |
| Expanded FP8 weights | 1.610–1.666 tok/s | Controls 1.619–1.620; no consistent gain; unselected |
| First exact row16 FP4 kernel | 1.725 tok/s | Matched control 1.634; 5.5% aggregate gain; insufficient per-run margin |
| Exact row16 FP4 with vector fallback | 1.754 tok/s | Matched control 1.633; 7.4% gain; both candidate requests exact |
| Independently audited selected HTTP endpoint | 1.800 tok/s | Two cached requests; zero downloaded bytes |

The live endpoint comparison measured 1.690 to 1.745 tok/s (3.3%). It did not reach the predeclared 5% serving threshold, so the previous implementation was restored. The candidate result and fallback reason are preserved in the promotion evidence. Selection is recorded in [the mutable selected manifest](deepseek-v41-selected.json); the snapshot audit records the exact PID/start identity and source hashes. The [live handoff evidence](results/deepseek-v41-lattice16b-promoted-0910/result.json) includes repeated HTTP controls and candidate checks.

The four-socket trial applied interleaving to new anonymous allocations; existing cached weight pages were retained at their previous locations. Its end-of-request thread-affinity snapshots describe threads remaining after inference. These results do not establish a limit for a fully optimized NUMA layout or a separate persistent native worker pool.

## Kernel and profiling evidence

The FP8 expansion preserves every E4M3 value and the original K32 reduction tree. It passes 286,624 exact-output checks and temporarily expands 290 common matrices into 10,989,076,480 additional bytes. Some hot component shapes improve, but the complete-model comparison provides no consistent speed gain. The serving implementation retains native FP8 storage.

The FP4 row16 integer path uses unchanged checkpoint nibbles and E8M0 scales. A block is eligible when each FP8 activation is exactly an integer multiple of 1/64. FP4 weights times two are integers with magnitude at most 12, and activation integers have magnitude at most 28,672. Every partial sum is bounded by 32 × 28,672 × 12 = 11,010,048, below 2^24. The original FP32 tree is therefore exact in units of 1/128 on these blocks. Other blocks evaluate that same FP32 tree. The second revision vectorizes the fallback across 16 output rows. It passes 411,900 grouped outputs plus 8,712 additional format outputs, including all FP8 activation codes and the cancellation/tie case that changes under an unrestricted integer reduction.

The revised component controls drift by 11% and 25%. Each candidate arm still beats both controls by at least 31%, which justified a diagnostic full-model comparison; the component ratios are not model speedups. An initial model launch stopped at the stricter component timing gate before constructing the model and restored its endpoint. Its failed record is preserved separately from the completed fresh comparison.

Warmup diagnostics from the real model, weighted by projection tasks:

| Projection rows × input width | Eligible blocks | Total blocks | Eligible share |
| --- | ---: | ---: | ---: |
| 2304x5120 | 681,144 | 691,200 | 98.55% |
| 5120x2304 | 132,026 | 155,520 | 84.89% |

The packed expert cache stays bounded at 64 GiB and releases copies with resident-expert eviction. The greeting's decode path packs 19,856,793,600 bytes. Prefill keeps the original arithmetic. The serving entry point leaves collection of activation-block diagnostics disabled.

The corrected wall profile of the fused path measures approximately 1.74 seconds in grouped expert GEMMs, 1.01 seconds in other FP8 GEMMs, 0.49 seconds in sparse attention, 0.39 seconds in the grouped BF16 attention output projection, and 0.27 seconds in the vocabulary head across nine decode passes. The full profiled control is about 5.56 seconds. Timers are inclusive and overlap where calls nest. A separate sampled CPU-cycle profile attributes 32.39% of sampled user cycles to FP4 and 29.35% to FP8 kernels; cycle shares are not wall-time or memory-bandwidth fractions.

Row-tiling variants, bounded OpenMP spin waits, the earlier exact standalone VNNI prototype, and the unrestricted grouped VNNI experiment remain unselected. Their source and result records are retained. The selected-source hashes and completed result records identify the serving implementation and each prototype's validation status.

## Current limits and next work

Context remains 256 combined tokens, output at most 128 tokens, greedy and text-only. The temporary native cache is bounded at 90 GiB; the complete approximately 510 GB checkpoint is not resident. New prompts can fetch missing expert tensors and Engram rows. Vision and DSpark remain disabled in the audited endpoint. The existing GLM Flash process was preserved during these experiments; it is not a requirement for future whole-server work.

DeepSeek IMC bandwidth has not been measured. These results establish no 250 GB/s, 85%, 93%, or over-10 tok/s claim. The profile points toward faster large projections, lower operator overhead, and improved native scheduling and memory placement. Further gains must be measured on repeated full-model requests, followed by longer varied generation.

Evidence: [fused model trial](results/deepseek-v41-goal-fused-0910/trial/result.json), [four-socket trial](results/deepseek-v41-four-socket-0910/trial/result.json), [FP8 fixtures](results/deepseek-v41-fp8-bf16-0910/kernel-check.json), [FP8 model trial and wall profile](results/deepseek-v41-fp8-expanded-model-0910/trial/result.json), [CPU dispatch profile](results/deepseek-v41-fp8-expanded-model-0910/trial/cprofile.json), [first row16 trial](results/deepseek-v41-lattice16-model-0910/trial/result.json), [revised fixtures](results/deepseek-v41-lattice16b-0910/kernel-check.json), [revised complete-model trial](results/deepseek-v41-lattice16b-model-0910b/trial/result.json), [initial timing-gate failure](results/deepseek-v41-lattice16b-model-0910/result.json), [independent endpoint audit](results/deepseek-v41-tuning-audit-0910.json), and [retained-expert baseline](DEEPSEEK-V41-RESIDENT-EXPERTS-20260910.md).
