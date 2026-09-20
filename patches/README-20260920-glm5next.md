# GLM-5.3-Flash decode patches, 09-18 to 09-20

Thirteen changes to the GLM runtime (`llama.cpp-glm5n-goal-0904` line) measured on a Lenovo SR950 (4x Xeon Gold 6242, 755 GiB DDR4-2400,
381.6 GB/s measured, no GPU), plus one candidate for upstream. Rates are single-stream decode tok/s of an actual Codex turn through the
Paseo wrapper at ~4,000 input tokens, modes compared inside one loaded server. All thirteen are in production (revision d, 2026-09-20).
Report: [benchmarks/glm53-flash-paseo-decode-20260920.md](../benchmarks/glm53-flash-paseo-decode-20260920.md).

| # | patch | library | switch | result | output |
|---|---|---|---|---|---|
| 1 | `glm5next-kpool-fusion-statecopy.patch` | libggml-cpu | `GGML_CPU_GLM_POOL_FUSION=1` `GGML_CPU_CPY_FLAT=1` | state copy 20.4 -> 1.2 ms per cycle; native 12.65/14.62/14.17 -> 14.10/16.21/15.71 for the copy alone | byte-identical |
| 2 | `glm5next-kpool-wide.patch` | libggml-cpu | `GGML_CPU_GLM_POOL_WIDE=1` | 14.60 -> 15.15 (+3.8%); at 29,930 tokens 9.05 -> 10.26 (+13.4%) | byte-identical |
| 3 | `glm5next-pool-result-cache.patch` | libggml-cpu | `GGML_CPU_GLM_POOL_CACHE=1` | 17.23 -> 17.61 (+2.3%) | identical |
| 4 | `glm5next-fa-mqa-cellsplit.patch` (rev 2) | libggml-cpu | `GGML_F18_FEATURES` bit 0 | with #5: 16.24 -> 17.23 (+6.1%); op 1.17 -> 0.28 ms | **changes** (stock is ~1% off float64); batch-invariant |
| 5 | `topk-select-tie-fallback.patch` | libggml-cpu | `GGML_F18_FEATURES` bit 1 | op 0.26 -> 0.06 ms per layer at 1,026 pools | identical set |
| 6 | `glm5next-gdn-row-split.patch` | libggml-cpu | `GGML_F18_FEATURES` bit 2 | verify graph 120.3 -> 119.5 ms | bit-identical |
| 7 | `cpu-numa-shared-team.patch` | libggml-cpu | `GGML_CPU_NUMA_SHARED_TEAM=1` (and `GOMP_SPINCOUNT=20000`, no code) | spin: 18.2-18.4 -> 18.81; team: 260 -> 200 threads, speed-neutral after the spin fix | identical |
| 8 | `glm-kv-seq-rm-used-prefix.patch` | libllama | `LLAMA_KV_SEQ_RM_USED_PREFIX=1` | 15.24 -> 15.61 (+2.4%) | identical |
| 9 | `glm5next-mtp-kv-only-catchup.patch` | libllama | `GGML_GLM5N_MTP_KV_ONLY=1` | 15.43 -> 15.82 (+2.5%) | identical |
| 10 | `glm5next-mtp-query-rows.patch` | libllama | `LLAMA_F18_MTP_QROWS=1` | with #6: 19.11 -> 19.55 (+2.3%); at 13,231 tokens +4.5% | bit-identical |
| 11 | `mtp-draft-merge-fastpick-pad.patch` | libllama-common | `GGML_F18_MTP_MERGE` `_FAST_PICK` `_PAD` | 17.97 -> 18.40 (merge +0.5%, padding +2.1%) | identical |
| 12 | `coupled-sampling-fast-sampler.patch` (A) | libllama-common | `GGML_F18_COUPLED=1` | sampled requests 17.89 -> 18.24, acceptance 72.4% -> 75.0% | exact sampler, same distribution |
| 13 | `coupled-sampling-fast-sampler.patch` (B) | libllama-common | `GGML_F18_FAST_SAMPLER=1` | 18.81 -> 19.11 (+1.6%) | identical |
| - | `coupled-sampling-offline-fit.patch` | libllama-common | `GGML_F18_COUPLE_LOG` `GGML_F18_COUPLED_DRAFT_TEMP` | window w7 only: replay of drafter settings against logged draws; deployed setting is optimal, coupling +3.1% tokens per cycle | none; not deployed |
| - | `upstream-cpu-fattn-f32-accumulate.patch` | upstream ggml-cpu at `b23efaa2` | none | error 6.4e-3 -> 8.6e-5, op +17-21% | toward float64; candidate, not submitted |

1-6 are one ordered series over parents that are already published with matching hashes
(`engineering/2026-09-12/archive/serving/fleet-0911/parallel-unary-0911/ggml-cpu.patched.c` and `ops.patched.cpp`); 7 touches a different
file of the same library. 8-10 are libllama deltas and carry the exact parent hashes; 10 applies on top of 9. 11 then 12/13 are one series
over `common/speculative.cpp` and `common/sampling.cpp`. Every header names its parent; every series was re-applied to its parent and
compared byte for byte with the source that was built and deployed. Every switch defaults off.
`GGML_F18_CONTROL_FILE`, `GGML_F18_SPEC_CONTROL_FILE`, `LLAMA_F18_MTP_QROWS_CONTROL_FILE` and the older `*_CONTROL_FILE` variables map a
4-byte file so a benchmark can flip a feature inside a loaded server; change them only while the server is idle.

## Reading the numbers

The rows were measured over three days against different controls, so the percentages do not multiply into one figure. What can be
said: production gave 12.0 tok/s on this fixture before #1 (one cold, sampled run), 15.5-16.2 greedy with #1, #2, #8, #9 deployed, and
19.2 greedy with all of them (18.2 on sampled requests exactly as Codex sends them, mean of twelve).

#4 is the only change whose output differs from the stock engine, and the reason is the stock engine: its kernel sums V in FP16
([report](../benchmarks/cpu-flash-attn-f16-accumulation.md)). Its gate is a float64 reference computed in situ, not byte parity.
Its first revision had a defect of its own: a query's result depended on the other queries of its batch. Revision 2 is bit-identical
alone and in any batch ([check](../tools/fa_mqa_invariance_check.cpp)), and everything after it is gated on byte parity with it.

#5 is not [topk-linear-selection.patch](topk-linear-selection.patch). That one changed the selected set whenever scores tied at `-inf`
and lost 7-17% of throughput through draft acceptance. This one returns the identical set or declines; it also only runs below 16,384
entries, because `std::partial_sort` is faster again on long rows.

#7 exists because of #13: the faster sampler made decode 2.7% slower until the two worker teams stopped colliding. The spin-count
setting needs no code and is what moved the number; the shared team makes the collision impossible.

#12 does not make the drafter better, it makes the verifier's randomness shareable. The verifier's pick is an exact sample whatever the
drafter does ([test](../tools/coupled_sampling_check.cpp)); a fixed seed now gives the same text with or without speculation.
Shared noise also makes drafter settings replayable offline ([couple_fit.py](../tools/couple_fit.py)): on 4,752 logged positions the
deployed setting is worth +3.1% tokens per cycle over a greedy drafter and nothing in its family does better.

## How they were built

Each library is the production link line with one or two objects replaced, after two lineage gates: the unmodified source recompiles to
the byte-identical production object, and the unmodified link reproduces the production library hash. Build scripts, generators
(`make_*.py`), window controllers and deploy records are in [engineering/2026-09-20](../engineering/2026-09-20/README.md).
Production drop-ins of revisions b, c and d, with rollback instructions, are in `archive/serving/fleet-0920-flash18/deploy-0920*/`.
