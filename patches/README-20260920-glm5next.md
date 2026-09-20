# GLM-5.3-Flash decode patches, 09-18 to 09-20

Seven changes to the GLM runtime (`llama.cpp-glm5n-goal-0904` line) measured on a Lenovo SR950 (4x Xeon Gold 6242, 755 GiB DDR4-2400,
381.6 GB/s measured, no GPU), plus one candidate for upstream. Rates are single-stream decode tok/s of an actual Codex turn through the
Paseo wrapper at ~4,000 input tokens, modes compared inside one loaded server. Report:
[benchmarks/glm53-flash-paseo-decode-20260920.md](../benchmarks/glm53-flash-paseo-decode-20260920.md).

| # | patch | library | switch | result | output | state |
|---|---|---|---|---|---|---|
| 1 | `glm5next-kpool-fusion-statecopy.patch` | libggml-cpu | `GGML_CPU_GLM_POOL_FUSION=1` `GGML_CPU_CPY_FLAT=1` | state copy 20.4 -> 1.2 ms per cycle; native 12.65/14.62/14.17 -> 14.10/16.21/15.71 for the copy alone | byte-identical | deployed |
| 2 | `glm5next-kpool-wide.patch` | libggml-cpu | `GGML_CPU_GLM_POOL_WIDE=1` | 14.60 -> 15.15 (+3.8%); at 29,930 tokens 9.05 -> 10.26 (+13.4%) | byte-identical | deployed |
| 3 | `glm5next-pool-result-cache.patch` | libggml-cpu | `GGML_CPU_GLM_POOL_CACHE=1` | 17.23 -> 17.61 (+2.3%) | identical | measured, not deployed |
| 4 | `glm5next-fa-mqa-cellsplit.patch` | libggml-cpu | `GGML_F18_FEATURES` bit 0 | with #5: 16.24 -> 17.23 (+6.1%); op 1.16 -> 0.23 ms | **changes** (stock is ~1% off float64) | measured, not deployed |
| 5 | `topk-select-tie-fallback.patch` | libggml-cpu | `GGML_F18_FEATURES` bit 1 | op 0.26 -> 0.06 ms per layer at 1,026 pools | identical set | measured with #4 only |
| 6 | `glm-kv-seq-rm-used-prefix.patch` | libllama | `LLAMA_KV_SEQ_RM_USED_PREFIX=1` | 15.24 -> 15.61 (+2.4%) | identical | deployed |
| 7 | `glm5next-mtp-kv-only-catchup.patch` | libllama | `GGML_GLM5N_MTP_KV_ONLY=1` | 15.43 -> 15.82 (+2.5%) | identical | deployed |
| - | `upstream-cpu-fattn-f32-accumulate.patch` | upstream ggml-cpu at `b23efaa2` | none | error 6.4e-3 -> 8.6e-5, op +17-21% | toward float64 | candidate, not submitted |

1-5 are one ordered series over parents that are already published with matching hashes
(`engineering/2026-09-12/archive/serving/fleet-0911/parallel-unary-0911/ggml-cpu.patched.c` and `ops.patched.cpp`); each header names its
parent and the resulting library hash. 6 and 7 are independent libllama deltas and carry the exact parent library hashes.
Every switch defaults off. `GGML_F18_CONTROL_FILE` and the `*_CONTROL_FILE` variables map a 4-byte file so a benchmark can flip a
feature inside a loaded server; change them only while the server is idle.

## Reading the numbers

The rows were measured over three days against different controls, so the percentages do not multiply into one figure. What can be
said: production gave 12.0 tok/s on this fixture before #1 (one cold, sampled run) and 15.5-16.2 greedy with #1, #2, #6, #7 deployed;
with #3-#5 added the same process measured 17.61 against 16.24.

#4 is the only change whose output differs from production, and the reason is production: the stock kernel sums V in FP16
([report](../benchmarks/cpu-flash-attn-f16-accumulation.md)). Its gate is a float64 reference computed in situ, not byte parity.

#5 is not [topk-linear-selection.patch](topk-linear-selection.patch). That one changed the selected set whenever scores tied at `-inf`
and lost 7-17% of throughput through draft acceptance. This one returns the identical set or declines; it also only runs below 16,384
entries, because `std::partial_sort` is faster again on long rows.

## How they were built

Each library is the production link line with one object replaced, after two lineage gates: the unmodified source recompiles to the
byte-identical production object, and the unmodified link reproduces the production library hash. Build scripts and the window
controller are in [engineering/2026-09-20](../engineering/2026-09-20/README.md).
