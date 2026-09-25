# GLM-5.3-Flash decode patches, 09-18 to 09-20

Seventeen changes to the GLM runtime (`llama.cpp-glm5n-goal-0904` line) measured on a Lenovo SR950 (4x Xeon Gold 6242, 755 GiB DDR4-2400,
381.6 GB/s measured, no GPU), plus one candidate for upstream. Rates are single-stream decode tok/s of an actual Codex turn through the
Paseo wrapper at ~4,000 input tokens, modes compared inside one loaded server. All seventeen are in production (revision h, 2026-09-20);
#16 and #17 were added on the evening of 09-20 and published on 09-24.
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
| 14 | `meta-backend-small-uploads-blocking-dispatch.patch` | libggml-base | `GGML_META_F18=3` | 19.30 -> 20.18 (+4.6%); sampled 17.98 -> 18.79; verify graph 119.7 -> 115.5 ms | identical |
| 15 | `q5k-x16-expert-prefetch.patch` | libggml-cpu | `GGML_F18_FEATURES` bit 3 | 19.57 -> 20.18 (+3.1%); expert down projection 77 -> ~95 GB/s per socket | bit-identical |
| 16 | `glm5next-pool-fusion-overlap-proof.patch` | libggml-cpu | none: part of `GGML_CPU_GLM_POOL_FUSION=1` | 4K: 19.98 / 20.26 / 20.33 against 20.15; the 8K-128K curve of a fresh session is unchanged (202 vs 198 ms per cycle at 128K); a session restored from the RAM prompt cache no longer drops its last DSA layer off the fused path | byte-identical |
| 17 | `glm5next-indexer-score-blocked.patch` | libggml-cpu | `GGML_F18_FEATURES` bit 4; the radix top-k rides bit 1 | 4K: 20.54 / 20.62 / 20.49 (the indexer is ~2 ms of a 105 ms graph there); score op 2.3x per core, top-k ~2.4x per row above 16,384 entries; the expected ~19 ms per cycle at 128K was not measured | bit-identical; identical set |
| - | `coupled-sampling-offline-fit.patch` | libllama-common | `GGML_F18_COUPLE_LOG` `GGML_F18_COUPLED_DRAFT_TEMP` | window w7 only: replay of drafter settings against logged draws; deployed setting is optimal, coupling +3.1% tokens per cycle | none; not deployed |
| - | `upstream-cpu-fattn-f32-accumulate.patch` | upstream ggml-cpu at `b23efaa2` | none | error 6.4e-3 -> 8.6e-5, op +17-21% | toward float64; candidate, not submitted |

1-6 are one ordered series over parents that are already published with matching hashes
(`engineering/2026-09-12/archive/serving/fleet-0911/parallel-unary-0911/ggml-cpu.patched.c` and `ops.patched.cpp`); 7 touches a different
file of the same library. 8-10 are libllama deltas and carry the exact parent hashes; 10 applies on top of 9. 11 then 12/13 are one series
over `common/speculative.cpp` and `common/sampling.cpp`. Every header names its parent; every series was re-applied to its parent and
compared byte for byte with the source that was built and deployed. Every switch defaults off.
`GGML_F18_CONTROL_FILE`, `GGML_F18_SPEC_CONTROL_FILE`, `LLAMA_F18_MTP_QROWS_CONTROL_FILE` and the older `*_CONTROL_FILE` variables map a
4-byte file so a benchmark can flip a feature inside a loaded server; change them only while the server is idle.

#16 and #17 are production revisions g and h. #16 fixes a rejection in the 14-node fusion matcher that a restored 128K session hit
(+51 ms per graph); a session that reaches 128K by appending never did, which is why the curve did not move
([correction](../benchmarks/glm53-flash-context-depth-20260920.md#where-the-growth-is-per-op-trace-same-session-one-sockets-verify-graph)).
Revisions i-l tried an F16 router, dense Q6_K requantisation and two composite n-gram/MTP drafters; all four were slower and were rolled
back ([numbers](../docs/models/glm-flash.md#revisions-gl-september-20-evening-what-24-toks-would-take)). Their drop-ins are archived
in the [2026-09-23 snapshot](../engineering/2026-09-23/README.md#glm-53-flash-september-20-evening).

## Reading the numbers

The rows were measured over three days against different controls, so the percentages do not multiply into one figure. What can be
said: production gave 12.0 tok/s on this fixture before #1 (one cold, sampled run), 15.5-16.2 greedy with #1, #2, #8, #9 deployed, and
20.2 greedy with all of them (19.7 on sampled requests exactly as Codex sends them, mean of twelve, none below 19).

#4 is the only change whose output differs from the stock engine, and the reason is the stock engine: its kernel sums V in FP16
([report](../benchmarks/cpu-flash-attn-f16-accumulation.md)). Its gate is a float64 reference computed in situ, not byte parity.
Its first revision had a defect of its own: a query's result depended on the other queries of its batch. Revision 2 is bit-identical
alone and in any batch ([check](../tools/fa_mqa_invariance_check.cpp)), and everything after it is gated on byte parity with it.

#5 is not [topk-linear-selection.patch](topk-linear-selection.patch). That one changed the selected set whenever scores tied at `-inf`
and lost 7-17% of throughput through draft acceptance. This one returns the identical set or declines; it also only runs below 16,384
entries, because `std::partial_sort` is faster again on long rows.

#7 exists because of #13: the faster sampler made decode 2.7% slower until the two worker teams stopped colliding. The spin-count
setting needs no code and is what moved the number; the shared team makes the collision impossible.

#15 corrects this repository's own claim that the expert kernels were finished: the Q5_K down projection ran 27% below the wall because
of its access order, and one prefetch per cache line fixed it. The prefetch distance is not a free parameter (four groups ahead: -65%).

#14 was found with `/proc` and strace, not with a profiler: the backend started one thread per NUMA device for every graph-input
upload (~93 thread creations per decode cycle). It also records the build recipe of the production libggml-base, which nobody had
written down: the engine's compile command plus `-mavx512f -mavx512bw` reproduces it byte for byte.

#12 does not make the drafter better, it makes the verifier's randomness shareable. The verifier's pick is an exact sample whatever the
drafter does ([test](../tools/coupled_sampling_check.cpp)); a fixed seed now gives the same text with or without speculation.
Shared noise also makes drafter settings replayable offline ([couple_fit.py](../tools/couple_fit.py)): on 4,752 logged positions the
deployed setting is worth +3.1% tokens per cycle over a greedy drafter and nothing in its family does better.

## How they were built

Each library is the production link line with one or two objects replaced, after two lineage gates: the unmodified source recompiles to
the byte-identical production object, and the unmodified link reproduces the production library hash. Build scripts, generators
(`make_*.py`), window controllers and deploy records are in [engineering/2026-09-20](../engineering/2026-09-20/README.md).
Production drop-ins of revisions b, c and d, with rollback instructions, are in `archive/serving/fleet-0920-flash18/deploy-0920*/`.

## Applying these to a reconstructed engine

Verified 2026-09-21 against a fresh reconstruction of the `llama.cpp-glm5n-goal-0904` bundle (base `2e0e57f1` of
unslothai/llama.cpp plus `engineering/2026-09-12/patches/llama.cpp-glm5n-goal-0904.patch`). Apply
[`glm5next-x16-moe-expert-bound.patch`](glm5next-x16-moe-expert-bound.patch) first or the model cannot decode a token at all.

**Six apply as they are** (`git apply <patch>`): `glm5next-x16-moe-expert-bound`, `cpu-numa-shared-team`,
`meta-backend-small-uploads-blocking-dispatch`, `mtp-draft-merge-fastpick-pad`, `q5k-x16-expert-prefetch`,
`glm5next-split-state-diagnostic`.

**Four were cut with bare filenames rather than repository-relative paths** and need the target directory supplied:

| patch | invocation |
|---|---|
| `glm5next-gdn-row-split` | `git apply --directory=ggml/src/ggml-cpu` |
| `topk-select-tie-fallback` | `git apply --directory=ggml/src/ggml-cpu` |
| `glm5next-mtp-kv-only-catchup` | `git apply --directory=src/models` |
| `glm-kv-seq-rm-used-prefix` | `git apply --directory=src` |

**The remaining eight are a chain and will not apply to the bare bundle.** They were cut against the private rebuilt sources
of the 09-11 and 09-19 workspaces, in this file's numbered order, and several depend on each other rather than only on the
base: `glm5next-kpool-fusion-statecopy` (#1) CREATES `pool-kernel.inc`, which `glm5next-kpool-wide` (#2) and
`glm5next-pool-result-cache` (#3) then modify, so neither can apply before it; `glm5next-mtp-query-rows` (#10) expects
`glm5next-mtp-kv-only-catchup` (#9) already applied; `glm5next-fa-mqa-cellsplit` (#4) expects the 09-11 `ops.cpp` overlay;
the two `coupled-sampling` patches expect the f18 `common/` overlays; and `meta-backend-trailing-subgraph` is a plain
`diff -u` carrying workspace paths and a timestamp rather than a git header.

The numbering in the table above is the application order. The earlier overlays those hunks assume live in the engineering
snapshots — start from `engineering/2026-09-12/archive/serving/fleet-0911/parallel-unary-0911/` and the `glm-*-0919`
directories under `engineering/2026-09-20/archive/serving/fleet-0912-ctx/`.

**A practical first pass**, if the goal is a working fast server rather than reproducing every measurement: take the base,
apply the expert bound, then the six-plus-four above. That gets the NUMA worker-team fix, the thread-free graph uploads
(+4.6%), the padded draft batches, the Q5_K expert prefetch (+3.1%), the bounded `seq_rm` scan (+2.4%), the row-split gated
delta net and the selection top-k, without touching the pooled-indexer chain.
