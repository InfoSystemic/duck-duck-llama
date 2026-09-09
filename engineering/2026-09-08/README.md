# CPU inference engineering snapshot, September 8, 2026

This snapshot publishes the GLM-5.3 Full, GLM-5.3-Flash, and Qwen3.8 engineering from the SR950 tuning campaign. It includes source patches, selected source overlays, numerical fixtures, benchmark controllers, analyses, launch settings, and unsuccessful experiments. It supersedes the older repository findings where the same model or implementation is discussed.

The server has four Xeon Gold 6242 sockets, 64 physical cores, 128 logical CPUs, about 811 decimal GB of RAM, and no GPU. The utilization denominator below is the requested approximately 380 decimal GB/s whole-server capacity. These are single-conversation decode measurements, with IMC counters and adjacent idle-traffic subtraction. They are not prefill rates or aggregate multi-user throughput.

## Retained measurements

| Model and configuration | Prose tok/s | Code tok/s | Adjusted decode GB/s, prose/code | Fraction of 380 GB/s |
| --- | ---: | ---: | ---: | ---: |
| Qwen Flash-Next UD-Q6_K_XL, selected Q8 MTP4 | 21.53 | 28.46 | 126.0 / 136.6 | 33-36% |
| GLM Flash Q8_0, separately measured MTP2 batching stack | 13.60 | 14.78 | 205.4 / 210.0 | 54-55% |
| GLM Flash Q8_0, retained raw stack, two runs | 11.42-11.47 | 11.45-11.49 | 234.5-237.1 across workloads | 61.7-62.4% |
| GLM Full, Q4-based mixed runtime, MTP default 2 | 7.89 | 9.64 | 196.7 / 199.0 | about 52% |

Exact counters, token counts, and evidence paths are in [measured-status.json](measured-status.json). The Flash MTP2 result predates the final raw stack; it is not a measurement of that combined stack with MTP enabled. Full's mixed runtime includes load-time requantization and a hybrid draft, so its filename alone does not describe runtime precision. Favorable Full file-edit replay results of 13.59-15.70 tok/s are a separate workload, not a general generation rate.

No measured configuration establishes 85-93% utilization. The current Qwen target, strictly above 30 generated tok/s and above 190 GB/s in the same configuration on fresh prose and code, remains unmet. A component kernel gain, a waiting-cycle percentage, or active-weight bytes multiplied by speculative output tok/s cannot establish that target.

## Source bundles and reconstruction

Every base patch is independent. Do not stack patches for different source lines. [source-bundles.json](source-bundles.json) records each full base commit, changed-file SHA-256, patch hash, and reconstructed Git tree. All eight patches passed `git apply --check` against their recorded bases, and applying each produced the expected source tree.

| Base patch | Repository and base commit | Purpose |
| --- | --- | --- |
| [qwen-flash-next-goal-source.patch](patches/qwen-flash-next-goal-source.patch) | ggml-org/llama.cpp `daef7b6874397a5a7c3d7e38b55e2ee0adf7da38` | Complete Qwen Flash goal engine before private library overlays |
| [glm-flash-goal-source.patch](patches/glm-flash-goal-source.patch) | unslothai/llama.cpp `2e0e57f1008053bae4902a772da85e3eb99d4aff` | Complete GLM Flash goal engine before private library overlays |
| [glm-full-sr950-source.patch](patches/glm-full-sr950-source.patch) | llama.cpp `a30273376ef669023334fc20ad02ae4ed8196a65` | Current Full integration, including server and speculative fixes |
| [qwen-flash-next-numa-source.patch](patches/qwen-flash-next-numa-source.patch) | llama.cpp `3173a56471c1753650cd806694145ffd6dcace67` | Earlier Qwen conversion, MTP and NUMA integration |
| [qwen-flash-next-bringup-source.patch](patches/qwen-flash-next-bringup-source.patch) | ggml-org/llama.cpp `daef7b6874397a5a7c3d7e38b55e2ee0adf7da38` | Earlier Qwen NUMA bring-up state |
| [glm-flash-bringup-source.patch](patches/glm-flash-bringup-source.patch) | unslothai/llama.cpp `2e0e57f1008053bae4902a772da85e3eb99d4aff` | Earlier GLM Flash backend and repack port |
| [qwen-dense-server-source.patch](patches/qwen-dense-server-source.patch) | llama.cpp `9d57ce456c94d241dde672b2db9cf18879766568` | Qwen dense chat/server fixes |
| [sr950-numa-source.patch](patches/sr950-numa-source.patch) | llama.cpp `9d57ce456c94d241dde672b2db9cf18879766568` | Earlier NUMA and scheduling source line |

For the selected Qwen Q6 source, apply `qwen-flash-next-goal-source.patch` followed by [qwen-q6-selected-overlay.patch](patches/qwen-q6-selected-overlay.patch). The overlay contains the 512-expert capacity fix, wider Q8 row batching, and balanced expert splitting. For the retained Flash Q8 raw source, apply `glm-flash-goal-source.patch` followed by [glm-flash-q8-raw-overlay.patch](patches/glm-flash-q8-raw-overlay.patch). [selected-overlays.json](selected-overlays.json) maps every overlay file back to its recorded source and hash. Each x86 overlay relocates one private absolute header include to `../../repack.h`; this changes no arithmetic and avoids mixing headers from two checkouts.

Example for Qwen, from a clean checkout of its recorded base:

```bash
git apply --check /path/to/llama-llama-duck/engineering/2026-09-08/patches/qwen-flash-next-goal-source.patch
git apply /path/to/llama-llama-duck/engineering/2026-09-08/patches/qwen-flash-next-goal-source.patch
git apply --check /path/to/llama-llama-duck/engineering/2026-09-08/patches/qwen-q6-selected-overlay.patch
git apply /path/to/llama-llama-duck/engineering/2026-09-08/patches/qwen-q6-selected-overlay.patch
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_OPENMP=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=OFF
cmake --build build --target llama-server -j4
```

Use the matching [parameterized profiles](../../profiles/20260908/README.md). Model shards and draft weights are separate downloads. Preserve a broad process affinity during NUMA device discovery; on this host the internal pools select 15 physical workers per socket. Initializing a multi-socket test with one-core affinity invalidates its parallel measurements.

The measured Qwen CPU library has SHA-256 `c79e19e38acfd6ada3f0637c136cf9cb10253127778cb214528b2f966fb6dfcf`, and its selected llama library is `d213ba477ef5d3f2a68568b30a31595ef9578b6dce4b3b81e95da6f83a3dac8b`. The Flash raw CPU library is `5ba125771f3e8bac0b634a845f6666700d53e25b90573581c62f8bbbb8718f2b`. These identify the original measurements. Clean builds can have different binary hashes and need fresh model validation; publishing the source does not certify a new binary's quality or speed.

## Engineering map

| Area | Implemented work and evidence entry point |
| --- | --- |
| NUMA correctness | Zero stale inactive-device buffers before direct reduction; handle empty attention slices; fix scheduler device recognition and buffer naming. See [Flash goal log](archive/serving/fleet-0903/FLASH-GOAL-20260904.md) and `patches/` under that archive. |
| Model sharding | Attention, MLA, shared-expert and HC projection splits; balanced expert FFN dimension splits; local repack placement and four-socket collectives. See [Qwen high-precision log](archive/serving/fleet-0903/QWEN-HIGH-QUANT-20260907.md). |
| Quantized kernels | IQ2/IQ3 exact layouts, x16 Q4/Q5/Q6/Q8, paired and batched activations, 512-expert capacity, wide Q8 verification rows, and dispatch checks. Source variants and fixtures are preserved beside their build records. |
| Graph work | Dense/MoE gate-up fusion, clamped SwiGLU, weighted down sums, fused reductions, parallel copies/sigmoid, exact top-k, pooling, guarded RMS and ordered narrow Q8 projections. See [bandwidth work log](archive/serving/fleet-0903/BANDWIDTH75-20260908.md). |
| MTP correctness | Detached draft extraction, hidden-state handling, recurrent/PLE history snapshots and rollback, cache-pool reuse, graph reuse, and launch-depth controls. The work logs distinguish repaired paths from unresolved prompt-cache behavior. |
| Measurement | [IMC reader](archive/serving/fleet-0903/dram_bandwidth.py), [decode measurement](archive/serving/fleet-0903/measure-model-bandwidth.py), operation/cycle profiles, resource gates, request monitors, and exact owned-process cleanup. |
| Serving | Selected model presets, exclusive model-load guard, fixed source/library identities, runtime packaging and model-quality probes. Full need not remain resident when another model is evaluated. |

## Selected and experimental status

Qwen keeps UD-Q6_K_XL with a Q8 draft and MTP4. The six target shards total 169,165,382,688 bytes at model repository revision `38bb39ee97821de2c9009abb7e93950eec396e66`. Fresh arithmetic, factual, generated-code and continuation checks passed within the recorded test scope. Explicit prompt-cache extension still fails parity; requests must use `cache_prompt=false`, and the selected preset disables reuse by default. This is not broad model-quality certification.

GLM Flash keeps the higher-precision quality objective. Q8 raw tuning is retained as an experimental source stack; no broad comparison with the source checkpoint establishes near-lossless quality. Lower-quant historical speed results do not replace that objective. The earlier lower-quant launcher in the archive is historical, not the selected Q8 profile.

The following candidates remain separate from the selected Qwen overlay:

| Candidate | Validation | Model status |
| --- | --- | --- |
| Packed-Q6 two/three-row batching | 432 arithmetic cases; 360 CPU graph cases plus four-NUMA validation | No established model speed gain. One prose bandwidth window had invalid idle subtraction; the qualification record rejects its attribution. |
| Single-row Q6 arithmetic variant | 432 cases, 146,880 exact outputs | Component attempts timed out at the CPU gate before measuring; no speed result or runtime integration. |
| Expanded lossless Q6 byte layout | Exact code/scale/output proof; about 30.5% larger Q6 storage | Qualified warm gains of 40-42%, but the cold 64-row single-activation case is about 17% slower. Global expansion is parked; the source transform remains uncompiled. |
| Dissemination barrier / OpenMP spin changes | 4,320 CPU graph cases plus NUMA checks | No repeatable Qwen model gain; not promoted. Flash candidate is parked. |
| HC-combine fusion | 384 CPU/NUMA cases, 24,676,920 bit-exact outputs; loader and build pass | Two off/on pairs show about 3-4% gains with exact output and draft-count agreement. Enabled repeat: 21.80/28.49 tok/s and 126.99/135.07 GB/s. Target remains unmet; no promotion. |
| HC multiply-and-mean fusion | 768 CPU/NUMA cases, 19,741,536 bit-exact outputs; private build and loader pass | First off/on pair, with HC-combine enabled: 21.98/28.90 tok/s and 127.51/137.04 GB/s. Outputs and draft counts match; gains of 2.87%/1.18% need repetition. Not promoted. |
| Flattened HC normalization | 768 CPU/NUMA cases, 35,892,640 bit-exact outputs; padded-input fallback, private build and loader pass | Clean completed pair: 21.95/28.66 tok/s disabled, 22.34/29.19 enabled, with 129.01/138.91 GB/s enabled. Outputs and draft counts match; gains of 1.75%/1.82% need repetition. Earlier code profile failed and remains separate; not promoted. |
| Q8 exact activation sums | 18,448 integer cases, 57,600 exact matrix outputs, 2,592 graph-case executions; private build and four-NUMA loader pass | First pair with all HC changes enabled: 22.69/29.78 tok/s disabled, 22.45/29.68 enabled. Complete outputs and draft counts match. No demonstrated model gain; not promoted. |
| Block-parallel activation quantization | 720 helper cases, 127,476,000 exact bytes, 2,025,000 block claims; private build, 1,728 CPU graph cases, 14 NUMA arms and loader pass | First pair with all HC changes enabled: 22.52/29.40 tok/s disabled, 22.22/16.91 enabled. Complete outputs and draft counts match. Prose is 1.36% slower and code is 42.48% slower; the large code slowdown is unexplained and unreplicated. Not promoted. |

See the [current Qwen log](archive/serving/fleet-0903/QWEN-30TPS-20260908.md) for exact failures, corrections, and evidence paths. The HC controller now rejects bandwidth attribution if either adjacent idle window exceeds 19 GB/s or if their difference exceeds 9.5 GB/s. This stricter qualification is not retroactively assumed for every older result.

The private HC source is preserved in the [stream-mean build](archive/serving/fleet-0903/results/qwen-hc-mix-build-0908/private-llama/) and [normalization build](archive/serving/fleet-0903/results/qwen-hc-norm-flat-build-0908/private-llama/), including complete changed model files, helper headers, patches, and manifests. These HC candidates keep the selected C79 CPU library and Q6/Q8 weights. The [stream-mean comparison](archive/serving/fleet-0903/results/qwen-private-comparison-hc-mix-0908.json) and [clean normalization comparison](archive/serving/fleet-0903/results/qwen-private-comparison-hc-norm-flat-0909.json) record their completed model pairs. The [earlier failed normalization controller](archive/serving/fleet-0903/results/qwen-private-hc-norm-flat-on-profile-0908/result.json) remains failed; its [validated prose analysis](archive/serving/fleet-0903/results/qwen-hc-norm-flat-prose-analysis-0909/result.json) qualifies only that capture, not the failed code arm. Profiled token rates are excluded from the table above.

The [Q8 sum build](archive/serving/fleet-0903/results/qwen-q8-sums-build-0909/private-cpu/) records the changed x86 source and exact-sum helper. Its [model comparison](archive/serving/fleet-0903/results/qwen-private-comparison-q8-sums-0909.json) uses a separate private CPU library with the existing HC normalization llama library.

The [block-quantization helper](archive/serving/fleet-0903/qwen-quantize-blocks-0909.h) and [helper correctness result](archive/serving/fleet-0903/results/qwen-quantize-blocks-proof-0909/result.json) are followed by the [integrated private CPU source](archive/serving/fleet-0903/results/qwen-quantize-blocks-build-0909/private-cpu/), [graph validation](archive/serving/fleet-0903/results/qwen-quantize-blocks-validation-0909/result.json), and [completed model comparison](archive/serving/fleet-0903/results/qwen-private-comparison-quantize-blocks-0909.json). This experiment uses CPU hash `c476ac4e475a9f5e541bf20a0182049fbdab8cece8a44a81254ebdb8a4e1d91f` with the HC normalization llama library. It excludes the Q8 sum candidate and leaves the selected deployment unchanged. Q6 NUMA checks exercise the new expert path; Q8 NUMA checks use the existing non-x16 dispatcher and establish compatibility only. The model pair has valid adjacent-idle bandwidth windows, but does not demonstrate a speed gain or reach either target on both workloads.

## Archive contract and verification

[archive-manifest.json](archive-manifest.json) inventories every exported source and evidence file, with its original and published SHA-256. Sources and notes are preserved byte for byte. JSON evidence is filtered to remove process inventories, full environment snapshots and response bodies; it is explicitly not byte-identical to the original raw evidence. Large arrays may be represented by a count and hash. Original raw logs, perf captures, model weights, libraries, binaries, credentials and private context are not part of this publication.

The archive preserves the historical directory layout, host paths, PIDs, experiment names and failure versions so the source can be audited. Its controllers are historical research code, not portable turnkey launchers. Many require the original object files, private runtime bundles, control files or process identities, and their integrity checks can intentionally reject a reconstructed environment. Use the source patches and parameterized profiles for a new deployment; use the archive to port individual experiments and understand their validation.

Run `python3 tools/verify_engineering_snapshot.py` from the repository root to verify the published hashes and syntax. [validation.json](validation.json) records publication-time checks separately from historical model results. The reconstructed Qwen and GLM Flash source stacks built a Release `llama-server`, passed `--version`, and exposed all four NUMA devices without loading model weights. Archive hashes, Python and shell syntax, and profile rendering checks passed; exact counts are recorded in the validation file. No model benchmark is rerun merely by verifying the archive.

Upstream-derived code retains its original notices. The base repositories, earlier [patch attribution](../../patches/ATTRIBUTION.md), and individual source diffs describe provenance; this snapshot does not claim original authorship of upstream model implementations.
