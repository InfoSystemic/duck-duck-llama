# DeepSeek-V4.1-Flash CPU bring-up — September 10, 2026

DeepSeek-V4.1-Flash has been added to the SR950 tuning work. The first native-format CPU component is implemented and tested. Its AVX-512 Engram lookup decoder improves the measured scalar component baseline by 1.48× on a 276.8 MB table fixture and 1.69× on a 270.3 KB fixture. Full-model loading, generated tok/s, and IMC bandwidth have not yet been measured; the V4.1 model graph is still missing from the checked llama.cpp runtimes.

The existing goal remains 250+ adjusted decode GB/s per model, including this fourth model. Qwen's separate Q6 goal remains 40+ generated tok/s. All models can use the server sequentially; no simultaneous residency with GLM Full or any other model is required.

## Pinned release and actual tensor inventory

The [official release](https://api-docs.deepseek.com/updates/#deepseek-v41-flash-release), [model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/fb2764a5cf321eaa5070ca8f9e892818f477c16d/README.md), and [configuration](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/fb2764a5cf321eaa5070ca8f9e892818f477c16d/config.json) identify a new `DeepseekV41ForCausalLM` architecture. The publisher describes 552B backbone parameters, 196B Engram parameters, 8B active parameters during prefill and 16B during decode, with native vision and three DSpark draft blocks. These are publisher architecture claims, not measurements on this server.

The [header audit](results/deepseek-v41-tensor-audit-0910/result.json) independently checks all 48 safetensors headers against all 96,085 entries in the official index. Tensor extents are contiguous, nonoverlapping, and account for the complete declared payload. Only 10,750,656 bytes of headers and sample rows were retrieved. The 510,296,708,312-byte checkpoint was not downloaded. Publisher whole-shard SHA-256 values are recorded; whole-shard verification has not been performed. Retrieved headers and sample slices have their own SHA-256 values.

| Tensor family | Native payload, decimal GB |
| --- | ---: |
| Routed backbone experts, FP4 plus E8M0 scales | 288.778 |
| Engram lookup tables, FP8 plus E8M0 scales | 202.758 |
| Other backbone tensors | 9.532 |
| DSpark draft tensors | 7.933 |
| Vision and aligner | 0.971 |
| Engram projections and normalization weights | 0.315 |

The intended baseline preserves the published FP4 expert codes and FP8 Engram tables. Dense matrix conversion still needs a separate precision audit. No lower-precision target or GGUF quant has been selected. The GGUF repository examined at revision `4c0d55c5207793ae77486c56a17c667f7507c493` contained a README and no weight files; its upload plans are not treated as usable weights.

## Native precision work completed

The [FP4 bridge](deepseek_v41_native_formats_0910.py) rearranges adjacent packed nibbles into ggml's MXFP4 blocks and preserves every code and scale byte. Four real expert samples, covering gate and down projection rows at two positions each, round-trip exactly and match ggml's finite decoded values. All 256 possible packed-byte values also round-trip. This is a bounded format primitive, not a complete V4.1 GGUF converter. The numeric comparison treats positive and negative zero equally; the packed bits themselves are preserved.

The [Engram decoder](deepseek-v41-engram-cpu-0910.cpp) reads the native FP8 row and its eight E8M0 scale bytes and emits BF16 with round-to-nearest-even. It has scalar and AVX-512 implementations, validates all row IDs before writing, and preserves the caller's MXCSR state. It is a standalone component awaiting graph integration.

The [numerical probe](results/deepseek-v41-native-cpu-0910/result.json) checks both implementations against an independent arithmetic reference over every one of the 65,536 FP8/scale byte combinations. It also checks 128 actual table rows, reordered and duplicated row IDs, invalid IDs, and empty inputs. The comparison canonicalizes NaNs rather than preserving NaN payloads. All sampled real rows are finite. The [reconciled validation audit](results/deepseek-v41-validation-audit-0910.json) includes 16 combinations of implementation, rounding mode, and flush mode, with unaligned buffers and output canaries. These checks establish the component's scope; they do not establish full-model quality or parity with the GPU reference runtime.

Keeping Engram tables native requires 202.758 GB, compared with 393.228 GB after full BF16 expansion: a 190.470 GB difference. Each text token selects 48 rows across two tables, totaling 12,672 bytes of row and scale payload. That byte count excludes cache lines, page faults, projections, and inter-socket transfers. The whole table is not read for every token.

## First CPU component timing

Eight runs use an off/on/on/off order for each footprint, pinned to CPU 48. Each invocation gathers 48 deterministic, varying row IDs. Output checksums match across implementations. Times below average the two run medians for each arm.

| Fixture | Scalar µs/call | AVX-512 µs/call | Scalar / AVX-512 |
| --- | ---: | ---: | ---: |
| 270.3 KB table | 18.147 | 10.753 | 1.688× |
| 276.8 MB table | 24.655 | 16.655 | 1.480× |

These are single-worker synthetic-table timings under recorded host load, with real weights used separately for correctness. They do not measure a complete 202.8 GB table, SSD lookup, multi-socket scaling, model speed, or memory-controller bandwidth. The small absolute lookup time also prevents treating this component ratio as a model speed multiplier.

## Remaining work before a model baseline

The [bring-up assessment](results/deepseek-v41-bringup-0910/result.json) checks architecture registration and conversion at upstream commit `4ea6d1bb6dac161f70be728983e2cd58e4d9246f` and in the current Qwen engine. Both lack V4.1 registration. The official reference implementation supplies the architecture, but its accelerator kernels are not a CPU serving path.

1. Add nested configuration and tensor conversion, CED/CSA2 KV sharing and hierarchical indexing, and single-pass mHC coefficient handoff.
2. Integrate the exact compressed-token map, n-gram history, native Engram gather, and projection/gate into the CPU graph; preserve image boundaries and request resets.
3. Add vision routing and the native vision path, then the three-block DSpark draft and verification lifecycle.
4. Resolve weight storage and transient load/repack memory. The observed free disk space was about 25.1 GB on the root volume and 7.7 GB on `/models`, which cannot hold a native checkpoint with a reserve. A sequential model switch is allowed; it does not create persistent disk capacity.
5. Establish deterministic text/vision correctness and fresh-request checks, then matched prose/code raw-decode and DSpark baselines using all 48 IMC counters and adjacent idle subtraction. Tune NUMA placement, thread count, expert kernels, and speculation from those results.

GLM Flash Q4 stayed healthy at PID 3647836 during these component experiments. No serving profile was promoted, no other model or weight was removed, and no 250 GB/s or model tok/s result is claimed.

## Reproduction and source notices

Run the one-shot scripts only into fresh output labels: [tensor audit](audit_deepseek_v41_tensors_0910.py), [native CPU probe](probe_deepseek_v41_native_cpu_0910.py), and [bring-up assessment](assess_deepseek_v41_bringup_0910.py). Their completed results bind inputs and binaries by SHA-256. The standalone C++ component can be built independently; the orchestration scripts use this server's lifecycle and measurement helpers.

[Pinned source provenance](results/deepseek-v41-intake-0910/sources.json) and [license provenance](results/deepseek-v41-intake-0910/license-sources.json) accompany the [DeepSeek MIT notice](results/deepseek-v41-intake-0910/official/LICENSE-DEEPSEEK-V41.md) and [llama.cpp MIT notice](results/deepseek-v41-intake-0910/upstream/LICENSE-LLAMA-CPP.md). Binary sample rows, built libraries, model weights, and raw logs are excluded from the source publication; their recorded hashes describe artifacts retained on the server.
