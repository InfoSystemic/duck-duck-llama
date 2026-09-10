# DeepSeek-V4.1 Engram hashing on CPU — September 10, 2026

The native token-history and hash component now matches the pinned publisher CPU reference exactly. Both the division baseline and reciprocal-division candidate pass 16,091,808 row-ID comparisons across 120 sequence cases. The compressed tokenizer vocabulary is exactly the required 99,092 IDs. This completes another standalone prerequisite for DeepSeek-V4.1-Flash; it does not yet make the full model loadable.

The reciprocal implementation reduces measured single-token hashing from 0.314 µs to 0.091 µs. Its 3.47–4.74× component speedup is small in absolute terms and is not a model-speed multiplier. The 250 GB/s objective, complete model graph, and model tok/s baseline remain open.

## Exact tokenizer and hash metadata

The [setup result](results/deepseek-v41-hash-reference-setup-0910/result.json) records an isolated CPU-only reference environment: NumPy 2.3.3, SymPy 1.14.0, Tokenizers 0.22.2, and PyTorch 2.10.0+cpu. The environment is separate from the serving runtimes. The 6,367,257-byte tokenizer and 801-byte tokenizer configuration come from official revision `fb2764a5cf321eaa5070ca8f9e892818f477c16d`; both are SHA-256 bound. No full checkpoint was downloaded.

The [reviewed publisher implementation](results/deepseek-v41-intake-0910/official/inference/engram.py) executes on CPU as the test oracle. Its exact normalization sequence, partial-UTF-8 handling, odd seeded multipliers, and disjoint prime-sized bucket layout produce:

| Item | Value |
| --- | ---: |
| Original vocabulary | 129,280 |
| Compressed vocabulary | 99,092 |
| Compressed pad ID | 2 |
| Engram layer IDs | 1 and 14 |
| Rows in the two tables | 384,006,168 and 384,016,682 |
| Hash columns per layer | 24 |
| Total selected rows per token | 48 |
| Serialized token map | 517,120 bytes |
| Token map plus layout/multipliers/header | 517,596 bytes |

The token-map SHA-256 is `c60a86322ec17b4142bfef3c57a8d81fb428550cdf487f88a4320cb59fe46481`. The [correctness result](results/deepseek-v41-engram-hash-0910/correctness.json) preserves all primes, multipliers, source hashes, version identities, and per-case output hashes. The binary map and metadata are retained on the server and recreated by the checker; source publication excludes binary artifacts and the large tokenizer JSON.

The serving component consumes this precomputed map and metadata. It does not need Python, PyTorch, a tokenizer library, or a random-number generator during inference.

## Native state and arithmetic

The [C++ component](deepseek-v41-engram-hash-0910.cpp) implements the fixed two-layer, four-token, eight-head V4.1 hash layout. It writes 48 integer row IDs per input token. Native integer products and XORs preserve the reference arithmetic; configuration validation bounds products to the signed 64-bit range used by the publisher.

The baseline uses ordinary integer remainder. The candidate precomputes `floor(2^64 / modulus)`, computes a high-half product to estimate the quotient, and conditionally subtracts the modulus once. The [checker](check_deepseek_v41_engram_hash_0910.py) independently verifies 316,946 remainder cases using Python arbitrary-precision arithmetic, including unsigned 64-bit boundaries beyond the model's product range.

Each mutable handle belongs to one sequence. A handle may append tokens, replace a populated suffix, rewind to an existing position, or clone its state for an independent speculative branch. Clones share immutable metadata and copy token history. Position-zero reuse and suffix replacement shorten the valid history; later appends cannot read stale tokens beyond that boundary. Image-mask tokens stop lookback, including when the boundary crosses a prefill/decode split.

Invalid tokens, masks, positions, output capacities, and sequence overruns are checked before output or history changes. Batch tests use independent native handles for each sequence. Callers must serialize operations on a mutable handle and provide valid, separate input/output buffers. This is a component API, not an integrated server request lifecycle or DSpark implementation.

## Validation and timing

The [completed probe](results/deepseek-v41-engram-hash-0910/result.json) compiles the shared component and direct C++ benchmark with warnings treated as errors. Both native implementations match the official CPU reference across the entire vocabulary in sequence, random batches, all 16 four-token mask patterns, every split of a 17-token sequence, variable chunks, image boundaries, branch clones, rewinds, suffix replacement, and request reuse. It also verifies 16 invalid input calls leave outputs and state unchanged and rejects 11 malformed configurations. Output canaries remain intact.

Twelve timing runs use off/on/on/off order at each batch size, pinned to CPU 48. Each arm contains 21 timing samples; the table averages the two arm medians. Deterministic varying input and output checksums prevent dead-work comparisons and match across modes.

| Tokens per call | Integer division, µs/call | Reciprocal, µs/call | Baseline / reciprocal |
| ---: | ---: | ---: | ---: |
| 1 | 0.3144 | 0.0905 | 3.473× |
| 128 | 37.7349 | 7.9530 | 4.745× |
| 4,096 | 1,193.0670 | 295.4040 | 4.039× |

These timings include token-history writes and row-ID output, with existing host load recorded. They do not include Engram table reads, projections, model compute, or IMC counters. The [independent audit](results/deepseek-v41-engram-hash-audit-0910.json) reconciles pinned reference sources, derived metadata, result counts, binary/source hashes, all timing logs, and preservation of Flash Q4 at PID 1173340.

## Integration still required

The [subsequent combined lookup](DEEPSEEK-V41-ENGRAM-LOOKUP-20260910.md) now connects these hashes to the native FP8 gather with exact CPU reference checks and 48 actual selected rows. Connect that lookup to the model's projection/gate next. Add the converter metadata and graph nodes, then the CED/CSA2 attention, mHC, vision, and DSpark paths described in the [main bring-up report](DEEPSEEK-V41-BRINGUP-20260910.md). Weight storage and transient loading still need a complete plan. No new serving profile has been selected and no DeepSeek model throughput has been measured.

Reproduction entry points are the [isolated setup](setup_deepseek_v41_hash_reference_0910.py), [component probe](probe_deepseek_v41_engram_hash_0910.py), and [audit](audit_deepseek_v41_engram_hash_0910.py). Completed inputs and outputs are frozen; use new output labels for reruns. The state/hash algorithm follows the pinned DeepSeek implementation under its [MIT notice](results/deepseek-v41-intake-0910/official/LICENSE-DEEPSEEK-V41.md).
