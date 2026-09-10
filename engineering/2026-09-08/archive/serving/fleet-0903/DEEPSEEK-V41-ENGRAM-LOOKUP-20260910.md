# Native DeepSeek-V4.1 hash-to-Engram lookup — September 10, 2026

The native token-history/hash component now feeds the native FP8 Engram decoder through a validated C API. Across eight variants, 267,583,488 BF16 values and 1,044,096 row IDs match the pinned official CPU reference exactly. This integrates lookup selection, state handling, row-shard dispatch, and native table decoding. The projection, residual gate, and complete V4.1 model graph are still pending; no model throughput or IMC bandwidth was measured.

The [completed probe](results/deepseek-v41-engram-lookup-0910/result.json) builds the new [lookup implementation](deepseek-v41-engram-lookup-0910.cpp) against the frozen [hash](deepseek-v41-engram-hash-0910.cpp) and [FP8 decoder](deepseek-v41-engram-cpu-0910.cpp). Its [public header](deepseek-v41-engram-lookup-0910.h) documents shapes, ownership, validation, and sequence semantics. Compilation uses C++17, optimization, and warnings as errors. No selected runtime or model weights were modified.

## Exact layout and state

Each call accepts one sequence chunk and emits BF16 bits shaped `[tokens][2][24][256]`. Optional selected-row output has shape `[tokens][2][24]`. The two table indices correspond to model layers 1 and 14. The API accepts unordered row-shard descriptors, sorts them at construction, and requires exact coverage of each table without gaps or overlaps. Native E4M3 weights and E8M0 scales remain in their original storage format; only selected rows are decoded.

All descriptors and hash metadata are copied. Weight and scale storage is borrowed, immutable, and must remain readable for the handle and its clones. The forward path allocates no heap memory and uses preallocated scratch IDs. Disjoint hash buckets make selected IDs ordered within each layer; consecutive IDs that land in one shard are gathered directly into their contiguous output span. This adds address dispatch, not a NUMA placement policy.

Construction validates table extents, buffer spans, modes, and capacities. Forward validates caller output sizes and token/history inputs before changing state. Empty calls leave history unchanged. Appends, suffix replacement, position-zero reset, rewind, and independent clones retain the preceding hash component's valid-history rules. Image-mask positions stop n-gram lookback but still produce pad-hashed rows, matching the official embedding path; the later residual gate must apply the mask to suppress its contribution.

Callers must serialize mutation of each handle and provide valid, disjoint buffers. Clones share immutable tables and metadata while retaining independent history and scratch space. This is not yet connected to llama.cpp request lifecycle or DSpark verification.

## Validation

The [checker](check_deepseek_v41_engram_lookup_0910.py) executes the pinned official `NgramHashState` and the reviewed `ParallelEngramEmbedding` class on CPU. It extracts only that embedding class from the publisher's model source, avoiding accelerator imports. A compact oracle table contains the selected rows remapped to dense indices; the native side uses the original large row IDs and table dimensions.

| Check | Result |
| --- | ---: |
| Sequence cases | 68 |
| Native variants: 1/4 shards × division/reciprocal × scalar/AVX-512 | 8 |
| Exact BF16 value comparisons | 267,583,488 |
| Exact selected-row comparisons | 1,044,096 |
| Rejected input calls, state/output unchanged | 112 |
| Rejected configurations | 19 |
| Actual publisher rows selected by token ID 42 at position zero | 48 |
| Retrieved real row and scale payload | 12,672 bytes |

Cases cover batches through independent sequence handles, chunks of 1–512 tokens, all 16 four-token image masks, every nonempty split of a 17-token sequence, variable prefill/decode chunks, optional row-ID output, clone branching, rewind, overwrite, request reset, and input errors. Output canaries remain intact. Matching appends after rejected calls also check that history survived those failures.

The checker reserves the complete 202,758,032,400-byte native table address spans with Linux sparse anonymous mappings. It populates 105,868 distinct fixture rows across both layers, not the complete tables. This exercises real-width row offsets and shard boundaries without downloading or physically allocating 202.8 GB of weights. One- and four-shard tests run on one CPU; they establish dispatch correctness, not four-socket scaling or cold-memory timing.

For the real-data case, 96 bounded HTTP range responses retrieve exactly the 48 selected weight rows and their scale rows from official revision `fb2764a5cf321eaa5070ca8f9e892818f477c16d`. The [fixture manifest](results/deepseek-v41-engram-lookup-0910/real-rows.json) records original public URLs, tensor names, byte extents, source-header hashes, and slice hashes. Requests require HTTP 206, an exact Content-Range, and the exact bounded response size. All selected real outputs are finite and agree bit-for-bit. Publisher whole-shard hashes are recorded but have not been verified; full shards and binary row slices are excluded from source publication.

The [independent audit](results/deepseek-v41-engram-lookup-audit-0910.json) reconciles all counts, source/binary identities, pinned source provenance, header offsets, and real-row bytes. It independently recomputes the 48 actual IDs using Python integer arithmetic and the real BF16 values using a separate arithmetic reference. Flash Q4 remains healthy at PID 2334769 throughout the probe and audit.

## Next integration work

Connect this lookup to the Engram projection and masked residual gate, then integrate it with the new CED/CSA2 attention, mHC, vision, converter, and DSpark paths in the [bring-up report](DEEPSEEK-V41-BRINGUP-20260910.md). Keeping the complete tables native still avoids about 190.5 GB of BF16 expansion. The combined lookup has no new timing claim; earlier isolated hashing and decoding speedups do not multiply model tok/s.

Reproduction uses the [guarded probe](probe_deepseek_v41_engram_lookup_0910.py), checker, and [independent audit](audit_deepseek_v41_engram_lookup_0910.py). Completed sources and results are frozen; use fresh labels for reruns. The official reference remains covered by its [retained MIT notice](results/deepseek-v41-intake-0910/official/LICENSE-DEEPSEEK-V41.md).
