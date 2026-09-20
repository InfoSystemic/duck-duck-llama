# Wider GLM kpool inner loop, 2026-09-19

Deployed and verified on the normal GLM endpoint (:18131), PID 3692141.
The existing first fusion was already deployed; this change accelerates its inner
loop. The A/B completed, restored the original runtime, and passed final controls
before the wider loop was separately enabled for production. The deployed library
is an immutable copy under `deploy/`, not the rebuild output under `build/`.

Candidate CPU SHA256:
`3f957a341321b940d93be53c250cdd068825093faa2d9efda142ebe56427b1b3`.
Parent deployed CPU SHA256:
`4793379e6894a9286168f79c4f323985388ec0b0ac65e54013dcd2957488af1f`.

## Implementation

The unchanged strict 14-node matcher recognizes the same F16-cache pooling graph.
The new kernel looks up four member-cell pointers once per pool segment and handles
8 adjacent channels together. F16 conversion, APE addition, maxima, normalization,
and weighted accumulation use AVX instructions. Exponentials still call the same
scalar `expf`; the exact identity `expf(+/-0) == 1` avoids unnecessary calls.
Products round to float before sequential double accumulation; normalization uses
the original double reciprocal followed by float conversion. No approximate exp,
reciprocal, reassociation, graph mutation, cached summaries, or split-rule change.
Tails and exceptional values use the original scalar arithmetic. Unsupported CPU
builds and a disabled flag retain the original implementation.

Only `ops.cpp.o` is rebuilt. Every other object, including the fusion scheduler,
comes from the frozen deployed build. `parent-provenance.json` and
`build-provenance.json` record input hashes. Build with `bash rebuild.sh`.

Enable the private library with the existing production environment plus:

```sh
LIB_PREPEND=/home/user/InfoSystemic/AI-Server/serving/fleet-0912-ctx/glm-kpool-wide-0919/build
GGML_CPU_GLM_POOL_FUSION=1
GGML_CPU_GLM_POOL_WIDE=1
```

The new flag defaults off. The optional benchmark-only
`GGML_CPU_GLM_POOL_WIDE_CONTROL_FILE` maps a 4-byte little-endian 0/1 word read-only.
The controller changes it only between idle requests, enabling A/B alternation
without reloading the model. Normal use does not need this file. The exported
`ggml_cpu_glm_pool_wide_count()` verifies activation in standalone tests; bounded
`GGML_CPU_GLM_POOL_PROBE=1` logs provide live engagement evidence.

## Numerical gate

`validation-1,2,3,15.json`: 77 cases, twice with changed inputs, at 1/2/3/15 workers,
across unfused production, deployed fusion, candidate fusion-off, candidate scalar,
and candidate wide. All 85,564,960 output bytes match in every arm. Tests include
32K and 100K pool sizes, multiple streams, padded strides, allocator reuse,
vector boundaries, half subnormals/extremes, varied finite half bit patterns,
signed zeros, reordered MUL operands, and the original strict fallback cases.
Expected activation counts and loaded library paths/hashes are checked.

`validation-control-copy.json`: repeated scalar/wide switching passes byte parity
at 3 and 15 workers, and the existing 14-case flat-copy regression suite remains
exact. No claim is made that the pre-existing full-model cache inconsistency is fixed.

## Performance scope

Actual Codex/Paseo ABBA results at fixed generated text and identical input,
cache, generated-token, and speculation counters:

| Actual input | Scalar control | Wider loop | Gain |
|---|---:|---:|---:|
| 3,998 tokens | 14.60 tok/s | 15.15 tok/s | 3.8% |
| 29,930 tokens | 9.05 tok/s | 10.26 tok/s | 13.4% |

The two scalar controls drifted approximately 1.0% and 0.1%, respectively.
The long input is synthetic context filler plus an LRU explanation request.
18+ tok/s has not been reached on this standard Codex task. Cold deep-context
prefill still took approximately 12 minutes; these rates measure decode only.

Separate profiles with matching 7,490-pool shapes attributed 54.714 ms to the old
pool loop and 24.284 ms to the wider loop. Those individual sampled graph timings
are diagnostic; the unprofiled A/B above supplies the end-to-end result.

All six native, twelve Codex A/B, two instrumented Codex, and nine corresponding
stateful regression outputs matched. The pre-existing cached/fresh discrepancy
remains unresolved. After deployment, native parity and cold/cached Codex smoke
checks passed; those final Codex runs measured 15.04 and 15.06 tok/s.

The service drop-in is:
`~/.config/systemd/user/glm53-flash-production.service.d/70-glm-kpool-wide-0919.conf`.
It selects `deploy/` and enables `GGML_CPU_GLM_POOL_WIDE=1`. The benchmark control
file is absent from the production environment. Only the CPU library and new
flag differ from the restored baseline; other runtime identities were verified.

Full results: [KPOOL-WIDE-RESULT.md](/home/user/sr950-strategy/codex-bench-0919/KPOOL-WIDE-RESULT.md).
Runtime identities and raw evidence are in `codex-bench-0919/kpool-*.json`;
`kpool-promotion-final.json` is the final verified production snapshot.
