# Private GLM kpool fusion candidate, 2026-09-19

Historical isolated candidate; its source and build remain frozen. This fusion
was subsequently measured and incorporated into the deployed pool+copy library.
See [the combined deployment status](../glm-pool-copy-0919/README.md). The source
switch defaults off; the production service explicitly enables it. The original
private-build instructions and test scope below are retained for provenance.

## Source and integration

The actual live CPU parent is `fleet-0911/parallel-unary-0911`, not the engine's
`ggml-cpu.c`. That parent already fuses SOFT_MAX + MUL + SUM_ROWS, and production
already exports `GGML_CPU_SOFTMAX_POOL_FUSION=1`.

`generate.py` derives private CPU sources and a relink script from that parent.
`pool-fusion.inc` recognizes exactly the 14-node GET_ROWS-through-SUM_ROWS graph
emitted by `glm5next.cpp:480-504`. It checks operand identities, types, dimensions,
strides, offsets, softmax parameters, external users, and output/input overlap.
Unsupported layouts and nonmatching graphs fall back to existing operations.
The model graph and meta-backend split rules are unchanged.

`pool-kernel.inc` gathers four key/gate pairs directly from the F16 cache, adds the
slot's F32 APE, and executes the existing deployed pool kernel's `ggml_vec_*`
sequence. Products remain float; sums and reciprocal use `ggml_float`, as before.
Each output row is independent, so worker assignment changes no arithmetic.

A scheduling prerequisite is included under the same flag: F16 GET_ROWS nodes
named `indexer_pool_members` use the existing parallel row-copy implementation.
Production otherwise hardcodes their task count to one, bypassing the fusion
selector completely. Thus a rejected fusion may still get a bit-preserving
parallel gather. Ordinary F16 GET_ROWS nodes retain their current scheduling.

This single-thread gather explains at least some of the slow pooling trace.
Existing documents' broader claims about shared physical destinations and
coherence remain hypotheses; no such assumption is needed by this candidate.

## Build and standalone correctness

From this directory:

```sh
python3 generate.py
taskset -c 124-125 bash rebuild.sh > build.log 2>&1
bash build-test.sh
```

`test.py` runs actual production unary-lib, candidate with the flag off, and
candidate with the flag on. It asserts loaded CPU/base library identities and
compares complete output bytes. It also checks exact expected fusion counts,
so a non-engaging kernel cannot pass as a fused implementation.

The harness has 21 cases, two executions with changed inputs each. Cases cover
128/129/tiny channel counts, 1/2/3 streams, odd pool counts, padded cache/cell
strides, the context allocator and graph allocator's storage reuse, reordered
MUL operands, signed zeros, F16 subnormals/extreme finite values, large gate
magnitudes, duplicate/missing pool members (cell zero), and 1,026 pools.

Fallback gates cover kpool=3, softmax scale != 1, an intermediate marked as an
output, another consumer of the gathered members, an extra APE VIEW node with
padded strides, F32 cache input, in-place ADD, and output/cache storage overlap.
Single-thread execution deliberately takes the old path. Set `TEST_THREADS=1`,
`2`, `3` (default), or `4` to test different worker counts.

Numerical agreement in this harness does not prove complete model correctness.
The final server test must include greedy text equality, pool integrity,
cache/rollback, context length, concurrent streams, and the existing battery.

## Coordinated server gate (owner: root)

Use the existing validated launcher, preserving its entire environment and
library chain. The only candidate settings are:

```sh
LIB_PREPEND=/home/user/InfoSystemic/AI-Server/serving/fleet-0912-ctx/glm-pool-0919/build
GGML_CPU_GLM_POOL_FUSION=1
GGML_CPU_GLM_POOL_PROBE=1
```

The launcher honors `PORT`, so the candidate can use the coordinated test port
18141 while production is stopped. Do not launch another loaded server without
coordinating service ownership and available memory with root.

Assert `/proc/PID/maps`, library SHA, and all activation variables before results
are trusted. Preserve `glm-fix`, `unary-lib`, private-cpu, and
validated-chunk16-bin in their existing order after LIB_PREPEND.

With the probe enabled, first eight matches print:

```
GLM_POOL_FUSED name=... d=128 pools=... streams=... threads=15
```

First sixteen rejected pooling graphs print `GLM_POOL_REJECT line=... name=...`.
These line numbers refer to `pool-fusion.inc`. The exported function
`ggml_cpu_glm_pool_fused_count()` supplies process-local counts to the standalone
harness. Omit the probe after engagement is established.

Promotion requires a positive, reproducible end-to-end result and all correctness
gates. No throughput claim is made from the earlier design's projections.

## Optional CPY diagnostic

`GGML_CPU_CPY_ADDRESS_PROBE=1` prints first eight thread-0 `cache_s_l0` CPYs,
including source/destination pointers, threadpool identity, dimensions, and
strides. It leaves the existing copy and synchronization unchanged. This tests
the older single-writer experiment's unverified shared-destination assumption.
The previous single-writer candidate failed text parity twice and must not be
re-enabled based on its apparent throughput.
