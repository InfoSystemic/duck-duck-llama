# Private all-device GLM state-copy candidate, 2026-09-19

Default off. Independent of the immutable `glm-pool-0919` candidate; built directly
from the production `fleet-0911/parallel-unary-0911` sources. No shared sources,
launchers, services, or model files changed. Full-model performance and promotion
remain unverified.

## Evidence and exact change

Production `ggml_compute_forward_dup_bytes` handles a noncontiguous same-type
copy by partitioning only `ne[1]`. Real recurrent-state copies have
`ne=[262144,1,2,1]`, source strides `[4,1048576,1048576,2097152]`, destination
strides `[4,1048576,2097152,4194304]`. The destination has a gap between planes,
so the fully contiguous shortcut fails. With `ne[1]=1`, thread zero copies all
2 MiB and the other workers do no copying.

A live pointer probe on 2026-09-19 also disproved the old single-writer patch's
shared-destination premise. Four devices write four different state destinations
(`0x762088bbc000`, `0x76207b164000`, `0x76206d70c000`, `0x762096614000`), consistent
with the meta backend's separate per-device buffer allocations. Evidence:
`/home/user/sr950-strategy/codex-bench-0919/pool-candidate-server.log`, lines 8–11.
The previous single-writer candidate failed greedy parity twice and is rejected.

This candidate keeps all devices' state writes and preserves the existing graph
barrier. It splits the logical packed-row byte stream among all local workers
at 64-byte boundaries; each worker copies its part of each row using memcpy.
There is no arithmetic change, cross-device synchronization, or writer skipping.

The path requires `GGML_CPU_CPY_FLAT=1`, a `cache_s_l` destination name,
identical nonquantized types and shapes, packed dim0, at least 64 KiB, internally
nonoverlapping monotonic rows, and disjoint source/destination byte spans.
Unsupported shapes, aliases, type conversions, ordinary copies, and small
copies retain the original implementation. Padding bytes are preserved.

## Rebuild and correctness

```sh
python3 generate.py
taskset -c 124-125 bash rebuild.sh > build.log 2>&1
bash build-test.sh
```

The harness compares production, candidate-off, and candidate-on byte for byte.
It checks actual loaded CPU/base library identities, exact flat-copy hit counts,
and complete destination allocations including padding and surrounding canaries.
Fourteen cases execute twice with fresh data. They include the real state shape,
1/2/3 planes, odd row tails, padded multidimensional views, F32/F16/I32, the exact
64 KiB threshold, and fallback cases for aliases, overlapping source rows,
nonpacked dim0, type conversion, ordinary names, and shape-changing copies.

`TEST_THREADS=1`, `2`, `3` (default), or `4` controls worker count. Wider worker
counts are for an explicitly coordinated idle window; this task limits local
CPU affinity to cores 124–127. A separate optional `bench` argument on test_copy
measures only the actual-shape copy graph, excluding input preparation, with
100 samples and ten discarded warmups. Microbenchmarks do not establish an
end-to-end token-rate improvement.

## Coordinated full-model gate

Use the validated launcher while preserving its full library/environment chain:

```sh
LIB_PREPEND=/home/user/InfoSystemic/AI-Server/serving/fleet-0912-ctx/glm-copy-0919/build
GGML_CPU_CPY_FLAT=1
GGML_CPU_CPY_FLAT_PROBE=1
```

First eight successful calls emit `CPY_FLAT` with source/destination addresses,
bytes, and thread count. All four devices must still write their state. The
process-local `ggml_cpu_cpy_flat_count()` function supplies standalone hit counts.

Verify mapped libraries/SHA/activation flags, actual engagement, greedy text
parity, cache/rollback behavior, the existing battery, multiple streams, and
short/context-length throughput before promotion. Any combined pool+copy library
must be separately derived and validated; this candidate contains no pool patch.

## Final standalone evidence

Library SHA256: `5a809c34142cdd76c32bf2fa9da73c5611d72447edee1d7bc0b85aeb01304527`.
`validation.json` records all 14 cases passing at 1, 2, 3, and 4 workers against
production and candidate-off. Each run compared 27,959,384 bytes including padding
and canaries; SHA256 `68b509ab68f08bb3045959c7edb7d55c59051068bcec1fd9ae15daabf205ae1c`.

`python3 microbench.py` ran production/candidate/production on the real 2 MiB
layout with input preparation outside the measured interval. Every arm produced
identical bytes. Medians in microseconds (100 samples, ten discarded warmups):

| Workers | Production | Candidate | Repeated production |
|---|---:|---:|---:|
| 1 | 155.70 | 154.57 | 154.97 |
| 3 | 157.45 | 79.31 | 156.82 |
| 4 | 156.97 | 76.19 | 157.24 |

That is about 2.0–2.1x for this local copy operation at 3–4 workers. It is not a
claim about model tok/s or the production 15-worker, four-device runtime. See
`microbench.json` for p10/p90, output hashes, affinity, and scope.
