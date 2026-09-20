# Deployed pool + state-copy kernel, 2026-09-19

This directory combines the immutable `glm-pool-0919` and `glm-copy-0919`
candidates. Both switches default off and remain independent. The isolated build preserved both individual candidates and shared engine source.
The user subsequently approved production deployment of this combined library.

`generate.py` verifies the frozen parent library hashes and source hashes from
their manifests, then combines their exact implementations. The only delta over
the pool candidate is the independently gated all-device flat state copy. See
`pool-parent-delta.patch` and `source-provenance.json`.

Parent CPU-library SHA256 values:

- Pool: `f504f67f5a0184cce86378ba56fe04e937f62e971cb1f51bd0e4678446996ee0`
- Copy: `5a809c34142cdd76c32bf2fa9da73c5611d72447edee1d7bc0b85aeb01304527`

Build and test:

```sh
python3 generate.py
taskset -c 124-125 bash rebuild.sh > build.log 2>&1
bash build-test.sh
```

`test.py` runs the 21-case pool harness and 14-case copy harness in five arms:
actual production, both off, pool only, copy only, both on. It compares all output
bytes and asserts expected path counts and loaded CPU/base library identities.
`TEST_THREADS` selects the worker count (default 3). Validation artifacts are
`validation-threads-N.json`; full-model validation remains a separate gate.

Settings for a coordinated server arm:

```sh
LIB_PREPEND=/home/user/InfoSystemic/AI-Server/serving/fleet-0912-ctx/glm-pool-copy-0919/build
GGML_CPU_GLM_POOL_FUSION=1
GGML_CPU_CPY_FLAT=1
```

Opt-in, bounded engagement logs use `GGML_CPU_GLM_POOL_PROBE=1` and
`GGML_CPU_CPY_FLAT_PROBE=1`. The original address-only CPY diagnostic remains
available as `GGML_CPU_CPY_ADDRESS_PROBE=1`.

Preserve the complete validated runtime environment and downstream library order.
Test copy alone first for attribution; then verify the combined library's greedy
outputs, cache/rollback, long context, concurrency, and actual Codex/Paseo behavior
before any promotion. Standalone correctness and component gains do not establish
combined end-to-end throughput.

## Locked validation

Final combined library SHA256: `4793379e6894a9286168f79c4f323985388ec0b0ac65e54013dcd2957488af1f`.
At three workers, both standalone suites passed in all five arms (actual
production plus all four independent-switch states). Pool output SHA256 remains
`546fda618461f6376e4b1f3170cd923dd8214083a0736c1276714cd931b0f9fe`;
copy output SHA256 remains
`68b509ab68f08bb3045959c7edb7d55c59051068bcec1fd9ae15daabf205ae1c`.
Expected path counts and exact loaded library identities were verified in every
arm. This is 21 pool cases and 14 copy cases, each executed twice per arm.
This exact combined library was measured and deployed to `../glm-cpu-fast-0919`
after explicit user approval. The live service enables both flags. It passed
three native fixed-output prompts, actual Codex/Paseo explanation and coding
checks, and all nine corresponding stateful regression outputs. The deployment
window measured 13.987 tok/s for the Codex explanation and 15.767 tok/s for the
coding test. These are workload-specific observations, not a universal rate.

The cached/fresh discrepancy already present in the original kernel remains
unresolved; stateful regression parity does not mean intrinsic consistency passed.
The earlier window did not establish the new long-context inner-loop gain.
See [the full experiment log](/home/user/sr950-strategy/GLM53-FLASH-CODEX-20260919.md)
and [the separate wider-loop candidate](../glm-kpool-wide-0919/README.md).
The individual frozen candidates remain available for attribution.

## Subsequent wider-loop deployment

The combined implementation above remains the frozen parent and rollback library.
The normal endpoint now uses the byte-validated eight-channel pool inner loop
from [glm-kpool-wide-0919](../glm-kpool-wide-0919/README.md), CPU SHA256
`3f957a341321b940d93be53c250cdd068825093faa2d9efda142ebe56427b1b3`.
Its measured same-process Codex gains are 3.8% at 3,998 input tokens and 13.4% at
29,930 tokens. Both original flags remain enabled, plus `GGML_CPU_GLM_POOL_WIDE=1`.
