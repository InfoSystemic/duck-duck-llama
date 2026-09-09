# GLM-5.3 SR950 release and cutover runbook

This is the release-day procedure for replacing GLM-5.2 without losing the
working service or assuming that a similarly named model has identical tensor
geometry. Commands are run from this directory unless noted otherwise.

## Current state and hard blocker

As of 2026-08-26 04:45 MDT, no public GLM-5.3 repository was found under the
watched official Z.ai or Unsloth names. The hourly watcher is active:

    systemctl --user status glm53-release-watch.timer
    ./check-glm53-release.sh
    jq . ~/.local/state/glm-sr950/glm53-release.json

Storage must be expanded before downloading another large GLM quant. Current
free space is about 39 GiB on the root NVMe and 24 GiB on the Samsung SATA SSD
mounted at `/models`. A side-by-side cutover needs at least 512 GB of additional
SSD space; 1 TB is preferred so the old model, new model, temporary download
files, and benchmark artifacts can coexist. Do not delete GLM-5.2 first.

## 1. Select and stage the artifact

Prefer an official or Unsloth GGUF whose quality/size tier is comparable to the
current UD-Q2_K_XL. Do not select by quant filename alone. Download every shard
to one directory on SSD, retain publisher checksums or manifest metadata, and
make the files read-only after verification.

Confirm that the numbered set is complete and inspect its headers without
loading tensor data:

    ./inspect-glm-gguf.py \
      /models/GLM-5.3-.../GLM-5.3-...-00001-of-000NN.gguf

Save a machine-readable record too:

    ./inspect-glm-gguf.py --json \
      /models/GLM-5.3-.../GLM-5.3-...-00001-of-000NN.gguf \
      > /models/GLM-5.3-.../sr950-preflight.json

The installed GLM-5.2 control reports:

- architecture `glm-dsa`
- 753,864,139,008 parameters
- 94.43% of parameters in IQ2_XS/IQ3_XXS
- 236.43 GiB GGUF tensor payload
- 322.73 GiB projected compact IQ2/IQ3 weights
- 327.14 GiB projected weights with all currently enabled repacks
- routed layouts dominated by `6144x2048x256` and `2048x6144x256`

Stop and update the fork before production if any of these change:

- `general.architecture` is not `glm-dsa`
- required tensor names or operators are unknown to the loader
- MTP metadata describes more than the one block currently asserted by the
  GLM-DSA draft implementation
- routed expert dimensions differ from the PGO corpus
- projected compact weights plus context/cache/host reserve cannot fit in
  755 GiB without swap

Different IQ2/IQ3 proportions are allowed, but they change both resident RAM
and the value of the compact kernels. A quant with neither type receives no
benefit from the IQ2/IQ3 gates and must be benchmarked on its actual formats.

## 2. Preserve a GLM-5.2 control

Before stopping production, record the exact binary, config, health, memory,
and matched benchmarks:

    systemctl --user show glm-sr950.service \
      -p MainPID -p MemoryCurrent -p MemorySwapCurrent
    GLM_BENCH_TEMPERATURE=0 GLM_BENCH_SPEC_N_MAX=0 \
      GLM_WORKLOAD_OUT=/tmp/glm52-precutover-raw.jsonl \
      ./benchmark-workloads.sh glm52-precutover-raw
    GLM_REPLAY_OUT=/tmp/glm52-precutover-replay.jsonl \
      ./benchmark-replay.sh glm52-precutover-replay

Current clean GLM-5.2 controls at 256K are 3.9949 tok/s raw, 5.2699 tok/s on
the general speculative suite, and 6.3139 cold / 7.4954 warm on the agentic
replay suite. The 32K speed-tier controls are 3.7814 raw and 7.0787 confirmed
replay. Compare only matched context, sampling, token count, workload,
single-tenant host state, and n-gram cache history.

## 3. Compatibility load before PGO

The existing PGO profile is trained on GLM-5.2's tensor-split shape and must
not be trusted blindly for changed geometry. If GLM-5.3 dimensions differ,
first build the same fork without profile-use, then validate compact repack and
collect a new profile on GLM-5.3's real shapes at 1/2/4/8/16 rows.

Required runtime gates for a compatible IQ2/IQ3 model are in `model.env`:

    GGML_CPU_NUMA_REPACK=1
    GGML_CPU_IQ2_XS_REPACK=1
    GGML_CPU_IQ3_XXS_REPACK=1
    GGML_CPU_Q5_K_REPACK=1
    GGML_CPU_REPACK_LOAD_THREADS=8

The validation order is:

1. focused `test-backend-ops` comparisons against the original tensor types;
2. full target-forward and fused gate/up correctness comparisons with repack off/on;
3. full model load at 32K with no speculation;
4. deterministic workload and replay suites;
5. only then collect and apply a new PGO profile if shapes changed.

Reject a PGO build if any 1/2/4/8/16-row case regresses. The accepted 5.2
profile used `-fprofile-update=atomic`; a small-shape corpus was rejected after
it improved decode but regressed higher-batch IQ3.

## 4. Transactional activation

Once the full shard set and engine pass the preflight, use the checked-in
activator:

    ./activate-glm-model.sh \
      /models/GLM-5.3-.../GLM-5.3-...-00001-of-000NN.gguf 5.3

It validates shard completeness and total size, saves `model.env`, changes only
the model path and aliases, restarts `glm-sr950.service`, waits for health,
checks both stable aliases, and sends default plus OpenAI/Anthropic agentic API
smoke requests. Any failure restores the previous config and restarts GLM-5.2.
The old model is never deleted.

For a cold source on `/models`, `../prefetch-model.sh FIRST_SHARD --wait` can
reduce load time, but it deliberately skips when available RAM is smaller than
the file set. Never force it while the 400+ GiB GLM process is resident. Use it
only in an announced single-tenant maintenance window after stopping GLM, and
retain enough memory headroom for the repacked destination.

## 5. Production acceptance gates

After activation, require all of the following before calling the cutover
successful:

- `/health` and `/v1/models` return normally under the stable alias
- `glm-sr950-agentic` applies its default on both OpenAI and Anthropic endpoints
- completion smoke test returns content
- `VmSwap` for the GLM PID is zero
- anonymous placement is within 5% across nodes 0-3
- deterministic raw suite completes all eight workloads
- replay suite passes all three exact edit checks
- Claude/Codex tool-call compatibility smoke passes
- Paseo and both Qwen gateway health checks remain healthy

Useful checks:

    pid=$(systemctl --user show glm-sr950.service -p MainPID --value)
    awk '/^(VmRSS|RssAnon|RssFile|VmSwap):/{print}' /proc/$pid/status
    numastat -p "$pid"
    curl -fsS http://127.0.0.1:18091/health
    curl -fsS http://127.0.0.1:18091/v1/models | jq .

Begin at 32K to establish the speed ceiling, then test 256K. Keep 256K only if
the context is operationally worth its measured throughput and RAM cost. Do
not start at 1M: on GLM-5.2 it used about 563.5 GiB process RSS and reduced
short-context replay throughput.

## 6. Speculation must be re-derived

Start with drafting disabled. Then test one-token MTP and the current composite
profiles. GLM-5.2 has separate general and replay optima:

    GLM_SPEC_TYPE=ngram-mod,draft-mtp
    GLM_SPEC_DRAFT_N_MAX=64
    GLM_SPEC_DRAFT_P_MIN=0.8
    GLM_SPEC_NGRAM_MOD_N_MATCH=24

    # Alias-scoped replay default; explicit request values still win.
    GLM_SPEC_ALIAS_P_MIN=glm-sr950-agentic:0.9

With the fused-Q5 engine, the general profile measured 5.2699 tok/s. On the
three-case replay suite, request-scoped `n_max=64,p_min=0.9` measured 7.4407
tok/s on a warmed candidate. The final production process measured 6.3139
tok/s on its first replay after restart and 7.4954 tok/s on an immediate warm
repeat; every check passed. `n_max=3,p_min=0` reached 7.0095, while 1/0 and
64/0.5 were slower. The 0.9 threshold is not a universal improvement: it
reduced the general suite to 4.3894 tok/s. Keep the stable alias on the general
default and use `glm-sr950-agentic` only for replay-heavy coding work. Record
cold and warm runs separately because n-gram history changes throughput.

Re-run both workload classes on GLM-5.3. Alias defaults are conveniences, not
evidence that 5.2's threshold transfers. Optimize aggregate tok/s and exact
correctness, not acceptance percentage.

The 5.2 production engine also fuses compact-IQ3 routed-down projection with
router weighting (`GGML_CPU_MOE_DOWN_WEIGHTED_SUM_FUSION=1`, tile 192).  Its
full 256-expert harness is bit-exact and its conservative final raw gain is
1.51%.  Keep it enabled for 5.3 only if the quant still uses IQ3_XXS down
experts and the full-topology validator passes.  Never copy the diagnostic
`GGML_CPU_MOE_DOWN_WEIGHTED_SUM_VALIDATE` flag into production; it is not safe
under the current four-backend warmup scheduler.

## 7. Rollback and retention

If a post-activation gate fails, restore the timestamped
`model.env.before-glm-5.3-*` file and restart only `glm-sr950.service`. Keep the
5.2 model and its known-good PGO binary until 5.3 has survived correctness,
Paseo/CLI use, a cold restart, and at least one day of service. Deleting the old
model is a separate destructive action and is not part of activation.
