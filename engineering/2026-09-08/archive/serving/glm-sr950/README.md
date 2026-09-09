# SR950 GLM serving profile

This profile is the production path for GLM-5.x on the four-socket Lenovo
SR950. Paseo and local Claude/Codex clients use the stable `glm-sr950` model
alias, so a model-version change does not require editing those clients.

## Live validation (2026-09-01)

The active Full GLM-5.3 process uses `build-dev2` and
`model.glm53-q4-fast.env`. Small fused NUMA reductions use one worker per
socket, selected by `GGML_CPU_NUMA_FUSED_REDUCE_SINGLE_MAX_ELEMENTS=32768`.
The matched development A/B improved 13.993 to 14.841 tok/s; the Full n2 suite
improved from the v2 reference of 8.588 to 8.882 tok/s with 74.13% acceptance.

The clean confirmation of the request-scoped agentic profile below produced
all three exact file edits correctly at **12.775 aggregate tok/s** with 88.46%
draft acceptance. Detailed rows are in
`/tmp/glm53-v5-n18p075-confirm.jsonl`. The `glm53-sr950.service` drop-in now
points to the canonical fast profile; the unit remains disabled unless an
operator explicitly enables it.

## Current production result (2026-08-29)

- Model: GLM-5.3 Full (non-Flash) UD-Q4_K_XL, 11 verified GGUF shards
- Engine: `llama.cpp-sr950-glm/build-sr950-glm`, fingerprint `b1-a302733`
- Context tier: 32K tokens, q8_0 K/V cache
- General default: request-inherited MTP `n_max=2`, `p_min=0`
- Agentic coding profile: request-scoped MTP `n_max=18`, `p_min=0.75`
- Draft: `GLM-5.3-MTP-HYBRID-Q4L-Q6H-OUTQ4.gguf`
- Final process RSS: about 470 GiB after four-way NUMA repack

The agentic profile was validated twice in clean windows on the same live
process. All six file-edit workloads passed their exact correctness checks:

| Run | Output tokens | Aggregate decode | Draft acceptance | Correct |
| --- | ---: | ---: | ---: | ---: |
| measured | 577 | 13.308 tok/s | 93.79% | 3/3 |
| confirmation | 565 | 12.546 tok/s | 92.35% | 3/3 |
| **combined** | **1,142** | **12.920 tok/s** | **93.07%** | **6/6** |

Send both request-scoped fields for the fast coding profile; the alias alone
does not change the safe general default:

```json
{
  "model": "glm-sr950-agentic",
  "speculative.n_max": 18,
  "speculative.p_min": 0.75
}
```

The two detailed validation files are
`/tmp/glm53-restored-n18-p075-clean.jsonl` and
`/tmp/glm53-restored-n18-p075-confirm.jsonl`. The Q4 shard hashes are pinned
in `GLM-5.3-UD-Q4_K_XL.sha256`.

## Previous GLM-5.2 production result (2026-08-26)

- Model: GLM-5.2 UD-Q2_K_XL, 753,864,139,008 parameters, 253,878,401,856 bytes
- Engine: `build-sr950-glm-pgo-q5-r8-20260826`
- Context tier: 256K tokens, q8_0 K/V cache
- Final process RSS: about 396.0 GiB; cgroup peak: about 414.9 GiB; GLM swap: zero
- General client alias: `glm-sr950`
- Replay-heavy agentic alias: `glm-sr950-agentic`

The production engine now combines four socket-local CPU backends, direct NUMA
all-reduce, GLM attention/MoE fusion, lossless compact IQ2_XS/IQ3_XXS VNNI
repacks, an exact Q5_K R8 expansion with fused dense gate/up execution, and a
real-shape PGO build. The repacks are allocated on the NUMA node that executes
their tensor shard. Each backend uses the 16 physical cores on its socket; SMT
was slower.

Measured tiers are intentionally reported separately. Context allocation and
speculative replay behavior both affect the final number:

| Profile | Deterministic raw decode | Agentic replay decode | Prompt decode |
| --- | ---: | ---: | ---: |
| Stock llama.cpp reference | 1.220 tok/s | not measured | not measured |
| PGO + compact IQ2/IQ3, 32K | 3.781 tok/s | 6.960 measured, 7.079 confirm | 10.169 tok/s |
| Prior PGO + compact IQ2/IQ3, 256K | 3.715 tok/s | 6.645 measured, 6.587 confirm | 10.125 tok/s |
| **Production + fused Q5 R8, 256K** | **3.995 tok/s** | **6.314 cold / 7.495 warm** | **10.738 tok/s** |
| Same engine, 1M viability test | not re-run raw | 6.848 measured | about 10.6 tok/s |

The 32K speed tier is 3.10x stock before speculation. The 32K PGO build
improved deterministic raw decode by
3.85-3.86% over the matched non-PGO compact-repack build. Replay is noisier
because n-gram draft opportunities differ between otherwise identical runs;
the confirmed 32K result is 8.05% above the older 6.551 tok/s production replay
reference.

A clean, single-tenant 256K no-draft rerun now establishes 3.7150 tok/s raw
decode and 10.1251 tok/s prompt ingest across the seeded eight-workload suite.
The earlier 3.5969/9.9229 run remains discarded because an uncoordinated
DeepSeek-V4 process had been resident throughout it.

Compact repack correctness was tested directly against the original target
forward path. Maximum absolute error was 1.56e-7 for IQ2_XS and 2.01e-7 for
IQ3_XXS. These are lossless weight re-encodings; the tiny output difference is
floating-point reduction order, not requantization.

The first Q5_K experiment accelerated individual matrices but bypassed the
server's dense gate/up fusion, so its speculative full-service arm regressed.
A fresh inference-only profile exposed that mistake: Q5_K was 28.63% of
exclusive samples, and most of its caller-attributed time was in
`ggml_cpu_try_fuse_ops`. The repack traits now implement the fused
`MUL_MAT_SWIGLU` path directly, quantize the shared activation once, execute
both repacked projections, and apply SwiGLU in the existing NUMA tile.

That corrected implementation is enabled with `GGML_CPU_Q5_K_REPACK=1`.
Against the same 256K PGO control it improved raw decode from 3.7150 to 3.9949
tok/s (+7.53%), raw prompt ingest from 10.1251 to 10.7384 tok/s (+6.06%), and
the seeded general speculative suite from 4.4560 to 5.2699 tok/s (+18.27%).
Direct fused-graph comparison passed at batch 1 and batch 4 with maximum
absolute error no larger than 5.50e-4 and normalized MSE no larger than
1.72e-13. The weights are not requantized; the small output difference is
floating-point reduction order.

Speculation has two measured optima on the same loaded model. The stable
`glm-sr950` alias keeps `n_max=64`, `p_min=0.8`, and `n_match=24`; it is the
5.2699 tok/s general-workload winner. `glm-sr950-agentic` defaults only
`p_min` to 0.9. On the final production process it measured 6.3139 tok/s on
the first post-restart replay and 7.4954 tok/s on the immediately repeated
warm replay, with all three deterministic checks correct in both runs. An
earlier warmed candidate measured 7.4407 tok/s. The range is expected because
`ngram-mod` benefits from reusable server history; 7.5 tok/s is not advertised
as cold-start throughput. Explicit dotted or nested request values override
the alias default. A single global threshold would make one workload class
slower, so Paseo, Claude Code, and Codex expose both aliases without loading a
second model copy.

The 1M context remains viable but consumed about 563.5 GiB process RSS and left
too little operational headroom for a default. The selected 256K tier uses
about 396 GiB process RSS and 415 GiB of cgroup memory, leaving substantially
more room for Paseo, Qwen, gateways, page cache, and a clean recovery path. Use
32K only when maximum short-context speed is more important than agent context
length.

No 12 tok/s GLM-5.2 novel-workload result has been measured. Do not derive or
advertise one from memory-bandwidth arithmetic. Final production measured
6.3139 tok/s on the first replay after restart and 7.4954 tok/s on an immediate
warm repeat through `glm-sr950-agentic`; the measured general profile is
5.2699 tok/s.

## Historical baseline and profiling

Before compact repack and PGO, the safe n-gram plus adaptive-MTP profile
measured 4.262 tok/s on the general workload suite and 5.663 tok/s on the
three-case agentic replay suite. Those results remain useful as historical A/B
baselines, but they no longer describe the production binary.

## Inference-only CPU profile

An August 25 `gprofng` experiment paused sampling during the roughly 7.5-minute
model load, then sampled only the same eight 96-token production workloads used
for service A/B tests. Exclusive CPU samples were concentrated in:

| Function or runtime region | Exclusive CPU samples |
| --- | ---: |
| `ggml_vec_dot_q5_K_q8_K` | 27.95% |
| `ggml_vec_dot_iq2_xs_q8_K` | 23.23% |
| libgomp worker wait/synchronization | 18.93% |
| `ggml_vec_dot_iq3_xxs_q8_K` | 11.51% |
| q8 tinyBLAS `gemm4xN<2>` | 6.19% |
| `ggml_vec_dot_q6_K_q8_K` | 3.26% |
| `ggml_vec_dot_q4_K_q8_K` | 2.78% |

Attention accounted for only about 0.54%, while graph and direct-NUMA metadata
overhead were small. Caller attribution showed most Q5 and IQ2 time inside the
fused dense and routed gate/up paths. This makes quantized dot products and
short-region worker synchronization the next optimization targets; NUMA
placement, storage, attention, and the base operating system are not the
remaining decode bottleneck.

The sampled run reported only 3.002 aggregate decode tokens/second because the
profiler imposed substantial overhead. That number must not be compared with
unprofiled throughput; only the sample proportions above are used as evidence.
The live process had about 458 GiB of anonymous huge pages, balanced socket
placement, and zero service swap during the experiment.

Passive OpenMP waiting was tested because it improved isolated fused-kernel
microbenchmarks. It failed the matched whole-service test and was reverted:

| OpenMP policy | Aggregate decode tok/s | Prompt tok/s | MTP acceptance |
| --- | ---: | ---: | ---: |
| libgomp default | **3.9552** | **9.7781** | 75.17% |
| `OMP_WAIT_POLICY=PASSIVE` | 3.5368 | 9.2999 | 69.51% |

Passive waiting reduced aggregate decode throughput by 10.6% and prompt
throughput by 4.9%, despite making the isolated fused Q5 kernel about 10%
faster. Wake latency across the complete graph outweighed the hot-loop gain.
Production intentionally leaves both `OMP_WAIT_POLICY` and `GOMP_SPINCOUNT`
unset.

A matched `GGML_OPENMP=OFF` build was also tested on the persistent
`CPU-NUMA2` backend, which mirrors one production socket. Its native worker
pool improved the dense fused Q5 projection by about 9.3%, but routed IQ2 was
about 2% slower. More importantly, the IQ3 routed projection regressed from
1.152 ms to 60.886 ms with the paired-row gate. Disabling that gate recovered
native IQ3 to 1.486 ms, still about 29% slower than OpenMP. The native build was
therefore rejected; production remains an OpenMP build.

The kernel harness now accepts `GGML_TEST_N_THREADS`. Always set it to 16 and
benchmark a persistent `CPU-NUMA<N>` device. The upstream default is all 128
logical CPUs, which oversubscribes a 16-core socket and produces misleading
results even if the process itself is pinned with `numactl`.

Use the checked-in wrapper for the production-representative fused kernels:

    ./benchmark-kernels.sh

It sources `model.env`, selects `CPU-NUMA2`, sets the harness to the same 16
physical cores used by each production backend, and measures the fused Q5,
routed IQ2, and routed IQ3 shapes that dominate the inference-only profile.

For a low-overhead whole-graph decode profile, leave the arm file absent during
load and run:

    ./capture-decode-op-profile.sh

The helper disables speculation for the request and creates the arm file only
after the first streamed event, so prompt setup is excluded. It removes the
trigger on exit, allowing a later run to re-arm without restarting the model.
Summarize the matching journal interval with:

    journalctl --user -u glm-sr950.service --since 'YYYY-MM-DD HH:MM:SS' --no-pager \
      | grep CPU_OP_PROFILE | ./summarize-op-profile.py

The summary ranks operations and normalized tensor names, and reports graph
totals per socket so a kernel bottleneck is not confused with load imbalance.

## Speculative-decoding production tuning

The first production profile used GLM's built-in MTP head with one drafted
token. That established a conservative baseline before longer confidence-gated
drafts and safe n-gram replay were tested.

Repeated 128-token API trials used the same 33-token prompt, disabled prompt
cache reuse, and sampled through the same loaded server:

| Request draft max | Mean decode tok/s | Range | Draft acceptance |
| ---: | ---: | ---: | ---: |
| 0 (MTP loaded, drafting disabled) | 3.450 | 3.436-3.466 | n/a |
| 1 | **3.948** | 3.828-4.089 | 74.7% |
| 2 | 3.292 | 3.008-3.537 | 42.4% |

A separate three-token trial reached only 3.580 tokens/second with 45.1%
acceptance. A single favorable two-token run reached 4.035 tokens/second, but
the repeated matrix exposed its poor tail and lower mean. One token is the
measured production choice for GLM-5.2 on this SR950.

The final whole-service A/B used eight 96-token coding, systems, database,
security, algorithm, and technical-writing workloads (768 output tokens per
configuration):

| Service configuration | Aggregate decode tok/s | Mean per-workload tok/s | Prompt tok/s |
| --- | ---: | ---: | ---: |
| No MTP | 3.633 | 3.595 | 10.093 |
| One-token MTP | **3.956** | **3.937** | 9.818 |

One-token MTP improved aggregate decode throughput by 8.9% and won seven of
eight workloads. It lost the one low-acceptance workload (44% acceptance), so
individual requests can still have a slower tail; across the suite its overall
draft acceptance was 72.4%. Prompt processing was 2.7% slower, a worthwhile
trade for sustained interactive generation on this profile.

The current profile combines `ngram-mod` with the built-in MTP head. It uses a
safe 24-token n-gram match, emits n-gram drafts only in the 48-64 token range,
and allows MTP to draft up to 64 tokens while stopping below 0.8 confidence.
The matched eight-workload rerun produced:

| Production profile | Aggregate decode tok/s | Prompt tok/s | Draft acceptance |
| --- | ---: | ---: | ---: |
| One-token MTP (`n_max=1`, `p_min=0`) | 3.951 | 9.701 | 75.46% |
| Safe n-gram + adaptive MTP (`n_max=64`, `p_min=0.8`) | **4.262** | **9.957** | **89.56%** |

That is a 7.88% decode improvement and a 2.64% prompt improvement on the same
prompts, seeds, sampling settings, and 768 output tokens. The corrected,
deterministic three-case agentic file-rewrite suite measured 3.207 tokens/second
with drafting disabled, 4.380 with one-token MTP, and 5.663 with the composite
profile. The composite profile is therefore 76.6% faster than no drafting and
29.3% faster than one-token MTP on 562 correctness-checked output tokens.

The fork supports request-scoped `speculative.n_max` and
`speculative.p_min`. Draft length is strictly bounded by the allocation made at
server startup, and confidence is bounded to `[0, 1]`. Both llama.cpp's dotted
keys and the natural nested `speculative` object are accepted. This lets a
loaded 254 GB model run controlled comparisons without resizing speculative
buffers or restarting.

The final post-inference production snapshot had zero GLM service swap, about
396.0 GiB process RSS, about 414.9 GiB cgroup memory, and 274 GiB host memory
still available. `numastat` showed 405,451 MB total distributed within 578 MB
across all four sockets, confirming that tensor placement remained balanced to
well under one percent.

Re-run the matrix after loading GLM-5.3:

    ./benchmark-mtp.sh 0 1
    ./benchmark-workloads.sh glm-5.3

`benchmark-workloads.sh` defaults to the production sampling temperature and
startup MTP limit. For a deterministic no-draft control against the same loaded
service, use:

    GLM_BENCH_TEMPERATURE=0 GLM_BENCH_SPEC_N_MAX=0 \
        ./benchmark-workloads.sh glm-5.3-greedy-no-mtp

`GLM_BENCH_TOKENS` changes the generated length (minimum two tokens), while
`GLM_BENCH_SPEC_P_MIN` overrides the confidence threshold, and
`GLM_WORKLOAD_OUT` preserves the per-workload JSONL instead of using a
temporary file. `benchmark-replay.sh` supports the same `SPEC_N_MAX` and
`SPEC_P_MIN` controls. Do not compare one-token runs: llama.cpp reports no
steady-state decode interval until at least two generated tokens are present.

All checked-in benchmark scripts take the nonblocking
`/tmp/glm-sr950-benchmark.lock`, so two harness runs cannot silently overlap.
They also call `assert-benchmark-window.sh` before the suite and before every
request. The guard scans `/proc` and refuses exit status 75 if an unapproved
`llama-server`, `llama-cli`, or `llama-bench` process is present. Its default
allowlist is the production GLM/Qwen service ports (18091, 18081, 5811, 5804).
Set `GLM_BENCH_ALLOWED_LLAMA_PORTS` only for an announced, controlled window.

The guard cannot prevent another process from being launched in the instant
after a check. Preserve detailed JSONL and check process timestamps whenever a
result is surprising. A 2026-08-26 256K raw suite was correctly retracted after
timestamps proved that an ad hoc model had overlapped the complete run.

To test a larger request value, first raise `GLM_SPEC_DRAFT_N_MAX` in
`model.env` and restart once; requests above the startup value are rejected.

## n-gram speculative decoding (added 2026-08-25)

`launch-glm-sr950.sh` previously rejected every `GLM_SPEC_TYPE` except `none`
and `draft-mtp`. The engine registers eleven types, including `ngram-mod`,
`ngram-simple`, `ngram-map-k`, `ngram-map-k4v`, and `ngram-cache`; `--spec-type`
takes a comma-separated list. The launcher now parses that list and emits the
per-type arguments, still rejecting unknown components.

### The clamp that made ngram-mod a no-op

Enabling `ngram-mod` alone changed nothing: 4.187 -> 4.274 tok/s on replay, with
draft counts identical to MTP-only (620 vs 622). The cause is in
`tools/server/server-context.cpp:458`:

    n_draft_max = std::min(n_draft_max, std::max(0, task->params.speculative.draft.n_max));

`draft.n_max` is the **MTP** depth, set by `--spec-draft-n-max`. The runtime
clamp applies it to *every* implementation, so with `GLM_SPEC_DRAFT_N_MAX=1`
each ngram-mod draft was truncated from 64 tokens to 1 in
`common/speculative.cpp` ("truncating draft to %d tokens"). The startup
allocator is already correct — `common_speculative_n_max()` takes the max across
all registered types (64 here) — so the buffers existed and went unused. This
looks like an upstream defect: the clamp should use `common_speculative_n_max()`,
not `draft.n_max`.

Raising `GLM_SPEC_DRAFT_N_MAX` alone is unsafe, because `--spec-draft-p-min 0.0`
disables MTP's confidence gate outright (`common/speculative.cpp:1220`,
`conf = params.p_min > 0.0f ? ... : nullptr`), leaving MTP to draft its full
depth every step. Verifying a 64-token MTP draft costs roughly 64 batched-token
evaluations for about three accepted tokens. `p_min > 0` restores the early stop
and lets MTP size its own drafts while ngram-mod keeps the full 64.

### Measured

Production config: `GLM_SPEC_TYPE=ngram-mod,draft-mtp`, `GLM_SPEC_DRAFT_N_MAX=64`,
`GLM_SPEC_DRAFT_P_MIN=0.8`, ngram geometry at engine defaults (64/48/24).

| Suite | Before (MTP n_max=1) | After | Change |
| --- | ---: | ---: | ---: |
| `benchmark-workloads.sh` (novel, 8 x 96 tok) | 3.951 | **4.262** | +7.9% |
| `benchmark-replay.sh` (3 deterministic file edits, 562 tok) | 4.380 | **5.663** | +29.3% |

Draft acceptance rose from 0.755 to 0.896 on the novel suite. On the replay
suite, one-token MTP accepted 280/282 drafts (99.29%), while the composite
profile accepted 538/797 (67.50%) and still won because accepted n-gram runs
can advance many tokens per target-model evaluation. Both suites improved, so
this supersedes the one-token MTP result above.

`benchmark-replay.sh` was added because the novel-prose suite cannot show this
effect — ngram-mod drafts from context repetition, which is the agentic
file-re-emission pattern that `claude-glm` and `codex-glm` actually produce.

Geometry note: `n_match=8` (the 64/8 setting from the previous host) measured
slightly faster on replay, 5.499 vs 5.198, but the engine warns
`ngram_mod n_match=8 is too small - poor quality is possible`. Defaults are kept.

### Determinism

The controlled replay checks use `temperature=0`, a fixed seed, and
`reasoning_effort=none`; no-draft and composite requests produced the same
requested file content and passed exact edit checks. With reasoning enabled,
identical requests can still produce different reasoning lengths because the
four-socket all-reduce (`GGML_CPU_NUMA_DIRECT_ALLREDUCE=1`) can vary
floating-point reduction order with thread scheduling. This predates the
change. Benchmark by aggregate tokens/second and treat single-sample sweeps as
noise.

Rollback: `launch-glm-sr950.sh.bak-20260825-prengram`, `model.env.bak-20260825-prengram`.

## Claude Code and Codex compatibility

The fork converts OpenAI Responses `custom` tools to the model's JSON function
representation and converts generated calls back to native
`custom_tool_call` events. This is required for Codex's freeform `apply_patch`
tool. Both the llama.cpp chat test suite and a real Codex streaming round trip
were verified: GLM called `apply_patch`, Codex changed an isolated probe file,
and the agent completed the turn successfully.

Current Codex `namespace` and `web_search` tool definitions are intentionally
skipped because llama.cpp has no equivalent local execution facility for them.
Core shell/function tools and freeform `apply_patch` work. A cold Codex turn can
still spend several minutes ingesting its multi-thousand-token tool prompt;
subsequent turns reuse the server prefix cache and are much faster.

## GLM-5.3 release and activation

`check-glm53-release.sh` checks likely official Z.ai and Unsloth Hugging Face
repo names, searches both publishers for newly named GLM-5.3 models, and tracks
revision changes to their existing GLM-5 repositories. It does not download
anything. The active `release-watch.timer` writes the latest result to:

    ~/.local/state/glm-sr950/glm53-release.json

Once the desired GGUF is fully staged, activate it transactionally:

    ./activate-glm-model.sh /path/to/GLM-5.3-...-00001-of-000NN.gguf 5.3

The script verifies every shard and a minimum total size, saves `model.env`,
restarts only `glm-sr950.service`, waits for health, verifies the stable model
alias, and sends a completion request. Any failure restores GLM-5.2 and
restarts it automatically. It never deletes the old model.

Before loading hundreds of gigabytes, inspect every shard header and calculate
the new quant's compact-repack coverage and projected weight footprint:

    ./inspect-glm-gguf.py /path/to/GLM-5.3-...-00001-of-000NN.gguf

The complete compatibility, benchmark, activation, acceptance, and rollback
procedure is in `GLM53-RUNBOOK.md`.

Do not assume compatibility based only on the version name. Before activation,
inspect GLM-5.3 metadata and run the existing correctness tests. If its tensor
shapes or operators differ from GLM-5.2, re-profile the fork before production.

## Storage constraint

The current root and `/models` SSD filesystems do not have enough combined free
space to stage another roughly 254 GB quant while retaining GLM-5.2. A safe
side-by-side cutover therefore requires at least 512 GB of additional SSD
capacity; 1 TB is preferred. Do not delete the working GLM-5.2 shards until the
GLM-5.3 shard set has passed activation, correctness, performance, and cold
restart tests.
