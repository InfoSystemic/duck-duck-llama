# GLM-5.3-Flash Codex/Paseo tuning — 2026-09-19

Work in progress. Goal remains faster, reliable GLM-5.3-Flash in Codex CLI inside
Paseo on this four-socket SR950, with coding/tool quality retained.

## Verified starting state

- Model: GLM-5.3-Flash UD-Q4_K_XL, MTP2, CPU-NUMA0..3 tensor parallel,
  15 threads/socket, native 1M server context, 2 unified slots.
- Live PID at capture: 2289587, port 18131. **The enabled production systemd unit
  is inactive**; this PID was manually restored by yesterday's session.
- Actual libraries and activation flags are captured in
  `codex-bench-0919/production-runtime.json`. In particular `glm-fix` and
  `unary-lib` override the private/engine libraries. Rebuilding from the engine
  directory alone does not reproduce this deployment.
- Three raw-completion prompts, 256 tokens, seed 42, temperature 0:
  12.65 / 14.62 / 14.17 tok/s. Raw response artifacts are
  `../InfoSystemic/AI-Server/serving/fleet-0912-ctx/results/bench3-goal0919-baseline-*.json`.

## Actual Codex-in-Paseo baseline

Provider `codex-local/glm-5.3-flash`, isolated workspace
`codex-bench-0919`, auto permissions, 200-word LRU explanation, no tools requested.
Expected final marker verified. No concurrent inference during measurement.

| Measure | Baseline |
|---|---:|
| Input tokens | 8,752 |
| Input cached | 0 |
| Prefill | 143.316 s / 61.07 tok/s |
| Generated tokens (including reasoning) | 312 |
| Decode | 32.374 s / 9.637 tok/s |
| Time to first token (Codex metric) | 149.616 s |
| Full turn (Codex metric) | 176.176 s |

Counters and rollout evidence: `codex-bench-0919/paseo-baseline-summary.json`.
This is a single representative observation, not a statistically established mean.
Decode throughput and full-turn latency measure different parts of the experience.

## Applied route correction

Implicit GLM Flash aliases now resolve only to production port 18131. Previously,
a healthy experimental server at 18094 took priority. Explicit endpoint overrides
and all other model routes remain available. The exact change passed shell syntax
checks, 24 Flash cases and 36 unchanged other-model cases, without live requests.

Patch, original backup, candidate and applied manifest are in `codex-bench-0919`.
An initial live-edit approval review was rejected; after preparing the exact patch,
rollback and regression evidence, re-review allowed application. No service restart.

## Current findings and work

- Paseo sends lowercase `glm-5.3-flash` through app-server RPC even though the shell
  wrapper starts with uppercase `GLM-5.3-Flash`. The custom catalog contains only
  the uppercase slug. Actual baseline session recorded the large stock Codex base
  instructions rather than the intended compact GLM instructions. Alias matching
  and an offline request-capture regression are being checked before changing it.
- Indexer kernel candidate is being developed privately in
  `../InfoSystemic/AI-Server/serving/fleet-0912-ctx/glm-pool-0919`.
  Production already fuses SOFT_MAX/MUL/SUM_ROWS; the remaining opportunity is
  the repeated GET_ROWS/CONT/ADD preceding that fusion. Default-off until validated.
- Yesterday's single-writer CPY experiment changed output almost immediately;
  its apparent speedup is invalid. Do not deploy it.

Still required: validate the catalog correction with the actual app-server RPC
shape; measure its cold and warm performance; exercise coding/tool use; complete
kernel parity and performance gates before any promotion; restore normal service
supervision without overlapping model loads. Do not treat this log as completion.

## Catalog correction applied; pool candidate measured

The lowercase catalog alias and intended `low` default are now applied. Original
catalog retained as `codex-bench-0919/glm-5.3-flash.catalog.before.json`; exact
request-capture and live benchmark results are saved alongside it. Fresh Codex
app-server processes are required to load the updated catalog. Existing sessions
may retain their original system instructions.

The reproducible app-server probe runs **inside a Paseo terminal**, through the
installed `codex-smeagol` wrapper, with the lowercase thread/turn RPC model setting.
It does not reproduce every injected item of Paseo's managed-agent RPC session:
its original-catalog request has 7,739 input tokens versus 8,752 in the initial
managed-agent observation. The paired comparison below uses the same probe in
all arms. A temporary loopback forwarding adapter forces `cache_prompt=false`;
all rows recorded zero cached tokens. Temperature and generated text vary, so
these are individual observations rather than fixed-output causal estimates.

| Configuration | Input | Decode tok/s | Prefill s | Whole turn s |
|---|---:|---:|---:|---:|
| Original catalog + production | 7,739 | 9.585 | 122.982 | 158.319 |
| Corrected catalog + production | 3,998 | 12.005 | 57.441 | 79.452 |
| Original catalog + pool candidate | 7,739 | 11.589 | 127.897 | 154.901 |
| Corrected catalog + pool candidate | 3,998 | 13.136 | 58.190 | 79.684 |

All completed successfully with the requested marker. Catalog-only is currently
applied; the pool library remains experimental. The pool candidate passed 21
standalone cases, two executions each, at 1/2/3/4 threads, with exact bits vs the
actual production library. Full-model short-prompt greedy text parity passed 3/3;
short speeds were 12.29/14.29/13.77, slightly below the earlier production values.
No general 18+ tok/s claim is supported yet. The user's explicit target is now
**18+ tok/s inside Codex/Paseo**, retaining useful coding behavior.

A live pointer probe also refutes the old CPY coherence hypothesis: four NUMA
devices have distinct state destinations. The real copy implementation divides
only `ne[1]`, but the live recurrent-state shape is `[262144,1,2,1]`, leaving one
copy worker. A separate candidate splits logical bytes across each device's own
workers and copies on every device. It is under standalone validation in
`fleet-0912-ctx/glm-copy-0919`; no skipped-device or global-barrier trick is used.

## Pool window restoration and copy window start

Pool experiment restored the original production library hashes and inference
environment under enabled, active `glm53-flash-production.service`, PID 280235.
`pool-window-report.json` confirms healthy restoration and no kernel promotion.

The all-device copy candidate passed all 14 standalone cases twice at 1–4 workers.
Frozen library SHA256 is
`5a809c34142cdd76c32bf2fa9da73c5611d72447edee1d7bc0b85aeb01304527`.
A serial full-model copy-only experiment began at about 19:49 UTC on private
port 18141, with identical launcher/environment except this CPU library and its
activation/probe flags. The harness restores original production in `finally`.
Output: `codex-bench-0919/copy-window.log`. No throughput claim yet.

A combined pool+copy library is independently validated but not yet measured:
SHA256 `4793379e6894a9286168f79c4f323985388ec0b0ac65e54013dcd2957488af1f`.
Both standalone suites pass production and all four switch states at 3 workers.

Source inspection found per-request speculative depth parameters disabled by
`#if 0` in this server's schema. Do not claim an API n_max sweep took effect.
A source-only LLAMA_MTP_DRAFT_N_FILE hook exists, but activation in the deployed
common library has not been established.

## Copy-only full-model result

The copy-only candidate engaged on all four distinct device destinations with
15 workers each. All three 256-token greedy outputs exactly match production.
Short-prompt rates: **14.10 / 16.21 / 15.71 tok/s**, compared with production
12.65 / 14.62 / 14.17. These are about 11% faster, pending repeated controls.

Corrected-catalog Codex cold probe: **12.838 tok/s**, 3,998 input tokens,
260 generated tokens, 56.517 s prefill, 20.252 s decode, 77.170 s whole turn,
71.0% draft acceptance. Marker passed. This is one stochastic observation;
it is not an established mean or an 18+ result. Pool-only previously gave 13.136.
The copy window is restoring original production; no kernel promotion.

The exact deployed common library contains LLAMA_MTP_DRAFT_N_FILE and the
activation marker, so an optional bounded MTP1/MTP2 sweep is now prepared for
an upcoming private load. No source rebuild is needed, and activation plus
fixed-output parity will be required.

## Repeated production and cache latency

Copy experiment restored the original library/environment under healthy
production PID 922334. Repeated production short prompts exactly match baseline;
rates **12.69 / 14.71 / 14.12 tok/s** closely match the earlier controls. This
supports the copy-only ~11% raw-prompt gain.

Fresh corrected-catalog production cold observation: 3,998 input / 262 output,
**11.266 tok/s**, 56.401 s prefill, 57.109 s first visible text, 80.078 s turn.
Immediate warm replay reused 3,994 tokens (4 new), generated 276 tokens at
**11.867 tok/s**, 0.290 s prefill, **0.936 s first visible text**, 23.903 s turn.
Prompt caching materially improves latency, but does not establish 18 tok/s.
These are stochastic requests; use the fixed short controls for kernel attribution.
The isolated Codex apply_patch and stateful production checks are still running.

## Codex tool smoke test and host sandbox limitation

The actual file-edit request correctly reached apply_patch. Codex then reported
that bubblewrap cannot create user namespaces on this host. The first test stopped
for review; no permission policy was changed. A second test used prompt cache and
paused for root review of the exact single-file patch. The patch hash, request id,
path, add-only status and contents were checked before accepting that one request.
Codex wrote the file and completed successfully. Behavioral tests pass all required
miss, update, eviction, capacity-zero/negative and value edge cases.

This first generated implementation used pop(next(iter(dict))) for eviction;
that is not a robust O(1) structure after repeated deletions. Do not call it a full
algorithm-quality pass. The integration probe is refined to explicitly require
OrderedDict, and a fresh target will preserve the original evidence. The initial
production-control driver stopped at the sandbox issue; separate reviewed and
stateful follow-up artifacts are authoritative for the remaining checks.

## Existing production cache consistency issue (unresolved)

The full stateful test recorded a real greedy-output mismatch between a cached
append and fresh evaluation of exactly the same prompt. Cached input reused
3,266 tokens, but the first output character and subsequent story differed.
Original artifacts are preserved in `stateful-production-control.json` and log;
its intrinsic `passed` remains false. This occurred with the original kernel.

Two simultaneously active slots produced exactly the same outputs as their serial
controls. Earlier-context divergence also matched fresh output, but reused zero
cached tokens, so this did **not** exercise checkpoint rollback. Ordinary mid-prompt
checkpoints are skipped by this server; a staged/chat-message test is needed for
that branch. Do not claim cache/rollback consistency fully passed.

For the next private kernel experiment, the separate regression gate compares
all nine outputs, including each cached/fresh scenario, against this exact
production reference and retains the failed intrinsic checks. Matching baseline
is not represented as fixing the existing consistency issue. No candidate has
been promoted. Root continues toward the 18+ Codex target while investigating it.

## Combined private window started

Frozen combined CPU library SHA256
`4793379e6894a9286168f79c4f323985388ec0b0ac65e54013dcd2957488af1f`
is loading under `glm53-flash-combo-test-0919.service`, private port 18141,
PID 1846758 at the startup check. Exact original production PID 922334 was stopped.
Normal production port 18131 is temporarily unavailable. The harness will restore
and verify the original unit regardless of experiment outcome. No overlapping loads.

This window runs fixed-output speed controls, the corrected-catalog Codex probe,
a reviewed single-file coding test, exact-scenario stateful regression checks,
MTP1 and repeated MTP2 controls through the existing library hook, and one bounded
operation trace excluded from throughput measurements. The existing production
cached-append mismatch stays explicitly failed in reports. No promotion is automatic.

The OrderedDict Codex probe passed behavior tests and manual complexity review.
Its server decode was 12.088 tok/s for 405 generated tokens over two API requests.
Client wall time includes explicit patch-review waiting and is not a latency benchmark.

## Combined validation and approved production deployment

The combined private candidate passed exact-output parity on all three fixed
prompts, both optimization engagement checks, and the reviewed OrderedDict
apply_patch coding test. The real Codex explanation measured **13.987 tok/s**;
the coding test measured **15.767 tok/s** across 306 generated tokens. The coding
client wall clock includes patch-review waiting and is not a latency benchmark.
The candidate matched all nine corresponding original-production stateful outputs.
The intrinsic cached/fresh mismatch remains failed on both kernels; no cache fix
is claimed. See `combo-window-report.json` and `stateful-combo.json`.

The MTP1 fixed prompts completed with identical output at 14.200 / 14.989 / 15.114
tok/s, about 1.9% worse in aggregate than the initial combined MTP2 controls.
The optional sweep then stopped on a diagnostic-log assertion: the activation
message is INFO level 3 but the server runs at verbosity 2. Repeated MTP2 and
profiling did not run in that window. The harness restored the original service
and exact library/environment, healthy PID 2352478. Its future activation check
now uses the actual draft-per-verification metric instead of a hidden log line.

The user explicitly approved applying this validated combined CPU kernel after
being told about its measured speed, brief restart, and unresolved cache mismatch.
The user-approved deployment is running under `promote_kernel.py`; it changes
only the CPU library prepend and enables `GGML_CPU_GLM_POOL_FUSION=1` and
`GGML_CPU_CPY_FLAT=1`. Model, MTP2, context, slots, and thread settings stay the same.
The kernel SHA256 is
`4793379e6894a9286168f79c4f323985388ec0b0ac65e54013dcd2957488af1f`.
The normal endpoint is healthy under PID 2656512. All three post-restart fixed
outputs match the original, at approximately 14.05 / 16.47 / 15.99 tok/s.
The cold/cached actual Codex checks are pending at this record's creation;
`promotion-report.json` will be authoritative for completion or rollback.

18+ tok/s in Codex/Paseo has not been achieved or established.

## Production deployment verified (2026-09-19)

Deployment completed successfully, PID 2656512, normal endpoint18131.
The kernel library hash, exact command, expected two flags, all non-CPU library
hashes, health and idle slots passed the final checks. The three deterministic
outputs still exactly match the original. The drop-in is installed at
`~/.config/systemd/user/glm53-flash-production.service.d/60-glm-kernel-0919.conf`.
`glm-kernel-promotion-manifest.json` is APPLIED and verified; authoritative report:
`codex-bench-0919/promotion-report.json`. No rollback occurred.

Actual Codex explanation through the normal endpoint and installed catalog:
- Cold: 13.375 tok/s; 3,998 input,
  266 generated; 56.922 s prefill;
  57.596 s first visible text; 77.224 s whole turn.
- Cached: 14.960 tok/s; 3,994 cached + 4 new input,
  261 generated; 0.260 s prefill;
  0.869 s first visible text; 18.058 s whole turn.
Both turns completed with the expected marker. Counter-based draft activation is
exactly 2.0 draft tokens per verification on each run. The previously validated
coding sample was 15.767 tok/s. These stochastic samples are not a guaranteed
throughput floor. The 18+ target and pre-existing cached/fresh greedy mismatch
remain unresolved. Start a fresh Codex/Paseo conversation to ensure the corrected
lowercase model metadata is loaded; the server kernel is already active.

## Bounded post-deployment decode profile completed

One diagnostic request on the healthy production endpoint used 3,919 prompt
tokens and generated 96 tokens. Profiling armed only after the first streamed
text arrived, so prompt ingestion was excluded. Exactly eight graph headers,
one arming marker and one completion marker were retained without journal
suppression. The arm file was removed afterward; service health and idle slots
were verified. Runtime identity is embedded in `profile-promoted4k.json`.
These instrumented request timings are not an additional speed benchmark.

The one captured target-model partition was a three-token verification graph
(MTP2 plus one sampled token), CPU 0, 7,239 nodes. It reported 120.274 ms of
operation time and a separate 15.906 ms in graph barriers. Individual operations
under the profiler's 5-microsecond reporting threshold are omitted from rows.
Measured rows included 41.163 ms of expert matrix work, 35.166 ms of dense matrix
work, 10.095 ms of fused indexer-pool work, 8.850 ms of flash attention, and
1.015 ms of copies. Source and attribution:
`profile-promoted4k.optrace.log` / `profile-promoted4k.optrace-summary.json`.
Do not combine this partition with separate graphs or claim an exact speed delta
against the old short-context four-token profile. The remaining cost is dominated
by matrix operations; the previously large copy category is now small in this
sample. This supports examining thread scaling and matrix kernels next.

No new thread sweep has run. The historical w69 sweep referred to Qwen, not GLM.
The deployed CPU library has a `GGML_CPU_NUMA_THREADS_FILE` control that can only
lower an already-created pool; its environment is read once. Production remains
at 15 workers/socket. A private 32-worker pool with measured activation and
15/16/other-worker controls could test this without rebuilding the CPU library,
but would require another serialized model load. Do not claim this is validated
or that 18+ tok/s is now attainable. Production remains on the approved combined
kernel, PID 2656512, with no profiling or MTP-control file enabled.

## Approved physical-core sweep started

The previous deployment turn made progress: the combined kernel was applied,
post-deployment fixed outputs matched, cold/cached Codex was measured, and a
bounded decode profile identified matrix operations as the remaining large cost.

New authoritative checks correct the earlier hypothetical 32-worker plan: the
exact deployed binary's --list-devices caps NUMA pools at 16 physical cores per
socket even when GGML_CPU_NUMA_THREADS=32 is requested. A tiny custom graph using
the exact approved CPU library directly observed ith/nth and CPU affinity at
8, 10, 12, 14, 15, 16, and 15 workers on all four sockets. All 28 cases passed;
`thread-control-probe.json` records them and the loaded kernel SHA256.

The Codex benchmark now has an optional --greedy mode (temperature 0, seed 42),
leaving default behavior unchanged. Its production check completed at 14.478
tok/s on 3,998 input / 270 generated tokens, expected marker present. Source
inspection confirms Responses conversion preserves the body and the chat parser
passes remaining sampling fields to the native request. This mode reduces output
and speculative-acceptance confounds for thread comparisons.

A complete isolated sweep is staged in thread_window.py. It leaves the normal
launcher/service files unchanged, creates a 16-worker private pool at port 18141,
uses the validated control file for 8-16 active workers, retains the exact approved
library/model/MTP2 settings, measures native and actual Codex requests with repeated
15-worker controls, and always restores approved production. Automatic approval
review initially rejected the concrete outage. No action ran on rejection. The
user then explicitly approved the 15-20 minute outage and the staged sweep.
It is now running (exec session 58991); do not start a duplicate. Native production
controls ran first. The authoritative evolving report is thread-window-report.json.
No thread setting will be promoted automatically.

The known cached/fresh output mismatch remains unresolved. Cached Codex rows in
this sweep are compared against the corresponding cached 15-worker control, and
native fixed-output checks use fresh prompts. This is not a claim of fixing or
fully validating cache equivalence. CPU frequency/governor interfaces were not
exposed under /sys/devices/system/cpu on this execution surface; no power-policy
change or thermal-limit conclusion was made.

## Thread sweep measurements completed; production restoration in progress

All seven native groups (three fixed 256-token prompts each) matched the original
outputs. All seven cached actual Codex requests also produced the same 270-token
output, reused 3,994 prompt tokens, and passed the completion marker.
Worker counts / native aggregate / actual Codex decode tok/s:
- 15: 15.043 / 14.689
- 16: 4.845 / 9.159
- 12: 14.134 / 13.789
- 8: 11.923 / 11.216
- 14: 15.271 / 14.161
- 10: 13.679 / 12.394
- 15: 15.392 / 14.426

15 workers remains the best tested Codex setting. Native 14-worker performance
falls within the 15-worker control drift; its Codex result is slower. 16 workers
is substantially slower in both contexts. No thread setting is promoted.
The harness is restoring the approved production configuration, now loading as
PID 3927330. Completion still requires health, exact identity/environment checks,
and the post-restoration native control; inspect thread-window-report.json.

The auxiliary read-only 15/16 scheduler observer missed its first trigger because
the cold-control report field was checkpointed only after the first cached result.
It was explicitly stopped; it did not affect inference or configuration. Do not
claim a 15/16 scheduler comparison. A separate incidental three-second scheduler
sample was at **8** workers (not 16), with that count recorded in its artifact.

New source/profile finding: 41 of 42 expert gate/up layers in the saved production
trace use Q4_K, and one uses Q5_K; the matching gate and up operations execute
separately. The preserved parent repack.cpp's clamped fused expert path is guarded
by expanded_iq and NB_COLS==16, so existing fusion flags alone do not enable that
path for this model's Q4/Q5 layers. Extending it is only a candidate idea; no new
kernel has been built or validated. Preserve exact arithmetic and existing expert
addressing/fast GEMV paths in any experiment.

The CPU library is built with OpenMP and loads libgomp. No OMP_, GOMP_, or KMP_
overrides were present in the running test process. GCC documents reduced busy
waiting when OpenMP worker count exceeds available CPUs:
https://gcc.gnu.org/onlinedocs/libgomp/GOMP_005fSPINCOUNT.html
https://gcc.gnu.org/onlinedocs/libgomp/OMP_005fWAIT_005fPOLICY.html
Whether that explains the 16-worker cliff is unproven. A small exact-library test
with eight NUMA backends and a retained loader/helper OpenMP team can examine the
threshold without loading another model. Changing wait policy, reducing the generic
CPU helper thread count, or building a native-thread backend remains untested.
Do not represent any of these as an established route to 18+.

cache_logits_probe.py is prepared (not run yet) to compare first-token probability
distributions for cached/fresh appends after 48 generated tokens versus a prefill-only
state. This is meant to distinguish numerical drift from a possible post-generation
state issue; greedy inequality alone is not proof of corruption.

## Thread sweep fully completed and production restored

The approved experiment completed successfully and restored healthy production
PID 3927330. Exact command, loaded library hashes, and inference
environment match the approved pre-sweep deployment. The normal endpoint18131 is
healthy and idle; private18141 has stopped. No thread or kernel change was promoted.
The post-restoration three-prompt control matched all original outputs. Native
aggregate throughput was 15.201 tok/s before
and 15.443 after the window
(ratio 1.0159). The manifest is marked COMPLETED.

The current validated setting stays at 15 workers/socket. Fixed-sampling cached
Codex controls measured 14.689 and 14.426 tok/s; all tested alternatives were slower
for this workload. This is evidence against obtaining 18+ merely by changing the
current worker-count setting; it is not a hardware upper-bound claim. The overall
tuning goal remains active, including the unresolved cache-state finding.

## Cache probability diagnostic completed; exact expert dispatch corrected

`cache-logits-promoted.json` completed six requests on healthy promoted production
PID 3927330. Initial 48-token output matches the saved original/combined reference.
After that generation (45 drafts, 23 accepted), cached append favors colon 0.400928
versus comma 0.367533; fresh identical append favors comma 0.454066 versus colon
0.302737. The greedy token differs. A prefill-only control (one sampled token,
zero drafts, sampled token excluded from append) selects comma in both cases,
but its probability differs: cached 0.536903 versus fresh 0.524936. Probability
differences therefore also occur without preceding speculative generation.
These controls have different final prompts; they do not establish MTP as the
cause, nor prove cache corruption or resolution. A fixed-continuation prefill
control is the next useful comparison.

Correction to the earlier expert candidate description: current environment uses
GGML_CPU_X16_Q4_K=1 and GGML_CPU_X16_Q5_K=1, which select tensor_traits_x16,
not the generic expanded-IQ trait with NB_COLS=16. The actual x16 clamped expert
guard accepts only Q8_0 under GGML_CPU_X16_Q8_CLAMP_FUSION. Its existing lower-level
forward_x16_moe_swiglu supports typed GEMV and optional clamps for Q4/Q5, but the
guard rejects them. Any private candidate must extend that x16 guard, preserve
current Q5 compact-p4 selection and Q4 addressing, and verify actual engagement.
No new kernel is deployed and no additional outage is authorized by these notes.

## Isolated x16 Q4/Q5 clamped fusion validated

Private library `glm-kquant-clamp-0919/libggml-cpu.so.0.22.0`, SHA256
63bf425e98ff1e426dfa89730dfee1d14b348e6f1e3688941d8902356ba8a6f6,
extends only the x16 clamped-expert guard for unchanged Q4_K/Q5_K weights.
GGML_CPU_KQUANT_CLAMP_FUSION defaults off. An optional counter, also off by
default, confirms engagement in standalone tests. Original frozen objects
relinked byte-for-byte to production before replacing repack.cpp.o.
No service, launcher, catalog, routing, or production library was changed.

280 small-shape cases and 80 real 4096x512x288-expert cases matched production
bit-for-bit with candidate disabled and enabled. All repeated outputs were
finite and identical. Coverage includes 1/3/4/15 threads, 1/2/3/4/8/32 tokens,
Q4/Q5/Q6 fallback, 24-row repack fallback, strided inputs, eight source rows,
reversed gate/up, clamps, zero inputs, and external consumers. The preliminary
engagement assertion wrongly expected single-thread fusion; saved source shows
ggml_cpu_node_is_single_task bypasses fusion for n_threads<=1. This expectation
was corrected without rerunning computations, and the preliminary report retained.
No numerical failure occurred. All actual engagement counts then matched.

Repeated isolated off/on/off/on speed controls (15 pinned physical cores, valid
quantized synthetic weights, 61 measured repetitions per case) retained exact
outputs. For three-token Q4 expert work, mean-control speedups were 1.297x with
identical expert routes and 1.108x with changing routes. Q5 was 1.113x/1.072x for
those cases; other Q5 cases were mixed. These are single-operation measurements,
not whole-model gains or proof of 18+ tok/s. End-to-end validation is still needed.
Reports are under glm-kquant-clamp-0919/validation-{small,real,speed}/report.json.

## Matched-input cache control completed

`cache-matched-promoted.json` used identical final prompt text (hash checked) for
cached state after a fixed continuation was prefilled versus generated, and fresh
re-evaluation. Prefilling the fixed continuation sampled only one excluded token
and used zero MTP drafts. Both cached paths chose colon; fresh chose comma.
Prefilled-cache colon/comma probabilities were 0.385347/0.368449; generated-cache
0.400928/0.367533; fresh 0.302737/0.454066. A repeated fresh request reproduced the
entire saved top-20 distribution exactly, and generated-cache also exactly matched
the earlier saved distribution. Generated continuation text matched its reference.

The prefilled-cache path reused 3263 tokens and processed 6, whereas generated-cache
reused 3266 and processed 3. This checkpoint/batch boundary difference remains a
confound when comparing those cached paths. The experiment nevertheless reproduces
the greedy cache/fresh mismatch without preceding speculative generation, so MTP
is not required to trigger this example. Root cause and cache consistency remain
unresolved. Final runtime identity was unchanged and healthy, PID 3927330.

## New kernel window approved and running

Automatic approval review initially rejected the separate Q4/Q5 candidate outage
because the earlier approval covered only the completed worker sweep. No command
ran on rejection. The user then explicitly approved the new 15-20 minute outage.
The reviewed kquant_window.py passed its preflight with production PID 3927330;
execution is now running as unified exec session 74756. Do not start a duplicate.
`kquant-window-report.json` and `kquant-window.log` are authoritative progress.
The private unit is glm53-flash-kquant-test-0919.service on port18141. The approved
production service will be restored, with exact runtime identity and before/after
output controls. No candidate promotion is authorized or automatic.
Pre-window native outputs matched at 14.9557 tok/s aggregate. Actual greedy Codex
controls measured 14.4238 cold and 14.2173 cached, with markers present. These are
the relevant controls for this candidate window, not historical stochastic runs.

## Q4/Q5 full-model checks passed; restoration in progress

The private candidate loaded as PID 246735 with exact reviewed library SHA,
unchanged MTP2/15-worker settings, and no non-CPU library/configuration differences.
All six fixed native outputs and the cold/two cached 270-token greedy Codex outputs
matched their production references. Native group aggregates were 14.4801 and
15.5590 tok/s (do not drop the slow first group). Actual Codex was 14.7686 cold,
14.5568 and 14.7565 cached. Pre-window controls were 14.4238/14.2173; post-restore
controls are still pending, so the small apparent gain is not yet a final result.
All Codex rows used 210 draft tokens, 164 accepted, 105 verification steps.

All nine stateful/parallel outputs and the intrinsic check results matched the
original production reference. The cached/fresh mismatch remains false; no cache
fix is claimed. Two concurrent slots were active and matched the serial controls.
The separately timed eight-graph profile captured a target graph with 41 Q4 gate
layers and one Q5 gate layer, all 42 down layers, and no separately executed expert
up layers. This proves full-model engagement alongside standalone counter checks.
The profile request and its complete output also exactly matched the earlier
promoted-kernel profile. Profile timings are excluded from throughput results.

The harness is now stopping the private model and restoring approved production.
Do not claim restoration complete until kquant-window-report.json confirms health,
exact identities, and final before/after control output parity. Exec74756 remains
active. No candidate promotion or normal service configuration change has occurred.

## Q4/Q5 window fully completed; production restored; no convincing speed gain

The approved window completed with exit0 and restored healthy, idle production
PID 722759 after 1030.053 seconds (17.17 minutes). Exact command, mapped library
hashes, and inference environment match the pre-window approved configuration.
Private unit glm53-flash-kquant-test-0919.service is stopped (MainPID0), private
port18141 closed, and the profiling arm is absent. Normal production still uses
15 workers/socket and approved CPU SHA4793379e6894a9286168f79c4f323985388ec0b0ac65e54013dcd2957488af1f.

All post-restoration native and cold/cached Codex outputs matched their pre-window
controls. Native before/after aggregate was 14.9557/15.4585 tok/s (3.36% drift).
Candidate groups were 14.4801 and 15.5590, combined 15.0002 versus controls15.2030.
The slow first candidate third prompt is included; no favorable-run selection.

Actual greedy Codex cold: production before14.4238, candidate14.7686, restored
production14.8898 tok/s. Cached: production before14.2173, candidate14.5568 and
14.7565, restored production14.8293. Candidate cached mean14.6567 versus mean
controls14.5233 is only +0.918%; cold advantage over mean controls is +0.763%.
Control cached after/before drift is +4.30%. Thus there is no convincing whole-model
speed gain; restored production is slightly faster than the candidate on the final
controls. Do not promote this candidate on these measurements. Correctness and
fusion engagement passed, but 18+ tok/s inside Codex remains unestablished.

`kquant-window-summary.json` contains the complete scoped result; the window and
candidate manifests are marked completed/validated-not-deployed. The normal endpoint
is healthy. The overall tuning goal remains active, and the existing cache mismatch
remains unresolved. A useful next read-only direction is Codex input overhead:
the current actual request has504 instruction characters but12947 serialized input
characters. Its developer/context payload has not yet been broken down; do not
assume tool-schema removal or disabling features is safe or will reach18.

## Codex prompt audit: feature JSON size is not model prompt size

Production was revalidated healthy and idle as PID722759, CPU SHA4793379..., with
no pending kernel window. The previous turn was progress: it completed an approved
candidate experiment and ruled out a convincing speed gain while verifying restore.
Installed Codex is0.154.0; Paseo's installed provider appends --enable goals, matching
the appserver harness. Private offline captures made no inference requests and did
not change production. Each baseline request has504 instruction characters,
12947 serialized input characters, and14991 serialized tool-schema characters.
The developer text consists mainly of skills5602chars and permissions4065chars.

Authoritative `/v1/responses/input_tokens` counts are3998 for baseline, multi_agent
disabled, goals disabled, and both disabled. Disabling multi-agent removes9756 JSON
characters but no model tokens: the actual Responses converter skips non-function
tools, including namespace, freeform, and web_search. Do not assume those tools are
available to GLM just because Codex advertised them. Prior successful file edits
used shell-mediated patch commands; no new direct freeform support was validated.

Tokenizer-only component attribution: removing the developer item counts1564,
removing tools counts2900, and removing the skills block counts2669. These are
attribution probes only, not proposed generation configurations; permissions and
skill guidance remain in production.

A private catalog switch test with include_skills_usage_instructions=false retains
the exact skill catalog and tool schemas but removes2554 characters/561 tokens of
usage guidance, reducing3998 to3437. It has not been applied or treated as a free
optimization. No-generation artifacts are in prompt-overhead-0919 and
prompt-skill-variants-0919. The remaining question is how to reduce actual compute
while preserving useful coding behavior; JSON payload reduction alone does not help.

## Q4 token batching: standalone validated, separate full-model window staged

A new candidate shares Q4 unpacking across two/three token activations in ordinary
x16 MUL_MAT and MUL_MAT_ID dispatch, preserving each token's exact accumulation
order. It does not include the previous Q4/Q5 clamped fusion candidate. Production
remains healthy and idle as PID722759, 15 workers/socket, approved CPU4793379...
No service restart, model request, or production configuration change occurred.

Prototype `fleet-0912-ctx/glm-q4-batch-0919` passed1080 direct comparisons. Initial
correctness used the older validated-bin base library; a hash guard caught the
identity mismatch before timing. Correctness was repeated with exact production
base598563... and CPU4793379..., producing the same digest. Only these exact-library
micro timings are accepted. Two/three-token 64-row tiles improved1.17–1.40x across
hot/rotating weights. Exploratory512-row results are not representative expert tiles.
Both batch template implementations have no ZMM stack spills.

Candidate `fleet-0912-ctx/glm-q4-batch-cpu-0919` CPU SHA
90282e20361e22cffbdd08e2aab25e4a510aeadc199df781727d1adf8ecc35e1
is default-off under GGML_CPU_X16_Q4_BATCH=1. Parent relink was byte-identical to
approved production; replaces only repack object and adds batch object. Optional
GGML_CPU_X16_Q4_BATCH_PROBE counts batch2/3 calls; disabled for timings. Other quant
types, single-token calls, and existing fused gate/up implementations are unchanged.

All1304 graph cases matched production bit-for-bit across production/off/on:
280 small experts,80 full4096x512x288 gate shape,80 full512x4096x288 down shape,
864 dense/broadcast graphs. Exact engagement counts passed. Two preliminary dense
count expectations were corrected to production's GGML_CPU_X16_CHUNK_MAX=16 and
fusion requiring input plane count=1 (including broadcast). Numerical outputs were
already identical; saved outputs were rechecked and all reports retained. Kernel
code was not changed by these test-expectation corrections.

15-pinned-core off/on/off/on graph timings used61 repeats per median, counters off,
and production task IDs477/242 remained unchanged. Three-token Q4 gate-shape graph:
1.325x same routes,1.040x mostly divergent routes. Down-shape:1.167x same,1.013x
divergent. These synthetic graphs include two matrix ops/clamps/SwiGLU; down shape
is dimensional coverage, not an exact real down-subgraph timing. Some unchanged
single-token/Q5 controls drifted by several percent. Full results, including every
row and control drift, are in q4batch-standalone-summary.json. No whole-model speed
claim,18tok/s claim, or cache fix is supported yet.

A distinct full-model window is staged in q4batch_window.py and
q4batch-window-manifest.json. Read-only preflight passed. It requires a NEW approval
for this different candidate; prior approved Q4/Q5 fusion outage was completed and
restored. It is not running. Window uses private18141/unitglm53-flash-q4batch-test-0919,
unchangedMTP2/15workers, no overlapping model loads, before/after production controls,
six native outputs, cold/two warm actual Codex outputs, stateful/parallel regression,
and graph-structure profile; finally restores exact production. No automatic
promotion. Profile alone does not directly count live batch calls. Manifest approval
is pending; do not execute or change it to approved without the user's new answer.

## Approved Q4 batch full-model window: production restored, final controls running

The user explicitly approved this separate batch-kernel outage. The staged harness
is running in unified exec session61581; do not launch it again. Private PID1282309
loaded exact CPU90282e... with unchanged non-CPU libraries and all production
settings except its private port/library and opt-in batch flag. All six native
outputs and cold/two warm 270-token greedy Codex outputs matched production.
Native candidate groups15.3923/15.1203 tok/s versus pre-control15.3928. Codex candidate
cold14.3518; warm14.6207/14.6017 versus pre-control14.4478cold/14.5325warm. No clear
performance benefit is established before post-controls.

All9 stateful/parallel outputs and intrinsic checks matched the production reference;
regression passed, intrinsic cached/fresh consistency remains false. Bounded profile
has all42 gate/up/down expert operations and unchanged7239-node target structure.
This is structural evidence, not a direct live batch-counter measurement. Instrumented
profile timings are excluded from throughput comparisons.

The private candidate is stopped. Approved production restored healthy as PID1756372
with exact original mapped hashes, inference environment and command line after
1032.210 seconds=17.20minutes. q4batch-window-restored.json is authoritative. The
harness is now collecting final native and cold/cached Codex controls; do not claim
window completion or finalize speed conclusions until it exits and the report has
codex_after. No candidate deployment or production config mutation occurred.

## Q4 batch window completed: no convincing overall gain; production healthy

Exec61581 exited0. All final native and cold/cached Codex controls matched their
before-window outputs. Production PID1756372 is healthy and idle on approved
CPU4793379..., 15workers/socket. Final command, inference environment, and mapped
hashes exactly equal original production. PrivatePID1282309 is dead, private18141
closed, profiling arm absent. Outage1032.210s=17.20min. No promotion/config change.

Native before15.3928, candidate groups15.3923/15.1203 (combined15.2551), after15.2309,
combined controls15.3114. Codex cold before14.4478, candidate14.3518, after14.5978.
Cached before14.5325, candidate14.6207/14.6017, after14.2929. Candidate cached mean
14.6112 versus controls14.4127 is+1.377%; cold is-1.177%. Cached control drift=-1.649%.
Thus no convincing overall speed gain; candidate's cached advantage is smaller than
control drift, while cold/native are slightly slower. Keep approved production.
18+tok/s still unachieved. All9 stateful/parallel regressions passed, existing intrinsic
cache/fresh mismatch remains false. No new full-model outage is pending or running.

Final human report Q4-BATCH-WINDOW-RESULT.md; complete q4batch-window-summary.json.
Window/candidate manifests marked completed/regression-passed/not-deployed.
Reviewed standalone manifest was archived before updating status metadata. All
reviewed files were verified unchanged at final runtime audit before metadata updates.
Do not rerun the completed q4batch_window.py. This turn completed an approved
experiment and ruled out another candidate; the broader tuning goal remains active.

## Live scheduling and standalone dispatch investigation completed

Production remains PID1756372, approved CPU4793379..., 15workers/socket. Final audit
`dispatch-probe/final-runtime.json` confirms exact original command, inference env,
mapped hashes, main affinity, and all nine previously altered helper/main affinities.
Health is OK, slots idle, profiling disarmed. No new full-model outage occurred or is
pending. Existing batch, fusion and worker windows remain completed; do not rerun them.

A read-only /proc observer on actual cached greedy Codex output recorded18.4218s:
120 pinned threads56.2868 logical CPU-seconds/wall-second,140unbound1.4993. Pinned
runqueue .30967s versus1036.906CPU s. Main+eight helpers account for most unbound
system CPU (22.44s total unbound system). Exact output270tokens,3994cached,210draft/
164accepted/105verify. Observer cost1.4916CPU s,maxsample.1117s;14.4394tok/s is NOT an
uninstrumented speed result. perf was blocked by kernel permissions; security settings
were not changed. Preserved base598563... source's 20ms sched_yield polling is a
supported hypothesis for helper systemCPU, not sampled call-stack proof.

Completed bounded helper-affinity comparison: original14.6532, spare-only13.2424/
13.3564, restored14.6381tok/s. All exact270-token outputs/3994cache. Spare-only
runqueue19.24/19.31s versusoriginal.030/.089s. All original masks restored in finally
and final live audit. Do not apply this affinity change. Cores15,31,47,63 are unused
by15-worker GLM compute but are NOT globally idle. The observer already recorded
82.51niceCPUseconds on those cores+SMTs over18.42s. A later process/affinity sample
identified Godot/container workloads on15,31,47,63,79,95,111,127. Other work was not
changed. This is a current confound, not proof of the old full-model16-worker cause.

Built `dispatch-probe` against exact production CPU/base/ggml. Separate target/draft
meta contexts across4nodes, Q4 dense/residual/SiLU/RMS graphs with fused all-reduce.
Synthetic weights are reused; no real attention/cache/MoE routing.47successfulcases,
1preserved allocationfailure. Initial16/spin0 failure was8GiB RLIMIT_AS, not numerics;
meta arenas reserve~6GiB alone. Runner raised its own virtual limit24GiB with sampled
RSSabort6GiB; maxsampleRSS<273MiB. Production idle/taskIDguards passed all47cases.
21finalcases saved complete output bytes, exact across3shapes/1213measuredcycles;
everycycle checked target/draft output against warm bytes. Earlier cases have equal
fingerprints only. Versioned source/build/runner archives retain provenance.

Lower dispatch spin greatly reduced systemCPU but long15-worker timing gains were
small/notrobust. GOMP_SPINCOUNT0 did not fix16-worker slowdown. The existing optional
GGML_CPU_OMP_SIMPLE_BARRIER1 looked1.417x faster in a probe with an artificial retained
128-thread OpenMP team (66.180→46.717ms mean runmedians). Crucial control WITHOUT that
extra team showed0.973x (53.756→55.264ms); aggregate ratios1.424x vs0.961x. Thus do NOT
promote or stage an outage based on the apparent40% gain.

Production's128earlyunboundthreads startedwithin.09s; server source defaults127HTTP
workers pluslistener. This supports HTTP origin, not retained OpenMP workers, but no
user-space stacks were sampled. Do not count HTTP threads as OpenMP managed threads.
Later16-worker no-loader case was also severely slow while unrelated workloads used
the extra4cores; the earlier attribution to added OpenMP loader team was premature.
No established production wait-policy/barrier improvement or18+ claim. Human report
`codex-bench-0919/DISPATCH-PROBE-RESULT.md`, machine summary`dispatch-probe/summary.json`.

Goal-turn classification: PROGRESS. Completed controlled observations, ruled out an
affinity change and an artificial standalone speedup, and documented real interference.
The overall tuning goal remains active; this is not a blocked turn or a completion.
No new permission request is pending. Cache consistency remains unresolved.

## Wider kpool inner loop built; final A/B started

The user redirected work to the existing kpool design, then explicitly requested
compilation and A/B testing on :18131. Inspection confirmed that the design's
first fusion is already deployed; its old “design only” status was stale. That
status and the two original candidate READMEs have been corrected. The claimed
+50% was a projection over the unfused September 18 runtime, not an incremental
projection over current approximately 14 tok/s performance.

A separate private implementation now groups eight channels, hoists member-cell
lookups out of the channel loop, and preserves scalar expf, float products,
sequential double accumulation, and the original reciprocal. Parent CPU objects
and strict fusion guards are unchanged. Candidate SHA256:
`3f957a341321b940d93be53c250cdd068825093faa2d9efda142ebe56427b1b3`.
Source/build/gates: `fleet-0912-ctx/glm-kpool-wide-0919`.

All 77 cases, each executed twice, passed full-byte equality across five arms and
four worker counts (1/2/3/15). Outputs include 32K/100K pool sizes; 85,564,960 bytes
match in each arm. Repeated dynamic switching at 3/15 workers and existing copy
regressions also pass. Preliminary standalone ABBA timings show about 2.0x/1.8x
faster pooling at 32K/100K; these are not end-to-end token rates.

`kpool_window.py` now runs the final binary on :18131, using a read-only mapped
0/1 control changed only between idle requests. It will compare cold and cached
Codex requests at approximately 4K and 30K contexts, native text parity, stateful
regression parity, and separately instrumented long-context profiles. Both the
controller's finally block and the temporary service's ExecStopPost restore the
approved production service. No promotion is automatic. The current cache
consistency failure remains outside this change.

### Preserved cache findings before the kpool redirect

`cache-boundary-short-scout.json` completed on the unchanged deployed runtime.
Using raw token IDs removed text-retokenization ambiguity. At final lengths 128
and 512, eight cached-prefix cases covered all four prefix residues modulo four.
Every first token matched the fresh control, but all eight top-20 probability
vectors differed (largest shared probability differences approximately 0.0065 to
0.0691). The repeated fresh 128-token control was exact. No speculative drafts
were generated in these one-token probes, and cached token counts matched the
requested prefixes exactly. This rules out crossing the 2,051-token sparse
selection limit, prior MTP generation, and a four-token rewind as necessary
triggers for these particular probability differences. It does not distinguish
cache corruption from legitimate differences in execution order/numerics.

A read-only comparison of existing deployed Q4/Q5 dense outputs found 720/720
bit-exact comparisons between 1/2/3/4/8-token outputs and corresponding columns of
32-token outputs across 144 shape/type/thread/mode groups. See
`batch-numerics-saved-dense.json`. That evidence narrows only those saved Q4/Q5
dense cases; it does not establish equivalence for Q8, attention, or the full model.
No cache-related runtime or source change was applied.

## Wider kpool A/B completed and kernel deployed

The final same-process ABBA comparison passed identical text and all input,
cache, generated-token, draft-token, accepted-draft, and verification-step counts:

- Actual Codex 3,998-token input: 14.6017 → 15.1545 tok/s, **+3.8%**.
- Actual Codex 29,930-token input: 9.0510 → 10.2604 tok/s, **+13.4%**.

The two scalar controls drifted about 1.0% (short) and 0.1% (long). Cold pairs were
14.3854 → 15.2879 and 8.9842 → 10.2578 tok/s; the long cold pair used one different
speculative verification, so matched ABBA is the primary estimate. Original-service
controls before/after were 14.4648/14.7807 tok/s, with a one-accepted-draft difference.
No 18+ or 20+ rate is claimed for this standard Codex task.

Both profiles had the same 29,952-cell / 7,490-pool / three-verification-token
shapes. Sampled attributed pooling work fell 54.714 → 24.284 ms. Other operations
varied between the sampled CPU workers, so graph total time is not used as the
throughput claim. Unprofiled full requests supply the end-to-end rates above.
All six native, twelve Codex A/B, two profiled Codex, and nine stateful regression
outputs matched. Cold and cached Codex text also matched for these fixtures.
The old stateful cached/fresh inconsistency remains failed and is not fixed.

The A/B window lasted 45.96 minutes including reloads, numerical/behavior checks,
and two roughly 12-minute cold long-context prefills. Its controller and temporary
unit restored the original exact runtime, healthy PID 3394183; its final control
passed. No second model was loaded concurrently. Source, results, and manifest:
`glm-kpool-wide-0919`, `KPOOL-WIDE-RESULT.md`, `kpool-window-summary.json`, and
`kpool-window-report.json`.

The tested wider loop was then enabled for the ongoing GLM optimization using
`promote_kpool.py`, which had passed a read-only proposal check and included
rollback on failed runtime/output verification. It installed only the separate
`70-glm-kpool-wide-0919.conf` drop-in, selecting the immutable `deploy/` library
and `GGML_CPU_GLM_POOL_WIDE=1`. Existing production configuration remains below
that drop-in. The benchmark control-file environment variable is absent.

Production PID **3692141** is healthy and verified. CPU SHA256 remains
`3f957a341321b940d93be53c250cdd068825093faa2d9efda142ebe56427b1b3`.
Other mapped libraries, model arguments, and inference flags match the restored
baseline. Native text parity passed after deployment, followed by actual Codex
cold/cached smoke checks at **15.0380 / 15.0586 tok/s**, with exact reference text.
The final snapshot is `kpool-promotion-final.json`; the deployment report is
`kpool-promotion-report.json`. The original library remains available for rollback.

## Historical GLM phase-accounting correction

The September 14 Qwen-specific parser omitted all 89-node GLM draft graphs. A new audit includes the draft catch-up and both prediction passes, separates prompt ingestion, and reconciles every target verification with saved response counters. Across 512 steady cycles, the corrected host residual is about 9.6-10.2 ms/cycle, with another 4.3-4.4 ms of draft preparation already included in graph time. These are old instrumented timings, not current throughput or a projected gain. The CPU op profiler also reports barriers separately from its node total. See `codex-bench-0919/PHASE-AUDIT-RESULT.md` and `glm-phase-audit.json`. Production remained unchanged and healthy on the wider-pool kernel.

## Bounded KV rollback A/B completed and deployed

Recovered a byte-identical libllama parent from 183 frozen objects and three preserved source revisions. The parent SHA256 exactly matches the original deployed `8c794722...c5c8eb08a`. The candidate changes only the KV-cache object, bounding rollback scans by the highest occupied physical cell plus one instead of scanning all 1,048,576 allocated cells. It changes no floating-point arithmetic. The original-source off branch is retained, but compiler optimization differs from the original binary; it is not an instruction-identical control.

The final candidate passes production/off/on/alternating-mode exact state comparison across 16 configurations, 5,650 scripted steps and 2,826,776 bytes per arm. Coverage includes sparse high indices, shared cells, checkpoint restore, sequence operations and allocator state. Full model validation then passed six native outputs, fourteen identical Codex texts (including profiles and original-production controls), and nine stateful regression outputs. The pre-existing cached/fresh consistency failure remains unresolved.

Actual Codex/Paseo at 3,998 input tokens, same process, eight cached runs in ABBA/BAAB order: **15.2387 → 15.6078 tok/s, +2.42%**. Blocks improved +3.19% and +1.66%; every input/cache, output, draft, acceptance and verification counter matches. The cold pair was 15.09 → 15.70 tok/s. Original production controls bracketing the window were 15.10 and 15.24 tok/s. No new long-context throughput measurement or 18+ claim is made.

Fresh coarse profiles separately account for all three draft passes. Across 101 steady cycles, outside-graph time fell from 9.616 to 6.032 ms; draft-model time remained around 21.4 ms, including 4.33 ms preparation. Target graph time varied, so uninstrumented repeated throughput is the primary result. Both profiles reconcile 105 verifications and 422 total graphs. Profiling is now disarmed.

The first live attempt stopped on an INFO probe filtered by production verbosity and restored the original configuration after 10.54 minutes. Raising only that opt-in probe to WARN changed one executable byte and the build ID; exact gates passed again. The successful R2 window completed in 19.79 minutes and restored the exact original runtime as PID 1266046 before promotion.

The separate verified promotion installed only `80-glm-kv-range-0919.conf`, selecting the immutable `glm-kv-range-0919/deploy-r2` library and enabling `LLAMA_KV_SEQ_RM_USED_PREFIX=1`. Wider-pool CPU SHA256 remains `3f957a34...27b1b3`. Final libllama SHA256 is `e77a573df69456745857677e71e4e9cc876b8294384fa7798629f3ebe7c3306a`. All other libraries and model arguments match. Production PID **1561764** is healthy and independently audited. Final native parity and actual Codex cold/cached checks passed at **15.5796 / 15.7127 tok/s**, with exact reference text. Both profiling arm files are absent; no benchmark switch or probe flag is deployed.

Human result: `codex-bench-0919/KV-RANGE-RESULT.md`. Evidence: `kv-range-r2-window-summary.json`, `kv-range-promotion-report.json`, `kv-range-final-audit.json`; source/build/gates in `fleet-0912-ctx/glm-kv-range-0919`. Goal-turn classification: PROGRESS. The broader tuning goal stays active.


## MTP cache-only catch-up A/B completed and deployed

The catch-up call requests zero output rows, so the guarded MTP branch retains exactly the normalization, combined projection, KV projection/normalization and cache write while omitting unused query/attention work. Its parent rebuild matches the current deployed libllama byte for byte; only models/glm5next.cpp.o changes. All four standalone modes at both 1 and 15 workers/socket match 170 records / 28,400,396 bytes, including actual cache bytes, sequence metadata, logits and hidden outputs.

Actual Codex/Paseo at 3,998 input tokens: eight cached runs in same-process ABBA/BAAB order give **15.43298 -> 15.82093 tok/s, +2.51377%**. Both blocks improve (+2.69088%, +2.33489%), with identical text and every input/cache, generated, draft, acceptance and verification counter. The matched cold pair is 15.28187 -> 15.77103 tok/s. Original-service controls are 15.84702 / 15.33847, with about 3.2% drift and a one-accepted-draft difference; they do not replace the balanced primary comparison. All six native, fourteen Codex and nine stateful outputs match their references. Existing cached/fresh inconsistency remains unresolved.

Both instrumented traces account for 422 graphs and 105 verifications. Across 101 steady cycles, total draft work falls 21.269 -> 18.152 ms, with catch-up 7.398 -> 4.411 ms. Target graph time remains about 138 ms and draft preparation about 3.349 ms. Profiles are excluded from throughput calculations. No new long-context throughput measurement or consistent 17+/18+ claim is made.

The first shared-port attempt stopped before enabling the candidate because another request was active, and restored production. Its contaminated control is excluded. The private-port retry finished every inference A/B and both traces, but its final parser rejected the expected new 15-node graph because the imported parser allowed only 7152/89 nodes. The raw failed report is unchanged. After exact automatic restoration (1180.28 seconds), saved traces passed offline mode-specific parsing and a final production control passed. The derived validation report records the error and hashes its raw evidence. No test outage was repeated for this parser fix.

The separate pinned promotion installed **90-glm-mtp-kv-only-0919.conf**, selecting immutable libllama **65670d4a19178bbdffcfbc79ee953e5319083875309ed52aa4de214393e20ce3** and GGML_GLM5N_MTP_KV_ONLY=1. Wider pooling and bounded rollback remain enabled. Production PID **3717968** is independently audited, healthy and idle, with exact expected command/environment/libraries. No benchmark control/probe or profiling arm is active. Final native parity passed; actual Codex cold/cached smoke rates are **15.66716 / 15.51947 tok/s**, exact text and clean request-owned counters. Deployment session 18401 completed successfully.

Human result: codex-bench-0919/MTP-KV-RESULT.md. Authoritative evidence: mtp-kv-only-r2-validation-report.json, mtp-kv-only-r2-window-summary.json, mtp-kv-only-r2-phase-breakdown.json, mtp-kv-only-promotion-report.json and mtp-kv-only-final-audit.json. The request accounting audit has 93 matches and one excluded shared-port mismatch. A source/profile audit ranks dense Q8 work at 32.473 ms and expert matrices at 42.253 ms in a saved target graph; Q8 multi-token batching is already deployed. Remaining priorities are documented in NEXT-PRIORITIES.md. Goal-turn classification: PROGRESS; the broader goal remains active.


## Q8 batching follow-up: exact, small synthetic gain, not deployed

Built a CPU candidate over a byte-identical wider-pool production parent, changing only the x86 repack object. Two integer chains plus inline activation sums for NR=2..4 / NC=16 pass 1,296 direct matrix cases and all 1,936 compiled graph cases, including dynamic off/on switching. All direct and graph output bytes match production at the corresponding worker count.

The isolated kernel improvement does not translate proportionally to four sockets: first separate-process median cycle gain 1.61%; refined same-process ABBA/BAAB gains 0.50% with four weight sets and 0.82% with 32 distinct sets. The latter target phase improves 1.45% on median; unchanged draft time also varies. An excluded diagnostic arm confirms actual candidate engagement. No artificial OpenMP team is used. This is a synthetic graph result, not an actual model throughput gain.

Candidate retained, not deployed. Production PID 3717968 and all mapped library/environment identities are unchanged; no outage occurred. Full evidence and decision: /home/user/InfoSystemic/AI-Server/serving/fleet-0912-ctx/glm-q8-batch-0920/RESULT.md.
