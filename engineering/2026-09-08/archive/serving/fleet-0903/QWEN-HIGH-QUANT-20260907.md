# Qwen3.8-Flash-Next higher-precision trial

Latest September 8 goal: get Qwen above 30 tok/s and above 190 GB/s (50% of
380 GB/s), retaining the selected Q6 precision. This supersedes the all-model
75% objective below. See [the current Qwen work log](QWEN-30TPS-20260908.md).

September 8 follow-on: the active all-model target is now 75% of 380 GB/s,
or 285 GB/s. It has not been achieved. The [new work log](BANDWIDTH75-20260908.md)
records decode-only operation traces and an isolated FP32 scheduling experiment.
That component candidate is not selected, and no new model-throughput result
supersedes the final Q6 measurements below.

Target: higher-precision Flash-Next exceeding 28 generated tok/s. Each model
may use the whole server; keeping GLM Full resident is not a requirement.

Selected artifact: `unsloth/Qwen3.8-Flash-Next-GGUF`, revision
`38bb39ee97821de2c9009abb7e93950eec396e66`, `UD-Q6_K_XL`.
All six shards, totaling 169165382688 bytes, are downloaded and SHA-256 verified.
The files are under `/home/user/.local/share/ai-models/Qwen3.8-Flash-Next-38bb39ee9782`.

The Q6 preset was promoted on port 18095 and is saved in
[`qwen-flash-20tps.json`](qwen-flash-20tps.json), consumed by the existing
[`launch-qwen-flash-20tps.py`](launch-qwen-flash-20tps.py) launcher. It uses
balanced NUMA splitting, 15 workers per socket, MTP4 with confidence 0.3, and
the validated wider Q8 batching library. Another session subsequently loaded
this Q6 preset while Flash Q8 was resident, and the kernel killed Flash under
node-0 memory pressure. The latest read-only preflight observes Qwen PID
1219506 on port 18095, idle with no queued requests. Qwen is preserved pending
coordination about exclusive server access. Its launcher and launcher generator
now include a checked refusal of overlapping model loads; the preset, weights,
and runtime settings are unchanged. Evidence is in
`results/exclusive-model-launch-0908/result.json` and
`results/glm-flash-dissemination-control-0908/preflight.json`.
Final fresh-prompt measurements are
**28.4619 generated tok/s on code and 21.5350 on prose**. The preceding wider-Q8
run measured 28.0952/21.3158. This meets the requested 28+ rate on the tested
code workload; it does not establish that rate across all prompts. MTP3's best
prose result is 22.9855. The final code run used 136.5827 GB/s after idle-traffic
subtraction, about 35.94% of the requested 380 GB/s capacity. 85% utilization
has not been reached.

Prompt reuse is off by default through `--no-cache-prompt`. Arithmetic/factual
checks, eight generated-code behavior cases, and all four default-versus-fresh
continuation cases pass; each default continuation reports zero reused prompt
tokens. Explicit `cache_prompt=true` still has the earlier extension-parity
failure and should remain disabled. This avoids the known path by default;
it is not a claim that the underlying explicit-reuse defect is repaired or that
model quality has been broadly certified. Full remains stopped. Q2/27B/model
files are retained, and the previous launcher configuration is backed up in
[`results/qwen-q6-promotion-0907/previous-q2-config.json`](results/qwen-q6-promotion-0907/previous-q2-config.json).

After the authorized Chrome close, the completed Q6 measurements are:

| Configuration | Prose tok/s | Code tok/s | Adjusted memory GB/s, prose/code |
| --- | ---: | ---: | ---: |
| Balanced MTP3, Q8 row batching | 22.8676 | 27.5151 | 132.7499 / 136.4534 |
| Balanced MTP4, Q8 row batching | 21.2349 | 27.8844 | 129.3367 / 133.9200 |
| Balanced MTP3, fused Q8 row batching | 22.6837 | 27.3346 | 131.4253 / 135.3389 |
| Balanced MTP3, Q8 expert x16, 16 workers | 18.6683 | 24.2975 | 118.6222 / 123.1732 |
| Balanced MTP3, Q8 expert x16, 15 workers | 20.9356 | 27.0086 | 132.4753 / 137.2408 |
| Balanced MTP3, additional dense Q8 x16 | 22.6442 | 27.2276 | 130.1620 / 131.5225 |
| Balanced MTP3, atomic barrier, spin 1000 | 22.9855 | 27.4694 | 135.2783 / 137.6660 |
| Balanced MTP4, confidence 0.3, run 1 | 20.8771 | 28.2275 | 122.1375 / 135.0683 |
| Balanced MTP4, confidence 0.3, repeat | 21.2700 | 27.9224 | 125.5056 / 133.8278 |
| Balanced MTP5, confidence 0.3 | 19.1087 | 12.2903 | 117.9777 / 70.6140 |
| Balanced MTP4, confidence 0.3, wider Q8 batching | 21.3158 | 28.0952 | 124.6812 / 134.6526 |
| Same wider Q8 configuration, 14 workers | 19.3612 | 27.2827 | 119.1790 / 129.3690 |
| Final wider Q8 MTP4, 15 workers, default prompt reuse off | 21.5350 | 28.4619 | 125.9915 / 136.5827 |

All thirteen runs passed their arithmetic/factual checks and captured valid memory
counters. The first two started below four background CPU cores. MTP3 remains faster
across these two workloads. Confidence-limited MTP4 crosses 28 tok/s on code in
one completed measurement but falls below it on repeat; a consistently higher
rate is not established. Its code draft counts exactly match uncapped MTP4,
so the small observed difference is not evidence of a cutoff-driven gain.
Extending the same bit-exact Q8 batching to fused dense gate/up passed
168480 kernel comparisons and 432 graph cases, but its full-model run showed no
measurable speed gain.

A separate component probe compared the existing Q8 expert kernels with the
x16 kernel on real per-socket gate/up and down-projection shapes. All 48 graph
cases passed canonical numerical checks. Most tested shapes improved by about
1.5-3.3 times; single-token down-projection was about 10% slower. These are
focused component timings, not full-model speed. A private opt-in selector is
built and undergoing four-NUMA, 512-expert validation before model use. The
selector preserves Q8 quantization; it does not promise identical FP32 output
bits to the previous kernel. Both four-NUMA validation suites subsequently
passed with all 512 experts: Q6 gate/up with Q8 down, and Q8 gate/up with MTP
geometry. Cross-layout comparisons cover 41779200 values with maximum scaled
error below 0.0000151. Both 64-token unfused prefill cases also pass. The
expert selector did not improve full-model throughput at either 15 or 16 workers,
so it is not selected for the next candidate.

The tensor inventory identified recurrent-output and hyperconnection matrices
that also bypass the optimized dense Q8 path. A 128-case component probe of
their real shapes passed canonical numerical checks. A new private selector
enables x16 for `ssm_out`, PLE key/value, and HC down projections; HC up remains
on its earlier path because single-token probes regressed there. The selector
passes 432 additional exact comparisons with the validated x16 implementation
using the real tensor names. Its completed MTP3 run with 15 workers per socket
did not improve throughput. Weight quantization remains unchanged.

An eight-second call-stack profile of this candidate found two separate teams
sharing each of the 60 pinned worker CPUs: target and MTP draft. Several OpenMP
waiting sites together account for over half of sampled CPU cycles. The
lower-consuming thread in each pinned pair contributed 18.36% of their combined
CPU ticks over the surrounding observation window. Some of that is useful draft
work; these measurements do not identify an equivalent removable wall-time
fraction. A private candidate reuses ggml's existing
atomic graph barrier and sets `GOMP_SPINCOUNT=1000`, allowing inactive OpenMP
teams to sleep without replacing active graph barriers with short spin waits.
The unmodified C source rebuild matches the installed object's instruction
bytes. The candidate passes 864 bit-exact Q8/Q6 graph comparisons and 60 NUMA
reduction/thread-count cases; original engine inputs are unchanged. Its first
Q6 fixture attempt incorrectly generated raw random bytes in a Q6 scale field
and failed with the new barrier disabled. That attempt is retained separately;
the corrected fixture uses the canonical Q6 quantizer with the same numerical
tolerances. Full-model throughput is essentially unchanged at 22.99/27.47
tok/s, so this experiment does not establish a speed gain or meet the target.
The follow-up profile attributes 44.27% of sampled cycles to the atomic graph
barrier, with Q8 x16 GEMV at 15.08% and Q6 x16 GEMV at 7.97%. Waiting therefore
remains substantial inside active computation; reducing inactive-team spinning
did not translate into a material decoding gain. The next trial returns to the
established Q8 batch library to test MTP4 with a 0.3 draft-confidence cutoff.

MTP5 with that cutoff regresses both workloads, especially code. The recorded
background process samples show ordinary load, and a subsequent live five-second
sample found no busy Chrome process. This does not isolate the slowdown's cause.
MTP5 is not selected. A private eight-row Q8 batch extension passes 842400 exact
kernel comparisons and 648 graph cases (ordinary/padded, 1/4/15 threads, one
through nine activation rows). Component timings are mixed. Its completed MTP4
measurement is 21.32/28.10 tok/s, again close to the previous configuration;
a material speed gain is not established.
The 14-worker comparison regresses to 19.36/27.28 tok/s, and its control file
was restored to 15. Another newly started headless Chrome instance consumed
about eight CPU cores during that run's background gate. Its exact main PID
and start time were verified, then it was closed through pidfd/SIGTERM under
the user's existing authorization. All 11 processes in that instance exited;
the parent workflow was not signaled. Measurement began after the quiet gate
passed at roughly 3.4 background cores.

A newly started headless Chrome instance also caused contention during the
64-core benchmark's background gate. It was closed under the user's existing
authorization, and all 11 processes in that instance exited. A separate
one-core build then finished, allowing measurement under the original
four-core background threshold. No benchmark requests were interrupted.

The first full-model startup exposed two 256-expert limits in the optimized
x16 CPU path. Qwen has 512 experts. A private build expands both checked
arrays to 512 entries; installed engine source and libraries remain unchanged.
The repair passes canonical numerical checks for one-token decoding,
three-token batches, and 64-token prefill activating all 512 experts. Both
the original split and balanced split pass, as does unfused prefill. Cross-layout
comparisons cover 20889600 output values with maximum scaled error below
0.0000151. These are runtime correctness checks, not a model-quality evaluation.

GLM Full has been stopped through its user systemd service. The previous Qwen
configuration was restored after the failed trial. The repaired Q6 model
loads and generates coherent responses. The standard split with MTP2 measured:

| Workload | Generated tokens | tok/s | Adjusted memory GB/s |
| --- | ---: | ---: | ---: |
| Refrigerator explanation | 512 | 20.7383 | 126.5417 |
| Sorted-list merge function | 313 | 23.4560 | 128.4587 |

These use a 4096-token context setting and short fresh prompts. Memory traffic
is system-wide IMC traffic with the larger adjacent idle baseline subtracted;
it remains an attribution estimate. A prior run was slower during a Chrome
workload consuming about 14 CPU cores. The 28 tok/s target is not yet met.

Arithmetic/capital checks and all eight generated-code behavior cases pass.
Three of four cached/fresh continuation cases match exactly. The extension
case changes quote placement around `make cold` and then reconverges. Its
numerical-versus-cache cause remains under investigation; exact cache parity
is not claimed.

The balanced split with MTP3 and 15 workers per socket measured 21.3522 tok/s
on prose and 25.3206 on code. Adjusted memory traffic was 120.6645 and
124.4658 GB/s. Reducing to 12 workers measured 18.7295 and 25.4096 tok/s,
so the worker control was restored to 15. These do not reach 28 tok/s.

An eight-second Q6 MTP3 cycle profile attributes 13.62% of sampled CPU cycles
to dense x16 Q8 GEMV, 6.63% to x16 Q6 GEMV, and 11.42% to Q8 8x8 GEMM/GEMV.
OpenMP waiting accounts for a substantial share of samples. Cycle shares
across all workers are not removable wall-time shares.

The MTP4 measurement was interrupted through its own HTTP client after a
severe slowdown. Partial output was coherent, but no completed decode timing
was returned. A bounded diagnostic measured only 1.5963 tok/s while Chrome
consumed about 18 CPU cores. One busy Chrome process's main thread was pinned
to CPUs 8-15, overlapping Qwen worker cores. The model thread audit shows
workers pinned to 0-14/16-30/32-46/48-62, with other threads allowed on all 128
logical CPUs; a pinned model controller on CPU 15 was not established.
This is a contended diagnostic, not a fair
MTP3/MTP4 speed comparison. Chrome was not modified during that diagnostic. Subsequent benchmark
starts require 20 seconds below four background CPU cores in addition to the
model-idle gate, and requests have an elapsed-time limit.

Two alternate integer accumulation schedules passed 18720 bit-exact kernel
comparisons but gave no consistent speed benefit, so neither was deployed.
Batching Q8 dense rows together subsequently passed 168480 bit-exact kernel
comparisons across actual Qwen widths, activation counts, tails, and the
temporary-buffer heap boundary. The private CPU build also passes 432 exact
off/on graph cases with 1/4/15 threads and ordinary/padded activations. Focused
streaming tests improve by roughly 22-43%; this is not a model speed claim.
The earlier balanced MTP3 trial with this private Q8 build loaded on port 18095. Its live
arithmetic/factual checks and generated merge function's eight behavior cases
pass. Cache-extension token equality still fails: in this run `add cold to`
and `create cold;` diverge after 21 common continuation tokens. Forced-prefix
one-token probes both choose `add` but report different probability values;
the cause has not been isolated. The cache checks are complete: repeat,
one-token trim, and three-token trim match exactly; extension does not.
Consequently the combined runtime correctness result is marked failed, even
though the arithmetic and generated-code checks pass. The later wider-Q8 MTP4
configuration also passes all eight code cases and the repeat/trim cache cases,
while explicit cache extension still fails. Prompt reuse is now disabled by
default; the explicit-reuse failure remains recorded and is not claimed fixed.

The first quiet-host speed test waited through sustained unrelated CPU load and
was canceled before sending any requests. A proposed CPU-affinity helper refused
the process before making any change. The user subsequently authorized closing
the busy Chrome instance. Its verified browser process received SIGTERM, and all
11 processes in that instance exited. The subsequent MTP3 and MTP4 results are
listed above. A separate one-core build temporarily delayed MTP4's background
gate, then exited; the run proceeded without changing the four-core threshold.
The fused-batching test permits up to five background cores and still records
adjacent idle memory-traffic baselines and unrelated CPU activity.

Evidence:

- [Verified download](results/qwen-q6-download-0907/status.json)
- [Weight traffic inventory](results/qwen-higher-quant-bandwidth-inventory-0907.json)
- [Four-line kernel repair](results/qwen-q6-expert-capacity-0907/private-cpu/x16-expert-capacity.patch)
- [512-expert numerical checks](results/qwen-q6-512-expert-validation-0907/result.json)
- [Service transition record](results/qwen-q6-trial-0907/state.json)
- [Standard split MTP2 measurements](results/qwen-q6-mtp2-standard-0907b/result.json)
- [Code and cache checks](results/qwen-q6-mtp2-standard-0907b-quality/result.json)
- [Balanced MTP3 measurements](results/qwen-q6-mtp3-even-0907/result.json)
- [Twelve-worker measurements](results/qwen-q6-mtp3-even-t12-0907/result.json)
- [Q6 CPU cycle profile](results/qwen-q6-mtp3-profile-0907/result.json)
- [Contended MTP4 diagnostic](results/qwen-q6-mtp4-regression-profile-0907/result.json)
- [Q8 batching kernel checks](results/qwen-q8-chains-batch-shapes-0907/result.json)
- [Q8 batching build and graph checks](results/qwen-q6-q8-batch-0907/result.json)
- [Live Q8 batching runtime checks](results/qwen-q6-q8batch-runtime-0907-quality/result.json)
- [Authorized Chrome close](results/chrome-close-0907.json)
- [Second authorized Chrome close](results/chrome-close-0907b.json)
- [Q8 batching MTP3 after Chrome close](results/qwen-q6-mtp3-q8batch-chromeclosed-0907/result.json)
- [Q8 batching MTP4 after Chrome close](results/qwen-q6-mtp4-q8batch-chromeclosed-0907/result.json)
- [Fused Q8 batching build and numerical checks](results/qwen-q6-q8-fused-batch-0907/result.json)
- [Fused Q8 batching MTP3 measurement](results/qwen-q6-mtp3-q8fused-chromeclosed-0907/result.json)
- [Q8 expert-kernel component probe](results/qwen-q8-expert-x16-probe-0907/result.json)
- [Q8 expert selector build](results/qwen-q6-q8-experts-0907/result.json)
- [512-expert Q6/Q8 numerical checks](results/qwen-x16-q8-experts-q6-512-0907/result.json)
- [512-expert Q8/MTP numerical checks](results/qwen-x16-q8-experts-q8-512-0907/result.json)
- [Q8 expert full-model run, 16 workers](results/qwen-q6-mtp3-q8experts-t16-0907/result.json)
- [Q8 expert full-model run, 15 workers](results/qwen-q6-mtp3-q8experts-t15-0907/result.json)
- [Additional dense Q8 shape probe](results/qwen-q8-dense-extra-probe-0907/result.json)
- [Additional dense Q8 build](results/qwen-q6-q8-dense-extra-0907/result.json)
- [Real tensor-name selector checks](results/qwen-q6-q8-dense-extra-0907/selector-check/result.json)
- [Additional dense Q8 full-model run](results/qwen-q6-mtp3-q8dense-extra-0907/result.json)
- [Q6 call-stack and thread profile](results/qwen-q6-dense-extra-callgraph-0907/result.json)
- [Third authorized Chrome close](results/chrome-close-0907c.json)
- [Atomic barrier build and validation](results/qwen-q6-simple-barrier-0907/result.json)
- [Atomic barrier full-model measurement](results/qwen-q6-mtp3-simplebarrier-spin1000-0907/result.json)
- [Atomic barrier follow-up profile](results/qwen-q6-simplebarrier-profile-0907/result.json)
- [Confidence-limited MTP4 measurement](results/qwen-q6-mtp4-pmin03-q8batch-0907/result.json)
- [Confidence-limited MTP4 repeat](results/qwen-q6-mtp4-pmin03-q8batch-repeat-0907/result.json)
- [Confidence-limited MTP5 measurement](results/qwen-q6-mtp5-pmin03-q8batch-0907/result.json)
- [Wider Q8 component checks](results/qwen-q8-chains-wide-0907/result.json)
- [Wider Q8 private build and graph checks](results/qwen-q6-q8-wide-batch-0907/result.json)
- [Wider Q8 full-model measurement](results/qwen-q6-mtp4-pmin03-q8wide-0907/result.json)
- [Wider Q8 14-worker comparison](results/qwen-q6-mtp4-pmin03-q8wide-t14-0907/result.json)
- [Fourth authorized Chrome close](results/chrome-close-0907d.json)
- [Final Q6 runtime measurement](results/qwen-q6-final-q8wide-0907/result.json)
- [Explicit reuse correctness check, extension failure retained](results/qwen-q6-mtp4-pmin03-q8wide-0907-quality/result.json)
- [Default reuse-off correctness checks](results/qwen-q6-default-cache-off-0907-quality/result.json)
- [Persistent Q6 promotion and Q2 backup](results/qwen-q6-promotion-0907/result.json)

The estimated Q6 active weight traffic is 6.963900416 GB per ordinary raw token.
At 93% of 380 GB/s this gives a weight-only ceiling of 50.75 tok/s; it excludes
runtime overhead and extra state traffic. It is not a measured decoding rate.
