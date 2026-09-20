# GLM dispatch and scheduling investigation

No setting from this investigation is promoted. The approved production service is
healthy as PID 1756372 on CPU library SHA-256
`4793379e6894a9286168f79c4f323985388ec0b0ac65e54013dcd2957488af1f`.
All original helper affinities, command arguments, inference environment and mapped
library hashes are restored/unchanged. No additional model outage occurred or is
pending. The completed Q4 batch experiment remains a regression pass without a
reliable throughput gain; see [its report](Q4-BATCH-WINDOW-RESULT.md).

## Actual Codex scheduling evidence

A read-only `/proc` observer recorded an 18.422-second generation window at the
normal 15 workers/socket. Pinned model threads used 56.287 logical CPU-seconds per
wall second; unbound threads used 1.499. Pinned run-queue wait was only 0.310 seconds
against 1,036.906 seconds of CPU execution. Most unbound CPU time was system time,
concentrated in the main thread and eight helpers. The preserved meta backend's
20,000-microsecond polling loops call `sched_yield()` after a short hard spin, which
is a plausible contributor. No user-space call stacks were sampled, so this is not
a proven attribution of those threads' system time.

The observer itself consumed 1.492 CPU seconds and individual samples took up to
0.112 seconds. Its 14.439 tok/s request is observational, not an uninstrumented speed
benchmark. Kernel perf permissions did not allow perf counters; no security settings
were changed.

A separate reversible comparison temporarily confined the main thread and eight
helpers to physical cores 15, 31, 47 and 63. All four actual Codex outputs were exactly
the same 270-token response with 3,994 cached prompt tokens.

| Helper affinity | Decode tok/s | Helper run-queue seconds |
|---|---:|---:|
| Original, before | 14.653 | 0.030 |
| Four selected cores, first | 13.242 | 19.243 |
| Four selected cores, second | 13.356 | 19.308 |
| Original, after | 14.638 | 0.089 |

The affinity restriction lost about 9% and was fully reverted. These cores are unused
by GLM's pinned 15-worker teams, but they are not generally idle machine resources.
The original scheduler observation already recorded 82.51 seconds of nice-priority
CPU work across those cores and their SMT siblings during 18.422 seconds of wall
time. A later process-name/affinity sample identified active Godot and container
workloads there. They were left untouched. This confounds claims about an intrinsic
16-worker threshold and about spare-core helper affinity. It does not establish the
cause of the earlier full-model 16-worker sweep, which lacked comparable observation.

## Standalone dispatch controls

The probe links the exact production CPU, base and ggml libraries. It constructs
separate target-like and draft-like contexts, each split across four CPU-NUMA devices,
with Q4 matrices, residual/SiLU/RMS operations and fused all-reduces. Each measured
cycle executes the target once and draft twice. Weights are reused and graphs omit
real attention, routing and cache behavior: these are synchronization controls, not
full-model performance predictors.

There were 47 successful cases and one preserved setup failure. The failure was an
8 GiB virtual-address limit during allocation, not an output mismatch. The meta
allocator reserves large tensor-metadata arenas. Only the probe runner was adjusted
to a 24 GiB virtual limit with a sampled 6 GiB resident-memory abort guard; observed
resident memory stayed below 273 MiB. Every run also checked production was idle,
aborted on traffic, and verified its task IDs were unchanged.

All successful output fingerprints matched by graph shape. The final 21 cases also
saved complete target/draft output bytes and compared them exactly across settings,
covering three shapes and 1,213 measured cycles. Each cycle checked final target and
draft output bytes against the warmed reference. Earlier fingerprint-only cases are
not presented as saved-byte comparisons.

Reducing dispatcher polling from 20 ms to zero greatly reduced system CPU time.
Long-graph timings at 15 workers did not establish a robust speed benefit. Tests with
`GOMP_SPINCOUNT=0` also did not resolve the observed 16-worker slowdown.

## Why the apparent barrier gain is not deployable evidence

The installed CPU library has an optional `GGML_CPU_OMP_SIMPLE_BARRIER=1` branch.
An exploratory probe added a retained 128-thread OpenMP team to imitate a possible
loader pool. GNU OpenMP changes its busy-wait behavior when managed OpenMP threads
exceed the CPU count; ordinary HTTP worker threads do not establish that condition.
See [GNU libgomp's spin-count documentation](https://gcc.gnu.org/onlinedocs/libgomp/GOMP_005fSPINCOUNT.html).

| 15-worker long-graph comparison | Existing barrier, ms | Optional barrier, ms | Speed ratio |
|---|---:|---:|---:|
| Artificial retained OpenMP team | 66.180 | 46.717 | 1.417x |
| No artificial OpenMP team | 53.756 | 55.264 | 0.973x |

Values are the mean of two run medians. Aggregate cycle ratios were respectively
1.424x and 0.961x. The apparent 40% gain disappeared in the control without the
artificial team.

Production's 128 early unbound threads were created within 0.09 seconds of startup.
The server source defaults to 127 HTTP workers on this host, plus its listener.
Together these observations support HTTP workers as that group; this is an inference,
not a stack-based identification. There is no evidence for a retained 128-thread
OpenMP loader team in production. The production observer's low pinned system CPU
also differs from the artificial-team probe's substantial system CPU overhead.

Later 16-worker cases were slow both with and without the artificial loader pool
while other applications occupied the additional four physical cores. Therefore the
initial attribution of the slowdown to the loader pool was premature. The recorded
results do not identify a production OpenMP threshold, a deployable barrier gain, or
a validated route to 18+ tok/s. No full-model barrier outage is staged or requested.

## Evidence

- [Standalone case summary](dispatch-probe/summary.json), [summarizer](dispatch-probe/summarize.py), [probe](dispatch-probe/probe.cpp), [runner](dispatch-probe/run.py).
- [Final production audit](dispatch-probe/final-runtime.json), [thread evidence](dispatch-probe/production-thread-evidence.json), [current background CPU sample](dispatch-probe/background-cpu.json).
- [Live observer](appserver-scheduler-production15.json), [completed affinity experiment](helper-affinity-window.json).

The known cached/fresh output inconsistency remains unresolved. Current validated
Codex performance remains approximately 14–15 tok/s, with workload-dependent higher
coding results reported earlier. No claim of 18+ is supported.
