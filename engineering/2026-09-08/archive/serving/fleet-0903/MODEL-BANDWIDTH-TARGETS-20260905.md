# Model speed at the requested memory bandwidth

Latest user correction, September 7: the user never instructed us to keep Full
running. The assistant incorrectly attributed that restriction to the user;
it is withdrawn. Model selection uses the whole server with loading as needed,
not a requirement to fit alongside Full. The latest Qwen target is higher
precision than the installed Q2 and more than 28 generated tok/s. Historical
occupancy restrictions in the log below are not current user requirements.

September 7 execution update: Full is stopped and Qwen UD-Q6_K_XL is selected.
MTP4 measured 28.23 tok/s on code, with a 27.92 repeat. The selected wider Q8
batching runtime measured 28.46 code / 21.53 prose after a 28.10/21.32 run;
MTP3's best prose result is 22.99. Q6 has
therefore exceeded 28 on this code workload, while a general 28 tok/s floor
has not been established. The selected MTP4 measurement used approximately
137 GB/s after adjacent idle traffic was subtracted, about 36% of 380 GB/s.
The Q6 launcher preset is saved, with the Q2 configuration backed up. Prompt
reuse is off by default to avoid the observed cached-extension mismatch.
85% memory-bandwidth utilization has not been demonstrated. See
[the Q6 trial report](QWEN-HIGH-QUANT-20260907.md) for evidence and limitations.

Latest comparison: [September 7 model choices](MODEL-CHOICES-20260907.md) adds
the user-proposed Q4 Flash candidate. Its header inventory gives 14.246 GB of
estimated packed target weights per raw token: 22.7 tok/s at 85%, 24.8 at 93%,
and 26.7 at 100%. No Q4 Flash model was loaded or benchmarked, and its quality
has not been qualified. Historical IQ2 numbers below remain artifact-specific.

September 7 requirement: GLM-5.3-Flash must preserve near-lossless quality;
Qwen3.8-Flash-Next is the option for additional speed. The historical IQ2
benchmarks and projections below do not establish an acceptable Flash serving
configuration. Q8 is the primary candidate for quality evaluation; Q6 requires
evidence that it meets the same requirement. Quality takes precedence over
speed and bandwidth targets. See [the current quality policy and capacity
check](GLM-FLASH-QUALITY-POLICY-20260907.md). That assessment changed no service
or launch configuration; its earlier permanent-Full constraint is withdrawn.

Latest measured status, September 6: the unchanged Full service reached
15.703 tokens/s on a short, exact file edit using an 18-token draft limit and
0.75 confidence threshold. Two edits passed exact whole-file checks under both
request profiles; a third added an unrequested class under both profiles and
is excluded from successful-throughput claims. The passing 18-token cases
used approximately 10.55-12.01 GB/output token, giving conditional bandwidth-only
projections of 29.4-33.5 tok/s at 93% (353.4 GB/s), or 31.6-36.0 at 100%.
These are favorable source-copy/edit workloads, not general reasoning rates.

The fresh 512-token Full reasoning windows measured 7.89-9.64 generated tok/s
and approximately 20.65-24.94 GB/generated token. Their conditional projections
remain 14.2-17.1 tok/s at 93%, or 15.2-18.4 at 100%. The replay comparison and
its quality/counter audit are detailed at the end of this report. No model has
a verified 304 GB/s decode result, and Full has not achieved 93% during decode.

Assume 380 decimal GB/s is available to one model across all four sockets.
85% is 323 GB/s; 90% is 342 GB/s; 93% is 353.4 GB/s; 95% is 361 GB/s.

| Installed model | Estimated packed target weights per raw token | At 85% | At 90% | At 95% |
| --- | ---: | ---: | ---: | ---: |
| GLM-5.3-Flash, UD-IQ2_XXS | 10.784 GB | 30.0 tok/s | 31.7 tok/s | 33.5 tok/s |
| Qwen3.8-Flash-Next, UD-Q2_K_XL | 4.870 GB | 66.3 tok/s | 70.2 tok/s | 74.1 tok/s |
| GLM-5.3 Full, UD-Q4_K_XL with current load-time requantization | 28.106 GB | 11.5 tok/s | 12.2 tok/s | 12.8 tok/s |

These are weight-only projections for batch-one decode, with one model using
the whole server. The calculation is bandwidth divided by estimated active
weight bytes per target token. They are not achieved rates or direct DRAM
measurements. The separate active optimization goal is >=80%, or 304 GB/s,
for each model; its corresponding projections are 28.2, 62.4, and 10.8 tok/s.

The inventory uses model metadata: GLM Flash selects 8 of 288 experts across
45 target layers; Qwen selects 10 of 512 across 48 target layers; Full selects
8 of 256 across 78 target layers. Dense and shared weights are included.
The estimate accounts approximately for the current repacked layouts and
Full's load-time attention/shared-expert/output requantization. It does not
simulate every per-NUMA tensor shape or runtime dispatch decision.

Qwen's smaller active weight stream explains its higher bandwidth projection.
Total model file size is not the number of bytes read for each MoE token.
KV/state accesses, activations, duplicated reads, socket replication, cache
reuse, and speculative draft/verification traffic can change actual bytes
per emitted token. MTP throughput needs a separate measurement; multiplying
these rates by the draft length would not account for its costs or acceptance.

## What 85% requires

If weight reads themselves sustain 380 GB/s, reaching an effective 323 GB/s
leaves the following budget for additional work that cannot overlap the reads:

| Model | Time per token at 323 GB/s | Weight-read time at 380 GB/s | Remaining time |
| --- | ---: | ---: | ---: |
| GLM Flash | 33.39 ms | 28.38 ms | 5.01 ms |
| Qwen Flash-Next | 15.08 ms | 12.82 ms | 2.26 ms |
| GLM Full | 87.02 ms | 73.96 ms | 13.05 ms |

This is an optimistic accounting model, not a proof that 85% is attainable.
Quantized arithmetic, attention/state operations, scheduling, and socket
synchronization must fit within that allowance or overlap memory accesses.
CPU utilization alone does not establish bandwidth saturation. Intel's
[CPU roofline documentation](https://www.intel.com/content/www/us/en/docs/advisor/get-started-guide/2023-0/identify-bottlenecks-using-cpu-roofline.html)
distinguishes compute and memory ceilings and calls for measured operations,
traffic, and timing.

## Evidence and current status

- Inventory: `results/bandwidth-roofline-inventory.json` and
  `bandwidth-roofline-inventory.py`. Older rough 36/76/11.5 tok/s estimates
  were superseded by the packing and expert-count audit.
- `dram_bandwidth.py` validates all 48 read/write counters: six channels on
  each of four sockets. The successful live idle pilot returned complete,
  unmultiplexed counters and 17.80 GB/s of systemwide background traffic.
- A 23.55-second passive Full observation measured 16.98 GB/s systemwide
  while slot 319 was processing a prompt, with zero decoded tokens. It is
  not a decode measurement and cannot establish model-attributable bandwidth.
- A separate six-second passive cycle profile during that prompt put 58.67%
  of sampled cycles in `ggml_backend_meta_fused_reduce_op`. Disassembly places
  almost all of that function's samples in its arrival/completion spin loops.
  This establishes substantial waiting in that prompt sample, not the decode
  bottleneck or a guaranteed speedup from removing a barrier.
- The attached Full benchmark passed its normal-reasoning arithmetic and
  geography checks and completed all four raw/MTP measurements. The two preceding
  setup attempts failed a truncated answer and a forced-zero-reasoning wrong
  answer respectively; neither produced a bandwidth measurement.
- The attempted GLM Flash raw server was cancelled during loading when an
  independently started Qwen server appeared. No GLM counter sample completed;
  the dependent MTP waiter was stopped. The existing Qwen service was subsequently
  measured with MTP2. No old Flash pipeline remains queued.
- The client separates a stable decode window from prompt processing and records
  background before and after it. IMC counts are systemwide; background
  subtraction is an attribution estimate.

### Full: first controlled counter measurements

`results/glm53-full-bandwidth-baseline-0905c/result.json` completed successfully,
with 512 generated tokens per sample, no competing inference, no guard abort,
and complete hardware counters. The server's response timing supplies token
rate; only complete counter intervals inside the stable streamed decode window
supply bandwidth. Subtracting the larger mean of the before/after background
windows gives the following attribution estimates:

| Full workload | Measured tok/s | Gross system GB/s | Adjusted GB/s | Adjusted / 380 | Approx. GB per output token | Projection at 323 GB/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw prose | 5.76 | 181.89 | 169.85 | 44.7% | 29.48 | 10.96 tok/s |
| Raw code | 6.97 | 217.71 | 207.00 | 54.5% | 29.68 | 10.88 tok/s |
| MTP2 prose | 8.50 | 204.18 | 195.06 | 51.3% | 22.94 | 14.08 tok/s |
| MTP2 code | 9.41 | 207.02 | 203.54 | 53.6% | 21.62 | 14.94 tok/s |

The measured raw traffic suggests about **10.9 raw tok/s at 85%**, refining
the 11.5 weight-only estimate above. With the measured MTP2 traffic and
acceptance pattern, the projection is **14-15 tok/s at 85%**, or 15.7-16.7 at
95%. Draft acceptance was 282/457 on prose and 307/408 on code. These are
conditional extrapolations; the utilization target has not been achieved.
Background changes and the slightly different timing/counter windows limit
the precision of derived bytes per token.

A September 6 source/profile audit narrows the meaning of "raw" here: these
requests set the maximum offered draft length to zero, but run on an MTP-enabled
server. After target decode, `common_speculative_process` still updates the
draft context. The zero-draft profile also contains matrix work in the later
thread pool. These results therefore include draft-context maintenance; they
are not measurements of a separately launched server with speculation absent.
The MTP2 measurements and their projections are unaffected by this clarification.

All four sockets carried similar raw decode traffic: about 45-46 GB/s each
on prose and 54-55 GB/s each on code. The low average utilization is therefore
not explained by one inactive memory socket in these samples. No cycle
profiler overlapped the controlled bandwidth measurements.

At the user's requested higher utilization, the same measured traffic gives:

| Full mode | 93%, 353.4 GB/s | 95%, 361 GB/s | 100%, 380 GB/s |
| --- | ---: | ---: | ---: |
| Raw | 11.9-12.0 tok/s | 12.2 tok/s | 12.8-12.9 tok/s |
| Current MTP2 | 15.4-16.3 tok/s | 15.7-16.7 tok/s | 16.6-17.6 tok/s |

These assume unchanged traffic and draft acceptance. They are single-stream
decode projections, not measured improvements or universal maximum rates.
20 tok/s at 353.4 GB/s would require at most 17.67 GB/output token, another
18-23% traffic reduction from these MTP2 samples as well as higher bandwidth.

In a simplified accounting model where reads run at 380 GB/s and additional
work cannot overlap them, the 93% target allows only 4.28-4.54 ms of that
additional work per MTP output token. Quantized arithmetic, attention,
routing, draft processing, and synchronization must fit within that budget
or overlap memory access. The bandwidth-only projection therefore does not
prove 93% is attainable for this workload. Better speculation or weight
reuse can raise tokens/s while reducing bytes per output token; batching
also changes this single-stream comparison.

### Qwen: attached MTP2 counter measurements

`results/qwen-existing-mtp-bandwidth-0905b/result.json` completed two 256-token
samples with arithmetic/geography checks passed, no competing inference, no
guard abort, and complete counters. The existing PID 2308651 / port 18095
matched the pinned Qwen runtime environment at measurement time.

| Qwen sample | tok/s | Gross system GB/s | Adjusted GB/s | Adjusted / 380 | Approx. GB/output token |
| --- | ---: | ---: | ---: | ---: | ---: |
| MTP2 prose | 11.29 | 68.80 | 52.02 | 13.69% | 4.608 |
| MTP2 code | 12.78 | 67.69 | 54.62 | 14.37% | 4.274 |

The host was contended by other work, including a Chrome process averaging
about 14.5 CPU cores. This is not a quiet-host throughput baseline or evidence
of regression from the earlier 20+ tok/s runs. Background-adjusted traffic
implies conditional rates of 66.0/71.1 tok/s at 304 GB/s and 70.1/75.6 at
323 GB/s on prose/code respectively; neither target is achieved. The first
attached attempt cancelled its own request when user work queued, before a
complete bandwidth sample. That failed attempt is preserved separately.

### Validated decode profiles and current experiment

`results/glm53-decode-profile-0905c` and
`results/qwen-existing-mtp-decode-profile-0905d` contain successful six-second
cycle profiles. Explicit monotonic timestamps verify that every sample lies
inside the observed generated-token window; profiler setup failures remain
separate. No cycle profiler overlapped the counter benchmarks above.

- Full raw: Q5/Q4 kernels accounted for 30.43%/15.80% of sampled cycles;
  the listed OpenMP wait symbols total 25.36%, and fused NUMA reduction 9.14%.
- Full MTP2: Q5/Q4 accounted for 28.67%/22.19%; the listed OpenMP wait symbols
  total 24.59%, and fused NUMA reduction 6.80%.
- Qwen MTP2: the listed OpenMP wait symbols total about 56.7%; Q5/Q8 x16
  kernels account for 5.45%/4.40%. Its profiled throughput was 19.41 tok/s,
  under different host conditions from the counter run.

Sampled CPU-cycle percentages do not equal removable elapsed time. Full's
Q5 work motivates a direct comparison of its current kernel with the newer
compact Q5 implementation on representative Full matrix sizes. The first
attempt built both fixture executables, then cancelled its partial first arm
when Qwen inference became active; it supplies no comparative timing verdict.
The revised runner, `run-full-q5-kernel-comparison.py`, checkpoints each complete
six-arm comparison and retries an interrupted case after another idle gate.
Result `full-q5-kernel-comparison-0905b` completed all 48 cases with identical
canonical weight digests and packed-output hashes across implementations, plus
native-reference numerical tolerance. Its median Full/compact ratio on the
eight primary cases was 0.7743: the broad replacement is not selected.

An initial 1.51x advantage on the large three-token attention matrix did not
survive longer sampling. `full-q5-kernel-comparison-0905c` used 1,000 graph
repetitions per arm and checked both standard and padded inputs for two dense
shapes. Full/compact ratios were 0.997/0.957 on 4096x6144 and 0.800/0.922 on
2048x4096. All four cases matched exactly, all compared library hashes remained
unchanged, and no run was interrupted. No compact Q5 kernel was ported to Full.

Qwen's recorded cycle periods were also grouped by sample-time CPU. All four
sockets show roughly 59-62% of cycles in libgomp when all its samples are
included. Non-leader workers total 63.98% libgomp; socket leaders total 18.42%
libgomp and 19.29% fused reduction. Auxiliary threads contribute less than
0.5% of total sampled cycles. Evidence is in the valid Qwen profile's
`draft2/cycles-by-cpu.json` and `draft2/cycles-leaders-workers.json`. This
supports investigating graph scheduling around the leaders, rather than
assuming auxiliary idle thread teams account for the waits.

The next opt-in Qwen source candidate declares already single-worker fused
reductions as single-task graph nodes. This permits the existing executor to
skip barriers between such a reduction and a following single-task operation.
Larger reductions retain their existing task count. The flag is
`GGML_CPU_NUMA_FUSED_REDUCE_TASK_HINT`, default off; pinned server binaries are
unchanged. `run-qwen-reduce-task-hint.py` builds only the private runtime and
compares graph outputs with the flag off/on, including two/four sockets,
inactive partitions, merged reductions, and sizes above/below the existing
single-worker threshold. `qwen-reduce-task-hint-0905` completed 32 graph cases,
160 changing-input iterations per arm, with identical off/on output hashes.
The separate timing run completed eight cases with 1,000 repetitions per arm
in off/on/on/off order. Its median ratio was 1.01360, with individual ratios
0.969-1.044. Outputs remained exact after repeated execution. This small and
mixed gain does not justify selecting the flag for a model trial; it remains
disabled. These tests establish graph behavior, not full-model bandwidth.

### Full: request-level speculative draft sweep

`glm53-draft-sweep-bandwidth-0905/result.json` completed all six 512-token
measurements with no error or competing inference. The existing server's
ngram/MTP configuration was preserved; only each request's maximum draft
length was changed. Both normal-reasoning correctness checks passed.

| Requested draft limit | Prose tok/s | Prose adjusted GB/s | Code tok/s | Code adjusted GB/s |
| --- | ---: | ---: | ---: | ---: |
| 1 | 8.80 | 204.70 | 9.05 | 208.38 |
| 3 | 7.42 | 191.43 | 8.85 | 198.05 |
| 4 | 6.69 | 186.97 | 8.17 | 174.76 |

Draft acceptance on prose fell from 228/282 at limit one to 305/614 at three
and 314/783 at four. More drafting did not improve either throughput or
bandwidth in this sweep. The maximum adjusted result was 208.38 GB/s, 54.84%
of 380. Background subtraction was larger on the final code sample (13.59
GB/s); comparisons across different runs remain sensitive to host conditions.
The earlier MTP2 result remains 8.50/9.41 tok/s and 195.06/203.54 GB/s. No
server default was changed.

A separate process-scoped CPU counter capture, `glm53-cpu-counters-0905`,
completed after the sweep. `analyze-cpu-counters.py` validated all five event
records for every thread, with complete active-thread counters running 100%.
Both roughly six-second collection intervals lie inside streamed decode.

| Full CPU sample | Instructions/cycle | Load page-walk active cycles | Retired L3 load misses / 1,000 instructions |
| --- | ---: | ---: | ---: |
| Raw | 0.377 | 1.211% | 18.97 |
| MTP2 | 0.556 | 1.049% | 12.14 |

IPC includes spinning. Walk-active cycles measure page-walk activity, not
stall time, and retired L3 misses exclude prefetch traffic. Among threads with
at least one billion recorded cycles, the maximum walk-active fraction was
1.473% raw and 1.230% MTP2. These observations do not support page walks as
the main explanation for the bandwidth gap.

Full's current environment leaves the small-reduction worker limit unset
(zero), unlike Qwen. Its existing library was checked with one worker per
socket below a size cutoff, using Full's NUMA environment and without rebuilding
any production library. `glm53-reduce-single-worker-0905` passed 32 graph
cases and 160 changing-input iterations per arm with exact off/on outputs.
The initial 65,536-element timing run improved the 6,144-wide cases but
regressed 49,152-element reductions. A 32,768-element cutoff excluded those
larger cases; its separate eight-case, 1,000-repetition ABBA run had only a
1.00551 overall median ratio and 1.02027 on the 6,144-wide cases. The earlier
large gains did not repeat, so neither setting is selected for deployment.

`glm53-precise-load-misses-0905` subsequently completed a separate MTP2
capture of `mem_load_retired.l3_miss:upp`, including data addresses. All samples
were inside decode and none were lost. Q5 and Q4 x16 matrix kernels accounted
for 56.15% and 37.05% of recorded load-miss event weight. Precise annotations
locate the misses at packed-weight loads, including Q5 high-bit loads and
Q4 low-nibble loads. The ordinary cycle profile often placed samples at a
following activation broadcast; those cycle positions alone did not identify
the missed data. Annotations and `hot-load-instructions.json` are retained.

### Sustained read-only calibration, September 6

`results/local-read-bandwidth-0906/result.json` completed four ten-second
read samples on the same 60 physical worker cores used by Full. Each worker
had a private 32 MiB buffer bound to its local NUMA node, for 1.875 GiB total.
All 3,840 sampled physical page locations per arm matched their intended
nodes. Independent AVX-512 integer accumulators consumed every byte; initial
and repeated-read checksums matched exactly. This is a hardware calibration,
not a model run or progress toward the model-attributable bandwidth target.

| Read mode, paired order | Logical bytes read / elapsed, GB/s | Adjusted IMC reads, GB/s | Adjusted IMC reads + writes, GB/s |
| --- | ---: | ---: | ---: |
| Cached loads, first | 384.40 | 370.70 | 363.70 |
| Streaming-load instruction, first | 382.00 | 379.20 | 379.48 |
| Streaming-load instruction, second | 377.84 | 375.68 | 379.21 |
| Cached loads, second | 378.69 | 375.44 | 375.29 |

All 48 hardware counters were complete and unmultiplexed. Each result uses
14-15 complete half-second intervals wholly inside the common worker window,
excluding one second at each boundary. Subtracting the larger before/after
background read rate gives 371-379 GB/s, 97.6-99.8% of the stated 380 GB/s.
The total column independently subtracts total background; changing background
writes can therefore make adjusted total traffic lower than adjusted reads.
The separate timing windows and small cache reuse limit comparison with the
logical byte rate, which was 0.6-3.6% above adjusted reads. Gross read traffic
was balanced at roughly 93-97 GB/s per socket.

Both inference services remained idle and untouched. Other host work remained
active, including Chrome. No prefetcher MSR, global memory policy, power
setting, or production environment changed. The result supports the server's
ability to supply the user's requested 353.4 GB/s read stream. It does not
establish that a complete autoregressive model can sustain it between graph
operations and socket synchronization.

### Full x16 kernels with a cold weight stream

Three subsequent private experiments completed: `cold-q5-prefetch-0906`,
`cold-q4-prefetch-0906`, and `cold-q5-three-token-0906`. They use the same 60
workers and private local weight pools, passing successive 64-row tiles through
the existing library. Q5 uses K=4096; Q4 uses K=6144. The three-token fixture
calls the current GEMV path for three activation rows while reusing each tile.
This isolates matrix arithmetic and memory access from model graph barriers,
expert routing, input quantization, and cross-socket reductions.

Every arm compared every tile on three changing activation inputs against
the untouched Full library, with exact output equality, then checked the
checksum of every repeated timed pass. All page-placement checks and counters
passed, no competing inference was detected, and compared source/library
hashes stayed unchanged. Complete source snapshots and binaries are retained.

| Existing Full kernel | Paired median logical unique-weight GB/s | Adjusted IMC read GB/s, two arms |
| --- | ---: | ---: |
| Q5, one activation row | 373.80 | 384.02 / 348.23 |
| Q4, one activation row | 383.72 | 383.82 / 379.20 |
| Q5, three activation rows | 299.33 | 303.88 / 301.18 |

The logical column counts each unique weight tile once per pass, including
when three activation rows reuse it. Hardware and logical timing windows
differ slightly. Host load drift and background subtraction explain part of
the Q5 single-row spread. One prefetch arm's after-background read/write
traffic jumped to 73.81/57.75 GB/s, making its adjusted counter result unsuitable
for a precise prefetch comparison; its logical tile throughput is retained.

The one/two/four-block-ahead Q5 prefetch variants had paired median logical
throughput ratios of 0.9856 / 0.9850 / 1.0003 against the same arithmetic
compiled without prefetch. Q4's one-block variant was 0.9910; Q5's three-token
variant was 1.0087. None is selected. No production source, library, or
configuration was changed.

These measurements show that the existing single-row Q5/Q4 kernels can
approach the physical read ceiling when supplied with continuous independent
work. They also show that weight reuse across three activation rows increases
arithmetic per unique weight byte and lowers the achieved DRAM rate to about
300 GB/s in this Q5 case. A high bandwidth percentage alone does not establish
maximum token throughput. Full's 93% throughput estimate remains conditional,
not a demonstrated or guaranteed operating point.

Saved Full cycle profiles were additionally grouped by thread-ID cohort and
sample-time CPU (`draft*/cycles-by-thread-pool-0906.json`). The earlier pool
accounts for 92.78% of raw-request cycles and 88.95% of MTP2 cycles; the later
pool accounts for 7.18% / 10.95%. Total libgomp cycle shares are 29.34% / 27.97%,
including unresolved symbols identified by their library. Only 4.79% / 6.38%
of total cycles are libgomp in the later pool. The main waiting cost therefore
does not come solely from an unused second pool. Both pools currently pin
workers to the same 60 logical CPUs; this observation alone does not prove
avoidable concurrent work or translate cycle shares into removable elapsed
time.

### Finite Full matrix graphs and controller correction, September 6

`cold-graph-check.cpp` now exercises eight Q5 projections through Full's
unchanged four-device NUMA backend. Each uses the attention-output dimensions:
global K=16384, 6144 output rows, with K=4096 on each socket. Each projection
feeds RMS normalization, requiring a completed cross-socket reduction. The
566.23 MB packed pool exceeds the combined last-level caches. Three changing
inputs are checked against independent native dot products for every output
row; timed outputs must remain exactly equal to the last checked result.
This is a synthetic graph, not a complete model or a decode-bandwidth result.

`cold-full-output-graph-chunks-0906c` completed six arms with one activation
row and chunk limits 64/32/16/16/32/64. Paired median times for the entire
eight-projection graph were 2.28065 / 2.25020 / 2.26520 ms, respectively.
The 1.35% and 0.68% changes are too small to select a different chunk limit.
Two preceding setup attempts failed before timing while establishing the
correct device-discovery affinity; neither contributes performance evidence.

`cold-full-output-graph-grouping-0906` completed six arms grouping one, two,
or four independent projections before their normalization consumers. Fused
reduction boundary counts were verified as eight, four, and two, with the
same eight tensors and exact output hashes. Paired speedups were 1.0220 and
1.0548 for groups two and four. No runtime change is selected: the synthetic
graph does not establish that equivalent independent groups exist in Full.

Both comparisons above restricted the fixture controller to CPU 63. That
restriction differs from production Full, which already permits controller
execution on reserved CPUs 15, 31, 47, and 63. The subsequent paired
`cold-full-graph-controller-0906` comparison measured 2.24772 ms with CPU 63
alone versus 2.01153 ms with all four reserved cores, a 1.1174x fixture
speedup. This corrects the benchmark; it is not a production improvement.
Future graph runs default to the four reserved controller cores.

All controller arms verified the complete local weight allocation after
reference checks: each node had 34,612 bound 4 KiB pages and zero nonlocal
pages in those mappings. Every output/hash check and all 48 IMC counters
passed, without competing inference. The four-controller arms recorded
257.07 and 206.43 GB/s of adjusted reads. A background spike affected the
second arm, so these adjusted counters cannot support a precise speedup
estimate; graph timing and counter attribution remain separate measures.

`cold-q5-scale-range-0906` additionally compared normal FP16 block scales
with nonzero FP16 subnormal scales in four continuous-kernel arms. Its
paired logical throughput ratio was 1.00085, with exact checks against the
existing kernel. This control does not support scale-conversion slowdown
as the cause of the finite-graph gap. No floating-point mode was changed.

The finite graph remains slower than the independent continuous-kernel
stream. Controller placement explains part of the original fixture gap;
these tests do not assign the remaining difference entirely to barriers.

### Three-row Full graph comparison, September 6

Two further experiments used three activation rows and all four reserved
controller cores. `cold-full-three-token-graph-chunks-0906` completed six
eight-second arms in 64/32/16/16/32/64 order. The graph times were:

| Chunk rows | First graph time, ms | Second graph time, ms |
| --- | ---: | ---: |
| 64 | 2.81810 | 3.85726 |
| 32 | 2.59116 | 3.75770 |
| 16 | 3.90886 | 3.81363 |

The paired median favored 32 over 64 by 5.14%, but substantial drift across
the run prevents treating that ratio as an isolated chunk-size improvement.
Adjusted reads ranged from 137.43 to 196.99 GB/s. These remain graph-only
measurements; no full-model bandwidth result was collected.

The longer `cold-full-three-token-graph-repeat-0906` ran twelve-second arms
in 32/64/64/32 order. Graph times were 2.69716 / 2.78429 / 4.73437 / 2.65700
ms. The first adjacent comparison favored 32 by 3.23%, while the second
64-row arm was much slower than the other three arms. Its adjusted reads
were 23% above the declared unique-weight stream, further limiting traffic
attribution. All samples are retained; the resulting 1.4043x paired median
must not be presented as a chunk-size speedup. No chunk change is selected.

All ten arms passed the three-input native references, exact final-output
checks, identical hashes across chunk sizes, complete IMC counters, and
inference guards. The longer run verified 34,716 bound 4 KiB pages per node
with zero nonlocal pages in every arm. Full source/library hashes remained
unchanged. These results led to the resident, interleaved comparisons below.

No model has yet been verified at 304 or 323 GB/s of model-attributable decode
traffic. Production PID 4005448 / port 18091 and the independently started
Qwen PID 2308651 / port 18095 are preserved. Benchmark requests
and temporary test servers yield to production work; no production restart or
global configuration change is part of these measurements.

The final audit, `results/finite-full-graph-0906-final-state.json`, confirms
all eight graph/setup/control jobs are terminal, no owned fixture or runner
remains active, and both protected services retain their original PIDs and
ports. Both services were idle with empty queues at that observation.

### Resident interleaved graph comparisons, September 6

`run-interleaved-full-graph.py` keeps two private weight pools loaded, then
executes one fixture at a time in balanced ABBA/BAAB order. Each experiment
has sixteen four-second phases, with completed warmups and settling between
phases. Both fixtures retain Full's four-device backend and four reserved
controller cores. Every phase verifies exact final outputs and checks the
inactive fixture's process CPU time. Only complete counter intervals within
the central two seconds contribute to that phase's bandwidth estimate.

Both completed runs passed every output, hash, NUMA-placement, counter,
source/library-integrity, and inference-isolation check. The inactive fixture
used zero measured CPU in all thirty-two timed phases. This validates the
comparison method; these remain synthetic graph measurements.

| Activation rows | Pairs favoring chunk 32 | Geometric mean paired speedup | Median paired speedup | Median adjusted reads, chunk 64 / 32 |
| --- | ---: | ---: | ---: | ---: |
| 3 | 8 of 8 | 1.05651x | 1.05586x | 191.60 / 199.48 GB/s |
| 1 | 6 of 8 | 0.84862x | 1.00965x | 260.91 / 260.91 GB/s |

`full-three-row-interleaved-0906` supports further investigation of chunk 32:
every adjacent pair favored it, by 3.77-7.65%. Median graph times were 2.75701
and 2.60492 ms for chunks 64 and 32. It does not establish a model gain.

`full-one-row-interleaved-0906` had a large timing disturbance: one chunk-32
phase took 8.86650 ms, compared with roughly 1.9-2.5 ms for the other phases.
All samples are retained. Its median pair difference is about 1%, while the
geometric mean is dominated by the disturbance. No global chunk setting is
selected from these results.

### Private Q5 shared-decoding candidate, September 6

Full's dense x16 executor currently calls the Q5 GEMV kernel separately for
each activation row. `cold-q5-batch.h` implements a private candidate that
shares weight decoding across two or three rows, retaining each row's integer
sum and per-block floating-point multiply-add order. It uses Full's existing
packed layout and has not been added to a production library.

`cold-q5-shared-decode3-0906` completed four continuous-kernel arms in
Full/candidate/candidate/Full order, using K=4096 and three activation rows.
All tiles on three changing inputs matched the unchanged library exactly;
NUMA-placement, checksums, counters, and inference guards passed.
Paired median logical unique-weight rates were 304.566 GB/s for Full and
295.319 GB/s for the candidate, a ratio of 0.96964. This version is not selected.

Static disassembly of its three-row kernel contains sixty instructions with
vector stack operands. A less-unrolled version was therefore compared in
`cold-q5-shared-decode3-loop-0906`. It passed the same exact-output, placement,
counter, and isolation checks. Median logical rates were 307.979 GB/s for
Full and 309.758 for the candidate, a ratio of 1.00578. Its three-row function
shrunk from 6,623 to 2,907 bytes and has no vector stack operands. Removing
that stack traffic recovered the initial slowdown but did not establish a
material gain. Neither version is selected for a runtime port. The two-row
implementation has not been validated by these three-row experiments.

### Completion-counter spacing in private Full libraries, September 6

Full's fused reduction stores the per-device completion counters in a single
aligned array of four-byte atomics. The four active sockets therefore update
the same cache line. `build-private-counter-layout.py` stages a copy of that
source and builds two private base libraries: four-byte spacing and 64-byte
spacing. Arithmetic and atomic memory ordering are unchanged. Both versions
reuse the other original object files without modifying them. Build commands,
source patch, headers/object hashes, library hashes, and loaded library paths
are recorded. These are test libraries, not a deployed Full configuration.

`full-counter-layout-one-row-0906` completed sixteen interleaved phases.
Seven of eight pairs favored separate cache lines; geometric mean speedup
was 1.01965 and paired median 1.01948. Median graph times were 1.97304 versus
1.93243 ms, with median adjusted reads of 261.47 versus 266.27 GB/s.
All numerical, placement, counter, source-integrity, and inference-isolation
checks passed. The inactive fixture used zero measured CPU in every phase,
and process maps confirmed each fixture loaded its intended private library.
`full-counter-layout-three-row-0906` also completed sixteen phases with all
checks passed. Seven of eight pairs favored padding, but the geometric mean
gain was only 1.00573x, with paired median 1.00535x. Median packed/padded graph
times were 2.77429 / 2.75403 ms; median adjusted reads were 190.84 / 192.16 GB/s.
Both fixtures again had exact outputs, verified private-library mappings,
local weights, and zero measured CPU in the inactive fixture.

The counter-layout effect is small in these finite graphs and does not
explain most of the gap from continuous kernel streams. No production
library or service has changed. A separate cycle profile of the finite
graph can locate the remaining time before selecting further runtime work.
No model has yet been verified at the all-three 304 GB/s objective.

The latest audit is `results/interleaved-kernels-counters-0906-final-state.json`.
All six experiments are terminal and no owned fixture or runner remains
active. Forty-nine original engine source/header/object/library hashes
were verified unchanged. Full PID 4005448 / port 18091 and Qwen PID 2308651
/ port 18095 remain intact; both were idle with empty queues at the audit.

### Finite-graph cycle profiles, September 6

`full-one-row-finite-cycles-0906` and `full-three-row-finite-cycles-0906`
profiled the private eight-projection Q5 graph at chunk limit 64, with all
four reserved controller cores. Cycle collection attached only to each
private fixture; no IMC collection overlapped it. Profiled timing is excluded
from throughput comparisons. Both runs passed native references, exact final
outputs, local weight placement, source/library hashes, and inference guards.

`analyze-finite-graph-cycles.py` validates every trace line's target PID,
timestamp, physical CPU, and positive cycle period. It also requires the
recorded selected sample count and zero reported lost samples. The one-row
and three-row traces contain 97,080 and 97,163 selected samples respectively,
each inside an eight-second central window of a ten-second graph phase.

| Activation rows | Q5 GEMV | OpenMP library | Fused NUMA reduction | Other |
| --- | ---: | ---: | ---: | ---: |
| 1 | 69.681% | 20.019% | 7.789% | 2.512% |
| 3 | 69.290% | 22.305% | 5.727% | 2.678% |

These are shares of period-weighted user cycles, not wall-time fractions or
estimates of removable overhead. The complete OpenMP-library total is used
here; summing only the report's individually displayed functions understates
it. Among nonleader worker CPUs, OpenMP accounts for 21.075% and 23.547% of
each cohort's cycles; among socket-leader CPUs it accounts for 6.834% and
6.320%. Reserved controller CPUs contribute only 0.568% and 0.442% of all
sampled cycles. All remaining sampled threads also execute Q5 multiplication;
there is no separate compute-free worker cohort dominating these fixtures.

Worker Q5 shares range from 66.24-72.89% in the one-row profile and
66.77-73.03% in the three-row profile. This is not evidence of a whole idle
socket or a few entirely unused workers. Chunk-queue operations themselves
account for only 0.260% and 0.154% of sampled cycles. Changing work granularity
can still affect load balance, cache reuse, and waiting outside that function;
its performance effect requires a separate unprofiled comparison.

### Larger chunks in the three-row graph, September 6

`full-three-row-large-chunk-0906` compared chunk limits 64 and 256 in sixteen
balanced, resident four-second phases. For local K4096 x 6144 and fifteen
workers, limit 256 makes the existing formula choose 208 rows: thirty work
items per matrix instead of ninety-six. A full 208-row Q5 chunk occupies
599,040 packed bytes. Arithmetic, weights, and activation inputs are unchanged.

All eight adjacent pairs favored the original limit 64. Candidate speed ratios
ranged from 0.81495 to 0.89715, with geometric mean 0.85068 and median 0.84965.
Median graph time increased from 2.94695 to 3.48237 ms. Median adjusted reads
were 178.62 / 174.34 GB/s for limits 64 / 256. These rates are physical-counter
estimates after background subtraction; declared unique-weight bytes are a
separate quantity and cannot substitute for them. The larger limit is rejected
for this graph. It supplies no basis for raising a production-wide chunk cap.

Every phase passed exact outputs, complete counters, local placement, and
source/library-integrity checks. The inactive fixture used zero measured CPU
in all sixteen phases. Both fixtures and all owned build processes exited
normally. No production change was made.

The final audit, `results/finite-cycles-chunks-0906-final-state.json`, verifies
both profiles and the chunk comparison are terminal, all recorded owned
processes have exited, and forty-nine original engine files retain their
hashes. Protected Full PID 4005448 / port 18091 and Qwen PID 2308651 / port
18095 remain intact and were idle with empty queues at that observation.
The all-three 304 GB/s model goal remains active and incomplete; Full's best
adjusted model measurement remains 208.38465 GB/s. The 93-100% speed table
above remains a conditional projection, with no new achieved model speed.

### Qwen expert split geometry, September 6

The current Qwen process still uses the pinned `validated-iq-batch3-bin`
libraries, four sockets, fifteen workers per socket, and split policy 13.
Its CPU and model libraries match their current build-tree copies; its
pinned base library differs from the later build-tree base library. Private
expert tests therefore link to the pinned directory and verify loaded library
paths, preserving the exact deployed base/CPU combination.

The model inventory contains 48 gate/up/down expert triplets. Every down
projection is IQ4_NL with shape [640,2560,512]. Gate/up shapes are
[2560,640,512], with IQ2_XS in 47 layers and IQ3_XXS in layer 2. Expert gate/up
row splits follow the down projection's quantization alignment. The existing
FFN policy increases that alignment to 128 elements. Equal socket fractions
then produce slices 128/128/128/256, with the larger slice rotating by layer.

`qwen-expert-split-geometry-0906c` evaluated the deployed library's split
function on weight-free metadata for all 144 expert tensors. Every returned
split had three 128-element slices and one 256-element slice; gate/up use
axis 1 and down uses axis 0. The four rotated arrangements each occurred
36 times. This directly tests the split function and does not attach to or
reconfigure the server. The successful check passed isolation and all original
input/library hash checks. The first attempt was stopped while waiting, before
any build; the second had a private fixture compile error. Neither produced
model or component timings.

At 128 rows, the current 16-row-aligned static gate/up partition gives work
to eight of fifteen socket workers. A 256-row slice uses all fifteen but
assigns two row groups to one worker. An even 160-row slice uses ten workers.
These facts motivate both socket balance and later within-socket work sharing;
they do not by themselves predict a model speedup.

### Qwen gate/up split comparisons, September 6

`qwen-expert-graph-check.cpp` compares the original 128-element split policy
with 32-element alignment, giving four 160-row slices. It uses the unchanged
pinned Qwen IQ2 r16 kernels, gate/up fusion setting, and batch3 setting. The
graph has independent expert gate/up projections with K2560, width640,
32 stored experts, and 10 selected per activation row. Adjacent activation
rows shift routes by two experts, exercising one-, two-, and three-row reuse.
Each set of four projections rotates the larger original slice evenly across
sockets. This is a component fixture, not a complete Qwen layer or model.

Both layouts check every output against native IQ2 dot products and SwiGLU
on three changing inputs and route sets. Timed phases require exact equality
to the last checked output; weight and output hashes must match across layouts.
The existing interleaved runner now supports this fixture, distinguishes total
packed allocation from the selected unique-weight stream, verifies local
placement and pinned library mappings, and retains all inference guards.

| Eight-projection graph | Pairs favoring even split | Geometric mean speedup | Median paired speedup | Median adjusted reads, original / even |
| --- | ---: | ---: | ---: | ---: |
| One activation row | 6 of 8 | 1.08145x | 1.07470x | 29.32 / 42.28 GB/s |
| Three activation rows | 8 of 8 | 1.18727x | 1.16804x | 58.91 / 75.25 GB/s |

The completed runs are `qwen-expert-even-split-one-row-0906b` and
`qwen-expert-even-split-three-row-0906`. Every output and guard check passed,
with zero measured CPU in the inactive fixture across all thirty-two phases.
Maximum native-reference errors were 7.45e-9 and 1.86e-8. The initial one-row
setup failed on private mirrored-input metadata before measurement; it remains
separate from the successful run.

The eight-projection weight pools have substantial cache reuse: adjusted
reads are well below the declared unique-weight rate. These are preliminary
speed comparisons and cannot establish model DRAM utilization. Larger
working-set comparisons are needed before selecting the split change.

`qwen-expert-even-split-cold-one-row-0906` doubled the graph to sixteen
projections, with 904,396,800 packed bytes per resident fixture and a
282,624,000-byte selected unique-weight stream per graph pass. All eight
paired comparisons favored equal slices: geometric mean speedup 1.11036x,
median paired speedup 1.11567x. Median original/even graph times were
2.06077 / 1.86885 ms; adjusted read medians were 98.40 / 117.07 GB/s.
The read-to-declared-stream median ratios rose to 0.712 / 0.763. This provides
a comparison with more DRAM traffic, although it is still not a pure uncached
stream or a model measurement. All sixteen phases passed every output,
placement, counter, source-integrity, and isolation check; inactive-fixture
CPU was zero throughout.

`results/qwen-expert-runtime-0906.json` verifies the live server's equal
1,1,1,1 tensor fractions and tensor split mode. The host exposes 1 MiB L2
per physical core and 22 MiB shared L3 per socket. L2 and L3 reuse helps
explain why the smaller component's declared weight stream should not be
treated as its DRAM traffic.

`build-private-qwen-split.py` prepares an optional private policy,
`GGML_Q4E_EXPERT_EVEN_SPLIT=1`, restricted to four-device Qwen4exp expert
tensors with width640 and 32-element down-projection quantization blocks.
It retains the quantization alignment and 16-row gate/up alignment while
removing the extra 128-element rounding. The default policy remains unchanged.
The helper writes a private source copy and library. Complete target and MTP
expert-path numerical coverage subsequently passed, as recorded below. An
actual model trial remains required before adopting this policy. The initial helper had an output-path
bug; its incident and correction are recorded below.

`qwen-expert-even-split-cold-three-row-0906` completed the sixteen-projection
three-row comparison. All eight pairs favored equal slices: geometric mean
1.12564x, median paired speedup 1.13292x. Median original/even graph times
were 3.11289 / 2.77190 ms, with adjusted read medians 110.25 / 125.64 GB/s.
Median read-to-declared-stream ratios were 0.863 / 0.875. Every numerical,
placement, counter, source-integrity, and isolation check passed, with zero
measured CPU in the inactive fixture. The larger one-row and three-row
comparisons support an 11-13% component improvement; no model improvement
has yet been measured.

### Private Qwen policy build and corrected output-path incident

The first policy build, `qwen-expert-even-split-policy-0906`, contained a
helper bug: while normalizing shared-library input paths, it also rewrote
the linker's `-o` argument to the pinned libllama path. It replaced that file
on disk. The run failed before candidate geometry validation and is retained
as an error result; it is not a successful candidate build.

The original file was restored from the build-tree copy, whose SHA-256 had
already matched the captured original. The restored hash is
`f14c91f2db56f7e1bf0d88e4494595d3ca34b816e4b5dec0cacaecaa8520b98e`.
The replacement output was preserved in the failed run's private directory.
Hashing the running Qwen process's mapped library through `/proc/2308651/map_files`
confirmed the same original SHA-256: its in-memory library backing file
retained the original bytes throughout. Both protected services kept their
PIDs/ports and responded with idle slots and empty queues after restoration.
Evidence is in that run's `restoration.json`,
`service-state-after-restoration.json`, and
`mapped-library-after-restoration.json`. No service was restarted.

The helper now excludes the output argument from dependency-path rewriting
and checks both compiler and linker outputs against the private destination
before execution. `check-private-qwen-build-output.py` exercises the actual
recipe transformation using a writer that refuses any output outside its
temporary directory; this check passed. The successful corrected build is
`qwen-expert-even-split-policy-0906b`. Its six build/check commands exited
normally. Disabled candidate metadata exactly matched all 144 original
splits; enabled metadata produced [160,160,160,160] for every expert tensor,
with unchanged split axes. The metadata fixture now includes layer 2's actual
IQ3 gate/up types. All recorded original inputs retained their hashes in the
corrected run. The optional library has not been deployed to a model service.

### IQ3 numerical coverage and final audit

`qwen-expert-even-split-iq3-check-0906` completed the four-projection,
three-row IQ3_XXS comparison. All eight phases passed exact output checks,
complete hardware counters, local NUMA placement, and inference isolation.
Inactive-fixture CPU was zero in every phase. Both split layouts produced
the same weight and output hashes; maximum error against the native reference
was 2.98023224e-7. Its 13 recorded source/library inputs were rechecked after
completion. This validates the tested IQ3 gate/up split geometry; it does
not validate down projections or execution of the private policy in a model.

The four paired timings were mixed: geometric mean 1.01759x, median paired
speedup 1.00830x, and two of four pairs favored equal slices. Median original
and even graph times were 1.02011 and 1.05726 ms. This small graph is a
numerical coverage check, not evidence for selecting a performance change.

`results/qwen-expert-split-0906-final-state.json` records the closing audit:
all 11 current Qwen experiments are terminal, including the four failed or
interrupted setup/build attempts; none of their 41 recorded runner/child PIDs
remain. All 308 recorded original engine inputs match their current contents.
This is a check of restored/original contents, not a claim that the pinned
Qwen library was unchanged throughout the earlier build incident. A fresh
mapped-file hash confirms that running Qwen still has the original library.
Full PID4005448/port18091 and Qwen PID2308651/port18095 were responsive,
idle, and had empty queues at the audit. Neither service was restarted.

No model has a verified 304 GB/s decode result. Full's best adjusted model
measurement remains 208.38465 GB/s (54.84% of 380 GB/s). Its current MTP2
traffic still projects to 15.4-16.3 tok/s at 93% or 16.6-17.6 tok/s at 100%;
these conditional projections do not establish the highest attainable speed.

### Complete Qwen expert paths with the actual split policy

The `expert_moe_split` mode extends the existing component fixture through
gate/up, SwiGLU, IQ4_NL down projection, and a residual add that requires
combining the socket partial sums. The original arm loads the pinned libllama;
the candidate loads the already validated private libllama with even splitting
enabled. Both use the pinned ggml arithmetic libraries and the actual exported
model split callback. The fixture verifies the expected split axis and slices
for every weight and exercises all four split rotations. It stores 32 experts,
selects 10 per activation row, and changes inputs and routes over three probes.
It does not include the full model, router, expert-weighted sum, attention,
or recurrent state. Its token dimension covers a component activation batch,
not a measured number of accepted speculative tokens.

Gate/up activations are checked against native quantized dots and SwiGLU.
Down outputs plus the residual are checked against native IQ4_NL/Q8_0 dots
using the gathered intermediate activations. Output arrays are also compared
between layouts, allowing 2e-5 * (1 + max(abs(a), abs(b))) for changed
floating-point reduction grouping. Repeated phases must remain bit-exact
within each layout. Packed weights, library mappings, local allocation,
hardware-counter completeness, input hashes, and inference isolation retain
the previous checks. Total resident packed pools stay below 2 GiB.

The first attempt, `qwen-expert-moe-split-three-row-0906`, stopped before any
timing: its fixture tried to retrieve an unmaterialized partial down output.
Adding the residual consumer forces the required reduction. The corrected
`qwen-expert-moe-split-three-row-0906b` completed all 16 timed phases with
eight IQ2 gate/up plus IQ4 down projections and three activation rows.
All eight pairs favored equal slices: geometric mean 1.26012x, paired median
1.24999x. Median graph time was 3.77300 / 2.97668 ms (original/even).
Adjusted read medians were 69.23 / 86.91 GB/s; adjusted read-plus-write
medians were 75.99 / 95.17 GB/s. Median read-to-declared-stream ratios were
0.877 / 0.866, so cache effects remain in this component measurement.

The 1,843,200 output values differed by at most 4.76837158e-7 across layouts;
maximum scaled difference was 2.03643644e-7. Maximum native down-reference
error was 2.38418579e-7 / 4.76837158e-7. Both arms exercised eight fused
reduction boundaries. Every timed phase passed, with zero measured CPU in
the inactive fixture. All 19 outer inputs and 256 private-build inputs
retained their hashes. This establishes a 26% improvement for the tested
complete expert path, without establishing model throughput or 304 GB/s.

The remaining target expert-path cases also completed:

| Case | Projections | Pairs favoring even | Geometric mean speedup | Median original/even graph time |
| --- | ---: | ---: | ---: | ---: |
| IQ2, one activation row | 8 | 8/8 | 1.30929x | 2.64793 / 2.07218 ms |
| IQ3, three activation rows | 4 | 4/4 | 1.18691x | 2.27454 / 1.88484 ms |
| IQ3, one activation row | 4 | 4/4 | 1.09045x | 1.39783 / 1.28293 ms |

These are `qwen-expert-moe-split-one-row-0906`,
`qwen-expert-moe-split-iq3-three-row-0906`, and
`qwen-expert-moe-split-iq3-one-row-0906`. Every phase passed the numerical,
mapping, placement, counter, input-integrity, and isolation checks, with no
measured CPU in the inactive fixture. Maximum cross-layout errors were
2.38418579e-7, 3.81469727e-6, and 3.81469727e-6 respectively. The smaller IQ3
graphs were substantially cached: median adjusted-read/logical-stream ratios
were approximately 0.50 for three rows and 0.19-0.22 for one row. Their
timings extend component coverage and do not measure model DRAM utilization.
The larger IQ2 cases support 26-31% improvement for these expert paths.

### MTP coverage discovered before model adoption

A header-only read of the installed draft model is recorded in
`results/qwen-mtp-expert-metadata-0906.json`. It declares architecture qwen4exp,
49 blocks with one next-token prediction block, expert width640, 512 experts,
and ten selected experts. Its three expert tensors in block48 are all Q8_0:
gate/up [2560,640,512] and down [640,2560,512]. Therefore the optional even
split also applies to the MTP draft model, beyond the IQ2/IQ3 target weights.

The fixture now supports this Q8_0/Q8_0 expert path with the corresponding
Q8_0 activations and native references. It includes block48's actual metadata
and equivalent rotations, checks the draft header and file identity, and
requires the deployed q8_0_8x8 repacked kernels. Both draft-path cases completed:

| Q8 draft case | Pairs favoring even | Geometric mean speedup | Median original/even graph time | Median adjusted reads, original/even |
| --- | ---: | ---: | ---: | ---: |
| One activation row | 4/4 | 1.12675x | 1.90286 / 1.70484 ms | 70.68 / 75.68 GB/s |
| Three activation rows | 4/4 | 1.17647x | 2.89281 / 2.43426 ms | 82.51 / 97.13 GB/s |

The runs are `qwen-expert-moe-split-q8-one-row-0906` and
`qwen-expert-moe-split-q8-three-row-0906`. Both used four projections and
completed eight phases. Maximum cross-layout absolute error was 1.52587891e-5
in both; maximum scaled error was 7.49214320e-6 against the 2e-5 tolerance.
Maximum native down-reference scaled errors were 1.26086697e-5 for the original
and 1.03452712e-5 for even slices. All repeated outputs remained exact within
each layout. All mappings, placement, counter, input-integrity, and isolation
checks passed, with zero CPU in the inactive fixture. The one-row case still
had considerable cache reuse; three-row read/logical medians were 0.830/0.811.

`results/qwen-expert-moe-split-0906-final-state.json` closes this round. It
verifies six completed target/draft cases, 64 valid timed phases, and
4,915,200 cross-layout output values. The one initial fixture setup failure
is retained separately with no timing phases. All seven runs are terminal,
none of their 27 recorded runner/child PIDs remain, and all 309 recorded
original engine inputs match their current contents. Saved source snapshots,
fixture binaries, output arrays, and the private policy source/object/library
also pass their recorded hashes. Both protected services kept their original
PIDs and ports and were responsive, idle, and unqueued at the audit. A fresh
mapped-file hash confirms Qwen still runs its original library.

The private policy is ready for an actual-model trial of the tested target
and draft paths. It has not been deployed. Such a trial still needs the single
Flash-model slot currently occupied by the independent Qwen service. No model
has a verified 304 GB/s decode result; Full's best adjusted sample remains
208.38465 GB/s. Component speedups do not satisfy the all-three-model goal.

### Fresh Qwen baseline and model-trial preparation, September 6

The existing Qwen service completed a fresh MTP2 baseline in
`results/qwen-even-split-existing-baseline-0906/result.json`. The guarded
idle interval was 61.20 seconds; arithmetic and geography checks both passed.
Both requests used a 512-token limit, temperature zero, seed 42, and disabled
prompt caching. The code response finished naturally at 325 tokens.

| Existing Qwen MTP2 sample | Output tokens | tok/s | Gross GB/s | Adjusted GB/s | Adjusted / 380 | Approx. GB/output token | Draft acceptance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Prose | 512 | 21.80687 | 106.67605 | 100.18915 | 26.37% | 4.59439 | 312/398 |
| Code | 325 | 24.17193 | 108.25977 | 103.74915 | 27.30% | 4.29213 | 212/228 |

There were 44 complete stable-decode intervals over 22.14 seconds for prose
and 23 over 11.59 seconds for code. Every interval in both complete captures
had all 48 counters running at 100%. Subtracted background was 6.48691 and
4.51062 GB/s respectively. All four sockets carried similar gross decode
traffic: 25.60-28.21 GB/s for prose and 26.59-27.44 for code.

No request was cancelled and no other inference process appeared or changed.
Full used only 0.17%/0.08% of one CPU in the recorded windows. Ordinary host
applications remained active: the leading Chrome process used about 0.92/0.64
CPU, and containerd, dockerd, and Paseo each used about 0.56-0.63 CPU. This is
an attached-service baseline under current host conditions, not an exclusive
machine benchmark. The earlier 11.29/12.78 tok/s Qwen samples had much heavier
Chrome contention; the difference does not establish an engine improvement.
IMC counters remain systemwide and background subtraction remains an estimate.

`audit-qwen-existing-baseline-0906.py` produced `validation-audit.json` beside
the result. It reconciled saved perf rows with the parsed per-socket byte
counts, recomputed the stable decode summaries, checked authoritative stream
timing and acceptance counts, and verified the five saved measurement-source
copies against their originals. It also verified all 21 pinned binary/library
entries, the private candidate library, and the original backing bytes mapped
by the running Qwen process. Both protected services retained their original
identities and ports, were idle and unqueued at the audit, and no measurement
or graph runner remained active. Port 18155 had no listener.

Before this baseline, `model_measurement_guard.py` added queue and process
identity monitoring to the attached measurement client. It checks target and
peer queues, additional active slots, inference-process changes, and peer CPU
activity every half second during the request. Incoming work or monitoring
failure closes only the benchmark's HTTP socket. Nine disposable localhost
HTTP checks passed, covering normal completion and eight cancellation cases;
the disposable server remained responsive in every case. Those checks sent
zero real model requests. Their evidence is
`results/qwen-even-split-model-trial-0906-staging/measurement-guard-check.json`.

The same staging directory records the exact original and candidate commands
and environments in `plan.json`. The actual pinned server's ELF loader was
checked with `LD_TRACE_LOADED_OBJECTS=1`: it resolves the private split-policy
libllama with the pinned CPU/base libraries. This did not execute the server
main function or load a model. The candidate changes only that private library
resolution and enables `GGML_Q4E_EXPERT_EVEN_SPLIT=1`; its trial endpoint is
localhost:18155. The original service uses port 18095 and all three aliases
`qwen-goal,qwen3.8-flash-next,flash-next`.

The recorded original working directory is the AI-Server root, and its main
process affinity is CPUs 0-127. A trial or restoration must preserve that
affinity; the benchmark controller's CPU 63 restriction must not leak into the
server launch. Exact configuration preparation and the attached baseline are
complete. The model lifecycle trial has not run or been queued; the independent
Qwen service still occupies the one-Flash-model slot. No candidate is deployed.
Full remains unchanged, with best measured adjusted bandwidth 208.38465 GB/s
and fastest saved MTP2 decode in this bandwidth series at that point of
9.41315 tok/s in a separate sample. Its current
MTP2 traffic still projects to 15.4-16.3 tok/s at 93% and 16.6-17.6 at 100%,
conditional on unchanged traffic and acceptance. The 304 GB/s model goal
remains active and incomplete.

### Qwen lifecycle controller completed; interruption decision pending

`qwen_split_trial.py` now implements an original/candidate/original model
comparison with restoration. Its default invocation is read-only and passed
against the current services and files. An explicitly authorized execution
would measure the original Qwen, wait for idle, terminate the exact original
process using a pidfd, test the private candidate, restore the original
configuration, and repeat the original measurement. It preserves the original
working directory, complete in-memory launch environment, main affinity,
port, and aliases. It never signals Full or overwrites the pinned libraries.

`check-qwen-split-trial.py` passed 17 checks with disposable localhost HTTP
processes. Ten transaction cases cover success and failures before/after
termination, candidate work, cancellation, cleanup, and the final measurement.
Four checks cover protected/reused/mismatched process identities and the
handoff of the restored service. Three exercise the production restoration
loop with injected idle-monitor failure, incoming peer work, and load-monitor
failure. All disposable processes were reaped; these checks sent zero
requests or signals to actual model services. The child launcher retained
CPUs 0-127 while the controller stayed on CPU 63. The first 14-case run passed;
the later run additionally covers restoration-monitor retry behavior.

`results/qwen-even-split-model-trial-0906-staging/lifecycle-check.json` and
`controller-preflight.json` record the evidence and source hashes. The
reviewable procedure and availability implications are in `TRIAL.md` in that
directory. A user decision was requested on temporarily interrupting the
independently running Qwen service, because the current measurement constraints
preserve it. No approval has arrived, no trial is queued, and no service has
been restarted. The controller will not be executed without that decision.

### Fresh Full MTP2 measurements while the Qwen decision is pending

`measure-full-bandwidth-0906.py` is a separate attached measurement client.
It keeps the successful Full baseline's default reasoning for arithmetic and
geography checks; its measured prose/code request payloads match the current
client. The original Qwen trial client and its verified hashes are unchanged.

`results/glm53-full-current-bandwidth-0906/result.json` completed after a
60.86-second idle gate. Both short checks passed. Each measured request
generated 512 tokens and stopped at the length limit:

| Full MTP2 sample | Generated tok/s | Gross GB/s | Adjusted GB/s | Adjusted / 380 | Approx. GB/generated token | Draft acceptance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prose | 7.88710 | 203.63402 | 196.66740 | 51.75% | 24.93531 | 267/486 |
| Code | 9.63604 | 206.02509 | 199.01980 | 52.37% | 20.65368 | 319/383 |

The new and earlier Full MTP2 windows contain reasoning tokens, with no final
answer content before the 512-token limit. These rates measure generated
tokens, including reasoning; they do not measure completed-answer speed.
Reasoning text differs between runs. Prose draft acceptance changed from
282/457 to 267/486, while code changed from 307/408 to 319/383. The new code
sample was the fastest Full decode sample in this prose/reasoning bandwidth
series, but this is an unchanged engine under a different generated-token
stream, not a measured engine
improvement. Full bandwidth remains near 200 GB/s under current host activity.

The stable windows contain 126/103 complete intervals over 63.29/51.74 seconds.
All 48 counters ran at 100% throughout both captures. Subtracted background was
6.96662/7.00529 GB/s. No request abort or inference churn occurred; Qwen used
0.18%/0.15% of one CPU. Host applications still ran: Chrome used roughly one
CPU, and an additional Python process used roughly one CPU in the code window.
The counters remain systemwide; background subtraction remains an estimate.

`audit-full-current-bandwidth-0906.py` reconciled the saved perf rows, socket
bytes, decode windows, stream timing, and all five measurement-source copies.
It verified Full's command/environment against the saved draft-sweep capture
and six recorded Full binary/library entries. Both protected services retained
their original identities, ports, and affinities and were idle and unqueued
at the audit. The first audit invocation used an older result schema without
runtime fields; it stopped before producing an audit. The corrected audit uses
the existing draft-sweep runtime reference and passed; it is saved beside the
measurement as `validation-audit.json`.

The new traffic gives conditional 93% projections of 14.17 tok/s for prose and
17.11 for code, or 15.24/18.40 at 100%. This widens the workload dependence of
the earlier 15.4-16.3 tok/s at 93% estimate. Acceptance and traffic must remain
unchanged for any such extrapolation; these are not achieved rates. Full's
highest adjusted model bandwidth remains 208.38465 GB/s, and all three models
remain below the verified 304 GB/s requirement.

### Full Q5 accumulator-chain candidates: no deployment selected

`cold-q5-two-chain-0906` and `cold-q5-asymmetric-chain-0906` each completed
six paired private component arms after a queue-aware 60-second idle gate.
The fixtures use 60 workers, 32 MiB per worker, K=4096, three activation rows,
and ten-second timed passes. The original one-chain candidate is unchanged.
The read-fixture runner now records protected process start identities and
uses the shared queue/CPU-aware idle guard; its source snapshots are retained
separately for each experiment.

| Q5 comparison | Full median logical GB/s | One-chain median | New candidate median | Candidate / Full |
| --- | ---: | ---: | ---: | ---: |
| Two low and two high chains per row | 308.949 | 312.241 | 298.621 | 0.96657x |
| Two low and one high chain per row | 308.202 | 309.890 | 311.008 | 1.00910x |

Both candidates passed the fixture's exact output checks on changing inputs
and every timed pass. All twelve arms completed with 3,840 local page samples
per arm, complete unmultiplexed counters, and no inference contention or churn.
The two-chain assembly has 26 static vector stack accesses, versus zero for
the one-chain kernel. The asymmetric version eliminates these accesses but
improves this component by only 0.9% over Full and 0.36% over one-chain. Neither
candidate is selected for a full-model change. These logical unique-weight
rates are component measurements, not model bandwidth or generated-token rates.

The two-chain experiment's arm 1 background rose from 4.09 to 80.43 GB/s and
arm 2 also had substantial background drift. Their adjusted IMC attribution
is unsuitable for a precise kernel comparison; all measurements are retained.
The saved analysis and assembly comparisons document the limitation.

### Full replay: faster exact edits and a workload-dependent 93% ceiling

Earlier project replay files in `serving/glm-sr950/devloop/results` already
recorded about 13.5 tok/s with an 18-token draft limit and 0.75 confidence
threshold. Thus earlier statements that 9.41 or 9.64 tok/s were Full's
"fastest saved" rate apply only to this prose/reasoning bandwidth series.
The historical replay script used substring quality checks and no IMC
counters. Its aggregate also counts all generated tokens, whereas llama's
per-request decode timing uses generated tokens minus one.

`measure-full-replay-bandwidth-0906.py` reused the original source and prompts
from `serving/glm-sr950/benchmark-replay.sh` with the current attached-request
guard. It sends request-only settings: temperature 0, seed 7100, low reasoning,
400-token limit, no prompt cache, and either n=2/p=0 or n=18/p=0.75. It does not
restart, reconfigure, or rebuild Full. Adjacent workload pairs alternate profile
order. The new validator requires exactly one Python code block, valid syntax,
the exact requested source edit, and a natural stop. Generated code is never run.

`results/glm53-full-replay-bandwidth-0906/result.json` completed all six requests
after 60.89 seconds idle:

| Edit | Draft limit / threshold | Generated tok/s | Adjusted GB/s | Approx. GB/generated token | Exact output |
| --- | --- | ---: | ---: | ---: | --- |
| Rename function | 2 / 0 | 10.80751 | 201.68139 | 18.66122 | Pass |
| Rename function | 18 / 0.75 | 15.70296 | 165.65088 | 10.54903 | Pass |
| Add specified docstrings | 18 / 0.75 | 13.59200 | 163.27971 | 12.01293 | Pass |
| Add specified docstrings | 2 / 0 | 10.61647 | 206.45858 | 19.44702 | Pass |
| Replace exception name | 2 / 0 | 10.96626 | 209.30898 | 19.08663 | Fail: extra class |
| Replace exception name | 18 / 0.75 | 13.95798 | 170.38281 | 12.20684 | Fail: extra class |

The first two edits return identical correct file content under both settings,
with no reasoning text. The third adds `class ConfigError(ValueError): pass`
under both profiles, violating the request to leave everything else unchanged.
This is preserved as two failures. The 209.309 GB/s sample therefore does not
establish a successful task at that bandwidth. The previous prose/reasoning
series maximum remains 208.385 GB/s, far below 304 GB/s.

For the two exact edits, n=18/p=0.75 improves decode speed by 45.3% and 28.0%
in this comparison, while reducing estimated traffic per output token by
43.5% and 38.2%. Their combined decode rate, using the sum of predicted_n-1
divided by total predicted time, is 14.514 versus 10.706 tok/s. Draft acceptance
is 171/192 and 187/224 at n=18, versus 121/122 and 132/140 at n=2. This is a
request-profile result on source-copy/edit workloads, not a general engine
speedup or evidence that longer drafts help ordinary reasoning. The earlier
p=0 prose/reasoning draft sweep regressed at larger limits.

For the current quantization, one request stream, and unchanged traffic and
acceptance patterns, bandwidth divided by bytes per output token gives:

| Full workload/profile | Approx. GB/generated token | At 93%, 353.4 GB/s | At 100%, 380 GB/s |
| --- | ---: | ---: | ---: |
| Earlier zero-offered-draft windows, including draft maintenance | 29.48-29.68 | 11.9-12.0 tok/s | 12.8-12.9 tok/s |
| Fresh MTP2 prose/reasoning windows | 20.65-24.94 | 14.2-17.1 tok/s | 15.2-18.4 tok/s |
| Two exact edits, n=2/p=0 | 18.66-19.45 | 18.2-18.9 tok/s | 19.5-20.4 tok/s |
| Two exact edits, n=18/p=0.75 | 10.55-12.01 | 29.4-33.5 tok/s | 31.6-36.0 tok/s |

These are conditional bandwidth-only projections, not achieved speeds, a
compute-inclusive ceiling, or a universal maximum. Different quantization,
context lengths, output predictability, draft acceptance, or concurrent requests
change the calculation. The replay decode windows contain only 8.04-17.08
seconds of complete counter intervals, while token rates use whole-request
decode timing. Burst delivery and these different windows make the byte/token
estimates approximate. The measured prompt processing adds 6.84-7.71 seconds
for the passing n=18 requests; decode tok/s does not include that delay.

The server's separate 371-379 GB/s local-read calibration establishes physical
read capability. Full must also perform quantized arithmetic, attention,
speculative verification, scheduling, and cross-socket reductions. Even the
isolated three-row Q5 kernel currently moves only about 300-311 logical GB/s.
There is no evidence yet that the full decode path can average 353.4 GB/s.
Faster useful generation can coincide with lower memory utilization, as this
replay comparison demonstrates. This follows the distinction between a memory
roof and attainable performance in the original
[Roofline model](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.pdf).

`audit-full-replay-bandwidth-0906.py` independently reconciled all six replay
captures and twelve component captures with their saved perf rows. All 48
counters ran at 100%. It checked replay timings, exact output comparisons,
source snapshots, unchanged Full command/environment, 309 original engine
hashes, and Qwen's original mapped library hash. The initial audit stopped
because Linux restricts access to the mapped inode; using a read-only privileged
hash completed that check. Both services retained their original identities,
ports, and affinities and were idle/unqueued at audit time. All 15 recorded
benchmark/fixture PIDs were absent. The audit is saved as
`results/glm53-full-replay-bandwidth-0906/validation-audit.json`.

This turn made progress through a measured request-profile improvement and
two rejected component candidates. No service was restarted or engine change
deployed. The Qwen interruption decision remains unanswered and its trial has
not run. The all-three-model 304 GB/s goal remains active and incomplete.

### Qwen expert row scheduling: useful alone, little extra with even splitting

The deployed IQ r16 fused gate/up path divides each expert's output rows
statically among fifteen workers. A 128-row slice has only eight 16-row tiles,
so seven workers receive no matrix rows for that expert. A 256-row slice also
has an uneven static allocation. This motivates sharing row tasks across the
selected experts, instead of retaining the same row range on every expert.

`build-private-qwen-task-rows.py` constructs a private `libggml-cpu` with an
opt-in `GGML_CPU_IQ_MOE_TASK_ROWS` setting. The candidate uses the existing
thread-pool work counter to claim disjoint expert/row tiles. Input quantization,
IQ arithmetic, gate/up fusion, SwiGLU, physical weight offsets, and down
projection remain the same. No additional workspace or weight allocation is
introduced. The candidate is restricted to IQ r16, K=2560, output slices at
most 256 rows aligned to sixteen, and at most three activation rows. The
default is off. Only the 16-row setting was tested; 32 and 64 are untested.

The private library is
`results/qwen-expert-task16-one-row-0906/private-cpu/libggml-cpu.so.0.22.0`,
SHA256 `658d56d6cf59c1602c1b7bb389d8231c2d70885e0aaf361256ee80873743198b`.
The builder writes a source patch, object, library, and complete input manifest
under that private directory. It verifies that the sole compiler/linker output
stays there and leaves all original inputs untouched. The compiler emitted an
existing unrelated const-cast warning in the router function.

`run-qwen-expert-task-rows.py` derives from the existing guarded interleaved
graph runner. It compares the pinned CPU library with the private one while
holding the tensor split identical within each pair. It checks protected start
identities and executables as well as ports, uses a 60-second idle gate, and
terminates only owned fixtures/builds on model activity or monitor failure.
Later cases additionally record ordinary host CPU activity. The prepared
split-only model trial and its inputs are unchanged.

Five experiments each completed eight four-second phases in balanced adjacent
pairs, with sixty workers and four reserved controller cores. The graphs
include fused gate/up, SwiGLU, down projection, and a mirrored consumer that
exercises the fused NUMA reduction. They exclude a full model's remaining
operations and use synthetic weights/routes. IQ cases use eight graphs; the
Q8 fallback case uses four to stay within the resident-memory budget.

| Format / activation rows | Tensor split | Paired geometric mean speedup | Pairs favoring private library | Median graph ms, pinned / private | Median adjusted reads GB/s, pinned / private |
| --- | --- | ---: | ---: | ---: | ---: |
| IQ2 / 1 | Existing uneven | 1.16201x | 4/4 | 2.41291 / 2.09206 | 60.07 / 72.51 |
| IQ2 / 3 | Existing uneven | 1.18731x | 4/4 | 3.84022 / 3.29470 | 66.87 / 80.15 |
| IQ2 / 3 | Prepared even | 1.02266x | 4/4 | 2.99708 / 2.90082 | 87.40 / 91.53 |
| IQ3 / 3 | Existing uneven | 1.18799x | 4/4 | 3.89553 / 3.31535 | 66.78 / 78.97 |
| Q8 / 3, IQ scheduler inactive | Existing uneven | 0.98489x | 1/4 | 2.73462 / 2.74335 | 88.07 / 87.36 |

The 16-19% component improvements with the current split do not combine
multiplicatively with the earlier even-split gains. The direct combination adds
only 2.3% in the tested three-row IQ2 graph, while the Q8 fallback comparison
is slightly slower. No combined CPU-library/model change is selected. The
previously prepared split-only candidate remains the next model trial; the row
scheduler is retained as a separately measured option.

All five cases matched outputs exactly across three changing input/route
probes, totaling 7,065,600 compared float values, and passed the native gate/up
and down references. Every timed phase and warmup ended with output identical
to its reference. The fixture checks at phase boundaries, not after every
inner graph execution; an early progress update overstated that coverage and
was corrected. Partial expert overlap exercises one-, two-, and three-row IQ
batch paths. The Q8 candidate logs confirm that the IQ scheduler stayed inactive.
All sampled bound pages were local to their intended nodes.

All forty timed phases have complete, unmultiplexed 48-counter captures. Their
stable windows contain three complete half-second intervals per phase. These
are short, systemwide counter samples with estimated background subtraction;
cache reuse and ordinary host activity affect attribution. In the first case,
read/declared-stream ratios span roughly 0.65-0.71. None of these rates is a
model-decode result. Host activity in the IQ3 case includes a Python process
near one CPU and a final-phase Node burst near 2.6 CPUs; all four adjacent
pairs still favor the candidate, without proving a universally quiet-host gain.

`audit-qwen-expert-task-rows-0906.py` reconciled the saved counter rows, phase
windows, timings, output files, runtime mappings, and source snapshots for all
five cases. It verified 549 original engine-file hashes, Qwen's original mapped
library backing hash, unchanged service commands/environments/affinities, and
the absence of all 22 recorded runner/build/fixture PIDs. Both original services
were idle and unqueued at audit time. Evidence is in
`results/qwen-expert-task-rows-0906-audit.json`. The private runner evolved
between cases; each measurement's saved version is checked against its own
recorded hash rather than incorrectly requiring it to equal the latest version.

### Qwen request draft overrides are unavailable in the loaded fork

`qwen-existing-draft01-bandwidth-0906` attempted an attached request-only
comparison with zero and one offered draft token. After the 60-second idle
gate, the arithmetic and geography checks passed. The first 512-token prose
request then returned 398 generated drafts and 312 accepted drafts, despite
`speculative.n_max=0`. The client stopped at its existing validity assertion
before recording a successful zero-draft measurement or attempting n=1.
The observed 21.86901 tok/s belongs to the unchanged MTP2 behavior.

The source explains the result: Qwen's `tools/server/server-schema.cpp`
explicitly places the per-request n_max, n_min, and p_min fields under `#if 0`.
Its slot draft-limit function also lacks Full's request-level cap. The loaded
MTP driver uses its launch configuration. The README's sample JSON containing
these fields does not establish support in this fork. Further request-only
draft sweeps against this server would not test the intended conditions.
Full's separately customized fork does implement the overrides used in its
successful measurements; this finding does not invalidate those Full results.

`audit-qwen-draft-request-capability-0906.py` verified the rejected capture,
all saved client sources, complete counters, authoritative draft counts,
source excerpts, and all 21 pinned Qwen binary/library entries. It deliberately
does not reconstruct adjusted bandwidth: the failed request's background
bounds were not saved before the assertion. Evidence is preserved in
`results/qwen-existing-draft01-bandwidth-0906/failure-audit.json`; the original
failed result remains unchanged and is not promoted to a successful benchmark.

At the final audit both original model services retained their identities,
commands, runtime environments, ports, and affinities, and were idle/unqueued.
No measurement or scheduling runner remained active. No engine change was
deployed and no model service was restarted. The earlier interruption question
remains unanswered; the prepared split-only Qwen trial has not run.

This goal turn made progress through the private scheduler comparisons and
the verified API limitation, which rules out an ineffective tuning route.
The full objective is unchanged: each of GLM Flash, Qwen Flash-Next, and Full
must independently demonstrate at least 304 GB/s of model-attributable decode.
No model has met that requirement. The goal remains active and incomplete.

### Exact IQ byte expansion: rejected because computation became slower

The preceding user-facing ceiling calculation was read-only and did not change
model performance. This continuation revalidated both protected services and
implemented a new private IQ2_XS/IQ3_XXS candidate. The staged Qwen lifecycle
preflight again returned `ready: true, execute: false`; its interruption decision
remains unanswered and no lifecycle was executed.

`build-private-qwen-iq-bytes.py` reuses the existing private compiler/linker
builder without changing it or the engine. The opt-in `GGML_CPU_IQ_R16_BYTES`
layout expands each paired-nibble LUT index into its exact unsigned byte value,
retaining the bias of 64, half multipliers, subblock scales, integer reduction
order, and floating-point FMA order. Allocation sizes and row/expert offsets
use the selected layout consistently. The r16 block grows from 2,208 to 4,256
bytes. The private library SHA256 is
`6853764320d762ab6cac75cd038714a9de6ce567c05fb672627f0f7a88c4131d`.

`run-qwen-iq-byte-rows.py` compares the original CPU library against this private
library with the prepared even split held constant in both arms. Each fixture
contains eight complete gate/up, SwiGLU, IQ4 down, and NUMA-reduction graphs,
using 60 workers and four controller cores. Each experiment starts after 60
seconds idle and runs eight balanced four-second phases. Actual resident
weights are 688,128,000 bytes in the original arm and 1,107,558,400 in the byte
arm; combined storage stays below 2 GiB. Per-arm byte accounting and NUMA
placement checks reflect that difference.

| Activation rows | Original / byte median graph ms | Geometric throughput ratio | Pairs favoring bytes | Original / byte adjusted reads GB/s |
| --- | ---: | ---: | ---: | ---: |
| 3 | 3.01045 / 3.58132 | 0.84676x | 0/4 | 86.68 / 123.54 |
| 1 | 2.11084 / 2.52875 | 0.82820x | 0/4 | 70.78 / 111.37 |

Thus byte expansion raises physical traffic but reduces throughput by about
15-17%. It is rejected for deployment. The three-row assembly removes all four
LUT shuffles from the inner loop, retains twelve dot-product instructions, and
has no vector stack accesses in either version. Removing those instructions
did not compensate for the larger weight stream in these complete paths.

The existing native fixture passed 1,920 cases across original/private
libraries, IQ2/IQ3, standard/padded inputs, one to nine activation rows, and
batch3 disabled/enabled. All 960 corresponding cross-layout case hashes are
identical. These include r8 fallback shapes as well as r16 shapes. The complete
expert fixtures additionally compare 2,457,600 floating-point outputs across
three changing inputs/routes with exact equality between layouts. Native
gate/up/down checks and exact timed phase-boundary checks pass; exactness is
not claimed for every inner timing iteration.

`audit-qwen-iq-bytes-0906.py` reconciled all 16 timed phases, all 48 unmultiplexed
IMC counters, source/binary/output hashes, per-arm byte counts, and NUMA page
placement. It verified 549 original engine input hashes and that all 19 recorded
runner/build/fixture PIDs were absent. Both original services retained their
identities, commands, runtime environments, affinities, and ports and were idle
and unqueued. Evidence is in `results/qwen-iq-bytes-0906-audit.json` and the
`qwen-iq-bytes-even-{three,one}-row-0906` result directories.

Counter windows remain systemwide with estimated background subtraction, and
cache reuse is visible: median adjusted-read/declared-byte ratios are 0.88/0.92
for the three-row original/byte arms and 0.69/0.83 for one row. These are
component measurements, not model-decode bandwidth. The goal remains active;
none of the three models has verified 304 GB/s of attributable decode traffic.

### Current Qwen profiles: OpenMP remains prominent across both worker groups

`profile-qwen-current-0906.py` uses the shared ModelMeasurementGuard, records
host and thread activity, verifies protected identities/commands/environments,
and profiles the existing launched MTP2 configuration without sending unsupported
speculative overrides. It passed a 60-second idle gate and the 391/Paris checks,
then sampled eight seconds of cycles during each of the current prose/code
requests. No model was restarted or reconfigured.

`results/qwen-current-mtp2-profile-0906/result.json` completed both captures.
Profiled request rates were 21.24197 and 23.85793 tok/s, near the latest
unprofiled 21.80687/24.17193 baseline. Prose generated 512 tokens and reached
the length limit; code generated 325 and stopped naturally. Draft counts and
acceptance match the baseline: 312/398 and 212/228. Sampling can affect timing,
and these diagnostic rates are not a new engine speedup or bandwidth result.

All 194,511 samples belong to Qwen and fall inside their observed decode
windows, with no reported loss. The complete recorded cycle periods give:

| Category | Prose cycles | Code cycles |
| --- | ---: | ---: |
| OpenMP library | 60.635% | 60.457% |
| Q8 matrix kernels | 8.612% | 8.731% |
| Q5 matrix kernels | 5.366% | 5.294% |
| IQ r16 matrix kernels | 3.973% | 4.313% |
| 4-bit LUT matrix kernels | 3.004% | 2.986% |
| tinyBLAS | 3.239% | 3.360% |
| Fused NUMA reduction | 1.220% | 1.112% |

Each socket contributes approximately one quarter of sampled cycles: the
observed range is 24.84-25.32%. Ordinary host work remained present, including
roughly two Chrome CPU cores during prose and one during code, plus other
processes. Full averaged only 0.1% of one CPU in both captures. The earlier
profile's complete OpenMP-library proportion was 61.063%; the previously
reported 56.7% summed selected displayed wait symbols. The new profiles do
not establish a material shift in that bottleneck.

The saved thread identities and affinities distinguish two groups of 56 pinned
workers. The earlier group contributes 74.47-75.26% of all sampled cycles and
executes IQ, Q5, Q8, and other matrix work. Its OpenMP samples account for
45.65-46.49% of all cycles. The later group contributes 18.50-19.32% of all
cycles, including actual Q8 matrix work; its OpenMP contribution is another
13.16-13.86%. Both groups perform matrix work. The large OpenMP total cannot
be assigned solely to a wholly unused worker pool.

Cycle proportions do not measure removable wall time. They include active
waits and runtime work and can reflect dependencies, uneven tasks, and other
threads' compute/memory costs. This supports retaining load balancing and
synchronization as the next areas to test, rather than selecting the rejected
byte expansion. The prepared split-only full-model comparison remains the
strongest next experiment and still requires the pending interruption decision.
The earlier GLM spin-limit reduction regressed and is not evidence for a Qwen
speedup; no global or process runtime policy was changed here.

`audit-qwen-current-profile-0906.py` reconciled sample periods with perf totals,
decode-window bounds, thread identities/affinities, source/capture hashes, 18
runtime library hashes, and Qwen's original mapped libllama bytes. Both owned
perf PIDs are absent, and both protected services are idle, unqueued, and
unchanged at the final audit. The first audit needed privileged read access to
root-owned perf files. A subsequent review corrected anonymous-namespace C++
symbol parsing; the initial category audit is preserved separately and the
corrected `validation-audit.json` is authoritative. Raw samples are unchanged.

All work launched in this continuation has finished. The Qwen lifecycle trial
has not executed, and no new job is queued. This continuation made progress
through a tested, rejected kernel and a current, audited thread-level profile.
The full goal remains active and incomplete for GLM Flash, Qwen Flash-Next,
and Full; none has verified at least 304 GB/s of model-attributable decode.

### Repeated interruption decision: prepared work is complete, model trial blocked

The preceding goal turn was progress: it completed the exact byte-layout
comparison and current Qwen thread profiles. This continuation revalidated
the actual service identities, commands, runtime environments, affinities,
queues, staged library/source hashes, and the 17 completed lifecycle checks.
The controller remains `ready: true, execute: false`. Neither protected service
has changed, no owned benchmark remains active, and no lifecycle is queued.
The read-only evidence is in `results/model-bandwidth-decision-block-0906.json`.

The same unanswered Qwen interruption decision has persisted across at least
three goal turns, including the row-scheduling/API-capability work, the
byte-layout/current-profile work, and this revalidation. Independent work
continued while useful experiments remained. Those experiments are now
complete, and the next necessary step is the prepared complete-model
original/candidate/original comparison. Additional component-only tests or
repeats of the unchanged services cannot resolve that comparison.

The goal is blocked on that existing user decision, not achieved. Execution
requires the explicit exception described in the staged `TRIAL.md`: Qwen is
temporarily unavailable while the candidate is loaded/measured and the
original service is restored. Full remains running and is never signalled.
No assumption of approval is made from automatic goal continuations.

The objective remains unchanged: GLM Flash, Qwen Flash-Next, and Full must each
verify at least 304 GB/s of model-attributable decode. The current audited Full
and Qwen captures remain below that requirement; GLM Flash still lacks a
completed actual IMC decode baseline while Qwen occupies the single Flash
slot. The existing trial plan is concrete and ready for the pending decision.
