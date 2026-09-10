# Qwen decode scheduling experiments, September 9

These are private, same-precision experiments on the corrected Qwen Q6/Q8
runtime. An initial HC-only pair measures a roughly 2.2% gain, but the later
complete ten-route comparison establishes no meaningful overall speed gain:
-0.60% on prose and +0.10% on code. The 250 GB/s target remains unmet.
The separately launched Qwen service and its selected configuration remain
unchanged; the combined scheduling candidate stays disabled there.

## Narrow HC projections

The executed-operation captures contain Q8 projections with 10,240 input
values and 64 or 96 output rows per socket. Their names select the generic
Q8 r8 repack implementation, rather than the x16 attention implementation.
The current row scheduler provides eight or twelve weight-row groups for
fifteen workers. Four-row activation packing and leftover GEMV rows run
within each weight-row job.

The first proof extends the earlier Flash ordered-K implementation to eight
activation rows. It compares its arithmetic against graphs running on the
unchanged corrected Qwen CPU. All 96 cases pass bit-for-bit, including
ordinary versus packed-four activation quantization, padded input rows,
preserved weights/inputs, and guarded partial/output buffers. Each case uses
three input samples. Input widths are 4,096, 10,240 and 32,768; output widths
are 64/96; activation rows span one through eight.

- [Arithmetic proof](results/qwen-hc-ordered-k-proof-0909/result.json)
- [Ordered-K private build](results/qwen-hc-ordered-k-build-0909/result.json)
- [Token-row private build](results/qwen-hc-row-split-build-0909/result.json)

Each of the first two private libraries reproduces the current parent object
and complete CPU library exactly before replacing only repack.cpp.o. Five
96-case graph arms check parent, disabled, enabled, one-worker fallback and
four-worker fallback. All outputs agree exactly and audited dispatch counts
match the intended shape restrictions. The weight quantization is unchanged.

The first version partitions input blocks, records integer dot products and
scales, and then performs floating-point accumulation in the original block
order. The second uses the existing GEMV kernel for separate activation-row
jobs, avoiding partial-result storage during verification. Neither broad
version is suitable for a model trial:

| Rotating-weight case | Ordered-K time change | Token-row version time change |
| --- | ---: | ---: |
| 64 outputs, one token | -20.00% | -17.20% |
| 64 outputs, four tokens | -1.70% | -4.89% |
| 64 outputs, five tokens | -17.03% | -12.84% |
| 96 outputs, one token | +9.15% | +0.36% (original kernel retained) |
| 96 outputs, four tokens | +35.25% | +30.29% |
| 96 outputs, five tokens | +21.87% | +27.59% |

Negative values mean shorter component time. Each comparison averages two
candidate and two parent runs. The fixture uses 96 distinct matrices per
graph, totaling 66.85 or 100.27 MB, and forty timing samples. Single-matrix
cached measurements are also retained but show larger differences between
runs. No component result is counted as model bandwidth utilization.

## Bounded candidate and Q6 expert tiles

The third private build restricts both HC changes to 64 outputs: ordered-K
for one activation row, and token-row jobs for two through eight rows.
The 96-output matrices retain their original kernels. It also exposes a
separate, disabled-by-default expert tile setting for Q6 weights with
K=2,560, 160 outputs and 512 experts. Its eight-route predicate was a mistaken
prototype assumption: the actual target and draft both route to ten experts
per token. The prototype allows at most eight tokens and correctly skips the
ten-route model workload. The following component results cover eight routes.
The options are 16/32/48 rows instead of 64. Fixed stack buffers retain the
64-row capacity. Per-output quantized dot products and SwiGLU are unchanged.

The resulting private CPU SHA-256 is
45f35424e9213a7cfe279f64f620f189184e9f653f0e14304c99bce0617ea363.
Both parent object and complete-library reproduction pass. The HC graph
suite again passes all five 96-case arms, with 48 selected dispatches in
the enabled arm. The expert suite passes seven 28-case arms, including
padding, an additional gate consumer, one/four/fifteen workers, three input
samples, and 64-token prefill fallback. All compared outputs are exact;
input/route buffers and packed weights are preserved.

Ten expert timing arms compare the parent, private 64-row control and each
candidate twice. Every rotating case visits all 512 experts. The 32-row
candidate's mean times versus the parent are:

| Rotating expert workload | Parent microseconds | 32-row microseconds | Time change |
| --- | ---: | ---: | ---: |
| One token, eight routes | 92.654 | 82.548 | -10.91% |
| Five tokens, overlapping routes | 316.591 | 314.103 | -0.79% |
| Five tokens, distinct routes | 240.205 | 234.592 | -2.34% |

All timing-output hashes match. The predeclared component gate admits all
three tile options; 32 rows has the highest geometric mean improvement
across the three rotating workloads, 1.0502x. This gate applied to the wrong
route count. It does not qualify the ten-route workload or project a model
speedup, and the experiment is not a selected serving configuration.

- [Bounded build, correctness and timings](results/qwen-decode-scheduling-build-0909/result.json)
- [Source transformation](qwen_decode_scheduling_transform_0909.py)
- [Expert fixture](check-qwen-q6-expert-tiles-0909.cpp)

One unchanged HC fallback case, 96 outputs and four activation rows, measured
9.29% slower in the third build's first timing set. Its two candidate medians
were 44.09 and 39.39 microseconds, versus parent medians of 38.21 and 38.17.
The separate retiming controller uses the same binaries, both feature states,
two sockets, balanced arm order, and a fresh background gate before each arm.
All twelve arms completed. Its two-socket geometric mean speedups are
1.2497/1.0613/1.1393 for 64 outputs at one/four/five tokens, and
0.9860/0.9636/0.9648 for the unchanged 96-output cases. These pass the
predeclared model-test gate: no aggregate rotating case over 5% slower, and
at least one changed case over 10% faster. The underlying timing variation
is not explained or removed from the evidence.

A workspace-history follow-up was prepared after source inspection showed
that the CPU backend grows and retains its scratch buffer. That additional
fixture and controller were not executed: the completed two-socket recheck
already admits a model test. They make no performance claim.

- [Retiming controller](recheck_qwen_decode_scheduling_0909.py)
- [Completed two-socket recheck](results/qwen-decode-scheduling-recheck-0909/result.json)
- [Executed, incomplete model comparison controller](run_qwen_decode_scheduling_0909.py)

The model controller retains the existing Q6 target, Q8 MTP4 draft, corrected
gather, shared dispatch, and fifteen workers per socket. It checks the private
loader and all four NUMA devices before a parent/candidate/candidate/parent
sequence. Complete outputs and generated/drafted/accepted counts must match;
all 48 IMC counters and adjacent idle subtraction must qualify. Every owned
model is released and the separately launched peer is preserved. The private
loader passes and exposes all four NUMA devices. The first parent and
candidate children completed both requests, preserved the peer, released
shared dispatch groups and exited with status zero. Their results are:

| Workload | Parent tok/s | HC-only tok/s | Parent GB/s | HC-only GB/s |
| --- | ---: | ---: | ---: | ---: |
| Prose | 24.205 | 24.738 | 139.329 | 143.214 |
| Code | 31.447 | 32.122 | 148.501 | 152.104 |

All four counter windows qualify, with exact complete outputs and
generated/drafted/accepted counts. The candidate logs both HC execution
markers. It does not log the expert marker: its eight-route predicate is
ineligible for the actual model. The outer comparison fails on that explicit
check before its second candidate or final parent run. This is one HC-only
pair, not a completed ABBA, repeated gain, expert-tile gain or 250 GB/s result.

- [Independent partial-pair analysis](results/qwen-hc-partial-pair-0909.json)
- [Failed outer comparison](results/qwen-decode-scheduling-model-0909/result.json)
- [Cleanup and preserved-peer audit](results/qwen-decode-scheduling-host-audit-0909.json)

The corrected `0909b` experiment uses ten routes and scales the rotating
expert offset accordingly. It tests all 512 experts, five-token overlap
(22 active experts), five-token distinct routing (50), and eight-route
fallback. Only the route predicate and once-only log marker change in the
library; the HC implementation is identical to the verified prototype.
All seven 30-case graph arms pass, totaling 210 cases with exact output
dumps and preserved input, routing and weight buffers. The 16-row option
is the only candidate to pass the original performance gate for ten routes:

| Rotating ten-route workload | Parent microseconds | 16-row microseconds | Time change |
| --- | ---: | ---: | ---: |
| One token | 99.260 | 94.965 | -4.33% |
| Five tokens, overlapping routes | 387.700 | 366.890 | -5.37% |
| Five tokens, distinct routes | 290.557 | 292.617 | +0.71% |

The geometric mean speedup is 1.0313x for 16 rows, 1.0095x for 32 and
1.0033x for 48. Timing-output hashes agree in all ten arms. The selected
model-test tile is therefore 16, rather than the earlier eight-route
prototype's 32. These are component results; a fresh parent/candidate/
candidate/parent model comparison completes with all eight counter windows
qualified and exact complete outputs and generated/drafted/accepted counts.
Both candidate runs log all three execution markers, including ten expert
routes and 16-row tiles. The complete results are:

| Arm | Prose tok/s | Code tok/s | Prose GB/s | Code GB/s |
| --- | ---: | ---: | ---: | ---: |
| First parent | 24.1888 | 31.7578 | 139.6153 | 150.1700 |
| First candidate | 23.8684 | 31.5261 | 138.8574 | 149.4166 |
| Second candidate | 24.8748 | 32.1467 | 143.5546 | 152.5486 |
| Final parent | 24.8468 | 31.8495 | 143.1092 | 150.2464 |
| Parent mean | 24.5178 | 31.8037 | 141.3622 | 150.2082 |
| Candidate mean | 24.3716 | 31.8364 | 141.2060 | 150.9826 |

The first paired speed changes are -1.32%/-0.73%; the reverse-order pair
changes are +0.11%/+0.93%. Mean changes are -0.60%/+0.10%. The combined
candidate does not establish a repeatable model gain or either bandwidth
target, and is not promoted. It also does not isolate the HC change from
expert tiling. All four owned models exit zero and release shared groups.
The final audit verifies unchanged peer identity/runtime, inactive Full,
free private ports and lifecycle lock, and about 359.6 GB available memory.

The corrected private CPU SHA-256 is
f437355d2ae73fe9ebf5f461e51ca2410f1c9fb88892cd2c68a8ed168f1f6034.

- [Ten-route build and validation](results/qwen-decode-scheduling-build-0909b/result.json)
- [Ten-route model controller](run_qwen_decode_scheduling_0909b.py)
- [Completed repeated model comparison](results/qwen-decode-scheduling-model-0909b/result.json)
- [Post-comparison cleanup audit](results/qwen-decode-scheduling-host-audit-0909b.json)
- [Source patch reconstruction](results/qwen-decode-scheduling-source-publication-0909b.json)

The optional published overlay reconstructs the tested repack source and HC
header exactly. Its intrinsic helper requires the SR950's x86 AVX-512 VNNI
build configuration; non-VNNI and non-x86 builds were not tested. The overlay
does not by itself recreate the experimental shared-dispatch and HC-normalization
runtime, whose separate private sources and manifests remain in the archive.

The expert timing fixtures rotate through all 512 experts using contiguous
route indices before wraparound. They test working-set rotation and route
overlap, but not the model's actual expert-ID distribution. This is a
remaining component-to-model coverage limit; scattered-ID timing is a
possible follow-up, not an established explanation for the model result.
