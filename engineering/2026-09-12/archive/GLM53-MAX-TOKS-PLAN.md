# GLM-5.3 Full on the SR950: the path to maximum tok/s

*Written 2026-09-01 (session in progress). Target set by the maintainer: GLM-5.3 Full (non-Flash)
at >= 12 tok/s single-stream on this box. Companion: `serving/glm-sr950/DEV-LOOP-20260901.md`
(measurement method, dev model, micro-benchmark results).*

## 0. The arithmetic that governs everything

Decode at batch 1 is a weight-streaming problem: `tok/s = effective GB/s / GB read per token`.

**What the machine can supply (measured, not theoretical):**

| scope | GB/s |
|---|---:|
| one socket, NUMA-local streaming (16 cores) | 95-98 |
| four sockets concurrently, NUMA-local | 360-380 |
| four sockets, interleaved (any non-TP backend) | 139 |

DIMMs are 24 x 32 GB, all 6 channels per socket populated. SMBIOS reports them
**configured at 2400 MT/s** (theoretical 461 GB/s); Xeon Gold 6242 supports 2933.
See section 5 - this is the one hardware knob worth checking.

**What GLM-5.3 UD-Q4_K_XL reads per token (from the GGUF tensor inventory):**

| component | stored type | GB/token | share |
|---|---|---:|---:|
| routed experts (8 of 256, 76 layers) | Q4_K gate/up, Q5_K/Q6_K down | 13.9 | 41% |
| attention (q_a, q_b, kv_a, k_b, v_b, output, indexer), 79 layers | **Q8_0** | 14.7 | 43% |
| shared expert + router | Q8_0 / F32 | 3.5 | 10% |
| dense FFN (3 layers) + lm_head | Q8_0 | 1.7 | 5% |
| **total** | | **33.9** | |

(Correction 2026-09-01 evening: the fork's `block_q5_K_r8` layout used for the Q5_K
down-experts is 5.625 bpw, i.e. compact; the 8.625 bpw "expansion" applies only to the
dense Q5_K path of the GLM-5.2 era. Bytes per token are 33.9 GB as stored.)

`attn_output` alone is [16384 x 6144] Q8_0 = 107 MB per layer = 8.4 GB per token, the
single largest tensor family. The Unsloth "UD" scheme keeps attention at 8.5 bits for
quality; on a bandwidth-bound CPU that is a 43% tax.

**Ceilings:** 33.9 GB at 380 GB/s = **11.2 tok/s raw** even at 100% extraction. So >= 12
raw at this quant *as loaded* is physically impossible on this memory. The path must
combine three things: (a) fewer bytes per token, (b) higher extraction, (c) more than one
emitted token per weight pass (speculative decoding). No single one is enough.

**Where we are (quiet host, measured earlier sessions):** raw 6.95 tok/s = ~250 GB/s
effective = **~65% extraction**. Production speculative default (MTP n=2) ~8-9 general;
agentic replay with n=18/p=0.75 measured 12.9 (workload-specific, not general).

## 1. What is already right and must not be rebuilt

- **Tensor-parallel across the 4 sockets with node-local weights** (`--device
  CPU-NUMA0..3 --split-mode tensor`, `GGML_CPU_NUMA_REPACK=1`). Each socket holds an
  exact quarter (125 GB/node, balanced within 1%) and streams only that quarter. This is
  the correct topology: expert-parallel (whole experts per socket) would unbalance
  per-token load (8 experts over 4 sockets -> the busiest socket gets 3-4), and any
  single-backend layout is capped at 139 GB/s. Keep TP.
- **Direct F32 all-reduce** at layer boundaries (+37% over the generic butterfly).
- **Per-device 16-thread pools**, `GGML_CPU_NUMA_POLL=100`.
- **Kernels are bandwidth-bound when streaming** (measured today, node 0, 16 cores,
  contiguous rows: upstream AVX2 8x8 Q4_K gemv 94.4 GB/s; new AVX-512 VNNI 16-row gemv
  98.5 GB/s; the VNNI kernel does 400 GB/s aggregate from cache = 4x compute headroom).
  Raw kernel speed is *not* the primary loss at batch 1.
- **MTP draft head** from the same checkpoint (`draft-mtp` alone; the composite
  `ngram-mod,draft-mtp` 500s every request on 5.3 - a bug to fix, see 4d).

## 2. The loss budget (what the ~35% of missing bandwidth is)

At 6.95 tok/s a token takes 144 ms; pure streaming of 36.7 GB at 380 GB/s is 97 ms.
The ~47 ms residual comes from, in my current order of belief (the full-model per-op
profile being captured this session decides the order):

1. **Cross-device synchronization.** The meta backend splits each token into one
   subgraph per all-reduce point (>= 2 per layer, ~160-320 per token). Each subgraph is
   dispatched to the 4 device threads through a mutex + condition variable and collected
   the same way: two futex round-trips across sockets per subgraph, ~50-150 us each.
   Estimated 10-30 ms per token.
2. **Per-op barriers inside each device.** ~60 nodes per layer x 79 layers ~ 5,000 ops,
   each followed by an OpenMP barrier over 16 threads, including for tiny 1-token ops
   (norms, rope, scales, top-k, router softmax, the indexer). ~1-3 us each -> 5-15 ms.
3. **Redundant (mirrored) compute.** The tensor-parallel split rules mirror the shared
   expert (40 MB/layer), `attn_q_a` (13 MB) and `attn_kv_a_mqa` (4 MB) onto every socket,
   so each socket streams them in full instead of a quarter: measured 22 + 11 + 12 + 4 ms
   of a ~155 ms token on every device (profile of 2026-09-01). This is the largest
   single software loss found so far and is fixed by split rules, not kernels.
4. **Sub-optimal streaming pattern in MoE ops:** each thread streams 16 separate ~110 KB
   slices per layer (32 rows x 8 experts x 2 matrices) instead of a few long streams;
   thread 0 does serial setup (routing table, activation quantization) while 15 wait.
5. **Imbalance / straggler waits** ("libgomp wait" was 28% of samples in an older
   profile): uneven chunks and the thread-0 serial sections above.

## 3. The path, in order of tok/s per hour of work

### Phase 1 - Cut the fixed per-token cost (no quality change). Raw 7 -> ~8.5-9

1a. **Spin-wait dispatch.** Replace the condvar hand-off in `ggml_backend_meta_context::
    dispatch_cpu_numa` with socket-pinned dispatch threads spinning on an atomic epoch,
    and completion via an atomic counter the main thread spins on. Removes ~2 futex
    wakes per subgraph. Small change in `ggml/src/ggml-backend-meta.cpp`.
1b. **One barrier per layer boundary, not two dispatches.** Target design: the four
    persistent socket teams run each device's whole sub-graph in lock-step and perform
    the all-reduce *inside* the worker threads - each thread sums its slice of the
    6144-float residual across the four device buffers behind a cross-socket spin
    barrier (atomic counter), then continues. No main-thread round trip at all.
    The residual is 24 KB; the cost becomes ~5-10 us per boundary instead of 50-150 us.
1c. **Skip barriers between consecutive single-threaded ops.** In
    `ggml_graph_compute_thread`, when node n and n+1 both have `n_tasks == 1`, only
    thread 0 runs them and the barrier is deferred to the next multi-threaded node.
    Also fold the remaining small ops (norm+mul, rope, scale) into fused ops where ggml
    already has fusion hooks.
1d. **Parallelize thread-0 serial work** in `forward_mul_mat_id*` (routing table build,
    activation quantization already split by blocks - verify) and re-sweep threads per
    device (16 vs 20 vs 32 with SMT) *after* 1a-1c, since the optimum moves once
    barrier cost drops.
1e. **Verify the subgraph count.** With `GGML_GLM_ATTN_TP=1` and the delayed MoE
    all-reduce the count should be ~2 per layer (~158/token). The `META_NUMA_PROFILE`
    line prints it; if it is higher, find the extra PARTIAL nodes and delay them.

### Phase 2 - Fewer bytes per token without touching routed-expert precision. Raw -> 10.5-12

2a. **16-row-interleaved VNNI kernel family** (`block_q4_K_x16` validated today;
    Q5_K/Q6_K/Q8_0 variants follow the same design: one 64-byte vector = 16 rows x 4
    weights, `vpdpbusd` against a broadcast of 4 activations, per-row scales applied
    with `vpmulld`, no horizontal reductions). Needed as the fast Q6_K/Q5_K path that
    2b requires (x86 has only generic C for Q6_K 8x8 today) and for batched verify.
2b. **Load-time requantization of attention and shared-expert tensors** from Q8_0 to
    Q6_K (or Q5_K) inside the NUMA repack step. No new GGUF on disk (both disks are
    full); the mmap'd Q8_0 bytes are dequantized and requantized into the node-local
    buffer once per load. Attention 14.7 -> 11.3 GB (Q6_K) / 9.5 GB (Q5_K); shared
    experts 3.0 -> 2.3 / 2.0; lm_head 1.0 -> 0.78; router weights F32 -> F16 (-0.24).
    Q6_K attention is standard in Q5_K_M/Q6_K quants and near-lossless; validate with a
    KL-divergence probe against the Q8_0 run before adopting Q5_K.
2c. Resulting bytes per token: **~29 GB (Q6_K attention) / ~27 GB (Q5_K attention)**,
    down from 33.9. Raw ceiling at 380 GB/s: 13-14 tok/s.

### Phase 3 - Extraction: make the real graph stream like the micro-benchmark. 65% -> ~85%

3a. **Long streams per thread in MoE.** Partition rows so each thread owns a few large
    contiguous runs per layer (e.g. half an expert = 256 rows = 0.9 MB) instead of
    sixteen 110 KB slices; keep the 64-row L1 tile for the fused gate/up SwiGLU.
3b. **Prefetch and non-temporal loads** in the streaming kernels (`prefetchnta` two
    row-groups ahead, `_mm512_stream_load` for weights): weights are touched once per
    token, so they should not displace the KV cache and activations from L2/L3.
3c. **Balance the residual small ops**: run the indexer/top-k/router ops on the device
    that owns the data with the smallest possible task count.
3d. Re-run the whole-token profile and iterate until the sum of matmul time equals
    bytes/380 GB/s within ~15%.

### Phase 4 - The speculative multiplier (the only way past the raw ceiling). x1.3-1.8

4a. **Model the verify cost honestly.** On MoE, a verify batch of n+1 tokens routes to
    up to 8(n+1) distinct experts, so routed-expert bytes scale ~linearly with n while
    attention/shared/dense bytes are amortized. After Phase 2 the dense share shrinks,
    so the optimal n *drops*. Re-derive `n_max`/`p_min` per workload after each phase;
    optimize aggregate tok/s, never acceptance (measured anti-correlated).
4b. **Make drafts cheaper.** Each MTP draft token reads the NextN layer plus an lm_head
    (Q8_0 1.0 GB; the OUTQ4 hybrid already cuts it to 0.5 GB). Go further: IQ2/Q3 lm_head
    for the *draft only* (draft precision only affects acceptance), and skip the draft's
    full-vocab softmax by taking the top-1 from a reduced head.
4c. **Batched verify kernels.** The 16-row VNNI kernels process up to 4 activation rows
    per weight load, so tokens that share an expert in the verify batch read it once.
4d. **Fix the composite `ngram-mod,draft-mtp` path** on 5.3 (every request 500s while
    `/health` is green). On the Qwen twin ngram-mod took replay from 17 to 23.6 tok/s;
    for agentic/file-edit traffic this is worth more than any kernel.
4e. **Adaptive depth**: draft until the draft's own confidence falls below `p_min`
    (already supported) - tune per alias (`glm-sr950` general, `glm-sr950-agentic`).

### Phase 5 - Firmware and platform (needs the maintainer: BMC/BIOS, no sudo from sessions)

5a. **Memory speed.** SMBIOS shows 2400 MT/s configured. If the DIMMs are 2666/2933-rated,
    setting the UEFI memory speed to maximum is +11-22% bandwidth across *every* phase.
    Check in XCC (Lenovo BMC) -> Inventory -> Memory, or `sudo dmidecode -t 17`
    ("Speed" vs "Configured Memory Speed").
5b. **Uncore frequency and C-states.** Set OS/UEFI to Maximum Performance, disable
    uncore frequency scaling, disable C6. Bandwidth-bound decode is sensitive to the
    uncore clock; the box idles between tokens long enough for scaling to bite.
5c. **Huge pages for the node-local weight buffers** (`GGML_CPU_NUMA_HUGEPAGES=1`,
    THP is `madvise`): 470 GB of streaming through 4 KB pages costs TLB walks; it was
    measured harmful on Qwen-27B but never on GLM-5.3. Measure, do not assume.
5d. **Model on NVMe.** The 20-minute cold load from the SATA SSD is the dev-loop
    bottleneck. The NVMe root holds 194 GB of GLM-5.2 shards that production no longer
    uses; retiring them would fit GLM-5.3 on NVMe (~2.5 min loads).

### Phase 6 - Beyond this memory system (optional, hardware purchase)

The DDR4 bus is the hard limit. One 24 GB GPU holding the dense parts (attention 14.7 GB
+ shared experts + lm_head, all Q8_0, ~19 GB) while the routed experts stay in CPU RAM
(the KTransformers split) would cut CPU-side bytes to ~14 GB/token: a raw ceiling of
~25 tok/s from the same DIMMs. It is the only single change that doubles the ceiling.

## 4. Expected trajectory (single-stream, general prose, quiet host)

| after | GB/token | extraction | raw tok/s | with MTP |
|---|---:|---:|---:|---:|
| today (quiet) | 33.9 | 65% | 6.95 | 8-9 |
| today (contended, 2026-09-01 evening) | 33.9 | 40% | 3.8 | 6.5 |
| Phase 1 (incl. de-mirroring) | 33.9 | 75-80% | 8-8.5 | 10-11 |
| Phase 2 | 27-29 | 75-80% | 10-11 | 12.5-14 |
| Phase 3 | 27-29 | 85% | 11.5-12.5 | 14-17 |
| + 5a if DIMMs run 2933 | | | x1.15 | x1.15 |

The 12 tok/s target is reached at Phase 2 + MTP on general prose, and at Phase 3 raw.
Every row above is a projection; each phase is measured on the live server with the
fixed harness before the next is started.

## 5. What not to do (measured dead ends, do not re-run)

- Smaller routed-expert quants (Q3/Q2): measured slower - extraction collapses faster
  than bytes shrink. Q4_K_XL is the fastest expert tier here.
- Any single-backend/interleaved layout: 139 GB/s cap, 2.4x slower than TP.
- Expert-parallel across sockets: unbalanced per token.
- `OMP_WAIT_POLICY=PASSIVE`, `--cpu-strict`, OMP place/bind env: all measured worse.
- The composite spec type as shipped: 500s on every request (fix, then re-enable).
- Optimizing draft acceptance: anti-correlated with throughput.
- Benchmarking on a busy host: contention cost up to 81% on other models here. Check
  `top` first; the internal-dashboard build and containers are the usual offenders.

## 6. Ordering rationale

Phases 1 and 2 are independent code paths (meta backend / repack) and can proceed in
parallel; Phase 3 depends on the profile after Phase 1; Phase 4 tuning must be redone
after every byte-count change; Phase 5a is a five-minute BMC check with the largest
free upside, so ask for it first.

## 7. Progress log (2026-09-01 evening, same session)

Measured on the 8-layer dev model unless noted (tok/s, medians of 2 reps, noisy host):

| change | 8-layer tok/s | mechanism |
|---|---:|---|
| production binary | 44.4 | baseline |
| + spin-wait dispatch (1a) | 46.4 | no futex round trips per subgraph |
| + shared-expert split (`GGML_GLM_SHEXP_TP=1`) | 48.3 | stops mirroring 40 MB/layer on every socket; bit-identical logits |
| + fused in-graph all-reduce (`GGML_CPU_NUMA_FUSED_REDUCE=1`, 1b) | 52.5 | one dispatch per token instead of one per boundary (~150 us saved per boundary) |
| + q_a/kv_a k-split (`GGML_GLM_QA_TP=2`) | 54.9-56.0 | stops mirroring 17 MB/layer; **was numerically wrong until the strided-row fix below** |
| barrier-skip for tiny ops (1c) | +1% | small |
| OMP_WAIT_POLICY=ACTIVE | 0% | threads already spin |
| huge pages | -2.5% | keep off |

**The whole-token profile (section 2) was right about the loss being scheduling and
mirroring, not arithmetic.** Per-op time on the production binary: routed experts stream
at 76 GB/s/socket, attn_output at 97, while shexp/q_a/kv_a showed ~22 GB/s because each
socket streamed the full tensor (mirrored). Dispatch/sync outside ops was 50-90 ms of a
220-270 ms contended token.

**Kernel family (Phase 2a) landed and verified.** `block_{q4_K,q5_K,q6_K,q8_0}_x16`
with AVX-512 VNNI gemv: reference error ~1e-4, streaming 92-99.5 GB/s per socket
(= ceiling), cache-resident 316-643 GB/s (3-6x headroom). In the fork they are
bit-identical to the existing paths on the 4-layer oracle. Load-time requantization of
Q8_0 attention/shexp/output to Q6_K or Q5_K (`GGML_CPU_*_REQUANT`) drifts the first-token
logprobs by only 0.02 on the oracle; no new GGUF needed.

**Root cause of the q_a/kv_a k-split drift:** the repack traits quantize activation rows
four at a time assuming contiguous rows; a column-sliced view of a multi-token batch has
row stride 6144 floats, so prompt processing read interleaved garbage while single-token
decode was fine. Fixed by gathering strided rows into a temporary before the 4-row
quantizer (both call sites). Re-verification queued (chain 8).

**Correctness discipline that worked:** a 4-layer truncated model is deterministic
(top-10 logprob diff 0.0 across identical runs), so any config drifting > 0.01 is wrong;
the 8-layer model is chaotic (0.1 between identical runs) and only good for timing.

**Full-model v1 (spin + shexp split + fused reduce, production kernels/bytes), 2026-09-01
18:50Z, same background load as the baseline:** raw decode **4.74 tok/s** mean
(4.67-4.83, 8 runs) vs 3.76 mean (3.10-4.66) for the production binary earlier in the
evening; `17*23 -> 391` correct. Per-byte the full model is ~1.4x slower than the 8-layer
dev model extrapolation; one identified cause is that the 4 layers with Q6_K
down-experts run a scalar generic kernel on x86 (no SIMD Q6_K 8x8 exists), which the
x16 Q6_K path in v2 replaces.

**Chains 7/8 (2026-09-01 19:00Z), 8-layer dev model, dev2 binary (fused + shexp):**
base 53.6 | x16 experts 55.2 | x16 for Q8_0 too 54.7 (keep Q8_0 on the 8x8 VNNI path) |
+Q6_K requant of attention/shexp/output 57.3 | Q5_K instead 58.7 | +SMT (32 thr/socket) 59.0.
Strided-row fix verified: with llama-eval-callback on a 5-token prompt every op agrees
between mirrored and k-split q_a/kv_a to ~5e-6 relative (was 2x off on `q-0`), so
`GGML_GLM_QA_TP=2` is correct; the residual 0.015 probe drift is rounding amplified by the
untrained 4-layer head. Full-model v2 = fast profile (fused + shexp + qa + x16 experts +
Q6_K requant + dynamic MoE tiles) is running.

**Straggler finding:** in the baseline profile, ops with dynamic chunking (attn_output,
q_b) take the same time on all four sockets, while statically partitioned ops (MoE
experts, mirrored shexp/q_a, the F32 router) run 1.2-1.8x slower on sockets 2/3. That is
a shared-core/straggler signature (background dockerd/containerd/mssql threads), not
bandwidth. The x16 MoE paths now use dynamic (expert, 64-row tile) work stealing.

**v3 (15 threads/socket + F16 router + OMP_WAIT_POLICY=ACTIVE) FAILED: 0.08 tok/s.** The
server holds two contexts (target + MTP draft), each with 4 x 15 OpenMP workers on the
same cores; ACTIVE keeps the idle context's 60 threads spinning, so 120 runnable threads
fight for 64 cores (run-queue wait measured at 93% of run time). Never set it here; the
dispatch and fused-reduce spin paths already cover intra-token gaps. v4 = v3 minus ACTIVE
plus Q5_K attention/shexp is running; v5 adds merged all-reduce boundaries
(`GGML_CPU_NUMA_MERGE_REDUCE=1`, q_a+kv_a and MoE+shexp share one reduce) and confines the
main/HTTP threads to the spare core of each socket (`taskset -c 15,31,47,63`).

Note: another session edited this fork concurrently during the evening (meta backend,
speculative.cpp, server-context.cpp). Builds after ~21:30Z include those changes; every
build is re-checked on the 4-layer oracle before a full-model run.

**Item 2 (host isolation) applied 2026-09-01 23:16 local** via
`host-setup/isolate-background.sh`: docker/containerd/netdata + [client]-db confined to the spare
cores 15,31,47,63 (+siblings), CI runners to 64-127, CPUWeight 50 / cpu-shares 256, daemons
restarted ([client]-db restarted as a side effect). dockerd+containerd fell from ~285% to ~120%
CPU and no longer touch worker cores. Isolated re-benchmark of v4 with a per-thread run-queue
probe is queued as the first measurement.

**Item 5 decisions:** RMS-norm fusion skipped (< 1% after barrier-skip); draft refresh left
alone (another session is rewriting `common/speculative.cpp`); implemented instead:
per-head x16 path for k_b/v_b (`GGML_CPU_X16_ATTN3D=1`), dense attention below the indexer
top-k (`GGML_GLM_DSA_DENSE=1`, skips ~5.5 ms/token of indexer work plus 3 mask ops per layer
at contexts <= 2048; mathematically identical), and a dual gate+up Q4_K kernel
(`GGML_CPU_X16_DUAL=1`). Each is verified on the 4-layer oracle before entering a full run
(v5: merge + taskset; v6: + attn3d + dense; v7: + dual).

**v5 (2026-09-02 00:05) was invalid**: launching under `taskset -c 15,31,47,63` made the
CPU-NUMA devices enumerate only the allowed CPUs (one worker per socket, 0.65 tok/s).
Relaunched as v5b without the restriction; the main/HTTP threads are pinned to the spare
cores after startup (`pin_main.sh`) and a per-role run-queue probe runs during the bench.
Queue (chain 22): v5b -> agentic replay -> CUDA build -> dense/dual oracle+timing -> v6 ->
MTP sweep.

**v5b (2026-09-02 00:34, isolated host, correct launch):** raw **7.51 tok/s** (7.42-7.59),
MTP n2 **9.55** (structured 10.8-11.1). **Agentic replay (project harness, 3/3 correct,
n18/p0.75): 13.32 / 14.07 / 12.87 = 13.4 tok/s mean.** Cumulative from the production binary
on the same box: raw 3.76 -> 7.51 (2.0x), general MTP 6.8 -> 9.55, agentic replay 12.9
(08-29, quiet) -> 13.4 under the new stack with the host isolated. The 13+ goal is met for
agentic/replay traffic on CPU alone; general prose needs the GPU split (section 9) or
faster DIMMs. v6 (dense-below-top-k + dual gate/up) and the MTP re-sweep are queued.

**v6 CORRUPTS OUTPUT (2026-09-02 01:30) - do NOT ship.** v6 (= v5b + dual gate/up +
dense-below-top-k) measured raw 7.85, MTP n2 11.9, and an MTP sweep to 14.1 (n6/p0.6), but
the target model produces fluent nonsense: raw n0 on "17*23" returns "0.5, 0.5, 0.5", agentic
replay 0/3 correct. The 4-layer oracle passed at 0.0 because it only checks the FIRST token of
a short prompt; the corruption shows only in multi-token generation. This is the qwen4exp
tensor-split lesson again: a throughput-only sweep selects for the fast garbage path. The MTP
sweep's 14.1 is void. Isolation (dual-only vs dense-only, exact 64-token output on the 8-layer
model) is running. **v5b remains the correct high-water mark: raw 7.51, general MTP 9.55,
agentic replay 13.48 (3/3 correct).**

Full-model validation of the scheduler fixes (spin + shexp + fused) is running as `v1`;
the next full run (`v2`) adds x16 experts + Q6_K attention/shexp/output requant + the
k-split. Numbers land in `serving/glm-sr950/DEV-LOOP-20260901.md`.

## 8. Operating the fast profile (handoff)

Launch (same launcher, new env file; the binary is `build-dev2`):

```
cd ~/InfoSystemic/AI-Server/serving/glm-sr950
GLM_SR950_CONFIG=$PWD/model.glm53-q4-fast-v6.env setsid nohup ./launch-glm-sr950.sh 18091 > /tmp/glm-fast.log 2>&1 &
# then, once /health is ok, pin the main/HTTP threads to the spare cores:
/dev/shm/glm-dev/pin_main.sh   # (copy lives in host-setup/ too)
```

Rollback = the same command with `model.glm53-q4.env` (production binary and knobs).
Every new knob is opt-in and off by default, so the production profile is unaffected by
the source changes; `build-sr950-glm` was never rebuilt.

Verify after any change: one real completion (`17 * 23` -> `391`) and the raw
(`speculative.n_max=0`) decode rate from `decode_bench.py`, on a quiet host. The
per-request speculative sweep (`/dev/shm/glm-dev/mtp_sweep.sh`) re-derives `n_max`/`p_min`
after every byte-count change, because the optimum moves when the dense share shrinks.

## 9. GPU hybrid plan (V100/P40 on the SR950 riser)

Readiness verified 2026-09-01: driver 580-server + CUDA 12.0 installed (both still support
sm_61 Pascal and sm_70 Volta; CUDA 13 would not), the fork's CUDA backend has the GLM-DSA
lightning-indexer op, and `build-cuda` is configured for `61;70` (compile queued behind the
benchmarks; binary lands in `engines/llama.cpp-sr950-glm/build-cuda/bin/llama-server`).

Split: GPU holds every per-token dense tensor (attention q_a/q_b/kv_a/k_b/v_b/output,
indexer, shared expert, router, lm_head, dense-FFN layers 0-2, the MTP draft's dense parts)
plus the KV caches; the four CPU sockets keep streaming only the routed experts (13.9 GB/token).

| card | GPU-side per token | est. raw | est. MTP prose | agentic |
|---|---:|---:|---:|---:|
| V100 32 GB | ~16 ms | ~13 | 16-18 | 20+ |
| V100 16 GB (Q5_K attention, <=16K ctx) | ~14 ms | ~13 | 16-18 | 20+ |
| P40 24 GB | ~42 ms | ~10 | ~13 | ~16 |

Code change required (one place): `llama_prepare_model_devices` in `src/llama.cpp` folds
every listed device into the single meta (tensor-parallel) device. Hybrid mode must build
the meta device from the `CPU-NUMA*` devices only and keep `CUDA0` as a separate device,
give all layers to CUDA0 and route `ffn_*_exps` to the meta buffer type with tensor
overrides (`-ot`). The scheduler already moves the 24 KB residual GPU<->CPU per layer.
Validate on the 4-layer oracle (bit-exact vs CPU-only is not expected because CUDA
kernels accumulate differently; require top-10 logprob drift < 0.05 and `17*23 -> 391`).
