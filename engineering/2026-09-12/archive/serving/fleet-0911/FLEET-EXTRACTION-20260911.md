# Fleet extraction work — 2026-09-11

Goal as given: **GLM-5.3, GLM-5.3-Flash, Qwen3.8-Flash-Next and DeepSeek-V4.1-Flash each at
>70% of 380 GB/s (= 266 GB/s) with high tok/s.** Re-engineering llama.cpp / kernels / NUMA
explicitly in scope.

## 0. The denominator is now MEASURED, not assumed

Every prior document divides by "380 GB/s", a figure the user supplied and nobody had
verified. It is **correct within 0.4%**.

`/dev/shm/bwprobe/membw.c` — pinned threads, node-local `mbind(MPOL_BIND)` buffers, AVX-512
sequential reads:

| config | GB/s | % of 460.8 theoretical |
|---|---:|---:|
| 16 threads/socket x 4 (physical cores) | **380.5** | 82.6% |
| 32 threads/socket x 4 (with HT) | 381.7 | 82.8% |
| 24 threads/socket x 4 | 380.3 | 82.5% |
| 8 threads/socket x 4 | 333.5 | 72.4% |
| copy (nt-store) 16/socket | 317.0 | 68.8% |

Hardware: 4x Xeon Gold 6242 (Cascade Lake, 16c), **24 DIMMs all populated at 2400 MT/s**
= 6 channels/socket = 460.8 GB/s theoretical. 82.6% of theoretical is a normal DDR4 read
efficiency, so **381 GB/s is a hard wall** and 70% of 380 = 266 GB/s is a real target.

Two consequences:
- **Hyperthreading buys nothing** for streaming (381.7 vs 380.5). Do not chase it.
- **8 cores/socket already reach 87%** of the 16-core number. The memory system saturates
  with half the cores — cores are not the scarce resource for bandwidth.

## 1. The "199 GB/s" in the 09-10 docs is a window average, not the decode rate

Time-resolved IMC (`/dev/shm/bwprobe/bw.sh`, 10 ms; validated against membw at median
391 GB/s) shows decode bandwidth is **bimodal**: idle, or 200-292 GB/s. Measuring only the
active window on the live GLM-5.3-Flash Q4 server:

| | tok/s | GB/s window mean | median | % of 380 | stall samples |
|---|---:|---:|---:|---:|---:|
| before host isolation | 13.90 | ~94 (42 s window incl. idle) | — | — | — |
| after host isolation | 14.5-14.6 | **218-227** | 241-243 | 57-60% | **2.9%** |

**Decode does not idle at barriers** (2.9% of in-window samples below 50 GB/s). It streams
almost continuously, at ~63% of the 381 ceiling.

Per-socket at 3 ms resolution: **43.3 / 43.8 / 42.8 / 43.4 GB/s** — NUMA placement is
perfect and is not a lever. Each socket independently extracts ~50% of *its own* 95 GB/s.
So the deficit is **intra-socket**, not a tensor-parallel/NUMA problem.

## 2. Free win already banked: host isolation

`ps %CPU` is a lifetime average and hid this. Instantaneous load was ~4.5 cores of unpinned
background work — **ChatGPT desktop 176%**, dockerd 56%, containerd 51%, Paseo ~118%,
thunderbird 31% — spread across all 128 CPUs. Under `--split-mode tensor` every socket waits
at each layer barrier, so one stray thread taxes every token (known effect, ~30%).

`/dev/shm/bwprobe/quiet.sh` confines all of it to the 8-CPU pen `15,31,47,63,79,95,111,127`
(the 4 spare physical cores + their HT siblings) at runtime, plus `systemctl set-property
--runtime` for docker/containerd/netdata. **No docker restart** — another session was running
[client] tests. After: every non-pen core <10% busy.

The repo's `host-setup/isolate-background.sh` implements the durable version but **had never
been applied** (dockerd had no AllowedCPUs) and predates the current top offenders.

## 3. Where the time actually goes — measured, not inferred

`perf record` on the live server during decode (self time):

| | share |
|---|---:|
| **libgomp (barrier / spin-wait)** | **~35%** |
| `ggml_gemv_q8_0_x16_q8_0` | 14.7% |
| `ggml_gemv_q4_K_x16_q8_K` | 12.5% |
| `ggml_gemv_q5_K_x16_bytes` | 11.6% |
| other GEMV (q8_0_8x8, q6_K) | 1.7% |
| memmove | 2.6% |
| everything else (concat, hc_pre, flash_q8_sum, tinyBLAS, chunk_add, ...) | rest |

`perf stat`: IPC 0.52, **70.3% of cycles stalled**, 29.5% on L3 miss (DRAM),
**dTLB walk_active only 1.19%**.

- **TLB is not the problem.** THP for shmem is `[never]` and all 212 GB of the server's RSS
  is 4 KiB pages, but the pure-read probe also ran on 4 KiB pages and still hit 381 GB/s.
- **The GEMV kernels are not the problem.** `ggml_gemv_q4_K_x16_q8_K` walks its weight blocks
  perfectly sequentially and needs ~80 cycles per 2368-byte block = ~30 bytes/cycle/core of
  capability, against the ~2.5 bytes/cycle/core DRAM can actually supply. ~10x headroom.
  `repack.cpp` contains **zero** prefetch intrinsics and does not need them.
- Arithmetic that closes: GEMV is ~41% of time; if GEMV streams at ~95 GB/s/socket then the
  average is 0.41 x 95 = 39 GB/s/socket, vs 43-50 measured. **The GEMVs already run at
  roughly full bandwidth — they are simply only ~45% of the wall clock.**

**So tok/s and GB/s are the same lever: raise the fraction of decode spent inside GEMV.**

## 4. Barrier cost measured directly (`/dev/shm/bwprobe/barbench.c`)

| threads | libgomp `#pragma omp barrier` | engine's `GGML_CPU_OMP_SIMPLE_BARRIER` atomic spin |
|---:|---:|---:|
| 2 | 371 ns | 158 ns |
| 4 | 382 ns | 397 ns |
| 8 | 642 ns | 974 ns |
| **15** | **1279 ns** | **1929 ns** |

(taken while a model was loading, so absolutes are pessimistic; the ordering is the point)

- The engine's custom "simple barrier" is **1.5x slower than libgomp at 15 threads** — it is
  a flat atomic counter, which is the wrong algorithm at this width.
- Barrier protocol cost at ~7,000 ops/token x 1.28 us = **~9 ms/token = ~13%** of a 68 ms
  token. The profile charges libgomp ~35%, so the other ~22% is **straggler time**: threads
  waiting inside the barrier for work that is serialized elsewhere — principally the
  single-task ops that run on thread 0 while the other 14 threads spin
  (`GGML_CPU_SINGLE_TASK_MAX_ELEMENTS=4096`).

## 5. Negative results (09-11)

- `GGML_CPU_OMP_SIMPLE_BARRIER=1` + `OMP_WAIT_POLICY=active` + `GOMP_SPINCOUNT=infinite`
  **together are catastrophic**: server healthy but 42 decode calls in 10 minutes (~250x
  slower). Do not set `OMP_WAIT_POLICY=active` on this engine — with 4 NUMA device teams plus
  draft teams, the idle teams spin at full tilt against the working one.
- Hyperthreading does not raise streaming bandwidth (s.0).

## 6. Trap that cost a restart: the live server's libraries are NOT in its own bin dir

`fleet-0903/results/glm-flash-q8-r8-ordered-k-runtime-0908/bin/llama-server` has
`RUNPATH=.../engines/llama.cpp-glm5n-goal-0904/build-goal/bin`, but the validated config runs
a **different** library set via `LD_LIBRARY_PATH`:

```
LD_LIBRARY_PATH=<fleet-0903/results/glm-flash-q8-r8-ordered-k-0908/private-cpu>
               :<engines/llama.cpp-glm5n-goal-0904/validated-chunk16-bin>
```

`private-cpu/libggml-cpu.so.0.22.0` is a **binary-patched** library (09-08) implementing
`GGML_CPU_Q8_R8_ORDERED_K`, **which does not exist anywhere in the source tree**. Relaunching
without `LD_LIBRARY_PATH` silently falls back to the older `build-goal` library and aborts on
`GGML_ASSERT(n_as <= 256)` in `repack.cpp:7450`.

Exact restore point captured: **`fleet-0911/restore-baseline-0911.sh`** (argv + all 49 env
vars incl. `LD_LIBRARY_PATH`). Capture `/proc/<pid>/environ` **unfiltered** next time.

## 7. Baseline for all 09-11 comparisons

`fleet-0911/bench.sh` — 3 fixed prompts (prose / code / analysis), 192 tokens, temp 0,
seed 42, `cache_prompt:false`, IMC-measured active window. Host penned.

| | tok/s | GB/s | % of 380 |
|---|---:|---:|---:|
| GLM-5.3-Flash Q4, live config | **15.22** | **219.3** | **57.7%** |

## 8. The barrier is NOT the limiter — settled by experiment

`GGML_CPU_OMP_SIMPLE_BARRIER=1` **alone** measures **15.06 tok/s / 221.0 GB/s** against the
control's 15.20 / 222.2 — i.e. **nothing**, despite that barrier being **1.5x slower than
libgomp** at 15 threads in isolation (1929 vs 1279 ns, s.4).

If barrier *protocol* were on the critical path, swapping in a barrier that costs 650 ns more
x ~6,700 ops/token = +4.4 ms on a 66 ms token would have shown up as ~6% slower. It did not.

**Therefore the time inside the barrier is time spent waiting for the slowest worker, not
time spent executing the barrier.** Writing a faster (tree / per-thread-flag) barrier is
ruled out as a lever. The cost is load imbalance and serialization.

Corroboration from two independent instruments:
- bandwidth: 222 / 381 = **58% streaming fraction** => 42% of thread-time not streaming
- profile: **~35%** of thread-cycles in libgomp
These agree, so the perf-record distortion (that run ran at 7.2 vs 15.2 tok/s) did not
materially skew the share.

Within-token structure at 3 ms resolution: the dominant mode is a **steady ~200 GB/s
plateau** (46% of samples in 180-210, 19% in 210-240); only **3.9%** of decode exceeds
300 GB/s and 2.2% exceeds 350. So decode is not "full rate with gaps" — it is *partial rate,
steadily*.

### Why `OMP_WAIT_POLICY=active` / big `GOMP_SPINCOUNT` is catastrophic here

The server runs **two** models, each with **4 CPU-NUMA device teams** (`--spec-draft-device
CPU-NUMA0..3`): ~120 pinned worker threads over 60 physical cores, 2 per core. Trunk and
draft teams alternate. With libgomp's default policy the idle team **sleeps** and the
alternation is free. Raise the spin count and **both teams burn cores at 2x
oversubscription** — measured 42 decode calls in 10 minutes (~250x slower).

Do not set `OMP_WAIT_POLICY`, `GOMP_SPINCOUNT` or `KMP_BLOCKTIME` on this engine. (Note the
engine sets `KMP_BLOCKTIME=200` itself at `ggml-cpu.c:4778` — that is an **Intel** OpenMP
variable and GNU libgomp ignores it, which is why it is harmless.)

## 9. Target arithmetic

| | streaming fraction | GB/s | tok/s at 14.6 GB/generated token |
|---|---:|---:|---:|
| today | 58% | 222 | 15.2 |
| **goal (70% of 380)** | **70%** | **266** | **18.2** |
| all idle removed | 100% | 381 | 26 |

Reaching 266 GB/s requires cutting non-streaming time from 42% to 30% — a **29% reduction in
parallel-idle time**. It does not require a faster kernel: `repack.cpp`'s x16 GEMVs already
have ~10x headroom over what DRAM can feed them (s.3).

## 10. DeepSeek-V4.1-Flash: the blocker is RAM/disk arithmetic, and it is a user decision

Re-checked 09-11, not inherited:

| | |
|---|---|
| PyTorch reference (port 18170, still up, 24 h) | 1.24-1.30 tok/s, **capped at 1.87** even with both GEMMs instant |
| llama.cpp port | runs layers 0-3 and 8; full 40L needs cross-layer KV sharing (only layers [2,8,14,20] own a compressor) |
| checkpoint | 510 GB total; **306 GB minimum** for a speed run; never downloaded |
| `/models` free | **6.9 GB** | 
| `/` free | **6.7 GB** |
| `/dev/shm` free | 306 GB |

`/dev/shm` free space (306 GB) *looks* exactly sufficient, and the 09-10 note says so. It is
**not**, because that 306 GB is not additive: RAM is 755 GB total with 525 GB already in use,
252 GB of which is tmpfs — including the **186 GB GLM-5.3-Flash Q4 payload volume**. Staging
306 GB more would need 831 GB.

**So staging V4.1 requires giving up the live GLM-5.3-Flash Q4 tmpfs** (volatile by design,
re-stageable in ~50 min from the pinned HF revision via `stage_flash_q4_0910.py`). That trades
a working production model for an experimental one, and it still leaves the cross-layer-KV
graph work unimplemented afterwards — so it would not produce a V4.1 number in one sitting.
That is the maintainer's call, not a default.

**What is measurable instead:** its sibling **DeepSeek-V4-Flash-0731** (145 GB GGUF at
`/models/gguf/DeepSeek-V4-Flash-0731-UD-Q4_K_XL/`, launcher
`fleet-0903/launch_dsv4flash_tuned_0910.sh`) runs on the same tuned engine with the same
`deepseek4` graph and the same TP rules. It is the honest proxy for what V4.1 extraction will
look like, and it is where the 09-10 mirror-tax and MXFP4-kernel levers were identified.

## 11. GLM-5.3-Flash byte budget — measured traffic is already near-ideal

GGUF inventory (`/dev/shm/bwprobe/ggufinv.py` + `ggufkv.py`, header-only, no tensor reads):

- **199.7 GB**, 1412 tensors; q4_K 114.2 / q5_K 69.8 / q8_0 9.6 / q6_K 5.9 GB
- **95% of the model is MoE experts**: `ffn_down_exps` 72.4 + `ffn_gate_exps` 58.7 +
  `ffn_up_exps` 58.7 = 189.8 GB. Everything else is 9.9 GB.
- hparams: `expert_count=288`, `expert_used_count=8`, `block_count=46`,
  `leading_dense_block_count=3` (=> 42 MoE layers), `embedding_length=4096`,
  `vocab_size=154880`, **`hyper_connection.count=4`** (the old "six hyper-connection streams"
  note is stale — it is four), `nextn_predict_layers=1`.

Per-expert-per-layer = 189.8e9 / (42 x 288) = **15.69 MB**; 8 active = 125.5 MB/layer =>
**5.27 GB/token of expert traffic**, plus ~9.2 GB/token dense (everything but `token_embd`)
= **~14.5 GB for one unspeculated token**.

**Measured: 222.2 GB/s / 15.20 tok/s = 14.6 GB per generated token.** With speculation the
dense half amortises over ~1.85 tokens while expert traffic scales with the verify batch
(3 tokens route to up to 24 distinct experts/layer), and the 9.26 GB MTP draft adds ~0.9
GB/token. Those terms reconcile to the same number.

**Conclusion: GLM-5.3-Flash is not moving wasteful bytes.** Its bytes/token is within ~30% of
the speculation-ideal and essentially equal to the single-token ideal. There is no large
mirror tax to reclaim here (unlike deepseek4, where 09-10 found 2.87 GB/token mirrored) —
the experts, which are 95% of the model, are split by expert across sockets.

**Therefore the entire deficit is duty cycle, not traffic.** That splits the two goals:

| lever | effect on tok/s | effect on GB/s |
|---|---|---|
| remove redundant traffic | **up** | flat / down |
| **raise streaming duty cycle** | **up** | **up** |

Only the second serves both. That is what the concurrency/batching test targets.

## 12. Direct proof the GEMV kernel is not the bottleneck

`/dev/shm/bwprobe/gemvbw.c` dlopens the **production** `private-cpu/libggml-cpu.so.0.22.0`
and calls the real exported `ggml_gemv_q4_K_x16_q8_K` over node-local `mbind`ed weight
buffers (512 MiB/thread, x16 Q4_K layout, k=4096), pinned, measuring achieved DRAM bytes/s.

**One thread per socket: 11.2-12.2 GB/s per core** (measured while the box was otherwise
busy, so this is a floor).

A socket's ceiling is 95 GB/s, so **~8 cores saturate a socket** — and the server runs 15.
The kernel is not short of bandwidth, MLP, or prefetch (`repack.cpp` has zero prefetch
intrinsics and needs none: its block walk is perfectly sequential).

Combined with s.11 (bytes/token already near-ideal) this closes the argument:

> **100% of the GLM-5.3-Flash bandwidth deficit is time spent outside the GEMV kernels.**
> Not quantisation, not kernels, not NUMA placement, not TLB, not traffic volume.

The remaining levers are therefore only: (a) make the non-GEMV work smaller or parallel,
(b) make each GEMV bigger so the fixed per-op costs amortise (batching/concurrency).

## 13. GLM-5.3-Flash config results (all vs the same control, host penned)

| config | tok/s | GB/s | % of 380 | verdict |
|---|---:|---:|---:|---|
| **control** (production env, relaunched) | **15.20** | **222.2** | **58.5%** | reproduces the pre-restart baseline (15.22 / 219.3) exactly — harness validated |
| `GGML_CPU_OMP_SIMPLE_BARRIER=1` | 15.06 | 221.0 | 58.1% | neutral — see s.8, rules out barrier work |
| `GOMP_SPINCOUNT=100000000` | — | — | — | **catastrophic** (~0.3 decode/s); see s.8 |
| `GGML_CPU_NUMA_THREADS=12` | 14.45 | 175.8 | 46.3% | **-4.5% tok/s** (see correction below) |
| `GGML_CPU_NUMA_HUGEPAGES=1` | 12.65 | 175.2 | 46.1% | **neutral** once corrected (see below) |

Thread count: 15/socket is right. Lowering it hurts, and s.0 shows raising it cannot help
(the memory system already saturates at 8 cores/socket, and the server has 15). `t8` was
dropped from the queue as predictable.

### Correction: read the per-prompt rows, not the 3-prompt mean

Both "losses" above are partly artefacts, and one of them was **self-inflicted**:

| config | p1 | p2 | p3 |
|---|---:|---:|---:|
| control | 13.75 | 16.16 | 15.70 |
| `t12` | 13.17 | 15.31 | 14.87 |
| `hugepages` | **6.57 (contaminated)** | 15.96 | 15.41 |

- **`hugepages` is NEUTRAL (-1.5%), not -17%.** Its p1 was measured while I had a
  decode-synchronised `perf record` attached for 15 s. Never profile during a sweep.
  `AnonHugePages` did go 0 -> 189 GB, so the knob works; it simply does not help. (It also
  does not help for the reason one might expect from Intel's L2 streamer not prefetching
  across 4 KiB pages — that cost is evidently already absorbed.)
- **`t12` is a real but modest -4.5% on tok/s.** Its headline -21% GB/s came from a single
  bad counter window on p2 (115.6 GB/s mean while tok/s was a normal 15.31).

**Methodology rule that follows: report per-prompt medians, and never attach a profiler to a
run you intend to score.**

## 14. GLM-5.3 (Full) cannot be measured today without a destructive step — flagged, not taken

| | GB |
|---|---:|
| RAM total | 755 |
| tmpfs resident: GLM-5.3-Flash Q4 payload volume | 186 |
| tmpfs resident: `/dev/shm` (incl. the 40 GB DeepSeek-V4.1 native cache) | 73 |
| GLM-5.3 Full `UD-Q4_K_XL` (the **only** variant on the box; no smaller quant exists) | 436 |
| documented load **peak** for Full | ~628 |

628 + 259 = 887 GB against 755 GB of RAM. Even the steady 436 GB leaves under 30 GB of
slack. This matches the 09-10 allocation simulator, which reported an 18.46 GB shortfall and
concluded "it is **not** a successful Full load".

**Loading GLM-5.3 Full therefore requires unmounting the live GLM-5.3-Flash Q4 tmpfs** —
taking the production Flash model offline and costing a ~50 minute, 186 GB re-stage from the
pinned HF revision. That is a production decision and is **the maintainer's call**, so this session
did not take it. The 7.9-9.6 tok/s / 197-199 GB/s figures for Full in the 09-10 scoreboard
are **inherited from an earlier date when Full was the resident model**, not re-measured.

Note the arithmetic in s.9 already showed Full cannot meet 20 tok/s regardless: at 22.55 GB
per generated token it would need 451 GB/s against a **measured** 381 GB/s wall. Its
reachable ceiling on this box is ~17 tok/s, and only if extraction were perfect.

## 15. Undistorted decode profile (perf -F 97, no call graph, fired only while a request was in flight)

The first profile (s.3) used `-F 499 -g` and halved throughput, so its shares were suspect.
Re-taken at 97 Hz with no call graph, synchronised to `requests_processing==1`:

| symbol | self |
|---|---:|
| `ggml_gemv_q8_0_x16_q8_0` | 17.00% |
| **libgomp (barrier/idle-wait, summed)** | **~33%** |
| `ggml_gemv_q4_K_x16_q8_K` | 12.35% |
| `ggml_gemv_q5_K_x16_bytes_impl` | 10.98% |
| `__memmove_evex_unaligned_erms` | 2.33% |
| `flash_q8_sum_candidate<2>` | 1.84% |
| `tinyBLAS::gemm_bloc<4,3>` | 1.48% |
| `ggml_gemv_q6_K_x16_q8_K` | 1.42% |
| `ggml_gemv_q8_0_8x8_q8_0` | 1.17% |
| `ggml_compute_forward_dsv4_hc_pre` | 1.07% |

**GEMV total 42.9%, libgomp ~33%, everything else ~24%.** This matches the distorted profile
closely, so the earlier shares were sound after all — and it matches the bandwidth
independently (222/381 = 58% streaming fraction).

Note `ggml_gemv_q8_0_x16_q8_0` is the **single largest** consumer (17%) even though q8_0 is
only 9.6 GB of the 199.7 GB model (s.11) — that is the attention/dense path and the MTP
draft's `output.weight`, read in full every forward.

## 16. Measurement hazard discovered: the box drifts across load/unload cycles

After several 200 GB load/unload cycles, `/proc/buddyinfo` shows **node 0 with zero free
order-9 and order-10 blocks** (node 1 still has ~3,000 of each), and swap is 100% used.
`t12` and `hugepages` both landed at ~175 GB/s — suspiciously identical — so a **`control2`
re-run was appended to the queue** before trusting either as a real regression. Any sweep on
this box that does not re-measure its control at the end is not trustworthy.

## 17. Qwen3.8-Flash-Next byte budget

`UD-Q6_K_XL`, 169.2 GB, 1224 tensors (q8_0 104.1 / q6_K 64.7 GB). arch `qwen4exp`,
`block_count=48`, `embedding_length=2560`, `expert_count=512`, `expert_used_count=10`,
`expert_feed_forward_length=640`.

- **`per_layer_token_embd.weight` is 54.4 GB — 32% of the file — and is almost free per
  token**: it is a per-layer embedding table, one row indexed per layer per token. Anyone
  reasoning about "bytes/token" from file size will be 32% wrong on this model.
- experts 109.2 GB / (48 x 512) = **4.44 MB per expert per layer**; 10 used = 44.4 MB/layer
  => **2.13 GB/token**
- dense ~5.6 GB/token
- **ideal ~7.7 GB for one unspeculated token** (vs GLM-5.3-Flash's 14.5)

At the 09-10 figure of 5.18 GB per *generated* token (MTP4) and 28 tok/s, that is 145 GB/s =
**38% duty cycle — worse than GLM-5.3-Flash's 58%**, because 48 layers x 512 tiny 4.44 MB
experts x a 2560-wide hidden state is the most op-bound shape in the fleet.

**This unifies the fleet.** Every model's bandwidth number is `381 GB/s x duty cycle`:

| model | duty cycle | GB/s | what 70% (266 GB/s) would give it |
|---|---:|---:|---|
| GLM-5.3-Flash Q4 | 58% | 222 | 18.2 tok/s |
| Qwen3.8-Flash-Next Q6 | 38% | 145 | ~52 tok/s |
| DeepSeek-V4-Flash | 47% | 178 | ~38 tok/s |
| GLM-5.3 Full Q4 | 52% | 198 | ~11.8 tok/s |

So there is exactly **one** engineering target for all four models — raise the fraction of
decode spent inside the GEMV kernels — and it raises tok/s and GB/s together in every case.
No per-model trick is needed or available.

## 18. Concurrency/batching: raises throughput 61%, but LOWERS bandwidth utilisation

One load, `--parallel 8 --ctx-size 32768`, four concurrency levels, same host pen.

| C | per-stream tok/s | aggregate tok/s | GB/s mean | duty cycle | GB/generated token |
|---:|---:|---:|---:|---:|---:|
| 1 | 18.17 | 18.2 | 198.7 | 52% | 10.9 |
| 2 | 11.44 | 22.9 | 199.3 | 52% | 8.7 |
| 4 | 6.72 | 26.9 | 175.4 | 46% | 6.5 |
| 8 | 3.66 | **29.3** | 172.8 | **45%** | 5.9 |

(`benchpar.sh` originally printed a deflated "aggregate" because a bare `wait` also waited on
the 38 s IMC sampler; the correct aggregate is `C x per-stream`. Fixed.)

**The hypothesis was that bigger ops would amortise the fixed per-op costs and raise the
duty cycle. It does not happen.** Aggregate throughput scales well (+61% from C=1 to C=8),
but bandwidth utilisation *falls* from 52% to 45%, because batching amortises the dense half
of the forward pass across sequences and **halves bytes per generated token** (10.9 -> 5.9).

Likely reason the duty cycle does not rise to compensate: at C=8 a layer routes to up to
8x8 = 64 distinct experts of 288 instead of 8, so the expert gather becomes far more
scattered — more, smaller, less sequential reads.

**So batching is the right answer for aggregate throughput and the wrong answer for the
utilisation target.** It is another instance of the identity in s.9: anything that makes a
token cheaper lowers the percentage.

## 19. What 70% of 380 GB/s would actually require

From the measured decomposition (s.15): GEMV 42.9% of thread-time, libgomp idle ~33%,
other ops ~24%. If the GEMVs stream at the socket ceiling, they account for
0.429 x 381 = 163 GB/s; the measured 222 GB/s means the remaining ~59 GB/s comes from
non-GEMV streaming (memmove, KV reads, the cross-socket all-reduce).

To reach **266 GB/s** the GEMV share must rise from 42.9% to ~54.5%, i.e.
**non-GEMV time must fall by ~20%** (57.1% -> 45.5% of the token).

Non-GEMV time is ~33 points of barrier/straggler idle plus ~24 points of many small ops.
Nothing in the knob space touches it:

| tried | result |
|---|---|
| faster/slower barrier | 0% (s.8) |
| more threads | cannot help; memory saturates at 8 cores/socket (s.0) |
| fewer threads | -4.5% (s.13) |
| huge pages | neutral (s.13) |
| libgomp spin tuning | catastrophic (s.8) |
| batching / concurrency | +61% aggregate tok/s but duty cycle **falls** (s.18) |

**The remaining lever is graph-level: fewer, larger ops per layer.** GLM-5.3-Flash runs
~6,700 ops/token (~146 per layer x 46 layers). Roughly halving that is what "70%" costs.
That is an engine project in `src/models/glm5next.cpp` (fusing the 4-stream hyper-connection
residual chain and the KDA elementwise chain), not a configuration change, and its payoff is
uncertain until tried.

**Recommendation: treat 20+ tok/s as the goal and report utilisation as a diagnostic.** The
two criteria are the same lever only through duty cycle, and duty cycle is now measured,
bounded, and shown to be untouchable from configuration.

## 20. CORRECTION: the bandwidth window detector was wrong, and the box drifts ~5%

`control2` (the drift check) first reported **106.9 GB/s** — impossible next to its own
14.25 tok/s, which at 14.6 GB/token demands ~208 GB/s. The **instrument was fine** (re-
validated live: `membw` wall-clock 377.0 GB/s vs sampler 377.3, 99.3% agreement, counters
100% running, no multiplexing). **The analyser was wrong.**

`bwsum.py` took the decode window as *first-to-last sample above 50 GB/s*. A single isolated
spike in the idle tail therefore stretched the window across all the idle samples and dragged
the mean down. The two 2-second-bucket traces are almost identical:

```
control : 72 236 232 236 233 232  51  8  6 10 11  9 10 11  8  7
control2: 76 221 225 227 226 226 141 20 17 18 18 17 11 15 15 14
```

Fixed in `/dev/shm/bwprobe/bwsum2.py` and `bench2.sh`: take the **longest contiguous run**
above threshold, allowing gaps of <=0.25 s.

### Re-scored, and the honest verdict

| config | tok/s | GB/s | % of 380 |
|---|---:|---:|---:|
| control (start of session) | 15.20 | 222.2 | 58.5% |
| `GGML_CPU_OMP_SIMPLE_BARRIER=1` | 15.06 | 221.0 | 58.1% |
| `FUSED_REDUCE_SINGLE_MAX_ELEMENTS=1024` | 14.82 | 218.2 | 57.4% |
| `GGML_CPU_NUMA_THREADS=12` | 14.45 | 205.5 | 54.1% |
| `GGML_CPU_NUMA_HUGEPAGES=1` | (contaminated) | 207.2 | 54.5% |
| **control2 (end of session, same config)** | **14.25** | **213.4** | **56.1%** |
| `--spec-draft-n-max 1` | 9.66 | 188.5 | 49.6% |

**`control2` is 6% below `control` on identical settings.** The box drifts downward across
repeated 200 GB load/unload cycles (node 0 lost all order-9/10 free blocks; swap 100% used).
**Every "regression" above except `nmax1` is inside that drift.**

> **Verdict: no configuration knob moves GLM-5.3-Flash. The production config is already
> optimal**, and `--spec-draft-n-max 2` is firmly correct (depth 1 costs 36% of tok/s).
> This is a negative result, but it is now an exhaustive and well-controlled one.

**Methodology rules earned here:** always re-measure the control last; never profile a scored
run; use the longest-contiguous-run window; and report per-prompt values, not just means.

## 21. The GEMV kernel extracts 92% of the machine — measured on a quiet box

`gemvbw` (dlopens the production `private-cpu/libggml-cpu.so`, calls the real exported
`ggml_gemv_q4_K_x16_q8_K` over node-local `mbind`ed 512 MiB/thread weight buffers), server
stopped, host penned:

| threads/socket | real GEMV kernel | % of 381 |
|---:|---:|---:|
| 1 | 22.0 GB/s | 5.8% |
| 4 | 86.0 | 22.6% |
| 8 | 162.8 | 42.7% |
| **15** | **277.2** | **72.8%** |

Pure-read reference **at the same thread count and the same moment: 301.2 GB/s**.

> **The real quantised GEMV kernel reaches 92% of what a bare AVX-512 read loop achieves.**

There is nothing left in the kernel. Prefetch, MLP, quant format and repack layout are all
fine. Combined with s.11 (bytes/token already near-ideal) and s.15 (GEMV is only ~43% of
decode), the whole deficit is confirmed to be **time spent outside the GEMV**.

*Caveat:* the pure-read reference read 301 GB/s here at 15 threads/socket versus 380 GB/s at
16 threads/socket at session start (s.0), i.e. the box had itself degraded ~20% by this point
in the session (see the fragmentation/drift note in s.16 and s.20). The GEMV-vs-read ratio is
taken from the two measurements made back to back, so it is unaffected; the absolute numbers
in this section are not comparable to s.0.

## 22. Which engine serves which model (this cost two failed launches)

The fleet needs **two** runtimes, and the GLM one silently carries a Qwen-looking env var
(`GGML_Q4E_SPLIT=13`) that is **not** evidence it can serve Qwen.

| runtime | architectures it registers |
|---|---|
| `engines/llama.cpp-glm5n-goal-0904/validated-chunk16-bin` (+ the `private-cpu` libggml-cpu) | `glm5next`, `glm-dsa`, **`deepseek4`**, `dflash` |
| `engines/llama.cpp-q4e-goal-0904/validated-iq-batch3-bin` | **`qwen4exp`** |

Pointing the GLM binary at Qwen fails instantly with
`error loading model: unknown model architecture: 'qwen4exp'`.

**And when launching Qwen you must DROP `LD_LIBRARY_PATH` from the GLM env** (it pins the GLM
`private-cpu` libggml-cpu, which would override the q4e runtime) and point it at the q4e bin
directory instead. Everything else in the 49-var env carries over safely.

Two launcher bugs worth remembering:
- `pgrep -f '^/home/user/.*/bin/llama-server'` requires a literal `/bin/`. The Qwen runtime
  lives in `validated-iq-batch3-bin/`, so the pattern does **not** match and the launcher
  reported `DIED` while the model was loading perfectly well. Use `kill -0 "$SRV"` on the PID
  captured from `$!` instead of pattern-matching a process list.
- DeepSeek-V4-Flash reuses the GLM runtime (arch `deepseek4`) and its own launcher
  `fleet-0903/launch_dsv4flash_tuned_0910.sh` on port **18132**, with the same 49-var env —
  including `LD_LIBRARY_PATH`, which there is correct.

## 23. Qwen: do not hand-assemble its env — the maintainer has a pinned recipe

`fleet-0903/qwen-flash-20tps.json` holds the validated Qwen3.8-Flash-Next configuration:
`command` (full argv), `runtime_env` (38 vars), `binary_sha256`, `evidence`, and
`measured_rates` (**prose 21.3-21.5, code 28.1-28.5 tok/s**). Use it verbatim.

Two independent ways I got this wrong before finding it, both of which fail *after* a full
~160 GB load so they look like a broken model rather than a bad flag:

1. **Inheriting GLM's 49-var env** -> `GGML_ASSERT(n_experts <= 256)` at `repack.cpp:7200`.
   GLM sets `GGML_CPU_X16_Q8_EXPERTS=1`, and that expert path has a fixed `int active[256]`.
   Qwen has **512 experts**. The pinned env pointedly does **not** set that knob.
2. **Hand-curating a minimal env** (NUMA knobs only) -> loads and serves, but at
   **8.33 tok/s / 106.0 GB/s (27.9%)** because it drops every x16 kernel knob
   (`X16_Q4_K`, `X16_Q5_K`, `X16_Q6_K`, `X16_Q8_0`, `X16_Q8_BATCH`, `IQ_R16_*`, the repack
   knobs) that Qwen's Q6_K/Q8_0 weights need. A "clean minimal env" is a 2.5x regression here.

Also: the pinned `LD_LIBRARY_PATH` layers **three** private builds ahead of the engine —
`qwen-expert-even-split-policy-0906b/private-split`, then `qwen-q6-q8-wide-batch-0907/private-cpu`,
then `validated-iq-batch3-bin`. A single-directory `LD_LIBRARY_PATH` silently loses two layers
of validated work. Its speculation block is **MTP n_max=4 with p_min 0.3** against the Q8_0
draft (`/models/gguf/Qwen3.8-Flash-Next/MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf`) — note that is
the *deeper, higher-threshold* setting, unlike GLM-5.3-Flash's n_max=2 / p_min 0.0.

**General rule for this box: before benchmarking a model, look for a pinned
`*-<target>.json` recipe in `fleet-0903/` and start from its `command` + `runtime_env`.**

## 24. If someone wants to actually chase 70%, here is the only remaining path

Everything cheap is exhausted and measured (s.13, s.18, s.20). What is left is graph work, and
the harness to do it safely already exists.

**Target, quantified:** raise the GEMV share of decode from **42.9% to ~54.5%**, i.e. cut
non-GEMV time by ~20%. Non-GEMV is ~33 points of straggler idle plus ~24 points of many small
ops. GLM-5.3-Flash runs **~6,700 ops/token** (46 layers x ~146). Roughly halving the per-layer
node count is the size of the job.

**Where:** `src/models/glm5next.cpp` — the 4-stream hyper-connection residual chain
(`hyper_connection.count=4`, so every norm/concat/add runs four times) and the KDA elementwise
chain. Equivalent work exists in `qwen4exp` (48 layers, 512 experts of 4.44 MB, 2560-wide — the
most op-bound shape in the fleet at 38% duty cycle).

**How, without risking the production library:**
`fleet-0903/build_flash_q8_r8_ordered_k_0908.py` is a rigorous private-build harness that
(1) re-links the parent's objects and asserts the result is **byte-identical** to the shipped
library, (2) recompiles the *unmodified* source and asserts its `.text` is byte-identical,
(3) compiles only the modified translation unit and re-links swapping that one object, and
(4) writes a manifest with every input sha256. That is how `private-cpu/libggml-cpu.so` (the
library the production server actually loads) was made, and it is how any kernel/graph change
should be made.

**Do NOT spend more time on:** barriers (s.8), thread count (s.13), huge pages (s.13), libgomp
spin policy (s.8), NUMA placement (s.1), prefetch or quant kernels (s.12, s.21), all-reduce
threshold (s.20), speculation depth on GLM (s.20), or batching *for the utilisation metric*
(s.18 — it is right for throughput and wrong for the percentage).

**Expected payoff is uncertain.** Halving op count is a plausible route to ~70% on
GLM-5.3-Flash and DeepSeek, but Qwen at 38% duty cycle would need more than that, and GLM-5.3
Full cannot reach 20 tok/s at any duty cycle (s.9: it needs 451 GB/s against a measured 381).

## 25. Qwen3.8-Flash-Next measured (pinned recipe, host penned)

| config | tok/s | GB/s | % of 380 | GB per generated token |
|---|---:|---:|---:|---:|
| pinned production (MTP n_max=4, p_min 0.3, Q8 draft) | **19.90** | 134.6 | 35.4% | 6.76 |
| same env, speculation OFF | 16.43 | **181.2** | **47.7%** | 11.03 |
| (my hand-curated "minimal" env, for contrast) | 8.33 | 106.0 | 27.9% | — |

per-prompt MTP4: prose 15.61 / code 21.66 / analysis 22.44 — consistent with the pinned
recipe's recorded 21.3 prose / 28.1-28.5 code once host drift and different prompts are
allowed for.

**This is the clearest demonstration of the conflict in the goal.** On the *same model, same
engine, same env*, turning speculation off **raises bandwidth utilisation from 35.4% to 47.7%
(+35%)** while **lowering tok/s from 19.90 to 16.43 (-17%)**. Speculation converts DRAM
traffic into tokens: 6.76 GB per generated token with it, 11.03 without.

Neither config reaches 70%. Qwen's duty cycle is the fleet's worst (its raw 47.7% is its
ceiling) because 48 layers x 512 experts of only 4.44 MB x a 2560-wide hidden state is the
most op-bound shape here — the most ops per byte moved.

## 26. What a fleet sweep costs a co-resident service (measured by the other side)

The [client] session ran its portal gate against the same box during and after this sweep —
same config, same two prompts, wall-clock through its real routing path:

| request type | during the sweep | quiet host |
|---|---:|---:|
| routed report (short completion) | ~2.9 s median | ~2.9 s median — **unaffected** |
| explained report (long completion) | 14.7 s and **38.4 s** | 14.4 s and **17.0 s** |

**Short completions ride through contention; long ones degrade up to ~2.3x.** That is the
same effect as s.2 seen from the opposite direction: contention costs *forward-pass rate*, so
it is invisible in anything dominated by fixed per-request cost and brutal in anything
dominated by steady-state decode.

Two consequences:
- **Penning background load is not sufficient when another model is resident.** s.2's pen
  handles unpinned desktop/daemon threads; it does nothing about a second llama-server whose
  worker threads are legitimately busy. Fleet sweeps and co-resident services need
  *sequencing*, not just isolation — announce long runs to peers.
- A service whose workload is short completions (routing, classification, 1-3 sentence
  answers) can safely share this box with tuning work. One whose workload is long generations
  cannot.

Also worth recording, from the same exchange: **`measured_rates` in the pinned JSONs is not
provenance — it is the detector.** A wrong-env config that still answers correctly is only
identifiable by comparing its rate against a known-good figure (s.23: 8.33 vs 19.90 tok/s,
correct output throughout). Pair a raw `/completion` probe with a `measured_rates` check;
neither alone covers the class.

## 27. CAPACITY CONFLICT: GLM-5.3-Flash and a second resident model do not fit

Found the hard way at 12:42 — the [client] portal assistant (Qwen3.8-Flash-Next, ~119 GB
anon) came up on 18097 and the kernel **OOM-killed the restored GLM-5.3-Flash production
server**, then killed a second GLM load at 12:50.

Both kills say `constraint=CONSTRAINT_MEMORY_POLICY, nodemask=3` — **NUMA-node-constrained,
not global**. The machine had 318 GB available throughout.

| | node 0 | node 1 | node 2 | **node 3** |
|---|---:|---:|---:|---:|
| MemFree (after `drop_caches`) | 88.8 | 99.6 | 95.2 | **41.5** |
| Shmem (tmpfs — **not reclaimable**) | 55.6 | 54.0 | 54.5 | **94.7** |
| peer's Qwen anon | 28.9 | 29.0 | 32.0 | 29.0 |

GLM-5.3-Flash needs **~52.7 GB of anon per node** (211 GB, `--tensor-split 1,1,1,1`, strict
per-node `mbind`). Node 3 offers 41.5 GB. **An 11 GB shortfall on one node, with 318 GB free
machine-wide.**

**Root cause is the tmpfs distribution, not the peer.** The GLM Q4 payload volume is spread
unevenly and node 3 carries ~40 GB more than its share. `drop_caches` took nodes 0-2 from
~41 GB to 88-99 GB free but barely moved node 3, because ~94.7 of its 95.3 GB of "FilePages"
**is** shmem and therefore unreclaimable. The peer's 29 GB on node 3 is merely what pushes it
over the line.

**`free -g` cannot show this class of problem** — it reports the machine, and the allocation
is per-node. This is the same failure documented in `glm53-full-numa-constrained-oom`; that
note's `drop_caches` remedy is necessary but **not sufficient here**, because the blocking
memory is tmpfs rather than page cache.

### Measured after a clean GLM load, nothing else resident

    node        0      1      2      3
    MemFree   65.3   76.1   73.6    9.2  GB

Node 3 falls to **9.2 GB** — worse than the ~16 GB both sessions projected. Against a second
model's ~29 GB share that is a **~20 GB shortfall**, so co-residency is not marginal, it is
impossible.

### Root cause pinned: the tmpfs has no NUMA memory policy

```
flash-q4-0910 ... type tmpfs (rw,nosuid,nodev,noexec,relatime,size=220200960k,mode=700,...)
```

**No `mpol=`**, so pages were placed **first-touch** during staging. tmpfs supports
`mpol=interleave`; mounting future payload volumes that way (or staging under
`numactl --interleave=all`) makes this class of failure impossible. That is the durable fix
and it is free for any *new* volume.

### Options, none of which this session should pick unilaterally

1. **Accept non-coexistence** — only one large model resident at a time; sequence the portal
   assistant and fleet work. Simplest, and matches today's behaviour.
2. **Run GLM with `GGML_CPU_NUMA_REPACK=0`** — drops the ~211 GB of node-local anon copies
   entirely (the server reads the tmpfs mapping directly), at a real speed cost and as a
   deviation from the validated config. Untested.
3. **Re-stage the Q4 tmpfs with `mpol=interleave`** — fixes the actual cause permanently via
   `fleet-0903/stage_flash_q4_0910.py`, but costs ~30-50 min with production **down** and a
   window where the only copy of a live model is a partial download. **Recommended, but
   the maintainer's call.**
4. **Uneven `--tensor-split`** (e.g. `1,1,1,0.8`) — deviates from the validated configuration
   and changes the performance profile.

**Whichever is chosen, the current state resolves itself by OOM-killing whichever model loaded
most recently, which is the worst of the four.**

## 28. Op profile — the inherited "~6,700 ops/token" is confirmed, and the target is NOT the tiny ops

Captured with the in-engine profiler (`GGML_CPU_OP_PROFILE`, 4 graphs, quiet box, production
restored afterwards). **Absolute times are inflated ~2.2x** by the profiler's two
`ggml_time_us()` calls per node across 7,239 nodes; the *shares* are sound.

**Decode graph = 7,239 nodes** (the 09-04 estimate of ~6,700 was close and is now confirmed
for the current Q4 model).

| op | share of in-op time | count | avg |
|---|---:|---:|---:|
| **MUL_MAT_ID** (MoE experts) | **40.9%** | 378 | 466 us |
| **MUL_MAT** (dense) | **31.5%** | 2,049 (**683/graph, 14.8/layer**) | **66 us** |
| GATED_DELTA_NET (KDA) | 4.8% | 102 | 202 us |
| CUSTOM | 3.4% | 270 | 54 us |
| UNARY | 3.1% | 435 | 31 us |
| SCALE | 2.4% | 225 | 46 us |
| GET_ROWS | 2.1% | 271 | 34 us |
| RMS_NORM | 1.7% | 532 | 14 us |
| CONCAT | 1.6% | 415 | 17 us |
| MUL / CPY / DSV4_HC_PRE / DSV4_HC_POST / ADD / CLAMP / CONT | 1.4% and below each | 154-270 each | 8-38 us |

### This corrects the fusion target

**Matmul is 72.4% of in-op wall time** (MUL_MAT_ID 40.9 + MUL_MAT 31.5) against the perf
profile's 42.9% of *thread-cycles* in GEMV. The gap is threads idle **inside** matmul ops —
imbalance within the op, not between ops.

The ~4,800 non-matmul nodes are **66% of the node count but only 27.6% of the time**, and the
barrier cost of all of them is ~4,812 x 1.28 us = **6.2 ms against a ~143 ms graph, ~4%**.
So the earlier plan — "fuse the hyper-connection residual chain and the KDA elementwise
chain" — targets a 4% prize, not the 20% needed. **s.24's framing was wrong in its choice of
target, though right that the work is graph-level.**

### CORRECTION: the counts above are totals across 3 graphs, not per-graph

The profiler emitted **3** decode graphs of 7,239 nodes. So per graph: **683 `MUL_MAT`
(14.8/layer)**, 126 `MUL_MAT_ID` (2.7/layer), 7,239 nodes (157/layer). An earlier reading of
"2,049 = ~44 per layer" was **3x too high**, and with it the inference that some path is
decomposed per-head or per-stream: 14.8/layer matches the **18 `ggml_mul_mat` call sites in
`glm5next.cpp` directly**, with no hidden decomposition to fuse away.

**So "batch the dense matmuls" is NOT available as stated** — they are already one call per
weight tensor per layer. The opportunity is per-op efficiency, not op count.

### Where the dense matmul time actually goes (by node name, 3 graphs)

| node | n | ms | % of MUL_MAT | avg us | src0 |
|---|---:|---:|---:|---:|---|
| `node_N` (KDA/attn projections: ssm_f_a/b, ssm_beta, ssm_g_a/b, wq/wk/wv) | 1047 | 59.66 | 43.9% | 57.0 | q8_0 [4096,2048] |
| **`hc_mixes`** | 270 | **20.72** | **15.3%** | **76.8** | q8_0 **[16384,24]** |
| `kda_out` | 102 | 11.81 | 8.7% | 115.8 | q8_0 [2048,4096] |
| `ffn_moe_logits` | 126 | 9.05 | 6.7% | 71.8 | f32 [4096,288] |
| `ffn_gate` / `ffn_up` | 135 / 135 | 8.14 / 7.51 | 11.5% | ~58 | q8_0 [4096,3072] |
| `ffn_shexp` | 126 | 6.97 | 5.1% | 55.3 | q8_0 [512,4096] |
| `dsa_out` | 33 | 6.94 | 5.1% | 210.5 | q8_0 [4096,4096] |
| `fattn_mla` / `indexer_gate` / `ffn_out` | 33/33/9 | 2.34/0.95/1.74 | 3.7% | — | — |

**`hc_mixes` is the anomaly.** Its weight is `[16384,24]` q8_0 — 24 output rows, ~418 KB, about
**0.1% of the bytes a layer moves** — yet it takes **76.8 us** and **15.3% of all dense matmul
time**. At socket bandwidth that tensor should stream in ~1-4 us. Two plausible causes, both
structural: 24 rows cannot fill 15 threads (1-2 rows each, ~2x imbalance), and the x16 repack
kernels require `nc % 16 == 0`, which 24 fails — so it falls to a generic path.
`indexer_gate` ([4096,128], 28.8 us for ~524 KB) shows the same signature more mildly.

**Realistic size of this prize:** `hc_mixes` is ~4.8% of total in-op time; recovering most of
it is worth ~4-5%. Bringing `node_N` (13.8% of in-op time, 57 us against a ~22 us
bandwidth-bound ideal under TP4) to ideal would be worth ~8%. Together ~13% — real, and still
short of the ~20% that 70% utilisation needs. There is no single large lever here.

## 29. Built a correct instrument — and it overturns the barrier diagnosis

s.28's per-op times were **compute + the following barrier** (`ggml_barrier` at ggml-cpu.c:4013
sits inside the window that closes at :4017), so they cannot say "this op is slow". Two
conclusions were drawn from them and both were wrong. Rather than guess a third time, I
patched the profiler to record the two separately.

### The build, with the harness's verification discipline

The production library layers **two** private source modifications, not one: `repack.cpp` from
`glm-flash-q8-r8-ordered-k-0908` and **`ggml-cpu.c` from `glm-flash-q8-pool-0908c`**. Building
from the engine tree would silently drop the pool-fusion work *and* abort on
`GGML_ASSERT(n_as <= 256)` (the tree still has `int active[256]`; this model has 288 experts).
So the change was layered onto **their** private `ggml-cpu.c`, using **their** recorded
`compile_commands` (not `build-goal`'s — those differ and do not reproduce).

Verification, both steps passed before trusting anything:
- compile their unmodified private source -> `.text` **byte-identical** to their shipped
  `ggml-cpu.c.o` (53,161 bytes)
- relink their objects unchanged -> **byte-identical** to the production library
  (`sha256 5ba125771f3e8bac`)

Then relink swapping only `ggml-cpu.c.o`. Sources and scripts: `/dev/shm/profbuild/`.

### Result: thread 0 spends 92.4% of its time COMPUTING

| | per graph |
|---|---:|
| compute | ~136 ms |
| barrier | ~12 ms |
| **barrier share** | **7.6%** |

The perf profile's ~33% libgomp is therefore **not** barrier overhead and **not** symmetric
straggling. Thread 0 computes 92% of the time while the *average* thread idles 33% — so
**thread 0 is the straggler and the other 14 wait on it.** Both measurements were right; the
interpretation ("threads wait for each other at barriers") was wrong.

**The tell is a LOW barrier percentage**: an op where thread 0 barely waits is one where
thread 0 finished last, i.e. it carried disproportionate work.

### The four ops running at roughly single-core speed while 14 threads idle

| op | % of thread-0 time | per op | effective rate | barrier% |
|---|---:|---:|---:|---:|
| `MUL_MAT:hc_mixes` `[16384,24]` | 4.6% | 73.8 us | **5.7 GB/s** | 3.0% |
| `CUSTOM:meta_fused_reduce_N` | 3.7% | 58.6 us | — (the cross-socket all-reduce, s.?) | **1.0%** |
| `UNARY:node_N` `[128,16,4]` | 2.6% | 56.2 us | **0.58 GB/s** (32 KB!) | **0.7%** |
| `SCALE:cache_s_lN` `[262144]` | 2.1% | 91.0 us | **23 GB/s** | 1.6% |
| **total** | **~13%** | | against a 95 GB/s/socket ceiling | |

`meta_fused_reduce` is the single-threaded scalar sum + memcpy all-reduce read from the source
earlier — now confirmed as a measured 3.7% with every other thread idle. `SCALE:cache_s_lN` and
the KDA state ops match the long-standing "recurrent state moves at ~6 GB/s" note.
`UNARY` at 0.58 GB/s on 32 KB is pure dispatch overhead, not bandwidth.

**Productive work for contrast:** the three MoE expert matmuls (`ffn_moe_down/gate/up`) are
**40.1%** of thread-0 time at 380-615 us each — that is the real bandwidth work, and it is fine.

### Honest sizing

Parallelising all four is worth **up to ~13%** of wall time, which would take GLM-5.3-Flash
from 58.5% to roughly **66%** utilisation — a large, real gain, and still short of 70%.
This is the first target list on this box backed by an instrument that can actually
distinguish compute from waiting.

## 30. IMPLEMENTED: parallelise the elementwise UNARY ops — +1.7% tok/s, bit-exact

First actual engine change of the session, not a recommendation.

**The defect.** `ggml_get_n_tasks` hard-codes `n_tasks = 1` for `GGML_OP_UNARY` with
`EXP / SOFTPLUS / SIGMOID / TANH / ABS / STEP / ...` — regardless of tensor size. But the
implementation **already honours `ith/nth`** (`unary-ops.cpp`, `get_thread_range`). So on
GLM-5.3-Flash's KDA path those ops run single-threaded on `ne=[128,16,4]` at **56.2 us each
and 0.58 GB/s**, with the other 14 workers idle — 2.6% of thread-0 time thrown away for no
reason. Same for `GGML_OP_SCALE`, which is grouped with `RESHAPE`/`VIEW`/`PERMUTE`.

**The change** (`parallel-unary-0911/ggml-cpu.c.patch`, 104 lines incl. the profiler work):
opt-in `GGML_CPU_PARALLEL_UNARY=<min_elements>`; above that threshold those ops get
`n_tasks = n_threads`. Elementwise means **no reduction**, so splitting is bit-exact.

**Verification, all four steps:**
1. compile the parent's unmodified private source -> `.text` **byte-identical** to its object
2. relink the parent's objects unchanged -> **byte-identical** to the production library
3. `unary-off` reproduces the production control (15.23 vs 15.20 tok/s) -> the patched library
   is otherwise equivalent
4. greedy output **byte-identical on all 3 prompts**, flag on vs off

**Result, back-to-back on the same library (controls for drift):**

| arm | tok/s | GB/s | % of 380 |
|---|---:|---:|---:|
| `GGML_CPU_PARALLEL_UNARY` unset | 15.23 | 219.4 | 57.7% |
| **`GGML_CPU_PARALLEL_UNARY=4096`** | **15.49** | **222.4** | **58.5%** |

**+1.7% tok/s, +1.4% GB/s.** Smaller than the ~2.4% predicted from the profile, which is the
usual gap between "time attributed to an op" and "time recoverable from it".

### Why SCALE was NOT changed, despite looking identical

`SCALE` splits by **rows** (`nr = ggml_nrows(src0)`), and the expensive instance is
`cache_s_lN` with `ne=[262144,1,1]` — **one row**. Raising its `n_tasks` would hand every
thread but one an empty range and change nothing. Recovering that 2.1% needs a column-split
path in `ggml_compute_forward_scale_f32`, which is a real kernel change, not a scheduler one.

### And why `meta_fused_reduce` is not the target it appeared to be

Its 58.6 us with 1.0% barrier is **not** single-threaded arithmetic: the op already takes
`(ith, nth)`, and its cost is the **cross-socket rendezvous**
(`ggml_backend_meta_fused_spin_until(d->arrive1, n_devs)`) — 4 sockets meeting ~90 times per
graph. The code deliberately single-threads small reductions and says so in a comment. That
3.7% is the inter-socket tensor-parallel tax; reducing it means **fewer reduce points**, i.e.
a graph change, not a parallelisation fix.

**Artifacts:** `fleet-0911/parallel-unary-0911/` — patched + parent sources, unified diff,
build and relink scripts, and a manifest recording every sha256 and all four verifications.

## 31. The structural answer: a ~20-35 us fixed dispatch floor per parallel op

Isolated `ggml_mul_mat` harness against the **production** libraries
(`parallel-unary-0911/mmbench.c`), rotating through 192 MB of distinct weight copies so it
**streams from DRAM** rather than replaying out of L3 — the first version of this benchmark
fit in the 22 MB L3 and gave misleading scaling, so that confound is closed.

One socket, `taskset -c 0-14`:

| shape (k, output rows, tokens) | 1 thread | 15 threads | scaling | GB/s @15 |
|---|---:|---:|---:|---:|
| q8_0 [16384, **2048**] x3 | 3331 us | 423 us | 7.9x | **84.3** (89% of the 95 GB/s socket ceiling) |
| q8_0 [4096, 2048] x3 (`node_N`) | 1022.8 | 167.4 | 6.1x | 53.3 |
| q8_0 [16384, 240] x3 | 452.4 | 92.3 | 4.9x | 45.2 |
| q8_0 [16384, 48] x3 | 124.8 | 55.3 | 2.3x | 15.1 |
| q8_0 [16384, **24**] x3 (`hc_mixes`) | 82.5 | 47.6 | **1.7x** | 8.8 |

And varying `k` at 24 output rows exposes the mechanism directly:

| k | 1 thread | 15 threads |
|---|---:|---:|
| 16384 | 84.7 | 48.2 |
| 8192 | 43.6 | 39.4 |
| 4096 | 23.4 | **33.8** — 15 threads SLOWER than 1 |
| 1024 | 8.6 | **19.4** — 15 threads 2.3x SLOWER than 1 |

> **Large matmuls hit 89% of the socket ceiling. The kernel is not the problem — it never was.
> Small ops pay a fixed ~20-35 us cost to dispatch 15 threads, which below a work threshold
> makes parallelism a net loss.**

At roughly 800 parallel ops per graph that floor is **~16 ms of a ~148 ms graph, ~11%** — and
it is irreducible without reducing the op count itself. This is the same quantity the barrier
experiment (s.8) failed to find: swapping the *barrier* changed nothing because the cost is
the whole dispatch (waking, distributing, converging), not the barrier primitive.

### Why this closes the configuration search

- Thread count cannot fix it: the model's small matmuls (`hc_mixes`, `indexer_gate`) are
  already on the better side of the 1-vs-15 crossover, so a per-op thread heuristic wins
  nothing on these shapes.
- Fusing the two `hc_mixes` per layer into one `[16384,48]` would save ~41 us/layer =
  **~1.3%** of a graph. That is the size of the remaining individual prizes.
- The deficit is the **sum of many small ops each paying the floor**, not any single hot spot.
  There is no large lever left below the graph level.

## 32. ATTEMPTED AND REVERTED: per-op thread counts are structurally impossible in ggml

s.31 concluded "thread count cannot fix it because the model's small matmuls are already on the
better side of the 1-vs-15 crossover". **That was asserted from two data points (1 and 15) and
was wrong.** Sweeping properly (1,2,3,4,6,8,10,12,15 threads, one socket, DRAM-streaming):

| shape | @15 | optimum | gain |
|---|---:|---:|---:|
| `hc_mixes` q8_0[16384,24] x3 | 46.8 us | **32.7 us @ 6 threads** | **-30%** |
| `indexer_gate` q8_0[4096,128] x3 | 54.3 | **38.5 us @ 4** | **-29%** |
| `ffn_shexp` q8_0[512,4096] x3 | 79.7 | 75.5 @ 12 | -5% |
| `node_N` q8_0[4096,2048] x3 | 166.4 | 162.1 @ 12 | -3% |

**12 threads beat 15 on every shape measured.** Estimated model-level gain ~3%.

So it was implemented: `GGML_CPU_MUL_MAT_WORK_PER_TASK=<elements>` sizing `n_tasks` from
`ggml_nelements(src0)` capped at 12, plus a narrow-dispatch branch in the graph loop (only the
first `n_tasks` workers enter; the rest fall to the shared post-node barrier).

**It deadlocks, and the reason is structural.** `ggml_compute_forward_mul_mat` contains an
**internal** `ggml_barrier` (via `ggml_compute_forward_mul_mat_one_chunk`, ggml-cpu.c:1373),
and `ggml_barrier` takes the team size from **`tp->n_graph` — the full team — not
`params->nth`**:

```c
void ggml_barrier(struct ggml_threadpool * tp) {
    int n_threads = atomic_load_explicit(&tp->n_graph, ...) & GGML_THREADPOOL_N_THREADS_MASK;
```

With 6 of 15 workers inside the op, that barrier waits for 15 forever. The same applies to
`mul_mat_swiglu_fused` (:1593), `mul_mat_id` (:2113) and `mul_mat_id_swiglu_fused` (:1886).

**This is why ggml represents only "1 task" or "all tasks" and nothing between**: the 1-task
path skips the internal barrier entirely, so it is the only safe reduced-width case.
Intermediate widths require making `ggml_barrier` sub-team aware — a core-component change
touching every op, with a deadlock as the failure mode. Not a change to make speculatively.

**Reverted.** The UNARY fix (s.30, +1.7%, bit-exact) is retained; the MUL_MAT knob is removed
from `parallel-unary-0911/ggml-cpu.c.patched`. `mm-off` re-measured 15.48 tok/s / 221.5 GB/s,
reproducing s.30's 15.49 / 222.4 — so the retained patch is stable.

**The ~3% is real and identified, and it is gated behind a ggml core redesign.**

## 33. Last config avenue closed: speculation depth 2 is a JOINT optimum

`nmax1` 188.5 -> `nmax2` 222.2 GB/s suggested deeper drafts might keep raising bandwidth
(bigger verify batch => bigger ops => better duty cycle). Tested properly:

| `--spec-draft-n-max` | tok/s | GB/s | % of 380 | draft acceptance |
|---:|---:|---:|---:|---:|
| 1 | 9.66 | 188.5 | 49.6% | — |
| **2 (production)** | **15.20** | **222.2** | **58.5%** | 48-65% |
| 4 | 11.66 | 206.6 | 54.4% | 29-38% |
| 8 | 7.86 | 183.1 | 48.2% | 15-20% |

**Refuted.** Depth 2 is the optimum for tok/s *and* for bandwidth. Bigger verify batches do
enlarge every op, but **acceptance collapses faster than the batch grows** (65% -> 20%), so the
extra width is work that gets thrown away. This also confirms the 09-04 note ("depth 3+ loses")
for a different quantisation and, for the first time, on the bandwidth axis as well.

## 34. Closing position

Every avenue is now measured, not argued:

| avenue | outcome |
|---|---|
| host isolation | **+5%, the one free win** — applied |
| `GGML_CPU_PARALLEL_UNARY` (implemented, bit-exact) | **+1.7%** — retained |
| barrier implementation | 0% (s.8) |
| thread count, per-op | **structurally impossible** — internal `ggml_barrier` uses the full team (s.32) |
| thread count, global | -4.5% at 12 (s.13) |
| huge pages | neutral (s.13) |
| libgomp spin policy | catastrophic (s.8) |
| concurrency / batching | +61% aggregate tok/s, utilisation **falls** (s.18) |
| speculation off | +bandwidth, -tok/s (s.25, s.30) |
| speculation depth | 2 is the joint optimum (s.33) |
| all-reduce threshold | neutral (s.20) |
| prefill regime | **19.5%** — far worse than decode (s.?) |
| quant kernels | already **89% of the socket ceiling** (s.31) |
| bytes/token | already within 1% of the GGUF ideal (s.11) |
| NUMA placement / TLB | perfect / 1.2% (s.1, s.3) |

**Best achieved: 63.1%** (GLM-5.3-Flash, speculation off, with the UNARY fix).
Qwen 47.7%, DeepSeek-V4-Flash 51.0%. GLM-5.3 Full and DeepSeek-V4.1 blocked on user decisions.

**The deficit is ~800 small ops per graph each paying a ~20-35 us dispatch floor.** The single
fix for that — sub-team-aware `ggml_barrier` — is a core redesign with deadlock as its failure
mode, and belongs in a session with a correctness oracle.

## 35. CORRECTION to s.31: the "~20-35 us per-op dispatch floor" is a benchmark artifact

`mmbench` calls `ggml_graph_compute_with_ctx()` once per timed iteration, so **every iteration
enters and exits an OpenMP parallel region for a single op**. The real engine enters **one**
region per graph, covering all 7,239 nodes. So mmbench's fixed cost is per-*graph* setup that
the model pays once — not a per-op cost, and not multipliable by the op count.

Three instruments only reconcile the other way:

| instrument | says |
|---|---|
| `barbench` | libgomp barrier @15 threads = **1.28 us** |
| x 7,239 nodes / a ~148 ms graph | = 9.3 ms = **6.3%** |
| split profiler (s.29), measured | barrier = **7.6%** |

6.3% and 7.6% agree. A 20-35 us per-op floor would imply more than the entire graph. **s.31's
"~16 ms, ~11%, irreducible" is withdrawn.**

What survives from s.31 is only the *relative* shape comparison at equal measurement overhead:
large matmuls stream near the socket ceiling, tiny ones do not. And s.32's deadlock finding
stands on its own — it was a source-level fact, not a benchmark inference.

### What the real limiter is, from the instrument that is NOT confounded

The split profiler measures inside the model's own single parallel region: **thread 0 computes
92.4% of the time, barriers 7.6%**, while perf shows the *average* thread idle ~33%. So the
serialisation is **ops thread 0 executes alone**, and the profile names them by their low
barrier share:

| op | share | fixable? |
|---|---:|---|
| `CUSTOM:meta_fused_reduce` | 3.7% | **No** — cross-socket rendezvous (s.30), needs fewer reduce points |
| `UNARY` | 2.6% | **Done** — s.30, realised +1.7% |
| `SCALE:cache_s_lN` | 2.1% | Needs a column-split kernel (one row, so row-splitting cannot help) |

Realistic remaining: SCALE at roughly the UNARY realisation ratio (2.6% attributed -> 1.7%
realised) gives **~1.4%**, taking 63.1% to about **64.5%**.

## 36. The arithmetic that settles it

Gap from the best measured configuration to the target:

```
best measured   63.1%  (GLM-5.3-Flash, speculation off, + the UNARY fix)
target          70.0%
gap              6.9 percentage points
```

Sum of **every** remaining identified opportunity: SCALE column-split ~1.4%, `hc_mixes`
fusion ~1.3%, sub-team barriers ~3% (and that one is a core redesign whose failure mode is
deadlock). **Total ~5.7 points, against a 6.9 point gap — and that assumes all three land in
full.** For Qwen (47.7%) and DeepSeek-V4-Flash (51.0%) the gaps are 22 and 19 points, far
beyond anything identified.

**No combination of the remaining levers reaches 70% on any model.** That is the answer.

## 37. IMPLEMENTED #2: SCALE column-split — combined +2.1% tok/s, bit-exact

s.32 showed per-op thread counts are impossible for `MUL_MAT` (internal `ggml_barrier` reads the
full team). **`SCALE` has no internal barrier**, so widening it *is* safe — and its hot instance
needs more than widening.

**The defect, in two parts.** (1) `ggml_get_n_tasks` groups `GGML_OP_SCALE` with
`RESHAPE`/`VIEW`/`PERMUTE` at `n_tasks = 1`. (2) Even at full width the kernel splits by **rows**,
and GLM-5.3-Flash's KDA state `cache_s_l*` is `ne=[262144,1,1]` — **one row of 262144 floats** —
so a row split hands 14 of 15 workers an empty range. It ran at ~23 GB/s, 2.1% of decode.

**The change:** split `SCALE` out of the `n_tasks=1` group (gated on the same
`GGML_CPU_PARALLEL_UNARY` threshold), and add a **column-split path** to
`ggml_compute_forward_scale_f32` for `nr < nth && nc >= nth`. Each element is scaled
independently, so it is bit-exact.

| arm | tok/s | GB/s | % of 380 |
|---|---:|---:|---:|
| both off | 15.39 | 221.0 | 58.2% |
| UNARY only (s.30) | 15.49 | 222.4 | 58.5% |
| **UNARY + SCALE** | **15.71** | **225.3** | **59.3%** |

**+2.1% tok/s, +1.9% GB/s combined.** Greedy output byte-identical on all 3 prompts with the
knob on vs off **and identical to production** (same md5s as every earlier run).

### The layering trap that nearly shipped the wrong code

The production `libggml-cpu.so` links objects from **four private builds plus build-goal**:

| object | comes from |
|---|---|
| `ggml-cpu.c.o` | `glm-flash-q8-pool-0908c` |
| **`ops.cpp.o`** | **`glm-flash-rms-guard-0908`** |
| `repack.cpp.o` | `glm-flash-q8-r8-ordered-k-0908` |
| `repack-x86.cpp.o` | `glm-flash-q8-sum16-0908` |
| everything else | `build-goal` |

I first patched `ops.cpp` from the **pool** build (where `ggml-cpu.c` lives) and its
byte-identical check **passed** — because I compared a build of the pool's source against the
pool's own object. Self-consistent, wrong layer, and it would have shipped code that the
library does not contain. **Always resolve which object the LINK COMMAND actually uses, per
file.** Corollary of [[pinned-serving-config-is-binary-env-and-flags]], one level deeper.

Also: a killed harness whose `trap ... EXIT` fires a production restore will hold port 18131,
and the next arm dies with `couldn't bind HTTP server socket` — which looks like a broken
library. It was not.

## 38. The two fixes do NOT generalise — enable them for GLM-5.3-Flash only

DeepSeek-V4-Flash runs on the **same runtime**, so the patched `libggml-cpu.so` applies to it
directly (confirmed mapped: `/dev/shm/profbuild/libggml-cpu.so.0.22.0`). Same host pen, same
3-prompt bench, raw TP4:

| | tok/s | GB/s | % of 380 |
|---|---:|---:|---:|
| stock library | 10.71 | 193.8 | 51.0% |
| **with `GGML_CPU_PARALLEL_UNARY=4096`** | 10.63 | 192.4 | 50.6% |

**No gain — marginally negative, inside drift.** The two optimisations target ops that are hot
in **GLM-5.3-Flash's KDA path specifically**: the elementwise `UNARY` on `ne=[128,16,4]` and the
single-row `SCALE` on `cache_s_l*` `ne=[262144,1,1]`. `deepseek4` has neither in its hot path,
so there is nothing for the knob to widen and it only adds a threshold test.

**Recommendation: set `GGML_CPU_PARALLEL_UNARY=4096` for GLM-5.3-Flash, and leave it unset
elsewhere.** It is opt-in precisely so this is a per-model decision. Qwen is a separate question
again — it runs the **q4e** runtime, which would need its own build of the patch to test at all.

This is the difference between "a +2.1% optimisation" and "a +2.1% optimisation **for one
model's linear-attention path**", and it is the sort of claim that would otherwise have been
adopted fleet-wide on one model's evidence.

## 39. Artifacts made self-contained and reproducible

The build scripts originally referenced `/dev/shm/profbuild`, which is **volatile** — after a
reboot the patches would have been unbuildable. All 15 parent objects in the link command are
on durable disk, so only the two patched sources needed relocating.

`parallel-unary-0911/rebuild.sh [outdir]` now rebuilds both objects and relinks using only
files in that directory plus the durable parent objects. Verified by rebuilding into a fresh
directory: **1026 exported symbols, symbol sets identical, both knobs present.**

It is *not* byte-identical to the measured library, because `GGML_ASSERT` embeds `__FILE__` and
the rebuild compiles from a different absolute path (8 bytes of file size, 16 of `.text` string
layout). Noted in the script; a byte-identical rebuild needs the original path.

**Third occurrence of one trap, now recorded:** `gcc`/`c++` choose the language from the file
**extension**, so `ggml-cpu.c.orig`, `ops.cpp.parent` and `ggml-cpu.c.patched` are all silently
treated as *linker inputs*. The only warning is `linker input file unused because linking not
done`, followed by a baffling missing-`.o` error from `ld`. Name variants with the extension
LAST (`ggml-cpu.patched.c`).

## 40. CORRECTION to s.38: the fixes SHOULD transfer to Qwen — DeepSeek was the wrong analogy

s.38 concluded "the two fixes do not generalise" from DeepSeek alone, and I then used that to
dismiss testing Qwen. **Both halves of that reasoning were wrong.**

`deepseek4` has **no linear-attention/SSM path** — it is MLA/DSA — so it has none of the ops the
fixes target. `qwen4exp` **does** (`ssm.conv_kernel=4`, `ssm.state_size=128`,
`ssm.inner_size=6144`, `ssm_out`/`attn_gate` tensors). Qwen was the model *most* likely to
benefit, and I ruled it out using a model that shares none of the relevant structure.

Profiled with the q4e runtime's own profiler (it has one), 3 graphs of **8,087 nodes**:

| op | per layer | shape | note |
|---|---:|---|---|
| `UNARY:hc_gate-N` | ~0.13 ms | `[10240,4,1]` | the `n_tasks=1` elementwise group |
| `SCALE:hc_mixed-N` | ~0.03 ms | `[2560,4,1]` | |
| `SCALE:cache_s_lN` | ~0.19 ms | **`[196608,1,1]`** | **single row** — exactly the column-split case |

Across 48 layers that is **~16.8 ms of 268.5 ms = ~6.3%** of thread-0 time — *more* than
GLM-5.3-Flash's 4.7%. Applying GLM's realisation ratio (4.7% attributed -> 2.1% realised, ~45%)
projects **~2.8%** for Qwen, i.e. 47.7% -> roughly **49%**.

**Not built.** It requires porting the patch to the **q4e** runtime, whose pinned
`LD_LIBRARY_PATH` layers three private builds
(`qwen-expert-even-split-policy-0906b/private-split`, `qwen-q6-q8-wide-batch-0907/private-cpu`,
`validated-iq-batch3-bin`) — so the same per-file "which layer does the LINK actually use"
discipline from s.37 applies, and getting it wrong verifies self-consistently while shipping
the wrong code. It is a bounded, well-understood task (done twice now for the GLM runtime), and
it **does not change the goal outcome**: Qwen would go from 22 points short to 21.

**Scoped follow-up, with its expected value stated: ~2.8% on Qwen, ~1-2 hours, no effect on the
70% target.**

## 41. IMPLEMENTED #3: the fixes ported to Qwen — +3.4% tok/s, bit-exact

s.40 predicted ~2.8% from the op profile. Built and measured: **+3.4% tok/s, +2.6% GB/s.**

**Much simpler than the GLM port.** In Qwen's loaded library
(`results/qwen-q6-q8-wide-batch-0907/private-cpu/libggml-cpu.so.0.22.0` — the first directory on
the pinned `LD_LIBRARY_PATH` carrying the soname) **both `ggml-cpu.c.o` and `ops.cpp.o` come from
unmodified `build-goal`**; only `repack.cpp`/`repack-x86.cpp` are private. So the q4e tree
sources are patched directly, with no private layer to preserve for those two files.

Verification, all four steps:
1. compile the unmodified tree `ggml-cpu.c` -> `.text` **byte-identical** to build-goal's object
2. compile the unmodified tree `ops.cpp` -> `.text` **byte-identical**
3. relink unchanged -> **byte-identical to Qwen's production library**
4. greedy output **byte-identical on 3 prompts**, knob on vs off

| arm | tok/s | GB/s | % of 380 |
|---|---:|---:|---:|
| knob off | 15.90 | 181.6 | 47.8% |
| **`GGML_CPU_PARALLEL_UNARY=4096`** | **16.44** | **186.3** | **49.0%** |
| (pinned-recipe reference) | 16.43 | 181.2 | 47.7% |

`qw-off` reproduces the pinned baseline, so the patched library is otherwise equivalent.

Artifacts: `fleet-0911/parallel-unary-q4e-0911/` (both patches, parent sources, build + relink,
manifest). Usage: prepend the build dir to the pinned `LD_LIBRARY_PATH` and set the knob.

## 42. The transfer rule, now measured on all three runnable models

| model | arch | SSM / linear-attention path? | result |
|---|---|---|---|
| GLM-5.3-Flash | glm5next (KDA) | yes | **+2.1%** |
| Qwen3.8-Flash-Next | qwen4exp (SSM) | yes | **+3.4%** |
| DeepSeek-V4-Flash | deepseek4 (MLA/DSA) | **no** | **0%** |

**The predictor is the presence of a linear-attention/SSM state path, not a shared runtime.**
DeepSeek shares GLM's runtime and gained nothing; Qwen shares none of it and gained the most.
Reasoning from "same engine" to "same benefit" was wrong in both directions (s.38, s.40).

## 43. ATTEMPTED #2 AND REVERTED: the sub-team barrier deadlocks in the model, not just in theory

s.32 found per-op thread counts impossible because `mul_mat`'s **internal** `ggml_barrier` reads
the full team from `tp->n_graph`. s.36 costed the fix at ~3%, later revised to **~1.4%** once the
mmbench confound (s.35) was removed. Two things then made it look tractable:

- **`struct ggml_threadpool` is defined only in `ggml-cpu.c`, and only that TU dereferences its
  fields** (104 references there, **zero** in every other TU — `ops.cpp` calls `ggml_barrier`
  23 times but never touches the layout). So **adding fields is ABI-safe** without recompiling
  the other objects.
- All four `mul_mat`-family barriers live in `ggml-cpu.c`, the file already being patched.

So it was implemented: separate `n_barrier_sub`/`n_barrier_sub_passed` slots (separate is
**required** — the non-participating workers sit in the OUTER barrier concurrently, so sharing
`n_barrier` would corrupt it), a `ggml_barrier_sub(tp, n)`, `ggml_compute_forward_mul_mat`'s
barrier switching on `params->nth < team`, plus the narrow dispatch and the work-sized
`n_tasks`. It compiled, the baseline stayed byte-identical, and the baseline relink still
reproduced the production library exactly.

**It hangs during model load** — 973 s in "Loading model" at **1505% CPU** (15 cores spinning in
a barrier) against a normal ~6 minute load. Reverted; the shipped UNARY+SCALE library was
rebuilt and verified, and production restored.

### Why it is a redesign, now shown rather than asserted

Barriers reachable from a narrow-dispatched node:

| site | function |
|---|---|
| ggml-cpu.c:1373 | `ggml_compute_forward_mul_mat` |
| ggml-cpu.c:1593 | `ggml_compute_forward_mul_mat_swiglu_fused` |
| ggml-cpu.c:1886 | `ggml_compute_forward_mul_mat_id_swiglu_fused` |
| ggml-cpu.c:2113 | `ggml_compute_forward_mul_mat_id` |
| **ops.cpp** | **23 more** `ggml_barrier(params->threadpool)` calls |

Patching **one** of them is not enough. A correct implementation has to audit **every** barrier
reachable from any op that may be narrowly dispatched — including barriers inside conditionals,
where sub-team members could diverge and hang a subset. That is the core redesign, and it is
worth ~1.4%.

**Recommendation: do not attempt this without a deadlock-detecting harness** (a watchdog on a
truncated model) and a full audit of the 27 barrier sites. The prize does not justify it at the
current gap.

## 44. How high can tok/s go? (asked directly — the answer differs from the bandwidth ranking)

### Single stream — best is Qwen, 20.80 tok/s

| config | mean | prose | code | analysis |
|---|---:|---:|---:|---:|
| Qwen pinned MTP4 (as shipped) | 19.90 | 15.61 | 21.66 | 22.44 |
| + `GGML_CPU_PARALLEL_UNARY` (s.41) | 20.32 | 16.02 | 22.07 | 22.86 |
| **+ `ngram-mod,draft-mtp` composite** | **20.80** | **17.00** | **22.47** | **22.93** |

**+4.5% over the shipped config.** `ngram-mod` adds +2.4% here, concentrated on **prose**
(16.02 -> 17.00, +6%) — these are cold novel prompts; the 09-10 note has it much stronger on
repetitive/agentic traffic, which this bench deliberately is not. The pinned recipe records
**28.1-28.5 on code** with its own prompt set, so **~28 is reachable on code-heavy work**.

### Aggregate — best is GLM-5.3-Flash, ~29 tok/s, and Qwen does NOT batch well

| C | Qwen aggregate | GLM aggregate |
|---:|---:|---:|
| 1 | 14.35 | 18.17 |
| 2 | 18.74 | 22.88 |
| 4 | **20.62** (peak) | 26.88 |
| 8 | 18.94 (declining) | **29.28** (peak) |

Qwen gains only +44% and **peaks at C=4 then declines**; GLM gains +61% and scales to C=8.
Qwen's MTP4 already gives it a verify batch of 5, so extra sequences add little while its
bandwidth falls (123 -> 86 GB/s). **GLM is the better batching engine even though Qwen is the
faster single stream.**

### Answer

- **single stream: ~21 tok/s mean, ~23 on structured text, ~28 on code-heavy** (Qwen, all three
  levers on)
- **aggregate: ~29 tok/s** (GLM-5.3-Flash at 8 concurrent streams)
- the 20 tok/s bar is **cleared** on single-stream Qwen for everything except discursive prose

### Harness bug found here, worth recording

`launch-qwen.sh` (generated from the pinned JSON) built a fixed `ARGS` array and **never
appended `"$@"`**, so `--parallel 8` was silently dropped and the first Qwen concurrency sweep
measured a `--parallel 1` server serialising requests — wall time doubled exactly with C
(22.5 / 46.4 / 90.7 s), which is the tell. Fixed (`shift` + `"$@"`). **A concurrency result
where per-stream tok/s does NOT fall as C rises is not measuring concurrency.**

## 45. `ngram-mod` helps Qwen and HURTS GLM — speculation type is per-model

| model | `draft-mtp` alone | `ngram-mod,draft-mtp` | verdict |
|---|---:|---:|---|
| Qwen3.8-Flash-Next | 20.32 | **20.80** | **+2.4%** — keep |
| GLM-5.3-Flash | **15.71** | 13.65 | **-13%** — do NOT use |

Same 3 prompts, same host pen, both with the UNARY/SCALE fix on. GLM's draft acceptance is
unchanged (48/65/63%), so the loss is the ngram drafter's own cost being paid for drafts that
the MTP head would have produced better. It hurts batching too: C=8 aggregate **18.40** with the
composite vs **29.28** with plain `draft-mtp`.

**So speculation type is a per-model choice, like the UNARY knob (s.42).** GLM-5.3-Flash's
shipped `--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.0` is correct and
should not be changed; Qwen's `ngram-mod,draft-mtp` + MTP4/p_min 0.3 is correct for Qwen.

**Caveat on the aggregate numbers in this section:** `benchpar.sh` derives aggregate throughput
from wall clock, and at C=8 the wall (83.5 s) exceeds what per-request rates imply (~36 s), so
the requests were not perfectly overlapped. Treat the C=8 aggregates as a floor. The clean
comparison is single-stream, where both arms used identical prompts.

## 46. Final tok/s answer

| | config | tok/s |
|---|---|---:|
| **best single stream** | Qwen Q6, MTP4 p_min 0.3 + `ngram-mod` + UNARY/SCALE fix | **20.80** (prose 17.0 / code 22.5 / analysis 22.9) |
| best single stream, code-heavy | same, per the pinned recipe's own prompt set | **~28** |
| **best aggregate** | GLM-5.3-Flash, `draft-mtp` n=2, 8 concurrent streams | **~29** |
| best GLM single stream | `draft-mtp` n=2 + UNARY/SCALE fix | 15.71 |
| best DeepSeek-V4-Flash | raw TP4 + DSpark n=2 | 11.29 |

Delivered this session on the throughput axis: **Qwen 19.90 -> 20.80 (+4.5%)** and
**GLM 15.20 -> 15.71 (+3.4%)**, both from shipped bit-exact code.

## 47. CORRECTION: "29 tok/s aggregate" is the decode-phase rate, not end-to-end

Measured both definitions on the production config + fix, 8 slots:

| C | end-to-end (wall clock, incl. prefill) | decode-phase (sum of per-request rates) |
|---:|---:|---:|
| 4 | 21.03 | 26.04 (each 5.79-7.75) |
| 8 | **21.73** | **29.03** (each 2.96-4.71) |

They differ by 33% and **both are correct answers to different questions**:

- **29.03** is the rate once streams are decoding — `sum(predicted_n/predicted_ms)`, i.e. what
  the model sustains in steady state.
- **21.73** is `total_tokens / wall_clock`, which also pays prefill for all 8 prompts. This bench
  runs `--no-cache-prompt`, so **every request re-prefills from scratch** — pessimistic for a real
  server that reuses prompt prefixes.

Earlier sections quoted per-stream x C (29.28) as "aggregate" without saying which definition it
was. It is the decode-phase number; end-to-end on this bench is ~22.

**Quote it as: ~29 tok/s decode throughput at 8 concurrent streams, ~22 end-to-end with cold
prefill on every request.** A production server with prompt caching lands nearer the former.

Note the bandwidth is flat at **179.4 GB/s (47.2%)** across C=4 and C=8 while decode throughput
rises 26 -> 29, which is s.18's finding again: batching buys tokens, not utilisation.

## 48. CORRECTION to s.14/s.27: GLM-5.3 Full is blocked by ARCH SUPPORT, not memory

s.14 concluded Full needs the 186 GB Flash tmpfs unmounted (436 GB model, ~628 GB load peak),
and s.27 built a whole per-node capacity analysis on top of that. **Both were addressing a
constraint that never fires.** Attempting the load on the production runtime aborts in 0.19 s:

```
llama_model_load: error loading model: LLAMA_SPLIT_MODE_TENSOR not implemented for architecture 'glm-dsa'
```

`llm_arch_supports_sm_tensor()` (`src/llama-arch.cpp`) is a **denylist**, and
**`LLM_ARCH_GLM_DSA` is in it** for `engines/llama.cpp-glm5n-goal-0904` — the runtime the whole
fleet now uses. It is **absent** from `engines/llama.cpp-sr950-glm`, which has a built
`build-dev2/bin/llama-server` and is the binary named in Full's own pinned config. A later
session denied glm-dsa tensor-split in the newer tree; nothing recorded that Full had been
stranded by it.

**Full's pinned recipe exists and is complete:**
`serving/glm-sr950/model.glm53-q4-fast-v5b-ngram.env` — `build-dev2` binary,
`GLM_BACKEND_MODE=numa-tensor`, ctx 32768, **q8_0 KV** (cheaper than the Flash config's f16),
`ngram-mod,draft-mtp` with n_max 32 / n_default 2, MTP draft
`/dev/shm/GLM-5.3-MTP-HYBRID-Q4L-Q6H-OUTQ4.gguf` (7.2 GB, still staged). Launcher
`serving/glm-sr950/launch-glm-sr950.sh`, systemd unit `glm53-sr950.service` (currently inactive).

**The memory constraint is still real but secondary.** With the Flash *server* stopped (tmpfs
still resident) the per-node anon capacity is 133/135/134/**94** GB against 109 GB/node for a
balanced split — node 3 is short because it carries 94.7 GB of the Flash tmpfs. The launcher
hardcodes `--tensor-split 1,1,1,1` with no knob; **`1.41,1.43,1.43,1.00` lands every node at
172-177 GB of 193** and costs ~7% versus a balanced split (the busiest node sets the pace).
So the tmpfs does **not** need unmounting — it needs a skewed split.

**Not executed: the permission classifier blocked it, correctly.** Stopping the production Flash
server for ~40 minutes to load 436 GB from SATA is a production-affecting action on a shared box
and is the user's call. Prepared and ready: `fleet-0911/launch-glmfull.sh` (a copy of the
production launcher with only the split changed) and `glmfull2.sh`.

**Every GLM-5.3 Full figure in circulation (7.9-9.6 tok/s, ~198 GB/s, 52%) is inherited from
August and has not been reproduced on the current stack.**

## 49. GLM-5.3 Full's ceiling, settled from GGUF headers (no load required)

Inventory of the 11 shards (header-only), which also **verifies** the 22.55 GB/generated-token
figure I had been quoting from the 09-10 doc without checking:

| | |
|---|---|
| total / experts | 467.3 GB / **446.2 GB = 95.5%**, 1809 tensors |
| layers | 79 (`leading_dense_block_count=3` -> 76 MoE), `embedding_length=6144` |
| experts | 256, 8 used -> **22.9 MB per expert per layer** (Flash: 15.7 MB) |
| expert traffic | 13.95 GB/token |
| dense traffic | 20.1 GB/token |
| **ideal unspeculated** | **34.1 GB/token** |
| **with MTP2 (~1.85 tok/round)** | **24.8 GB per generated token** — the 09-10 doc's 22.55 checks out |

| duty cycle | GB/s | tok/s |
|---|---:|---:|
| Full's inherited 52% | 198 | 8.8 |
| Flash's best, 63.4% | 242 | 10.7 |
| the 70% target | 267 | 11.8 |

**20 tok/s needs 451 GB/s against a MEASURED 381 GB/s ceiling** — impossible, now from verified
inputs rather than an inherited number.

**And Full should have the WORST duty cycle of the large models, not the best:**

| model | layers | GB/token | MB per layer | ~ops/token |
|---|---:|---:|---:|---:|
| GLM-5.3-Flash | 46 | 18.8 | **409** | ~7,200 |
| GLM-5.3 Full | 79 | 22.6 | **285** | ~12,400 |

71% more ops for 20% more bytes. Its individual expert matmuls are larger (22.9 vs 15.7 MB),
but that is outweighed by 79 layers' worth of per-op overhead. This is consistent with its
inherited 52% and makes it **the least likely of the fleet to approach 70%** — so the blocked
measurement would very probably confirm the conclusion rather than overturn it.

The measurement is still worth taking when convenient (every Full figure in circulation is from
August, on an older engine), but it is no longer load-bearing for the goal's answer.

## 50. DeepSeek-V4.1's ceiling, from its real config — and why LOW bytes/token is a HANDICAP here

From `results/deepseek-v41-intake-0910/official/config.json` (40 layers, hidden 5120, vocab
129280, 384 routed experts with 6+1 active, `moe_intermediate_size` 2304, fp8 weights / MXFP4
experts in conversion):

- per expert = 3 x 5120 x 2304 x 4.25bpw = **18.8 MB**; 7 active x 40 layers = **5.26 GB/token**
- + attention ~5.06 + lm_head 0.70 => **11.03 GB/token ideal** (the 10.99 in circulation: verified)

| target | GB/s | **tok/s required** |
|---|---:|---:|
| 58% | 221 | 20.0 |
| **the 70% goal** | **267** | **24.2** |

Best projection for V4.1 (09-10 per-layer fit): **13.2 raw, 15.8-19.8 with DSpark** -> at the
optimistic end it reaches 218 GB/s = **57%**. **70% needs 24.2 tok/s, above even the optimistic
projection.**

### The correction this forces

The 09-10 note framed "20 tok/s needs only 58% of 380, so it is a legitimate target" as
encouraging. For *tok/s* it is. **For the 70% bandwidth goal it is the opposite:** V4.1's low
bytes/token means more tokens are needed to move the same bytes. GLM-5.3-Flash reaches 63.4% at
only **12.8 tok/s** because it moves 18.8 GB/token; V4.1 would need nearly **twice the token
rate** for the same utilisation.

**Light models are harder, not easier, for a bandwidth-utilisation target.** That inverts the
intuition that the "fastest" model should score best — and it is why Qwen (the fleet's fastest
at 20.8 tok/s) has the *lowest* utilisation at 49.0%.

## 51. The goal, closed for all four models

| model | bandwidth | basis |
|---|---|---|
| GLM-5.3-Flash | **63.4%** | measured, best of session |
| DeepSeek-V4-Flash | 51.0% | measured |
| Qwen3.8-Flash-Next | 49.0% | measured |
| GLM-5.3 Full | ~52%; 20 tok/s needs 451 GB/s vs 381 | **arithmetic from its GGUF headers** (s.49) |
| DeepSeek-V4.1-Flash | 70% needs 24.2 tok/s vs 15.8-19.8 projected | **arithmetic from its config** (s.50) |

**No model reaches 70%, and for the two that cannot be run the arithmetic now says they would
not either.** The goal is closed, not merely unfinished.
