# GLM-5.3-Flash decode as an agent sees it: 12.0 to 20.0 tok/s, September 18-20

**Status (2026-09-20, revision e in production):** a Codex agent inside Paseo gets **20.0 tok/s** on the fixed-output fixture
(19.96-20.08 over four warm runs) and **19.0 tok/s on requests exactly as Codex sends them** (twelve runs, 17.7-20.1, ten of them at
or above 18.0). On 09-19 the same service gave 15.5-16.2, on 09-18 12.0. Output has been byte-identical since revision b.

Machine: Lenovo SR950, 4x Xeon Gold 6242 (16 cores each, AVX-512 VNNI, no AMX), 24 DIMMs DDR4-2400, 755 GiB, no GPU.
Measured aggregate read ceiling 381.6 GB/s ([tools/membw.c](../tools/membw.c)). Model: GLM-5.3-Flash UD-Q4_K_XL (186 GiB), Q8_0 MTP
sidecar, draft depth 2, four CPU-NUMA devices in tensor-parallel, 15 workers per socket, 1,048,576-token context, two unified slots.

## How the number is measured

The rate that matters is the one an agent gets, so the probe is an actual Codex turn through the installed Paseo wrapper
([tools/paseo_codex_bench.py](../tools/paseo_codex_bench.py)): app-server, `thread/start` and `turn/start` with the lowercase model
name Paseo sends, one fixed no-tools prompt, about 4,000 input tokens of instructions and tool schemas, 250-310 generated tokens.
Decode tok/s is `tokens_predicted / tokens_predicted_seconds` from the server's `/metrics` delta, so prefill and client time are excluded.

Three things make comparisons valid:

- **Same process.** Separate loads of the same configuration differ by 2-3%, and a production load reads about 2% below the same
  stack in a test window. Every gain below compares modes inside ONE loaded server, switched through memory-mapped control words
  while both slots are idle, in balanced order.
- **Same tokens.** A loopback adapter sets `temperature=0, seed=42, cache_prompt=true`, and each pair is checked for identical
  generated text, draft, accepted-draft and verification counts. From revision b on, every mode of every window and every production
  revision produced ONE text (hash `424f201528`, 175 of 214 drafts accepted).
- **And the real thing.** Codex sends **no sampling fields**, so real turns use the server defaults (the GGUF carries temp 1.0,
  top_p 0.95; top_k 40 and min_p 0.05 come from the server). Sampled runs differ in text and in draft acceptance from run to run
  (68-81%), so they are reported as means over 8-12 runs and never used for a small A/B.

## What moved it

| Change | Same-process result (tok/s) | Output | In production |
| --- | --- | --- | --- |
| Codex model catalog had no lowercase alias, so Paseo's requests fell back to 20,751 characters of default instructions | input 7,739 -> 3,998 tokens; 9.6 -> 12.0 (cold, sampled, single observations) | n/a | 09-19 |
| Fused pooled-indexer compressor + flat recurrent-state copy | native greedy 12.65/14.62/14.17 -> 14.10/16.21/15.71 for the copy alone | byte-identical | 09-19 |
| Eight-channel pool inner loop | 14.60 -> 15.15 (+3.8%); at 29,930 tokens 9.05 -> 10.26 (+13.4%) | byte-identical | 09-19 |
| KV rollback scans the used prefix, not 1,048,576 cells | 15.24 -> 15.61 (+2.4%) | identical | 09-19 |
| MTP catch-up with zero outputs stops after the KV write | 15.43 -> 15.82 (+2.5%) | identical | 09-19 |
| Cell-split MQA attention in F32 + selection top-k | 16.24 -> 17.23 (+6.1%) | **changes**, toward float64 | rev b |
| Content-verified pooled-result cache | 17.23 -> 17.61 (+2.3%) | identical | rev b |
| MTP draft loop: catch-up merged into the first draft pass, direct pick, constant-shape (padded) draft batches | merge 18.01 -> 18.10 (w2); then padding + pick 17.76 -> 18.15 (w3) and 17.97 -> 18.40 (w4) | identical | rev b |
| OpenMP idle spin 300,000 -> 20,000 (`GOMP_SPINCOUNT`) | 18.2-18.4 (three loads) -> 18.81 (separate loads) | identical | rev c |
| Sampler: top-k candidates by threshold scan, candidate-free clone | 18.81 -> 19.11 (+1.6%) | identical | rev c |
| Coupled draft/verifier sampling (Gumbel-max, shared counter-based noise) | sampled requests 17.89 -> 18.24 (+2.0%), acceptance 72.4% -> 75.0%; greedy untouched | exact sampler, same distribution | rev c |
| One worker team per device for both contexts | 19.11 with or without once the spin is short | identical | rev d |
| Gated delta net split by state row + MTP query side only for the predicting row | 19.11 -> 19.55 (+2.3%); at 13,231 tokens 14.75 -> 15.41 (+4.5%) | bit-identical | rev d |
| Tensor-parallel backend: graph-input uploads without a thread per device, dispatch waits that block | 19.30 -> 20.18 (+4.6%); sampled 17.98 -> 18.79; at 13,231 tokens 16.15 -> 16.98 | identical | rev e |

Production after each revision, same fixture through Paseo: revision b 18.09-18.40 greedy, 17.14 sampled (n=4); revision c
18.46-18.76, 17.94 sampled (n=10); revision d 19.17-19.32, 18.18 sampled (n=12); revision e 19.96-20.08, 19.00 sampled (n=12).
Windows: w1 (kernels), w2-w4 (draft loop, batch-invariant attention), w5 (spin, sampler, coupling), w6 (revision d), w7 (sampled
traffic with both sides of the coupled sampler logged), w8 (revision e), w9 (draft depth 3 against 2).
Patches and their evidence: [patches/README-20260920-glm5next.md](../patches/README-20260920-glm5next.md).
Records, controllers and logs: [engineering/2026-09-20](../engineering/2026-09-20/README.md).

## Four findings that are not kernels

**Two worker teams per core.** Every `llama_context` creates a backend per CPU-NUMA device, and every such backend owned a
dispatcher thread that is the master of its own OpenMP team. Trunk plus MTP draft is 8 teams on 60 cores, two pinned workers per
core, and the teams alternate. A team that has just finished keeps spinning at libgomp's dock: the default 300,000 iterations are
documented as "3 ms", but `pause` takes ~140 cycles on Skylake/Cascade Lake, so it is ~15 ms, on exactly the CPUs the other team
needs next. It surfaced when a change that saved 1.2 ms of host time per cycle made decode 2.7% *slower*: the first draft graph now
started 1 ms earlier, deeper into the trunk team's spin (8.3 -> 10.3 ms for the same graph). Earlier trials here had bracketed the
useful range without entering it (1000: -14%, the waits inside a graph go to sleep; PASSIVE: -10.6%; 0: no help; ACTIVE or 1e8:
~250x slower). On an 8-layer proxy the trunk graph went 26.5 -> 23.1 ms and the first draft graph 8.3 -> 5.9 ms for any value from
8,000 to 150,000. [Patch and both fixes](../patches/cpu-numa-shared-team.patch).

**A thread per device for every graph input.** perf is not available on this host (`perf_event_paranoid=4`, no root), so the
host side had never been looked at. What an unprivileged user does have was enough: per-thread `/proc/PID/task/*/stat` deltas over
one decode ([tools/thread_profile.py](../tools/thread_profile.py)) and `strace -f --seccomp-bpf` on a proxy started as strace's
own child. strace showed 5,322 `madvise(8 MB, MADV_DONTNEED)` calls in a 57-cycle decode: thread stacks being recycled, ~93 thread
creations per cycle. The tensor-parallel backend's `set_tensor` started one `std::thread` per NUMA device for every call on a
CPU-NUMA buffer group: right for weights, and also the path of every token, position, mask and index upload, including uploads
inside graph compute. The transient threads landed on worker CPUs (the pinned workers were preempted 32 times per cycle, 6 after
the fix) and every exit sent TLB shootdowns into 60 computing cores. Uploads below 1 MiB now run in the caller: the verify graph
went from 119.7 to 115.5 ms and the draft graphs from 10.8 to 9.2 ms per cycle, the largest single step of the day after the
attention kernel. The 8-layer proxy showed it as +13%. [Patch](../patches/meta-backend-small-uploads-blocking-dispatch.patch).

**Batch-composition invariance.** The first version of the attention kernel let each worker take a slice of the cells that *any*
query of the batch could see. That made one query's summation order depend on its batch companions, i.e. a verified token's numbers
depended on the draft tokens beside it, and with padded draft batches greedy text alternated between two outputs from run to run.
Revision 2 deals cells to workers by index, takes each row's maximum before the softmax pass and keeps the weight sum as a running
double; a query alone and inside any batch is now bit-identical ([check](../tools/fa_mqa_invariance_check.cpp)). Any kernel that sees
a speculative verify batch needs this test.

**A drafter cannot be allowed to matter, and then it can be helped.** With sampled requests the verifier draws a token and a
greedy draft is accepted only if the draw happens to be its argmax. Replacing the final `dist` step by
`argmax(logit + Gumbel(salt, position, token))` is an exact sampler with a random stream the drafter can share; the drafter adds
the same noise to its own top-10. Same seed now gives the same text with or without a drafter. Acceptance rose 72.4% -> 75.0%, far
less than a synthetic drafter suggests (0.39 -> 0.88 of drafts at logit noise 0.3): the MTP head is not close to the trunk where
the trunk is uncertain. [Patch, exactness test](../patches/coupled-sampling-fast-sampler.patch).

## Where a cycle goes

One speculative cycle verifies three tokens and yields 2.63 on the greedy fixture (about 2.5 sampled).

| Part | 09-19 production, ms | revision d, ms | revision e, ms | Note |
| --- | ---: | ---: | ---: | --- |
| target verify graph | 137 | 119.5 | 115.5 | 7,239 nodes |
| MTP draft | 18.2 (catch-up + two passes, 3.3 of it graph rebuild) | 10.7 (two passes, both reused) | 9.2 | |
| outside graphs (sampling, rollback, server) | 6.0 | 2.5 | 1.9 | |
| **cycle** | **161** | **133** | **127** at ~200 tokens of context, **131** at the 4K fixture | |

Worker-0 attribution inside the verify graph, deployed kernels (106 ms of ops + 11 ms of waits): routed experts 42.2, dense matmul
36.6, flash attention 4.5, cross-socket reduces 3.5, gated delta net 2.6 (+2.2 waiting, before the row split), top-k 0.6,
about 13 in small ops.

**The experts are finished, and so is the dense part.** 24 (token, expert) pairs x 42 layers x 3.8 MB per socket in 42 ms is ~96 GB/s
per socket, the machine's measured ceiling, with every expert already split four ways. The large dense Q8_0 projections measure
84-92 GB/s per socket (`[4096,2048]` x102: 88; the LM head: 92). Together that is 79 of the 106 ms. What can still move is
everything a GPU-hybrid engine would hand to the GPU: attention, the indexer, the hyper-connection small ops, the draft loop,
host work and scheduling. All gains in the table came from there.

The marginal verified row costs 24 ms on revision e (window w9: verify graph 110 ms with three rows, 133.5 ms with four; 14 ms of
it is expert traffic that cannot shrink, and part of the rest is that the expert kernels are specialised for batches of three).
That is why depth 3 still loses: 3.12 tokens per cycle instead of 2.63, but 151 ms instead of 121 (greedy 19.56 -> 18.73,
sampled 19.05 -> 18.29, one process). The verify batch must also keep ONE shape: the backend holds a single compiled graph per
context, so a change of shape costs 43 ms of graph build plus ~60 ms of re-splitting.

## What did not work

Kept because each one looked right first ([09-19 log](../engineering/2026-09-20/archive/sr950-strategy/GLM53-FLASH-CODEX-20260919.md),
[09-18 windows](../engineering/2026-09-20/archive/serving/fleet-0912-ctx/RESULTS-LOG.md)):

- MTP depth 4: -22% (09-18); depth 3 on revision e: -4% greedy and sampled (w9), see above.
- Variable-length verification, simulated offline on 2,605 logged sampled cycles with the measured costs: dropping the second
  draft when the first one's confidence is low would gain at most +3%, and only if a shape change were free, which it is not.
- MoE/FFN fusion switches: output changes on all three prompts, no speed signal.
- Porting a wider expert kernel: cancelled by arithmetic before any code; the experts already run at bandwidth.
- Q4 token batching, Q4/Q5 clamp fusion, a batched Q8 kernel (13-34% faster alone, +0.8% in a four-socket synthetic cycle): no model gain.
- 16 workers per socket, zero-spin dispatch, pinning helper threads to the spare cores (-9%).
- A retained 128-thread OpenMP team made an optional barrier look 1.4x faster; without the artificial team it is 0.97x.
- Per-request `speculative.n_max` is silently ignored by this build; depth changes need a restart.
- Selection top-k loses to `std::partial_sort` above ~25,000 entries, so the new top-k is gated to short rows.
- A multi-slot graph cache in `llama-context.cpp` (keep both draft graph shapes alive): the Meta backend keeps a two-entry
  ping-pong of per-device tensors per buffer, so two live graphs over the same weights destroy each other. Constant-shape
  batches get the same saving without it.
- A faster sampler on the default OpenMP spin: -2.7% (see above). It is +1.6% once the teams stop colliding.
- A Q4-extracted MTP sidecar (5.5 GiB instead of 9.3): 0.15 ms per draft pass. The draft pass is 46% LM head but it is latency,
  not bytes, that is left in it.
- One shared worker team is speed-neutral on the full model once the spin is short; it is deployed for robustness
  (one worker per core, 260 -> 200 threads), not for speed.
- The row-split gated delta net removes a two-heads-on-one-worker straggler in all 34 KDA layers and still gains under 1 ms:
  the op is dominated by copying a 64 KB state per head per token for speculative rollback.
- Tuning the coupled drafter. Both sides share their noise, so a log of the verifier's candidates and the drafter's top-10 lets
  any drafter setting be replayed offline against the draws that really happened ([tool](../tools/couple_fit.py),
  [patch](../patches/coupled-sampling-offline-fit.patch)). On 4,752 positions of sampled traffic the deployed setting gives 2.425
  tokens per cycle against 2.352 for a greedy drafter (+3.1%, a cleaner number than the +2.0% of window w5), and no drafter
  temperature (0.5-3.0), truncation or top-1 bias adds more than 0.35%. The verifier's token is inside the drafter's top-10 98%
  of the time, and in 59-64% of positions the verifier's chain leaves a single candidate. What is left is the MTP head.

## Open

- Production loads read about 2% below test-window loads of the same stack (revision d: 19.2 against 19.55 greedy); the cause is not
  found (same launcher, environment, limits and cgroup settings; no memory or CPU pressure).
- The main thread still burns ~7 ms of user CPU per cycle although the wall time outside graphs is 1.9 ms; what it does during
  graph compute is unknown (no profiler). The first request after a restart also takes ~146 MB of page faults on the main thread
  (prompt-cache state), which is time to first token, not decode.
- The 90 `hc_mixes` projections per graph (Q8_0, 16,384 -> 24) take 29 us each for ~5 us of arithmetic; with the other tiny
  matmuls that is 2-3 ms per cycle of pure launch cost. `nextn.eh_proj` is mirrored on all sockets (35.7 MB read per socket per
  draft pass); a split rule would save ~0.6 ms per cycle.
- 13,231 tokens of context measure 15.4 tok/s. Nothing longer was re-measured on 09-20; production measured 10.26 tok/s at
  29,930 tokens on 09-19.
- The cached-versus-fresh output discrepancy recorded on 09-19 is unresolved.
