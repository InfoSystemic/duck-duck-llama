# GLM-5.3-Flash

**Status (2026-09-21):** production revision h delivers 19.8–20.6 tok/s to a Codex agent inside Paseo on the ~4K-token greedy fixture, and 18.7–19.9 tok/s on requests as Codex sends them. Since 2026-09-21 the service has been stopped to make room for [MiMo-V2.6-Pro](mimo-v26-pro.md); the two cannot share the machine's RAM.

Earlier status points, for comparison:

| Date | Speed |
| --- | --- |
| 09-19 | 15.5–16.2 tok/s |
| 09-18 | 12.0 tok/s |

Speed with depth in one append-only session:

| Context | Speed |
| --- | --- |
| 25K | 19.6 tok/s |
| 67K | 15.5 tok/s |
| 128K | 12.6 tok/s |

At 4K the cycle is at a hardware limit for this quantisation (see below). The remaining engineering value is at depth.

## Decode through Paseo, September 18–20

The measurement is an actual Codex turn through the Paseo wrapper, modes switched inside one loaded server ([report](../../benchmarks/glm53-flash-paseo-decode-20260920.md), [patches](../../patches/README-20260920-glm5next.md), [records](../../engineering/2026-09-20/README.md)). Everything below is in production.

- **09-19, all output-identical:** fused pooled-indexer compressor and a flat recurrent-state copy (the state copy had been running on one worker: 20.4 → 1.2 ms per cycle), an eight-channel pool loop (+3.8%, +13.4% at 30K), a KV rollback that scans the used prefix instead of 1,048,576 cells (+2.4%), and an MTP catch-up graph that stops after the KV write when no output is requested (+2.5%).
- **09-20, kernels:** a cell-split F32 attention kernel for the MQA cache with a selection top-k (16.24 → 17.23) and a content-verified pooled-result cache (→ 17.61). The attention kernel changes generated text because the stock kernel sums V in FP16 and is about 1% off a float64 reference on real tensors ([report](../../benchmarks/cpu-flash-attn-f16-accumulation.md)); its second revision is also bit-identical for a query alone and inside any batch, which the first was not. A gated-delta-net kernel split by state row and an MTP graph that builds queries only for the predicting row are bit-exact (+2.3% together, +4.5% at 13K).
- **09-20, draft loop and sampling:** catch-up merged into the first draft pass, constant-shape draft batches so both draft graphs are reused (+2.1%), a direct draft pick, top-k candidates taken straight from the logits (+1.6%), and coupled draft/verifier sampling: the verifier's pick becomes Gumbel-max with noise the drafter shares, an exact sampler that raised sampled-request acceptance from 72.4% to 75.0%.
- **09-20, scheduling:** trunk and draft each owned an OpenMP team pinned to the same 60 cores, and libgomp's default idle spin (~15 ms on this CPU, not the documented 3) made every hand-off a collision. `GOMP_SPINCOUNT=20000` was worth more than any single kernel that day (18.2–18.4 → 18.8); one shared team per device makes the collision impossible.
- **09-20, the host side:** the tensor-parallel backend started one thread per NUMA device for every graph-input upload, about 93 thread creations per decode cycle, found with `/proc` and strace because perf is not available. Uploads below 1 MiB now run in the caller and the dispatch waits block instead of spinning: 19.30 → 20.18, output identical.
- **09-20, one expert matrix was not at the wall:** the Q5_K down projection streamed at ~77 GB/s per socket against ~97 for the Q4_K gate and up, because its kernel reads a block group in a strided order the hardware prefetcher does not follow. A software prefetch of the next group: +3.1%, bit-identical.
- **An integration fix worth as much as a kernel:** the Codex model catalog lacked the lowercase alias Paseo sends, so requests carried 20,751 characters of default instructions. Adding it cut the prompt from 7,739 to 3,998 tokens; decode is context-sensitive, so this alone moved 9.6 → 12.0 tok/s.
- **Codex sends no sampling fields.** Real turns run at the server defaults (the GGUF carries temp 1.0) and accept fewer drafts, which is why two numbers are reported. Greedy numbers are for comparisons, not a promise.

Where the time is: a cycle is now ~127–131 ms (161 on 09-19). The routed experts take 42 ms and the large dense projections 37, and both already stream at the machine's measured DRAM ceiling with every matrix split across the four sockets, so those kernels are finished. What moved was attention, the indexer, the draft loop, host work and scheduling. Draft depth above 2 loses (−4% at depth 3 on the current stack, −22% at depth 4 on 09-18; a verified row costs 24 ms); fusion switches, Q4/Q8 token batching, 16 workers per socket, helper pinning, a multi-slot graph cache and a Q4 draft sidecar gave nothing.

Open: single sampled turns still fall below 18 (three of twelve); ~90 tiny hyper-connection projections per graph cost 2–3 ms of launch overhead; a cached-versus-fresh output discrepancy recorded on 09-19 is unresolved.

## Revisions g–l, September 20 evening: what 24+ tok/s would take

[Records](../../engineering/2026-09-23/README.md#glm-53-flash-september-20-evening) · [depth plan and progress log](../../engineering/2026-09-23/archive/serving/fleet-0920-flash18/DEPTH-CAMPAIGN-PLAN-20260920.md)

Every revision was measured on the same 4K Codex fixture (greedy, seed 42). Revision f read 20.15 tok/s there, with acceptance 0.775 and text hash `ae3c80b549`.

| Revision | Change | Outcome |
| --- | --- | --- |
| g | Pooled-indexer fusion made safe against an output placed over its inputs ([patch](../../patches/glm5next-pool-fusion-overlap-proof.patch)) | **Kept.** 19.98 / 20.26 / 20.33 tok/s with the same text hash. Across the 8K→128K curve it is unchanged against f (202 vs 198 ms per cycle at 128K), because a fresh session never trips the rejection. This is a robustness fix for sessions restored from the prompt cache. |
| h | Lightning-indexer score kernel blocked 16 rows per query pass (2.3× per core); radix top-k above 16,384 entries; both bit-identical ([patch](../../patches/glm5next-indexer-score-blocked.patch)) | **Production.** 20.54 / 20.62 / 20.49 tok/s with the same text hash. The expected gain is at depth (about 19 ms per cycle at 128K); it has not been measured. |
| i | Router weights stored as F16 | **Rejected.** 19.69–19.83 tok/s. Acceptance fell 0.775 → 0.763 and the text changed: on the full model, an F16 router flips near-tie expert choices, though a 4-layer check had called it bit-identical. |
| j | Dense Q8_0 weights requantised to Q6_K at load | **Rejected.** 19.69–19.82 tok/s. **Fewer bytes were slower:** these matmuls already run at 71–94 GB/s per socket against a ~95 GB/s wall, and Q6_K costs more to unpack than it saves in bandwidth. The perplexity gate was never needed. |
| k | Composite drafter: n-gram matches first, MTP otherwise (`n_match` 8) | **Rejected.** 16.1–18.5 tok/s on unseen prompts, with acceptance 0.51–0.66. An n-gram match *replaces* a 77.5%-accurate MTP draft. A repeated prompt reads 25.3–25.5 at acceptance 1.000, but that is the n-gram cache replaying its stored answer. Never quote a warm n-gram number. |
| l | Same composite at `n_match` 24 | **Rejected.** 17.2–18.9 tok/s, acceptance 0.60–0.69. Rolled back to h. |

The k and l numbers come from unseen prompts, while h's come from the fixture. No h run on those prompts exists, so the comparison is indicative.

A post-rollback check of h read 19.77–19.84 tok/s with acceptance 0.776, but with text hash `b7c6851941` instead of `ae3c80b549`. The recorded request shape was identical. No cause was found; a date-dependent field in the Codex request is one unverified possibility.

A column-split activation quantiser was built and never staged. It rested on a per-socket bandwidth figure that turned out to be an arithmetic error: the traced shapes were already per-socket.

**Where the ceiling is at 4K.** Every matrix already streams at 71–94 GB/s per socket against a measured ~95 wall, so bytes cannot be traded for time in either direction. The exact levers left total about 6%:

- a draft output head with fewer bytes: about 2%. This is output-exact, because [draft quality is free](../case-studies/block-drafter.md#draft-quality-is-free-target-quality-is-not);
- the 90 tiny hyper-connection matmuls: about 2%;
- small-operation fusion: about 2%.

That is about 21.8 tok/s. Past that, the verify cycle has to yield more tokens. GLM-5.3-Flash ships one MTP head, and drafting depth 3 on it measured 4% slower. **24+ tok/s single-stream at 4K is a model limit, not an engineering one.**

**Where the value is: depth.** At 41K, the indexer's pool scoring and validation are 15.9% of the verify graph, and they grow linearly with depth. The remaining steps in the depth plan are all exact and all unbuilt:

- epoch-based pool persistence instead of content validation (about −33 ms per cycle at 128K);
- tensor-parallel indexer scoring;
- sparse prefill.

Sparse prefill is the largest lever for long sessions. Prefill at 40K spends 21.6 of 35.4 s per micro-batch running the dense attention kernel over every cell behind a −inf mask. Gathering each query's 2,048 selected cells instead is projected to take prefill at 128K from 14.5 to about 60 tok/s. [Depth benchmark](../../benchmarks/glm53-flash-context-depth-20260920.md).

## Runtime and evidence

The four-socket Flash work covers tensor distribution, quantized kernels, shared/expert execution, barriers, Q8 MTP2, and later Q4 staging. The controlled Q8 barrier experiment reached 239.34–241.37 adjusted GB/s in raw decode. Its MTP2 effect was small and mixed. [Controlled comparison](../../engineering/2026-09-08/archive/serving/fleet-0903/FLASH-BANDWIDTH250-20260910.md).

The later Q4 target with a Q8 MTP2 draft measured 15.76–15.94 prose and 15.68–15.93 code tok/s under the report's recorded background load. Those observations are separate from the Q8 barrier comparison. [Q4 switch and qualification](../../engineering/2026-09-08/archive/serving/fleet-0903/FLASH-Q4-SWITCH-20260910.md).

## Latest prepared recipe

The September 11 unary/scale comparison records 15.39 to 15.71 mean decode tok/s with three identical greedy outputs. The [September 12 recipe](../../engineering/2026-09-12/archive/serving/fleet-0912/glmflash/README.md) rebuilds the durable candidate library and checks the launch dependencies. Those build checks are not a new speed measurement.

## Reproduction entry points

- [Current host recipe, source overlays, and rebuild instructions](../../engineering/2026-09-12/archive/serving/fleet-0912/glmflash/README.md).
- [Fresh A/B/A controller](../../engineering/2026-09-12/archive/serving/fleet-0912/AB-PROFILES.md).
- [Parameterized Q8 and Q4 profiles](../../profiles/20260908/README.md).
- [Source bundle inventory](../../engineering/2026-09-12/source-bundles.json): distinguish the goal source from earlier bring-up trees.

The Q4 payload is staged on a volatile RAM-backed volume, approximately 186 GiB. Reboot staging, per-node placement, and coexistence with Full affect the operational recipe. Fresh controls, completed-answer checks, and larger-context measurements remain open.

## Before the first launch

The published engine needs [one patch](../../patches/glm5next-x16-moe-expert-bound.patch) to run this model at all: the x16
mixed-expert path sizes its active-expert list at 256 and this model has 288 routed experts, so the first decoded token aborts
on `GGML_ASSERT(n_as <= 256)`. The four-socket host ran a private object with the larger bound, so no recorded result is
affected, but nothing in this repository exercised the published path at that expert count until an outside tester did.

Speculative decoding also needs the MTP draft as its own file, and there is no download for it. The `nextn` tensors ship
inside the main GGUF — a stock loader lists them as `blk.45.nextn.*` unused and ignores them — and
[extract-glm5next-mtp-gguf.py](../../engineering/2026-09-08/archive/serving/glm53-flash/extract-glm5next-mtp-gguf.py) writes
them out as the standalone sidecar that `--spec-draft-model` expects.

## Running it on fewer sockets

Every number on this page is from four sockets of sixteen cores with twenty-four DDR4-2400 channels, measured at 381.6 GB/s
aggregate read. Decode is bandwidth-bound, so on a smaller machine the rate scales with memory bandwidth first and core count
second. Measure the ceiling before downloading 186 GiB of weights — [tools/membw.c](../../tools/membw.c) reports it per socket:

```bash
gcc -O3 -march=native -mavx512f -fopenmp -o membw tools/membw.c
NODES=$(numactl -H | awk '/^node [0-9]+ size/ {n++} END{print n+0}')
CORES=$(lscpu | awk -F: '/^Core\(s\) per socket/{gsub(/ /,"",$2); print $2}')
for n in $(seq 0 $((NODES - 1))); do (numactl --cpunodebind=$n --membind=$n ./membw 8 "$CORES" &); done; wait
```

Add the per-node figures together: that sum is the number these results are bound by.

Scale the figures above by your total over 381.6 GB/s for a first estimate, then subtract for the work that is not at the memory
wall — the lightning-indexer scoring kernel and the small-operation chains follow core count, not bandwidth. A dual-socket
Cascade Lake with six channels per socket lands near half this machine's bandwidth and under a third of its cores, which puts
it around half the decode rate.

[`profiles/glm-flash-any-sockets.json`](../../profiles/glm-flash-any-sockets.json) is parameterized for this: it takes a
device list, a tensor split, a thread count and a context size, and its `GGML_CPU_NUMA_THREADS` follows the thread count
rather than the four-socket constant. Render it with `tools/render_profile.py` and read the command before running it.

What must change in the recipe:

- `--device CPU-NUMA0,CPU-NUMA1 --split-mode tensor --tensor-split 1,1` for two nodes, and one worker thread per physical core
  less one per socket, which leaves a core for that device's dispatcher.
- `GGML_CPU_NUMA_DEVICES=1` is required or the CPU-NUMA devices are never created; the failure looks like missing support
  rather than a missing switch. See [PORTING.md](../../patches/PORTING.md).
- Do not launch the server under a restrictive `taskset`: a restricted mask yields one worker per socket.
- Context costs memory per node. Tensor-splitting across two nodes mirrors less than across four, so the resident set is
  smaller, but a million-token context still adds tens of gibibytes per node. Start at 32K and grow it.

An AVX-512 machine without VNNI will load and run, but the 16-column integer kernels that produce these rates do not apply and
the generic paths are much slower. Cascade Lake and later have VNNI; Skylake-SP does not.

