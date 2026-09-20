# GLM-5.3-Flash

**Status (2026-09-20):** the Q4 service delivers 20.0 tok/s to a Codex agent inside Paseo at ~4K tokens of context on the fixed-output fixture, and 19.0 tok/s on requests exactly as Codex sends them (mean of twelve, 17.7–20.1). It was 15.5–16.2 on 09-19 and 12.0 on 09-18. At 13,231 tokens it measures 17.0 tok/s; at ~30K production measured 10.26 tok/s on 09-19 and has not been re-measured.

## Decode through Paseo, September 18–20

The measurement is an actual Codex turn through the Paseo wrapper, modes switched inside one loaded server ([report](../../benchmarks/glm53-flash-paseo-decode-20260920.md), [patches](../../patches/README-20260920-glm5next.md), [records](../../engineering/2026-09-20/README.md)). Everything below is in production.

- **09-19, all output-identical:** fused pooled-indexer compressor and a flat recurrent-state copy (the state copy had been running on one worker: 20.4 → 1.2 ms per cycle), an eight-channel pool loop (+3.8%, +13.4% at 30K), a KV rollback that scans the used prefix instead of 1,048,576 cells (+2.4%), and an MTP catch-up graph that stops after the KV write when no output is requested (+2.5%).
- **09-20, kernels:** a cell-split F32 attention kernel for the MQA cache with a selection top-k (16.24 → 17.23) and a content-verified pooled-result cache (→ 17.61). The attention kernel changes generated text because the stock kernel sums V in FP16 and is about 1% off a float64 reference on real tensors ([report](../../benchmarks/cpu-flash-attn-f16-accumulation.md)); its second revision is also bit-identical for a query alone and inside any batch, which the first was not. A gated-delta-net kernel split by state row and an MTP graph that builds queries only for the predicting row are bit-exact (+2.3% together, +4.5% at 13K).
- **09-20, draft loop and sampling:** catch-up merged into the first draft pass, constant-shape draft batches so both draft graphs are reused (+2.1%), a direct draft pick, top-k candidates taken straight from the logits (+1.6%), and coupled draft/verifier sampling: the verifier's pick becomes Gumbel-max with noise the drafter shares, an exact sampler that raised sampled-request acceptance from 72.4% to 75.0%.
- **09-20, scheduling:** trunk and draft each owned an OpenMP team pinned to the same 60 cores, and libgomp's default idle spin (~15 ms on this CPU, not the documented 3) made every hand-off a collision. `GOMP_SPINCOUNT=20000` was worth more than any single kernel that day (18.2–18.4 → 18.8); one shared team per device makes the collision impossible.
- **09-20, the host side:** the tensor-parallel backend started one thread per NUMA device for every graph-input upload, about 93 thread creations per decode cycle, found with `/proc` and strace because perf is not available. Uploads below 1 MiB now run in the caller and the dispatch waits block instead of spinning: 19.30 → 20.18, output identical.
- **An integration fix worth as much as a kernel:** the Codex model catalog lacked the lowercase alias Paseo sends, so requests carried 20,751 characters of default instructions. Adding it cut the prompt from 7,739 to 3,998 tokens; decode is context-sensitive, so this alone moved 9.6 → 12.0 tok/s.
- **Codex sends no sampling fields.** Real turns run at the server defaults (the GGUF carries temp 1.0) and accept fewer drafts, which is why two numbers are reported. Greedy numbers are for comparisons, not a promise.

Where the time is: a cycle is now ~127–131 ms (161 on 09-19). The routed experts take 42 ms and the large dense projections 37, and both already stream at the machine's measured DRAM ceiling with every matrix split across the four sockets, so those kernels are finished. What moved was attention, the indexer, the draft loop, host work and scheduling. Draft depth above 2 loses (−4% at depth 3 on the current stack, −22% at depth 4 on 09-18; a verified row costs 24 ms); fusion switches, Q4/Q8 token batching, 16 workers per socket, helper pinning, a multi-slot graph cache and a Q4 draft sidecar gave nothing.

Open: single sampled turns still fall below 18 (three of twelve); ~90 tiny hyper-connection projections per graph cost 2–3 ms of launch overhead; a cached-versus-fresh output discrepancy recorded on 09-19 is unresolved.

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
