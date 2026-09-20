# GLM-5.3-Flash

**Status (2026-09-20):** the Q4 service delivers 15.5–16.2 tok/s to a Codex agent inside Paseo at ~4K tokens of context. A candidate kernel stack measured 17.61 tok/s in the same loaded process and is not deployed. 18 tok/s is not reached. At ~30K tokens production measured 10.26 tok/s on 09-19.

## Decode through Paseo, September 18–20

The measurement is an actual Codex turn through the Paseo wrapper, modes switched inside one loaded server ([report](../../benchmarks/glm53-flash-paseo-decode-20260920.md), [patches](../../patches/README-20260920-glm5next.md), [records](../../engineering/2026-09-20/README.md)).

- **Deployed 09-19, all output-identical:** fused pooled-indexer compressor and a flat recurrent-state copy (the state copy had been running on one worker: 20.4 → 1.2 ms per cycle), an eight-channel pool loop (+3.8%, +13.4% at 30K), a KV rollback that scans the used prefix instead of 1,048,576 cells (+2.4%), and an MTP catch-up graph that stops after the KV write when no output is requested (+2.5%).
- **Measured 09-20, not deployed:** a cell-split F32 attention kernel for the MQA cache with a selection top-k (16.24 → 17.23), and a content-verified pooled-result cache on top (→ 17.61). The attention kernel changes generated text because the stock kernel sums V in FP16 and is about 1% off a float64 reference on real tensors ([report](../../benchmarks/cpu-flash-attn-f16-accumulation.md)); promotion is an owner decision.
- **An integration fix worth as much as a kernel:** the Codex model catalog lacked the lowercase alias Paseo sends, so requests carried 20,751 characters of default instructions. Adding it cut the prompt from 7,739 to 3,998 tokens; decode is context-sensitive, so this alone moved 9.6 → 12.0 tok/s.
- **Codex sends no sampling fields.** Real turns run at the server defaults (the GGUF carries temp 1.0) and accept fewer drafts: 70% against 78% greedy in one pair of runs, 14.8 against 15.6 tok/s. Greedy numbers are for comparisons, not a promise.

Where the time is: the routed experts take 42 ms of a 161 ms cycle and already stream at the machine's measured DRAM ceiling with every expert split across the four sockets, so expert kernels are finished. The remaining ~119 ms is attention, the indexer, small hyper-connection ops, the draft loop and host work, which is where every gain above came from. Draft depth above 2 loses (−22% at depth 4); fusion switches, Q4/Q8 token batching, 16 workers per socket and helper pinning gave nothing.

Open: a draft-loop restructure is written and has not run; no 30K measurement exists for the 09-20 kernels; a cached-versus-fresh output discrepancy recorded on 09-19 is unresolved.

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
