# Open engineering work

The objective remains to make every target model work and run as well as possible on this CPU server. MiMo-V2.6-Pro joined the four original models on 2026-09-21. The following work has not been completed by publishing the repository.

| Priority | Next decisive work | Evidence required |
| --- | --- | --- |
| Full loadability | Resolve node-3 allocation pressure; evaluate corrected unequal splits and cache placement | Completed load, measured per-node headroom, model/asset identity, semantic and repeatability checks |
| Flash and Qwen latest recipes | Run the prepared fresh-server A/B/A comparisons | Stable controls, identical requests/outputs, cache-state accounting, sufficient repetitions |
| DeepSeek practical serving | Reduce cold tensor/Engram costs and validate longer generation | Varied real-model requests, storage/network accounting, useful completed-answer latency |
| DeepSeek extended context | Validate the 4K/16K wrapper or complete the native engine path | Real-weight attention/logit/text checks across boundaries, interruption/reset behavior, bounded memory and cache use |
| GLM Flash at depth | Build the remaining exact depth steps: per-cell write epochs instead of pool validation, tensor-parallel indexer scoring, sparse prefill; measure revision h's score kernel at 128K | In-process A/B at 41K and 128K on sessions built by appending (never a restored one), greedy text parity, the 8K–128K curve |
| MiMo tokens per cycle | The verify cycle is at 79% of the memory wall, so only acceptance moves the rate: acceptance at depth (0.98 → 0.48 by 64K), prose, a quiet-machine re-measure of build 0922j | Per-workload rates with golden parity; drafter changes judged on acceptance |
| MiMo coexistence | Its per-node memory binding rules out any process above ~45 GiB on one NUMA node during load | A scheduling decision between workloads, then a guarded reload that completes |
| Cross-model quality | Expand beyond short semantic probes and fixed workloads | A documented task suite, quality results at each selected precision/context, repeatable comparisons |
| Portability | Reconstruct and qualify the source/profile stacks on a clean host | Dependency provenance, clean builds, topology-aware configuration, model-level regression evidence |

Candidate code, metadata tests, and successful builds are useful intermediate results. Promotion requires the corresponding full-model evidence. Existing throughput and bandwidth goals remain objectives, not achieved capabilities.
