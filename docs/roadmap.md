# Open engineering work

The objective remains to make all four target models work and run as well as possible on this CPU server. The following work has not been completed by publishing the repository.

| Priority | Next decisive work | Evidence required |
| --- | --- | --- |
| Full loadability | Resolve node-3 allocation pressure; evaluate corrected unequal splits and cache placement | Completed load, measured per-node headroom, model/asset identity, semantic and repeatability checks |
| Flash and Qwen latest recipes | Run the prepared fresh-server A/B/A comparisons | Stable controls, identical requests/outputs, cache-state accounting, sufficient repetitions |
| DeepSeek practical serving | Reduce cold tensor/Engram costs and validate longer generation | Varied real-model requests, storage/network accounting, useful completed-answer latency |
| DeepSeek extended context | Validate the 4K/16K wrapper or complete the native engine path | Real-weight attention/logit/text checks across boundaries, interruption/reset behavior, bounded memory and cache use |
| Cross-model quality | Expand beyond short semantic probes and fixed workloads | A documented task suite, quality results at each selected precision/context, repeatable comparisons |
| Portability | Reconstruct and qualify the source/profile stacks on a clean host | Dependency provenance, clean builds, topology-aware configuration, model-level regression evidence |

Candidate code, metadata tests, and successful builds are useful intermediate results. Promotion requires the corresponding full-model evidence. Existing throughput and bandwidth goals remain objectives, not achieved capabilities.
