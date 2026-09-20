# GLM phase-accounting correction

The archived September 14 GLM timing report overstated time outside graph execution. Its Qwen-specific parser classified graphs below 100 nodes as `tiny`, then omitted that class from the cycle sum. GLM's draft model has 89 nodes in these logs. All 800 draft graphs in each arm were omitted, including prompt ingestion and decode.

A normal MTP2 cycle runs the target once, catches up the draft KV state once with the target batch, and predicts two draft tokens in two separate passes. The draft catch-up is visible in the logs and in `common_speculative_impl_draft_mtp::process()`; the other two calls come from `draft()`.

The replacement audit accounts for every graph, checks prompt token totals against each saved response, and checks target verification counts against `predicted_n - 1 - draft_n_accepted`. The first generated token comes from prompt evaluation. Counting `predicted_n - draft_n_accepted` introduces an extra cycle.

## Corrected historical measurements

Each row summarizes 256 steady cycles across three sequential requests. A selected cycle starts at one completed, reused, three-token target graph and ends at the next, with exactly one three-token draft catch-up and two one-token draft passes between them. Prompt evaluation, target rebuilds and tail shapes are excluded from this steady-cycle comparison. Their time remains included in the separate full-request accounting in the JSON artifact.

| September 14 arm | Wall per cycle | Target graph | All three draft passes | Outside graph phases | Draft preparation, included in draft time |
|---|---:|---:|---:|---:|---:|
| Original | 161.52 ms | 133.59 ms | 17.70 ms | 10.23 ms | 4.29 ms |
| Split-state cache fix | 161.29 ms | 133.63 ms | 18.10 ms | 9.57 ms | 4.39 ms |

These are instrumented historical measurements, not current production timing or a new throughput gain. Logging, synchronization, sampling, cache management and server work can contribute to the residual. The old report's approximately 17-22 ms outside-graph estimate is not a valid attribution.

The CPU node profiler has a separate accounting distinction: its `total` sums node execution only; the `barrier` field is additional. The recent wider-pool target sample therefore represents 162.518 ms of node work plus 19.278 ms of reported barriers, or 181.796 ms, excluding its final graph barrier. It must not be compared with end-to-end wall time using node work alone. Samples from different NUMA workers or different draft calls cannot simply be summed into a cycle.

## Implications

The old logs support investigating about 10 ms of host work and about 4.3 ms of draft-graph preparation per cycle. They do not establish the current cost, nor a path from 15 to 18 tok/s by themselves. A fresh phase profile requires starting a process with `LLAMA_GRAPH_PHASE_ARM_FILE` set; that variable is absent from the current process and its first-use value is cached. No runtime environment injection or profiling restart was attempted for this audit.

The existing bounded KV rollback scan remains a candidate. Its correctness argument is that cells beyond `used_max_p1()` are empty and cannot match the nonnegative removal range. Rebuilding it requires an exact production library baseline; work to reconstruct that baseline is isolated under `glm-kv-range-0919`.

Production remained PID 3692141 with the wider-pool CPU hash `3f957a341321b940d93be53c250cdd068825093faa2d9efda142ebe56427b1b3`. Command, start time, inference environment and mapped library hashes matched the deployment audit. The endpoint was healthy and idle, and operation profiling was disarmed.

Evidence: [reproducible audit](audit_glm_phases.py), [full results with source hashes and cycle line references](glm-phase-audit.json), [runtime snapshot](phase-audit-runtime.json). The separate measured pooling improvement and deployment are recorded in [KPOOL-WIDE-RESULT.md](KPOOL-WIDE-RESULT.md).
