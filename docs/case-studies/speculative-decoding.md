# Speculative decoding as a serving feature

Speculation can increase emitted tokens per target forward pass. Its benefit depends on the draft model, workload, state handling, verification, and the cost of rejected work.

## Engineering

The project integrates detached MTP heads and request-level draft controls, then tests acceptance, stale state, rollback, and repeated requests. Qwen work includes a fresh-sequence MTP reset that passes eight repeated-request comparisons. [Request-state and timing report](../../engineering/2026-09-08/archive/serving/fleet-0903/QWEN-MTP-STATE-TIMELINES-20260910.md).

Retained Qwen UD-Q6_K_XL with a Q8 MTP4 draft measured 21.53 prose and 28.46 code generated tok/s. Later corrected-gather experiments reached 24.04–24.35 and 30.80–31.62 on their recorded workloads. These results identify different source stacks; they are not one aggregate improvement. [Recorded configurations and measurements](../../engineering/2026-09-08/README.md#retained-measurements).

## Controls that change the conclusion

Earlier Qwen sidecars achieved zero acceptance and made decode slower. Later source and sidecar work enabled useful MTP. A Full replay configuration produced a favorable result that a different later replay did not reproduce. Both are retained in the [earlier model-profile report](../../benchmarks/sr950-model-profiles.md).

The September 11 Qwen unary/ngram work records three-prompt means of 19.90, 20.32, and 20.80 tok/s for baseline, unary, and composite configurations. The [durable recipe](../../engineering/2026-09-12/archive/serving/fleet-0912/qwen/README.md) rebuilds its library and validates launch dependencies. A fresh matched comparison of that recipe remains pending; repeated prompts must not benefit from warmed ngram history unnoticed.

## Reusable comparison tooling

The [new A/B/A runner](../../engineering/2026-09-12/archive/serving/fleet-0912/AB-PROFILES.md) starts a fresh server for each arm, checks PID start identity and socket ownership, records libraries and requests, and refuses to count short or cached completions as throughput samples. It stops only its own verified child processes. If observation expires, the same live process can be resumed.

These controls make a result reviewable. They do not replace broader quality tests or enough repetitions to estimate variance. The next comparison should retain raw decode as well as speculation, because acceptance rate alone is not the optimization target.
