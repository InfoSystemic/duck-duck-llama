# Flash Q8 whole-server comparison, September 10, 2026 (UTC)

The eight-run control/candidate/candidate/control comparison completed for raw decode and MTP2. Each run measured fresh single-conversation prose and code with 512 generated tokens. The Q8 target, Q8 draft, 15 workers per socket, and all other arithmetic options were retained. The candidate enables the validated dissemination barrier.

| Configuration, two runs each | Prose tok/s | Code tok/s | Prose adjusted GB/s | Code adjusted GB/s |
| --- | ---: | ---: | ---: | ---: |
| Raw control | 11.48-11.55 | 11.48-11.53 | 236.56-238.39 | 236.90-238.76 |
| Raw barrier | 11.61-11.68 | 11.69-11.69 | 239.34-239.56 | 240.36-241.37 |
| MTP2 control | 15.20-15.24 | 16.37-16.54 | 229.66-230.99 | 233.24-235.78 |
| MTP2 barrier | 15.25-15.38 | 16.38-16.39 | 230.21-232.66 | 233.55-234.21 |

Measured mean candidate speed changes relative to the two matched controls:

| Mode | Prose | Code |
| --- | ---: | ---: |
| raw | +1.12% | +1.62% |
| mtp | +0.60% | -0.45% |

Retain the barrier for further raw tuning. Its MTP2 effect is small and mixed, so keep the control setting for MTP2. No production profile was changed.

All 16 stable decode windows pass coverage for the 48 IMC counters. The independent audit reparses the raw counter CSVs, reproduces the saved decode summaries, and checks adjacent idle traffic. Reported GB/s subtracts the larger adjacent idle value from systemwide traffic, so attribution remains an estimate. Every complete output stream and generated/drafted/accepted count agrees with its matching mode control.

The earlier MTP2 measurement used an older CPU stack and measured 13.60/14.78 generated tok/s. All four new MTP2 runs preserve its full output and its 426/297 prose and 386/317 code drafted/accepted counts. That older measurement was not interleaved with this comparison.

All GLM windows exhaust the 512-token limit during reasoning. These are generated-token rates; they do not measure completed-answer latency or certify near-lossless quality against the source checkpoint. The experiment retains the chosen Q8 precision.

None of the four configurations met the repeated 250 GB/s criterion. The three-model 250 GB/s objective remains incomplete. Full and Qwen have no new model measurements in this comparison.

The temporary model handoff ran under the existing authorization for whole-server model optimization. The earlier assistant-imposed peer-pause question was unnecessary and no longer blocks these trials. Qwen was restored as PID 3147059, start ticks 115029368, with its exact command, full environment, working directory, affinity, and library bindings. The independent audit confirms its new identity and runtime, the inactive Full service, free private ports, and released trial state. The private restoration environment is excluded from publication.

The next prepared Flash experiment changes only the existing NUMA huge-page advice. Its seven simulated handoff failure checks pass. It checks actual page backing before and after each measurement and refuses an unstable or ineffective page-size comparison. No huge-page model trial has run as part of this publication.

Full allocation planning uses its existing no_alloc model/context simulator with an 8 GiB virtual-address limit. The helper never calls decode. Its estimate is separate from an actual model load and includes no throughput claim.

- [Completed trial](results/flash-bandwidth250-comparison-0909/result.json)
- [Independent counter and restoration audit](results/flash-bandwidth250-assessment-0910.json)
- [Executed controller](flash_bandwidth250_trial_0909.py)
- [Huge-page controller](flash_hugepages250_trial_0910.py)
- [Huge-page lifecycle checks](results/flash-hugepages250-lifecycle-0910.json)
- [Full asset inspection](results/full-bandwidth250-assets-0910/result.json)
- [Full allocation planner](plan_full_memory_0910b.py)
- [Preserved first link failure](results/full-memory-plan-0910/result.json)
- [Full allocation planning result](results/full-memory-plan-0910b/result.json)

The allocation simulator reports 468.34 GB total (466.30 GB model, 1.66 GB context, 0.38 GB compute), using 0.09 GB peak actual RSS. With a 32 GiB global reserve, the estimated shortfall after pausing Qwen is 18.46 GB. This excludes the separate MTP model and temporary load copies; it is not a successful Full load.
