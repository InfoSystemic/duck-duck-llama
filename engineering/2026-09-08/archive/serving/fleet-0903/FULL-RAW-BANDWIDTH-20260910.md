# GLM-5.3 Full raw decode observation — September 10, 2026

Full produced 7.44–7.57 generated tok/s and 219.13–223.91 adjusted decode GB/s across four measurements. It did not reach either 240 or 250 GB/s. The trial failed its repeated-output check: both prose and code diverged between identical greedy requests. These are retained hardware traffic observations, not a qualified tuning comparison or a selected runtime.

The [frozen controller result](results/full-raw-baseline-0910/result.json) records that failure and successful restoration of the selected Flash Q4/Q8-MTP2 service at PID 2334769. The [independent failure-aware audit](results/full-raw-baseline-audit-0910b.json) passes its evidence and restoration checks; it explicitly keeps `model_run_passed: false`. An audit passing does not turn the model run into a successful repeatability test.

## Measured traffic

| Repetition | Workload | Generated tok/s | Total decode GB/s | Adjusted decode GB/s |
| ---: | --- | ---: | ---: | ---: |
| 1 | prose | 7.5670 | 229.562 | 223.574 |
| 1 | code | 7.5339 | 229.152 | 222.056 |
| 2 | prose | 7.4368 | 227.401 | 219.128 |
| 2 | code | 7.5251 | 229.263 | 223.910 |

Both pairs share one fresh Full process. They are repeated measurements, not independent loads. The runtime uses the existing mixed Q4 precision, Q8 KV, context 32,768, 15 workers per socket, and equal tensor splitting over all four sockets. The draft model is absent (`--spec-type none`); all four rows have zero drafted, accepted-draft, and cached tokens.

The [measurement script](measure_full_raw_0910.py) requests temperature zero, seed 42, a 512-token cap, and `enable_thinking: false`. Despite that template argument, all four measured responses contain reasoning only and hit the 512-token cap without a completed answer. These generated-token rates must not be presented as completed-answer throughput or a verified non-thinking baseline. The short arithmetic and capital-city checks pass in both repetitions.

All 48 physical IMC read/write counters are present in each retained interval. The audit reparses raw counters and SSE output independently. Boundary intervals are excluded; decode windows cover 66.43–67.45 seconds. Adjustment subtracts the larger of the immediately adjacent five-second idle measurements (4.63–8.27 GB/s). All four idle attribution checks qualify. Traffic is system-wide, and this subtraction remains an attribution estimate.

Per-socket traffic is balanced at approximately 56–58 GB/s. Adjusted traffic is 57.7–58.9% of the user's 380 GB/s server denominator. Reaching 240 GB/s would require about 7.2–9.5% more adjusted traffic; reaching 250 GB/s would require about 11.7–14.1%. Those gaps do not establish equivalent achievable tok/s gains or the cause of the remaining stalls.

## Repeatability and lifecycle

The two prose outputs share 531 initial reasoning characters before diverging; the two code outputs share 472. Counts remain 512 in each case. Source, runtime settings, and mapped-library checks match; the cause of the output divergence is unresolved. A new controlled correctness investigation is required before accepting a Full tuning comparison. Historical MTP and profiled raw results use different conditions and do not isolate the effect of removing the draft model.

The [admission assessment](results/full-raw-admission-0910.json) verifies that the existing Full allocation plan fits after pausing Flash, with global and per-node reserves. It credits only Flash's anonymous non-file memory. No weights were deleted. The [controller](run_full_raw_baseline_0910.py) owns the lifecycle lock, monitors memory throughout load and measurement, and has [12 checked failure/restoration cases](results/full-raw-controller-checks-0910.json).

Full took roughly 30 minutes to load. Initial whole-file mapping prefetch preceded anonymous NUMA allocation and repacking; cold-load I/O is separate from decode bandwidth. No loader optimization was applied in this trial. On the repeatability failure, the controller stopped its own Full process and restored Flash's command, full environment, working directory, affinity, and mapped libraries. The independent audit confirms the owned Full PID is gone, Flash is healthy, and the lifecycle lock is available. Private restoration context is neither read by the audit nor included in publication.

Reproduction uses the [admission helper](assess_full_raw_admission_0910.py), controller, measurement script, [lifecycle checker](check_full_raw_controller_0910.py), and [failure-aware audit](audit_full_raw_baseline_0910b.py). The earlier success-only audit was prepared but not executed. Executed sources and outputs are frozen; reruns require fresh labels. No runtime promotion occurred, and the bandwidth objective remains open.
