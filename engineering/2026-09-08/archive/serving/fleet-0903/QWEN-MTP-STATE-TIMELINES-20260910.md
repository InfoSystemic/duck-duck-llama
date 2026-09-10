# Qwen MTP fresh-request state and four-socket timing evidence

The 40 generated tok/s Qwen Q6 target and all-model 250 GB/s target remain incomplete. This follow-up isolates a request-history issue, tests a private MTP carry reset, and obtains output-matched socket timelines. No controlled model speedup is established.

The reset removed the observed fresh-request variation in this test: 8 of 8 repeat requests matched their own references in both output and token/draft counts. The candidate retains the Q6 target, Q8 draft, MTP4, and original corrected CPU kernels. It is not promoted to the saved serving profile.

## Request history and the isolated change

The original corrected CPU reproduced the mismatch with instrumentation disabled. In the controlled diagnostic run, code → prose and code → explicit slot erasure → prose produced the same changed response. Repeated prose also varied. Cache erasure alone does not reset the MTP cross-batch hidden vector.

The MTP process hook shifts target hidden rows and seeds the first row from pending_h. That carry previously survived a fresh position-zero request. The candidate zeros that vector at position zero, before draft prefill consumes it; nonzero continuation positions keep their carry. This restores the constructor's fresh-sequence seed at each new sequence.

The baseline object and pinned common library were reproduced byte for byte before the small patch. A build guard initially mistook its owned --version command for competing inference; a separate version-only validation completed. An initial model candidate was stopped before requests because a reused map filter excluded libllama-common. The corrected controller verifies that library directly and records its actual mapped path. Both failed attempts remain in the archive.

Private candidate common-library SHA-256: `bd121009dcbb121005a587262bfda25008cd63dc8d46016c3f4fbd50d5458f8d`.

This verifies the listed fresh-request workloads. It does not establish target-only numerical parity, general task quality, or a performance gain. The selected Flash Q4 command, environment, affinity, and libraries were restored after every model experiment.

## Uninstrumented reference measurements

These runs record existing host load and use different request-state conditions for the candidate. They are not an A/B speed comparison. Bandwidth is DRAM read plus write, with the larger adjacent idle baseline subtracted. The audit reparses all 48 IMC counters.

| Run | Prose tok/s | Code tok/s | Adjusted GB/s, prose / code |
| --- | ---: | ---: | ---: |
| qwen-whole-server-timeline-0910 | 22.85 | 28.42 | 131.24 / 131.26 |
| qwen-whole-server-timeline-0910b | 23.40 | 29.15 | 136.42 / 137.68 |
| qwen-repeatability-0910c | 22.48 | 28.61 | 131.34 / 134.90 |
| qwen-order-control-0910d | 22.28 | 28.72 | 123.78 / 136.58 |
| qwen-mtp-fresh-seed-0910f | 21.72 | 21.46 | 125.64 / 130.80 |

The [speed and headroom assessment](MODEL-SPEED-HEADROOM-20260910.md) records prior peaks and the conditional 40 tok/s bandwidth budget. These new experiments establish neither 40 tok/s nor 250 GB/s.

## Matched timeline observations

Both diagnostic captures match their unarmed references in full output hashes and generated/drafted/accepted counts. Each captures 160 graphs. Timestamps use a shared monotonic clock; concurrent socket intervals are counted by their union. The samples cover 5–6 complete target waves and 26–27 complete draft waves per workload.

| Workload | Target interval union, ms | Draft interval union, ms | Matrix share of last-finishing target stage time | Draft output projection share of last-finishing draft stage time |
| --- | ---: | ---: | ---: | ---: |
| prose | 595.752 | 87.533 | 62.3% | 42.7% |
| code | 527.550 | 87.761 | 59.6% | 45.4% |

Target matrix stages are the first optimization priority. The audit includes weight families, quantization types, shapes, and outer-barrier times for each matrix group. Stage time includes waits; the last-finishing socket does not prove a removable fraction of request latency. Instrumentation and logging overhead also occur outside graph intervals. Profile throughput is excluded from speed claims.

The raw timeline logs remain on the server and are identified by size and SHA-256 in the compact audit. They are excluded from the filtered repository export. Private restoration context is neither read by the audit nor exported.

## Evidence

- [Combined counter, output, restoration, and timeline audit](results/qwen-order-and-timeline-audit-0910b.json)
- [Original CPU request-order probe](results/qwen-repeatability-0910c/result.json)
- [Controlled erasure and matched timelines](results/qwen-order-control-0910d/result.json)
- [Private common-library build](results/qwen-mtp-fresh-seed-build-0910/result.json)
- [Completed runtime validation](results/qwen-mtp-fresh-seed-runtime-0910/result.json)
- [Corrected candidate model experiment](results/qwen-mtp-fresh-seed-0910f/result.json)
- [Lifecycle and library-map fault checks](results/qwen-mtp-fresh-seed-checks-0910f.json)
