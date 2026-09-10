# GLM-5.3-Flash Q4 selection, September 10, 2026 (UTC)

The user explicitly instructed: "Okay switch GLM-5.3-Flash to Q4." GLM Flash now selects UD-Q4_K_XL with the retained Q8 MTP draft and depth 2. It runs on localhost port 18131 with aliases `glm-flash-goal`, `glm-flash-q4`, and `GLM-5.3-Flash`. Qwen retains its Q6 configuration and weights and is unloaded while Flash uses the server. Full is inactive.

All six Q4 shards, totaling 199,707,321,347 bytes, match the publisher SHA-256 manifest at revision `621d456e93e926e4b52f85cff5f634358c1828f9`. Five shards completed in the original resumable downloader. The final slow connection was stopped, its progress preserved, and its missing ranges fetched with six bounded workers. Its entire payload then passed SHA-256 verification. The completion audit records this provenance explicitly.

The Q8 files were retained. Q4 occupies a separate owner-only 210 GiB tmpfs model volume. The selected configuration persists on disk; the Q4 payload volume is volatile across reboot. Restaging and refreshing the verified file identities are required if those RAM-backed files are lost.

The first activation plan waited on a clean-benchmark CPU gate while Chrome used about 14 cores. It was stopped before any model mutation. Activation was separated from performance qualification. A subsequent attempt stopped at the per-node RAM reserve and restored Qwen exactly. Moving 12 GiB of inactive Q4 pages from node 2 to node 1 repaired the imbalance; the affected full shard checksum remained unchanged. The following activation passed arithmetic, geography, and four default-cache-off continuation checks. Eleven bounded lifecycle/launcher fault checks also pass.

The selected launcher verifies model identities and runtime hashes and refuses insufficient RAM or a competing loaded model. The controller retains exact Qwen rollback on an activation failure. Private restoration environments are excluded from publication.

Performance requests use fresh single-conversation prose/code, a 4096-token context setting, a 512-token generation limit, Max reasoning, no reused prompt cache, and 15 workers per NUMA socket. The runtime retains the validated dense Q8 optimizations and existing Q4/Q5/Q6 expert kernels, with no extra lower-precision load-time requantization. Huge-page advice and the dissemination barrier are off.

| Run | Prose generated tok/s | Code generated tok/s | Prose adjusted GB/s | Code adjusted GB/s | Adjacent idle qualifies | Background cores before/after |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| 1 | 15.94 | 15.68 | 181.74 | 179.72 | True | 17.6/18.4 |
| 2 | 15.76 | 15.93 | 178.49 | 184.05 | True | 17.9/14.6 |

The benchmark records current host load instead of waiting indefinitely for other applications. These rates are not an uncontended maximum or an isolated Q4/Q8 speed comparison. All reported counter windows are checked against the 48 IMC read/write counters and stable streamed decode intervals. Adjacent idle qualification requires each baseline at most 19 GB/s and a before/after difference at most 9.5 GB/s. Systemwide traffic minus the larger adjacent idle baseline estimates model-attributable traffic. Generated rates include reasoning; they are not completed-answer latency.

The repeated measurement controller passed: **True**. It establishes repeated 250 GB/s on both workloads: **False**. The all-model bandwidth goal remains incomplete. This user selection supersedes the earlier Q8-only selection rule and does not certify Q4 as near-lossless.

Qwen Q4_K_XL reduces the inventoried active target weight estimate from 6.964 to 6.359 GB per raw token, an 8.69% byte reduction and 9.51% weight-only speed headroom at equal bandwidth. About 4.8 GB of dense/shared/other weights remains unchanged. The large indexed table shrinks on disk but is not streamed in full per token. State traffic, kernel behavior, and MTP acceptance need separate measurements. Qwen remains Q6; its Q4 variant has not been benchmarked or selected.

The earlier Q8 huge-page experiment completed one control run, then was cancelled during candidate loading when the user selected Q4. Qwen restoration passed. With no candidate measurement, it establishes no huge-page gain or regression.

- [Selected configuration](glm-flash-selected.json)
- [Successful activation controller](select_flash_q4_0910c.py)
- [Activation result](results/glm-flash-q4-selection-0910c/result.json)
- [Independent activation audit](results/glm-flash-q4-activation-audit-0910c.json)
- [Current-host-load measurements](results/glm-flash-q4-measured-0910/result.json)
- [Independent measurement audit](results/glm-flash-q4-measurement-audit-0910.json)
- [Benchmark controller and counter validation](benchmark_flash_q4_selected_0910.py)
- [Download verification](results/glm-flash-q4-download-0910/status.json)
- [Parallel completion audit](results/glm-flash-q4-final-shard-0910/result.json)
- [Verified NUMA page rebalance](results/glm-flash-q4-page-rebalance-0910/result.json)
- [Lifecycle checks](results/flash-q4-selection-checks-0910c.json)
- [Q4 byte estimates and superseded Q8 trial audit](results/q4-switch-assessment-0910.json)
