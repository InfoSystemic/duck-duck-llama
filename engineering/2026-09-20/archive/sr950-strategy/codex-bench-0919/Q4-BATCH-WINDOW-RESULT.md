# Q4 batch-kernel test result

The approved test completed successfully. Production was restored after **17.20 minutes** and is healthy on its original kernel, with 15 workers per socket. The candidate was not deployed.

**The batch kernel did not produce a convincing overall speed gain, and it did not reach 18 tok/s in Codex.**

| Measurement | Production before | Candidate | Production after |
|---|---:|---:|---:|
| Native aggregate | 15.393 tok/s | 15.255 tok/s across both groups | 15.231 tok/s |
| Codex cold | 14.448 tok/s | 14.352 tok/s | 14.598 tok/s |
| Codex cached | 14.533 tok/s | 14.621 and 14.602 tok/s | 14.293 tok/s |

The candidate's cached Codex mean was 1.38% above the mean production controls, while those controls drifted by 1.65%. Cold Codex was 1.18% slower than the mean controls. Native aggregate was slightly slower than the combined controls. This small sample does not establish an overall improvement; all measurements are included.

All six native candidate outputs and all three candidate Codex outputs matched their corresponding production controls exactly. The Codex runs generated the same 270 tokens, with the same 210 draft tokens, 164 accepted drafts, and 105 verification steps. All nine stateful/parallel scenarios also matched their production references. The existing cached-versus-fresh inconsistency is unchanged.

Standalone verification had already passed 1,080 direct kernel cases and 1,304 distinct graph cases. Those synthetic gains depended on how often tokens shared expert routes; they did not translate into a clear full-model gain here. The bounded full-model profile confirmed unchanged graph structure, not a direct count of batch kernel calls. Instrumented profile timings are excluded from throughput comparisons.

Final production PID: `1756372`. Approved CPU SHA256: `4793379e6894a9286168f79c4f323985388ec0b0ac65e54013dcd2957488af1f`. The command, inference environment, and mapped library hashes matched the original runtime. Both slots were idle at final verification. Private PID `1282309` is dead, port `18141` is closed, and profiling is disarmed.

Candidate SHA256: `90282e20361e22cffbdd08e2aab25e4a510aeadc199df781727d1adf8ecc35e1`. Keep the current production kernel. No promotion or production configuration change occurred.

Complete measurements are in [q4batch-window-summary.json](q4batch-window-summary.json), with the original run in [q4batch-window-report.json](q4batch-window-report.json) and final runtime verification in [q4batch-window-final-runtime.json](q4batch-window-final-runtime.json). The completed window must not be rerun.
