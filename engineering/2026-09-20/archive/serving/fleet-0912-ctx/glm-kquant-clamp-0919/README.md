# Q4/Q5 x16 clamped expert fusion experiment

This private library is **not deployed**. Its opt-in flag is
`GGML_CPU_KQUANT_CLAMP_FUSION=1`; the default is disabled.
The library preserves the approved pool/copy kernel and all frozen parent objects,
replacing only `repack.cpp.o`. `manifest.json` records exact source, object,
compiler, linker, and library hashes. `candidate.patch` contains the change.

The parent objects first relinked byte-for-byte to production. Tests used valid
reference-quantized weights and compared candidate off/on against that production
library. All 360 unique correctness cases passed exact float-output parity,
including full 4096x512x288 expert dimensions and required fallbacks. Optional
`GGML_CPU_KQUANT_CLAMP_PROBE=1` counters verified dispatch. The saved preliminary
small-test report reflects a corrected harness expectation: existing single-thread
graphs intentionally skip fusion. No numerical failure occurred.

The isolated Q4 operation improved, but this did not produce a convincing measured
Codex improvement. Full-model native, Codex, stateful, concurrent, and profiled
outputs matched references. A bounded trace confirmed all 42 expert layers fused.
The existing cached/fresh mismatch remains unresolved.

Actual greedy Codex decode tok/s:

| Setting | Cold | Cached |
| --- | ---: | ---: |
| Production before | 14.4238 | 14.2173 |
| Candidate | 14.7686 | 14.5568, 14.7565 |
| Production restored | 14.8898 | 14.8293 |

The candidate's 0.8-0.9% apparent advantage over mean controls is smaller than
3-4% control drift. All native groups were retained, including the slow first
candidate third prompt. Production was restored unchanged after 17.2 minutes;
15 workers/socket and the approved combined pool/copy kernel remain active.
18+ tok/s in Codex was not established. This candidate is not recommended for
promotion on these measurements alone.

The completed window is documented in
`/home/user/sr950-strategy/codex-bench-0919/kquant-window-summary.json`.
Do not rerun the outage harness: it refuses to overwrite the completed experiment.
A new full-model outage requires a separately staged plan and authorization.
