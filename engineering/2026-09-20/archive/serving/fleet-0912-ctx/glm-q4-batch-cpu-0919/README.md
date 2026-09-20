# Q4 token batching candidate — 2026-09-19

Status: full-model regression checks passed; no convincing overall speed gain; not deployed. The approved pool/copy kernel is restored and healthy as PID1756372 with 15 workers per socket.

This candidate shares Q4 weight loads and nibble decoding across two or three token activations. Every token retains production's integer operations and per-block floating FMA order. Pointer arrays support nonadjacent routed inputs and outputs. It applies only to ordinary Q4_K x16 MUL_MAT and MUL_MAT_ID dispatch. Single-token calls, other quant types, and existing fused gate/up paths retain their previous implementation. This does not include the earlier Q4/Q5 clamped fusion experiment.

Enable only in a private test with `GGML_CPU_X16_Q4_BATCH=1`. The default is off. Optional `GGML_CPU_X16_Q4_BATCH_PROBE=1` counts calls through `ggml_cpu_q4_batch_count(2|3)`; counters were disabled for timings.

Candidate SHA256: `90282e20361e22cffbdd08e2aab25e4a510aeadc199df781727d1adf8ecc35e1`.
Approved parent SHA256: `4793379e6894a9286168f79c4f323985388ec0b0ac65e54013dcd2957488af1f`.
The parent relink was byte-identical. The candidate replaces the preserved repack.cpp object and adds one batch-kernel object; all other approved objects remain unchanged.

Validation passed:

- 1,080 direct kernel comparisons: bit-identical output and intact guards; multiple dimensions, zero/scaled inputs, nonadjacent and unaligned rows, two-/three-token batches and single-token fallback. No vector register spills in the compiled batch loops.
- 1,304 distinct graph cases, each compared across production, candidate disabled, and candidate enabled: 280 small expert, 80 full gate-shape expert, 80 full down-shape expert, and 864 dense/broadcast cases. All numerical outputs and expected batch counts matched.
- Full expert shapes are 4096×512×288 and 512×4096×288. Tests include 1/3/4/15 threads, token tails, padded inputs, eight activation rows per token, changing routes, extra consumers, clamps, and fusion fallbacks.
- Two dense counter expectations initially failed because the test assumed a 64-row maximum instead of production's 16-row setting, and did not account for broadcast planes disabling a fused path. All saved numerical outputs were already exact. Corrected count checks passed without changing the kernel. Preliminary reports are retained.

The first direct correctness run used the older validated-bin base library. An identity guard caught this before timing. Correctness was repeated using the exact production base library, with the same output digest; only matched-library timing results are reported.

Performance is limited to synthetic operation graphs, not whole-model tok/s. Each graph has two matrix operations, clamps, and SwiGLU; the down-shape graph tests its dimensions rather than reproducing the entire real down-projection subgraph. Timings use 15 pinned physical cores, counters disabled, off/on/off/on ordering, and 61 repetitions per median. Production task IDs were unchanged throughout both graph timing runs.

| Three-token Q4 graph | Same expert routes | Mostly different routes |
|---|---:|---:|
| Gate shape | 1.325× | 1.040× |
| Down shape | 1.167× | 1.013× |

Some unchanged single-token controls drifted by several percent. The small divergent-route differences are not convincing whole-model evidence. One-core microbenchmarks of the production 64-row expert tile showed 1.17–1.40× for two/three tokens across hot and rotating weight sets. Exploratory 512-row tiles are reported separately and are not representative of production's expert work tile.

The full result, including every timing row and control drift, is `/home/user/sr950-strategy/codex-bench-0919/q4batch-standalone-summary.json`. No evidence yet establishes 18 tok/s in Codex. The pre-existing cached/fresh mismatch remains unresolved.

The separately approved full-model window completed and restored exact production after 17.20 minutes. All six native and three candidate Codex outputs matched production, as did all nine stateful/parallel scenarios. The existing cache inconsistency remained unchanged. The bounded profile retained all 42 gate/up/down expert operations; it does not directly count live batch calls.

Candidate native aggregate was 15.255 tok/s versus combined production controls 15.311. Candidate Codex was 14.352 cold and 14.621/14.602 cached. Production before/after was 14.448/14.598 cold and 14.533/14.293 cached. The cached candidate's +1.38% versus mean controls is smaller than the controls' 1.65% drift; cold Codex was 1.18% slower. There is no convincing overall gain, and 18 tok/s was not reached. Do not promote this candidate on these results.

Final report: `/home/user/sr950-strategy/codex-bench-0919/Q4-BATCH-WINDOW-RESULT.md`; complete data: `q4batch-window-summary.json` in that directory. The private process and port are stopped, profiling is disarmed, and final production was verified healthy and idle. No production configuration change or automatic promotion occurred. Do not rerun the completed window.
