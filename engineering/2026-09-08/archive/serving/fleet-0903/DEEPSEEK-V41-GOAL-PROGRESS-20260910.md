# DeepSeek V4.1 Flash: progress toward >10 tokens/s

The goal is **not yet achieved**. The fastest complete-model result currently verified is **1.780877 decode tokens/s**, compared with a matched **1.096775 tokens/s** baseline (1.62374×). Both are batch-one warm generation on one 16-core socket. The two candidate measurements are 1.77275 and 1.78908 tokens/s. All fixed-16-Torch-thread candidate logits match the original CPU implementation exactly.

Authoritative selection: `deepseek-v41-selected.json`; always inspect its PID/start identity and endpoint health before operating. Endpoint: `http://127.0.0.1:18170/v1`. The selected GLM peer on port 18131 is preserved. The DeepSeek server owns the shared fleet lifecycle lock continuously; the separate `results/deepseek-v41-controller.lock` serializes DeepSeek trials and handoffs. Host process inspection and actual inference require host execution (`require_escalated`); the default shell has isolated `/proc` and `/dev/shm` namespaces.

**Measurement correction (16:39 UTC):** current DeepSeek affinity is physical cores **48–63**. Reserve their SMT siblings **112–127** too. **CPU127 is not a separate core**: heavy fixtures there share core63 and can stall the entire model worker team. Only lightweight sleeping controllers should use127. All new CPU-heavy fixtures/builds in this team use CPU32 or0. A Torch1 fixture on127 from16:32:43 to16:34:30 directly overlapped the sparse and later arms of the first exact16 trial. Those correctness checks remain valid, but the affected speed comparisons are confounded. A quiet repeat is queued as `deepseek-v41-goal-exact16-quiet-0910`.

## Completed real-model work

- `results/deepseek-v41-goal-wall-0910/trial/result.json`: corrected request-thread wall profile. Warm decode 8.2601 seconds for nine forward steps. Native FP4 GEMM 1.9469 s, FP8 GEMM 1.4969 s, activation quantization 1.5316 s, HC Sinkhorn .5003 s, sparse attention .5354 s. Timers are inclusive and can overlap.
- `results/deepseek-v41-goal-candidates-0910/trial/result.json`: native activation quantization improves decode to 1.3361 tokens/s; reuse alone to 1.2751. LUT GEMM alone 1.0723 versus baseline 1.1039, so it is not selected. Trial aborted when the first 8-thread run needed additional expert downloads; original endpoint was restored. That cold run is not a speed result.
- `results/deepseek-v41-goal-fused-0910/trial/result.json`: passed complete-model test of native quantization, input quantization reuse, grouped MoE projections, native HC, scripted sparse attention. Matched rates stated above. Eight Torch threads yield 1.4456 and four yield .9072 tokens/s after their own warmups, with changed logits but identical greeting tokens. Fixed Torch16/native8 is exact but slower (.8875). Sixteen workers remain the selected candidate configuration.
- `results/deepseek-v41-goal-promoted-0910/result.json`: passed promotion and two actual HTTP validations. Warm endpoint rate 1.77215 decode tokens/s. The selected entry is `deepseek_v41_goal_server_0910.py`.
- `results/deepseek-v41-goal-numa-0910/trial/result.json`: passed bounded native row-copy placement checks on the actual model (38.5 GB copied, physical node quarters sampled). Torch/native64 measured 1.1061 and 1.1508; 32 measured 1.2936. Same greeting tokens but changed logits. Regression; not selected. Endpoint restored.
- `results/deepseek-v41-goal-vnni-0910/trial/result.json`: all full-model greeting logits exact. Matched fused baselines 1.6244 and 1.5964; prepacked VNNI single calls 1.2163; grouped VNNI 1.4340 and 1.4245. No measured warm downloads; retained packed cache 32.76 GB. Grouped native inclusive time regressed from about 1.74 s to 2.38 s per nine decode steps. Small hot-matrix speedups did not carry to the model. Not selected; controller result confirms endpoint restored (PID 713415 at this writing, re-read selection before use).
- `results/deepseek-v41-goal-exact16-0910/trial/result.json`: all complete-model logits exact. Exact16 two runs1.7149/1.7859; baseline before1.5957/after1.4871; sparse1.5256, combined1.6897. **Later arms and baseline-after overlap the SMT-conflicting fixture above; do not promote from these averages.** Packed cache17.77GB/2,835projections; sampled native pages allnode3, packed pages mostlynode3 with two sampled late entries onnode0. One OpenMP library and stable worker affinity.
- `results/deepseek-v41-goal-chunk-0910/trial/result.json`: partial perfect-draft verification test. Sequential1.6530, width3=1.5677, width5=1.7351 ideal verified positions/s, exact logits. Width8 changed logits and downloaded94MB; the cold-data assertion aborted this arm and endpoint restoration passed. These are not served generation rates and exclude drafting. Per-token native HC in verifier v2 fixes the identified width8 sigmoid scalar/vector rounding difference in fixtures.

The benchmark prompt is `Hi.` and output is `Hello! How can I help you today?`. Ten output tokens include EOS; decode timing counts nine forward passes after prefill. The measured warm requests download zero bytes. These short runs establish exact parity against the existing CPU path; they do not establish broad quality, official GPU parity, long-context performance, or >10 tokens/s.

## Implemented candidates

`goal_runtime_0910.py` composes independently selectable native quantization, scoped expert input reuse, grouped MoE, native HC, and scripted sparse attention. `deepseek_v41_goal_server_0910.py` serves all five with 16 workers. `promote_deepseek_v41_goal_0910.py` promotes this configuration from the completed matched evidence, validates the actual endpoint, and restores the previous implementation on failed promotion. Promotion state must be read from its result and `deepseek-v41-selected.json`, not inferred from this document.

Fixture evidence:

- Native quant: `results/goal_native_quant_0910/fixture-check.json`, 2,757,184 input values, exact FP8/scales/inplace BF16.
- Grouped GEMM: `results/deepseek-v41-native-grouped-goal-0910/kernel-check.json`, 551,768 exact BF16 outputs. Grouped MoE integration checks are in `goal_check_grouped_moe_0910.py`.
- HC: `results/goal_hc_0910/fixture-check.json`, 1,187 cases with exact float32 output bytes, using the same installed MKL exponential routine.
- Sparse attention: `goal_check_sparse_script_0910.py`, exact outputs across 21 cases and the original 64-position chunk ordering.

## Next work

1. Diagnose VNNI's model regression with matched hot and larger-than-LLC streaming component measurements, including native-only time and actual page locations.
2. `goal_numa_0910.py` and `goal_numa_store_0910.py` provide bounded row copies; the completed real-model trial above regressed. More workers alone have not helped.
3. New exact-integer FP4 grouped and experimental FP8 VNNI candidates are under development. Existing FP4 VNNI preserves native values but a crafted cancellation/tie fixture differs by one BF16 ULP, despite exact measured greeting logits. Packed cache eviction is implemented and fixture-verified in `goal_vnni_runtime_0910.py`.
4. `goal_trace_thread_parity_0910.py` can trace the first Torch16/Torch8 decode divergence; it has not run on the real model.
5. DSpark draft weights are ~7.93 GB but are not loaded. A causal chunk verifier and rollback need correctness and perfect-draft throughput tests before loading the draft stages. Simply enabling the draft graph cannot satisfy the goal.
6. A separate native sparse-attention candidate is being developed with MKL matmuls/exp and exact ATen row sums to reduce dispatch and allocation cost.

Additional completed components: `goal_sparse_native_0910.py` (exact one-core fixtures and real-model logits); `goal_chunk_verify2_0910.py` (16 full 40-layer fixture cases, rollback/shared-state exact); `goal_chunk_grouped_moe_0910.py` plus `deepseek_v41_ragged_int16_goal_0910.py` (native variable-token expert grouping, full-width and64-expert fixture parity); `results/deepseek-v41-ragged-int16-exact-goal-0910/kernel-check.json` (682,488 baseline bit comparisons). New combined perfect-verifier benchmark is `benchmark_deepseek_v41_goal_chunk_grouped_0910.py`; it is prepared for launch after the quiet exact16 repeat. Optional DSpark isolated native store/loader is prepared in `goal_dspark_store_0910.py` and `goal_dspark_loader_0910.py`; no draft weights were downloaded and no speculative generation runs yet.

Native cache is temporary, currently bounded at 90 GiB; the full ~510 GB checkpoint is not resident. Context remains 256 combined tokens, text-only and greedy, with max output 128. Longer, varied generation and sustained >10 tokens/s remain required work.
