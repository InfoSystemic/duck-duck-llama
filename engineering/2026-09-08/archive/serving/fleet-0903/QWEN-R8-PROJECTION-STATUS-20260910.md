# Qwen R8 projection work — September 10, 2026

The scoped R8 candidate passes exact component and model-output checks, but does not demonstrate a model speed gain. It remains unselected. Qwen retains its Q6 target and Q8 MTP4 draft; the 40 tok/s and 250 GB/s goals remain open. The selected Flash Q4 service was restored and independently verified healthy after the four-arm trial.

The [existing-tile probe](results/qwen-r8-tiles-probe-0910/result.json) runs tiles 1/2/4/8/8/4/2/1 on the corrected `12c61b...` CPU. It covers 12 native/repacked cases per arm, with 512 experts, 10 rotating routes, and activation batch sizes 1/3/5. Full packed-output hashes agree across all eight arms; native numerical tolerance also passes. Wider tiles consistently help the dense SSM shape, while expert timing changes substantially across the run order. This does not qualify a global tile change or a model speed gain.

The [candidate build](results/qwen-r8-projection-build-0910/result.json) reproduces the original x86 object and corrected CPU library before replacing the isolated x86 object. Candidate CPU SHA-256 is `fc9623ec3390caef64d2e9bbb7fa5899d666986e9b974028dd19ce2f61e47f97`.

The [kernel](qwen-q8-r8-k160-0910.h) and [transform](qwen_r8_projection_transform_0910.py) add two independent opt-ins. `GGML_CPU_Q8_R8_K160_PREP=1` prepares activation sums/scales once for K=160 NR=1, retaining contiguous expert weight traversal and the original accumulation order. `GGML_CPU_Q8_R8_SSM_TILE8=1` scopes tile 8 to K=1536. The optional K160 audit counter should be disabled for timing. The runtime bundle includes the preceding validated fresh-MTP-state common library.

The [component validation](results/qwen-r8-projection-validation-0910/result.json) covers 330 correctness cases across parent/off/prep/SSM/both configurations, one and 15 workers, padded layouts, activation counts 1/3/5, real projection shapes, and fallback shapes. Packed outputs match exactly within equal layout, worker count, and route schedule, and the native numerical tolerance passes. The audit counter proves the K160 path executes; the trace proves the scoped SSM tile selection. K160 dense calls also use the preparation path when the dispatcher invokes NR=1, while the three-token dense fixture uses its existing batched path. The 24 separate component timings contain host drift and are not model performance evidence.

## Four fresh model launches

The [controller](run_qwen_r8_projection_0910.py) passed [12 lifecycle checks](results/qwen-r8-projection-controller-checks-0910.json), then completed four fresh Q6/Q8-MTP4 launches in off/on/on/off order. All arms use the same candidate CPU and corrected fresh-MTP common library, 15 workers per socket, all four NUMA devices, context 4096, and shared dispatch. Only the two scoped feature flags vary. Audit counters and operation profiling are disabled during model timing.

| Arm | Flags | Prose tok/s | Code tok/s | Adjusted GB/s, prose/code | Adjacent idle qualification |
| --- | --- | ---: | ---: | ---: | --- |
| 0 | Off | 20.920 | 26.952 | 115.75 / 123.49 | Both fail |
| 1 | On | 21.669 | 29.060 | 125.69 / 136.01 | Both pass |
| 2 | On | 22.256 | 29.065 | 129.04 / 134.97 | Both pass |
| 3 | Off | 22.341 | 29.150 | 130.28 / 135.99 | Both pass |

All four runs produce the same output hashes and speculative counts: prose emits 512 tokens with 586 drafted and 359 accepted; code emits 318 with 290 drafted and 244 accepted. Cache reuse is zero. Prose reaches its token cap; code finishes its answer. Both simple pre-measurement checks also pass in every arm.

The first control's adjacent idle traffic exceeds the attribution limit. Its aggregate ABBA comparison therefore cannot establish the apparent 1.53% prose / 3.61% code improvement. Against the qualified final control, the candidate mean is 1.70% slower on prose and 0.30% slower on code. The nearest candidate is 0.38% / 0.29% slower. These results support leaving the candidate unselected; they do not establish a general regression or an uncontended maximum.

The [independent audit](results/qwen-r8-projection-audit-0910c.json) reparses all 48 IMC counters in every interval, reconciles retained responses, hashes and counts, verifies fixture and controller source identities, checks the two-flag-only comparison, and confirms all four Qwen processes exited. The frozen controller verifies mapped CPU/base/llama/common libraries and exact Flash restoration. The independent audit verifies the live Flash identity, command, and manager state; it never reads private restoration context. Earlier audit script versions contain audit-only branch/schema mistakes and produced no accepted result.

The [build](build_qwen_r8_projection_0910.py), [component validator](validate_qwen_r8_projection_0910.py), [model result](results/qwen-r8-projection-model-0910/result.json), and their executed inputs remain frozen. Component arithmetic is validated; full-model use is experimental and has not earned promotion.
