# Quantization, layout, and numerical correctness

A model's quantization name is not a complete execution plan. Tensor types, block widths, row shapes, repack eligibility, and the operator that consumes the tensor determine the actual CPU path.

## Engineering

The archive contains compact IQ kernels, Q4/Q5/Q6/Q8 layouts, x86 vector paths, fused MoE operations, and direct/graph/model checks. Experiments distinguish a dense matrix multiply from routed expert multiplication; an improvement in one does not transfer automatically to the other.

An earlier microbenchmark interpretation compared unequal work across dense and MoE operators. The [historical overview](../history/legacy-overview-20260912.md) preserves the correction. Within-operator comparisons remain useful; the withdrawn cross-operator inference does not support a performance claim.

## Worked example: correcting Qwen gather

A parallel FP32 copy branch had been placed in a quantized gather path. The correction puts it in the intended FP32/I32 gather function, preserving quantized dequantization and enabling the intended wide FP32 copy.

The parent reproduces 24 wide-quantized failures. Each corrected direct arm passes 114 cases and checks 46,262,752 values; four graph arms pass another 184 cases in total. The model comparison then runs parent/corrected/corrected/parent with identical outputs and draft counts. [Detailed validation and comparison](../../engineering/2026-09-08/archive/serving/fleet-0903/BANDWIDTH250-20260909.md).

| Workload | Corrected decode tok/s, two runs | Mean gain over the two controls |
| --- | ---: | ---: |
| Prose | 24.3505 / 24.0436 | +3.94% |
| Code | 31.6158 / 30.8030 | +2.50% |

The [portable correction patch](../../engineering/2026-09-08/patches/qwen-get-rows-columns-correction.patch) and [source reconstruction metadata](../../engineering/2026-09-08/correctness-overlays.json) preserve the change. The comparison did not promote a runtime, and the result is not a general quality certification.

## Worked example: Full split granularity

Unequal splits exposed an independent failure in GLM's DSA/shared-expert tensor rules. Several reduction dimensions were divided at granularity 1 even though the source quantization requires 32-element blocks and the selected requantization path needs 256-element alignment.

The correction uses the least common multiple of the source block size and 256 for the affected patterns. It checks all 1,809 actual Full tensors across 32 schedules: 57,888 metadata cases. Balanced splits remain unchanged; the old unequal schedule reproduces 234 invalid splits. [Source, metadata, and validation](../../engineering/2026-09-12/archive/serving/fleet-0912/glmfull-split/README.md).

The candidate library also builds from the existing object recipe, with a byte-identical baseline relink and verified loader selection. Full-model unequal-split execution is still pending. The distinction matters: valid tensor metadata is necessary, but graph propagation and model outputs must also be checked.
