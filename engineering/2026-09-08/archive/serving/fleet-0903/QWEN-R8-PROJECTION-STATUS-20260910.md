# Qwen R8 projection work — September 10, 2026

The existing R8 tile probe completed before DeepSeek-V4.1-Flash was added to the campaign. The new scoped CPU candidate compiled successfully but has not passed functional fixtures or model measurements. It remains unselected. Qwen retains its Q6 target and Q8 draft; the 40 tok/s and 250 GB/s goals remain open.

The [existing-tile probe](results/qwen-r8-tiles-probe-0910/result.json) runs tiles 1/2/4/8/8/4/2/1 on the corrected `12c61b...` CPU. It covers 12 native/repacked cases per arm, with 512 experts, 10 rotating routes, and activation batch sizes 1/3/5. Full packed-output hashes agree across all eight arms; native numerical tolerance also passes. Wider tiles consistently help the dense SSM shape, while expert timing changes substantially across the run order. This does not qualify a global tile change or a model speed gain.

The [candidate build](results/qwen-r8-projection-build-0910/result.json) reproduces the original x86 object and corrected CPU library before replacing the isolated x86 object. Candidate CPU SHA-256 is `fc9623ec3390caef64d2e9bbb7fa5899d666986e9b974028dd19ce2f61e47f97`.

The [kernel](qwen-q8-r8-k160-0910.h) and [transform](qwen_r8_projection_transform_0910.py) add two independent opt-ins. `GGML_CPU_Q8_R8_K160_PREP=1` prepares activation sums/scales once for K=160 NR=1, retaining contiguous expert weight traversal and the original accumulation order. `GGML_CPU_Q8_R8_SSM_TILE8=1` scopes tile 8 to K=1536. The optional K160 audit counter should be disabled for timing. The runtime bundle includes the preceding validated fresh-MTP-state common library.

Next, check both flags separately and together against the parent across exact output, real tensor shapes, thread counts, fallback shapes and layouts. If those pass, use matched fresh-state model comparisons with IMC counters before any promotion. The [build script](build_qwen_r8_projection_0910.py) and executed artifacts remain frozen. The selected Flash Q4 service was preserved.
