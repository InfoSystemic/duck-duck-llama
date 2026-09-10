// SPDX-License-Identifier: MIT
// Native FP8 projection and BF16 residual gate; standalone CPU bring-up component.
#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

// Quantize finite BF16 inputs, [tokens][width], in groups of 32, using the
// publisher's max(amax, 1e-4)/448 rounded UP to a power of two. FP8 conversion is
// round-to-nearest-even with saturation at +/-448. Scale output is native E8M0.
// All spans are element counts. Invalid inputs leave both outputs unchanged.
// Nonempty buffers must be valid and disjoint. Empty calls are no-ops.
int deepseek_v41_act_quant(const uint16_t * input, int64_t tokens, int64_t width,
    uint8_t * output, uint64_t output_elements, uint8_t * scales, uint64_t scale_elements);

typedef struct {
    int64_t input_dim;             // Multiple of 32, at most 6144; released model: 6144.
    int64_t dim;                   // Multiple of 32, at most 5120; released model: 5120.
    int64_t hc_mult;               // Released model: 4; this component supports 1..4.
    const uint8_t * weight;        // Native E4M3, [(hc_mult+1)*dim][input_dim].
    uint64_t weight_bytes;
    const uint8_t * scale;         // E8M0, [ceil((hc_mult+1)*dim/32)][input_dim/32].
    uint64_t scale_bytes;
    const uint16_t * q_weight;     // BF16 [hc_mult][dim]. Copied as FP32 q*k.
    const uint16_t * k_weight;
    uint64_t norm_elements;
    float norm_eps;                // Released model: 1e-20.
} deepseek_v41_project_config;

// Copies metadata and q*k products; borrows immutable weight/scale buffers for
// the handle's lifetime. Creation checks every weight/scale code for finiteness.
// Scratch buffers are allocated here; apply performs no explicit heap allocation.
// OpenMP may initialize its own runtime/worker storage on the first call.
// The scalar and AVX-512 modes use the same specified FP32 reduction tree.
void * deepseek_v41_project_create(const deepseek_v41_project_config * config,
    int64_t max_chunk, int vectorize, int workers);
void deepseek_v41_project_destroy(void * handle);

// lookup BF16: [tokens][input_dim], selecting ONE layer from native lookup output.
// hidden/output BF16: [tokens][hc_mult][dim]. mask optional: 0 passes through.
// Optional diagnostics: projected BF16 [tokens][(hc_mult+1)*dim], raw FP32
// residual [tokens][hc_mult][dim], and FP32 gates [tokens][hc_mult].
// Sizes must be zero for omitted diagnostics. Serialize calls on each handle.
// Buffers must be valid/disjoint, except output may equal hidden (full in-place).
// On error all caller outputs remain unchanged. -1: invalid input; -2: unavailable
// SIMD; -3: arithmetic became nonfinite. This does not mutate token history.
// FP32 reductions need not be bit-identical to a GPU tensor-core implementation.
int deepseek_v41_project_apply(void * handle, const uint16_t * lookup,
    const uint16_t * hidden, const uint8_t * mask, int64_t tokens,
    uint16_t * output, uint64_t output_elements,
    uint16_t * projected, uint64_t projected_elements,
    float * raw_output, uint64_t raw_elements, float * gates, uint64_t gate_elements);

#ifdef __cplusplus
}
#endif
