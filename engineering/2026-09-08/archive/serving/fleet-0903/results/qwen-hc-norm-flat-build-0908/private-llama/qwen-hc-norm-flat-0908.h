#pragma once
#include "ggml.h"

static ggml_tensor * qwen_hc_norm_flat(ggml_context * ctx, ggml_cgraph * graph,
        ggml_tensor * x, ggml_tensor * weight, float eps) {
    const int64_t width = x->ne[0], streams = x->ne[1], tokens = x->ne[2];
    GGML_ASSERT(x->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_F32 && x->ne[3] == 1);
    GGML_ASSERT(weight->ne[0] == width * streams && ggml_nelements(weight) == width * streams);
    if (!ggml_is_contiguous(x) || !ggml_is_contiguous(weight)) {
        auto normalized = ggml_rms_norm(ctx, x, eps);
        return ggml_mul(ctx, ggml_reshape_2d(ctx, normalized, width * streams, tokens), weight);
    }
    auto row_weights = ggml_reshape_2d(ctx, weight, width, streams);
    // Place the weight view before the adjacent RMS_NORM and MUL nodes.
    ggml_build_forward_expand(graph, row_weights);
    auto rows = ggml_reshape_2d(ctx, x, width, streams * tokens);
    auto normalized = ggml_rms_norm(ctx, rows, eps);
    return ggml_reshape_2d(ctx, ggml_mul(ctx, normalized, row_weights), width * streams, tokens);
}
