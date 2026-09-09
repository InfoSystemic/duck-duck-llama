#pragma once

#include "ggml.h"
#include <algorithm>
#include <atomic>
#include <cstdint>

// Keep the rounded multiply followed by add used by the original graph.
__attribute__((optimize("fp-contract=off")))
static void qwen_hc_combine_f32(ggml_tensor * dst, const ggml_tensor * residual,
        const ggml_tensor * block, const ggml_tensor * weight,
        int ith, int nth, void * userdata) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32 && residual->type == GGML_TYPE_F32 &&
                block->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_are_same_shape(dst, residual));
    const int64_t width = residual->ne[0], streams = residual->ne[1], tokens = residual->ne[2];
    GGML_ASSERT(residual->ne[3] == 1 && block->ne[0] == width && block->ne[1] == 1 &&
                block->ne[2] == tokens && block->ne[3] == 1);
    GGML_ASSERT(weight->ne[0] == 1 && weight->ne[1] == streams &&
                weight->ne[2] == tokens && weight->ne[3] == 1);
    GGML_ASSERT(dst->nb[0] == sizeof(float) && residual->nb[0] == sizeof(float) &&
                block->nb[0] == sizeof(float) && weight->nb[0] == sizeof(float));
    if (userdata && ith == 0)
        static_cast<std::atomic<uint64_t> *>(userdata)->fetch_add(1, std::memory_order_relaxed);
    constexpr int64_t tile = 256;
    const int64_t tiles = (width + tile - 1) / tile;
    for (int64_t item = ith; item < tokens * streams * tiles; item += nth) {
        const int64_t token = item / (streams * tiles);
        const int64_t stream = (item / tiles) % streams;
        const int64_t begin = (item % tiles) * tile;
        const int64_t end = std::min(begin + tile, width);
        const float * r = reinterpret_cast<const float *>(reinterpret_cast<const char *>(residual->data)
                            + token * residual->nb[2] + stream * residual->nb[1]);
        const float * b = reinterpret_cast<const float *>(reinterpret_cast<const char *>(block->data)
                            + token * block->nb[2]);
        const float w = *reinterpret_cast<const float *>(reinterpret_cast<const char *>(weight->data)
                            + token * weight->nb[2] + stream * weight->nb[1]);
        float * out = reinterpret_cast<float *>(reinterpret_cast<char *>(dst->data)
                            + token * dst->nb[2] + stream * dst->nb[1]);
        for (int64_t j = begin; j < end; ++j) {
            const float product = b[j] * w;
            out[j] = r[j] + product;
        }
    }
}
