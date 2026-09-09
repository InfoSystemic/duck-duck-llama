#pragma once

#include "ggml.h"
#include <algorithm>
#include <atomic>
#include <cstdint>

// Preserve the graph's rounded products, left-to-right additions, and final scale.
__attribute__((optimize("fp-contract=off")))
static void qwen_hc_mix_f32(ggml_tensor * dst, int ith, int nth, void * userdata) {
    const ggml_tensor * values = dst->src[0];
    const ggml_tensor * gates = dst->src[1];
    GGML_ASSERT(dst->type == GGML_TYPE_F32 && values->type == GGML_TYPE_F32 && gates->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_are_same_shape(values, gates));
    const int64_t width = dst->ne[0], tokens = dst->ne[1];
    GGML_ASSERT(width > 0 && values->ne[0] % width == 0 && values->ne[1] == tokens);
    GGML_ASSERT(dst->ne[2] == 1 && dst->ne[3] == 1 && values->ne[2] == 1 && values->ne[3] == 1);
    GGML_ASSERT(dst->nb[0] == sizeof(float) && values->nb[0] == sizeof(float) && gates->nb[0] == sizeof(float));
    const int64_t streams = values->ne[0] / width;
    GGML_ASSERT(streams > 0 && nth > 0);
    if (userdata && ith == 0)
        static_cast<std::atomic<uint64_t> *>(userdata)->fetch_add(1, std::memory_order_relaxed);
    const float scale = 1.0f / static_cast<float>(streams);
    constexpr int64_t tile = 256;
    const int64_t tiles = (width + tile - 1) / tile;
    for (int64_t item = ith; item < tokens * tiles; item += nth) {
        const int64_t token = item / tiles;
        const int64_t begin = (item % tiles) * tile;
        const int64_t end = std::min(begin + tile, width);
        const float * x = reinterpret_cast<const float *>(reinterpret_cast<const char *>(values->data) + token * values->nb[1]);
        const float * g = reinterpret_cast<const float *>(reinterpret_cast<const char *>(gates->data) + token * gates->nb[1]);
        float * out = reinterpret_cast<float *>(reinterpret_cast<char *>(dst->data) + token * dst->nb[1]);
        for (int64_t j = begin; j < end; ++j) out[j] = x[j] * g[j];
        for (int64_t stream = 1; stream < streams; ++stream) {
            const float * xs = x + stream * width;
            const float * gs = g + stream * width;
            for (int64_t j = begin; j < end; ++j) {
                const float product = xs[j] * gs[j];
                out[j] = out[j] + product;
            }
        }
        for (int64_t j = begin; j < end; ++j) out[j] *= scale;
    }
}
