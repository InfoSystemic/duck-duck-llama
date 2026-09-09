#pragma once

static inline void qwen_quantize_blocks(const ggml_compute_params * params, const ggml_tensor * src,
        ggml_type type, ggml_from_float_t quantize, char * dst, size_t row_bytes) {
    GGML_ASSERT(type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q8_K);
    GGML_ASSERT(src->type == GGML_TYPE_F32 && src->nb[0] == sizeof(float));
    const int64_t block = ggml_blck_size(type);
    const size_t bytes = ggml_type_size(type);
    GGML_ASSERT(src->ne[0] > 0 && src->ne[0] % block == 0);
    const int64_t blocks = src->ne[0] / block;
    const int64_t rows = src->ne[1] * src->ne[2] * src->ne[3];
    GGML_ASSERT(row_bytes >= size_t(blocks) * bytes);
    const int64_t first = rows * blocks * params->ith / params->nth;
    const int64_t last = rows * blocks * (params->ith + 1) / params->nth;
    for (int64_t item = first; item < last;) {
        const int64_t row = item / blocks;
        const int64_t begin = item % blocks;
        const int64_t count = std::min(blocks - begin, last - item);
        const int64_t i1 = row % src->ne[1];
        const int64_t i2 = (row / src->ne[1]) % src->ne[2];
        const int64_t i3 = row / (src->ne[1] * src->ne[2]);
        const char * input = (const char *) src->data + i1 * src->nb[1] + i2 * src->nb[2] + i3 * src->nb[3];
        quantize((const float *) input + begin * block, dst + row * row_bytes + begin * bytes, count * block);
        item += count;
    }
}
