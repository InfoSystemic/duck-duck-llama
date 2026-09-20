// GET_ROWS -> key/gate transposes -> APE ADD -> existing softmax-pool arithmetic.
// Only the strict matcher in pool-fusion.inc may call this kernel.
static std::atomic<uint64_t> glm_pool_fused_calls{0};

extern "C" uint64_t ggml_cpu_glm_pool_fused_count(void);
extern "C" uint64_t ggml_cpu_glm_pool_fused_count(void) {
    return glm_pool_fused_calls.load(std::memory_order_relaxed);
}

static void ggml_compute_forward_glm_pool_scalar(
        const ggml_compute_params * params,
        const ggml_tensor * members,
        const ggml_tensor * ape,
        const ggml_tensor * probs,
        const ggml_tensor * mul,
        ggml_tensor * dst) {
    if (params->ith == 0) {
        const uint64_t call = glm_pool_fused_calls.fetch_add(1, std::memory_order_relaxed);
        static const bool probe = [] { const char * v = std::getenv("GGML_CPU_GLM_POOL_PROBE"); return v && std::atoi(v) == 1; }();
        if (probe && call < 8) {
            fprintf(stderr, "GLM_POOL_FUSED name=%s d=%lld pools=%lld streams=%lld threads=%d\n",
                members->name, (long long) dst->ne[1], (long long) dst->ne[2], (long long) dst->ne[3], params->nth);
        }
    }
    const ggml_tensor * kg = members->src[0];
    const ggml_tensor * cells = members->src[1];
    const int64_t d = dst->ne[1];
    const int64_t pools = dst->ne[2];
    const int64_t rows = d * pools * dst->ne[3];
    const int64_t begin = rows * params->ith / params->nth;
    const int64_t end = rows * (params->ith + 1) / params->nth;
    const bool probs_first = mul->src[0] == probs;
    float scale;
    memcpy(&scale, (const float *) probs->op_params, sizeof(float));
    for (int64_t row = begin; row < end; ++row) {
        const int64_t channel = row % d;
        const int64_t pool = (row / d) % pools;
        const int64_t stream = row / (d * pools);
        float work[4], keys[4], probability[4], product[4];
        for (int m = 0; m < 4; ++m) {
            const int32_t cell = *(const int32_t *) ((const char *) cells->data +
                (4*pool + m)*cells->nb[0] + stream*cells->nb[1]);
            GGML_ASSERT(cell >= 0 && cell < kg->ne[1]);
            const ggml_fp16_t * cell_data = (const ggml_fp16_t *) ((const char *) kg->data +
                cell*kg->nb[1] + stream*kg->nb[2]);
            keys[m] = GGML_CPU_FP16_TO_FP32(cell_data[channel]);
            const float gate = GGML_CPU_FP16_TO_FP32(cell_data[d + channel]);
            const float position = *(const float *) ((const char *) ape->data +
                channel*ape->nb[0] + m*ape->nb[1]);
            work[m] = gate + position;
        }
        // Keep every arithmetic stage identical to the deployed pool kernel,
        // including float products and ggml_float accumulation/reciprocal.
        ggml_vec_scale_f32(4, work, scale);
        float max = -INFINITY;
        ggml_vec_max_f32(4, &max, work);
        ggml_float sum = ggml_vec_soft_max_f32(4, probability, work, max);
        sum = 1.0 / sum;
        ggml_vec_scale_f32(4, probability, sum);
        ggml_vec_mul_f32(4, product, probs_first ? probability : keys, probs_first ? keys : probability);
        ggml_vec_sum_f32(4, (float *) dst->data + row, product);
    }
}
