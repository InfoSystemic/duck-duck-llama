// The deployed scalar kernel is retained unchanged behind the new flag.
#include "pool-scalar.inc"

// Optional benchmark-only control. Writers change one aligned 32-bit word only
// while all server slots are idle. Mapping it read-only avoids file I/O in decode.
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
static bool glm_pool_wide_enabled() {
    static const bool enabled = [] {
        const char * v = std::getenv("GGML_CPU_GLM_POOL_WIDE");
        return v && std::strcmp(v, "1") == 0;
    }();
    if (!enabled) return false;
    static const uint32_t * control = [] () -> const uint32_t * {
        const char * path = std::getenv("GGML_CPU_GLM_POOL_WIDE_CONTROL_FILE");
        if (!path || !*path) return nullptr;
        const int fd = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
        GGML_ASSERT(fd >= 0);
        struct stat info;
        GGML_ASSERT(fstat(fd, &info) == 0 && S_ISREG(info.st_mode) && info.st_size == sizeof(uint32_t));
        void * mapping = mmap(nullptr, sizeof(uint32_t), PROT_READ, MAP_SHARED, fd, 0);
        close(fd);
        GGML_ASSERT(mapping != MAP_FAILED);
        return (const uint32_t *) mapping;
    }();
    if (!control) return true;
    const uint32_t value = __atomic_load_n(control, __ATOMIC_ACQUIRE);
    GGML_ASSERT(value <= 1);
    return value == 1;
}

static std::atomic<uint64_t> glm_pool_wide_calls{0};
extern "C" uint64_t ggml_cpu_glm_pool_wide_count(void);
extern "C" uint64_t ggml_cpu_glm_pool_wide_count(void) {
    return glm_pool_wide_calls.load(std::memory_order_relaxed);
}

// Original arithmetic for tails and exceptional values. Keep MUL operand order.
static inline void glm_pool_scalar_row(const ggml_fp16_t * const cell[4],
        const ggml_tensor * ape, int64_t d, int64_t ch, bool probs_first, float * dst) {
    float work[4], keys[4], probability[4], product[4];
    for (int m = 0; m < 4; ++m) {
        keys[m] = GGML_CPU_FP16_TO_FP32(cell[m][ch]);
        const float gate = GGML_CPU_FP16_TO_FP32(cell[m][d + ch]);
        const float position = *(const float *) ((const char *) ape->data + ch*ape->nb[0] + m*ape->nb[1]);
        work[m] = gate + position;
    }
    ggml_vec_scale_f32(4, work, 1.0f);
    float max = -INFINITY;
    ggml_vec_max_f32(4, &max, work);
    ggml_float sum = ggml_vec_soft_max_f32(4, probability, work, max);
    sum = 1.0 / sum;
    ggml_vec_scale_f32(4, probability, sum);
    ggml_vec_mul_f32(4, product, probs_first ? probability : keys, probs_first ? keys : probability);
    ggml_vec_sum_f32(4, dst, product);
}

void ggml_compute_forward_glm_pool_fused(
        const ggml_compute_params * params,
        const ggml_tensor * members,
        const ggml_tensor * ape,
        const ggml_tensor * probs,
        const ggml_tensor * mul,
        ggml_tensor * dst) {
    const bool enabled = glm_pool_wide_enabled();
#if defined(__AVX512F__) && defined(__AVX2__) && defined(__F16C__)
    if (!enabled) {
        ggml_compute_forward_glm_pool_scalar(params, members, ape, probs, mul, dst);
        return;
    }
    if (params->ith == 0) {
        glm_pool_fused_calls.fetch_add(1, std::memory_order_relaxed);
        const uint64_t call = glm_pool_wide_calls.fetch_add(1, std::memory_order_relaxed);
        static const bool probe = [] { const char * v = std::getenv("GGML_CPU_GLM_POOL_PROBE"); return v && std::atoi(v) == 1; }();
        if (probe && call < 8) {
            fprintf(stderr, "GLM_POOL_WIDE name=%s d=%lld pools=%lld streams=%lld threads=%d\n",
                members->name, (long long) dst->ne[1], (long long) dst->ne[2], (long long) dst->ne[3], params->nth);
        }
    }
    const ggml_tensor * kg = members->src[0];
    const ggml_tensor * cells = members->src[1];
    const int64_t d = dst->ne[1], pools = dst->ne[2];
    const int64_t blocks_per_pool = (d + 7) / 8;
    const int64_t blocks = blocks_per_pool * pools * dst->ne[3];
    const int64_t begin = blocks * params->ith / params->nth;
    const int64_t end = blocks * (params->ith + 1) / params->nth;
    const bool probs_first = mul->src[0] == probs;
    const __m256 inf = _mm256_set1_ps(INFINITY);
    const __m256 sign = _mm256_set1_ps(-0.0f);
    const __m512d one = _mm512_set1_pd(1.0);
    for (int64_t b = begin; b < end;) {
        const int64_t pool_stream = b / blocks_per_pool;
        const int64_t pool = pool_stream % pools, stream = pool_stream / pools;
        const int64_t pool_end = std::min(end, (pool_stream + 1) * blocks_per_pool);
        const ggml_fp16_t * cell[4];
        for (int m = 0; m < 4; ++m) {
            const int32_t idx = *(const int32_t *) ((const char *) cells->data +
                (4*pool + m)*cells->nb[0] + stream*cells->nb[1]);
            GGML_ASSERT(idx >= 0 && idx < kg->ne[1]);
            cell[m] = (const ggml_fp16_t *) ((const char *) kg->data + idx*kg->nb[1] + stream*kg->nb[2]);
        }
        for (; b < pool_end; ++b) {
            const int64_t ch = (b - pool_stream*blocks_per_pool) * 8;
            float * output = (float *) dst->data + pool_stream*d + ch;
            if (ch + 8 > d) {
                for (int64_t c = ch; c < d; ++c) glm_pool_scalar_row(cell, ape, d, c, probs_first, output + c - ch);
                continue;
            }
            __m256 keys[4], work[4];
            __m256 max = _mm256_set1_ps(-INFINITY);
            bool finite = true;
            for (int m = 0; m < 4; ++m) {
                keys[m] = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (cell[m] + ch)));
                const __m256 gate = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *) (cell[m] + d + ch)));
                const __m256 position = _mm256_loadu_ps((const float *) ((const char *) ape->data + m*ape->nb[1]) + ch);
                work[m] = _mm256_add_ps(gate, position);
                finite &= _mm256_movemask_ps(_mm256_cmp_ps(_mm256_andnot_ps(sign, work[m]), inf, _CMP_LT_OQ)) == 255;
                finite &= _mm256_movemask_ps(_mm256_cmp_ps(_mm256_andnot_ps(sign, keys[m]), inf, _CMP_LT_OQ)) == 255;
                max = _mm256_max_ps(max, work[m]);
            }
            if (!finite) {
                for (int c = 0; c < 8; ++c) glm_pool_scalar_row(cell, ape, d, ch + c, probs_first, output + c);
                continue;
            }
            alignas(64) float shifted[4][8], exponent[4][8];
            for (int m = 0; m < 4; ++m) {
                _mm256_store_ps(shifted[m], _mm256_sub_ps(work[m], max));
                // Use the same scalar libm expf as ggml_vec_soft_max_f32(n=4).
                // expf(+/-0) is exactly 1; no approximate vector exponential.
                for (int c = 0; c < 8; ++c) {
                    exponent[m][c] = shifted[m][c] == 0.0f ? 1.0f : expf(shifted[m][c]);
                }
            }
            __m256 exps[4];
            __m512d sum = _mm512_setzero_pd();
            for (int m = 0; m < 4; ++m) {
                exps[m] = _mm256_load_ps(exponent[m]);
                sum = _mm512_add_pd(sum, _mm512_cvtps_pd(exps[m]));
            }
            const __m256 reciprocal = _mm512_cvtpd_ps(_mm512_div_pd(one, sum));
            __m512d weighted = _mm512_setzero_pd();
            for (int m = 0; m < 4; ++m) {
                const __m256 probability = _mm256_mul_ps(exps[m], reciprocal);
                const __m256 product = probs_first ? _mm256_mul_ps(probability, keys[m]) : _mm256_mul_ps(keys[m], probability);
                weighted = _mm512_add_pd(weighted, _mm512_cvtps_pd(product));
            }
            _mm256_storeu_ps(output, _mm512_cvtpd_ps(weighted));
        }
    }
#else
    GGML_UNUSED(enabled);
    ggml_compute_forward_glm_pool_scalar(params, members, ape, probs, mul, dst);
#endif
}
