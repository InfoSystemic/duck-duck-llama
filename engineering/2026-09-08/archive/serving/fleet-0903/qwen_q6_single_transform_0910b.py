"""Dispatch the proven NR=1 Q6 helper without changing expert scheduling."""


def transform(source):
    marker = 'void ggml_gemv_q6_K_x16_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {'
    assert source.count(marker) == 1
    support = r'''
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
#include "qwen-q6-packed-single-0908.h"

static std::atomic<uint64_t> qwen_q6_single_calls{0};
extern "C" {
GGML_BACKEND_API uint64_t ggml_cpu_qwen_q6_single_count();
}
extern "C" uint64_t ggml_cpu_qwen_q6_single_count() {
    return qwen_q6_single_calls.load(std::memory_order_relaxed);
}

static bool qwen_q6_single_enabled() {
    static const bool enabled = [] {
        const char * value = getenv("GGML_CPU_QWEN_Q6_PACKED_SINGLE");
        const bool active = value && atoi(value) == 1;
        if (active) GGML_LOG_INFO("QWEN_Q6_PACKED_SINGLE nr=1 storage=unchanged\n");
        return active;
    }();
    return enabled;
}

static void qwen_q6_single_audit() {
    static const bool enabled = [] {
        const char * value = getenv("GGML_CPU_QWEN_Q6_PACKED_SINGLE_AUDIT");
        return value && atoi(value) == 1;
    }();
    if (enabled) qwen_q6_single_calls.fetch_add(1, std::memory_order_relaxed);
}
#endif

'''
    source = source.replace(marker, support + marker)
    anchor = marker + '\n#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)\n    GGML_ASSERT(nr == 1 && n % QK_K == 0 && nc % 16 == 0);'
    assert source.count(anchor) == 1
    return source.replace(anchor, anchor + r'''
    if (qwen_q6_single_enabled()) {
        const block_q8_K * activation[] = { (const block_q8_K *) vy };
        float * output[] = { s };
        qwen_q6_packed_batch_impl<1>(n, nc, (const block_q6_K_x16 *) vx, activation, output);
        qwen_q6_single_audit();
        return;
    }''')
