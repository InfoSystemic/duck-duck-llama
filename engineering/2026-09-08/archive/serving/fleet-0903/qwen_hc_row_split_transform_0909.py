"""Use token-row work for HC verification and keep ordered-K only for 64-row raw decode."""
from qwen_hc_ordered_k_transform_0909 import transform as ordered_transform
from qwen_hc_ordered_k_transform_0909 import graph_fixture as ordered_fixture


def transform(source):
    source = ordered_transform(source)
    old = '(a->ne[1] == 64 || a->ne[1] == 96) && n_threads > a->ne[1] / 8'
    assert source.count(old) == 1
    source = source.replace(old, 'a->ne[1] == 64 && n_threads > a->ne[1] / 8')
    old = 'b->ne[1] >= 1 && b->ne[1] <= 8 && b->ne[2] == 1 && b->ne[3] == 1'
    assert source.count(old) == 1
    source = source.replace(old, 'b->ne[1] == 1 && b->ne[2] == 1 && b->ne[3] == 1')
    marker = '#include "qwen-q8-hc-ordered-k-0909.h"\n'
    assert source.count(marker) == 1
    source = source.replace(marker, marker + '''
static bool qwen_hc_row_split_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_CPU_QWEN_HC_ROW_SPLIT");
        return value && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}
static std::atomic<uint64_t> qwen_hc_row_split_calls{0};
extern "C" uint64_t ggml_cpu_qwen_hc_row_split_count();
extern "C" uint64_t ggml_cpu_qwen_hc_row_split_count() {
    return qwen_hc_row_split_calls.load(std::memory_order_relaxed);
}
''')
    marker = '    static bool hc_ordered_k_eligible(int n_threads, const ggml_tensor * op) {'
    assert source.count(marker) == 1
    source = source.replace(marker, '''    static bool hc_row_split_eligible(int n_threads, const ggml_tensor * op) {
        if constexpr (!std::is_same_v<BLOC_TYPE, block_q8_0> || INTER_SIZE != 8 || NB_COLS != 8 || PARAM_TYPE != GGML_TYPE_Q8_0) {
            return false;
        }
        if (!qwen_hc_row_split_enabled() || op->op != GGML_OP_MUL_MAT) return false;
        const ggml_tensor * a = op->src[0], * b = op->src[1];
        return a->type == GGML_TYPE_Q8_0 && b->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
            a->ne[0] == 10240 && (a->ne[1] == 64 || a->ne[1] == 96) && n_threads > a->ne[1] / 8 &&
            a->ne[2] == 1 && a->ne[3] == 1 && b->ne[0] == a->ne[0] &&
            b->ne[1] >= 2 && b->ne[1] <= 8 && b->ne[2] == 1 && b->ne[3] == 1 &&
            op->ne[0] == a->ne[1] && op->ne[1] == b->ne[1] && op->ne[2] == 1 && op->ne[3] == 1 &&
            b->nb[0] == sizeof(float) && op->nb[0] == sizeof(float) && op->nb[1] % sizeof(float) == 0;
    }

''' + marker)
    marker = '        if (!params->use_ref && hc_ordered_k_eligible(nth, op)) {'
    assert source.count(marker) == 1
    source = source.replace(marker, '''        if (!params->use_ref && hc_row_split_eligible(nth, op)) {
            const int blocks = ne00 / QK8_0;
            const int first = ith * blocks / nth, last = (ith + 1) * blocks / nth;
            for (int row = 0; row < ne11; ++row) {
                if (last > first) from_float(
                    (const float *) ((const char *) src1->data + row * nb11 + first * QK8_0 * sizeof(float)),
                    wdata + row * nbw1 + first * sizeof(block_q8_0), (last - first) * QK8_0);
            }
            ggml_barrier(params->threadpool);
            const int groups = ne01 / 8;
            for (int task = ith; task < groups * ne11; task += nth) {
                const int row = task / groups, group = task % groups;
                forward_mul_mat_one_chunk(params, op, group * 8, (group + 1) * 8, row, row + 1);
            }
            if (ith == 0) {
                static std::once_flag first_call;
                std::call_once(first_call, [=] {
                    GGML_LOG_INFO("QWEN_HC_ROW_SPLIT k=%lld nc=%lld nr=%lld nth=%d\\n",
                                  (long long) ne00, (long long) ne01, (long long) ne11, nth);
                });
                if (qwen_hc_ordered_k_audit()) qwen_hc_row_split_calls.fetch_add(1, std::memory_order_relaxed);
            }
            return;
        }

''' + marker)
    return source


def graph_fixture(source):
    source = ordered_fixture(source)
    marker = '        const char * flag = std::getenv("GGML_CPU_QWEN_HC_ORDERED_K");'
    assert source.count(marker) == 1
    source = source.replace(marker, '''        const auto row_counter = reinterpret_cast<count_fn>(dlsym(RTLD_DEFAULT, "ggml_cpu_qwen_hc_row_split_count"));
        const char * row_flag = std::getenv("GGML_CPU_QWEN_HC_ROW_SPLIT");
''' + marker)
    old = '        const bool selected = counter && flag && !std::strcmp(flag,"1") && k == 10240 && test_threads > nc/8;'
    assert source.count(old) == 1
    source = source.replace(old, '''        const bool selected = k == 10240 && test_threads > nc/8 &&
            ((counter && flag && !std::strcmp(flag,"1") && nc == 64 && nr == 1) ||
             (row_counter && row_flag && !std::strcmp(row_flag,"1") && nr >= 2));''')
    source = source.replace('const uint64_t before_count = counter ? counter() : 0;',
                            'const uint64_t before_count = (counter ? counter() : 0) + (row_counter ? row_counter() : 0);')
    source = source.replace('if (counter && counter()-before_count != uint64_t(selected)) std::abort();',
                            'if (counter && counter()+row_counter()-before_count != uint64_t(selected)) std::abort();')
    old = '    std::printf("HC_COUNTER_PRESENT %d\\n", audit_count != nullptr);'
    assert source.count(old) == 1
    source = source.replace(old, '''    const auto row_audit_count = reinterpret_cast<uint64_t (*)()>(dlsym(RTLD_DEFAULT, "ggml_cpu_qwen_hc_row_split_count"));
    if ((audit_count != nullptr) != (row_audit_count != nullptr)) std::abort();
    std::printf("HC_COUNTER_PRESENT %d\\n", audit_count != nullptr && row_audit_count != nullptr);''')
    old = '(unsigned long long)(audit_count ? audit_count() : 0)'
    assert source.count(old) == 1
    source = source.replace(old, '(unsigned long long)((audit_count ? audit_count() : 0) + (row_audit_count ? row_audit_count() : 0))')
    return source
