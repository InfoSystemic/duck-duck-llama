"""Add an opt-in ordered-K path for the observed Qwen HC Q8 projections."""


def transform(source):
    marker = '#include "ggml-quants.h"\n'
    assert source.count(marker) == 1
    source = source.replace(marker, marker + '''
#include "qwen-q8-hc-ordered-k-0909.h"

static bool qwen_hc_ordered_k_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_CPU_QWEN_HC_ORDERED_K");
        return value && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}
static bool qwen_hc_ordered_k_audit() {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_CPU_QWEN_HC_ORDERED_K_AUDIT");
        return value && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}
static std::atomic<uint64_t> qwen_hc_ordered_k_calls{0};
extern "C" uint64_t ggml_cpu_qwen_hc_ordered_k_count();
extern "C" uint64_t ggml_cpu_qwen_hc_ordered_k_count() {
    return qwen_hc_ordered_k_calls.load(std::memory_order_relaxed);
}
''')
    start = source.index('template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS, ggml_type PARAM_TYPE> class tensor_traits')
    end = source.index('class tensor_traits_router_f16', start)
    body = source[start:end]
    marker = '    bool work_size(int n_threads, const struct ggml_tensor * op, size_t & size) override {'
    assert body.count(marker) == 1
    body = body.replace(marker, '''    static bool hc_ordered_k_eligible(int n_threads, const ggml_tensor * op) {
        if constexpr (!std::is_same_v<BLOC_TYPE, block_q8_0> || INTER_SIZE != 8 || NB_COLS != 8 || PARAM_TYPE != GGML_TYPE_Q8_0) {
            return false;
        }
        if (!qwen_hc_ordered_k_enabled() || op->op != GGML_OP_MUL_MAT) return false;
        const ggml_tensor * a = op->src[0], * b = op->src[1];
        return a->type == GGML_TYPE_Q8_0 && b->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
            a->ne[0] == 10240 && (a->ne[1] == 64 || a->ne[1] == 96) && n_threads > a->ne[1] / 8 &&
            a->ne[2] == 1 && a->ne[3] == 1 && b->ne[0] == a->ne[0] &&
            b->ne[1] >= 1 && b->ne[1] <= 8 && b->ne[2] == 1 && b->ne[3] == 1 &&
            op->ne[0] == a->ne[1] && op->ne[1] == b->ne[1] && op->ne[2] == 1 && op->ne[3] == 1 &&
            b->nb[0] == sizeof(float) && op->nb[0] == sizeof(float) && op->nb[1] % sizeof(float) == 0;
    }

''' + marker)
    old = '''                    size = ggml_row_size(PARAM_TYPE, ggml_nelements(op->src[1]));
                    return true;'''
    assert body.count(old) == 1
    body = body.replace(old, '''                    size = ggml_row_size(PARAM_TYPE, ggml_nelements(op->src[1]));
                    if (hc_ordered_k_eligible(n_threads, op)) {
                        size += alignof(qwen_q8_hc_partial) - 1 + sizeof(qwen_q8_hc_partial) *
                            (op->src[0]->ne[0] / QK8_0) * (op->src[0]->ne[1] / 8) * op->src[1]->ne[1];
                    }
                    return true;''')
    mat_start = body.index('    void forward_mul_mat(ggml_compute_params * params, ggml_tensor * op) {')
    mat_end = body.index('    void forward_mul_mat_id(', mat_start)
    mat = body[mat_start:mat_end]
    marker = '        // INFO: Quantization is done in planes to avoid extra complexity in chunking.'
    assert mat.count(marker) == 1
    mat = mat.replace(marker, '''        if (!params->use_ref && hc_ordered_k_eligible(nth, op)) {
            const int blocks = ne00 / QK8_0;
            const int first = ith * blocks / nth, last = (ith + 1) * blocks / nth;
            const uintptr_t begin = GGML_PAD((uintptr_t) wdata + nbw2, alignof(qwen_q8_hc_partial));
            auto * partial = (qwen_q8_hc_partial *) begin;
            GGML_ASSERT(begin + sizeof(*partial) * blocks * (ne01 / 8) * ne11 <= (uintptr_t) wdata + params->wsize);
            for (int row = 0; row < ne11; ++row) {
                if (last > first) from_float(
                    (const float *) ((const char *) src1->data + row * nb11 + first * QK8_0 * sizeof(float)),
                    wdata + row * nbw1 + first * sizeof(block_q8_0), (last - first) * QK8_0);
            }
            qwen_q8_hc_prepare(ne00, ne01, ne11, (const block_q8_0x8 *) src0->data,
                               (const block_q8_0 *) wdata, first, last, partial);
            ggml_barrier(params->threadpool);
            qwen_q8_hc_finish(ne00, ne01, ne11, partial, (float *) dst->data, nb1 / sizeof(float), ith, nth);
            if (ith == 0) {
                static std::once_flag first_call;
                std::call_once(first_call, [=] {
                    GGML_LOG_INFO("QWEN_HC_ORDERED_K k=%lld nc=%lld nr=%lld nth=%d\\n",
                                  (long long) ne00, (long long) ne01, (long long) ne11, nth);
                });
                if (qwen_hc_ordered_k_audit()) qwen_hc_ordered_k_calls.fetch_add(1, std::memory_order_relaxed);
            }
            return;
        }

''' + marker)
    body = body[:mat_start] + mat + body[mat_end:]
    return source[:start] + body + source[end:]


def graph_fixture(source):
    source = source.replace('static bool check(', 'static FILE * output_dump;\nstatic int test_threads = 15;\n\nstatic bool check(', 1)
    marker = '        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) std::abort();'
    assert source.count(marker) == 1
    source = source.replace(marker, '''        using count_fn = uint64_t (*)();
        const auto counter = reinterpret_cast<count_fn>(dlsym(RTLD_DEFAULT, "ggml_cpu_qwen_hc_ordered_k_count"));
        const char * flag = std::getenv("GGML_CPU_QWEN_HC_ORDERED_K");
        const bool selected = counter && flag && !std::strcmp(flag,"1") && k == 10240 && test_threads > nc/8;
        const uint64_t before_count = counter ? counter() : 0;
''' + marker + '''
        if (counter && counter()-before_count != uint64_t(selected)) std::abort();
        if (std::fwrite(y->data, 1, ggml_nbytes(y), output_dump) != ggml_nbytes(y)) std::abort();''')
    source = source.replace('int main() {', '''int main(int argc, char ** argv) {
    if (argc != 3) return 2;
    test_threads = std::atoi(argv[2]);
    if (test_threads != 1 && test_threads != 4 && test_threads != 15) return 2;
    output_dump = std::fopen(argv[1], "wb");
    if (!output_dump) return 2;
    const auto audit_count = reinterpret_cast<uint64_t (*)()>(dlsym(RTLD_DEFAULT, "ggml_cpu_qwen_hc_ordered_k_count"));
    std::printf("HC_COUNTER_PRESENT %d\\n", audit_count != nullptr);''')
    source = source.replace('ggml_threadpool_params_default(15)', 'ggml_threadpool_params_default(test_threads)')
    source = source.replace('i < CPU_SETSIZE && CPU_ISSET(i, &available);', 'i < test_threads && CPU_ISSET(i, &available);')
    source = source.replace('ggml_backend_cpu_set_n_threads(backend, 15);', 'ggml_backend_cpu_set_n_threads(backend, test_threads);')
    marker = '    return failures ? 1 : 0;'
    assert source.count(marker) == 1
    source = source.replace(marker, '''    if (std::fclose(output_dump)) ++failures;
    std::printf("HC_TOTAL_CALLS %llu\\n", (unsigned long long)(audit_count ? audit_count() : 0));
''' + marker)
    return source
