"""Add an opt-in, lossless expanded Q6 layout to the selected CPU source."""


def replace_once(source, old, new):
    assert source.count(old) == 1, old[:100]
    return source.replace(old, new)


def transform(source):
    source = replace_once(source, '#include "repack.h"',
                          '#include "repack.h"\n#include "qwen-q6-lossless-bytes-0908.h"')
    marker = 'static void ggml_x16_pack_q8_0('
    source = replace_once(source, marker, '''static std::atomic<uint64_t> qwen_q6_bytes_counts[4]{};

extern "C" {
GGML_BACKEND_API uint64_t ggml_cpu_q6_bytes_count(int kind);
}

extern "C" uint64_t ggml_cpu_q6_bytes_count(int kind) {
    return kind >= 0 && kind < 4 ? qwen_q6_bytes_counts[kind].load(std::memory_order_relaxed) : 0;
}

static bool qwen_q6_bytes_audit_enabled() {
    static const bool enabled = [] {
        const char * value = getenv("GGML_CPU_X16_Q6_BYTES_AUDIT");
        return value && atoi(value) == 1;
    }();
    return enabled;
}

static void ggml_x16_pack_q6_K_bytes(const block_q6_K * in, int nb, qwen_q6_bytes_x16 * out) {
    std::vector<block_q6_K_x16> compact(nb);
    ggml_x16_pack_q6_K(in, nb, compact.data());
    qwen_q6_expand_x16(compact.data(), out, nb);
    if (qwen_q6_bytes_audit_enabled()) {
        qwen_q6_bytes_counts[0].fetch_add(1, std::memory_order_relaxed);
        qwen_q6_bytes_counts[2].fetch_add(size_t(nb) * sizeof(qwen_q6_bytes_x16), std::memory_order_relaxed);
    }
}

static void ggml_gemv_q6_K_x16_bytes_q8_K(int n, float * s, size_t bs,
        const void * vx, const void * vy, int nr, int nc) {
    GGML_ASSERT(nr == 1 && n > 0 && n % QK_K == 0 && nc > 0 && nc % 16 == 0);
    GGML_UNUSED(bs);
    qwen_gemv_q6_bytes_x16_q8_K(n, s, static_cast<const qwen_q6_bytes_x16 *>(vx),
                               static_cast<const block_q8_K *>(vy), nc);
    if (qwen_q6_bytes_audit_enabled()) {
        qwen_q6_bytes_counts[1].fetch_add(1, std::memory_order_relaxed);
        qwen_q6_bytes_counts[3].fetch_add(nc, std::memory_order_relaxed);
    }
}

''' + marker)
    old = '        case GGML_TYPE_Q6_K: ggml_x16_pack_q6_K((const block_q6_K *) rows, nb, (block_q6_K_x16 *) dst); break;'
    source = replace_once(source, old, '''        case GGML_TYPE_Q6_K:
            if (sp.x16_bytes == sizeof(qwen_q6_bytes_x16)) {
                ggml_x16_pack_q6_K_bytes((const block_q6_K *) rows, nb, (qwen_q6_bytes_x16 *) dst);
            } else {
                ggml_x16_pack_q6_K((const block_q6_K *) rows, nb, (block_q6_K_x16 *) dst);
            }
            break;''')
    marker = '    static const x16_spec spec_q5_bytes = '
    source = replace_once(source, marker, '''    static const x16_spec spec_q6_bytes = { GGML_TYPE_Q6_K, GGML_TYPE_Q6_K, QK_K, sizeof(qwen_q6_bytes_x16), GGML_TYPE_Q8_K, ggml_gemv_q6_K_x16_bytes_q8_K, "x16 q6_K bytes" };
    static const tensor_traits_x16 t_q6_bytes(spec_q6_bytes);
    static const bool q6_bytes = [] {
        const char * value = getenv("GGML_CPU_X16_Q6_BYTES");
        return value && atoi(value) == 1;
    }();
''' + marker)
    old = '    if (cur->type == GGML_TYPE_Q6_K && x16_q6_K && cur->ne[0] % QK_K == 0) return &t_q6_K;'
    return replace_once(source, old,
        '    if (cur->type == GGML_TYPE_Q6_K && x16_q6_K && cur->ne[0] % QK_K == 0) return q6_bytes ? &t_q6_bytes : &t_q6_K;')
