"""Restrict HC work to 64 outputs and expose a bounded Q6 expert-tile experiment."""
from qwen_hc_row_split_transform_0909 import transform as hc_transform
from qwen_hc_row_split_transform_0909 import graph_fixture as hc_fixture


def transform(source):
    source = hc_transform(source)
    old = '(a->ne[1] == 64 || a->ne[1] == 96) && n_threads > a->ne[1] / 8'
    assert source.count(old) == 1
    source = source.replace(old, 'a->ne[1] == 64 && n_threads > a->ne[1] / 8')
    marker = '#include "qwen-q8-hc-ordered-k-0909.h"\n'
    assert source.count(marker) == 1
    source = source.replace(marker, marker + '''
static std::atomic<uint64_t> qwen_q6_moe_tile_calls{0};
extern "C" uint64_t ggml_cpu_qwen_q6_moe_tile_count();
extern "C" uint64_t ggml_cpu_qwen_q6_moe_tile_count() {
    return qwen_q6_moe_tile_calls.load(std::memory_order_relaxed);
}
static int64_t qwen_q6_moe_tile(const ggml_compute_params * params, ggml_type type,
                                int64_t k, int64_t nc, int64_t experts, int64_t used, int64_t tokens) {
    static const int64_t requested = [] {
        const char * value = std::getenv("GGML_CPU_QWEN_Q6_MOE_TILE_ROWS");
        if (value && !std::strcmp(value,"16")) return int64_t(16);
        if (value && !std::strcmp(value,"32")) return int64_t(32);
        if (value && !std::strcmp(value,"48")) return int64_t(48);
        return int64_t(64);
    }();
    if (requested == 64 || params->nth <= 1 || params->use_ref || type != GGML_TYPE_Q6_K ||
            k != 2560 || nc != 160 || experts != 512 || used != 8 || tokens < 1 || tokens > 8) return 64;
    if (params->ith == 0) {
        static std::once_flag first_call;
        std::call_once(first_call, [] {
            GGML_LOG_INFO("QWEN_Q6_MOE_TILE rows=%lld\\n", (long long) requested);
        });
        static const bool audit = [] {
            const char * value = std::getenv("GGML_CPU_QWEN_Q6_MOE_TILE_AUDIT");
            return value && !std::strcmp(value,"1");
        }();
        if (audit) qwen_q6_moe_tile_calls.fetch_add(1,std::memory_order_relaxed);
    }
    return requested;
}
''')
    start = source.index('class tensor_traits_x16 : public tensor_traits_base')
    end = source.index('\n}  // namespace ggml::cpu::repack', start)
    body = source[start:end]
    old = '        constexpr int64_t tile = 64;'
    assert body.count(old) == 2
    first = body.index(old)
    body = body[:first] + body[first:].replace(old, '''        const int64_t tile = qwen_q6_moe_tile(params, sp.dst_type, k, nr0, n_as, n_ids, ne12);''', 1)
    body = body.replace('''        // Dynamic work items (expert, 64-row tile): threads steal tiles, so a core that is
        // shared with another process does not stall the whole socket at the next barrier.
        constexpr int64_t tile = 64;''', '''        const int64_t tile = qwen_q6_moe_tile(params, sp.dst_type, k, n_out, n_experts, n_used, n_tokens);''')
    old = '        float gate_tmp[tile]; float up_tmp[tile];'
    assert body.count(old) == 1
    body = body.replace(old, '        float gate_tmp[64]; float up_tmp[64];')
    return source[:start] + body + source[end:]


def graph_fixture(source):
    source = hc_fixture(source)
    old = '(row_counter && row_flag && !std::strcmp(row_flag,"1") && nr >= 2)'
    assert source.count(old) == 1
    return source.replace(old, '(row_counter && row_flag && !std::strcmp(row_flag,"1") && nc == 64 && nr >= 2)')
