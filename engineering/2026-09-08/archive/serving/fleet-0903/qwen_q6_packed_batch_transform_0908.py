"""Add opt-in Q6 expert batching to the selected private CPU source."""


def transform(source):
    include = '#include "repack.h"'
    assert source.count(include) == 1
    source = source.replace(include, include + '\n#include "qwen-q6-packed-batch-0908.h"')
    marker = 'class tensor_traits_x16 : public tensor_traits_base {'
    assert source.count(marker) == 1
    source = source.replace(marker, '''static std::atomic<uint64_t> qwen_q6_batch_counts[4]{};

extern "C" GGML_BACKEND_API uint64_t ggml_cpu_q6_expert_batch_count(int rows) {
    return rows >= 0 && rows < 4 ? qwen_q6_batch_counts[rows].load(std::memory_order_relaxed) : 0;
}

static bool qwen_q6_expert_batch_enabled() {
    static const bool enabled = [] {
        const char * value = getenv("GGML_CPU_X16_Q6_EXPERT_BATCH");
        return value && atoi(value) == 1;
    }();
    return enabled;
}

static void qwen_q6_expert_batch_audit(int rows) {
    static const bool enabled = [] {
        const char * value = getenv("GGML_CPU_X16_Q6_EXPERT_BATCH_AUDIT");
        return value && atoi(value) == 1;
    }();
    if (enabled) qwen_q6_batch_counts[rows].fetch_add(1, std::memory_order_relaxed);
}

''' + marker)
    old = '''            for (int64_t ir = 0; ir < cnt; ir++) {
                const row_map m = rows[e * ne12 + ir];'''
    new = '''            for (int64_t ir = 0; ir < cnt; ir++) {
                if (qwen_q6_expert_batch_enabled() && sp.gemv == ggml_gemv_q6_K_x16_q8_K && ir + 1 < cnt) {
                    const int nr = (int) std::min<int64_t>(3, cnt - ir);
                    const block_q8_K * activation[3];
                    float * output[3];
                    for (int r = 0; r < nr; ++r) {
                        const row_map m = rows[e * ne12 + ir + r];
                        activation[r] = (const block_q8_K *) (wdata + (m.i1 % ne11) * row_bytes + m.i2 * nbw2);
                        output[r] = (float *) ((char *) dst->data + m.i1 * dst->nb[1] + m.i2 * dst->nb[2]) + t0;
                    }
                    qwen_q6_packed_batch((int) k, (int) tr, (const block_q6_K_x16 *) w, activation, output, nr);
                    qwen_q6_expert_batch_audit(nr);
                    ir += nr - 1;
                    continue;
                }
                const row_map m = rows[e * ne12 + ir];'''
    assert source.count(old) == 1
    source = source.replace(old,new)
    old = '''            for (int64_t ir = 0; ir < cnt; ir++) {
                const row_map m = rows[e * n_tokens + ir];'''
    new = '''            for (int64_t ir = 0; ir < cnt; ir++) {
                if (qwen_q6_expert_batch_enabled() && sp.gemv == ggml_gemv_q6_K_x16_q8_K && ir + 1 < cnt) {
                    const int nr = (int) std::min<int64_t>(3, cnt - ir);
                    const block_q8_K * activation[3];
                    float gates[3][tile], ups[3][tile];
                    float * gate_output[3] = { gates[0], gates[1], gates[2] };
                    float * up_output[3] = { ups[0], ups[1], ups[2] };
                    for (int r = 0; r < nr; ++r) {
                        const row_map m = rows[e * n_tokens + ir + r];
                        activation[r] = (const block_q8_K *) (wdata + m.i2 * plane_bytes + (m.i1 % n_src_rows) * row_bytes);
                    }
                    qwen_q6_packed_batch((int) k, (int) tr, (const block_q6_K_x16 *) gw, activation, gate_output, nr);
                    qwen_q6_packed_batch((int) k, (int) tr, (const block_q6_K_x16 *) uw, activation, up_output, nr);
                    for (int r = 0; r < nr; ++r) {
                        const row_map m = rows[e * n_tokens + ir + r];
                        float * out = (float *) ((char *) dst->data + m.i2 * dst->nb[2] + m.i1 * dst->nb[1]);
                        ggml_vec_swiglu_f32(tr, out + t0, gates[r], ups[r]);
                    }
                    qwen_q6_expert_batch_audit(nr);
                    ir += nr - 1;
                    continue;
                }
                const row_map m = rows[e * n_tokens + ir];'''
    assert source.count(old) == 1
    return source.replace(old,new)
