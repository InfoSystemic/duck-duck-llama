"""Private RMS/scale/repacked-Q8 fusion; retain the existing arithmetic."""

DECLARATIONS = '''bool ggml_cpu_repack_compute_forward_rms_norm_matmul(const struct ggml_compute_params * params, struct ggml_tensor * norm, struct ggml_tensor * activation, struct ggml_tensor * dst);
uint64_t ggml_cpu_rms_matmul_fusion_count(int weighted);
'''

DISPATCH = '''    if (ggml_cpu_rms_matmul_fusion && node->op == GGML_OP_RMS_NORM) {
        const enum ggml_op weighted_ops[] = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_MUL_MAT };
        const enum ggml_op plain_ops[] = { GGML_OP_RMS_NORM, GGML_OP_MUL_MAT };
        for (int count = 3; count >= 2; --count) {
            if (node_n + count > cgraph->n_nodes) continue;
            const int outputs[] = { node_n + count - 1 };
            if (ggml_can_fuse_subgraph(cgraph, node_n, count,
                    count == 3 ? weighted_ops : plain_ops, outputs, 1)) {
                struct ggml_tensor * activation = cgraph->nodes[node_n + count - 2];
                struct ggml_tensor * dst = cgraph->nodes[node_n + count - 1];
                if (ggml_cpu_repack_compute_forward_rms_norm_matmul(params, node, activation, dst)) {
                    return count - 1;
                }
            }
        }
    }

'''

HOOK = '''
static std::atomic<uint64_t> rms_matmul_audit_counts[2] = {};

uint64_t ggml_cpu_rms_matmul_fusion_count(int weighted) {
    return weighted == 0 || weighted == 1 ? rms_matmul_audit_counts[weighted].load(std::memory_order_relaxed) : 0;
}

static bool rms_matmul_overlaps(const ggml_tensor * a, const ggml_tensor * b) {
    const uintptr_t ap = (uintptr_t) a->data, bp = (uintptr_t) b->data;
    return ap <= bp ? bp - ap < ggml_nbytes(a) : ap - bp < ggml_nbytes(b);
}

bool ggml_cpu_repack_compute_forward_rms_norm_matmul(
        const ggml_compute_params * params,
        ggml_tensor * norm,
        ggml_tensor * activation,
        ggml_tensor * dst) {
    const ggml_tensor * x = norm->src[0];
    const ggml_tensor * w = dst->src[0];
    const bool weighted = activation != norm;
    if (params->use_ref || norm->op != GGML_OP_RMS_NORM || dst->op != GGML_OP_MUL_MAT ||
            dst->src[1] != activation || !x || !w || !w->buffer || !w->extra ||
            !ggml_backend_cpu_is_repack_buffer_type(w->buffer->buft) ||
            w->type != GGML_TYPE_Q8_0 || ggml_n_dims(w) != 2 || !ggml_is_contiguous(w) ||
            x->type != GGML_TYPE_F32 || norm->type != GGML_TYPE_F32 ||
            activation->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
            norm->ne[1] < 1 || norm->ne[1] > 3 || norm->ne[2] != 1 || norm->ne[3] != 1 ||
            !ggml_are_same_shape(x, norm) || !ggml_are_same_shape(norm, activation) ||
            !ggml_is_contiguous(x) || !ggml_is_contiguous(norm) || !ggml_is_contiguous(activation) ||
            !ggml_is_contiguous(dst) || norm->view_src || activation->view_src ||
            w->ne[0] != activation->ne[0] || dst->ne[0] != w->ne[1] ||
            dst->ne[1] != activation->ne[1] || dst->ne[2] != 1 || dst->ne[3] != 1 ||
            rms_matmul_overlaps(activation, x)) {
        return false;
    }
    if (weighted) {
        if (activation->op != GGML_OP_MUL) return false;
        const ggml_tensor * scale = activation->src[0] == norm ? activation->src[1] :
            activation->src[1] == norm ? activation->src[0] : nullptr;
        if (!scale || scale->type != GGML_TYPE_F32 || scale->ne[0] != norm->ne[0] ||
                scale->ne[1] < 1 || norm->ne[1] % scale->ne[1] != 0 || scale->ne[2] != 1 || scale->ne[3] != 1 ||
                !ggml_is_contiguous(scale) || rms_matmul_overlaps(activation, scale)) return false;
    }

    // For <=3 rows on one plane, RMS and both repack quantizers assign
    // each complete row to ith. The matmul's existing barrier publishes
    // every quantized row before any output write, including buffer reuse.
    if (weighted) ggml_compute_forward_rms_norm_mul_fused(params, norm, activation);
    else ggml_compute_forward_rms_norm(params, norm);
    auto * traits = static_cast<ggml::cpu::tensor_traits *>(w->extra);
    ggml_compute_params mutable_params = *params;
    // Both private repack traits unconditionally handle MUL_MAT.
    const bool computed = traits->compute_forward(&mutable_params, dst);
    GGML_ASSERT(computed);
    if (params->ith == 0) {
        static const bool audit = [] {
            const char * value = getenv("GGML_CPU_RMS_MATMUL_AUDIT");
            return value && strcmp(value, "1") == 0;
        }();
        if (audit) rms_matmul_audit_counts[weighted ? 1 : 0].fetch_add(1, std::memory_order_relaxed);
        static std::atomic<bool> logged{false};
        if (!logged.load(std::memory_order_relaxed) && !logged.exchange(true, std::memory_order_relaxed)) {
            GGML_LOG_INFO("RMS_MATMUL_FUSION_ACTIVE\\n");
        }
    }
    return true;
}
'''


def transform(cpu, repack, header):
    def replace_once(text, old, new):
        assert text.count(old) == 1, old
        return text.replace(old, new)

    cpu = replace_once(cpu, 'static struct ggml_state g_state = {0};',
                       'static struct ggml_state g_state = {0};\nstatic bool ggml_cpu_rms_matmul_fusion = false;')
    marker = '    if (node->op == GGML_OP_RMS_NORM) {\n        // RMS_NORM + MUL fusion'
    cpu = replace_once(cpu, marker, DISPATCH + marker)
    marker = '        {\n            const char * env = getenv("GGML_CPU_DISABLE_FUSION");'
    cpu = replace_once(cpu, marker, '''        {
            const char * env = getenv("GGML_CPU_RMS_MATMUL_FUSION");
            ggml_cpu_rms_matmul_fusion = env != NULL && atoi(env) == 1;
        }
''' + marker)
    header = replace_once(header, 'bool ggml_cpu_parallel_copy_enabled(void);',
                          DECLARATIONS + '\nbool ggml_cpu_parallel_copy_enabled(void);')
    repack = replace_once(repack, '#include "traits.h"', '#include "traits.h"\n#include "ops.h"')
    repack += HOOK
    return {'ggml-cpu.c': cpu, 'repack.cpp': repack, 'ops.h': header}
