#!/usr/bin/env python3
"""Prepare the next private GLM patch; only --apply changes engine sources."""
import difflib
from pathlib import Path
import sys

base = Path(__file__).resolve().parent
engine = base.parents[1] / 'engines/llama.cpp-glm5n-goal-0904'
changes = {}

def replace(rel, before, after):
    path = engine / rel
    original, current = changes.get(path, (path.read_text(), path.read_text()))
    assert current.count(before) == 1, (rel, before[:80], current.count(before))
    changes[path] = original, current.replace(before, after)

traits = 'ggml/src/ggml-cpu/traits.h'
replace(traits, 'bool ggml_cpu_extra_compute_forward_mul_mat_id_weighted_sum(',
'''bool ggml_cpu_extra_compute_forward_mul_mat_id_swiglu_clamped(
        const struct ggml_compute_params * params,
        struct ggml_tensor * gate_clamp,
        struct ggml_tensor * up_clamp,
        struct ggml_tensor * dst);
bool ggml_cpu_extra_compute_forward_mul_mat_id_weighted_sum(''')
replace(traits, '    virtual bool compute_forward_mul_mat_id_weighted_sum(',
'''    virtual bool compute_forward_mul_mat_id_swiglu_clamped(
            struct ggml_compute_params *,
            struct ggml_tensor *,
            struct ggml_tensor *,
            struct ggml_tensor *) {
        return false;
    }
    virtual bool compute_forward_mul_mat_id_weighted_sum(''')
replace('ggml/src/ggml-cpu/traits.cpp', 'bool ggml_cpu_extra_compute_forward_mul_mat_id_weighted_sum(',
'''bool ggml_cpu_extra_compute_forward_mul_mat_id_swiglu_clamped(
        const struct ggml_compute_params * params,
        struct ggml_tensor * gate_clamp,
        struct ggml_tensor * up_clamp,
        struct ggml_tensor * dst) {
    for (auto extra : ggml_backend_cpu_get_extra_buffer_types()) {
        if (!extra || !extra->context) {
            continue;
        }
        auto buf_extra = (ggml::cpu::extra_buffer_type *) extra->context;
        auto gate_traits = buf_extra->get_tensor_traits(gate_clamp->src[0]);
        auto up_traits = buf_extra->get_tensor_traits(up_clamp->src[0]);
        if (gate_traits && gate_traits == up_traits &&
                gate_traits->compute_forward_mul_mat_id_swiglu_clamped(
                    const_cast<ggml_compute_params *>(params), gate_clamp, up_clamp, dst)) {
            return true;
        }
    }
    return false;
}

bool ggml_cpu_extra_compute_forward_mul_mat_id_weighted_sum(''')
cpu = 'ggml/src/ggml-cpu/ggml-cpu.c'
replace(cpu, 'static bool ggml_cpu_moe_gate_up_fusion = false;',
'''static bool ggml_cpu_moe_gate_up_fusion = false;
static bool ggml_cpu_moe_clamp_fusion = false;''')
replace(cpu, '            ggml_cpu_moe_gate_up_fusion = (env != NULL && atoi(env) == 1);',
'''            ggml_cpu_moe_gate_up_fusion = (env != NULL && atoi(env) == 1);
        }
        {
            const char * env = getenv("GGML_CPU_MOE_CLAMP_FUSION");
            ggml_cpu_moe_clamp_fusion = (env != NULL && atoi(env) == 1);''')
needle = '    if (ggml_cpu_moe_gate_up_fusion && node->op == GGML_OP_MUL_MAT_ID && node_n + 2 < cgraph->n_nodes) {'
replace(cpu, needle, '''    if (ggml_cpu_moe_gate_up_fusion && ggml_cpu_moe_clamp_fusion &&
            node->op == GGML_OP_MUL_MAT_ID && node_n + 4 < cgraph->n_nodes) {
        const enum ggml_op fuse_ops[] = {
            GGML_OP_MUL_MAT_ID, GGML_OP_CLAMP, GGML_OP_MUL_MAT_ID, GGML_OP_CLAMP, GGML_OP_GLU
        };
        const int outputs[] = { node_n + 4 };
        if (ggml_can_fuse_subgraph(cgraph, node_n, 5, fuse_ops, outputs, 1)) {
            struct ggml_tensor * mm0 = cgraph->nodes[node_n];
            struct ggml_tensor * clamp0 = cgraph->nodes[node_n + 1];
            struct ggml_tensor * mm1 = cgraph->nodes[node_n + 2];
            struct ggml_tensor * clamp1 = cgraph->nodes[node_n + 3];
            struct ggml_tensor * glu = cgraph->nodes[node_n + 4];
            const bool operands_match = clamp0->src[0] == mm0 && clamp1->src[0] == mm1 &&
                ((glu->src[0] == clamp0 && glu->src[1] == clamp1) ||
                 (glu->src[0] == clamp1 && glu->src[1] == clamp0));
            if (operands_match && ggml_get_glu_op(glu) == GGML_GLU_OP_SWIGLU &&
                    mm0->src[1] == mm1->src[1] && mm0->src[2] == mm1->src[2] &&
                    mm0->type == GGML_TYPE_F32 && mm1->type == GGML_TYPE_F32 &&
                    clamp0->type == GGML_TYPE_F32 && clamp1->type == GGML_TYPE_F32 && glu->type == GGML_TYPE_F32 &&
                    mm0->src[0]->type == mm1->src[0]->type &&
                    ggml_are_same_shape(mm0->src[0], mm1->src[0]) &&
                    ggml_are_same_shape(mm0, clamp0) && ggml_are_same_shape(mm1, clamp1) &&
                    ggml_are_same_shape(mm0, mm1) && ggml_are_same_shape(mm0, glu) &&
                    ggml_cpu_extra_compute_forward_mul_mat_id_swiglu_clamped(params, glu->src[0], glu->src[1], glu)) {
                return 4;
            }
        }
    }

''' + needle)
repack = 'ggml/src/ggml-cpu/repack.cpp'
needle = '''    void forward_mul_mat_id_swiglu(
            ggml_compute_params * params,
            ggml_tensor * gate,
            ggml_tensor * up,
            ggml_tensor * dst) {'''
replace(repack, needle, '''    bool compute_forward_mul_mat_id_swiglu_clamped(
            ggml_compute_params * params,
            ggml_tensor * gate_clamp,
            ggml_tensor * up_clamp,
            ggml_tensor * dst) override {
        if constexpr (!expanded_iq || NB_COLS != 16) {
            return false;
        } else {
            ggml_tensor * gate = gate_clamp->src[0];
            ggml_tensor * up = up_clamp->src[0];
            if (gate_clamp->op != GGML_OP_CLAMP || up_clamp->op != GGML_OP_CLAMP ||
                    gate->op != GGML_OP_MUL_MAT_ID || up->op != GGML_OP_MUL_MAT_ID ||
                    gate->src[0]->extra != this || up->src[0]->extra != this) {
                return false;
            }
            static std::atomic<bool> logged{false};
            if (params->ith == 0 && !logged.load(std::memory_order_relaxed) &&
                    !logged.exchange(true, std::memory_order_relaxed)) {
                GGML_LOG_INFO("MOE_CLAMP_FUSION_ACTIVE type=%s\\n", ggml_type_name(gate->src[0]->type));
            }
            forward_mul_mat_id_swiglu(params, gate, up, dst, gate_clamp, up_clamp);
            return true;
        }
    }

    void forward_mul_mat_id_swiglu(
            ggml_compute_params * params,
            ggml_tensor * gate,
            ggml_tensor * up,
            ggml_tensor * dst,
            const ggml_tensor * gate_clamp = nullptr,
            const ggml_tensor * up_clamp = nullptr) {''')
needle = '''        const size_t src1_row_size = ggml_row_size(PARAM_TYPE, k);
        const size_t src1_plane_size = src1_row_size * n_src_rows;
        const size_t src1_size = src1_plane_size * n_tokens;

        struct row_mapping {'''
replace(repack, needle, '''        float gate_bounds[2] = {}, up_bounds[2] = {};
        if (gate_clamp && up_clamp) {
            memcpy(gate_bounds, gate_clamp->op_params, sizeof(gate_bounds));
            memcpy(up_bounds, up_clamp->op_params, sizeof(up_bounds));
        }
        const auto apply_clamps = [&](int64_t count, float * gate_values, float * up_values) {
            if (gate_clamp && up_clamp) {
                for (int64_t i = 0; i < count; ++i) {
                    gate_values[i] = MAX(MIN(gate_values[i], gate_bounds[1]), gate_bounds[0]);
                    up_values[i] = MAX(MIN(up_values[i], up_bounds[1]), up_bounds[0]);
                }
            }
        };

''' + needle)
needle = '                                ggml_vec_swiglu_f32(tile_rows, outputs[y] + tile_start, gate_tmp[y], up_tmp[y]);'
replace(repack, needle, '                                apply_clamps(tile_rows, gate_tmp[y], up_tmp[y]);\n' + needle)
needle = '''                    ggml_vec_swiglu_f32(
                        tile_rows, out_dst + tile_start - row_start, gate_tmp, up_tmp);'''
replace(repack, needle, '                    apply_clamps(tile_rows, gate_tmp, up_tmp);\n' + needle)

patch = ''
for path, (old, new) in changes.items():
    rel = str(path.relative_to(engine))
    patch += ''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
        fromfile='a/' + rel, tofile='b/' + rel))
    if '--apply' in sys.argv:
        backup = path.with_name(path.name + '.before-goal-clamped-moe-fusion')
        assert not backup.exists()
        backup.write_text(old)
        path.write_text(new)
(base / 'glm-clamped-moe-fusion.patch').write_text(patch)
print('Applied' if '--apply' in sys.argv else 'Prepared without applying', len(changes), 'files')
