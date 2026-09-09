"""Private, opt-in clamped SwiGLU dispatch for the existing Q8 x16 expert path."""


def once(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new)


def transform(source):
    signature = '    bool compute_forward_mul_mat_id_swiglu(struct ggml_compute_params * params, struct ggml_tensor * gate, struct ggml_tensor * up, struct ggml_tensor * dst) override {'
    assert source.count(signature) == 1
    begin = source.index(signature)
    end = source.index('    bool compute_forward_mul_mat_swiglu(', begin)
    method = source[begin:end]
    wrapper = '''    bool compute_forward_mul_mat_id_swiglu(struct ggml_compute_params * params, struct ggml_tensor * gate, struct ggml_tensor * up, struct ggml_tensor * dst) override {
        return forward_x16_moe_swiglu(params, gate, up, dst);
    }

    bool compute_forward_mul_mat_id_swiglu_clamped(
            ggml_compute_params * params, ggml_tensor * gate_clamp,
            ggml_tensor * up_clamp, ggml_tensor * dst) override {
        static const bool enabled = [] {
            const char * value = getenv("GGML_CPU_X16_Q8_CLAMP_FUSION");
            return value && atoi(value) == 1;
        }();
        if (!enabled || sp.dst_type != GGML_TYPE_Q8_0 ||
                gate_clamp->op != GGML_OP_CLAMP || up_clamp->op != GGML_OP_CLAMP) {
            return false;
        }
        const bool okay = forward_x16_moe_swiglu(
            params, gate_clamp->src[0], up_clamp->src[0], dst, gate_clamp, up_clamp);
        static std::atomic<bool> logged{false};
        if (okay && params->ith == 0 && !logged.exchange(true, std::memory_order_relaxed)) {
            GGML_LOG_INFO("Q8_X16_CLAMP_FUSION_ACTIVE\\n");
        }
        return okay;
    }

    bool forward_x16_moe_swiglu(
            ggml_compute_params * params, ggml_tensor * gate, ggml_tensor * up,
            ggml_tensor * dst, const ggml_tensor * gate_clamp = nullptr,
            const ggml_tensor * up_clamp = nullptr) {'''
    method = once(method, signature, wrapper)
    method = once(method, '        float gate_tmp[tile]; float up_tmp[tile];', '''        float gate_bounds[2] = {}, up_bounds[2] = {};
        GGML_ASSERT((gate_clamp == nullptr) == (up_clamp == nullptr));
        if (gate_clamp) {
            GGML_ASSERT(ggml_are_same_shape(gate, gate_clamp) && ggml_are_same_shape(up, up_clamp));
            memcpy(gate_bounds, gate_clamp->op_params, sizeof(gate_bounds));
            memcpy(up_bounds, up_clamp->op_params, sizeof(up_bounds));
        }
        float gate_tmp[tile]; float up_tmp[tile];''')
    method = once(method, '                ggml_vec_swiglu_f32(tr, out + t0, gate_tmp, up_tmp);', '''                if (gate_clamp) {
                    for (int64_t i = 0; i < tr; ++i) {
                        gate_tmp[i] = MAX(MIN(gate_tmp[i], gate_bounds[1]), gate_bounds[0]);
                        up_tmp[i] = MAX(MIN(up_tmp[i], up_bounds[1]), up_bounds[0]);
                    }
                }
                ggml_vec_swiglu_f32(tr, out + t0, gate_tmp, up_tmp);''')
    return source[:begin] + method + source[end:]
