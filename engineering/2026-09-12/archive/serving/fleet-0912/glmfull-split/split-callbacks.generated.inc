struct ggml_backend_meta_split_state split_parent(const struct ggml_tensor * tensor, void * userdata) {
    const llama_meta_device_get_split_state_userdata * ud = (const llama_meta_device_get_split_state_userdata *) userdata;
    const llama_hparams & hparams = ud->model->hparams;
    const std::string tensor_name = tensor->name;
    static const bool glm_attn_tp = []() {
        const char * value = std::getenv("GGML_GLM_ATTN_TP");
        return value != nullptr && std::atoi(value) != 0;
    }();
    static const bool glm_output_tp = []() {
        const char * value = std::getenv("GGML_GLM_OUTPUT_TP");
        return value != nullptr && std::atoi(value) != 0;
    }();
    static const int glm_attn_tp_layers = []() {
        const char * value = std::getenv("GGML_GLM_ATTN_TP_LAYERS");
        return value != nullptr ? std::max(0, std::atoi(value)) : std::numeric_limits<int>::max();
    }();
    int tensor_layer = -1;
    if (tensor_name.rfind("blk.", 0) == 0) {
        const size_t layer_end = tensor_name.find('.', 4);
        if (layer_end != std::string::npos) {
            tensor_layer = std::stoi(tensor_name.substr(4, layer_end - 4));
        }
    }
    const bool glm_attn_tp_tensor = glm_attn_tp && tensor_layer >= 0 && tensor_layer < glm_attn_tp_layers;

    static const std::regex pattern_q_weight        ("blk\\.\\d*\\.attn_q.weight");
    static const std::regex pattern_kv_weight       ("blk\\.\\d*\\.attn_(k|v).weight");
    static const std::regex pattern_qkv_weight      ("blk\\.\\d*\\.attn_qkv.weight");
    static const std::regex pattern_q_bias          ("blk\\.\\d*\\.attn_q\\.bias");
    static const std::regex pattern_kv_bias         ("blk\\.\\d*\\.attn_(k|v)\\.bias");
    static const std::regex pattern_qkv_bias        ("blk\\.\\d*\\.attn_qkv.bias");
    static const std::regex pattern_qk_norm         ("blk\\.\\d*\\.attn_(q|k)_norm\\.weight");
    static const std::regex pattern_kv_cache        ("cache_(k|v)_l\\d*");
    static const std::regex pattern_attn_sinks      ("blk\\.\\d*\\.attn_sinks.weight");
    static const std::regex pattern_attn_out_weight ("blk\\.\\d*\\.attn_output.weight");
    static const std::regex pattern_attn_out_bias   ("blk\\.\\d*\\.attn_output.bias");
    static const std::regex pattern_attn_gate_weight("blk\\.\\d*\\.attn_gate.weight");
    static const std::regex pattern_attn_q_b_weight ("blk\\.\\d*\\.attn_q_b.weight");
    static const std::regex pattern_attn_kv_b_weight("blk\\.\\d*\\.attn_(k|v)_b.weight");

    static const std::regex pattern_ssm_dt          ("blk\\.\\d*\\.ssm_dt.bias");
    static const std::regex pattern_ssm_a           ("blk\\.\\d*\\.ssm_a");
    static const std::regex pattern_ssm_alpha       ("blk\\.\\d*\\.ssm_alpha.weight");
    static const std::regex pattern_ssm_beta        ("blk\\.\\d*\\.ssm_beta.weight");
    static const std::regex pattern_ssm_beta_alpha  ("blk\\.\\d*\\.ssm_ba.weight");
    static const std::regex pattern_r_cache         ("cache_r_l\\d*");
    static const std::regex pattern_s_cache         ("cache_s_l\\d*");
    static const std::regex pattern_ssm_conv1d      ("blk\\.\\d*\\.ssm_conv1d.weight");
    static const std::regex pattern_ssm_out_weight  ("blk\\.\\d*\\.ssm_out.weight");

    static const std::regex pattern_ffn_up_weight     ("blk\\.\\d*\\.ffn_up(_exps)?.weight");
    static const std::regex pattern_ffn_up_bias       ("blk\\.\\d*\\.ffn_up(_exps)?.bias");
    static const std::regex pattern_ffn_gate_weight   ("blk\\.\\d*\\.ffn_gate(_exps)?.weight");
    static const std::regex pattern_ffn_gate_bias     ("blk\\.\\d*\\.ffn_gate(_exps)?.bias");
    static const std::regex pattern_ffn_gate_up_weight("blk\\.\\d*\\.ffn_gate_up(_exps)?.weight");
    static const std::regex pattern_ffn_down_weight   ("blk\\.\\d*\\.ffn_down(_exps)?.weight");
    static const std::regex pattern_ffn_down_bias     ("blk\\.\\d*\\.ffn_down.bias");
    static const std::regex pattern_ffn_down_exps_bias("blk\\.\\d*\\.ffn_down_exps.bias");

    static const std::regex pattern_output_weight("output\\.weight");
    static const std::regex pattern_output_bias  ("output\\.bias");

    // GLM shared expert: shard like the routed experts (gate/up by output rows,
    // down by input columns -> PARTIAL) instead of mirroring ~40 MB per layer
    // onto every socket.  Opt-in via GGML_GLM_SHEXP_TP=1.
    static const std::regex pattern_ffn_shexp_gate_up_weight("blk\\.\\d*\\.ffn_(up|gate)_shexp\\.weight");
    static const std::regex pattern_ffn_shexp_down_weight   ("blk\\.\\d*\\.ffn_down_shexp\\.weight");
    static const bool glm_shexp_tp = []() {
        const char * value = std::getenv("GGML_GLM_SHEXP_TP");
        return value != nullptr && std::atoi(value) != 0;
    }();
    // GLM MLA down-projections (q_a, kv_a_mqa): split along the input dimension so each
    // socket streams a quarter and contributes a partial that is all-reduced before the
    // following RMS norm.  Opt-in via GGML_GLM_QA_TP=1.
    static const std::regex pattern_attn_q_a_weight    ("blk\\.\\d*\\.attn_q_a\\.weight");
    static const std::regex pattern_attn_kv_a_weight   ("blk\\.\\d*\\.attn_kv_a_mqa\\.weight");
    // 1 = q_a only, 2 = q_a + kv_a_mqa, 3 = kv_a_mqa only (debug knob)
    static const int glm_qa_tp = []() {
        const char * value = std::getenv("GGML_GLM_QA_TP");
        return value != nullptr ? std::atoi(value) : 0;
    }();

    struct tensor_config {
        ggml_backend_meta_split_axis axis;

        const ggml_tensor * tensor_axis_0;

        uint32_t il;
        size_t   rotation; // when assigning tensor slices, rotate how the rounding is done for more even allocation
    };

    auto get_tensor_config_impl = [&](
                const ggml_backend_meta_split_axis axis, const std::string & suffix = "", const std::string & suffix_fallback = "") -> tensor_config {
        // the layers in a tensor can be inhomogeneous, if the pattern is cleanly divided by the number of GPUs there can be aliasing effects,
        //     count only the same type of previous layers to avoid this
        auto get_il_eff = [&](const size_t il){
            size_t ret = 0;
            const bool il_is_recr = hparams.is_recr(il);
            const bool il_is_swa  = hparams.is_swa(il);
            for (size_t il_prev = 0; il_prev < il; il_prev++) {
                ret += hparams.is_recr(il_prev) == il_is_recr && hparams.is_swa(il_prev) == il_is_swa;
            }
            return ret;
        };

        uint32_t il;
        std::string prefix;
        size_t rotation;
        if (tensor_name.substr(0, 4) == "blk.") {
            const size_t length_prefix = tensor_name.find('.', 4);
            GGML_ASSERT(length_prefix != std::string::npos);
            prefix = tensor_name.substr(0, length_prefix + 1);
            il = std::stoull(tensor_name.substr(4, length_prefix));
            rotation = get_il_eff(il) % ud->n_devices;
        } else if (tensor_name.substr(0, 6) == "cache_") {
            const size_t layer_index_start = tensor_name.find("_l", 6);
            GGML_ASSERT(layer_index_start != std::string::npos);
            il = std::stoull(tensor_name.substr(layer_index_start + 2));
            prefix = "blk." + std::to_string(il) + ".";
            rotation = get_il_eff(il) % ud->n_devices;
        } else {
            il = 0;
            rotation = hparams.n_layer() % ud->n_devices;
        }
        const ggml_tensor * tensor_axis_0 = suffix.empty() ? tensor : ud->model->get_tensor((prefix + suffix).c_str());
        if (tensor_axis_0 == nullptr) {
            GGML_ASSERT(!suffix_fallback.empty());
            tensor_axis_0 = ud->model->get_tensor((prefix + suffix_fallback).c_str());
        }
        GGML_ASSERT(tensor_axis_0 != nullptr);
        return {axis, tensor_axis_0, il, rotation};
    };

    auto get_tensor_config = [&]() -> tensor_config {
        // GLM DSA uses MLA and 3D expert tensors. Shard the decompressed MLA
        // heads together: wq_b produces head-local Q, wk_b/wv_b carry the same
        // head axis, and wo consumes it to produce a partial residual that is
        // all-reduced. The compressed MQA K/V path and DSA indexer stay mirrored.
        if (ud->model->arch == LLM_ARCH_GLM_DSA) {
            if (glm_attn_tp_tensor && std::regex_match(tensor_name, pattern_attn_q_b_weight)) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "attn_output.weight");
            }
            if (glm_attn_tp_tensor && std::regex_match(tensor_name, pattern_attn_kv_b_weight)) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_2);
            }
            if (glm_attn_tp_tensor && std::regex_match(tensor_name, pattern_attn_out_weight)) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0);
            }
            if ((glm_qa_tp == 1 || glm_qa_tp == 2) && std::regex_match(tensor_name, pattern_attn_q_a_weight)) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0);
            }
            if ((glm_qa_tp == 2 || glm_qa_tp == 3) && std::regex_match(tensor_name, pattern_attn_kv_a_weight)) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0);
            }
            if (glm_shexp_tp && std::regex_match(tensor_name, pattern_ffn_shexp_gate_up_weight)) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "ffn_down_shexp.weight");
            }
            if (glm_shexp_tp && std::regex_match(tensor_name, pattern_ffn_shexp_down_weight)) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "ffn_down_shexp.weight");
            }
            if (std::regex_match(tensor_name, pattern_ffn_up_weight) ||
                    std::regex_match(tensor_name, pattern_ffn_gate_weight)) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "ffn_down.weight", "ffn_down_exps.weight");
            }
            if (std::regex_match(tensor_name, pattern_ffn_up_bias) ||
                    std::regex_match(tensor_name, pattern_ffn_gate_bias)) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "ffn_down.weight", "ffn_down_exps.weight");
            }
            if (std::regex_match(tensor_name, pattern_ffn_gate_up_weight)) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "ffn_down.weight", "ffn_down_exps.weight");
            }
            if (std::regex_match(tensor_name, pattern_ffn_down_weight)) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "ffn_down.weight", "ffn_down_exps.weight");
            }
            if (std::regex_match(tensor_name, pattern_ffn_down_bias)) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            }
            if (std::regex_match(tensor_name, pattern_ffn_down_exps_bias)) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_PARTIAL);
            }
            // The vocabulary projection is naturally row-parallel: every
            // device consumes the same hidden state and produces a disjoint
            // vocabulary slice which the meta backend gathers for sampling.
            // Mirroring it makes all sockets redundantly read and compute the
            // full 154880-row GLM head on every target and MTP decode step.
            if (glm_output_tp && std::regex_match(tensor_name, pattern_output_weight)) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1);
            }
            if (glm_output_tp && std::regex_match(tensor_name, pattern_output_bias)) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0);
            }
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        }

        // standard attention
        if (std::regex_match(tensor_name, pattern_q_weight) || std::regex_match(tensor_name, pattern_kv_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "attn_output.weight", "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_q_bias) || std::regex_match(tensor_name, pattern_kv_bias)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "attn_output.weight", "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_qkv_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "attn_output.weight", "ssm_out.weight");
        }
        if ( std::regex_match(tensor_name, pattern_qkv_bias)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "attn_output.weight", "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_qk_norm)) {
            return get_tensor_config_impl(tensor->ne[1] == 1 ? GGML_BACKEND_SPLIT_AXIS_MIRRORED : GGML_BACKEND_SPLIT_AXIS_1, "attn_output.weight");
        }
        if (std::regex_match(tensor_name, pattern_kv_cache) || std::regex_match(tensor_name, pattern_attn_sinks)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "attn_output.weight");
        }
        if (std::regex_match(tensor_name, pattern_attn_out_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0);
        }
        if (std::regex_match(tensor_name, pattern_attn_out_bias)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        }

        if (std::regex_match(tensor_name, pattern_attn_gate_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "attn_output.weight", "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_ssm_dt) || std::regex_match(tensor_name, pattern_ssm_a)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_ssm_alpha) || std::regex_match(tensor_name, pattern_ssm_beta) ||
                std::regex_match(tensor_name, pattern_ssm_beta_alpha)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_r_cache) || std::regex_match(tensor_name, pattern_s_cache)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_ssm_conv1d)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_ssm_out_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0);
        }

        // FFN
        if (std::regex_match(tensor_name, pattern_ffn_up_weight) || std::regex_match(tensor_name, pattern_ffn_gate_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "ffn_down.weight", "ffn_down_exps.weight");
        }
        if (std::regex_match(tensor_name, pattern_ffn_up_bias) || std::regex_match(tensor_name, pattern_ffn_gate_bias)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "ffn_down.weight", "ffn_down_exps.weight");
        }
        if (std::regex_match(tensor_name, pattern_ffn_gate_up_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "ffn_down.weight", "ffn_down_exps.weight");
        }
        if (std::regex_match(tensor_name, pattern_ffn_down_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "ffn_down.weight", "ffn_down_exps.weight");
        }
        if (std::regex_match(tensor_name, pattern_ffn_down_bias)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        }
        if (std::regex_match(tensor_name, pattern_ffn_down_exps_bias)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_PARTIAL);
        }

        // output
        if (std::regex_match(tensor_name, pattern_output_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1);
        }
        if (std::regex_match(tensor_name, pattern_output_bias)) {
            const ggml_tensor * output_weight = ud->model->get_tensor("output.weight");
            GGML_ASSERT(output_weight != nullptr);
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0);
        }

        // everything else
        return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
    };

    auto get_split_segments = [&](int axis, uint32_t il) -> std::vector<std::pair<int64_t, uint32_t>> {
        if (ud->model->arch == LLM_ARCH_QWEN3NEXT || ud->model->arch == LLM_ARCH_QWEN35 || ud->model->arch == LLM_ARCH_QWEN35MOE) {
            const int64_t head_k_dim = hparams.ssm_d_state;
            const int64_t head_v_dim = hparams.ssm_d_state;
            const int64_t n_k_heads  = hparams.ssm_n_group;
            const int64_t n_v_heads  = hparams.ssm_dt_rank;
            const int64_t key_dim    = head_k_dim * n_k_heads;
            const int64_t value_dim  = head_v_dim * n_v_heads;

            // both Qwen 3 Next and Qwen 3.5 support n_v_heads > n_k_heads but the broadcasting pattern is different:
            //   - Qwen 3 Next: [k0_v0, k0_v1, k1_v2, k1_v3] (this is the default split pattern)
            //   - Qwen 3.5:    [k0_v0, k1_v1, k0_v2, k1_v3] (needs segmenting of V on the scale of K to get the correct pattern)
            if (ud->model->arch == LLM_ARCH_QWEN3NEXT) {
                if (std::regex_match(tensor_name, pattern_qkv_weight) || std::regex_match(tensor_name, pattern_ssm_conv1d)) {
                    GGML_ASSERT(tensor->ne[axis] == 2*key_dim + value_dim);
                    return {{key_dim, 2}, {value_dim, 1}};
                }
            } else {
                const int64_t head_ratio = n_v_heads / n_k_heads;
                if (std::regex_match(tensor_name, pattern_qkv_weight) || std::regex_match(tensor_name, pattern_ssm_conv1d)) {
                    GGML_ASSERT(tensor->ne[axis] == 2*key_dim + value_dim);
                    return {{key_dim, 2 + head_ratio}};
                }
                if (std::regex_match(tensor_name, pattern_attn_gate_weight) || std::regex_match(tensor_name, pattern_ssm_out_weight)) {
                    return {{key_dim, head_ratio}};
                }
                if (std::regex_match(tensor_name, pattern_ssm_dt) || std::regex_match(tensor_name, pattern_ssm_a) ||
                        std::regex_match(tensor_name, pattern_ssm_alpha) || std::regex_match(tensor_name, pattern_ssm_beta)) {
                    return {{n_k_heads, head_ratio}};
                }
                if (std::regex_match(tensor_name, pattern_r_cache)) {
                    return {{key_dim * (hparams.ssm_d_conv - 1), 2 + head_ratio}};
                }
                if (std::regex_match(tensor_name, pattern_s_cache)) {
                    return {{n_k_heads * head_v_dim * head_v_dim, head_ratio}};
                }
            }

            // the FFN is the same for Qwen 3 Next and Qwen 3.5:
            if (std::regex_match(tensor_name, pattern_ffn_gate_up_weight)) {
                const int64_t n_ff_exp = hparams.n_ff_exp;
                GGML_ASSERT(tensor->ne[axis] == 2*n_ff_exp);
                return {{n_ff_exp, 2}};
            }
            return {{tensor->ne[axis], 1}};
        }

        if (std::regex_match(tensor_name, pattern_qkv_weight) || std::regex_match(tensor_name, pattern_qkv_bias)) {
            const int64_t n_embd      = hparams.n_embd;
            const int64_t n_embd_gqa  = hparams.n_embd_v_gqa(il);
            GGML_ASSERT(hparams.n_embd_k_gqa() == n_embd_gqa);
            GGML_ASSERT(tensor->ne[axis] == n_embd + 2*n_embd_gqa);
            return {{n_embd, 1}, {n_embd_gqa, 2}};
        }
        if (std::regex_match(tensor_name, pattern_ffn_up_weight) || std::regex_match(tensor_name, pattern_ffn_up_bias)) {
            const int64_t n_ff = hparams.n_ff(il);
            // some models such as Phi 3 have fused up + gate tensors named "up" tensors, which need to be segmented
            if (tensor->ne[axis] == 2*n_ff) {
                return {{n_ff, 2}};
            }
            return {{tensor->ne[axis], 1}};
        }
        if (std::regex_match(tensor_name, pattern_ffn_gate_up_weight)) {
            const int64_t n_ff_exp = hparams.n_ff_exp;
            GGML_ASSERT(tensor->ne[axis] == 2*n_ff_exp);
            return {{n_ff_exp, 2}};
        }
        return {{tensor->ne[axis], 1}};
    };

    auto get_split_granularity = [&](int64_t blck_size, uint32_t il, const std::vector<std::pair<int64_t, uint32_t>> & segments) -> std::vector<int64_t> {
        if (ud->model->arch == LLM_ARCH_GLM_DSA && glm_attn_tp_tensor) {
            if (std::regex_match(tensor_name, pattern_attn_q_b_weight)) {
                GGML_ASSERT(segments.size() == 1);
                return {std::lcm(blck_size, tensor->ne[1] / hparams.n_head(il))};
            }
            if (std::regex_match(tensor_name, pattern_attn_kv_b_weight)) {
                GGML_ASSERT(segments.size() == 1);
                return {1};
            }
            if (std::regex_match(tensor_name, pattern_attn_out_weight)) {
                GGML_ASSERT(segments.size() == 1);
                return {std::lcm(blck_size, tensor->ne[0] / hparams.n_head(il))};
            }
        }

        // for better performance it may make sense to round up blck_size to a higher power of 2 so that more efficient kernels can be used
        if (hparams.is_recr(il)) {
            // linear attention
            const int64_t head_dim        = hparams.ssm_d_state;
            const int64_t blck_size_perf  = std::lcm(blck_size, 128);
            const int64_t granularity_qkv = std::lcm(blck_size_perf, head_dim);
            if (std::regex_match(tensor_name, pattern_qkv_weight) || std::regex_match(tensor_name, pattern_attn_gate_weight) ||
                    std::regex_match(tensor_name, pattern_ssm_conv1d) || std::regex_match(tensor_name, pattern_ssm_out_weight)) {
                return std::vector<int64_t>(segments.size(), granularity_qkv);
            }
            if (std::regex_match(tensor_name, pattern_ssm_dt) || std::regex_match(tensor_name, pattern_ssm_a) ||
                    std::regex_match(tensor_name, pattern_ssm_alpha) || std::regex_match(tensor_name, pattern_ssm_beta)) {
                return std::vector<int64_t>(segments.size(), granularity_qkv / head_dim);
            }
            if (std::regex_match(tensor_name, pattern_ssm_beta_alpha)) {
                return std::vector<int64_t>(segments.size(), 2 * (granularity_qkv / head_dim));
            }
            if (std::regex_match(tensor_name, pattern_r_cache)) {
                return std::vector<int64_t>(segments.size(), granularity_qkv * (hparams.ssm_d_conv - 1));
            }
            if (std::regex_match(tensor_name, pattern_s_cache)) {
                return std::vector<int64_t>(segments.size(), granularity_qkv * head_dim);
            }
        } else {
            // regular attention
            const uint32_t n_gqa    = hparams.n_gqa(il);
            const uint32_t n_embd_q = n_gqa * hparams.n_embd_head_k(il);

            // to handle head sizes like 80, only increase granularity while it doesn't cause underutilization
            int64_t blck_size_perf = blck_size;
            while (blck_size_perf < 128 && blck_size_perf*ud->n_devices < n_embd_q) {
                blck_size_perf *= 2;
            }

            if (std::regex_match(tensor_name, pattern_attn_sinks)) {
                GGML_ASSERT(segments.size() == 1);
                return {std::lcm(n_embd_q, blck_size_perf)/n_embd_q * n_gqa};
            }

            const int64_t granularity_q = std::lcm(n_embd_q, blck_size_perf);
            if (std::regex_match(tensor_name, pattern_q_weight) || std::regex_match(tensor_name, pattern_q_bias)) {
                GGML_ASSERT(segments.size() == 1);
                // some models have Q gate tensors, for those cases the granularity needs to be doubled:
                if (ud->model->arch == LLM_ARCH_QWEN3NEXT || ud->model->arch == LLM_ARCH_QWEN35 || ud->model->arch == LLM_ARCH_QWEN35MOE) {
                    return {std::lcm(2*n_embd_q, blck_size_perf)};
                }
                return {granularity_q};
            }
            if (std::regex_match(tensor_name, pattern_attn_out_weight)) {
                GGML_ASSERT(segments.size() == 1);
                return {granularity_q};
            }

            const int64_t granularity_kv = granularity_q / n_gqa;
            if (std::regex_match(tensor_name, pattern_kv_weight) ||
                std::regex_match(tensor_name, pattern_kv_bias) ||
                std::regex_match(tensor_name, pattern_kv_cache)) {
                GGML_ASSERT(segments.size() == 1);
                return {granularity_kv};
            }
            if (std::regex_match(tensor_name, pattern_qkv_weight) || std::regex_match(tensor_name, pattern_qkv_bias)) {
                GGML_ASSERT(segments.size() == 2);
                return {granularity_q, granularity_kv};
            }
        }

        // FFN
        if (std::regex_match(tensor_name, pattern_ffn_up_weight) || std::regex_match(tensor_name, pattern_ffn_up_bias) ||
                std::regex_match(tensor_name, pattern_ffn_gate_weight) || std::regex_match(tensor_name, pattern_ffn_gate_bias) ||
                std::regex_match(tensor_name, pattern_ffn_gate_up_weight) || std::regex_match(tensor_name, pattern_ffn_down_weight)) {
            const int64_t blck_size_perf = std::lcm(blck_size, 128);
            GGML_ASSERT(segments.size() == 1);
            return {blck_size_perf};
        }

        // everything else
        GGML_ASSERT(segments.size() == 1);
        return {1};
    };

    ggml_backend_meta_split_state split_state;
    memset(&split_state, 0, sizeof(split_state));
    tensor_config tc = get_tensor_config();
    split_state.axis = tc.axis;
    if (split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS) {
        const int64_t blck_size = ggml_blck_size(tc.tensor_axis_0->type);
        const float * tensor_split = ud->model->tensor_split();
        std::vector<float> tensor_split_scan;
        tensor_split_scan.reserve(ud->n_devices);
        for (size_t j = 0; j < ud->n_devices; j++) {
            tensor_split_scan.push_back(tensor_split == nullptr ? 0.0f : tensor_split[(j + tc.rotation) % ud->n_devices]);
            if (j > 0) {
                tensor_split_scan[j] += tensor_split_scan[j - 1];
            }
        }
        const std::vector<std::pair<int64_t, uint32_t>> segments = get_split_segments(split_state.axis, tc.il);
        const std::vector<int64_t> granularity = get_split_granularity(blck_size, tc.il, segments);
        for (size_t is = 0; is < segments.size(); is++) {
            const int64_t  ne_s = segments[is].first;
            const uint32_t nr_s = segments[is].second;
            const int64_t  g_s  = granularity[is];
            int64_t low = 0;
            size_t j = 0;
            for (; j < ud->n_devices - 1; j++) {
                int64_t high = tensor_split_scan.back() == 0.0f ?
                    ne_s * (j+1)/ud->n_devices : ne_s * tensor_split_scan[j]/tensor_split_scan.back();
                if (high % g_s != 0) {
                    high -= high % g_s;
                }
                split_state.ne[is*ud->n_devices + (j + tc.rotation) % ud->n_devices] = high - low;
                low = high;
            }
            split_state.ne[is*ud->n_devices + (j + tc.rotation) % ud->n_devices] = ne_s - low;
            split_state.nr[is] = nr_s;
        }
        split_state.n_segments = segments.size();
    } else {
        memset(split_state.ne, 0, sizeof(split_state.ne));
        split_state.nr[0] = 1;
        split_state.n_segments = 1;
    }
    return split_state;
    GGML_UNUSED(userdata);
}
struct ggml_backend_meta_split_state split_candidate(const struct ggml_tensor * tensor, void * userdata) {
    const llama_meta_device_get_split_state_userdata * ud = (const llama_meta_device_get_split_state_userdata *) userdata;
    const llama_hparams & hparams = ud->model->hparams;
    const std::string tensor_name = tensor->name;
    static const bool glm_attn_tp = []() {
        const char * value = std::getenv("GGML_GLM_ATTN_TP");
        return value != nullptr && std::atoi(value) != 0;
    }();
    static const bool glm_output_tp = []() {
        const char * value = std::getenv("GGML_GLM_OUTPUT_TP");
        return value != nullptr && std::atoi(value) != 0;
    }();
    static const int glm_attn_tp_layers = []() {
        const char * value = std::getenv("GGML_GLM_ATTN_TP_LAYERS");
        return value != nullptr ? std::max(0, std::atoi(value)) : std::numeric_limits<int>::max();
    }();
    int tensor_layer = -1;
    if (tensor_name.rfind("blk.", 0) == 0) {
        const size_t layer_end = tensor_name.find('.', 4);
        if (layer_end != std::string::npos) {
            tensor_layer = std::stoi(tensor_name.substr(4, layer_end - 4));
        }
    }
    const bool glm_attn_tp_tensor = glm_attn_tp && tensor_layer >= 0 && tensor_layer < glm_attn_tp_layers;

    static const std::regex pattern_q_weight        ("blk\\.\\d*\\.attn_q.weight");
    static const std::regex pattern_kv_weight       ("blk\\.\\d*\\.attn_(k|v).weight");
    static const std::regex pattern_qkv_weight      ("blk\\.\\d*\\.attn_qkv.weight");
    static const std::regex pattern_q_bias          ("blk\\.\\d*\\.attn_q\\.bias");
    static const std::regex pattern_kv_bias         ("blk\\.\\d*\\.attn_(k|v)\\.bias");
    static const std::regex pattern_qkv_bias        ("blk\\.\\d*\\.attn_qkv.bias");
    static const std::regex pattern_qk_norm         ("blk\\.\\d*\\.attn_(q|k)_norm\\.weight");
    static const std::regex pattern_kv_cache        ("cache_(k|v)_l\\d*");
    static const std::regex pattern_attn_sinks      ("blk\\.\\d*\\.attn_sinks.weight");
    static const std::regex pattern_attn_out_weight ("blk\\.\\d*\\.attn_output.weight");
    static const std::regex pattern_attn_out_bias   ("blk\\.\\d*\\.attn_output.bias");
    static const std::regex pattern_attn_gate_weight("blk\\.\\d*\\.attn_gate.weight");
    static const std::regex pattern_attn_q_b_weight ("blk\\.\\d*\\.attn_q_b.weight");
    static const std::regex pattern_attn_kv_b_weight("blk\\.\\d*\\.attn_(k|v)_b.weight");

    static const std::regex pattern_ssm_dt          ("blk\\.\\d*\\.ssm_dt.bias");
    static const std::regex pattern_ssm_a           ("blk\\.\\d*\\.ssm_a");
    static const std::regex pattern_ssm_alpha       ("blk\\.\\d*\\.ssm_alpha.weight");
    static const std::regex pattern_ssm_beta        ("blk\\.\\d*\\.ssm_beta.weight");
    static const std::regex pattern_ssm_beta_alpha  ("blk\\.\\d*\\.ssm_ba.weight");
    static const std::regex pattern_r_cache         ("cache_r_l\\d*");
    static const std::regex pattern_s_cache         ("cache_s_l\\d*");
    static const std::regex pattern_ssm_conv1d      ("blk\\.\\d*\\.ssm_conv1d.weight");
    static const std::regex pattern_ssm_out_weight  ("blk\\.\\d*\\.ssm_out.weight");

    static const std::regex pattern_ffn_up_weight     ("blk\\.\\d*\\.ffn_up(_exps)?.weight");
    static const std::regex pattern_ffn_up_bias       ("blk\\.\\d*\\.ffn_up(_exps)?.bias");
    static const std::regex pattern_ffn_gate_weight   ("blk\\.\\d*\\.ffn_gate(_exps)?.weight");
    static const std::regex pattern_ffn_gate_bias     ("blk\\.\\d*\\.ffn_gate(_exps)?.bias");
    static const std::regex pattern_ffn_gate_up_weight("blk\\.\\d*\\.ffn_gate_up(_exps)?.weight");
    static const std::regex pattern_ffn_down_weight   ("blk\\.\\d*\\.ffn_down(_exps)?.weight");
    static const std::regex pattern_ffn_down_bias     ("blk\\.\\d*\\.ffn_down.bias");
    static const std::regex pattern_ffn_down_exps_bias("blk\\.\\d*\\.ffn_down_exps.bias");

    static const std::regex pattern_output_weight("output\\.weight");
    static const std::regex pattern_output_bias  ("output\\.bias");

    // GLM shared expert: shard like the routed experts (gate/up by output rows,
    // down by input columns -> PARTIAL) instead of mirroring ~40 MB per layer
    // onto every socket.  Opt-in via GGML_GLM_SHEXP_TP=1.
    static const std::regex pattern_ffn_shexp_gate_up_weight("blk\\.\\d*\\.ffn_(up|gate)_shexp\\.weight");
    static const std::regex pattern_ffn_shexp_down_weight   ("blk\\.\\d*\\.ffn_down_shexp\\.weight");
    static const bool glm_shexp_tp = []() {
        const char * value = std::getenv("GGML_GLM_SHEXP_TP");
        return value != nullptr && std::atoi(value) != 0;
    }();
    // GLM MLA down-projections (q_a, kv_a_mqa): split along the input dimension so each
    // socket streams a quarter and contributes a partial that is all-reduced before the
    // following RMS norm.  Opt-in via GGML_GLM_QA_TP=1.
    static const std::regex pattern_attn_q_a_weight    ("blk\\.\\d*\\.attn_q_a\\.weight");
    static const std::regex pattern_attn_kv_a_weight   ("blk\\.\\d*\\.attn_kv_a_mqa\\.weight");
    // 1 = q_a only, 2 = q_a + kv_a_mqa, 3 = kv_a_mqa only (debug knob)
    static const int glm_qa_tp = []() {
        const char * value = std::getenv("GGML_GLM_QA_TP");
        return value != nullptr ? std::atoi(value) : 0;
    }();

    struct tensor_config {
        ggml_backend_meta_split_axis axis;

        const ggml_tensor * tensor_axis_0;

        uint32_t il;
        size_t   rotation; // when assigning tensor slices, rotate how the rounding is done for more even allocation
    };

    auto get_tensor_config_impl = [&](
                const ggml_backend_meta_split_axis axis, const std::string & suffix = "", const std::string & suffix_fallback = "") -> tensor_config {
        // the layers in a tensor can be inhomogeneous, if the pattern is cleanly divided by the number of GPUs there can be aliasing effects,
        //     count only the same type of previous layers to avoid this
        auto get_il_eff = [&](const size_t il){
            size_t ret = 0;
            const bool il_is_recr = hparams.is_recr(il);
            const bool il_is_swa  = hparams.is_swa(il);
            for (size_t il_prev = 0; il_prev < il; il_prev++) {
                ret += hparams.is_recr(il_prev) == il_is_recr && hparams.is_swa(il_prev) == il_is_swa;
            }
            return ret;
        };

        uint32_t il;
        std::string prefix;
        size_t rotation;
        if (tensor_name.substr(0, 4) == "blk.") {
            const size_t length_prefix = tensor_name.find('.', 4);
            GGML_ASSERT(length_prefix != std::string::npos);
            prefix = tensor_name.substr(0, length_prefix + 1);
            il = std::stoull(tensor_name.substr(4, length_prefix));
            rotation = get_il_eff(il) % ud->n_devices;
        } else if (tensor_name.substr(0, 6) == "cache_") {
            const size_t layer_index_start = tensor_name.find("_l", 6);
            GGML_ASSERT(layer_index_start != std::string::npos);
            il = std::stoull(tensor_name.substr(layer_index_start + 2));
            prefix = "blk." + std::to_string(il) + ".";
            rotation = get_il_eff(il) % ud->n_devices;
        } else {
            il = 0;
            rotation = hparams.n_layer() % ud->n_devices;
        }
        const ggml_tensor * tensor_axis_0 = suffix.empty() ? tensor : ud->model->get_tensor((prefix + suffix).c_str());
        if (tensor_axis_0 == nullptr) {
            GGML_ASSERT(!suffix_fallback.empty());
            tensor_axis_0 = ud->model->get_tensor((prefix + suffix_fallback).c_str());
        }
        GGML_ASSERT(tensor_axis_0 != nullptr);
        return {axis, tensor_axis_0, il, rotation};
    };

    auto get_tensor_config = [&]() -> tensor_config {
        // GLM DSA uses MLA and 3D expert tensors. Shard the decompressed MLA
        // heads together: wq_b produces head-local Q, wk_b/wv_b carry the same
        // head axis, and wo consumes it to produce a partial residual that is
        // all-reduced. The compressed MQA K/V path and DSA indexer stay mirrored.
        if (ud->model->arch == LLM_ARCH_GLM_DSA) {
            if (glm_attn_tp_tensor && std::regex_match(tensor_name, pattern_attn_q_b_weight)) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "attn_output.weight");
            }
            if (glm_attn_tp_tensor && std::regex_match(tensor_name, pattern_attn_kv_b_weight)) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_2);
            }
            if (glm_attn_tp_tensor && std::regex_match(tensor_name, pattern_attn_out_weight)) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0);
            }
            if ((glm_qa_tp == 1 || glm_qa_tp == 2) && std::regex_match(tensor_name, pattern_attn_q_a_weight)) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0);
            }
            if ((glm_qa_tp == 2 || glm_qa_tp == 3) && std::regex_match(tensor_name, pattern_attn_kv_a_weight)) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0);
            }
            if (glm_shexp_tp && std::regex_match(tensor_name, pattern_ffn_shexp_gate_up_weight)) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "ffn_down_shexp.weight");
            }
            if (glm_shexp_tp && std::regex_match(tensor_name, pattern_ffn_shexp_down_weight)) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "ffn_down_shexp.weight");
            }
            if (std::regex_match(tensor_name, pattern_ffn_up_weight) ||
                    std::regex_match(tensor_name, pattern_ffn_gate_weight)) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "ffn_down.weight", "ffn_down_exps.weight");
            }
            if (std::regex_match(tensor_name, pattern_ffn_up_bias) ||
                    std::regex_match(tensor_name, pattern_ffn_gate_bias)) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "ffn_down.weight", "ffn_down_exps.weight");
            }
            if (std::regex_match(tensor_name, pattern_ffn_gate_up_weight)) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "ffn_down.weight", "ffn_down_exps.weight");
            }
            if (std::regex_match(tensor_name, pattern_ffn_down_weight)) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "ffn_down.weight", "ffn_down_exps.weight");
            }
            if (std::regex_match(tensor_name, pattern_ffn_down_bias)) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
            }
            if (std::regex_match(tensor_name, pattern_ffn_down_exps_bias)) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_PARTIAL);
            }
            // The vocabulary projection is naturally row-parallel: every
            // device consumes the same hidden state and produces a disjoint
            // vocabulary slice which the meta backend gathers for sampling.
            // Mirroring it makes all sockets redundantly read and compute the
            // full 154880-row GLM head on every target and MTP decode step.
            if (glm_output_tp && std::regex_match(tensor_name, pattern_output_weight)) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1);
            }
            if (glm_output_tp && std::regex_match(tensor_name, pattern_output_bias)) {
                return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0);
            }
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        }

        // standard attention
        if (std::regex_match(tensor_name, pattern_q_weight) || std::regex_match(tensor_name, pattern_kv_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "attn_output.weight", "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_q_bias) || std::regex_match(tensor_name, pattern_kv_bias)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "attn_output.weight", "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_qkv_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "attn_output.weight", "ssm_out.weight");
        }
        if ( std::regex_match(tensor_name, pattern_qkv_bias)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "attn_output.weight", "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_qk_norm)) {
            return get_tensor_config_impl(tensor->ne[1] == 1 ? GGML_BACKEND_SPLIT_AXIS_MIRRORED : GGML_BACKEND_SPLIT_AXIS_1, "attn_output.weight");
        }
        if (std::regex_match(tensor_name, pattern_kv_cache) || std::regex_match(tensor_name, pattern_attn_sinks)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "attn_output.weight");
        }
        if (std::regex_match(tensor_name, pattern_attn_out_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0);
        }
        if (std::regex_match(tensor_name, pattern_attn_out_bias)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        }

        if (std::regex_match(tensor_name, pattern_attn_gate_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "attn_output.weight", "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_ssm_dt) || std::regex_match(tensor_name, pattern_ssm_a)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_ssm_alpha) || std::regex_match(tensor_name, pattern_ssm_beta) ||
                std::regex_match(tensor_name, pattern_ssm_beta_alpha)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_r_cache) || std::regex_match(tensor_name, pattern_s_cache)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_ssm_conv1d)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "ssm_out.weight");
        }
        if (std::regex_match(tensor_name, pattern_ssm_out_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0);
        }

        // FFN
        if (std::regex_match(tensor_name, pattern_ffn_up_weight) || std::regex_match(tensor_name, pattern_ffn_gate_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "ffn_down.weight", "ffn_down_exps.weight");
        }
        if (std::regex_match(tensor_name, pattern_ffn_up_bias) || std::regex_match(tensor_name, pattern_ffn_gate_bias)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "ffn_down.weight", "ffn_down_exps.weight");
        }
        if (std::regex_match(tensor_name, pattern_ffn_gate_up_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1, "ffn_down.weight", "ffn_down_exps.weight");
        }
        if (std::regex_match(tensor_name, pattern_ffn_down_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0, "ffn_down.weight", "ffn_down_exps.weight");
        }
        if (std::regex_match(tensor_name, pattern_ffn_down_bias)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
        }
        if (std::regex_match(tensor_name, pattern_ffn_down_exps_bias)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_PARTIAL);
        }

        // output
        if (std::regex_match(tensor_name, pattern_output_weight)) {
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_1);
        }
        if (std::regex_match(tensor_name, pattern_output_bias)) {
            const ggml_tensor * output_weight = ud->model->get_tensor("output.weight");
            GGML_ASSERT(output_weight != nullptr);
            return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_0);
        }

        // everything else
        return get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
    };

    auto get_split_segments = [&](int axis, uint32_t il) -> std::vector<std::pair<int64_t, uint32_t>> {
        if (ud->model->arch == LLM_ARCH_QWEN3NEXT || ud->model->arch == LLM_ARCH_QWEN35 || ud->model->arch == LLM_ARCH_QWEN35MOE) {
            const int64_t head_k_dim = hparams.ssm_d_state;
            const int64_t head_v_dim = hparams.ssm_d_state;
            const int64_t n_k_heads  = hparams.ssm_n_group;
            const int64_t n_v_heads  = hparams.ssm_dt_rank;
            const int64_t key_dim    = head_k_dim * n_k_heads;
            const int64_t value_dim  = head_v_dim * n_v_heads;

            // both Qwen 3 Next and Qwen 3.5 support n_v_heads > n_k_heads but the broadcasting pattern is different:
            //   - Qwen 3 Next: [k0_v0, k0_v1, k1_v2, k1_v3] (this is the default split pattern)
            //   - Qwen 3.5:    [k0_v0, k1_v1, k0_v2, k1_v3] (needs segmenting of V on the scale of K to get the correct pattern)
            if (ud->model->arch == LLM_ARCH_QWEN3NEXT) {
                if (std::regex_match(tensor_name, pattern_qkv_weight) || std::regex_match(tensor_name, pattern_ssm_conv1d)) {
                    GGML_ASSERT(tensor->ne[axis] == 2*key_dim + value_dim);
                    return {{key_dim, 2}, {value_dim, 1}};
                }
            } else {
                const int64_t head_ratio = n_v_heads / n_k_heads;
                if (std::regex_match(tensor_name, pattern_qkv_weight) || std::regex_match(tensor_name, pattern_ssm_conv1d)) {
                    GGML_ASSERT(tensor->ne[axis] == 2*key_dim + value_dim);
                    return {{key_dim, 2 + head_ratio}};
                }
                if (std::regex_match(tensor_name, pattern_attn_gate_weight) || std::regex_match(tensor_name, pattern_ssm_out_weight)) {
                    return {{key_dim, head_ratio}};
                }
                if (std::regex_match(tensor_name, pattern_ssm_dt) || std::regex_match(tensor_name, pattern_ssm_a) ||
                        std::regex_match(tensor_name, pattern_ssm_alpha) || std::regex_match(tensor_name, pattern_ssm_beta)) {
                    return {{n_k_heads, head_ratio}};
                }
                if (std::regex_match(tensor_name, pattern_r_cache)) {
                    return {{key_dim * (hparams.ssm_d_conv - 1), 2 + head_ratio}};
                }
                if (std::regex_match(tensor_name, pattern_s_cache)) {
                    return {{n_k_heads * head_v_dim * head_v_dim, head_ratio}};
                }
            }

            // the FFN is the same for Qwen 3 Next and Qwen 3.5:
            if (std::regex_match(tensor_name, pattern_ffn_gate_up_weight)) {
                const int64_t n_ff_exp = hparams.n_ff_exp;
                GGML_ASSERT(tensor->ne[axis] == 2*n_ff_exp);
                return {{n_ff_exp, 2}};
            }
            return {{tensor->ne[axis], 1}};
        }

        if (std::regex_match(tensor_name, pattern_qkv_weight) || std::regex_match(tensor_name, pattern_qkv_bias)) {
            const int64_t n_embd      = hparams.n_embd;
            const int64_t n_embd_gqa  = hparams.n_embd_v_gqa(il);
            GGML_ASSERT(hparams.n_embd_k_gqa() == n_embd_gqa);
            GGML_ASSERT(tensor->ne[axis] == n_embd + 2*n_embd_gqa);
            return {{n_embd, 1}, {n_embd_gqa, 2}};
        }
        if (std::regex_match(tensor_name, pattern_ffn_up_weight) || std::regex_match(tensor_name, pattern_ffn_up_bias)) {
            const int64_t n_ff = hparams.n_ff(il);
            // some models such as Phi 3 have fused up + gate tensors named "up" tensors, which need to be segmented
            if (tensor->ne[axis] == 2*n_ff) {
                return {{n_ff, 2}};
            }
            return {{tensor->ne[axis], 1}};
        }
        if (std::regex_match(tensor_name, pattern_ffn_gate_up_weight)) {
            const int64_t n_ff_exp = hparams.n_ff_exp;
            GGML_ASSERT(tensor->ne[axis] == 2*n_ff_exp);
            return {{n_ff_exp, 2}};
        }
        return {{tensor->ne[axis], 1}};
    };

    auto get_split_granularity = [&](int64_t blck_size, uint32_t il, const std::vector<std::pair<int64_t, uint32_t>> & segments) -> std::vector<int64_t> {
        if (ud->model->arch == LLM_ARCH_GLM_DSA &&
                (std::regex_match(tensor_name, pattern_attn_q_a_weight) ||
                 std::regex_match(tensor_name, pattern_attn_kv_a_weight) ||
                 std::regex_match(tensor_name, pattern_ffn_shexp_gate_up_weight) ||
                 std::regex_match(tensor_name, pattern_ffn_shexp_down_weight))) {
            // Match shared gate/up to down, including K-quant repacking blocks.
            GGML_ASSERT(segments.size() == 1);
            return {std::lcm(blck_size, int64_t(256))};
        }

        if (ud->model->arch == LLM_ARCH_GLM_DSA && glm_attn_tp_tensor) {
            if (std::regex_match(tensor_name, pattern_attn_q_b_weight)) {
                GGML_ASSERT(segments.size() == 1);
                return {std::lcm(blck_size, tensor->ne[1] / hparams.n_head(il))};
            }
            if (std::regex_match(tensor_name, pattern_attn_kv_b_weight)) {
                GGML_ASSERT(segments.size() == 1);
                return {1};
            }
            if (std::regex_match(tensor_name, pattern_attn_out_weight)) {
                GGML_ASSERT(segments.size() == 1);
                return {std::lcm(blck_size, tensor->ne[0] / hparams.n_head(il))};
            }
        }

        // for better performance it may make sense to round up blck_size to a higher power of 2 so that more efficient kernels can be used
        if (hparams.is_recr(il)) {
            // linear attention
            const int64_t head_dim        = hparams.ssm_d_state;
            const int64_t blck_size_perf  = std::lcm(blck_size, 128);
            const int64_t granularity_qkv = std::lcm(blck_size_perf, head_dim);
            if (std::regex_match(tensor_name, pattern_qkv_weight) || std::regex_match(tensor_name, pattern_attn_gate_weight) ||
                    std::regex_match(tensor_name, pattern_ssm_conv1d) || std::regex_match(tensor_name, pattern_ssm_out_weight)) {
                return std::vector<int64_t>(segments.size(), granularity_qkv);
            }
            if (std::regex_match(tensor_name, pattern_ssm_dt) || std::regex_match(tensor_name, pattern_ssm_a) ||
                    std::regex_match(tensor_name, pattern_ssm_alpha) || std::regex_match(tensor_name, pattern_ssm_beta)) {
                return std::vector<int64_t>(segments.size(), granularity_qkv / head_dim);
            }
            if (std::regex_match(tensor_name, pattern_ssm_beta_alpha)) {
                return std::vector<int64_t>(segments.size(), 2 * (granularity_qkv / head_dim));
            }
            if (std::regex_match(tensor_name, pattern_r_cache)) {
                return std::vector<int64_t>(segments.size(), granularity_qkv * (hparams.ssm_d_conv - 1));
            }
            if (std::regex_match(tensor_name, pattern_s_cache)) {
                return std::vector<int64_t>(segments.size(), granularity_qkv * head_dim);
            }
        } else {
            // regular attention
            const uint32_t n_gqa    = hparams.n_gqa(il);
            const uint32_t n_embd_q = n_gqa * hparams.n_embd_head_k(il);

            // to handle head sizes like 80, only increase granularity while it doesn't cause underutilization
            int64_t blck_size_perf = blck_size;
            while (blck_size_perf < 128 && blck_size_perf*ud->n_devices < n_embd_q) {
                blck_size_perf *= 2;
            }

            if (std::regex_match(tensor_name, pattern_attn_sinks)) {
                GGML_ASSERT(segments.size() == 1);
                return {std::lcm(n_embd_q, blck_size_perf)/n_embd_q * n_gqa};
            }

            const int64_t granularity_q = std::lcm(n_embd_q, blck_size_perf);
            if (std::regex_match(tensor_name, pattern_q_weight) || std::regex_match(tensor_name, pattern_q_bias)) {
                GGML_ASSERT(segments.size() == 1);
                // some models have Q gate tensors, for those cases the granularity needs to be doubled:
                if (ud->model->arch == LLM_ARCH_QWEN3NEXT || ud->model->arch == LLM_ARCH_QWEN35 || ud->model->arch == LLM_ARCH_QWEN35MOE) {
                    return {std::lcm(2*n_embd_q, blck_size_perf)};
                }
                return {granularity_q};
            }
            if (std::regex_match(tensor_name, pattern_attn_out_weight)) {
                GGML_ASSERT(segments.size() == 1);
                return {granularity_q};
            }

            const int64_t granularity_kv = granularity_q / n_gqa;
            if (std::regex_match(tensor_name, pattern_kv_weight) ||
                std::regex_match(tensor_name, pattern_kv_bias) ||
                std::regex_match(tensor_name, pattern_kv_cache)) {
                GGML_ASSERT(segments.size() == 1);
                return {granularity_kv};
            }
            if (std::regex_match(tensor_name, pattern_qkv_weight) || std::regex_match(tensor_name, pattern_qkv_bias)) {
                GGML_ASSERT(segments.size() == 2);
                return {granularity_q, granularity_kv};
            }
        }

        // FFN
        if (std::regex_match(tensor_name, pattern_ffn_up_weight) || std::regex_match(tensor_name, pattern_ffn_up_bias) ||
                std::regex_match(tensor_name, pattern_ffn_gate_weight) || std::regex_match(tensor_name, pattern_ffn_gate_bias) ||
                std::regex_match(tensor_name, pattern_ffn_gate_up_weight) || std::regex_match(tensor_name, pattern_ffn_down_weight)) {
            const int64_t blck_size_perf = std::lcm(blck_size, 128);
            GGML_ASSERT(segments.size() == 1);
            return {blck_size_perf};
        }

        // everything else
        GGML_ASSERT(segments.size() == 1);
        return {1};
    };

    ggml_backend_meta_split_state split_state;
    memset(&split_state, 0, sizeof(split_state));
    tensor_config tc = get_tensor_config();
    split_state.axis = tc.axis;
    if (split_state.axis >= 0 && split_state.axis < GGML_MAX_DIMS) {
        const int64_t blck_size = ggml_blck_size(tc.tensor_axis_0->type);
        const float * tensor_split = ud->model->tensor_split();
        std::vector<float> tensor_split_scan;
        tensor_split_scan.reserve(ud->n_devices);
        for (size_t j = 0; j < ud->n_devices; j++) {
            tensor_split_scan.push_back(tensor_split == nullptr ? 0.0f : tensor_split[(j + tc.rotation) % ud->n_devices]);
            if (j > 0) {
                tensor_split_scan[j] += tensor_split_scan[j - 1];
            }
        }
        const std::vector<std::pair<int64_t, uint32_t>> segments = get_split_segments(split_state.axis, tc.il);
        const std::vector<int64_t> granularity = get_split_granularity(blck_size, tc.il, segments);
        for (size_t is = 0; is < segments.size(); is++) {
            const int64_t  ne_s = segments[is].first;
            const uint32_t nr_s = segments[is].second;
            const int64_t  g_s  = granularity[is];
            int64_t low = 0;
            size_t j = 0;
            for (; j < ud->n_devices - 1; j++) {
                int64_t high = tensor_split_scan.back() == 0.0f ?
                    ne_s * (j+1)/ud->n_devices : ne_s * tensor_split_scan[j]/tensor_split_scan.back();
                if (high % g_s != 0) {
                    high -= high % g_s;
                }
                split_state.ne[is*ud->n_devices + (j + tc.rotation) % ud->n_devices] = high - low;
                low = high;
            }
            split_state.ne[is*ud->n_devices + (j + tc.rotation) % ud->n_devices] = ne_s - low;
            split_state.nr[is] = nr_s;
        }
        split_state.n_segments = segments.size();
    } else {
        memset(split_state.ne, 0, sizeof(split_state.ne));
        split_state.nr[0] = 1;
        split_state.n_segments = 1;
    }
    return split_state;
    GGML_UNUSED(userdata);
}
