#include "ggml.h"
#include "ggml-backend.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

enum llm_arch { LLM_ARCH_GLM_DSA, LLM_ARCH_QWEN3NEXT, LLM_ARCH_QWEN35, LLM_ARCH_QWEN35MOE };
struct llama_hparams {
    int64_t layers, n_embd, ff, n_ff_exp, heads, kv_heads, head_k, head_v;
    int64_t ssm_d_state = 0, ssm_n_group = 0, ssm_dt_rank = 0, ssm_d_conv = 0;
    bool is_recr(size_t) const { return false; }
    bool is_swa(size_t) const { return false; }
    int64_t n_layer() const { return layers; }
    int64_t n_head(uint32_t) const { return heads; }
    int64_t n_ff(uint32_t) const { return ff; }
    int64_t n_gqa(uint32_t) const { return heads / kv_heads; }
    int64_t n_embd_head_k(uint32_t) const { return head_k; }
    int64_t n_embd_v_gqa(uint32_t) const { return kv_heads * head_v; }
    int64_t n_embd_k_gqa() const { return kv_heads * head_k; }
};
struct llama_model {
    llm_arch arch = LLM_ARCH_GLM_DSA;
    llama_hparams hparams;
    std::map<std::string, ggml_tensor> tensors;
    std::array<float, 4> split;
    const float * tensor_split() const { return split.data(); }
    const ggml_tensor * get_tensor(const char * name) const {
        const auto found = tensors.find(name);
        return found == tensors.end() ? nullptr : &found->second;
    }
};
struct llama_meta_device_get_split_state_userdata {
    size_t n_devices;
    const llama_model * model;
};

#include "split-callbacks.generated.inc"

static void check(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}

static bool valid(const ggml_tensor & tensor, const ggml_backend_meta_split_state & state) {
    if (state.axis < 0 || state.axis >= GGML_MAX_DIMS) return true;
    const int64_t granularity = state.axis == GGML_BACKEND_SPLIT_AXIS_0 ? ggml_blck_size(tensor.type) : 1;
    int64_t total = 0;
    for (uint32_t segment = 0; segment < state.n_segments; ++segment) {
        for (size_t device = 0; device < 4; ++device) {
            const int64_t n = state.ne[segment * 4 + device];
            if (n < 0 || n % granularity != 0) return false;
            total += n * state.nr[segment];
        }
    }
    return total == tensor.ne[state.axis];
}

static bool same_slices(const ggml_backend_meta_split_state & a, const ggml_backend_meta_split_state & b) {
    if (a.n_segments != b.n_segments) return false;
    for (uint32_t s = 0; s < a.n_segments; ++s) {
        if (a.nr[s] != b.nr[s]) return false;
        for (size_t d = 0; d < 4; ++d) if (a.ne[s*4+d] != b.ne[s*4+d]) return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    check(argc == 2, "provide tensor metadata TSV");
    setenv("GGML_GLM_ATTN_TP", "1", 1);
    setenv("GGML_GLM_ATTN_TP_LAYERS", "2147483647", 1);
    setenv("GGML_GLM_OUTPUT_TP", "1", 1);
    setenv("GGML_GLM_SHEXP_TP", "1", 1);
    setenv("GGML_GLM_QA_TP", "2", 1);
    llama_model model;
    std::ifstream input(argv[1]);
    auto & h = model.hparams;
    input >> h.layers >> h.n_embd >> h.ff >> h.n_ff_exp >> h.heads >> h.kv_heads >> h.head_k >> h.head_v;
    check(bool(input), "invalid hparams header");
    std::string name;
    int type;
    int64_t n0, n1, n2, n3;
    while (input >> name >> type >> n0 >> n1 >> n2 >> n3) {
        check(name.size() < GGML_MAX_NAME, "tensor name exceeds GGML_MAX_NAME");
        ggml_tensor tensor{};
        std::strcpy(tensor.name, name.c_str());
        tensor.type = ggml_type(type);
        tensor.ne[0] = n0; tensor.ne[1] = n1; tensor.ne[2] = n2; tensor.ne[3] = n3;
        check(model.tensors.emplace(name, tensor).second, "duplicate tensor");
    }
    check(!model.tensors.empty(), "no tensors");
    llama_meta_device_get_split_state_userdata ud{4, &model};
    std::vector<std::array<float, 4>> ratios{{1,1,1,1}, {1.41f,1.43f,1.43f,1.00f},
        {0,0,0,0}, {1,0,0,0}, {0,1,0,1}, {1,2,3,4}, {.001f,100.f,.01f,.1f}};
    std::mt19937 random(530912);
    std::uniform_real_distribution<float> ratio(.01f, 3.f);
    for (int i = 0; i < 25; ++i) ratios.push_back({ratio(random),ratio(random),ratio(random),ratio(random)});
    size_t balanced_checked = 0, total_checked = 0, shared_checked = 0;
    size_t parent_bad = 0, requant_alignment_checked = 0, attention_heads_checked = 0;
    std::string first_bad;
    for (size_t scenario = 0; scenario < ratios.size(); ++scenario) {
        model.split = ratios[scenario];
        std::map<std::string, ggml_backend_meta_split_state> states;
        for (const auto & entry : model.tensors) {
            const auto & tensor = entry.second;
            auto parent = split_parent(&tensor, &ud);
            auto candidate = split_candidate(&tensor, &ud);
            check(valid(tensor, candidate), "candidate invalid split: " + entry.first);
            states.emplace(entry.first, candidate);
            ++total_checked;
            if (scenario == 0) {
                check(parent.axis == candidate.axis && same_slices(parent, candidate), "balanced split changed: " + entry.first);
                ++balanced_checked;
            }
            if (scenario == 1 && !valid(tensor, parent)) {
                ++parent_bad;
                if (first_bad.empty()) first_bad = entry.first;
            }
            if (candidate.axis == 0 && (entry.first.find("attn_q_a.weight") != std::string::npos ||
                    entry.first.find("attn_kv_a_mqa.weight") != std::string::npos ||
                    entry.first.find("ffn_down_shexp.weight") != std::string::npos)) {
                for (size_t d = 0; d < 4; ++d) check(candidate.ne[d] % 256 == 0, "K-quant repack alignment");
                ++requant_alignment_checked;
            }
        }
        for (const auto & entry : model.tensors) {
            const auto suffix = entry.first.find("ffn_down_shexp.weight");
            if (suffix != std::string::npos) {
                const std::string prefix = entry.first.substr(0, suffix);
                const auto & down = states.at(entry.first);
                const auto & gate = states.at(prefix + "ffn_gate_shexp.weight");
                const auto & up = states.at(prefix + "ffn_up_shexp.weight");
                check(down.axis == 0 && gate.axis == 1 && up.axis == 1, "shared expert axes");
                check(same_slices(down, gate) && same_slices(down, up), "shared expert channel boundaries");
                ++shared_checked;
            }
            const auto out_suffix = entry.first.find("attn_output.weight");
            if (out_suffix != std::string::npos) {
                const std::string prefix = entry.first.substr(0, out_suffix);
                const auto & out = states.at(entry.first);
                const auto & q = states.at(prefix + "attn_q_b.weight");
                const auto & key = states.at(prefix + "attn_k_b.weight");
                const auto & value = states.at(prefix + "attn_v_b.weight");
                const int64_t out_head = entry.second.ne[0] / h.heads;
                const int64_t q_head = model.tensors.at(prefix + "attn_q_b.weight").ne[1] / h.heads;
                for (size_t d = 0; d < 4; ++d) {
                    check(out.ne[d] % out_head == 0 && q.ne[d] % q_head == 0, "partial attention head");
                    check(out.ne[d]/out_head == q.ne[d]/q_head && key.ne[d] == value.ne[d] &&
                          key.ne[d] == out.ne[d]/out_head, "attention head split mismatch");
                }
                ++attention_heads_checked;
            }
        }
    }
    check(parent_bad > 0 && shared_checked > 0, "regression not exercised");
    std::cout << "{\"passed\":true,\"tensor_count\":" << model.tensors.size()
              << ",\"ratio_scenarios\":" << ratios.size() << ",\"tensor_checks\":" << total_checked
              << ",\"balanced_unchanged\":" << balanced_checked
              << ",\"shared_expert_triplet_checks\":" << shared_checked
              << ",\"attention_head_consistency_checks\":" << attention_heads_checked
              << ",\"requant_256_alignment_checks\":" << requant_alignment_checked
              << ",\"parent_unequal_invalid_axis0\":" << parent_bad
              << ",\"first_parent_invalid_tensor\":\"" << first_bad << "\"}\n";
}
