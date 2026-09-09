#include "llama-model.h"
#include "ggml.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <memory>
#include <string>
#include <vector>

int main() {
    setenv("GGML_Q4E_SPLIT", "13", 1);
    setenv("GGML_Q4E_HC_TP", "1", 1);
    auto params = llama_model_default_params();
    params.split_mode = LLAMA_SPLIT_MODE_TENSOR;
    std::vector<float> fractions(llama_max_devices(), 0.0f);
    std::fill_n(fractions.begin(), 4, 1.0f);
    params.tensor_split = fractions.data();
    std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_QWEN4EXP, params));
    if (!model || model->arch != LLM_ARCH_QWEN4EXP) return 2;
    auto & h = model->hparams;
    h.n_layer_all = 48;
    h.n_embd = 2560;
    h.n_expert = 512;
    h.n_expert_used = 10;
    h.n_ff_exp = 640;
    h.n_embd_head_k_full = h.n_embd_head_v_full = 256;
    h.n_head_arr.fill(24);
    h.n_head_kv_arr.fill(2);
    h.is_swa_impl.fill(0);
    h.is_ple_impl.reset();
    h.set_recr_pattern(4, false);
    h.ssm_d_state = 128;
    h.ssm_n_group = 16;
    h.ssm_dt_rank = 48;
    auto * context = ggml_init({160 * ggml_tensor_overhead(), nullptr, true});
    if (!context) return 2;
    for (int layer = 0; layer < 48; ++layer) {
        for (const char * kind : {"gate", "up", "down"}) {
            const bool down = std::string(kind) == "down";
            const auto type = down ? GGML_TYPE_IQ4_NL : (layer == 2 ? GGML_TYPE_IQ3_XXS : GGML_TYPE_IQ2_XS);
            auto * tensor = ggml_new_tensor_3d(context, type,
                                               down ? 640 : 2560, down ? 2560 : 640, 512);
            const std::string name = "blk." + std::to_string(layer) + ".ffn_" + kind + "_exps.weight";
            ggml_set_name(tensor, name.c_str());
            model->tensors_by_name.emplace_back(name, tensor);
        }
    }
    Dl_info library = {};
    if (!dladdr(reinterpret_cast<void *>(llama_meta_device_get_split_state), &library)) return 2;
    std::printf("{\"event\":\"library\",\"path\":\"%s\",\"metadata_only\":true}\n", library.dli_fname);
    llama_meta_device_get_split_state_userdata user{4, model.get()};
    for (const auto & item : model->tensors_by_name) {
        const auto split = llama_meta_device_get_split_state(item.second, &user);
        if (split.n_segments != 1 || split.nr[0] != 1) return 2;
        int64_t sum = 0;
        std::printf("{\"event\":\"split\",\"tensor\":\"%s\",\"axis\":%d,\"slices\":[", item.first.c_str(), int(split.axis));
        for (int socket = 0; socket < 4; ++socket) {
            if (split.ne[socket] <= 0) return 2;
            sum += split.ne[socket];
            std::printf("%s%lld", socket ? "," : "", (long long) split.ne[socket]);
        }
        if (sum != 640) return 2;
        std::printf("]}\n");
    }
    model.reset();
    ggml_free(context);
    return 0;
}
