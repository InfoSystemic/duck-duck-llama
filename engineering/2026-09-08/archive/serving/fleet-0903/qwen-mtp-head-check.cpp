#include "llama.h"
#include "llama-ext.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// Check real sidecar loading, multi-token hidden input, and the next draft step.
// Synthetic hidden states test execution and TP equivalence, not draft quality.
int main(int argc, char ** argv) {
    if (argc != 4) return 2;
    const int devices_n = std::atoi(argv[3]);
    ggml_backend_load_all();
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.load_mtp = true;
    const char * repack = std::getenv("GGML_CPU_NUMA_REPACK");
    mp.use_extra_bufts = repack && std::atoi(repack) != 0;
    std::vector<ggml_backend_dev_t> devices;
    if (devices_n > 0) {
        for (int i = 0; i < devices_n; ++i) {
            const auto name = "CPU-NUMA" + std::to_string(i);
            auto dev = ggml_backend_dev_by_name(name.c_str());
            if (!dev) return 2;
            devices.push_back(dev);
        }
        devices.push_back(nullptr);
        mp.devices = devices.data();
        mp.n_gpu_layers = 999;
        mp.split_mode = LLAMA_SPLIT_MODE_TENSOR;
    } else {
        mp.n_gpu_layers = 0;
    }
    auto model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 3;
    auto cp = llama_context_default_params();
    cp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    cp.n_ctx = 128;
    cp.n_batch = 8;
    cp.n_ubatch = 8;
    cp.n_threads = 1;
    cp.n_threads_batch = 1;
    cp.embeddings = llama_model_n_embd_out(model) == llama_model_n_embd(model);
    cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    auto ctx = llama_init_from_model(model, cp);
    if (!ctx) return 4;
    llama_set_embeddings_nextn(ctx, true, true);
    const int width = llama_model_n_embd_out(model);
    const int vocab_n = llama_vocab_n_tokens(llama_model_get_vocab(model));
    auto batch = llama_batch_init(8, width, 1);
    batch.token = static_cast<llama_token *>(std::malloc(8 * sizeof(llama_token)));
    auto out = std::fopen(argv[2], "wb");
    if (!out) return 5;
    std::vector<float> hidden(width);
    bool finite = true;
    for (int step = 0; step < 2; ++step) {
        batch.n_tokens = step == 0 ? 4 : 1;
        for (int j = 0; j < batch.n_tokens; ++j) {
            batch.token[j] = 100 + j + step;
            batch.pos[j] = step == 0 ? j : 4;
            batch.n_seq_id[j] = 1;
            batch.seq_id[j][0] = 0;
            batch.logits[j] = true;
            for (int k = 0; k < width; ++k) {
                batch.embd[j * width + k] = step == 0 ? std::sin(float(k + 17 * j) * 0.03f) : hidden[k];
            }
        }
        if (llama_decode(ctx, batch)) return 6;
        for (int j = 0; j < batch.n_tokens; ++j) {
            const auto logits = llama_get_logits_ith(ctx, j);
            const auto h = llama_get_embeddings_nextn_ith(ctx, j);
            if (!logits || !h) return 7;
            bool hidden_nonzero = false;
            for (int k = 0; k < vocab_n; ++k) finite &= std::isfinite(logits[k]);
            for (int k = 0; k < width; ++k) {
                finite &= std::isfinite(h[k]);
                hidden_nonzero |= h[k] != 0.0f;
            }
            finite &= hidden_nonzero;
            if (width == llama_model_n_embd(model)) {
                const auto expected_h = llama_get_embeddings_ith(ctx, j);
                if (!expected_h) return 9;
                for (int k = 0; k < width; ++k) {
                    finite &= std::abs(h[k] - expected_h[k]) <= 1e-6f;
                }
            }
            std::fwrite(logits, sizeof(float), vocab_n, out);
            std::fwrite(h, sizeof(float), width, out);
            hidden.assign(h, h + width);
        }
    }
    std::fclose(out);
    std::printf("MTP head devices=%d width=%d vocab=%d finite_and_hidden=%s rows=5\n", devices_n, width, vocab_n, finite ? "PASS" : "FAIL");
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return finite ? 0 : 8;
}
