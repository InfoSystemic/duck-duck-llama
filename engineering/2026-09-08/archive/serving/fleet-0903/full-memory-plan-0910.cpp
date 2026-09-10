#include "llama.h"
#include "llama-ext.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <sys/resource.h>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: full-memory-plan LIBRARY_DIRECTORY MODEL\n");
        return 2;
    }
    ggml_backend_load_all_from_path(argv[1]);
    llama_backend_init();
    std::vector<ggml_backend_dev_t> devices;
    for (const char * name : {"CPU-NUMA0", "CPU-NUMA1", "CPU-NUMA2", "CPU-NUMA3"}) {
        auto * dev = ggml_backend_dev_by_name(name);
        if (!dev) {
            std::fprintf(stderr, "missing device %s\n", name);
            return 3;
        }
        devices.push_back(dev);
    }
    devices.push_back(nullptr);
    std::vector<float> split(llama_max_devices(), 0.0f);
    for (int i = 0; i < 4; ++i) {
        split[i] = 1.0f;
    }
    auto mp = llama_model_default_params();
    mp.devices = devices.data();
    mp.tensor_split = split.data();
    mp.n_gpu_layers = 999;
    mp.split_mode = LLAMA_SPLIT_MODE_TENSOR;
    mp.load_mode = LLAMA_LOAD_MODE_NONE;
    mp.no_alloc = true;
    mp.load_mtp = false;
    auto * model = llama_model_load_from_file(argv[2], mp);
    if (!model) {
        return 4;
    }
    auto cp = llama_context_default_params();
    cp.n_ctx = 32768;
    cp.n_batch = 512;
    cp.n_ubatch = 256;
    cp.n_seq_max = 1;
    cp.n_threads = 15;
    cp.n_threads_batch = 15;
    cp.type_k = GGML_TYPE_Q8_0;
    cp.type_v = GGML_TYPE_Q8_0;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.no_perf = true;
    auto * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        llama_model_free(model);
        return 5;
    }
    const auto breakdown = llama_get_memory_breakdown(ctx);
    llama_memory_breakdown_data total;
    std::printf("{\"event\":\"full_memory_plan\",\"no_alloc\":true,\"load_mtp\":false,\"context\":32768,\"buffers\":[");
    bool first = true;
    for (const auto & [buft, mb] : breakdown) {
        const char * name = ggml_backend_buft_name(buft);
        if (std::strpbrk(name, "\"\\\n\r")) {
            throw std::runtime_error("unexpected buffer name");
        }
        std::printf("%s{\"name\":\"%s\",\"model_bytes\":%zu,\"context_bytes\":%zu,\"compute_bytes\":%zu}",
                    first ? "" : ",", name, mb.model, mb.context, mb.compute);
        first = false;
        total.model += mb.model;
        total.context += mb.context;
        total.compute += mb.compute;
    }
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return 6;
    }
    std::printf("],\"model_bytes\":%zu,\"context_bytes\":%zu,\"compute_bytes\":%zu,\"total_bytes\":%zu,\"peak_rss_bytes\":%zu}\n",
                total.model, total.context, total.compute, total.total(), size_t(usage.ru_maxrss) * 1024);
    std::fflush(stdout);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
