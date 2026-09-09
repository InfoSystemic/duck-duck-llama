#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

static void report_threads(ggml_tensor * output, int ith, int nth, void *) {
    for (int64_t i = ith; i < ggml_nelements(output); i += nth) {
        ((float *) output->data)[i] = float(nth);
    }
}

int main(int argc, char ** argv) {
    if (argc != 2) { return 2; }
    const std::string path = argv[1];
    setenv("GGML_CPU_NUMA_DEVICES", "1", 1);
    setenv("GGML_CPU_NUMA_THREADS", "16", 1);
    setenv("GGML_CPU_NUMA_THREADS_FILE", path.c_str(), 1);
    ggml_backend_load_all();
    int failures = 0;
    int cases = 0;
    for (int node = 0; node < 4; ++node) {
        const std::string name = "CPU-NUMA" + std::to_string(node);
        auto device = ggml_backend_dev_by_name(name.c_str());
        if (!device) { return 2; }
        auto backend = ggml_backend_dev_init(device, nullptr);
        auto ctx = ggml_init({8*1024*1024, nullptr, false});
        auto weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 512);
        auto input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 3);
        for (int64_t i = 0; i < ggml_nelements(weights); ++i) {
            ((float *) weights->data)[i] = std::sin(float(i)*0.017f)*0.02f;
        }
        for (int64_t i = 0; i < ggml_nelements(input); ++i) {
            ((float *) input->data)[i] = std::cos(float(i)*0.023f);
        }
        auto product = ggml_silu(ctx, ggml_mul_mat(ctx, weights, input));
        ggml_tensor * args[] = {product};
        auto observed = ggml_custom_4d(ctx, GGML_TYPE_F32, 16, 1, 1, 1,
            args, 1, report_threads, GGML_N_TASKS_MAX, nullptr);
        auto graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, observed);
        std::vector<float> reference;
        for (const std::string setting : {"16", "15", "12", "8", "1", "16", "0", "32", "bad", "missing"}) {
            if (setting == "missing") {
                std::remove(path.c_str());
            } else {
                std::ofstream file(path);
                file << setting << '\n';
            }
            const int requested = std::atoi(setting.c_str());
            const int expected = requested > 0 ? std::min(16, requested) : 16;
            bool ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
            ggml_backend_synchronize(backend);
            for (int i = 0; i < 16; ++i) {
                ok = ok && ((float *) observed->data)[i] == float(expected);
            }
            auto data = (const float *) product->data;
            if (reference.empty()) {
                reference.assign(data, data + ggml_nelements(product));
            }
            float max_error = 0;
            for (size_t i = 0; i < reference.size(); ++i) {
                const float error = std::fabs(data[i] - reference[i]);
                ok = ok && std::isfinite(data[i]) && error <= 2e-4f*(1 + std::fabs(reference[i]));
                max_error = std::max(max_error, error);
            }
            std::printf("%s node=%d setting=%s expected=%d observed=%.0f max_error=%g\n",
                ok ? "PASS" : "FAIL", node, setting.c_str(), expected,
                ((float *) observed->data)[0], max_error);
            failures += !ok;
            ++cases;
        }
        ggml_backend_free(backend);
        ggml_free(ctx);
    }
    std::printf("cases=%d failures=%d\n", cases, failures);
    return failures ? 1 : 0;
}
