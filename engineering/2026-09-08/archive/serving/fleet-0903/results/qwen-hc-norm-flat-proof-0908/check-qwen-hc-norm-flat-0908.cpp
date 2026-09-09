#include "qwen-hc-norm-flat-0908.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-impl.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <random>
#include <sched.h>
#include <string>
#include <vector>

static ggml_backend_meta_split_state mirrored(const ggml_tensor *, void *) {
    return {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
}

int main(int argc, char ** argv) {
    if (argc != 2 || (std::string(argv[1]) != "cpu" && std::string(argv[1]) != "numa")) return 2;
    const bool numa = std::string(argv[1]) == "numa";
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0, sizeof(mask), &mask) || CPU_COUNT(&mask) != 128) return 2;
    ggml_backend_load_all();
    ggml_backend_t backend = nullptr;
    if (numa) {
        std::vector<ggml_backend_dev_t> devices;
        for (int i = 0; i < 4; ++i) {
            auto device = ggml_backend_dev_by_name(("CPU-NUMA" + std::to_string(i)).c_str());
            if (!device) return 3;
            devices.push_back(device);
        }
        backend = ggml_backend_dev_init(ggml_backend_meta_device(devices.data(), devices.size(), mirrored, nullptr), nullptr);
    } else {
        backend = ggml_backend_cpu_init();
    }
    if (!backend) return 3;
    Dl_info cpu_library{}, base_library{};
    if (!dladdr(reinterpret_cast<void *>(ggml_backend_cpu_init), &cpu_library) ||
        !dladdr(reinterpret_cast<void *>(ggml_rms_norm), &base_library)) return 3;
    std::mt19937 random(419875);
    size_t cases = 0, outputs = 0, eligible_cases = 0, fallback_cases = 0;
    uint64_t hash = 14695981039346656037ULL;
    const std::vector<int> workers = numa ? std::vector<int>{15} : std::vector<int>{1,4,15};
    for (int nth : workers) {
        if (!numa) ggml_backend_cpu_set_n_threads(backend, nth);
        for (int width : {256,257,2560}) for (int streams : {1,4})
        for (int tokens : {1,3,5,64}) for (int pad : {0,7}) {
            auto inputs = ggml_init({1024 * 1024, nullptr, true});
            auto context = ggml_init({2 * 1024 * 1024, nullptr, true});
            auto xb = ggml_new_tensor_3d(inputs, GGML_TYPE_F32, width + pad, streams, tokens);
            auto wb = ggml_new_tensor_1d(inputs, GGML_TYPE_F32, width * streams + 7);
            ggml_set_input(xb); ggml_set_input(wb);
            auto input_buffer = ggml_backend_alloc_ctx_tensors(inputs, backend);
            if (!input_buffer) return 4;
            auto x = ggml_view_3d(context, xb, width, streams, tokens, xb->nb[1], xb->nb[2], 0);
            auto weight = ggml_view_1d(context, wb, width * streams, 0);
            const float eps = 1.0e-6f;
            auto original_norm = ggml_rms_norm(context, x, eps);
            auto original = ggml_mul(context, ggml_reshape_2d(context, original_norm, width * streams, tokens), weight);
            auto graph = ggml_new_graph_custom(context, 128, false);
            ggml_build_forward_expand(graph, original);
            auto candidate = qwen_hc_norm_flat(context, graph, x, weight, eps);
            ggml_set_output(original); ggml_set_output(candidate);
            ggml_build_forward_expand(graph, candidate);
            if (pad == 0) {
                auto multiply = candidate->src[0];
                auto norm = multiply->src[0];
                if (candidate->op != GGML_OP_RESHAPE || multiply->op != GGML_OP_MUL ||
                    norm->op != GGML_OP_RMS_NORM || norm->ne[1] != streams * tokens || norm->ne[2] != 1) return 5;
                int node_index = -1;
                for (int i = 0; i < ggml_graph_n_nodes(graph); ++i)
                    if (ggml_graph_node(graph, i) == norm) node_index = i;
                if (node_index < 0 || !ggml_can_fuse(graph, node_index, {GGML_OP_RMS_NORM, GGML_OP_MUL})) return 5;
            }
            auto allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            if (!ggml_gallocr_alloc_graph(allocator, graph)) return 4;
            std::vector<float> xv(ggml_nelements(xb)), wv(ggml_nelements(wb));
            std::vector<float> reference(size_t(width) * streams * tokens), actual(reference.size());
            for (int probe = 0; probe < 4; ++probe) {
                for (auto & value : xv) value = probe == 3 ? -0.0f : float(int(random() % 65536) - 32768) * (probe == 2 ? 123456.75f : .00123f);
                for (auto & value : wv) value = probe == 1 ? float(random() & 1) : float(int(random() % 65536) - 32768) / 16384.f;
                ggml_backend_tensor_set(xb, xv.data(), 0, ggml_nbytes(xb));
                ggml_backend_tensor_set(wb, wv.data(), 0, ggml_nbytes(wb));
                if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return 6;
                ggml_backend_synchronize(backend);
                ggml_backend_tensor_get(original, reference.data(), 0, ggml_nbytes(original));
                ggml_backend_tensor_get(candidate, actual.data(), 0, ggml_nbytes(candidate));
                if (std::memcmp(reference.data(), actual.data(), actual.size() * sizeof(float))) {
                    std::fprintf(stderr, "graph mismatch width=%d streams=%d tokens=%d pad=%d probe=%d\n", width, streams, tokens, pad, probe);
                    return 7;
                }
                for (int token = 0; token < tokens; ++token) for (int stream = 0; stream < streams; ++stream) {
                    const auto * row = xv.data() + size_t(token * streams + stream) * (width + pad);
                    volatile double sum = 0.0;
                    for (int j = 0; j < width; ++j) {
                        volatile float square = row[j] * row[j];
                        sum = sum + double(square);
                    }
                    const float mean = sum / width;
                    const float scale = 1.0f / std::sqrt(mean + eps);
                    for (int j = 0; j < width; ++j) {
                        volatile float normalized = row[j] * scale;
                        const float expected = normalized * wv[stream * width + j];
                        if (std::memcmp(&expected, &actual[(size_t(token) * streams + stream) * width + j], sizeof(float))) return 8;
                    }
                }
                for (auto pair : {std::make_pair(xb, &xv), std::make_pair(wb, &wv)}) {
                    std::vector<float> after(pair.second->size());
                    ggml_backend_tensor_get(pair.first, after.data(), 0, ggml_nbytes(pair.first));
                    if (std::memcmp(after.data(), pair.second->data(), after.size() * sizeof(float))) return 9;
                }
                const auto * bytes = reinterpret_cast<const unsigned char *>(actual.data());
                for (size_t i = 0; i < actual.size() * sizeof(float); ++i) hash = (hash ^ bytes[i]) * 1099511628211ULL;
                outputs += actual.size();
                ++cases;
                pad ? ++fallback_cases : ++eligible_cases;
            }
            ggml_gallocr_free(allocator);
            ggml_backend_buffer_free(input_buffer);
            ggml_free(context); ggml_free(inputs);
        }
    }
    std::printf("{\"passed\":true,\"bit_exact\":true,\"scalar_exact\":true,\"inputs_preserved\":true,\"mode\":\"%s\",\"cases\":%zu,\"outputs\":%zu,\"fusion_eligible_cases\":%zu,\"padded_fallback_cases\":%zu,\"hash\":\"%016llx\",\"cpu_library\":\"%s\",\"base_library\":\"%s\"}\n", argv[1], cases, outputs, eligible_cases, fallback_cases, (unsigned long long) hash, cpu_library.dli_fname, base_library.dli_fname);
    ggml_backend_free(backend);
}
