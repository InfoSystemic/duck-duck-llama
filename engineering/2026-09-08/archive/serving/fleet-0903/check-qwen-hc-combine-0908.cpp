#include "qwen-hc-combine-0908.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
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

static uint64_t hash_values(const std::vector<float> & values, uint64_t hash) {
    const auto * data = reinterpret_cast<const unsigned char *>(values.data());
    for (size_t i = 0; i < values.size() * sizeof(float); ++i) hash = (hash ^ data[i]) * 1099511628211ULL;
    return hash;
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
        for (int j = 0; j < 4; ++j) {
            auto dev = ggml_backend_dev_by_name(("CPU-NUMA" + std::to_string(j)).c_str());
            if (!dev) return 3;
            devices.push_back(dev);
        }
        auto meta = ggml_backend_meta_device(devices.data(), devices.size(), mirrored, nullptr);
        backend = ggml_backend_dev_init(meta, nullptr);
    } else {
        backend = ggml_backend_cpu_init();
    }
    if (!backend) return 3;
    Dl_info cpu_library{}, base_library{};
    if (!dladdr(reinterpret_cast<void *>(ggml_backend_cpu_init), &cpu_library) ||
        !dladdr(reinterpret_cast<void *>(ggml_map_custom3), &base_library)) return 3;
    std::atomic<uint64_t> calls{0};
    std::mt19937 random(982173);
    size_t cases = 0, comparisons = 0;
    uint64_t hash = 14695981039346656037ULL;
    const std::vector<int> workers = numa ? std::vector<int>{15} : std::vector<int>{1,4,15};
    for (int nth : workers) {
        if (!numa) ggml_backend_cpu_set_n_threads(backend, nth);
        for (int width : {257,2560}) for (int streams : {1,4})
        for (int tokens : {1,3,5,64}) for (int pad : {0,7}) {
            auto inputs = ggml_init({1024 * 1024, nullptr, true});
            auto context = ggml_init({2 * 1024 * 1024, nullptr, true});
            auto rb = ggml_new_tensor_3d(inputs, GGML_TYPE_F32, width + pad, streams, tokens);
            auto bb = ggml_new_tensor_3d(inputs, GGML_TYPE_F32, width + pad, 1, tokens);
            auto wb = ggml_new_tensor_3d(inputs, GGML_TYPE_F32, 1 + pad, streams, tokens);
            ggml_set_input(rb); ggml_set_input(bb); ggml_set_input(wb);
            ggml_set_name(rb, "residual"); ggml_set_name(bb, "block"); ggml_set_name(wb, "scatter");
            auto ib = ggml_backend_alloc_ctx_tensors(inputs, backend);
            if (!ib) return 4;
            auto r = ggml_view_3d(context, rb, width, streams, tokens, rb->nb[1], rb->nb[2], 0);
            auto b = ggml_view_3d(context, bb, width, 1, tokens, bb->nb[1], bb->nb[2], 0);
            auto w = ggml_view_3d(context, wb, 1, streams, tokens, wb->nb[1], wb->nb[2], 0);
            auto repeated = ggml_repeat_4d(context, b, width, streams, tokens, 1);
            auto baseline = ggml_add(context, r, ggml_mul(context, repeated, w));
            auto candidate = ggml_map_custom3(context, r, b, w, qwen_hc_combine_f32, GGML_N_TASKS_MAX, &calls);
            ggml_set_output(baseline); ggml_set_output(candidate);
            auto graph = ggml_new_graph_custom(context, 128, false);
            ggml_build_forward_expand(graph, baseline);
            ggml_build_forward_expand(graph, candidate);
            auto allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            if (!ggml_gallocr_alloc_graph(allocator, graph)) return 4;
            std::vector<float> rv(ggml_nelements(rb)), bv(ggml_nelements(bb)), wv(ggml_nelements(wb));
            std::vector<float> expected(size_t(width) * streams * tokens), actual(expected.size());
            for (int probe = 0; probe < 3; ++probe) {
                for (auto & v : rv) v = float(int(random() % 65536) - 32768) * (probe == 2 ? 123456.75f : .00123f);
                for (auto & v : bv) v = float(int(random() % 65536) - 32768) * (probe == 2 ? 123456.75f : .000917f);
                for (auto & v : wv) v = probe == 1 ? (random() & 1 ? 0.f : 2.f) : float(random() % 65536) / 32768.f;
                ggml_backend_tensor_set(rb, rv.data(), 0, ggml_nbytes(rb));
                ggml_backend_tensor_set(bb, bv.data(), 0, ggml_nbytes(bb));
                ggml_backend_tensor_set(wb, wv.data(), 0, ggml_nbytes(wb));
                if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return 5;
                ggml_backend_synchronize(backend);
                ggml_backend_tensor_get(baseline, expected.data(), 0, ggml_nbytes(baseline));
                ggml_backend_tensor_get(candidate, actual.data(), 0, ggml_nbytes(candidate));
                if (std::memcmp(expected.data(), actual.data(), actual.size() * sizeof(float))) return 6;
                for (int t = 0; t < tokens; ++t) for (int s = 0; s < streams; ++s) for (int j = 0; j < width; ++j) {
                    volatile float product = bv[size_t(t) * (width + pad) + j] * wv[(size_t(t) * streams + s) * (1 + pad)];
                    const float reference = rv[(size_t(t) * streams + s) * (width + pad) + j] + product;
                    const size_t index = (size_t(t) * streams + s) * width + j;
                    if (std::memcmp(&reference, &actual[index], sizeof(float))) return 7;
                }
                for (auto pair : {std::make_pair(rb, &rv), std::make_pair(bb, &bv), std::make_pair(wb, &wv)}) {
                    std::vector<float> after(pair.second->size());
                    ggml_backend_tensor_get(pair.first, after.data(), 0, ggml_nbytes(pair.first));
                    if (std::memcmp(after.data(), pair.second->data(), after.size() * sizeof(float))) return 8;
                }
                comparisons += actual.size();
                hash = hash_values(actual, hash);
                ++cases;
            }
            ggml_gallocr_free(allocator);
            ggml_backend_buffer_free(ib);
            ggml_free(context); ggml_free(inputs);
        }
    }
    if (!calls.load()) return 9;
    std::printf("{\"passed\":true,\"bit_exact\":true,\"scalar_exact\":true,\"inputs_preserved\":true,\"mode\":\"%s\",\"cases\":%zu,\"outputs\":%zu,\"callback_calls\":%llu,\"hash\":\"%016llx\",\"cpu_library\":\"%s\",\"base_library\":\"%s\"}\n", argv[1], cases, comparisons, (unsigned long long) calls.load(), (unsigned long long) hash, cpu_library.dli_fname, base_library.dli_fname);
    ggml_backend_free(backend);
}
