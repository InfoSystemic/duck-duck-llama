#include "qwen-hc-mix-0908.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
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
        !dladdr(reinterpret_cast<void *>(ggml_custom_4d), &base_library)) return 3;
    std::atomic<uint64_t> calls{0};
    std::mt19937 random(982174);
    size_t cases = 0, comparisons = 0;
    uint64_t hash = 14695981039346656037ULL;
    const std::vector<int> workers = numa ? std::vector<int>{15} : std::vector<int>{1,4,15};
    for (int nth : workers) {
        if (!numa) ggml_backend_cpu_set_n_threads(backend, nth);
        for (int width : {257,2560}) for (int streams : {1,3,4})
        for (int tokens : {1,3,5,64}) for (int pad : {0,7}) {
            auto inputs = ggml_init({1024 * 1024, nullptr, true});
            auto context = ggml_init({2 * 1024 * 1024, nullptr, true});
            auto xb = ggml_new_tensor_2d(inputs, GGML_TYPE_F32, width * streams + pad, tokens);
            auto gb = ggml_new_tensor_2d(inputs, GGML_TYPE_F32, width * streams + pad + 3, tokens);
            ggml_set_input(xb); ggml_set_input(gb);
            ggml_set_name(xb, "hc_values"); ggml_set_name(gb, "hc_gates");
            auto ib = ggml_backend_alloc_ctx_tensors(inputs, backend);
            if (!ib) return 4;
            auto x = ggml_view_2d(context, xb, width * streams, tokens, xb->nb[1], 0);
            auto g = ggml_view_2d(context, gb, width * streams, tokens, gb->nb[1], 0);
            auto gated = ggml_mul(context, x, g);
            gated = ggml_reshape_3d(context, gated, width, streams, tokens);
            auto baseline = ggml_cont(context, ggml_view_2d(context, gated, width, tokens,
                                      width * streams * sizeof(float), 0));
            for (int stream = 1; stream < streams; ++stream) {
                auto part = ggml_view_2d(context, gated, width, tokens, width * streams * sizeof(float),
                                         width * stream * sizeof(float));
                baseline = ggml_add(context, baseline, part);
            }
            baseline = ggml_scale(context, baseline, 1.0f / static_cast<float>(streams));
            ggml_tensor * args[] = {x, g};
            auto candidate = ggml_custom_4d(context, GGML_TYPE_F32, width, tokens, 1, 1,
                                            args, 2, qwen_hc_mix_f32, GGML_N_TASKS_MAX, &calls);
            ggml_set_output(baseline); ggml_set_output(candidate);
            auto graph = ggml_new_graph_custom(context, 128, false);
            ggml_build_forward_expand(graph, baseline);
            ggml_build_forward_expand(graph, candidate);
            auto allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            if (!ggml_gallocr_alloc_graph(allocator, graph)) return 4;
            std::vector<float> xv(ggml_nelements(xb)), gv(ggml_nelements(gb));
            std::vector<float> expected(size_t(width) * tokens), actual(expected.size());
            for (int probe = 0; probe < 4; ++probe) {
                for (auto & v : xv) v = probe == 3 ? -0.0f : float(int(random() % 65536) - 32768) * (probe == 2 ? 123456.75f : .00123f);
                for (auto & v : gv) v = probe == 3 ? 1.0f : probe == 1 ? float(random() & 1) : float(random() % 65536) / 65536.f;
                ggml_backend_tensor_set(xb, xv.data(), 0, ggml_nbytes(xb));
                ggml_backend_tensor_set(gb, gv.data(), 0, ggml_nbytes(gb));
                if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return 5;
                ggml_backend_synchronize(backend);
                ggml_backend_tensor_get(baseline, expected.data(), 0, ggml_nbytes(baseline));
                ggml_backend_tensor_get(candidate, actual.data(), 0, ggml_nbytes(candidate));
                if (std::memcmp(expected.data(), actual.data(), actual.size() * sizeof(float))) {
                    std::fprintf(stderr, "graph mismatch width=%d streams=%d tokens=%d pad=%d probe=%d\n", width, streams, tokens, pad, probe);
                    return 6;
                }
                for (int t = 0; t < tokens; ++t) for (int j = 0; j < width; ++j) {
                    const float * xr = xv.data() + size_t(t) * (width * streams + pad);
                    const float * gr = gv.data() + size_t(t) * (width * streams + pad + 3);
                    volatile float sum = xr[j] * gr[j];
                    for (int stream = 1; stream < streams; ++stream) {
                        volatile float product = xr[stream * width + j] * gr[stream * width + j];
                        sum = sum + product;
                    }
                    const float reference = sum * (1.0f / static_cast<float>(streams));
                    if (std::memcmp(&reference, &actual[size_t(t) * width + j], sizeof(float))) return 7;
                }
                for (auto pair : {std::make_pair(xb, &xv), std::make_pair(gb, &gv)}) {
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
    if (calls.load() != cases * (numa ? 4 : 1)) return 9;
    std::printf("{\"passed\":true,\"bit_exact\":true,\"scalar_exact\":true,\"inputs_preserved\":true,\"mode\":\"%s\",\"cases\":%zu,\"outputs\":%zu,\"callback_calls\":%llu,\"hash\":\"%016llx\",\"cpu_library\":\"%s\",\"base_library\":\"%s\"}\n", argv[1], cases, comparisons, (unsigned long long) calls.load(), (unsigned long long) hash, cpu_library.dli_fname, base_library.dli_fname);
    ggml_backend_free(backend);
}
