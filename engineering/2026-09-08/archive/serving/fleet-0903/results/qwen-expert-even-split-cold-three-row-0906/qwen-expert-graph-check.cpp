// Compare expert row splits through Qwen's unchanged four-NUMA backend.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-quants.h"
#include "repack.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sched.h>
#include <string>
#include <time.h>
#include <unistd.h>
#include <vector>

static double now() {
    timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t)) std::abort();
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static uint64_t hash_bytes(const void * data, size_t count, uint64_t hash = 14695981039346656037ULL) {
    const auto * bytes = (const uint8_t *) data;
    for (size_t i = 0; i < count; ++i) hash = (hash ^ bytes[i]) * 1099511628211ULL;
    return hash;
}

static ggml_backend_meta_split_state split_leaf(const ggml_tensor * tensor, void * user) {
    const int granularity = *(const int *) user;
    ggml_backend_meta_split_state state = {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
    int layer = -1;
    if (std::sscanf(tensor->name, "blk.%d.", &layer) == 1 && std::strstr(tensor->name, "_exps.weight")) {
        state.axis = GGML_BACKEND_SPLIT_AXIS_1;
        state.n_segments = 1;
        state.nr[0] = 1;
        int64_t low = 0;
        for (int device = 0; device < 4; ++device) {
            int64_t high = device == 3 ? tensor->ne[1] : tensor->ne[1] * (device + 1) / 4;
            if (device != 3) high -= high % granularity;
            state.ne[(device + layer) % 4] = high - low;
            low = high;
        }
    }
    return state;
}

int main(int argc, char ** argv) {
    if (argc != 5) return 2;
    const int granularity = std::stoi(argv[1]), workers = std::stoi(argv[2]);
    const double seconds = std::stod(argv[4]);
    const int k = std::stoi(std::getenv("COLD_GRAPH_K_PER_SOCKET"));
    const int rows = std::stoi(std::getenv("COLD_GRAPH_ROWS"));
    const int tokens = std::stoi(std::getenv("COLD_GRAPH_TOKENS"));
    const int copies = std::stoi(std::getenv("COLD_GRAPH_MATRICES"));
    constexpr int experts = 32, used = 10, packed_block_bytes = 2208;
    if ((granularity != 32 && granularity != 128) || workers != 15 || k != 2560 || rows != 640 ||
        tokens < 1 || tokens > 3 || copies < 4 || copies > 16 || copies % 4 || seconds < 3 || seconds > 10) return 2;
    ggml_backend_load_all();
    std::vector<ggml_backend_dev_t> devices;
    for (int j = 0; j < 4; ++j) {
        auto device = ggml_backend_dev_by_name(("CPU-NUMA" + std::to_string(j)).c_str());
        if (!device) return 2;
        devices.push_back(device);
    }
    cpu_set_t controller_mask;
    CPU_ZERO(&controller_mask);
    for (int cpu : {15, 31, 47, 63}) CPU_SET(cpu, &controller_mask);
    if (sched_setaffinity(0, sizeof(controller_mask), &controller_mask)) return 2;
    auto meta = ggml_backend_meta_device(devices.data(), devices.size(), split_leaf, (void *) &granularity);
    auto backend = ggml_backend_dev_init(meta, nullptr);
    auto extra = ggml_backend_meta_device_get_extra_bufts(meta);
    ggml_backend_buffer_type_t packed_type = nullptr;
    for (int i = 0; extra && extra[i]; ++i) {
        if (std::strstr(ggml_backend_buft_name(extra[i]), "REPACK")) { packed_type = extra[i]; break; }
    }
    if (!backend || !packed_type) return 2;
    auto wc = ggml_init({size_t(2 * copies + 8) * ggml_tensor_overhead(), nullptr, true});
    auto ic = ggml_init({size_t(copies + 8) * ggml_tensor_overhead(), nullptr, true});
    auto gc = ggml_init({4 * 1024 * 1024, nullptr, true});
    std::vector<ggml_tensor *> weights, outputs, routes;
    for (int m = 0; m < copies; ++m) {
        for (const char * kind : {"gate", "up"}) {
            auto w = ggml_new_tensor_3d(wc, GGML_TYPE_IQ2_XS, k, rows, experts);
            ggml_set_name(w, ("blk." + std::to_string(m) + ".ffn_" + kind + "_exps.weight").c_str());
            weights.push_back(w);
        }
        auto ids = ggml_new_tensor_2d(ic, GGML_TYPE_I32, used, tokens);
        ggml_set_name(ids, ("routes" + std::to_string(m)).c_str());
        ggml_set_input(ids);
        routes.push_back(ids);
    }
    auto x = ggml_new_tensor_3d(ic, GGML_TYPE_F32, k, 1, tokens);
    ggml_set_name(x, "input");
    ggml_set_input(x);
    auto wb = ggml_backend_alloc_ctx_tensors_from_buft(wc, packed_type);
    auto ib = ggml_backend_alloc_ctx_tensors(ic, backend);
    if (!wb || !ib) return 2;
    ggml_backend_buffer_set_usage(wb, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    auto graph = ggml_new_graph(gc);
    for (int m = 0; m < copies; ++m) {
        auto gate = ggml_mul_mat_id(gc, weights[2 * m], x, routes[m]);
        auto up = ggml_mul_mat_id(gc, weights[2 * m + 1], x, routes[m]);
        auto output = ggml_swiglu_split(gc, gate, up);
        ggml_set_output(output);
        outputs.push_back(output);
        ggml_build_forward_expand(graph, output);
    }
    auto allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(allocator, graph)) return 2;

    std::mt19937 rng(93481);
    block_iq2_xs templates[64];
    for (int i = 0; i < 64; ++i) {
        auto * bytes = reinterpret_cast<uint8_t *>(templates + i);
        for (size_t j = 0; j < sizeof(block_iq2_xs); ++j) bytes[j] = uint8_t(rng());
        templates[i].d = ggml_fp32_to_fp16(0.00003f * (1 + i % 5));
    }
    const int blocks = k / QK_K;
    std::vector<block_iq2_xs> canonical(size_t(experts) * rows * blocks);
    uint64_t weight_hash = 14695981039346656037ULL;
    for (int m = 0; m < 2 * copies; ++m) {
        for (int e = 0; e < experts; ++e) for (int r = 0; r < rows; ++r) for (int b = 0; b < blocks; ++b) {
            canonical[(size_t(e) * rows + r) * blocks + b] = templates[(r * 17 + b * 3 + m * 11 + e * 7) % 64];
        }
        weight_hash = hash_bytes(canonical.data(), canonical.size() * sizeof(block_iq2_xs), weight_hash);
        ggml_backend_tensor_set(weights[m], canonical.data(), 0, ggml_nbytes(weights[m]));
    }
    canonical.clear();
    canonical.shrink_to_fit();
    std::vector<float> input(size_t(k) * tokens);
    std::vector<block_q8_K> input_q(size_t(blocks) * tokens);
    std::vector<std::vector<float>> checked(copies, std::vector<float>(size_t(rows) * used * tokens));
    std::vector<std::vector<int32_t>> ids(copies, std::vector<int32_t>(used * tokens));
    float max_error = 0;
    uint64_t output_hash = 14695981039346656037ULL;
    for (int probe = 0; probe < 3; ++probe) {
        for (size_t i = 0; i < input.size(); ++i) input[i] = std::sin(float(i) * 0.031f + probe * 0.37f);
        ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));
        for (int t = 0; t < tokens; ++t) {
            ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float(input.data() + t * k, input_q.data() + t * blocks, k);
        }
        for (int m = 0; m < copies; ++m) {
            // Partial overlap exercises experts with one, two, and three rows.
            for (int t = 0; t < tokens; ++t) for (int u = 0; u < used; ++u) ids[m][t * used + u] = (u + 2 * t + 3 * m + probe) % experts;
            ggml_backend_tensor_set(routes[m], ids[m].data(), 0, ggml_nbytes(routes[m]));
        }
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return 2;
        ggml_backend_synchronize(backend);
        for (int m = 0; m < copies; ++m) {
            ggml_backend_tensor_get(outputs[m], checked[m].data(), 0, ggml_nbytes(outputs[m]));
            for (int t = 0; t < tokens; ++t) for (int u = 0; u < used; ++u) {
                const int e = ids[m][t * used + u];
                float expected[64];
                for (int r = 0; r < 64; ++r) {
                    float dots[2];
                    for (int which = 0; which < 2; ++which) {
                        block_iq2_xs weight_row[10];
                        for (int b = 0; b < blocks; ++b) weight_row[b] = templates[(r * 17 + b * 3 + (2 * m + which) * 11 + e * 7) % 64];
                        ggml_get_type_traits_cpu(GGML_TYPE_IQ2_XS)->vec_dot(k, dots + which, 0,
                            weight_row, 0, input_q.data() + t * blocks, 0, 1);
                    }
                    expected[r] = (dots[0] / (1.0f + std::exp(-dots[0]))) * dots[1];
                }
                for (int r = 0; r < rows; ++r) {
                    const float actual = checked[m][(size_t(t) * used + u) * rows + r];
                    const float error = std::abs(actual - expected[r % 64]);
                    max_error = std::max(max_error, error);
                    if (!std::isfinite(actual) || error > 2e-4f * (1 + std::abs(expected[r % 64]))) {
                        std::fprintf(stderr, "Reference mismatch: m=%d t=%d u=%d r=%d actual=%g expected=%g\n", m, t, u, r, actual, expected[r % 64]);
                        return 1;
                    }
                }
            }
            output_hash = hash_bytes(checked[m].data(), ggml_nbytes(outputs[m]), output_hash);
        }
    }
    const uint64_t bytes_per_expert = uint64_t(copies) * 2 * (rows / 16) * blocks * packed_block_bytes;
    const uint64_t bytes_per_pass = bytes_per_expert * (used + 2 * (tokens - 1));
    const uint64_t pool_bytes = bytes_per_expert * experts;
    std::printf("{\"event\":\"ready\",\"pid\":%d,\"workers\":60,\"chunk\":%d,\"group\":1,\"controller_cores\":4,\"k\":%d,\"rows\":%d,\"tokens\":%d,\"matrices\":%d,\"experts\":%d,\"used\":%d,\"bytes_per_pass\":%llu,\"packed_weight_bytes\":%llu,\"weight_hash\":\"%016llx\",\"output_hash\":\"%016llx\",\"max_reference_error\":%.9g}\n",
                getpid(), granularity, k, rows, tokens, copies, experts, used, (unsigned long long) bytes_per_pass,
                (unsigned long long) pool_bytes, (unsigned long long) weight_hash, (unsigned long long) output_hash, max_error);
    std::fflush(stdout);
    char command[16];
    unsigned phase = 0;
    while (true) {
        if (!std::fgets(command, sizeof(command), stdin)) return 2;
        if (!std::strcmp(command, "exit\n")) break;
        if (std::strcmp(command, "go\n")) return 2;
        const double start = now();
        uint64_t passes = 0;
        while (now() - start < seconds) {
            if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return 2;
            ++passes;
        }
        ggml_backend_synchronize(backend);
        const double end = now();
        bool exact = passes > 0;
        for (int m = 0; m < copies; ++m) {
            std::vector<float> values(checked[m].size());
            ggml_backend_tensor_get(outputs[m], values.data(), 0, ggml_nbytes(outputs[m]));
            exact &= std::memcmp(values.data(), checked[m].data(), ggml_nbytes(outputs[m])) == 0;
        }
        std::printf("{\"event\":\"done\",\"phase\":%u,\"start\":%.9f,\"end\":%.9f,\"passes\":%llu,\"bytes\":%llu,\"logical_gb_s\":%.9f,\"graph_ms\":%.9f,\"checksums_exact\":%s,\"threads\":[{\"start\":%.9f,\"end\":%.9f}]}\n",
                    phase++, start, end, (unsigned long long) passes, (unsigned long long) (passes * bytes_per_pass),
                    passes * bytes_per_pass / (end - start) / 1e9, 1000 * (end - start) / passes,
                    exact ? "true" : "false", start, end);
        std::fflush(stdout);
        if (!exact) return 1;
    }
    ggml_gallocr_free(allocator);
    ggml_backend_buffer_free(wb);
    ggml_backend_buffer_free(ib);
    ggml_free(gc);
    ggml_free(ic);
    ggml_free(wc);
    ggml_backend_free(backend);
    return 0;
}
