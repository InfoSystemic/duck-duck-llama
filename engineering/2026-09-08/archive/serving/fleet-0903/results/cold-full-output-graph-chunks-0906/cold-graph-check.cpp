// Finite, cold Q5 projections through Full's unchanged four-NUMA graph backend.
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

static ggml_backend_meta_split_state split_leaf(const ggml_tensor * tensor, void *) {
    ggml_backend_meta_split_state state = {};
    state.axis = GGML_BACKEND_SPLIT_AXIS_MIRRORED;
    if (std::strstr(tensor->name, ".weight") || !std::strcmp(tensor->name, "input")) {
        state.axis = GGML_BACKEND_SPLIT_AXIS_0;
        state.n_segments = 1;
        state.nr[0] = 1;
        if (tensor->ne[0] % 4) std::abort();
        for (int j = 0; j < 4; ++j) state.ne[j] = tensor->ne[0] / 4;
    }
    return state;
}

int main(int argc, char ** argv) {
    if (argc != 5) return 2;
    const int chunk = std::stoi(argv[1]), workers = std::stoi(argv[2]);
    const double seconds = std::stod(argv[4]);
    const int k = 4 * std::stoi(std::getenv("COLD_GRAPH_K_PER_SOCKET"));
    const int rows = std::stoi(std::getenv("COLD_GRAPH_ROWS"));
    const int tokens = std::stoi(std::getenv("COLD_GRAPH_TOKENS"));
    const int copies = std::stoi(std::getenv("COLD_GRAPH_MATRICES"));
    if ((chunk != 16 && chunk != 32 && chunk != 64) || workers != 15 ||
        k < 1024 || k > 32768 || k % 1024 || rows < 64 || rows > 16384 || rows % 64 ||
        tokens < 1 || tokens > 3 || copies < 1 || copies > 64 || seconds < 3 || seconds > 30) return 2;
    setenv("GGML_CPU_X16_CHUNK_MIN", "16", 1);
    setenv("GGML_CPU_X16_CHUNK_MAX", argv[1], 1);
    ggml_backend_load_all();
    std::vector<ggml_backend_dev_t> devices;
    for (int j = 0; j < 4; ++j) {
        auto device = ggml_backend_dev_by_name(("CPU-NUMA" + std::to_string(j)).c_str());
        if (!device) return 2;
        devices.push_back(device);
    }
    auto meta = ggml_backend_meta_device(devices.data(), devices.size(), split_leaf, nullptr);
    auto backend = ggml_backend_dev_init(meta, nullptr);
    auto extra = ggml_backend_meta_device_get_extra_bufts(meta);
    ggml_backend_buffer_type_t packed_type = nullptr;
    for (int i = 0; extra && extra[i]; ++i) {
        if (std::strstr(ggml_backend_buft_name(extra[i]), "REPACK")) { packed_type = extra[i]; break; }
    }
    if (!backend || !packed_type) {
        std::fprintf(stderr, "Missing meta backend or NUMA repack buffer\n");
        return 2;
    }
    auto wc = ggml_init({size_t(copies + 8) * ggml_tensor_overhead(), nullptr, true});
    auto ic = ggml_init({16 * ggml_tensor_overhead(), nullptr, true});
    auto gc = ggml_init({4 * 1024 * 1024, nullptr, true});
    std::vector<ggml_tensor *> weights, outputs;
    for (int m = 0; m < copies; ++m) {
        auto w = ggml_new_tensor_2d(wc, GGML_TYPE_Q5_K, k, rows);
        ggml_set_name(w, ("blk." + std::to_string(m) + ".attn_output.weight").c_str());
        weights.push_back(w);
    }
    auto x = ggml_new_tensor_2d(ic, GGML_TYPE_F32, k, tokens);
    ggml_set_name(x, "input");
    ggml_set_input(x);
    auto wb = ggml_backend_alloc_ctx_tensors_from_buft(wc, packed_type);
    auto ib = ggml_backend_alloc_ctx_tensors(ic, backend);
    if (!wb || !ib) return 2;
    ggml_backend_buffer_set_usage(wb, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    auto graph = ggml_new_graph(gc);
    for (auto * w : weights) {
        auto product = ggml_mul_mat(gc, w, x);
        auto output = ggml_rms_norm(gc, product, 1e-5f);
        ggml_set_output(output);
        outputs.push_back(output);
        ggml_build_forward_expand(graph, output);
    }
    auto allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(allocator, graph)) return 2;

    // Canonical quantized blocks with nontrivial scales and values. Rows repeat
    // every 64, allowing a complete independent native-dot reference cheaply.
    std::mt19937 rng(93471);
    std::uniform_real_distribution<float> distribution(-0.03f, 0.03f);
    block_q5_K templates[64];
    float source[QK_K];
    for (auto & block : templates) {
        for (float & value : source) value = distribution(rng);
        quantize_row_q5_K_ref(source, &block, QK_K);
    }
    const int blocks = k / QK_K;
    std::vector<block_q5_K> canonical(size_t(rows) * blocks);
    std::vector<std::vector<block_q5_K>> reference_rows(copies);
    uint64_t weight_hash = 14695981039346656037ULL;
    for (int m = 0; m < copies; ++m) {
        for (int r = 0; r < rows; ++r) for (int b = 0; b < blocks; ++b) {
            canonical[size_t(r) * blocks + b] = templates[(r * 17 + b * 3 + m * 11) % 64];
        }
        weight_hash = hash_bytes(canonical.data(), canonical.size() * sizeof(block_q5_K), weight_hash);
        reference_rows[m].assign(canonical.begin(), canonical.begin() + 64 * blocks);
        ggml_backend_tensor_set(weights[m], canonical.data(), 0, ggml_nbytes(weights[m]));
    }
    canonical.clear();
    canonical.shrink_to_fit();
    std::vector<float> input(size_t(k) * tokens);
    std::vector<block_q8_K> input_q(size_t(blocks) * tokens);
    std::vector<std::vector<float>> checked(copies, std::vector<float>(size_t(rows) * tokens));
    float max_error = 0;
    uint64_t output_hash = 14695981039346656037ULL;
    for (int probe = 0; probe < 3; ++probe) {
        for (size_t i = 0; i < input.size(); ++i) input[i] = std::sin(float(i) * 0.031f + probe * 0.37f);
        ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));
        for (int t = 0; t < tokens; ++t) {
            ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float(input.data() + t * k, input_q.data() + t * blocks, k);
        }
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return 2;
        ggml_backend_synchronize(backend);
        for (int m = 0; m < copies; ++m) {
            ggml_backend_tensor_get(outputs[m], checked[m].data(), 0, ggml_nbytes(outputs[m]));
            for (int t = 0; t < tokens; ++t) {
                float dots[64];
                double squares = 0;
                for (int r = 0; r < 64; ++r) {
                    ggml_get_type_traits_cpu(GGML_TYPE_Q5_K)->vec_dot(k, dots + r, 0,
                        reference_rows[m].data() + r * blocks, 0, input_q.data() + t * blocks, 0, 1);
                    squares += double(dots[r]) * dots[r];
                }
                const double scale = 1.0 / std::sqrt(squares / 64 + 1e-5f);
                for (int r = 0; r < rows; ++r) {
                    const float actual = checked[m][size_t(t) * rows + r];
                    const double expected = dots[r % 64] * scale;
                    const float error = std::abs(actual - expected);
                    max_error = std::max(max_error, error);
                    if (!std::isfinite(actual) || error > 2e-4 * (1 + std::abs(expected))) {
                        std::fprintf(stderr, "Graph reference mismatch: m=%d t=%d r=%d actual=%g expected=%g\n",
                                     m, t, r, actual, expected);
                        return 1;
                    }
                }
            }
            output_hash = hash_bytes(checked[m].data(), ggml_nbytes(outputs[m]), output_hash);
        }
    }
    const uint64_t bytes_per_pass = uint64_t(copies) * (rows / 16) * blocks * sizeof(block_q5_K_x16);
    std::printf("{\"event\":\"ready\",\"pid\":%d,\"workers\":60,\"chunk\":%d,\"k\":%d,\"rows\":%d,\"tokens\":%d,\"matrices\":%d,\"bytes_per_pass\":%llu,\"weight_hash\":\"%016llx\",\"output_hash\":\"%016llx\",\"max_reference_error\":%.9g}\n",
                getpid(), chunk, k, rows, tokens, copies, (unsigned long long) bytes_per_pass,
                (unsigned long long) weight_hash, (unsigned long long) output_hash, max_error);
    std::fflush(stdout);
    char command[16];
    if (!std::fgets(command, sizeof(command), stdin) || std::strcmp(command, "go\n")) return 2;
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
    std::printf("{\"event\":\"done\",\"start\":%.9f,\"end\":%.9f,\"passes\":%llu,\"bytes\":%llu,\"logical_gb_s\":%.9f,\"graph_ms\":%.9f,\"checksums_exact\":%s,\"threads\":[{\"start\":%.9f,\"end\":%.9f}]}\n",
                start, end, (unsigned long long) passes, (unsigned long long) (passes * bytes_per_pass),
                passes * bytes_per_pass / (end - start) / 1e9, 1000 * (end - start) / passes,
                exact ? "true" : "false", start, end);
    std::fflush(stdout);
    if (!std::fgets(command, sizeof(command), stdin) || std::strcmp(command, "exit\n")) return 2;
    ggml_gallocr_free(allocator);
    ggml_backend_buffer_free(wb);
    ggml_backend_buffer_free(ib);
    ggml_free(gc);
    ggml_free(ic);
    ggml_free(wc);
    ggml_backend_free(backend);
    return exact ? 0 : 1;
}
