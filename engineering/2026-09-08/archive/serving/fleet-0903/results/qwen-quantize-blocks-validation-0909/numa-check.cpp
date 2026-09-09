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

#if defined(QWEN_MOE_CHECK)
#include "llama-model.h"
#include <dlfcn.h>
#include <memory>
static constexpr bool check_down = true;
#else
static constexpr bool check_down = false;
#endif
static constexpr int weight_stride = check_down ? 3 : 2;

#if defined(QWEN_Q6_CHECK)
using qwen_weight_block = block_q6_K;
static constexpr ggml_type qwen_weight_type = GGML_TYPE_Q6_K;
#elif defined(QWEN_Q8_CHECK)
using qwen_weight_block = block_q8_0;
static constexpr ggml_type qwen_weight_type = GGML_TYPE_Q8_0;
#elif defined(QWEN_IQ3_CHECK)
using qwen_weight_block = block_iq3_xxs;
static constexpr ggml_type qwen_weight_type = GGML_TYPE_IQ3_XXS;
#else
using qwen_weight_block = block_iq2_xs;
static constexpr ggml_type qwen_weight_type = GGML_TYPE_IQ2_XS;
#endif
#if defined(QWEN_Q6_CHECK)
using qwen_activation_block = block_q8_K;
using qwen_down_block = block_q8_0;
static constexpr ggml_type qwen_activation_type = GGML_TYPE_Q8_K;
static constexpr ggml_type qwen_down_type = GGML_TYPE_Q8_0;
static constexpr bool draft_geometry = false;
#elif defined(QWEN_Q8_CHECK)
using qwen_activation_block = block_q8_0;
using qwen_down_block = block_q8_0;
static constexpr ggml_type qwen_activation_type = GGML_TYPE_Q8_0;
static constexpr ggml_type qwen_down_type = GGML_TYPE_Q8_0;
static constexpr bool draft_geometry = true;
#else
using qwen_activation_block = block_q8_K;
using qwen_down_block = block_iq4_nl;
static constexpr ggml_type qwen_activation_type = GGML_TYPE_Q8_K;
static constexpr ggml_type qwen_down_type = GGML_TYPE_IQ4_NL;
static constexpr bool draft_geometry = false;
#endif

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
#if defined(QWEN_MOE_CHECK)
    return llama_meta_device_get_split_state(tensor, user);
#else
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
#endif
}

int main(int argc, char ** argv) {
    Dl_info runtime{};
    if (!dladdr(reinterpret_cast<void *>(ggml_backend_cpu_init), &runtime)) std::abort();
    std::printf("QUANT_CPU_LIBRARY %s\n", runtime.dli_fname);
    std::atexit([] {
        using counter_fn = uint64_t (*)(int);
        const auto counter = reinterpret_cast<counter_fn>(dlsym(RTLD_DEFAULT, "ggml_cpu_qwen_quantize_blocks_count"));
        std::printf("QUANT_BLOCK_CALLS %llu %llu %llu\n", (unsigned long long) (counter ? counter(0) : 0),
                    (unsigned long long) (counter ? counter(1) : 0), (unsigned long long) (counter ? counter(2) : 0));
    });

    if (argc != 5) return 2;
    const int granularity = std::stoi(argv[1]), workers = std::stoi(argv[2]);
    const double seconds = std::stod(argv[4]);
    const int k = std::stoi(std::getenv("COLD_GRAPH_K_PER_SOCKET"));
    const int rows = std::stoi(std::getenv("COLD_GRAPH_ROWS"));
    const int tokens = std::stoi(std::getenv("COLD_GRAPH_TOKENS"));
    const int copies = std::stoi(std::getenv("COLD_GRAPH_MATRICES"));
    constexpr int experts = 512, used = 10, packed_block_bytes = 16 * sizeof(block_q6_K);
    if ((granularity != 32 && granularity != 128) || workers != 15 || k != 2560 || rows != 640 ||
        tokens < 1 || tokens > 64 || copies < 4 || copies > 16 || copies % 4 || seconds < 3 || seconds > 10) return 2;
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
    void * split_user = (void *) &granularity;
    const char * policy_library = "";
#if defined(QWEN_MOE_CHECK)
    const char * enabled = std::getenv("GGML_Q4E_EXPERT_EVEN_SPLIT");
    if (!enabled || std::atoi(enabled) != (granularity == 32)) return 2;
    auto model_params = llama_model_default_params();
    model_params.split_mode = LLAMA_SPLIT_MODE_TENSOR;
    std::vector<float> fractions(llama_max_devices(), 0.0f);
    std::fill_n(fractions.begin(), 4, 1.0f);
    model_params.tensor_split = fractions.data();
    std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_QWEN4EXP, model_params));
    if (!model || model->arch != LLM_ARCH_QWEN4EXP) return 2;
    auto & h = model->hparams;
    h.n_layer_all = draft_geometry ? 49 : 48;
    h.n_layer_nextn = draft_geometry ? 1 : 0;
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
    if (draft_geometry) h.is_recr_impl[48] = false;
    h.ssm_d_state = 128;
    h.ssm_n_group = 16;
    h.ssm_dt_rank = 48;
    llama_meta_device_get_split_state_userdata split_data{4, model.get()};
    split_user = &split_data;
    Dl_info library = {};
    if (!dladdr(reinterpret_cast<void *>(llama_meta_device_get_split_state), &library)) return 2;
    policy_library = library.dli_fname;
#endif
    auto meta = ggml_backend_meta_device(devices.data(), devices.size(), split_leaf, split_user);
    auto backend = ggml_backend_dev_init(meta, nullptr);
    auto extra = ggml_backend_meta_device_get_extra_bufts(meta);
    ggml_backend_buffer_type_t packed_type = nullptr;
    for (int i = 0; extra && extra[i]; ++i) {
        if (std::strstr(ggml_backend_buft_name(extra[i]), "REPACK")) { packed_type = extra[i]; break; }
    }
    if (!backend || !packed_type) return 2;
    auto wc = ggml_init({size_t(weight_stride * copies + 8) * ggml_tensor_overhead(), nullptr, true});
    auto ic = ggml_init({size_t(copies + 8) * ggml_tensor_overhead(), nullptr, true});
    auto gc = ggml_init({4 * 1024 * 1024, nullptr, true});
    std::vector<ggml_tensor *> weights, outputs, hidden, routes;
    for (int m = 0; m < copies; ++m) {
        const int layer = draft_geometry ? (m == 0 ? 48 : 4 * m + 3) : check_down ? m + m / 3 : m;
        for (int which = 0; which < weight_stride; ++which) {
            const char * kind = which == 0 ? "gate" : which == 1 ? "up" : "down";
            auto w = ggml_new_tensor_3d(wc, which == 2 ? qwen_down_type : qwen_weight_type,
                                        which == 2 ? rows : k, which == 2 ? k : rows, experts);
            ggml_set_name(w, ("blk." + std::to_string(layer) + ".ffn_" + kind + "_exps.weight").c_str());
            weights.push_back(w);
#if defined(QWEN_MOE_CHECK)
            model->tensors_by_name.emplace_back(w->name, w);
#endif
        }
        auto ids = ggml_new_tensor_2d(ic, GGML_TYPE_I32, used, tokens);
        ggml_set_name(ids, ("routes" + std::to_string(m)).c_str());
        ggml_set_input(ids);
        routes.push_back(ids);
    }
    auto x = ggml_new_tensor_3d(ic, GGML_TYPE_F32, k, 1, tokens);
    ggml_set_name(x, "input");
    ggml_set_input(x);
    if (check_down) {
        for (int m = 0; m < copies; ++m) for (int which = 0; which < weight_stride; ++which) {
            const auto split = split_leaf(weights[weight_stride * m + which], split_user);
            if (split.axis != (which == 2 ? GGML_BACKEND_SPLIT_AXIS_0 : GGML_BACKEND_SPLIT_AXIS_1) ||
                split.n_segments != 1 || split.nr[0] != 1) return 2;
            int64_t ordered[4];
            std::copy_n(split.ne, 4, ordered);
            std::sort(ordered, ordered + 4);
            for (int j = 0; j < 4; ++j) {
                if (ordered[j] != (granularity == 32 ? 160 : j == 3 ? 256 : 128)) return 2;
            }
        }
    }
    auto wb = ggml_backend_alloc_ctx_tensors_from_buft(wc, packed_type);
    auto ib = ggml_backend_alloc_ctx_tensors(ic, backend);
    if (!wb || !ib) return 2;
    ggml_backend_buffer_set_usage(wb, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    auto graph = ggml_new_graph(gc);
    for (int m = 0; m < copies; ++m) {
        auto gate = ggml_mul_mat_id(gc, weights[weight_stride * m], x, routes[m]);
        auto up = ggml_mul_mat_id(gc, weights[weight_stride * m + 1], x, routes[m]);
        auto output = ggml_swiglu_split(gc, gate, up);
        hidden.push_back(output);
        if (check_down) {
            ggml_set_output(output);
            output = ggml_mul_mat_id(gc, weights[weight_stride * m + 2], output, routes[m]);
            // A mirrored consumer materializes the four partial down sums.
            output = ggml_add(gc, output, x);
        }
        ggml_set_output(output);
        outputs.push_back(output);
        ggml_build_forward_expand(graph, output);
    }
    auto allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(allocator, graph)) return 2;

    std::mt19937 rng(93481);
    qwen_weight_block templates[64];
    for (int i = 0; i < 64; ++i) {
        auto * bytes = reinterpret_cast<uint8_t *>(templates + i);
        for (size_t j = 0; j < sizeof(qwen_weight_block); ++j) bytes[j] = uint8_t(rng());
        templates[i].d = ggml_fp32_to_fp16(0.00003f * (1 + i % 5));
#if defined(QWEN_Q8_CHECK)
        templates[i].d = ggml_fp32_to_fp16(0.0003f * (1 + i % 5));
        for (auto & q : templates[i].qs) q = int(rng() % 255) - 127;
#endif
    }
    const int blocks = k / ggml_blck_size(qwen_weight_type);
    std::vector<qwen_weight_block> canonical(size_t(experts) * rows * blocks);
    uint64_t weight_hash = 14695981039346656037ULL;
    for (int m = 0; m < 2 * copies; ++m) {
        for (int e = 0; e < experts; ++e) for (int r = 0; r < rows; ++r) for (int b = 0; b < blocks; ++b) {
            canonical[(size_t(e) * rows + r) * blocks + b] = templates[(r * 17 + b * 3 + m * 11 + e * 7) % 64];
        }
        weight_hash = hash_bytes(canonical.data(), canonical.size() * sizeof(qwen_weight_block), weight_hash);
        auto w = weights[(m / 2) * weight_stride + m % 2];
        ggml_backend_tensor_set(w, canonical.data(), 0, ggml_nbytes(w));
    }
    canonical.clear();
    canonical.shrink_to_fit();
    qwen_down_block down_templates[64];
    const int down_blocks = rows / QK4_NL;
    if (check_down) {
        for (int i = 0; i < 64; ++i) {
            auto * bytes = reinterpret_cast<uint8_t *>(down_templates + i);
            for (size_t j = 0; j < sizeof(qwen_down_block); ++j) bytes[j] = uint8_t(rng());
            down_templates[i].d = ggml_fp32_to_fp16(0.007f * (1 + i % 5));
#if defined(QWEN_Q8_CHECK) || defined(QWEN_Q6_CHECK)
            down_templates[i].d = ggml_fp32_to_fp16(0.0007f * (1 + i % 5));
            for (auto & q : down_templates[i].qs) q = int(rng() % 255) - 127;
#endif
        }
        std::vector<qwen_down_block> down_weights(size_t(experts) * k * down_blocks);
        for (int m = 0; m < copies; ++m) {
            for (int e = 0; e < experts; ++e) for (int r = 0; r < k; ++r) for (int b = 0; b < down_blocks; ++b) {
                down_weights[(size_t(e) * k + r) * down_blocks + b] = down_templates[(r * 13 + b * 7 + m * 11 + e * 3) % 64];
            }
            weight_hash = hash_bytes(down_weights.data(), down_weights.size() * sizeof(qwen_down_block), weight_hash);
            auto w = weights[weight_stride * m + 2];
            ggml_backend_tensor_set(w, down_weights.data(), 0, ggml_nbytes(w));
        }
    }
    std::vector<float> input(size_t(k) * tokens);
    std::vector<qwen_activation_block> input_q(size_t(blocks) * tokens);
    const int output_rows = check_down ? k : rows;
    std::vector<std::vector<float>> checked(copies, std::vector<float>(size_t(output_rows) * used * tokens));
    std::vector<std::vector<int32_t>> ids(copies, std::vector<int32_t>(used * tokens));
    float max_error = 0;
    float max_down_error = 0, max_down_scaled_error = 0;
    uint64_t output_hash = 14695981039346656037ULL;
    FILE * output_file = nullptr;
    if (check_down) {
        const char * path = std::getenv("COLD_GRAPH_OUTPUT_PATH");
        if (!path || !(output_file = std::fopen(path, "wb"))) return 2;
    }
    for (int probe = 0; probe < 3; ++probe) {
        for (size_t i = 0; i < input.size(); ++i) input[i] = std::sin(float(i) * 0.031f + probe * 0.37f);
        ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));
        for (int t = 0; t < tokens; ++t) {
            ggml_get_type_traits_cpu(qwen_activation_type)->from_float(input.data() + t * k, input_q.data() + t * blocks, k);
        }
        for (int m = 0; m < copies; ++m) {
            // Partial overlap exercises experts with one, two, and three rows.
            for (int t = 0; t < tokens; ++t) for (int u = 0; u < used; ++u) ids[m][t * used + u] = (u + 2 * t + (m % 2 ? 508 : 250) + probe) % experts;
            ggml_backend_tensor_set(routes[m], ids[m].data(), 0, ggml_nbytes(routes[m]));
        }
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return 2;
        ggml_backend_synchronize(backend);
        for (int m = 0; m < copies; ++m) {
            ggml_backend_tensor_get(outputs[m], checked[m].data(), 0, ggml_nbytes(outputs[m]));
            std::vector<float> hidden_values;
            if (check_down) {
                hidden_values.resize(size_t(rows) * used * tokens);
                ggml_backend_tensor_get(hidden[m], hidden_values.data(), 0, ggml_nbytes(hidden[m]));
            }
            for (int t = 0; t < tokens; ++t) for (int u = 0; u < used; ++u) {
                const int e = ids[m][t * used + u];
                float expected[64];
                for (int r = 0; r < 64; ++r) {
                    float dots[2];
                    for (int which = 0; which < 2; ++which) {
                        std::vector<qwen_weight_block> weight_row(blocks);
                        for (int b = 0; b < blocks; ++b) weight_row[b] = templates[(r * 17 + b * 3 + (2 * m + which) * 11 + e * 7) % 64];
                        ggml_get_type_traits_cpu(qwen_weight_type)->vec_dot(k, dots + which, 0,
                            weight_row.data(), 0, input_q.data() + t * blocks, 0, 1);
                    }
                    expected[r] = (dots[0] / (1.0f + std::exp(-dots[0]))) * dots[1];
                }
                for (int r = 0; r < rows; ++r) {
                    const float actual = (check_down ? hidden_values : checked[m])[(size_t(t) * used + u) * rows + r];
                    const float error = std::abs(actual - expected[r % 64]);
                    max_error = std::max(max_error, error);
                    if (!std::isfinite(actual) || error > 2e-4f * (1 + std::abs(expected[r % 64]))) {
                        std::fprintf(stderr, "Reference mismatch: m=%d t=%d u=%d r=%d actual=%g expected=%g\n", m, t, u, r, actual, expected[r % 64]);
                        return 1;
                    }
                }
                if (check_down) {
                    std::vector<block_q8_0> quantized(down_blocks);
                    ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float(
                        hidden_values.data() + (size_t(t) * used + u) * rows, quantized.data(), rows);
                    float expected_down[64];
                    for (int r = 0; r < 64; ++r) {
                        qwen_down_block down_row[20];
                        for (int b = 0; b < down_blocks; ++b) down_row[b] = down_templates[(r * 13 + b * 7 + m * 11 + e * 3) % 64];
                        ggml_get_type_traits_cpu(qwen_down_type)->vec_dot(rows, expected_down + r, 0,
                            down_row, 0, quantized.data(), 0, 1);
                    }
                    for (int r = 0; r < k; ++r) {
                        const float actual = checked[m][(size_t(t) * used + u) * k + r];
                        const float reference = expected_down[r % 64] + input[size_t(t) * k + r];
                        const float error = std::abs(actual - reference);
                        const float scaled = error / (1 + std::abs(reference));
                        max_down_error = std::max(max_down_error, error);
                        max_down_scaled_error = std::max(max_down_scaled_error, scaled);
                        if (!std::isfinite(actual) || scaled > 2e-5f) {
                            std::fprintf(stderr, "Down mismatch: m=%d t=%d u=%d r=%d actual=%g expected=%g error=%g\n", m, t, u, r, actual, reference, error);
                            return 1;
                        }
                    }
                }
            }
            output_hash = hash_bytes(checked[m].data(), ggml_nbytes(outputs[m]), output_hash);
            if (output_file && std::fwrite(checked[m].data(), sizeof(float), checked[m].size(), output_file) != checked[m].size()) return 2;
        }
    }
    if (output_file && std::fclose(output_file)) return 2;
    const uint64_t gate_up_bytes = draft_geometry ? uint64_t(2) * rows * blocks * sizeof(qwen_weight_block) :
        uint64_t(2) * (rows / 16) * blocks * packed_block_bytes;
    const uint64_t bytes_per_expert = uint64_t(copies) * (gate_up_bytes +
        (check_down ? uint64_t(k) * down_blocks * sizeof(qwen_down_block) : 0));
    const uint64_t bytes_per_pass = bytes_per_expert * (used + 2 * (tokens - 1));
    const uint64_t pool_bytes = bytes_per_expert * experts;
    std::printf("{\"event\":\"ready\",\"pid\":%d,\"workers\":60,\"chunk\":%d,\"group\":1,\"controller_cores\":4,\"k\":%d,\"rows\":%d,\"tokens\":%d,\"matrices\":%d,\"experts\":%d,\"used\":%d,\"bytes_per_pass\":%llu,\"packed_weight_bytes\":%llu,\"weight_hash\":\"%016llx\",\"output_hash\":\"%016llx\",\"max_reference_error\":%.9g,\"weight_type\":\"%s\",\"down_checked\":%s,\"max_down_error\":%.9g,\"max_down_scaled_error\":%.9g,\"policy_library\":\"%s\"}\n",
                getpid(), granularity, k, rows, tokens, copies, experts, used, (unsigned long long) bytes_per_pass,
                (unsigned long long) pool_bytes, (unsigned long long) weight_hash, (unsigned long long) output_hash, max_error, ggml_type_name(qwen_weight_type),
                check_down ? "true" : "false", max_down_error, max_down_scaled_error, policy_library);
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
