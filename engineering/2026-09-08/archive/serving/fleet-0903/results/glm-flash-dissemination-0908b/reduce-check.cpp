#include "ggml-cpu.h"
#include <dlfcn.h>
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

struct split_config { int count; unsigned active; };

static ggml_backend_meta_split_state split_leaf(const ggml_tensor * tensor, void * data) {
    const auto * config = (const split_config *) data;
    ggml_backend_meta_split_state state = {};
    state.axis = GGML_BACKEND_SPLIT_AXIS_MIRRORED;
    if (std::strncmp(tensor->name, "split.", 6) == 0) {
        state.axis = GGML_BACKEND_SPLIT_AXIS_0;
        state.n_segments = 1;
        state.nr[0] = 1;
        int active_count = 0;
        for (int j = 0; j < config->count; ++j) active_count += !!(config->active & (1u << j));
        if (active_count == 0 || tensor->ne[0] % active_count) std::abort();
        for (int j = 0; j < config->count; ++j) {
            state.ne[j] = config->active & (1u << j) ? tensor->ne[0] / active_count : 0;
        }
    }
    return state;
}

static int check_fused_graph() {
    setenv("GGML_CPU_NUMA_DEVICES", "1", 1);
    setenv("GGML_CPU_NUMA_THREADS", "15", 1);
    setenv("GGML_CPU_NUMA_DIRECT_ALLREDUCE", "1", 1);
    setenv("GGML_CPU_NUMA_FUSED_REDUCE", "1", 1);
    setenv("GGML_CPU_NUMA_MERGE_REDUCE", "1", 1);
    setenv("GGML_CPU_NUMA_FUSED_REDUCE_SINGLE_MAX_ELEMENTS", "65536", 0);
    setenv("GGML_CPU_SINGLE_TASK_MAX_ELEMENTS", "4096", 0);
    ggml_backend_load_all();
    int failures = 0, cases = 0;
    const char * timing_env = std::getenv("NUMA_REDUCE_TEST_TIMING_REPEATS");
    const int timing_repeats = timing_env ? std::atoi(timing_env) : 0;
    if (timing_repeats < 0 || timing_repeats > 1000) std::abort();
    const std::vector<int> counts = timing_repeats ? std::vector<int>{4} : std::vector<int>{2, 4};
    const bool full_shapes = std::getenv("NUMA_REDUCE_TEST_FULL_SHAPES") != nullptr;
    const std::vector<int> widths = timing_repeats ? (full_shapes ? std::vector<int>{6144, 16384} : std::vector<int>{128, 4096}) : std::vector<int>{128, 32768};
    for (int count : counts)
    for (unsigned active : timing_repeats ? std::vector<unsigned>{(1u << count) - 1u} : std::vector<unsigned>{1u, (1u << count) - 1u})
    for (int rows : widths) for (int tokens : {1, 3}) for (bool merged : {false, true}) {
        split_config config{count, active};
        std::vector<ggml_backend_dev_t> devices;
        for (int j = 0; j < count; ++j) {
            auto device = ggml_backend_dev_by_name(("CPU-NUMA" + std::to_string(j)).c_str());
            if (!device) return 2;
            devices.push_back(device);
        }
        auto meta = ggml_backend_meta_device(devices.data(), devices.size(), split_leaf, &config);
        auto backend = ggml_backend_dev_init(meta, nullptr);
        auto ctx = ggml_init({4 * 1024 * 1024, nullptr, true});
        const int k = 64;
        auto w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, rows);
        auto v = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, rows);
        auto x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, tokens);
        ggml_set_name(w, "split.weight");
        ggml_set_name(v, "split.second_weight");
        ggml_set_name(x, "split.input");
        ggml_set_input(x);
        auto storage = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!storage) return 2;
        std::vector<float> weights(k * rows), second(k * rows), input(k * tokens);
        for (int i = 0; i < k * rows; ++i) {
            weights[i] = float(i % 19 - 9) / 512;
            second[i] = float(i % 17 - 8) / 512;
        }
        ggml_backend_tensor_set(w, weights.data(), 0, ggml_nbytes(w));
        ggml_backend_tensor_set(v, second.data(), 0, ggml_nbytes(v));
        auto graph = ggml_new_graph(ctx);
        auto product = ggml_mul_mat(ctx, w, x);
        ggml_build_forward_expand(graph, product);
        auto second_product = merged ? ggml_mul_mat(ctx, v, x) : nullptr;
        if (second_product) ggml_build_forward_expand(graph, second_product);
        auto output = ggml_scale(ctx, ggml_rms_norm(ctx, product, 1e-5f), 0.75f);
        if (second_product) output = ggml_add(ctx, output, ggml_rms_norm(ctx, second_product, 1e-5f));
        ggml_set_output(output);
        ggml_build_forward_expand(graph, output);
        auto allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(allocator, graph)) return 2;
        bool okay = true;
        float max_error = 0;
        uint64_t checksum = 14695981039346656037ULL;
        std::vector<float> final_checked_values;
        for (int iteration = 0; iteration < 5; ++iteration) {
            for (int i = 0; i < k * tokens; ++i) input[i] = float(i % 13 - 6 + iteration) / 32;
            ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));
            if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return 2;
            ggml_backend_synchronize(backend);
            std::vector<float> values(rows * tokens);
            ggml_backend_tensor_get(output, values.data(), 0, ggml_nbytes(output));
            for (int t = 0; t < tokens; ++t) {
                std::vector<double> a(rows), b(rows);
                double aa = 0, bb = 0;
                for (int r = 0; r < rows; ++r) {
                    for (int j = 0; j < k; ++j) {
                        a[r] += double(weights[r * k + j]) * input[t * k + j];
                        b[r] += double(second[r * k + j]) * input[t * k + j];
                    }
                    aa += a[r] * a[r]; bb += b[r] * b[r];
                }
                for (int r = 0; r < rows; ++r) {
                    const double expected = 0.75 * a[r] / std::sqrt(aa / rows + 1e-5f)
                        + (merged ? b[r] / std::sqrt(bb / rows + 1e-5f) : 0);
                    const float delta = std::abs(values[t * rows + r] - expected);
                    max_error = std::max(max_error, delta);
                    okay &= std::isfinite(values[t * rows + r]) && delta <= 2e-4 * (1 + std::abs(expected));
                }
            }
            const auto * bytes = reinterpret_cast<const uint8_t *>(values.data());
            for (size_t i = 0; i < values.size() * sizeof(float); ++i) checksum = (checksum ^ bytes[i]) * 1099511628211ULL;
            if (iteration == 4) final_checked_values = values;
        }
        double median_ms = 0;
        if (timing_repeats) {
            for (int i = 0; i < 20; ++i) {
                if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return 2;
            }
            std::vector<double> samples;
            for (int i = 0; i < timing_repeats; ++i) {
                const auto start = std::chrono::steady_clock::now();
                if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return 2;
                ggml_backend_synchronize(backend);
                samples.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
            }
            std::sort(samples.begin(), samples.end());
            median_ms = 0.5 * (samples[(timing_repeats - 1) / 2] + samples[timing_repeats / 2]);
            std::vector<float> values(final_checked_values.size());
            ggml_backend_tensor_get(output, values.data(), 0, ggml_nbytes(output));
            okay &= std::memcmp(values.data(), final_checked_values.data(), ggml_nbytes(output)) == 0;
        }
        std::printf("%s fused_graph devices=%d active=%u rows=%d tokens=%d merged=%d max_error=%g graph_ms=%.6f hash=%016llx\n",
                    okay ? "PASS" : "FAIL", count, active, rows, tokens, merged, max_error, median_ms, (unsigned long long) checksum);
        std::fflush(stdout);
        failures += !okay;
        ++cases;
        ggml_gallocr_free(allocator);
        ggml_backend_buffer_free(storage);
        ggml_free(ctx);
        ggml_backend_free(backend);
    }
    std::printf("Fused graph: %d cases, %d failures\n", cases, failures);
    return failures ? 1 : 0;
}

int main(int argc, char ** argv) {
    Dl_info barrier_runtime{};
    if (!dladdr(reinterpret_cast<void *>(ggml_backend_cpu_init), &barrier_runtime)) std::abort();
    std::printf("BARRIER_CPU_LIBRARY %s\n", barrier_runtime.dli_fname);
    std::atexit([] {
        using counter_fn = uint64_t (*)(int);
        const auto counter = reinterpret_cast<counter_fn>(dlsym(RTLD_DEFAULT, "ggml_cpu_dissemination_count"));
        const uint64_t count = counter ? counter(0) : 0;
        const char * flag = std::getenv("GGML_CPU_DISSEMINATION_BARRIER");
        const char * audit = std::getenv("GGML_CPU_DISSEMINATION_AUDIT");
        const bool enabled = flag && flag[0] == '1';
        const bool auditing = audit && audit[0] == '1';
        std::printf("DISSEMINATION_CALLS %llu\n", (unsigned long long) count);
        if (auditing && ((!enabled && count != 0) || (enabled && true && count == 0))) std::abort();
    });

    if (argc == 2 && std::string(argv[1]) == "--fused-graph") return check_fused_graph();
    if (argc != 1) return 2;
    setenv("GGML_CPU_NUMA_DEVICES", "1", 1);
    setenv("GGML_CPU_NUMA_DIRECT_ALLREDUCE", "1", 1);
    ggml_backend_load_all();
    int failures = 0;
    for (size_t count : {2, 4}) {
        std::vector<ggml_backend_t> backends;
        std::vector<ggml_context *> contexts;
        std::vector<ggml_backend_buffer_t> buffers;
        std::vector<ggml_tensor *> tensors;
        for (size_t j = 0; j < count; ++j) {
            const std::string name = "CPU-NUMA" + std::to_string(j);
            auto device = ggml_backend_dev_by_name(name.c_str());
            if (!device) { std::fprintf(stderr, "missing %s\n", name.c_str()); return 2; }
            auto backend = ggml_backend_dev_init(device, nullptr);
            auto context = ggml_init({ggml_tensor_overhead() + 1024, nullptr, true});
            auto tensor = ggml_new_tensor_1d(context, GGML_TYPE_F32, 16);
            auto buffer = ggml_backend_alloc_ctx_tensors(context, backend);
            backends.push_back(backend);
            contexts.push_back(context);
            tensors.push_back(tensor);
            buffers.push_back(buffer);
        }
        auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backends[0]));
        auto init = (ggml_backend_comm_init_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_init");
        auto reduce = (ggml_backend_comm_allreduce_tensor_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_allreduce_tensor");
        auto release = (ggml_backend_comm_free_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_free");
        if (!init || !reduce || !release) return 2;
        auto comm = init(backends.data(), count);
        if (!comm) return 2;
        for (unsigned mask = 0; mask < (1u << count); ++mask) {
            float expected[16] = {};
            for (size_t j = 0; j < count; ++j) {
                bool active = mask & (1u << j);
                tensors[j]->flags = active ? GGML_TENSOR_FLAG_COMPUTE : 0;
                float values[16];
                for (int k = 0; k < 16; ++k) {
                    values[k] = active ? float((j + 1) * (k + 1)) : std::numeric_limits<float>::quiet_NaN();
                    if (active) expected[k] += values[k];
                }
                ggml_backend_tensor_set(tensors[j], values, 0, sizeof(values));
            }
            bool ok = reduce(comm, tensors.data());
            for (size_t j = 0; j < count; ++j) {
                float values[16];
                ggml_backend_tensor_get(tensors[j], values, 0, sizeof(values));
                for (int k = 0; k < 16; ++k) ok &= values[k] == expected[k];
            }
            std::printf("backends=%zu active_mask=%u %s\n", count, mask, ok ? "PASS" : "FAIL");
            failures += !ok;
        }
        release(comm);
        for (size_t j = 0; j < count; ++j) {
            ggml_backend_buffer_free(buffers[j]);
            ggml_free(contexts[j]);
            ggml_backend_free(backends[j]);
        }
    }
    return failures ? 1 : 0;
}
