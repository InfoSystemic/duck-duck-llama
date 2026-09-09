#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <dlfcn.h>
#include <memory>
#include <sched.h>
#include <set>
#include <string>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>
#include <vector>

#define REQUIRE(value) do { if (!(value)) { std::fprintf(stderr, "CHECK FAILED line=%d: %s\n", __LINE__, #value); std::abort(); } } while (0)

static constexpr int width = 2560;
static constexpr int hidden = 640;
static constexpr int layers = 3;

static size_t thread_count() {
    DIR * dir = opendir("/proc/self/task");
    REQUIRE(dir != nullptr);
    size_t count = 0;
    while (dirent * entry = readdir(dir)) {
        count += entry->d_name[0] != '.';
    }
    REQUIRE(closedir(dir) == 0);
    return count;
}

static void expect_thread_count(size_t expected) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (thread_count() != expected && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    REQUIRE(thread_count() == expected);
}

struct marker_state {
    std::array<std::atomic<int>, 128> tids;
    std::atomic<int> calls{0};
    std::atomic<int> held_workers{0};
    std::atomic<bool> hold{false};

    marker_state() {
        for (auto & tid : tids) {
            tid.store(0);
        }
    }
};

static void mark_workers(ggml_tensor * dst, const ggml_tensor * src, int ith, int nth, void * userdata) {
    auto & marker = *static_cast<marker_state *>(userdata);
    REQUIRE(nth == 15 && ith >= 0 && ith < nth);
    const int cpu = sched_getcpu();
    REQUIRE(cpu >= 0 && cpu < 64 && cpu % 16 < 15);
    const int tid = int(syscall(SYS_gettid));
    int previous = 0;
    REQUIRE(marker.tids[cpu].compare_exchange_strong(previous, tid) || previous == tid);
    marker.calls.fetch_add(1);
    if (marker.hold.load()) {
        marker.held_workers.fetch_add(1);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (marker.hold.load()) {
            REQUIRE(std::chrono::steady_clock::now() < deadline);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    REQUIRE(src->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    REQUIRE(ggml_is_contiguous(src) && ggml_is_contiguous(dst));
    const int64_t count = ggml_nelements(src);
    REQUIRE(count == ggml_nelements(dst));
    const int64_t first = count * ith / nth;
    const int64_t last = count * (ith + 1) / nth;
    std::memcpy(static_cast<float *>(dst->data) + first, static_cast<const float *>(src->data) + first,
                size_t(last - first) * sizeof(float));
}

static ggml_backend_meta_split_state split_leaf(const ggml_tensor * tensor, void *) {
    ggml_backend_meta_split_state split = {GGML_BACKEND_SPLIT_AXIS_MIRRORED, {0}, {1}, 1};
    if (std::strncmp(tensor->name, "up.", 3) == 0 || std::strncmp(tensor->name, "down.", 5) == 0) {
        split.axis = tensor->name[0] == 'u' ? GGML_BACKEND_SPLIT_AXIS_1 : GGML_BACKEND_SPLIT_AXIS_0;
        for (int rank = 0; rank < 4; ++rank) {
            split.ne[rank] = hidden / 4;
        }
    }
    return split;
}

static int up_index(int row, int layer, int tag) { return (row * 7 + layer * 13 + tag * 17) % width; }
static int down_index(int row, int layer, int tag) { return (row * 11 + layer * 19 + tag * 7) % hidden; }
static float up_scale(int row, int layer, int tag) { return std::ldexp(1.0f, -1 - (row + layer + tag) % 2); }
static float down_scale(int row, int layer, int tag) { return std::ldexp(1.0f, -1 - (row + layer + 2 * tag) % 2); }

struct context {
    int tokens;
    int tag;
    marker_state marker;
    ggml_backend_t backend = nullptr;
    ggml_context * weights_ctx = nullptr;
    ggml_context * inputs_ctx = nullptr;
    ggml_context * graph_ctx = nullptr;
    ggml_backend_buffer_t weights_buffer = nullptr;
    ggml_backend_buffer_t inputs_buffer = nullptr;
    ggml_gallocr_t allocator = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;

    context(ggml_backend_dev_t meta, int count, int identity) : tokens(count), tag(identity) {
        backend = ggml_backend_dev_init(meta, nullptr);
        REQUIRE(backend != nullptr);
        weights_ctx = ggml_init({64 * ggml_tensor_overhead(), nullptr, true});
        inputs_ctx = ggml_init({8 * ggml_tensor_overhead(), nullptr, true});
        graph_ctx = ggml_init({4 * 1024 * 1024, nullptr, true});
        REQUIRE(weights_ctx && inputs_ctx && graph_ctx);
        std::array<ggml_tensor *, layers> up{}, down{};
        for (int layer = 0; layer < layers; ++layer) {
            up[layer] = ggml_new_tensor_2d(weights_ctx, GGML_TYPE_F32, width, hidden);
            down[layer] = ggml_new_tensor_2d(weights_ctx, GGML_TYPE_F32, hidden, width);
            ggml_set_name(up[layer], ("up." + std::to_string(layer)).c_str());
            ggml_set_name(down[layer], ("down." + std::to_string(layer)).c_str());
        }
        weights_buffer = ggml_backend_alloc_ctx_tensors(weights_ctx, backend);
        REQUIRE(weights_buffer != nullptr);
        ggml_backend_buffer_set_usage(weights_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        input = ggml_new_tensor_2d(inputs_ctx, GGML_TYPE_F32, width, tokens);
        ggml_set_name(input, "input");
        ggml_set_input(input);
        inputs_buffer = ggml_backend_alloc_ctx_tensors(inputs_ctx, backend);
        REQUIRE(inputs_buffer != nullptr);
        graph = ggml_new_graph(graph_ctx);
        output = input;
        for (int layer = 0; layer < layers; ++layer) {
            auto projected = ggml_mul_mat(graph_ctx, up[layer], output);
            auto partial = ggml_mul_mat(graph_ctx, down[layer], projected);
            if (layer == layers - 1) {
                partial = ggml_map_custom1(graph_ctx, partial, mark_workers, GGML_N_TASKS_MAX, &marker);
                ggml_set_name(partial, "partial_worker_marker");
            }
            output = ggml_add(graph_ctx, partial, output);
        }
        ggml_set_output(output);
        ggml_build_forward_expand(graph, output);
        allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        REQUIRE(allocator && ggml_gallocr_alloc_graph(allocator, graph));
        for (int layer = 0; layer < layers; ++layer) {
            std::vector<float> up_weights(size_t(width) * hidden, 0.0f);
            std::vector<float> down_weights(size_t(hidden) * width, 0.0f);
            for (int row = 0; row < hidden; ++row) {
                up_weights[size_t(row) * width + up_index(row, layer, tag)] = up_scale(row, layer, tag);
            }
            for (int row = 0; row < width; ++row) {
                down_weights[size_t(row) * hidden + down_index(row, layer, tag)] = down_scale(row, layer, tag);
            }
            ggml_backend_tensor_set(up[layer], up_weights.data(), 0, ggml_nbytes(up[layer]));
            ggml_backend_tensor_set(down[layer], down_weights.data(), 0, ggml_nbytes(down[layer]));
        }
    }

    ~context() {
        ggml_backend_free(backend);
        ggml_gallocr_free(allocator);
        ggml_backend_buffer_free(inputs_buffer);
        ggml_backend_buffer_free(weights_buffer);
        ggml_free(graph_ctx);
        ggml_free(inputs_ctx);
        ggml_free(weights_ctx);
    }

    std::vector<float> run(int probe) {
        std::vector<float> values(size_t(width) * tokens);
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = float(1 + (i * 13 + probe * 17 + tag * 29) % 31) / 32.0f;
        }
        ggml_backend_tensor_set(input, values.data(), 0, ggml_nbytes(input));
        auto expected = values;
        for (int layer = 0; layer < layers; ++layer) {
            auto next = expected;
            for (int token = 0; token < tokens; ++token) {
                for (int row = 0; row < width; ++row) {
                    const int h = down_index(row, layer, tag);
                    const float projected = expected[size_t(token) * width + up_index(h, layer, tag)] * up_scale(h, layer, tag);
                    next[size_t(token) * width + row] += projected * down_scale(row, layer, tag);
                }
            }
            expected.swap(next);
        }
        const int calls_before = marker.calls.load();
        REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(output, values.data(), 0, ggml_nbytes(output));
        REQUIRE(marker.calls.load() - calls_before == 60);
        REQUIRE(std::memcmp(values.data(), expected.data(), values.size() * sizeof(float)) == 0);
        return values;
    }
};

static void compare_teams(const context & a, const context & b, bool shared) {
    std::set<int> a_tids, b_tids;
    for (int cpu = 0; cpu < 128; ++cpu) {
        const int ta = a.marker.tids[cpu].load(), tb = b.marker.tids[cpu].load();
        if (cpu < 64 && cpu % 16 < 15) {
            REQUIRE(ta > 0 && tb > 0);
            REQUIRE((ta == tb) == shared);
            a_tids.insert(ta);
            b_tids.insert(tb);
        } else {
            REQUIRE(ta == 0 && tb == 0);
        }
    }
    REQUIRE(a_tids.size() == 60 && b_tids.size() == 60);
    if (!shared) {
        for (int tid : a_tids) {
            REQUIRE(b_tids.count(tid) == 0);
        }
    }
}

int main(int argc, char ** argv) {
    REQUIRE(argc == 3);
    const bool shared = std::atoi(argv[1]) != 0;
    Dl_info cpu_library{}, base_library{};
    REQUIRE(dladdr(reinterpret_cast<void *>(ggml_backend_cpu_init), &cpu_library));
    REQUIRE(dladdr(reinterpret_cast<void *>(ggml_backend_graph_compute), &base_library));
    std::printf("LIFECYCLE_CPU_LIBRARY %s\nLIFECYCLE_BASE_LIBRARY %s\n", cpu_library.dli_fname, base_library.dli_fname);
    ggml_backend_load_all();
    std::array<ggml_backend_dev_t, 4> devices{};
    for (int rank = 0; rank < 4; ++rank) {
        devices[rank] = ggml_backend_dev_by_name(("CPU-NUMA" + std::to_string(rank)).c_str());
        REQUIRE(devices[rank] != nullptr);
    }
    auto meta = ggml_backend_meta_device(devices.data(), devices.size(), split_leaf, nullptr);
    REQUIRE(meta != nullptr);
    cpu_set_t controller;
    CPU_ZERO(&controller);
    for (int cpu : {15, 31, 47, 63}) {
        CPU_SET(cpu, &controller);
    }
    REQUIRE(sched_setaffinity(0, sizeof(controller), &controller) == 0);
    const size_t baseline_threads = thread_count();
    FILE * file = std::fopen(argv[2], "wb");
    REQUIRE(file != nullptr);
    size_t output_values = 0, runs = 0;
    auto record = [&](const std::vector<float> & values) {
        REQUIRE(std::fwrite(values.data(), sizeof(float), values.size(), file) == values.size());
        output_values += values.size();
        ++runs;
    };
    auto a = std::make_unique<context>(meta, 1, 1);
    auto b = std::make_unique<context>(meta, 5, 2);
    for (int round = 0; round < 2; ++round) {
        record(a->run(round));
        record(b->run(round));
    }
    compare_teams(*a, *b, shared);
    const size_t active_threads = thread_count();
    for (int round = 0; round < 8; ++round) {
        std::vector<float> out_a, out_b;
        std::atomic<int> ready{0};
        std::atomic<bool> start{false};
        auto submit = [&](context & ctx, int probe, std::vector<float> & output) {
            ready.fetch_add(1);
            while (!start.load()) {
                std::this_thread::yield();
            }
            output = ctx.run(probe);
        };
        std::thread producer_a([&]() { submit(*a, 10 + round, out_a); });
        std::thread producer_b([&]() { submit(*b, 20 + round, out_b); });
        while (ready.load() != 2) {
            std::this_thread::yield();
        }
        start.store(true);
        producer_a.join();
        producer_b.join();
        record(out_a);
        record(out_b);
    }
    compare_teams(*a, *b, shared);
    b->marker.hold.store(true);
    std::vector<float> held_output;
    std::thread producer([&]() { held_output = b->run(77); });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (b->marker.held_workers.load() != 60) {
        REQUIRE(std::chrono::steady_clock::now() < deadline);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    a.reset();
    b->marker.hold.store(false);
    producer.join();
    record(held_output);
    auto c = std::make_unique<context>(meta, 3, 3);
    record(c->run(0));
    record(b->run(78));
    compare_teams(*b, *c, shared);
    c.reset();
    b.reset();
    expect_thread_count(baseline_threads);
    auto d = std::make_unique<context>(meta, 9, 4);
    record(d->run(88));
    d.reset();
    expect_thread_count(baseline_threads);
    REQUIRE(std::fclose(file) == 0);
    REQUIRE(runs == 24 && output_values == 209920);
    std::printf("{\"passed\":true,\"runs\":%zu,\"exact_values\":%zu,\"shared_workers\":%s,\"workers_per_context\":60,\"baseline_threads\":%zu,\"active_threads_two_contexts\":%zu,\"final_threads\":%zu,\"concurrent_rounds\":8,\"destroyed_context_while_peer_workers_held\":true}\n",
                runs, output_values, shared ? "true" : "false", baseline_threads, active_threads, thread_count());
    return 0;
}
