#include "ops.h"
#include <dlfcn.h>
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

struct fixture {
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_context * weights = ggml_init({ggml_tensor_overhead() * 16, nullptr, true});
    ggml_context * graph_ctx = ggml_init({ggml_tensor_overhead() * 32 + ggml_graph_overhead(), nullptr, true});
    ggml_backend_buffer_t input_buffer = nullptr;
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    std::vector<ggml_tensor *> inputs;
    std::vector<std::vector<uint32_t>> input_values;

    ~fixture() {
        ggml_gallocr_free(alloc);
        ggml_backend_buffer_free(input_buffer);
        ggml_free(graph_ctx);
        ggml_free(weights);
        ggml_backend_free(backend);
    }

    ggml_tensor * add(ggml_type type, std::array<int64_t, 4> dims, uint32_t seed) {
        auto * tensor = ggml_new_tensor_4d(weights, type, dims[0], dims[1], dims[2], dims[3]);
        ggml_set_input(tensor);
        inputs.push_back(tensor);
        std::vector<uint32_t> values(ggml_nelements(tensor));
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = (uint32_t(i + seed) * 2654435761U & 0x807fffffU) | 0x3f000000U;
            if (i % 101 == 0) values[i] = 0x7fc12345U;
        }
        input_values.push_back(std::move(values));
        return tensor;
    }

    bool check(ggml_tensor * output, const std::vector<uint32_t> & expected, int threads, const char * label) {
        ggml_backend_cpu_set_n_threads(backend, threads);
        input_buffer = ggml_backend_alloc_ctx_tensors(weights, backend);
        if (!input_buffer) return false;
        auto * graph = ggml_new_graph(graph_ctx);
        ggml_set_output(output);
        ggml_build_forward_expand(graph, output);
        if (!ggml_gallocr_alloc_graph(alloc, graph)) return false;
        for (size_t i = 0; i < inputs.size(); ++i) {
            ggml_backend_tensor_set(inputs[i], input_values[i].data(), 0, ggml_nbytes(inputs[i]));
        }
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return false;
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < 8; ++i) {
            if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return false;
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / 8;
        std::vector<uint32_t> actual(ggml_nelements(output));
        ggml_backend_tensor_get(output, actual.data(), 0, ggml_nbytes(output));
        const bool ok = actual == expected;
        std::printf("%s %s threads=%d elements=%zu ms=%.4f\n", ok ? "PASS" : "FAIL", label, threads, actual.size(), ms);
        std::fflush(stdout);
        return ok;
    }
};

static bool concat_case(int dim, bool strided, bool large, int threads) {
    fixture f;
    std::array<int64_t, 4> a = large ? std::array<int64_t, 4>{3, 6144, 1, 1}
                                           : std::array<int64_t, 4>{5, 7, 3, 2};
    auto b = a;
    a[dim] = large ? 3 : 2;
    b[dim] = large ? 1 : 3;
    auto make = [&](std::array<int64_t, 4> dims, uint32_t seed) {
        if (strided) std::swap(dims[0], dims[1]);
        auto * parent = f.add(GGML_TYPE_F32, dims, seed);
        return strided ? ggml_permute(f.graph_ctx, parent, 1, 0, 2, 3) : parent;
    };
    auto * x = make(a, 17);
    auto * y = make(b, 89);
    auto * out = ggml_concat(f.graph_ctx, x, y, dim);
    std::vector<uint32_t> expected(ggml_nelements(out));
    for (size_t i = 0; i < expected.size(); ++i) {
        size_t rest = i;
        int64_t coord[4];
        for (int k = 0; k < 4; ++k) { coord[k] = rest % out->ne[k]; rest /= out->ne[k]; }
        int which = 0;
        if (coord[dim] >= a[dim]) { which = 1; coord[dim] -= a[dim]; }
        auto * src = which ? y : x;
        size_t offset = 0;
        for (int k = 0; k < 4; ++k) offset += coord[k] * src->nb[k];
        expected[i] = f.input_values[which][offset / sizeof(uint32_t)];
    }
    char label[128];
    std::snprintf(label, sizeof(label), "concat dim=%d strided=%d large=%d", dim, strided, large);
    return f.check(out, expected, threads, label);
}

static bool gather_case(int64_t nc, int64_t nr, bool padded, int threads, int64_t groups = 2) {
    fixture f;
    const int64_t physical_nc = nc + (padded ? 16 : 0);
    const int64_t physical_nr = nr + (padded ? 1 : 0);
    auto * source = f.add(GGML_TYPE_F32, {physical_nc, 3, groups, groups}, 137);
    auto * indices = f.add(GGML_TYPE_I32, {physical_nr, groups, groups, 1}, 0);
    for (size_t i = 0; i < f.input_values[1].size(); ++i) f.input_values[1][i] = (i + 1) % 3;
    auto * src = padded ? ggml_view_4d(f.graph_ctx, source, nc, 3, groups, groups,
        source->nb[1], source->nb[2], source->nb[3], 2 * sizeof(float)) : source;
    auto * ids = padded ? ggml_view_3d(f.graph_ctx, indices, nr, groups, groups,
        indices->nb[1], indices->nb[2], sizeof(int32_t)) : indices;
    auto * out = ggml_get_rows(f.graph_ctx, src, ids);
    std::vector<uint32_t> expected(ggml_nelements(out));
    for (int64_t i3 = 0; i3 < groups; ++i3) for (int64_t i2 = 0; i2 < groups; ++i2) for (int64_t i1 = 0; i1 < nr; ++i1) {
        const int64_t id_offset = i1 + physical_nr * (i2 + groups * i3) + (padded ? 1 : 0);
        const int64_t row = f.input_values[1][id_offset];
        const int64_t source_offset = physical_nc * (row + 3 * (i2 + groups * i3)) + (padded ? 2 : 0);
        const int64_t output_offset = nc * (i1 + nr * (i2 + groups * i3));
        std::memcpy(expected.data() + output_offset, f.input_values[0].data() + source_offset, nc * sizeof(float));
    }
    char label[128];
    std::snprintf(label, sizeof(label), "gather nc=%lld nr=%lld groups=%lld padded=%d", (long long) nc, (long long) nr, (long long) groups, padded);
    return f.check(out, expected, threads, label);
}

int main() {
    Dl_info loaded = {};
    GGML_ASSERT(dladdr(reinterpret_cast<void *>(ggml_compute_forward_get_rows), &loaded));
    std::printf("GATHER_LIBRARY %s\n", loaded.dli_fname);
    ggml_backend_load_all();
    int failures = 0, cases = 0;
    for (int threads : {1, 15}) {
        for (int dim = 0; dim < 4; ++dim) for (bool strided : {false, true}) {
            ++cases; failures += !concat_case(dim, strided, false, threads);
        }
        ++cases; failures += !concat_case(0, false, true, threads);
        ++cases; failures += !gather_case(262144, 1, true, threads, 1);
        for (int64_t nc : {17, 4096, 262144}) for (int64_t nr : {1, 3}) for (bool padded : {false, true}) {
            ++cases; failures += !gather_case(nc, nr, padded, threads);
        }
    }
    for (int64_t nr : {1, 5}) { ++cases; failures += !gather_case(786432, nr, true, 15, 1); }
    std::printf("Parallel copy: %d cases, %d failures\n", cases, failures);
    return failures ? 1 : 0;
}
