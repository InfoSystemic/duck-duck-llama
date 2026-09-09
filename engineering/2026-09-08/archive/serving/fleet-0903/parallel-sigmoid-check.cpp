#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

static float decode(const unsigned char * data, ggml_type type) {
    if (type == GGML_TYPE_F32) { float x; std::memcpy(&x, data, 4); return x; }
    if (type == GGML_TYPE_F16) { ggml_fp16_t x; std::memcpy(&x, data, 2); return ggml_fp16_to_fp32(x); }
    ggml_bf16_t x; std::memcpy(&x, data, 2); return ggml_bf16_to_fp32(x);
}

static void encode(unsigned char * data, ggml_type type, float x) {
    if (type == GGML_TYPE_F32) { std::memcpy(data, &x, 4); return; }
    if (type == GGML_TYPE_F16) { auto y = ggml_fp32_to_fp16(x); std::memcpy(data, &y, 2); return; }
    auto y = ggml_fp32_to_bf16(x); std::memcpy(data, &y, 2);
}

static bool check(std::array<int64_t, 4> dims, ggml_type type, bool padded, bool inplace, int threads) {
    auto backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, threads);
    auto * ctx = ggml_init({ggml_tensor_overhead() * 10 + ggml_graph_overhead(), nullptr, true});
    const int64_t width = dims[0] + (padded ? 7 : 0);
    auto * parent = ggml_new_tensor_4d(ctx, type, width, dims[1], dims[2], dims[3]);
    auto * input = padded ? ggml_view_4d(ctx, parent, dims[0], dims[1], dims[2], dims[3],
        parent->nb[1], parent->nb[2], parent->nb[3], 2 * ggml_type_size(type)) : parent;
    auto * out = inplace ? ggml_sigmoid_inplace(ctx, input) : ggml_sigmoid(ctx, input);
    ggml_set_input(parent); ggml_set_output(out);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    auto alloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    bool ok = ggml_gallocr_alloc_graph(alloc, graph);
    const size_t item_size = ggml_type_size(type);
    std::vector<unsigned char> data(ggml_nbytes(parent));
    for (int64_t i = 0; i < ggml_nelements(parent); ++i) {
        float x = float((i * 17) % 1021 - 510) / 19.f;
        switch (i % 101) {
            case 0: x = std::numeric_limits<float>::infinity(); break;
            case 1: x = -std::numeric_limits<float>::infinity(); break;
            case 2: x = std::numeric_limits<float>::quiet_NaN(); break;
            case 3: x = 0.f; break;
            case 4: x = -0.f; break;
            case 5: x = 100.f; break;
            case 6: x = -100.f; break;
            case 7: x = std::numeric_limits<float>::denorm_min(); break;
        }
        encode(data.data() + i * item_size, type, x);
    }
    std::vector<unsigned char> expected(ggml_nelements(out) * item_size);
    for (int64_t i = 0; i < ggml_nelements(out); ++i) {
        const int64_t src_index = i % dims[0] + (i / dims[0]) * width + (padded ? 2 : 0);
        const float x = decode(data.data() + src_index * item_size, type);
        encode(expected.data() + i * item_size, type, 1.f / (1.f + std::exp(-x)));
    }
    ggml_backend_tensor_set(parent, data.data(), 0, data.size());
    ok &= ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    std::vector<unsigned char> actual(ggml_nbytes(out));
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size());
    for (int64_t i = 0; i < ggml_nelements(out); ++i) {
        const int64_t out_index = inplace && padded ? i % dims[0] + (i / dims[0]) * width : i;
        const auto * got = actual.data() + out_index * item_size;
        const auto * ref = expected.data() + i * item_size;
        ok &= std::memcmp(got, ref, item_size) == 0 || (std::isnan(decode(got, type)) && std::isnan(decode(ref, type)));
    }
    std::printf("%s type=%s width=%lld rows=%lld padded=%d inplace=%d threads=%d\n",
        ok ? "PASS" : "FAIL", ggml_type_name(type), (long long) dims[0],
        (long long) (dims[1] * dims[2] * dims[3]), padded, inplace, threads);
    ggml_gallocr_free(alloc); ggml_free(ctx); ggml_backend_free(backend);
    return ok;
}

int main() {
    int failed = 0, total = 0;
    for (auto dims : {std::array<int64_t, 4>{1, 1, 1, 1}, {10240, 1, 1, 1}, {10240, 4, 1, 1}, {17, 3, 2, 2}})
    for (auto type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16})
    for (bool padded : {false, true}) for (bool inplace : {false, true}) for (int threads : {1, 15}) {
        failed += !check(dims, type, padded, inplace, threads); ++total;
    }
    std::printf("%d/%d passed\n", total-failed, total);
    return failed ? 1 : 0;
}
