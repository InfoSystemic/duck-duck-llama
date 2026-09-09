#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "repack.h"
#include "ggml-quants.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <random>
#include <vector>

struct shape { int k, rows, heads, broadcast; };

static std::vector<float> run(bool packed, shape s, int tokens, int threads, bool padded) {
    auto backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, threads);
    auto wc = ggml_init({ggml_tensor_overhead() * 4, nullptr, true});
    auto ic = ggml_init({ggml_tensor_overhead() * 4, nullptr, true});
    auto gc = ggml_init({ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true});
    auto w = ggml_new_tensor_3d(wc, GGML_TYPE_Q8_0, s.k, s.rows, s.heads);
    ggml_set_name(w, s.k == 256 ? "blk.3.attn_k_b.weight" : "blk.3.attn_v_b.weight");
    auto wb = ggml_backend_alloc_ctx_tensors_from_buft(wc,
        packed ? ggml_backend_cpu_repack_buffer_type() : ggml_backend_cpu_buffer_type());
    if (!wb || (packed && !w->extra)) std::abort();
    std::mt19937 rng(71993);
    std::normal_distribution<float> distribution(0, .02f);
    std::vector<float> row(s.k);
    std::vector<uint8_t> weights(ggml_nbytes(w));
    const size_t row_bytes = ggml_row_size(GGML_TYPE_Q8_0, s.k);
    for (int h = 0; h < s.heads; ++h) for (int r = 0; r < s.rows; ++r) {
        for (float & value : row) value = distribution(rng);
        quantize_row_q8_0_ref(row.data(), reinterpret_cast<block_q8_0 *>(
            weights.data() + (h * s.rows + r) * row_bytes), s.k);
    }
    ggml_backend_tensor_set(w, weights.data(), 0, weights.size());
    const int planes = s.heads * s.broadcast;
    auto storage = ggml_new_tensor_3d(ic, GGML_TYPE_F32, s.k + (padded ? 32 : 0), tokens, planes);
    auto x = padded ? ggml_view_3d(ic, storage, s.k, tokens, planes, storage->nb[1], storage->nb[2], 0) : storage;
    auto ib = ggml_backend_alloc_ctx_tensors_from_buft(ic, ggml_backend_cpu_buffer_type());
    if (!ib) std::abort();
    std::vector<float> activations(ggml_nelements(storage), -123.5f);
    for (int p = 0; p < planes; ++p) for (int t = 0; t < tokens; ++t) for (int k = 0; k < s.k; ++k) {
        activations[(p * tokens + t) * storage->ne[0] + k] =
            std::sin(float(k + 17 * t + 61 * p) * .0123f) + .3f * std::cos(float(k + 29 * p) * .079f);
    }
    ggml_backend_tensor_set(storage, activations.data(), 0, ggml_nbytes(storage));
    ggml_set_input(x);
    auto y = ggml_mul_mat(gc, w, x);
    ggml_set_output(y);
    auto graph = ggml_new_graph(gc);
    ggml_build_forward_expand(graph, y);
    auto alloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    if (!ggml_gallocr_alloc_graph(alloc, graph)) std::abort();
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) std::abort();
    std::vector<float> values(ggml_nelements(y));
    ggml_backend_tensor_get(y, values.data(), 0, ggml_nbytes(y));
    ggml_gallocr_free(alloc);
    ggml_backend_buffer_free(ib);
    ggml_backend_buffer_free(wb);
    ggml_free(gc); ggml_free(ic); ggml_free(wc);
    ggml_backend_free(backend);
    return values;
}

int main() {
    setenv("GGML_CPU_X16_Q8_0", "1", 1);
    setenv("GGML_CPU_X16_ATTN3D", "1", 1);
    setenv("GGML_CPU_Q8_0_REPACK", "1", 1);
    setenv("GGML_CPU_Q8_0_REPACK_FORCE", "1", 1);
    setenv("GGML_CPU_X16_CHUNK_MAX", "16", 1);
    ggml_backend_load_all();
    Dl_info info{};
    if (!dladdr(reinterpret_cast<void *>(&ggml_backend_cpu_init), &info)) std::abort();
    std::printf("CPU_LIBRARY %s\n", info.dli_fname);
    const std::vector<shape> shapes = {
        {256, 512, 16, 1}, {512, 256, 16, 1},
        {256, 512, 4, 2}, {512, 256, 4, 2},
        {256, 512, 64, 1}, {512, 256, 64, 1},
    };
    int cases = 0, failures = 0;
    for (auto s : shapes) for (int tokens : {1, 2, 3, 4, 5, 9})
    for (int threads : {1, 15}) for (bool padded : {false, true}) {
        const auto reference = run(false, s, tokens, threads, padded);
        const auto candidate = run(true, s, tokens, threads, padded);
        bool okay = reference.size() == candidate.size();
        float maximum = 0;
        for (size_t i = 0; i < reference.size(); ++i) {
            const float error = std::abs(reference[i] - candidate[i]) / (1 + std::abs(reference[i]));
            maximum = std::max(maximum, error);
            okay &= std::isfinite(candidate[i]) && error <= 2e-4f;
        }
        uint64_t digest = 14695981039346656037ULL;
        auto bytes = reinterpret_cast<const uint8_t *>(candidate.data());
        for (size_t i = 0; i < candidate.size() * sizeof(float); ++i) digest = (digest ^ bytes[i]) * 1099511628211ULL;
        std::printf("%s k=%d rows=%d heads=%d broadcast=%d tokens=%d threads=%d padded=%d max_scaled=%.9g hash=%016llx\n",
            okay ? "PASS" : "FAIL", s.k, s.rows, s.heads, s.broadcast, tokens, threads, padded, maximum, (unsigned long long)digest);
        std::fflush(stdout);
        ++cases; failures += !okay;
    }
    std::printf("Attention: %d cases, %d failures\n", cases, failures);
    return failures ? 1 : 0;
}
