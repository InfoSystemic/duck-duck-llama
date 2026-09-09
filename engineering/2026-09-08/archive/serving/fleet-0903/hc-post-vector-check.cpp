#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>

static float value(const ggml_tensor * t, int64_t i, int64_t j, int64_t k = 0) {
    return *(const float *) ((const char *) t->data + i*t->nb[0] + j*t->nb[1] + k*t->nb[2]);
}

int main() {
    ggml_backend_load_all();
    auto backend = ggml_backend_cpu_init();
    int cases = 0, failures = 0;
    for (int hc : {3, 4}) for (int width : {31, 3840}) for (int tokens : {1, 4, 17})
    for (int threads : {1, 15}) for (bool padded : {false, true}) {
        ggml_backend_cpu_set_n_threads(backend, threads);
        auto ctx = ggml_init({32*1024*1024, nullptr, false});
        auto make = [&](int64_t n0, int64_t n1, int64_t n2, int seed) {
            auto storage = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n0 + (padded ? 7 : 0), n1, n2);
            auto data = (float *) storage->data;
            for (int64_t i = 0; i < ggml_nelements(storage); ++i) {
                data[i] = 1.7f * std::sin(float(i + seed) * 0.131f) + 0.3f * std::cos(float(i + seed) * 0.017f);
            }
            return padded ? ggml_view_3d(ctx, storage, n0, n1, n2, storage->nb[1], storage->nb[2], 0) : storage;
        };
        auto x = make(width, tokens, 1, 13);
        auto residual = make(width, hc, tokens, 29);
        auto post = make(hc, tokens, 1, 67);
        auto comb = make(hc, hc, tokens, 97);
        auto out = ggml_dsv4_hc_post(ctx, x, residual, post, comb);
        auto graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, out);
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return 2;
        auto start = std::chrono::steady_clock::now();
        for (int r = 0; r < 20; ++r) {
            if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return 2;
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-start).count()/20;
        bool okay = true;
        float max_abs = 0, max_scaled = 0;
        uint64_t checksum = 14695981039346656037ULL;
        for (int t = 0; t < tokens; ++t) for (int d = 0; d < hc; ++d) for (int i = 0; i < width; ++i) {
            float ref = value(x,i,t) * value(post,d,t);
            for (int j = 0; j < hc; ++j) ref = std::fma(value(residual,i,j,t),value(comb,d,j,t),ref);
            const float got = value(out,i,d,t);
            const float delta = std::abs(ref-got);
            max_abs = std::max(max_abs,delta);
            max_scaled = std::max(max_scaled,delta/(1+std::abs(ref)));
            okay &= std::isfinite(got) && delta <= 2e-6f*(1+std::abs(ref));
            const auto * bytes = reinterpret_cast<const uint8_t *>(&got);
            for (int b = 0; b < 4; ++b) checksum = (checksum ^ bytes[b]) * 1099511628211ULL;
        }
        ++cases; failures += !okay;
        std::printf("%s hc=%d width=%d tokens=%d threads=%d padded=%d max_abs=%.8g max_scaled=%.8g ms=%.6f hash=%016llx\n",
                    okay ? "PASS" : "FAIL",hc,width,tokens,threads,padded,max_abs,max_scaled,ms,(unsigned long long)checksum);
        std::fflush(stdout);
        ggml_free(ctx);
    }
    ggml_backend_free(backend);
    std::printf("HC_POST: %d cases, %d failures\n",cases,failures);
    return failures ? 1 : 0;
}
