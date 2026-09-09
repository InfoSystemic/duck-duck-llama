#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static constexpr uint32_t sentinel = 0x7fc12345;

struct test_case {
    int width = 4, pools = 66, streams = 1, threads = 15;
    bool padded = false, swapped = false, output_probs = false, output_product = false;
    bool extra_use = false, scaled = false, masked = false, reference = false, inplace = false;
};

static bool run_case(const test_case & c, FILE * dump, bool timing, int id) {
    ggml_context * ctx = ggml_init({128U * 1024U * 1024U, nullptr, false});
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, c.threads);
    ggml_backend_cpu_set_use_ref(backend, c.reference);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    struct group { ggml_tensor * gate; ggml_tensor * keys; ggml_tensor * probs; ggml_tensor * product; ggml_tensor * out; ggml_tensor * mask; ggml_tensor * extra; };
    std::vector<group> groups;
    const int count = timing ? 11 : 1;
    for (int i = 0; i < count; ++i) {
        auto * parent = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.width + (c.padded ? 4 : 0), 128, c.pools, c.streams);
        auto * gate = c.padded ? ggml_view_4d(ctx, parent, c.width, 128, c.pools, c.streams,
            parent->nb[1], parent->nb[2], parent->nb[3], sizeof(float)) : parent;
        auto * keys = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.width, 128, c.pools, c.streams);
        auto * mask = c.masked ? ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.width, 128, 1, 1) : nullptr;
        auto * probs = c.inplace ? ggml_soft_max_inplace(ctx, gate)
            : ggml_soft_max_ext(ctx, gate, mask, c.scaled ? 0.5f : 1.0f, 0.0f);
        auto * product = c.swapped ? ggml_mul(ctx, probs, keys) : ggml_mul(ctx, keys, probs);
        auto * out = ggml_sum_rows(ctx, product);
        if (c.output_probs) ggml_set_output(probs);
        if (c.output_product) ggml_set_output(product);
        ggml_set_output(out);
        ggml_build_forward_expand(graph, out);
        auto * extra = c.extra_use ? ggml_scale(ctx, probs, 2.0f) : nullptr;
        if (extra) { ggml_set_output(extra); ggml_build_forward_expand(graph, extra); }
        groups.push_back({gate, keys, probs, product, out, mask, extra});
    }
    bool good = true;
    int fused = 0;
    const int samples = timing ? 1 : 3;
    for (int sample = 0; sample < samples; ++sample) {
        std::vector<std::vector<float>> original_gates;
        for (auto & g : groups) {
            std::vector<float> gate_values(ggml_nelements(g.keys));
            for (int64_t row = 0; row < ggml_nrows(g.keys); ++row) {
                const int64_t i1 = row % 128, i2 = row / 128 % c.pools, i3 = row / (128 * c.pools);
                auto * gate = (float *) ((char *) g.gate->data + i1*g.gate->nb[1] + i2*g.gate->nb[2] + i3*g.gate->nb[3]);
                for (int k = 0; k < c.width; ++k) {
                    size_t ix = row*c.width + k;
                    float value = float(int((ix*37 + sample*13) % 401) - 200) / (sample == 0 ? 8.0f : 1.0f);
                    if (row % 17 == 0) value = k & 1 ? -0.0f : 0.0f;
                    gate[k] = gate_values[ix] = value;
                    ((float *)g.keys->data)[ix] = float(int((ix*53 + sample*71) % 257) - 128) / 8.0f;
                }
            }
            original_gates.push_back(std::move(gate_values));
            if (g.mask) for (int64_t i = 0; i < ggml_nelements(g.mask); ++i) ((float *)g.mask->data)[i] = (i % 3) * -0.125f;
            if (!c.inplace) for (int64_t i = 0; i < ggml_nelements(g.probs); ++i) std::memcpy((char *)g.probs->data + i*4, &sentinel, 4);
            for (int64_t i = 0; i < ggml_nelements(g.product); ++i) std::memcpy((char *)g.product->data + i*4, &sentinel, 4);
        }
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) good = false;
        for (size_t j = 0; j < groups.size(); ++j) {
            auto & g = groups[j];
            uint32_t p, m;
            std::memcpy(&p, g.probs->data, 4); std::memcpy(&m, g.product->data, 4);
            const bool untouched = p == sentinel && m == sentinel;
            const bool requested = std::getenv("GGML_CPU_SOFTMAX_POOL_FUSION") && std::atoi(std::getenv("GGML_CPU_SOFTMAX_POOL_FUSION")) == 1;
            const bool disabled = std::getenv("GGML_CPU_DISABLE_FUSION") && std::atoi(std::getenv("GGML_CPU_DISABLE_FUSION")) == 1;
            const bool expected_fused = requested && !disabled && !c.reference && c.threads > 1 &&
                c.width == 4 && ggml_nelements(g.probs) > 4096 && !c.padded && !c.output_probs &&
                !c.output_product && !c.extra_use && !c.scaled && !c.masked && !c.inplace;
            if (untouched != expected_fused) good = false;
            fused += untouched;
            for (int64_t row = 0; row < ggml_nrows(g.keys); ++row) {
                double mx = -INFINITY, sum = 0, weighted = 0;
                double values[8];
                for (int k = 0; k < c.width; ++k) {
                    float value = original_gates[j][row*c.width+k] * (c.scaled ? 0.5f : 1.0f);
                    if (g.mask) value += ((float *)g.mask->data)[(row % 128)*c.width+k];
                    values[k] = value; mx = std::max(mx, double(value));
                }
                for (int k = 0; k < c.width; ++k) {
                    const double e = std::exp(values[k]-mx);
                    sum += e; weighted += e * ((float *)g.keys->data)[row*c.width+k];
                }
                const double expected = weighted/sum, actual = ((float *)g.out->data)[row];
                if (!std::isfinite(actual) || std::abs(actual-expected) > 0.00001 * std::max(1.0, std::abs(expected))) good = false;
            }
            if (dump) {
                std::fwrite(g.out->data, 1, ggml_nbytes(g.out), dump);
                if (c.output_probs) std::fwrite(g.probs->data, 1, ggml_nbytes(g.probs), dump);
                if (c.output_product) std::fwrite(g.product->data, 1, ggml_nbytes(g.product), dump);
                if (g.extra) std::fwrite(g.extra->data, 1, ggml_nbytes(g.extra), dump);
            }
        }
    }
    double ms = 0;
    if (timing && good) {
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < 160; ++i) if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) good = false;
        ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-start).count()/160;
    }
    std::printf("%s id=%d width=%d pools=%d streams=%d threads=%d fused=%d groups=%d ms=%.6f\n",
        good ? "PASS" : "FAIL", id, c.width, c.pools, c.streams, c.threads, fused, count, ms);
    std::fflush(stdout);
    ggml_backend_free(backend); ggml_free(ctx);
    return good;
}

int main(int argc, char ** argv) {
    const bool timing = argc == 2 && !std::strcmp(argv[1], "--timing");
    FILE * dump = timing ? nullptr : (argc == 2 ? std::fopen(argv[1], "wb") : nullptr);
    if (!timing && !dump) return 2;
    int failures = 0, id = 0;
    if (timing) {
        for (int pools : {66, 1024}) { test_case c; c.pools = pools; failures += !run_case(c, nullptr, true, id++); }
    } else {
        for (int threads : {1, 4, 15}) for (int pools : {1, 7, 66, 1024}) {
            test_case c; c.threads = threads; c.pools = pools;
            failures += !run_case(c, dump, false, id++);
            if (pools == 66) { c.streams = 3; c.swapped = true; failures += !run_case(c, dump, false, id++); }
        }
        for (int variant = 0; variant < 10; ++variant) {
            test_case c;
            switch (variant) {
                case 0: c.padded = true; break;
                case 1: c.width = 8; break;
                case 2: c.output_probs = true; break;
                case 3: c.output_product = true; break;
                case 4: c.extra_use = true; break;
                case 5: c.scaled = true; break;
                case 6: c.masked = true; break;
                case 7: c.reference = true; break;
                case 8: c.inplace = true; break;
                case 9: c.swapped = true; break;
            }
            failures += !run_case(c, dump, false, id++);
        }
        if (std::fclose(dump)) ++failures;
    }
    std::printf("SUMMARY cases=%d failures=%d\n", id, failures);
    return failures ? 1 : 0;
}
