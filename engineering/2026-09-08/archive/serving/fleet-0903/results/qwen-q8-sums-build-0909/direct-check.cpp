#include <immintrin.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <vector>
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "ggml-cpu-impl.h"
#include "simd-mappings.h"
#include "repack.h"
#include "q8-sums-candidates.h"

using Gemv = void (*)(int, float *, size_t, const void *, const void *, int, int);
static const Gemv functions[] = {ggml_gemv_q8_0_x16_q8_0, candidate_q8<0>, candidate_q8<1>, candidate_q8<2>};
static const char * names[] = {"library", "copy", "sad", "inline16"};
static uint64_t state = 0x4ba625f81dd731a9ULL;
static uint64_t random_word() {
    state ^= state << 13; state ^= state >> 7; state ^= state << 17;
    return state;
}
static void require(bool ok, const char * message) {
    if (!ok) { std::fprintf(stderr, "%s\n", message); std::abort(); }
}
static uint64_t hash(const std::vector<float> & values) {
    uint64_t result = 14695981039346656037ULL;
    for (float value : values) {
        uint32_t bits; std::memcpy(&bits, &value, sizeof(bits));
        result = (result ^ bits) * 1099511628211ULL;
    }
    return result;
}
static size_t sum_checks() {
    alignas(32) int8_t q[32];
    size_t count = 0;
    auto check = [&] {
        int sum = 0; for (int j = 0; j < 32; ++j) sum += q[j];
        require(sum == q8_activation_sum(q), "Integer sum differs");
        ++count;
    };
    for (int value = -128; value <= 127; ++value) {
        std::fill(q, q + 32, value); check();
    }
    for (int position = 0; position < 32; ++position) for (int value = -128; value <= 127; ++value) {
        for (int j = 0; j < 32; ++j) q[j] = int((j * 37 + position * 19) % 256) - 128;
        q[position] = value; check();
    }
    for (int probe = 0; probe < 10000; ++probe) {
        for (auto & value : q) value = int(random_word() % 256) - 128;
        check();
    }
    return count;
}
struct Data {
    int k, nc, nb;
    size_t tiles, blocks_per_tile, tile_bytes;
    std::vector<block_q8_0_x16> weights;
    std::vector<block_q8_0> activation;
    std::vector<float> output;
    Data(int k_, int nc_, bool rotating) : k(k_), nc(nc_), nb(k / 32) {
        blocks_per_tile = size_t(nc / 16) * nb;
        tile_bytes = blocks_per_tile * sizeof(block_q8_0_x16);
        tiles = rotating ? std::max<size_t>(1, (32ULL << 20) / tile_bytes) : 1;
        weights.resize(tiles * blocks_per_tile);
        activation.resize(nb); output.resize(tiles * nc);
        const uint16_t scales[] = {0, 0x8000, 1, 0x03ff, 0x0400, 0x1400, 0x3c00, 0xbc00, 0x7bff, 0x3555};
        for (auto & block : weights) {
            for (auto & value : block.qs) value = random_word() & 255;
            for (auto & value : block.d) value = scales[random_word() % 10];
        }
        set_activation(5);
    }
    void set_activation(int probe) {
        const uint16_t scales[] = {0, 0x8000, 1, 0x03ff, 0x0400, 0x1400, 0x3c00, 0xbc00, 0x7bff, 0x3555};
        for (int b = 0; b < nb; ++b) {
            auto & a = activation[b];
            a.d = scales[(b + probe * 3) % 10];
            for (int j = 0; j < 32; ++j) {
                if (probe == 0) a.qs[j] = -128;
                else if (probe == 1) a.qs[j] = 127;
                else if (probe == 2) a.qs[j] = 0;
                else if (probe == 3) a.qs[j] = j % 2 ? 127 : -128;
                else a.qs[j] = int(random_word() % 256) - 128;
            }
        }
    }
    void run(Gemv fn) {
        asm volatile("" : : "r"(weights.data()), "r"(activation.data()) : "memory");
        for (size_t tile = 0; tile < tiles; ++tile) {
            fn(k, output.data() + tile * nc, 0, weights.data() + tile * blocks_per_tile, activation.data(), 1, nc);
        }
    }
    std::vector<float> canonical() const {
        std::vector<float> result(tiles * nc);
        for (size_t tile = 0; tile < tiles; ++tile) for (int row = 0; row < nc; ++row) {
            float total = 0;
            for (int b = 0; b < nb; ++b) {
                const auto & w = weights[tile * blocks_per_tile + (row / 16) * nb + b];
                int dot = 0;
                for (int j = 0; j < 32; ++j) {
                    const int offset = (j / 4) * 64 + (row % 16) * 4 + j % 4;
                    dot += (int(w.qs[offset]) - 128) * int(activation[b].qs[j]);
                }
                const float scale = GGML_CPU_FP16_TO_FP32(w.d[row % 16]) * GGML_CPU_FP16_TO_FP32(activation[b].d);
                total = std::fma(float(dot), scale, total);
            }
            result[tile * nc + row] = total;
        }
        return result;
    }
};
static void exact(const std::vector<float> & a, const std::vector<float> & b) {
    require(a.size() == b.size(), "Output size differs");
    require(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0, "Output bits differ");
}
int main(int argc, char ** argv) {
    require(argc == 2, "Expected output path");
    ggml_cpu_init();
    require(GGML_CPU_FP16_TO_FP32(0x3c00) == 1.0f, "Conversion table is not initialized");
    Dl_info runtime{};
    require(dladdr(reinterpret_cast<void *>(ggml_gemv_q8_0_x16_q8_0), &runtime), "Missing CPU binding");
    std::printf("{\"event\":\"library\",\"path\":\"%s\"}\n", runtime.dli_fname);
    const size_t sums = sum_checks();
    FILE * dump = std::fopen(argv[1], "wb"); require(dump, "Cannot create output");
    size_t cases = 0, values = 0;
    for (int k : {32, 128, 512, 1536, 2048, 3072, 4096, 16384, 16416, 32768})
    for (int nc : {16, 32, 64, 128}) {
        Data data(k, nc, false);
        for (int probe = 0; probe < 6; ++probe) {
            data.set_activation(probe);
            const auto expected = data.canonical();
            for (float value : expected) require(std::isfinite(value), "Canonical result is not finite");
            for (auto fn : functions) {
                data.run(fn); exact(expected, data.output);
                require(std::fwrite(data.output.data(), sizeof(float), data.output.size(), dump) == data.output.size(), "Output write failed");
                values += data.output.size();
            }
            ++cases;
        }
    }
    require(std::fclose(dump) == 0, "Output close failed");
    std::printf("{\"event\":\"correctness\",\"sum_cases\":%zu,\"matrix_cases\":%zu,\"compared_values\":%zu,\"passed\":true}\n", sums, cases, values);
    std::fflush(stdout);
    if (std::getenv("Q8_SUMS_CHECK_ONLY")) {
        std::printf("{\"event\":\"done\",\"passed\":true}\n");
        return 0;
    }
    for (int k : {128, 512, 2048, 4096, 16384})
    for (int nc : {16, 64})
    for (bool rotating : {false, true}) {
        Data data(k, nc, rotating);
        data.run(functions[0]); const auto expected = data.output;
        for (int mode = 1; mode < 4; ++mode) { data.run(functions[mode]); exact(expected, data.output); }
        const int passes = rotating ? 1 : std::max<size_t>(1, (4ULL << 20) / data.tile_bytes);
        std::vector<double> samples[4];
        for (int sample = 0; sample < 17; ++sample) for (int mode : {0, 1, 2, 3, 3, 2, 1, 0}) {
            const auto before = std::chrono::steady_clock::now();
            for (int pass = 0; pass < passes; ++pass) data.run(functions[mode]);
            const auto after = std::chrono::steady_clock::now();
            samples[mode].push_back(std::chrono::duration<double, std::micro>(after - before).count() / (passes * data.tiles));
        }
        exact(expected, data.output);
        for (int mode = 0; mode < 4; ++mode) {
            auto & v = samples[mode]; std::sort(v.begin(), v.end());
            std::printf("{\"event\":\"timing\",\"k\":%d,\"nc\":%d,\"rotating\":%s,\"weight_bytes\":%zu,\"mode\":\"%s\",\"median_us\":%.9f,\"samples\":%zu,\"hash\":\"%016llx\"}\n",
                        k, nc, rotating ? "true" : "false", data.weights.size() * sizeof(block_q8_0_x16), names[mode],
                        (v[v.size()/2 - 1] + v[v.size()/2]) / 2, v.size(), (unsigned long long)hash(expected));
        }
        std::fflush(stdout);
    }
    std::printf("{\"event\":\"done\",\"passed\":true}\n");
}
