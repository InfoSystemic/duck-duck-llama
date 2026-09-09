// Cold Q8 NR=1 diagnostic. No live model tensors are read or changed.
#pragma once
#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-cpu-impl.h"
#include "simd-mappings.h"
#include "repack.h"
#include "cold-q8-partials.h"
#include <cmath>

namespace cold_x16 {
static int k = 0;
using Gemv = void (*)(int, float *, size_t, const void *, const void *, int, int);

static void configure() {
    k = std::stoi(std::getenv("COLD_Q8_K"));
    if (k < 256 || k > 8192 || k % 256) std::abort();
}

static Gemv function(const std::string & mode) {
    if (mode == "full") return ggml_gemv_q8_0_x16_q8_0;
    if (mode == "copy") return candidate_q8<1>;
    if (mode == "p2") return candidate_q8<2>;
    if (mode == "p4") return candidate_q8<4>;
    std::abort();
}

class Workload {
    uint64_t * data;
    size_t tile_bytes, tiles;
    std::vector<block_q8_0> activations;
    std::vector<float> output;
    Gemv selected;

    void set_activations(uint64_t seed) {
        for (size_t i = 0; i < activations.size(); ++i) {
            auto & a = activations[i];
            a.d = 0x1400 + (seed + i * 11) % 1024;
            for (int j = 0; j < QK8_0; ++j) a.qs[j] = int((seed + i * 17 + j * 37) % 256) - 128;
        }
    }

    uint64_t run(Gemv fn) {
        // Keep each timed pass dependent on memory rather than a hoisted result.
        asm volatile("" : : "r"(data) : "memory");
        for (size_t tile = 0; tile < tiles; ++tile) {
            fn(k, output.data() + tile * 64, 0, (const char *) data + tile * tile_bytes,
               activations.data(), 1, 64);
        }
        uint64_t hash = 14695981039346656037ULL;
        for (float value : output) {
            uint32_t bits;
            std::memcpy(&bits, &value, sizeof(bits));
            hash = (hash ^ bits) * 1099511628211ULL;
        }
        return hash;
    }

public:
    Workload(uint64_t * allocation, size_t bytes, uint64_t seed, const std::string & mode) : data(allocation) {
        tile_bytes = 4 * (k / QK8_0) * sizeof(block_q8_0_x16);
        tiles = bytes / tile_bytes;
        if (!tiles) std::abort();
        selected = function(mode);
        uint64_t state = seed;
        for (size_t i = 0; i < bytes / sizeof(uint64_t); ++i) {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            data[i] = state;
        }
        auto * blocks = (block_q8_0_x16 *) data;
        for (size_t i = 0; i < tiles * tile_bytes / sizeof(*blocks); ++i) {
            for (int row = 0; row < 16; ++row) blocks[i].d[row] = 0x1800 + (seed + i + row * 7) % 1024;
        }
        activations.resize(k / QK8_0);
        output.resize(tiles * 64);
        // Every tile, all output bits, three changing inputs, unchanged library.
        for (int probe = 0; probe < 3; ++probe) {
            set_activations(seed + probe * 23);
            run(function("full"));
            for (float value : output) if (!std::isfinite(value)) std::abort();
            const auto reference = output;
            run(selected);
            if (std::memcmp(reference.data(), output.data(), output.size() * sizeof(float))) {
                std::fprintf(stderr, "Q8 partials differ: k=%d mode=%s seed=%llu probe=%d\n",
                             k, mode.c_str(), (unsigned long long) seed, probe);
                std::abort();
            }
        }
    }

    uint64_t scan() { return run(selected); }
    size_t bytes_per_pass() const { return tiles * tile_bytes; }
};
} // namespace cold_x16
