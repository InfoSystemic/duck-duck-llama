// A continuous stream of 64-row tiles through Full's existing x16 kernels.
// Kernels and weights are private to the fixture. No server is loaded or changed.
#pragma once
#include "ggml.h"
#include "ggml-impl.h"
#include "repack.h"
#include "cold-x16-candidates.h"

#include <cmath>

namespace cold_x16 {
static int kind = 0, k = 0, tokens = 0;
using Gemv = void (*)(int, float *, size_t, const void *, const void *, int, int);

static void configure() {
    kind = std::stoi(std::getenv("COLD_X16_KIND"));
    k = std::stoi(std::getenv("COLD_X16_K"));
    tokens = std::stoi(std::getenv("COLD_X16_TOKENS"));
    if ((kind != 4 && kind != 5) || k < 256 || k > 8192 || k % 256 || tokens < 1 || tokens > 3) std::abort();
}

static Gemv function(const std::string & mode) {
    if (kind == 4) {
        if (mode == "full") return ggml_gemv_q4_K_x16_q8_K;
        if (mode == "copy") return candidate_q4<0>;
        if (mode == "pf1") return candidate_q4<1>;
        if (mode == "pf2") return candidate_q4<2>;
        if (mode == "pf4") return candidate_q4<4>;
    } else {
        if (mode == "full") return ggml_gemv_q5_K_x16_q8_K;
        if (mode == "copy") return candidate_q5<0>;
        if (mode == "pf1") return candidate_q5<1>;
        if (mode == "pf2") return candidate_q5<2>;
        if (mode == "pf4") return candidate_q5<4>;
    }
    std::abort();
}

template<typename Block>
static void metadata(Block * blocks, size_t count) {
    const bool small_scales = std::getenv("COLD_X16_SMALL_SCALES") &&
        std::atoi(std::getenv("COLD_X16_SMALL_SCALES")) != 0;
    for (size_t i = 0; i < count; ++i) {
        auto & b = blocks[i];
        for (int r = 0; r < 16; ++r) {
            b.d[r] = small_scales ? 0x0080 + ((i + r) % 512) : 0x1800 + ((i + r) % 1024);
            b.dmin[r] = 0x1400 + ((i + r * 7) % 1024);
            for (int s = 0; s < 8; ++s) {
                b.scales[s][r] &= 63;
                b.mins[s][r] &= 63;
            }
        }
    }
}

class Workload {
    uint64_t * data;
    size_t block_bytes, tile_bytes, tiles;
    std::vector<block_q8_K> activations;
    std::vector<float> output;
    Gemv selected;

    void set_activations(uint64_t seed) {
        for (size_t i = 0; i < activations.size(); ++i) {
            auto & a = activations[i];
            a.d = 0.002f * (1 + (seed + i) % 7);
            for (int j = 0; j < QK_K; ++j) a.qs[j] = int((seed + i * 17 + j * 37) % 255) - 127;
            for (int j = 0; j < QK_K / 16; ++j) {
                int sum = 0;
                for (int z = 0; z < 16; ++z) sum += a.qs[j * 16 + z];
                a.bsums[j] = sum;
            }
        }
    }

    uint64_t run(Gemv fn) {
        for (size_t tile = 0; tile < tiles; ++tile) {
            const auto * weights = (const char *) data + tile * tile_bytes;
            for (int t = 0; t < tokens; ++t) {
                fn(k, output.data() + (tile * tokens + t) * 64, 0, weights,
                   activations.data() + t * (k / QK_K), 1, 64);
            }
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
        block_bytes = kind == 4 ? sizeof(block_q4_K_x16) : sizeof(block_q5_K_x16);
        tile_bytes = 4 * (k / QK_K) * block_bytes;
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
        if (kind == 4) metadata((block_q4_K_x16 *) data, tiles * tile_bytes / block_bytes);
        else metadata((block_q5_K_x16 *) data, tiles * tile_bytes / block_bytes);
        activations.resize(tokens * (k / QK_K));
        output.resize(tiles * tokens * 64);
        // Verify all streamed tiles on changing inputs against the unchanged library.
        for (int probe = 0; probe < 3; ++probe) {
            set_activations(seed + probe * 23);
            run(function("full"));
            for (float value : output) if (!std::isfinite(value)) std::abort();
            const auto reference = output;
            run(selected);
            if (std::memcmp(reference.data(), output.data(), output.size() * sizeof(float))) {
                std::fprintf(stderr, "Cold x16 output differs: kind=%d k=%d tokens=%d mode=%s seed=%llu probe=%d\n",
                             kind, k, tokens, mode.c_str(), (unsigned long long) seed, probe);
                std::abort();
            }
        }
    }

    uint64_t scan() { return run(selected); }
    size_t bytes_per_pass() const { return tiles * tile_bytes; }
};
} // namespace cold_x16
