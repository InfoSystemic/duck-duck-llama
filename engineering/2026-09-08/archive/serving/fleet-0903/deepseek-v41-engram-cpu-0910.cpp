// SPDX-License-Identifier: MIT
// Native E4M3/E8M0 Engram row decode for DeepSeek-V4.1-Flash CPU bring-up.
// Reference semantics: DeepSeek inference/model.py ParallelEngramEmbedding.
// Standalone component; not yet wired into a llama.cpp model graph.
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <limits>
#include <vector>
#include <algorithm>

namespace {
struct Tables {
    alignas(64) std::array<float, 256> fp8;
    alignas(64) std::array<float, 256> scale;
    Tables() {
        for (unsigned code = 0; code < 256; ++code) {
            const unsigned magnitude = code & 127, exp = magnitude >> 3, mant = magnitude & 7;
            float value = magnitude == 127 ? std::numeric_limits<float>::quiet_NaN() :
                exp == 0 ? std::ldexp(float(mant), -9) : std::ldexp(float(8 + mant), int(exp) - 10);
            fp8[code] = code & 128 ? -value : value;
            scale[code] = code == 255 ? std::numeric_limits<float>::quiet_NaN() : std::ldexp(1.0f, int(code) - 127);
        }
    }
};

const Tables & tables() { static const Tables t; return t; }

uint16_t to_bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7fffffffU) > 0x7f800000U) return 0x7fc0;
    return uint16_t((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
}

void scalar_row(const uint8_t * weight, const uint8_t * scale, uint16_t * out) {
    const auto & t = tables();
    for (int k = 0; k < 256; ++k) out[k] = to_bf16(t.fp8[weight[k]] * t.scale[scale[k / 32]]);
}

__attribute__((target("avx512f,avx512bw,avx512vl")))
void vector_row(const uint8_t * weight, const uint8_t * scale, uint16_t * out) {
    const auto & t = tables();
    for (int block = 0; block < 8; ++block) {
        const __m512 factor = _mm512_set1_ps(t.scale[scale[block]]);
        for (int half = 0; half < 2; ++half) {
            const int offset = block * 32 + half * 16;
            const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i *>(weight + offset));
            const __m512i indices = _mm512_cvtepu8_epi32(bytes);
            const __m512 decoded = _mm512_i32gather_ps(indices, t.fp8.data(), 4);
            const __m512i bits = _mm512_castps_si512(_mm512_mul_ps(decoded, factor));
            const __mmask16 nan = _mm512_cmp_epu32_mask(
                _mm512_and_si512(bits, _mm512_set1_epi32(0x7fffffff)),
                _mm512_set1_epi32(0x7f800000), _MM_CMPINT_GT);
            const __m512i lsb = _mm512_and_si512(_mm512_srli_epi32(bits, 16), _mm512_set1_epi32(1));
            const __m512i rounded = _mm512_srli_epi32(
                _mm512_add_epi32(bits, _mm512_add_epi32(lsb, _mm512_set1_epi32(0x7fff))), 16);
            const __m512i canonical = _mm512_mask_mov_epi32(rounded, nan, _mm512_set1_epi32(0x7fc0));
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(out + offset), _mm512_cvtepi32_epi16(canonical));
        }
    }
}
}  // namespace

extern "C" int deepseek_v41_engram_has_avx512() {
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
           __builtin_cpu_supports("avx512vl");
}

extern "C" int deepseek_v41_engram_gather_bf16(
    const uint8_t * weight, const uint8_t * scale, int64_t table_rows,
    const int64_t * ids, int64_t count, uint16_t * out, int vectorize) {
    if (count < 0 || table_rows < 0 || table_rows > INT64_MAX / 256 ||
        (vectorize != 0 && vectorize != 1)) return -1;
    if (count == 0) return 0;
    if (!weight || !scale || !ids || !out || count > INT64_MAX / 256) return -1;
    if (vectorize && !deepseek_v41_engram_has_avx512()) return -2;
    // Validate the entire ID list before writing anything; callers can reject bad state cleanly.
    for (int64_t i = 0; i < count; ++i) if (ids[i] < 0 || ids[i] >= table_rows) return -1;
    // The reference uses IEEE round-to-nearest-even with gradual underflow. Restore caller state.
    const unsigned old_csr = _mm_getcsr();
    _mm_setcsr((old_csr & ~(0x8040U | 0x6000U)));
    const auto row = vectorize ? vector_row : scalar_row;
    for (int64_t i = 0; i < count; ++i)
        row(weight + ids[i] * 256, scale + ids[i] * 8, out + i * 256);
    _mm_setcsr(old_csr);
    return 0;
}

#ifdef DEEPSEEK_V41_ENGRAM_BENCH
int main(int argc, char ** argv) {
    if (argc != 3) return 2;
    const int mode = std::atoi(argv[1]);
    const int64_t rows = std::atoll(argv[2]);
    if (rows < 48 || rows > 4 * 1024 * 1024 || mode < 0 || mode > 1) return 2;
    std::vector<uint8_t> weight(rows * 256), scale(rows * 8);
    uint64_t rng = 0xd5eec410;
    auto next = [&]() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; };
    for (auto & value : weight) { value = uint8_t(next()); if ((value & 127) == 127) value ^= 1; }
    for (auto & value : scale) value = uint8_t(115 + next() % 13);
    std::array<int64_t, 48> ids{};
    std::array<uint16_t, 48 * 256> output{};
    std::vector<double> timings;
    uint64_t checksum = 0;
    constexpr int batches = 51, iterations = 256;
    // Small case is cache resident. Large case rotates rows over a table beyond one socket LLC.
    for (int batch = -2; batch < batches; ++batch) {
        std::vector<std::array<int64_t, 48>> requests(iterations);
        for (auto & request : requests) for (auto & id : request) id = int64_t(next() % uint64_t(rows));
        const auto started = std::chrono::steady_clock::now();
        for (int i = 0; i < iterations; ++i) {
            ids = requests[i];
            if (deepseek_v41_engram_gather_bf16(weight.data(), scale.data(), rows,
                    ids.data(), ids.size(), output.data(), mode)) return 3;
            checksum += output[(i * 41) % output.size()];
        }
        const auto finished = std::chrono::steady_clock::now();
        if (batch >= 0) timings.push_back(std::chrono::duration<double, std::micro>(finished - started).count() / iterations);
    }
    std::sort(timings.begin(), timings.end());
    std::printf("{\"mode\":%d,\"table_rows\":%lld,\"table_bytes\":%lld,\"lookups_per_call\":48,"
                "\"median_us\":%.6f,\"p10_us\":%.6f,\"p90_us\":%.6f,\"checksum\":\"%016llx\","
                "\"scope\":\"single-worker component; synthetic rows, no model or IMC claim\"}\n",
                mode, (long long)rows, (long long)(rows * 264), timings[25], timings[5], timings[45],
                (unsigned long long)checksum);
    return 0;
}
#endif
