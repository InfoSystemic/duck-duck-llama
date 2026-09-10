// SPDX-License-Identifier: MIT
// Independent caller-environment check for the isolated native Engram decoder.
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <immintrin.h>
#include <vector>

extern "C" int deepseek_v41_engram_gather_bf16(
    const uint8_t *, const uint8_t *, int64_t, const int64_t *, int64_t, uint16_t *, int);

int main() {
    constexpr int rows = 256, cols = 256;
    std::vector<uint8_t> weight(rows * cols + 1), scale(rows * 8 + 3);
    std::vector<uint16_t> reference(rows * cols + 2, 0xbeef), result(rows * cols + 2, 0xbeef);
    std::array<int64_t, rows> ids{};
    for (int row = 0; row < rows; ++row) {
        ids[row] = row;
        for (int col = 0; col < cols; ++col) weight[1 + row * cols + col] = uint8_t(col);
        for (int block = 0; block < 8; ++block) scale[3 + row * 8 + block] = uint8_t(row);
    }
    const unsigned initial = _mm_getcsr();
    _mm_setcsr(0x1f80);
    if (deepseek_v41_engram_gather_bf16(weight.data() + 1, scale.data() + 3, rows,
            ids.data(), rows, reference.data() + 1, 0)) return 2;
    int checks = 0;
    for (int mode : {0, 1}) for (unsigned rounding : {0U, 0x2000U, 0x4000U, 0x6000U})
    for (unsigned flush : {0U, 0x8040U}) {
        const unsigned requested = 0x1f80U | rounding | flush;
        _mm_setcsr(requested);
        if (deepseek_v41_engram_gather_bf16(weight.data() + 1, scale.data() + 3, rows,
                ids.data(), rows, result.data() + 1, mode)) return 3;
        if (_mm_getcsr() != requested || result != reference) return 4;
        if (result.front() != 0xbeef || result.back() != 0xbeef) return 5;
        ++checks;
    }
    _mm_setcsr(initial);
    std::printf("{\"passed\":true,\"environment_cases\":%d,\"values_per_case\":65536,"
                "\"unaligned_buffers\":true,\"canaries_preserved\":true,\"caller_mxcsr_restored\":true}\n", checks);
    return 0;
}
