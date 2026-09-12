// SPDX-License-Identifier: MIT
// Bounded CPU candidate for the official 32-element BF16 -> MXFP8 path.
// Conversion uses the installed PyTorch header-only primitives. No model state,
// workers, allocations, or floating-point environment changes live in this ABI.
#include <torch/headeronly/util/BFloat16.h>
#include <torch/headeronly/util/Float8_e4m3fn.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <immintrin.h>

namespace {
inline float from_bits(uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}
inline uint32_t bits_of(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}
}

// 0: completed; 1: invalid ABI arguments; 2: nonfinite input; 3: nonstandard
// FP environment. Rejections never alter any input or output storage.
extern "C" int ds41_act_quant32_bf16(
        const uint16_t *input, int64_t blocks, uint8_t *quant, uint8_t *scales,
        uint16_t *restored, int inplace) {
    if (!input || blocks <= 0 || blocks > INT64_MAX / 32 ||
        (inplace != 0 && inplace != 1) ||
        (inplace ? !restored : (!quant || !scales))) return 1;
    // PyTorch can explicitly enable denormal flushing. Defer to the reference
    // for those environments instead of silently choosing different arithmetic.
    if (_mm_getcsr() & (0x8040U | 0x6000U)) return 3;
    for (int64_t i = 0; i < blocks * 32; ++i)
        if ((input[i] & 0x7fffU) >= 0x7f80U) return 2;

    for (int64_t block = 0; block < blocks; ++block) {
        const int64_t offset = block * 32;
        uint16_t maximum = 0;
        for (int i = 0; i < 32; ++i)
            maximum = std::max(maximum, uint16_t(input[offset + i] & 0x7fffU));
        const float amax = std::max(from_bits(uint32_t(maximum) << 16), 1e-4f);
        // Match cpu._round_scale exactly: an FP32 multiply followed by exponent
        // extraction and ceiling whenever any FP32 mantissa bit is present.
        const uint32_t ratio = bits_of(amax * (1.0f / 448.0f));
        const uint8_t code = uint8_t(((ratio >> 23) & 255) + ((ratio & 0x7fffffU) != 0));
        const float scale = from_bits(uint32_t(code) << 23);
        if (!inplace) scales[block] = code;
        for (int i = 0; i < 32; ++i) {
            const float value = from_bits(uint32_t(input[offset + i]) << 16);
            const float scaled = std::max(-448.0f, std::min(448.0f, value / scale));
            const uint8_t encoded = c10::detail::fp8e4m3fn_from_fp32_value(scaled);
            if (inplace) {
                const float decoded = c10::detail::fp8e4m3fn_to_fp32_value(encoded);
                restored[offset + i] = c10::BFloat16(decoded * scale).x;
            } else {
                quant[offset + i] = encoded;
            }
        }
    }
    return 0;
}
