// SPDX-License-Identifier: MIT
// Fused scalar CPU candidate for one hc_mult=4 token. Explicit float32 staging
// follows deepseek_v41_cpu_reference_0910.py; no reassociation or fused multiply-add.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <immintrin.h>

// Installed ATen/cpu/vml.h dispatches FP32 torch.exp to MKL VML. Reuse that
// exported libtorch_cpu primitive; libc expf differs by one ULP on some inputs.
// The separate 4-value sigmoid paths use scalar libc expf in PyTorch.
extern "C" void vmsExp(int, const float *, float *, int64_t);

namespace {
inline float sum4(float a, float b, float c, float d) {
    return ((a + b) + c) + d;
}
}

// 0 success, 1 invalid ABI parameters, 2 nonfinite input/intermediate, 3 unsupported
// FP environment. The rejection paths leave all output storage untouched.
extern "C" int ds41_hc4_single(const float *mixes, const float *scale,
        const float *base, int iterations, float eps,
        float *pre, float *post, float *comb) {
    if (!mixes || !scale || !base || !pre || !post || !comb ||
        iterations < 1 || iterations > 256 || !std::isfinite(eps) || eps < 0) return 1;
    if (_mm_getcsr() & (0x8040U | 0x6000U)) return 3;
    for (int i = 0; i < 3; ++i) if (!std::isfinite(scale[i])) return 2;
    float transformed[24];
    for (int i = 0; i < 24; ++i) {
        if (!std::isfinite(mixes[i]) || !std::isfinite(base[i])) return 2;
        const float product = mixes[i] * scale[i < 4 ? 0 : i < 8 ? 1 : 2];
        transformed[i] = product + base[i];
        if (!std::isfinite(transformed[i])) return 2;
    }
    for (int i = 0; i < 4; ++i) {
        pre[i] = (1.0f / (1.0f + std::exp(-transformed[i]))) + eps;
        post[i] = 2.0f * (1.0f / (1.0f + std::exp(-transformed[4 + i])));
    }
    float centered[16];
    for (int row = 0; row < 4; ++row) {
        const float *x = transformed + 8 + row * 4;
        const float maximum = std::max(std::max(x[0], x[1]), std::max(x[2], x[3]));
        for (int col = 0; col < 4; ++col)
            centered[row * 4 + col] = x[col] - maximum;
    }
    // VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_IGNORE, matching ATen/cpu/vml.h.
    vmsExp(16, centered, comb, 0x140102);
    for (int row = 0; row < 4; ++row) {
        const float *v = comb + row * 4;
        const float sum = sum4(v[0], v[1], v[2], v[3]);
        for (int col = 0; col < 4; ++col)
            comb[row * 4 + col] = comb[row * 4 + col] / sum + eps;
    }
    for (int col = 0; col < 4; ++col) {
        const float sum = sum4(comb[col], comb[4 + col], comb[8 + col], comb[12 + col]) + eps;
        for (int row = 0; row < 4; ++row) comb[row * 4 + col] /= sum;
    }
    for (int iteration = 1; iteration < iterations; ++iteration) {
        for (int row = 0; row < 4; ++row) {
            const float *v = comb + row * 4;
            const float sum = sum4(v[0], v[1], v[2], v[3]) + eps;
            for (int col = 0; col < 4; ++col) comb[row * 4 + col] /= sum;
        }
        for (int col = 0; col < 4; ++col) {
            const float sum = sum4(comb[col], comb[4 + col], comb[8 + col], comb[12 + col]) + eps;
            for (int row = 0; row < 4; ++row) comb[row * 4 + col] /= sum;
        }
    }
    return 0;
}
