// SPDX-License-Identifier: MIT
// Decode-only fused CPU sparse attention. Both matmuls use PyTorch's linked MKL
// BLAS, exp uses its MKL VML path, and probability sums retain ATen reduction.
#include <ATen/ops/from_blob.h>
#include <ATen/ops/sum.h>
#include <torch/headeronly/util/BFloat16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <vector>

extern "C" void sgemm_(const char *, const char *, const int *, const int *, const int *,
                       const float *, const float *, const int *, const float *, const int *,
                       const float *, float *, const int *);
extern "C" void vmsExp(int, const float *, float *, int64_t);

namespace {
inline float expand_bf16(uint16_t value) {
    const uint32_t bits = uint32_t(value) << 16;
    float output;
    std::memcpy(&output, &bits, 4);
    return output;
}
inline void matmul(int mode, int h, int c, int d, const float *a, const float *v, float *out) {
    const float one = 1.0f, zero = 0.0f;
    const char trans = 'T', normal = 'N';
    if (mode == 0) {
        // Row-major q[H,D] @ values[C,D].T, expressed in column-major BLAS.
        sgemm_(&trans, &normal, &c, &h, &d, &one, v, &d, a, &d, &zero, out, &c);
    } else {
        // Row-major rounded_prob[H,C] @ values[C,D].
        sgemm_(&normal, &normal, &d, &h, &c, &one, v, &d, a, &c, &zero, out, &d);
    }
}
struct Workspace {
    std::vector<float> q, values, scores, rounded, numerator, partial;
    std::vector<float> maximum, next_maximum, denominator, correction, sums, sink;
    void resize(int h, int d) {
        q.resize(h*d); values.resize(64*d); scores.resize(h*64); rounded.resize(h*64);
        numerator.assign(h*d, 0.0f); partial.resize(h*d);
        maximum.assign(h, -1e30f); next_maximum.resize(h); denominator.assign(h, 0.0f);
        correction.resize(h); sums.resize(h); sink.resize(h);
    }
};
thread_local Workspace workspace;

int sparse_impl(const uint16_t *q, const uint16_t *kv, const float *sink, const void *ids,
                int index_bytes, int h, int d, int rows, int count,
                int64_t q_head_stride, int64_t kv_row_stride, int64_t sink_stride,
                int64_t index_stride, float scale, uint16_t *out) {
    if (!q || !kv || !sink || (!ids && count) || !out || h < 1 || h > 128 || d < 1 || d > 1024 ||
        rows < 1 || count < 0 || count > 2048 || q_head_stride < d || kv_row_stride < d ||
        sink_stride < 1 || index_stride < 1 || (index_bytes != 4 && index_bytes != 8) ||
        !std::isfinite(scale) || scale <= 0) return 1;
    if (_mm_getcsr() & (0x8040U | 0x6000U)) return 3;
    auto index = [&](int i) -> int64_t {
        return index_bytes == 4 ? static_cast<const int32_t *>(ids)[i * index_stride]
                               : static_cast<const int64_t *>(ids)[i * index_stride];
    };
    for (int i = 0; i < count; ++i) if (index(i) < -1 || index(i) >= rows) return 2;
    auto &w = workspace;
    w.resize(h, d);
    for (int head = 0; head < h; ++head) {
        if (!std::isfinite(sink[head * sink_stride])) return 3;
        for (int j = 0; j < d; ++j) {
            const uint16_t bits = q[head * q_head_stride + j];
            if ((bits & 0x7fffU) >= 0x7f80U) return 3;
            w.q[head*d+j] = expand_bf16(bits);
        }
    }
    const auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
    auto sums = at::from_blob(w.sums.data(), {h}, options);
    auto full_probs = at::from_blob(w.scores.data(), {h, 64}, options);
    for (int begin = 0; begin < count; begin += 64) {
        const int c = std::min(64, count - begin);
        for (int token = 0; token < c; ++token) {
            const int64_t row = index(begin + token);
            for (int j = 0; j < d; ++j) {
                const uint16_t bits = row < 0 ? 0 : kv[row * kv_row_stride + j];
                if ((bits & 0x7fffU) >= 0x7f80U) return 3;
                w.values[token*d+j] = expand_bf16(bits);
            }
        }
        matmul(0, h, c, d, w.q.data(), w.values.data(), w.scores.data());
        for (int head = 0; head < h; ++head) {
            float maximum = w.maximum[head];
            for (int token = 0; token < c; ++token) {
                float &value = w.scores[head*c+token];
                if (index(begin + token) < 0) value = -INFINITY;
                else if (!std::isfinite(value)) return 3;
                value = value * scale;
                if (std::isnan(value) || value == INFINITY) return 3;
                maximum = std::max(maximum, value);
            }
            w.next_maximum[head] = maximum;
            w.correction[head] = w.maximum[head] - maximum;
            for (int token = 0; token < c; ++token) w.scores[head*c+token] -= maximum;
        }
        vmsExp(h, w.correction.data(), w.correction.data(), 0x140102);
        vmsExp(h*c, w.scores.data(), w.scores.data(), 0x140102);
        auto probs = c == 64 ? full_probs : at::from_blob(w.scores.data(), {h, c}, options);
        at::sum_out(sums, probs, at::IntArrayRef{-1}, false, at::kFloat);
        for (int i = 0; i < h*c; ++i)
            w.rounded[i] = expand_bf16(c10::BFloat16(w.scores[i]).x);
        matmul(1, h, c, d, w.rounded.data(), w.values.data(), w.partial.data());
        for (int head = 0; head < h; ++head) {
            w.denominator[head] = w.denominator[head] * w.correction[head] + w.sums[head];
            for (int j = 0; j < d; ++j) {
                const int offset = head*d+j;
                w.numerator[offset] = w.numerator[offset] * w.correction[head] + w.partial[offset];
                if (!std::isfinite(w.numerator[offset])) return 3;
            }
            w.maximum[head] = w.next_maximum[head];
        }
    }
    for (int head = 0; head < h; ++head) w.sink[head] = sink[head * sink_stride] - w.maximum[head];
    vmsExp(h, w.sink.data(), w.sink.data(), 0x140102);
    for (int head = 0; head < h; ++head) {
        const float denominator = w.denominator[head] + w.sink[head];
        for (int j = 0; j < d; ++j)
            out[head*d+j] = c10::BFloat16(w.numerator[head*d+j] / denominator).x;
    }
    return 0;
}
}

extern "C" int ds41_sparse_decode(const uint16_t *q, const uint16_t *kv,
        const float *sink, const void *ids, int index_bytes, int h, int d, int rows, int count,
        int64_t q_head_stride, int64_t kv_row_stride, int64_t sink_stride, int64_t index_stride,
        float scale, uint16_t *out) {
    try {
        return sparse_impl(q, kv, sink, ids, index_bytes, h, d, rows, count,
                           q_head_stride, kv_row_stride, sink_stride, index_stride, scale, out);
    } catch (...) {
        return 4;
    }
}

// Fixture-only direct access proves both BLAS reduction paths against PyTorch.
extern "C" int ds41_sparse_matmul_fixture(int mode, int h, int c, int d,
        const float *a, const float *values, float *out) {
    if ((mode != 0 && mode != 1) || h < 1 || c < 1 || d < 1 || !a || !values || !out) return 1;
    matmul(mode, h, c, d, a, values, out);
    return 0;
}
