#pragma once
// Top-k by selection instead of a partial sort. The result is a set: the consumers scatter it and ignore the order.
// When the k-th value is tied with a value outside the set, the set is not unique: return false and let the caller sort.

#include <functional>

// std::partial_sort rejects most elements with one comparison, so it wins again once the row is long:
// measured break-even is ~25K entries for k = 512 (tools/topk_select_check.cpp). Select only below that.
static constexpr int64_t F18_TOPK_MAX_N = 16384;

static bool f18_top_k_row(const float * x, int64_t n, int64_t k, int32_t * out, float * tmp) {
    if (k >= n) {
        for (int64_t i = 0; i < n; ++i) {
            out[i] = (int32_t) i;
        }
        return k == n;
    }
    if (n > F18_TOPK_MAX_N) {
        return false;
    }

    for (int64_t i = 0; i < n; ++i) {
        if (x[i] != x[i]) {
            return false; // NaN: keep the reference order
        }
    }

    memcpy(tmp, x, (size_t) n*sizeof(float));
    std::nth_element(tmp, tmp + (k - 1), tmp + n, std::greater<float>());
    const float thr = tmp[k - 1];

    int64_t n_ge = 0;
    for (int64_t i = 0; i < n; ++i) {
        n_ge += x[i] >= thr;
    }
    if (n_ge != k) {
        return false; // tie across the cut
    }

    int64_t j = 0;
    for (int64_t i = 0; i < n; ++i) {
        if (x[i] >= thr) {
            out[j++] = (int32_t) i;
        }
    }
    return true;
}
