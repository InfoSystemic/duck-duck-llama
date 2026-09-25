#pragma once
// Top-k by selection instead of a partial sort. The result is a set: the consumers scatter it and ignore the order.
// When the k-th value is tied with a value outside the set, the set is not unique: return false and let the caller sort.

#include <functional>

// std::partial_sort rejects most elements with one comparison, so it wins again once the row is long:
// measured break-even is ~25K entries for k = 512 (tools/topk_select_check.cpp). Select with nth_element below that; above it,
// since 2026-09-20, find the k-th largest value with an MSB-first radix select over order-preserving key bits (one pass over the
// row plus passes over the shrinking candidate set: ~1.1 n operations, no comparisons), which is what the indexer needs at depth
// (34,178 pools at 128K context, 137K at 512K). The threshold is the same value either way, so the selected set is the same.
static constexpr int64_t F18_TOPK_MAX_N = 16384;

static inline uint32_t f18_float_key(float v) { // monotone: larger float -> larger key (no NaNs by construction)
    uint32_t u; memcpy(&u, &v, sizeof(u));
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
static inline float f18_key_float(uint32_t key) {
    const uint32_t u = (key & 0x80000000u) ? (key & 0x7fffffffu) : ~key;
    float v; memcpy(&v, &u, sizeof(v)); return v;
}
static float f18_kth_largest_radix(const float * x, int64_t n, int64_t k, uint32_t * key) {
    for (int64_t i = 0; i < n; ++i) key[i] = f18_float_key(x[i]);
    int64_t cnt = n, want = k; uint32_t prefix = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
        int64_t hist[256] = {0};
        for (int64_t i = 0; i < cnt; ++i) ++hist[(key[i] >> shift) & 255u];
        int64_t seen = 0; int b = 255;
        for (; b > 0; --b) { if (seen + hist[b] >= want) break; seen += hist[b]; }
        want -= seen; prefix |= (uint32_t) b << shift;
        if (shift == 0) break;
        int64_t w = 0;
        for (int64_t i = 0; i < cnt; ++i) if (((key[i] >> shift) & 255u) == (uint32_t) b) key[w++] = key[i];
        cnt = w;
    }
    return f18_key_float(prefix);
}

static bool f18_top_k_row(const float * x, int64_t n, int64_t k, int32_t * out, float * tmp) {
    if (k >= n) {
        for (int64_t i = 0; i < n; ++i) {
            out[i] = (int32_t) i;
        }
        return k == n;
    }

    for (int64_t i = 0; i < n; ++i) {
        if (x[i] != x[i]) {
            return false; // NaN: keep the reference order
        }
    }

    float thr;
    if (n > F18_TOPK_MAX_N) {
        static_assert(sizeof(uint32_t) == sizeof(float), "tmp doubles as the key buffer");
        thr = f18_kth_largest_radix(x, n, k, (uint32_t *) tmp);
    } else {
        memcpy(tmp, x, (size_t) n*sizeof(float));
        std::nth_element(tmp, tmp + (k - 1), tmp + n, std::greater<float>());
        thr = tmp[k - 1];
    }

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
