#pragma once
// The k largest logits of a vocabulary row, in descending order, lowest index first among equals; NaN never enters.
// This is what the top-k sampler computes with std::partial_sort over 154,880 twelve-byte candidates (0.19 ms) after the
// candidate array has been built (0.45 ms). A threshold scan over the raw floats does it in ~20 us.

#include <cstdint>
#include <cstdlib>

static constexpr int F18_FAST_K_MAX = 128; // std::partial_sort is what top-k itself uses up to here
// a row that keeps beating the cut (sorted ascending, say) would cost O(n*k): give up and let the caller sort
static constexpr int F18_TOPK_MAX_INSERTS = 8192;

// the k largest values in descending order, lowest index first among equals; NaN never enters.
// Returns how many were found, or -1 when the row is pathological (see F18_TOPK_MAX_INSERTS).
static inline void f18_topk_insert(float v, int32_t id, int k, int & m, float * best, int32_t * best_id) {
    int j = m < k ? m++ : k - 1;
    while (j > 0 && best[j - 1] < v) {
        best[j]    = best[j - 1];
        best_id[j] = best_id[j - 1];
        --j;
    }
    best[j]    = v;
    best_id[j] = id;
}

static int f18_topk_scan_scalar(const float * x, int n, int k, float * best, int32_t * best_id) {
    int m = 0;
    int n_ins = 0;
    for (int i = 0; i < n; ++i) {
        const float v = x[i];
        if (m == k ? v > best[k - 1] : v == v) {
            f18_topk_insert(v, i, k, m, best, best_id);
            if (++n_ins > F18_TOPK_MAX_INSERTS) {
                return -1;
            }
        }
    }
    return m;
}

#if defined(__x86_64__)
#include <immintrin.h>
__attribute__((target("avx512f")))
static int f18_topk_scan_avx512(const float * x, int n, int k, float * best, int32_t * best_id) {
    int m = 0;
    int i = 0;
    for (; i < n && m < k; ++i) { // fill
        if (x[i] == x[i]) {
            f18_topk_insert(x[i], i, k, m, best, best_id);
        }
    }
    if (m < k) {
        return m;
    }
    int n_ins = 0;
    __m512 thr = _mm512_set1_ps(best[k - 1]);
    for (; i + 16 <= n; i += 16) {
        __mmask16 mask = _mm512_cmp_ps_mask(_mm512_loadu_ps(x + i), thr, _CMP_GT_OQ);
        while (mask) {
            const int b = __builtin_ctz(mask);
            mask &= (__mmask16) (mask - 1);
            const float v = x[i + b];
            if (v > best[k - 1]) { // the cut may have moved inside this block
                f18_topk_insert(v, i + b, k, m, best, best_id);
                thr = _mm512_set1_ps(best[k - 1]);
                if (++n_ins > F18_TOPK_MAX_INSERTS) {
                    return -1;
                }
            }
        }
    }
    for (; i < n; ++i) {
        if (x[i] > best[k - 1]) {
            f18_topk_insert(x[i], i, k, m, best, best_id);
        }
    }
    return m;
}
#endif

static int f18_topk_scan(const float * x, int n, int k, float * best, int32_t * best_id) {
#if defined(__x86_64__)
    static const bool has_avx512 = __builtin_cpu_supports("avx512f") && !(getenv("GGML_F18_TOPK_SCALAR") && atoi(getenv("GGML_F18_TOPK_SCALAR")) != 0);
    if (has_avx512) {
        return f18_topk_scan_avx512(x, n, k, best, best_id);
    }
#endif
    return f18_topk_scan_scalar(x, n, k, best, best_id);
}
