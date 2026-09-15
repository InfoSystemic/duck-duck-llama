// Microbenchmark for the sparse-attention top-k selection, which is the prime suspect for the long-context decode
// collapse (29.22 -> 14.21 tok/s going from 234 to 38,056 context, a 36.2 ms/token penalty that neither KV bytes nor
// attention FLOPs can explain).
//
// The engine does, once per full-attention layer (12 of 48) per forward pass:
//     std::partial_sort(tmp, tmp + 2048, tmp + n_ctx, cmp{scores});   // indirect compare, O(n log k)
// on a tensor whose ggml_nrows() is 1, so it runs on ONE thread of 60.
//
// Three candidates measured here. The kernel's own comment ("emphasize that the order is not important" — it swaps
// dst[0] and dst[1]) says the output order is NOT required, which is what makes the cheaper selections legal.
//
// build: g++ -O2 -march=native -fopenmp -o bench_topk bench_topk.cpp
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <numeric>
#include <random>
#include <vector>
#include <chrono>
#ifdef _OPENMP
#include <omp.h>
#endif

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t) {
    return std::chrono::duration<double, std::milli>(clk::now() - t).count();
}

struct cmp_top_k { const float * d; bool operator()(int32_t a, int32_t b) const { return d[a] > d[b]; } };

// 1. what the engine does today
static void topk_partial_sort(const float * s, int n, int k, std::vector<int32_t> & idx, std::vector<int32_t> & out) {
    std::iota(idx.begin(), idx.begin() + n, 0);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.begin() + n, cmp_top_k{s});
    std::copy(idx.begin(), idx.begin() + k, out.begin());
}

// 2. same thing, O(n) average instead of O(n log k). Legal because the order is not required.
static void topk_nth_element(const float * s, int n, int k, std::vector<int32_t> & idx, std::vector<int32_t> & out) {
    std::iota(idx.begin(), idx.begin() + n, 0);
    std::nth_element(idx.begin(), idx.begin() + k - 1, idx.begin() + n, cmp_top_k{s});
    std::copy(idx.begin(), idx.begin() + k, out.begin());
}

// 3. parallel threshold selection: histogram the scores, find the bucket holding the k-th largest, collect above it.
//    O(n) with full thread parallelism and sequential (SIMD-friendly) access instead of indirect compares.
static void topk_threshold(const float * s, int n, int k, std::vector<int32_t> & out, int nth) {
    constexpr int NB = 1024;
    float lo = s[0], hi = s[0];
#pragma omp parallel for reduction(min:lo) reduction(max:hi) num_threads(nth) schedule(static)
    for (int i = 0; i < n; ++i) { lo = std::min(lo, s[i]); hi = std::max(hi, s[i]); }
    const float span = (hi > lo) ? (hi - lo) : 1.0f;
    const float scale = NB / span;
    std::vector<int64_t> hist(NB, 0);
#pragma omp parallel num_threads(nth)
    {
        std::vector<int64_t> loc(NB, 0);
#pragma omp for schedule(static) nowait
        for (int i = 0; i < n; ++i) {
            int b = (int)((s[i] - lo) * scale); b = b < 0 ? 0 : (b >= NB ? NB - 1 : b);
            loc[b]++;
        }
#pragma omp critical
        for (int b = 0; b < NB; ++b) hist[b] += loc[b];
    }
    // walk from the top bucket down until we have at least k
    int64_t acc = 0; int cut = NB - 1;
    for (; cut >= 0; --cut) { acc += hist[cut]; if (acc >= k) break; }
    const float thresh = lo + (float)cut / scale;      // everything strictly above this bucket is definitely in
    int64_t n_above = acc - hist[cut];
    // collect: definite winners first, then fill from the boundary bucket
    int pos = 0;
    for (int i = 0; i < n && pos < k; ++i) if (s[i] > thresh + span / NB) out[pos++] = i;
    for (int i = 0; i < n && pos < k; ++i) { float v = s[i]; if (v > thresh && v <= thresh + span / NB) out[pos++] = i; }
    for (int i = 0; i < n && pos < k; ++i) if (s[i] <= thresh) out[pos++] = i;   // degenerate ties
    (void)n_above;
}

// 4. the same threshold selection with NO threading. The parallel version above parallelises only the min/max and
//    histogram passes -- its collection loop is serial -- so most of its win may come from the algorithm rather than
//    the threads. That matters a lot for shipping: a single-threaded version drops straight into a ggml op, whereas a
//    parallel one needs intra-op barriers because the indexer's score tensor has exactly one row.
static void topk_threshold_st(const float * s, int n, int k, std::vector<int32_t> & out) {
    constexpr int NB = 1024;
    float lo = s[0], hi = s[0];
    for (int i = 1; i < n; ++i) { lo = std::min(lo, s[i]); hi = std::max(hi, s[i]); }
    if (!(hi > lo)) { for (int i = 0; i < k; ++i) out[i] = i; return; }
    const float scale = NB / (hi - lo);
    int hist[NB] = {0};
    for (int i = 0; i < n; ++i) {
        int bkt = (int)((s[i] - lo) * scale); bkt = bkt < 0 ? 0 : (bkt >= NB ? NB - 1 : bkt);
        hist[bkt]++;
    }
    long acc = 0; int cut = NB - 1;
    for (; cut > 0; --cut) { acc += hist[cut]; if (acc >= k) break; }
    const float hi_edge = lo + (float)(cut + 1) / scale;   // strictly above the cut bucket: definitely selected
    const float lo_edge = lo + (float)cut / scale;
    int pos = 0;
    for (int i = 0; i < n && pos < k; ++i) if (s[i] >= hi_edge) out[pos++] = i;
    for (int i = 0; i < n && pos < k; ++i) if (s[i] >= lo_edge && s[i] < hi_edge) out[pos++] = i;
    for (int i = 0; i < n && pos < k; ++i) if (s[i] <  lo_edge) out[pos++] = i;   // degenerate: k exceeds what qualifies
}

int main(int argc, char ** argv) {
    const int k = argc > 1 ? atoi(argv[1]) : 2048;
    const int reps = argc > 2 ? atoi(argv[2]) : 20;
    int nth = 1;
#ifdef _OPENMP
    nth = omp_get_max_threads(); if (nth > 15) nth = 15;   // one socket's worth, which is what a ggml op gets
#endif
    printf("top_k=%d, %d reps, %d threads for the parallel variant\n\n", k, reps, nth);
    printf("%10s %12s %12s %12s %12s   %s\n", "n_ctx", "partial_sort", "nth_element", "thr(par)", "thr(1 thread)",
           "vs partial_sort");
    std::mt19937 rng(1234);
    for (int n : {256, 2048, 9514, 38056, 100000, 262144}) {
        if (n < k) continue;
        std::vector<float> s(n);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto & x : s) x = nd(rng);
        std::vector<int32_t> idx(n), a(k), b(k), c(k), e(k);
        double t1 = 1e18, t2 = 1e18, t3 = 1e18, t4 = 1e18;
        for (int r = 0; r < reps; ++r) { auto t = clk::now(); topk_partial_sort(s.data(), n, k, idx, a); t1 = std::min(t1, ms_since(t)); }
        for (int r = 0; r < reps; ++r) { auto t = clk::now(); topk_nth_element (s.data(), n, k, idx, b); t2 = std::min(t2, ms_since(t)); }
        for (int r = 0; r < reps; ++r) { auto t = clk::now(); topk_threshold   (s.data(), n, k, c, nth);  t3 = std::min(t3, ms_since(t)); }
        for (int r = 0; r < reps; ++r) { auto t = clk::now(); topk_threshold_st(s.data(), n, k, e);       t4 = std::min(t4, ms_since(t)); }
        // Report HOW MANY indices differ, not just whether any do. The threshold variant resolves ties inside one
        // histogram bucket arbitrarily, so a handful of boundary differences is expected and harmless for attention
        // (the 2048th-best key is worth almost exactly what the 2049th is). Exact equality would call that broken.
        auto sorted = [](std::vector<int32_t> v){ std::sort(v.begin(), v.end()); return v; };
        auto A = sorted(a), B = sorted(b), C = sorted(c), E = sorted(e);
        auto diff = [&](const std::vector<int32_t> & X){
            std::vector<int32_t> d;
            std::set_symmetric_difference(A.begin(), A.end(), X.begin(), X.end(), std::back_inserter(d));
            return (int)d.size() / 2; };
        printf("%10d %12.3f %12.3f %12.3f %12.3f   nth %4d, thr %4d, thr1 %4d of %d differ\n",
               n, t1, t2, t3, t4, diff(B), diff(C), diff(E), k);
    }
    printf("\nThe engine runs this 12x per forward pass (one per full-attention layer), single-threaded.\n");
    return 0;
}
